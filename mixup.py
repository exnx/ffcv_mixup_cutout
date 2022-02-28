from typing import Tuple

from numba import objmode
import numpy as np
import torch as ch
import torch.nn.functional as F
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from numba import njit
import math

@njit
def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.flatten().astype("uint64")
    labels = np.full((x.size, num_classes), off_value)

    for i in range(x.size):
        labels[i, x[i]] = on_value

    return labels

@njit
def mixup_target(targets, num_classes, lam=1., smoothing=0.0):

    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(targets, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(np.flipud(targets), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)

@njit
def rand_bbox(batch_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        batch_shape (tuple): batch shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    _, img_h, img_w, _ = batch_shape

    # cant use int() for some reason
    cut_h, cut_w = int(round(img_h * ratio)), int(round(img_w * ratio))
    margin_y, margin_x = int(round(margin * cut_h)), int(round(margin * cut_w))

    cx = np.random.randint(0 + margin_y, img_h - margin_y)
    cy = np.random.randint(0 + margin_x, img_w - margin_x)

    # need to wrap in array
    y1_ = np.array([cy - cut_h // 2])
    yh_ = np.array([cy + cut_h // 2])
    xl_ = np.array([cx - cut_w // 2])
    xh_ = np.array([cx + cut_w // 2])

    yl = np.clip(y1_, 0, img_h)[0]  # grab the val from array
    yh = np.clip(yh_, 0, img_h)[0]
    xl = np.clip(xl_, 0, img_w)[0]
    xh = np.clip(xh_, 0, img_w)[0]
    return yl, yh, xl, xh

@njit
def rand_bbox_minmax(batch_shape, minmax):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.
    Args:
        batch_shape (tuple): batch shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
    """

    assert len(minmax) == 2
    _, img_h, img_w, _ = batch_shape

    h_start = int(round(img_h * minmax[0]))
    h_end = int(round(img_h * minmax[1]))

    w_start = int(round(img_w * minmax[0]))
    w_end = int(round(img_w * minmax[1]))

    cut_h = np.random.randint(h_start, h_end)
    cut_w = np.random.randint(w_start, w_end)
    yl = np.random.randint(0, img_h - cut_h)
    xl = np.random.randint(0, img_w - cut_w)

    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu

@njit
def cutmix_bbox_and_lam(batch_shape, lam, ratio_minmax=(0,0), correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if sum(ratio_minmax) > 0.:
        yl, yu, xl, xu = rand_bbox_minmax(batch_shape, ratio_minmax)
    else:
        yl, yu, xl, xu = rand_bbox(batch_shape, lam)
    if correct_lam or ratio_minmax is not None:

        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(batch_shape[1] * batch_shape[2])  # use h, w
    return (yl, yu, xl, xu), lam

@njit
def _params_per_batch(mixup_alpha, cutmix_alpha, mixup_prob, switch_prob, mixup_enabled):
    lam = 1.
    use_cutmix = False
    
    if mixup_enabled and np.random.rand() < mixup_prob:
        if mixup_alpha > 0. and cutmix_alpha > 0.:
            use_cutmix = np.random.rand() < switch_prob
            lam_mix = np.random.beta(cutmix_alpha, cutmix_alpha) if use_cutmix else \
                np.random.beta(mixup_alpha, mixup_alpha)
        elif mixup_alpha > 0.:
            lam_mix = np.random.beta(mixup_alpha, mixup_alpha)
        elif cutmix_alpha > 0.:
            use_cutmix = True
            lam_mix = np.random.beta(cutmix_alpha, cutmix_alpha)
        else:
            assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
        lam = float(lam_mix)
    return lam, use_cutmix

@njit
def _mix_batch(x, mixup_alpha, cutmix_alpha, cutmix_minmax, mixup_prob, switch_prob, correct_lam, mixup_enabled):
    lam, use_cutmix = _params_per_batch(mixup_alpha, cutmix_alpha, mixup_prob, switch_prob, mixup_enabled)
    if lam == 1.:
        return 1.
    if use_cutmix:

        (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
            x.shape, lam, ratio_minmax=cutmix_minmax, correct_lam=correct_lam)
        
        x_flip = np.flipud(x)
        x_final = x_flip[:, yl:yh, xl:xh, :]
        x[:, yl:yh, xl:xh, :] = x_final

    else:
        # for some reason can't use (when multiplying matrix)
        lam_inv = 1. -lam
        x_flipped = np.flipud(x) * lam_inv
        x[:,:,:,:] = x * lam + x_flipped
    return lam


class ImageMixup(Operation):

    def __init__(self,
        mixup_alpha=0.0, 
        cutmix_alpha=1.0,
        cutmix_minmax=(0,0),  # none or (0.2-0.3, 0.8-0.9), tuple
        mixup_prob=1.0,
        switch_prob=0.5,
        correct_lam=True,
        mixup_enabled = True
    ):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = mixup_prob
        self.switch_prob = switch_prob
        self.correct_lam = correct_lam
        self.mixup_enabled = mixup_enabled

    def generate_code(self) -> Callable:
        mixup_alpha = self.mixup_alpha
        cutmix_alpha = self.cutmix_alpha
        cutmix_minmax = self.cutmix_minmax
        mixup_prob = self.mixup_prob
        switch_prob = self.switch_prob
        correct_lam = self.correct_lam
        mixup_enabled = self.mixup_enabled

        my_range = Compiler.get_iterator()

        def image_mixer(x, dst, indices):
            np.random.seed(indices[-1])  # very important!

            assert len(x) % 2 == 0, 'Batch size should be even when using this'

            # normally we'd use lam to pass to mixup_target, but it's split in a different function,
            # so we need to calc lam again separate for targets
            lam = _mix_batch(x, mixup_alpha, cutmix_alpha, cutmix_minmax, mixup_prob, switch_prob, correct_lam, mixup_enabled)

            dst[:,:,:,:] = x[:,:,:,:]

            return dst

        image_mixer.is_parallel = True
        image_mixer.with_indices = True

        return image_mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (previous_state, AllocationQuery(shape=previous_state.shape,
                                                dtype=previous_state.dtype))


class LabelMixup(Operation):

    def __init__(self,
        mixup_prob=0.,
        mixup_alpha=0., 
        cutmix_alpha=0.,
        switch_prob=0.5,
        mixup_enabled=True,
        num_classes=1000,
        label_smoothing=0.1
    ):
        super().__init__()
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.mixup_enabled = mixup_enabled
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def generate_code(self) -> Callable:
        mixup_prob = self.mixup_prob
        mixup_alpha = self.mixup_alpha
        cutmix_alpha = self.cutmix_alpha
        switch_prob = self.switch_prob
        mixup_enabled = self.mixup_enabled
        num_classes = self.num_classes
        label_smoothing = self.label_smoothing

        my_range = Compiler.get_iterator()

        def label_mixer(targets, dst, indices):
            np.random.seed(indices[-1])  # very important!

            assert len(targets) % 2 == 0, 'Batch size should be even when using this'

            lam, use_cutmix = _params_per_batch(mixup_alpha, cutmix_alpha, mixup_prob, switch_prob, mixup_enabled)
            targets_new = mixup_target(targets, num_classes, lam, label_smoothing)
            dst[:,:] = targets_new[:,:]

            return dst

        label_mixer.is_parallel = True
        label_mixer.with_indices = True

        return label_mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        """
        The shape does change, so need to allocate for it, from (batch_size) to (batch_size, num_classes)
        """

        # batch_size = previous_state.shape

        new_shape = (self.num_classes, )  # needs to be a tuple
        new_state = replace(previous_state, shape=new_shape)

        mem_allocation = AllocationQuery(shape=new_shape, dtype=previous_state.dtype)

        return (new_state, mem_allocation)