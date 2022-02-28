""" Transforms Factory

Adapted from TIMM / Ross Wightman:
https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/data/transforms_factory.py#L44

"""
import math

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy
import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np

_LEVEL_DENOM = 10.  # denominator for conversion from 'Mx' magnitude scale to fractional aug level for op arguments

# only from branched version of FFCV, not in main yet.  Use ffcv-hippo2 conda env
from ffcv.transforms.randaugment import RandAugment

# for mixup (no cutout yet, different from TIMM)
# from ffcv.transforms.mixup import ImageMixup, LabelMixup, MixupToOneHot
from dataloaders.ffcv_custom.mixup import ImageMixup, LabelMixup


from ffcv.pipeline.operation import Operation
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder, ResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from dataloaders.ffcv_custom.cutout import Cutout
import ffcv

# for label pipeline
from ffcv.fields.basics import IntDecoder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100)
        'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10)
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
    num_ops = 2  # default to 2 ops per image
    num_magnitude_bins = 31 # default from FFCV

    weight_idx = None  # default to no probability weights for op choice
    # transforms = _RAND_TRANSFORMS
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param / randomization of magnitude values
            mstd = float(val)
            if mstd > 100:
                # use uniform sampling in 0 to magnitude if mstd is > 100
                mstd = float('inf')
            hparams.setdefault('magnitude_std', mstd)
        elif key == 'mmax':
            # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
            hparams.setdefault('magnitude_max', int(val))
        elif key == 'inc':
            # if bool(val):
            #     transforms = _RAND_INCREASING_TRANSFORMS
            pass # not implementing for now
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'

    return RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins)

def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        auto_augment=None,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        cutout_prob=0.,
        cutout_count=1,
        mixup_alpha=0.,
        cutmix_alpha=0,
        cutmix_minmax=(0,0),  # none or [0.2-0.3, 0.8-0.9]
        mixup_prob=0.,
        switch_prob=0.5,
        correct_lam=True,
        mixup_enabled=True,
        label_smoothing=0.,
        num_classes=1000,
    ):

    """
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """

    this_device = f'cuda:{torch.cuda.current_device()}'

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range

    # add the primary ffcv transforms here
    decoder = ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=(img_size, img_size), scale=scale, ratio=ratio)
    primary_tfl = [decoder]

    if hflip > 0.:
        primary_tfl += [ffcv.transforms.RandomHorizontalFlip(flip_prob=hflip)]
    # vflip not supported / used here    

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        # if interpolation and interpolation != 'random':
        #     aa_params['interpolation'] = str_to_pil_interp(interpolation)

        if auto_augment.startswith('rand'):  ####   we use this one  ####
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]

    final_tfl = []

    # which is same as cutout
    if cutout_prob > 0.:
        final_tfl += [
            # use custom cutout
            Cutout(
                min_count=cutout_count,
                prob=cutout_prob
            )
        ]

    if mixup_alpha > 0 or cutmix_alpha > 0:
        # final_tfl += [ImageMixup(alpha=mixup_alpha, same_lambda=True)]


        final_tfl += [
            ImageMixup(
                mixup_alpha=mixup_alpha, 
                cutmix_alpha=cutmix_alpha,
                cutmix_minmax=cutmix_minmax,  # none or [0.2-0.3, 0.8-0.9]
                mixup_prob=mixup_prob,
                switch_prob=switch_prob,
                correct_lam=correct_lam,
                mixup_enabled = mixup_enabled
            )
        ]
    
    final_tfl += [
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(torch.device(this_device), non_blocking=True),  # needs this
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(mean, std, np.float32)  # docs use float 16, maybe use later
    ]

    label_pipeline = [IntDecoder()]

    if mixup_alpha > 0 or cutmix_alpha > 0:
        # label_pipeline += [LabelMixup(alpha=mixup_alpha, same_lambda=True)]

        label_pipeline += [
            LabelMixup(
                mixup_prob=mixup_prob,
                mixup_alpha=mixup_alpha, 
                cutmix_alpha=cutmix_alpha,
                switch_prob=switch_prob,
                mixup_enabled=mixup_enabled,
                num_classes=num_classes,
                label_smoothing=label_smoothing
            )
        ]

    label_pipeline += [
        ToTensor(),
        Squeeze()
    ]

    # if mixup_alpha > 0 or cutmix_alpha > 0:
    #     label_pipeline += [MixupToOneHot(num_classes=num_classes)]

    label_pipeline += [
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    return (primary_tfl + secondary_tfl + final_tfl), label_pipeline


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    crop_pct = crop_pct or DEFAULT_CROP_PCT
    this_device = f'cuda:{torch.cuda.current_device()}'

    tfl = [
        ffcv.fields.rgb_image.CenterCropRGBImageDecoder((img_size, img_size), ratio=crop_pct),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(torch.device(this_device), non_blocking=True),  # needs this
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(mean, std, np.float32)  # docs use float 16, maybe use later
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    return tfl, label_pipeline

def create_transform(
        input_size,
        is_training=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        auto_augment=None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        cutout_prob=0.,
        cutout_count=1,
        crop_pct=None,
        mixup_alpha=0.,
        cutmix_alpha=0.,
        cutmix_minmax=(0, 0),  # none or [0.2-0.3, 0.8-0.9]
        mixup_prob=0.,
        switch_prob=0.5,
        correct_lam=True,
        mixup_enabled=True,
        label_smoothing=0.,
        num_classes=1000,
        tf_preprocessing=False,
    ):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training and not no_aug:
        transform = transforms_imagenet_train(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            auto_augment=auto_augment,
            mean=mean,
            std=std,
            cutout_prob=cutout_prob,
            cutout_count=cutout_count,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=cutmix_minmax,  # none or [0.2-0.3, 0.8-0.9]
            mixup_prob=mixup_prob,
            switch_prob=switch_prob,
            correct_lam=correct_lam,
            mixup_enabled=mixup_enabled,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )

    else:
        transform = transforms_imagenet_eval(
            img_size,
            mean=mean,
            std=std,
            crop_pct=crop_pct)

    return transform