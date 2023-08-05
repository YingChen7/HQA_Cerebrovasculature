#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from nnunet.training.data_augmentation.data_augmentation import augment_spatial, augment_gaussian_noise, augment_gaussian_blur,\
    augment_brightness_multiplicative, augment_brightness_additive, augment_contrast, augment_linear_downsampling_scipy, \
    augment_gamma, augment_mirroring, augment_mirroring_

from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
import abc
import numpy as np
import random
import torch
from typing import Tuple, Union, Callable

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None
    
def DataChannelSelectionTransform(data, select_channels):
    return data[:, select_channels]

def Convert3DTo2DTransform(data):
    shp = data.shape
    data = data.reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    return data, shp

def Convert2DTo3DTransform(data, ori_shp):
    current_shape = data.shape
    data = data.reshape((ori_shp[0], ori_shp[1], ori_shp[2], current_shape[-2], current_shape[-1]))
    return data


def SpatialTransform(data, patch_size_spatial, params, order_data = 3, border_val_seg = -1, order_seg = 1):
    if patch_size_spatial is None:
        if len(data.shape) == 4:
            patch_size = (data.shape[2], data.shape[3])
        elif len(data.shape) == 5:
            patch_size = (data.shape[2], data.shape[3], data.shape[4])
        else:
            raise ValueError("only support 2D/3D batch data.")
    else:
        patch_size = patch_size_spatial

    ret_val, _ = augment_spatial(data, seg = None, patch_size=patch_size,
                               patch_center_dist_from_border=None,
                               do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
                               sigma=params.get("elastic_deform_sigma"),
                               do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
                               angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
                               do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
                               border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
                               border_mode_seg="constant", border_cval_seg=border_val_seg,
                               order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
                               p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
                               independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"))
    return ret_val

def GaussianNoiseTransform(data, noise_variance=(0, 0.1), p_per_sample=1,
                           p_per_channel: float = 1, per_channel: bool = False,):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_gaussian_noise(data[b], noise_variance = noise_variance,  p_per_channel = p_per_channel, per_channel= per_channel)
    return data
    
def GaussianBlurTransform(data, blur_sigma: Tuple[float, float] = (1, 5), 
                          different_sigma_per_channel: bool = True, different_sigma_per_axis: bool = False, 
                          p_isotropic: float = 0, p_per_channel: float = 1,
                          p_per_sample: float = 1,):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_gaussian_blur(data[b], sigma_range = blur_sigma,  per_channel = different_sigma_per_channel,
                                            p_per_channel = p_per_channel, different_sigma_per_axis=different_sigma_per_axis,
                                            p_isotropic = p_isotropic)
    return data

def BrightnessMultiplicativeTransform(data, multiplier_range=(0.5, 2), per_channel=True, p_per_sample=1):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_brightness_multiplicative(data[b], multiplier_range, per_channel)
    return data
def BrightnessTransform(data, mu, sigma, per_channel=True, p_per_sample=1, p_per_channel=1):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_brightness_additive(data[b], mu, sigma, per_channel,
                                                      p_per_channel=p_per_channel)
    return data
def ContrastAugmentationTransform(data, contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                                  preserve_range: bool = True,
                                  per_channel: bool = True, p_per_sample: float = 1,
                                  p_per_channel: float = 1):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_contrast(data[b], contrast_range=contrast_range,
                                       preserve_range=preserve_range,
                                       per_channel=per_channel,
                                       p_per_channel=p_per_channel)
    return data
def SimulateLowResolutionTransform(data, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                                   channels=None, order_downsample=1, order_upsample=0, p_per_sample=1,
                                   ignore_axes=None):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_linear_downsampling_scipy(data[b], zoom_range=zoom_range,
                                                        per_channel=per_channel,p_per_channel=p_per_channel,
                                                        channels=channels, order_downsample=order_downsample,
                                                        order_upsample=order_upsample,ignore_axes=ignore_axes)
    return data
def GammaTransform(data, gamma_range=(0.5, 2), invert_image=False, per_channel=False,
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            data[b] = augment_gamma(data[b], gamma_range, invert_image,
                                    per_channel=per_channel, retain_stats=retain_stats)
    return data
def MirrorTransform(data, axes=(0, 1, 2), p_per_sample=1):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            ret_val = augment_mirroring(data[b], None, axes=axes)
            data[b] = ret_val[0]
    return data 

def MirrorTransform_(data, coord, axes=(0, 1, 2), p_per_sample=1):
    for b in range(len(data)):
        if np.random.uniform() < p_per_sample:
            ret_val = augment_mirroring_(data[b], coord[b], axes=axes)
            data[b], coord[b] = ret_val[0], ret_val[1]
    return data, coord 

def RandomCropTransform(data, patch_size):
    data_shape = data.shape
    res = np.zeros((data_shape[0], data_shape[1],  patch_size[0], patch_size[1], patch_size[2]))
    for b in range(len(data)):
        if (data_shape[2] - patch_size[0])==0:
            ori_dep = 0
        else:
            ori_dep = np.random.randint(0, data_shape[2] - patch_size[0])
        if (data_shape[3] - patch_size[1])==0:
            ori_heg =0
        else:
            ori_heg = np.random.randint(0, data_shape[3] - patch_size[1])
        if data_shape[4] - patch_size[2]==0:
            ori_wid = 0
        else:
            ori_wid = np.random.randint(0, data_shape[4] - patch_size[2])
        res[b] = data[b][:, ori_dep:ori_dep + patch_size[0], ori_heg:ori_heg + patch_size[1], ori_wid:ori_wid + patch_size[2]]
    return res


def NumpyToTensor(data, cast_to=None):
    tensor = torch.from_numpy(data)
    if cast_to is not None:
        if cast_to == 'half':
            tensor = tensor.half()
        elif cast_to == 'float':
            tensor = tensor.float()
        elif cast_to == 'long':
            tensor = tensor.long()
        else:
            raise ValueError('Unknown value for cast_to: %s' % cast_to)
    return tensor
    

def get_batch_augmentation(data, patch_size, coord = None, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, regions=None):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    origin_data = data.copy()
    if coord is not None:
        origin_coord = coord.copy()
    
    if params.get("selected_data_channels") is not None:
        data = DataChannelSelectionTransform(data, params.get("selected_data_channels"))


    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        data, ori_shape = Convert3DTo2DTransform(data)
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None
    
    # data = SpatialTransform(data, patch_size_spatial, params, order_data = order_data, border_val_seg = border_val_seg, order_seg = order_seg) 
    
    if params.get("dummy_2D"):
        data = Convert2DTo3DTransform(data, ori_shape)

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    data = GaussianNoiseTransform(data, p_per_sample=0.1)
    data = GaussianBlurTransform(data, (0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5)
    data = BrightnessMultiplicativeTransform(data, multiplier_range=(0.75, 1.25), p_per_sample=0.15)

    if params.get("do_additive_brightness"):
        data = BrightnessTransform(data, params.get("additive_brightness_mu"),
                                        params.get("additive_brightness_sigma"),
                                        True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                        p_per_channel=params.get("additive_brightness_p_per_channel"))

    data = ContrastAugmentationTransform(data, p_per_sample=0.15)
    data = SimulateLowResolutionTransform(data, zoom_range=(0.5, 1), per_channel=True,
                                          p_per_channel=0.5,
                                          order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                          ignore_axes=ignore_axes)
    data = GammaTransform(data, params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"), p_per_sample=0.1)

    if params.get("do_gamma"):
        data = GammaTransform(data, params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"])

    if params.get("do_mirror") or params.get("mirror"):
        if coord is not None:
            data, coord = MirrorTransform_(data, coord, params.get("mirror_axes"),  p_per_sample=0.5)
        else:
            data = MirrorTransform(data, params.get("mirror_axes"),  p_per_sample=0.5)
    origin_data = NumpyToTensor(origin_data, 'float')
    data = NumpyToTensor(data, 'float')
    
    if coord is not None:
        origin_coord = NumpyToTensor(origin_coord, 'float')
        coord = NumpyToTensor(coord, 'float')
        return origin_data, origin_coord, data, coord
    else:
        return origin_data, data

