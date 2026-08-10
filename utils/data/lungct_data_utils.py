import glob
import os
import sys

from typing import (Any, Callable, Dict, Hashable, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.config import KeysCollection
from monai.data import CacheDataset
from monai.transforms.transform import MapTransform
from monai.transforms import (CastToTyped, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandSpatialCropSamplesd, Resized,
                              ResizeWithPadOrCropd, ScaleIntensityd,
                              Spacingd, ToTensord)
from monai.utils import first, set_determinism
from monai.transforms import ScaleIntensityRanged as MonaiScaleIntensityRanged
from models import CFG
from itertools import combinations, permutations
from torch.utils.data import Dataset
import random
from ..path_utils import resolve_path
from ..transforms import (ScaleIntensityRanged, ForegroundMaskingd,
                            ResizeWithPadOrCropWithKeypointsd, ResizeWithInterpolationKeypointsd,
                            LungCTOneHotd)


class LoadCSVd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, *args, **kwargs) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            keypnt = pd.read_csv(d[key], delimiter=',', names=['x', 'y', 'z'])
            d[key] = keypnt.to_numpy()
        return d


def load_data_NLST(cfg: CFG,
                   *args,
                   **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.data_indexs:
        data_dicts.append({
            'exp': os.path.join(data_dir, 'imagesTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp': os.path.join(data_dir, 'imagesTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_mask': os.path.join(data_dir, 'masksTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp_mask': os.path.join(data_dir, 'masksTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_keypnt': os.path.join(data_dir, 'keypointsTr', f'NLST_{i:04d}_0000.csv'),
            'inp_keypnt': os.path.join(data_dir, 'keypointsTr', f'NLST_{i:04d}_0001.csv'),
        })

    data_transforms = Compose([
        LoadImaged(keys=['exp', 'inp', 'exp_mask', 'inp_mask']),
        LoadCSVd(keys=['exp_keypnt', 'inp_keypnt']),
        EnsureChannelFirstd(keys=['exp', 'inp', 'exp_mask', 'inp_mask'],
                            channel_dim='no_channel'),
        MonaiScaleIntensityRanged(keys=['exp', 'inp'],
                                  a_min=-1000,
                                  a_max=500,
                                  b_min=0.0,
                                  b_max=1.0,
                                  clip=True),
        ToTensord(keys=['exp', 'inp', 'exp_mask', 'inp_mask'], track_meta=False),
    ])
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)
    return dataset


def load_data_NLST_label(cfg: CFG,
                         *args,
                         **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.data_indexs:
        data_dicts.append({
            'exp': os.path.join(data_dir, 'corrected_imagesTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp': os.path.join(data_dir, 'corrected_imagesTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_mask': os.path.join(data_dir, 'masksTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp_mask': os.path.join(data_dir, 'masksTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_label': os.path.join(data_dir, 'labelsTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp_label': os.path.join(data_dir, 'labelsTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_keypnt': os.path.join(data_dir, 'keypointsTr', f'NLST_{i:04d}_0000.csv'),
            'inp_keypnt': os.path.join(data_dir, 'keypointsTr', f'NLST_{i:04d}_0001.csv'),
        })

    data_transforms = Compose([
        LoadImaged(keys=['exp', 'inp', 'exp_mask', 'inp_mask', 'exp_label', 'inp_label']),
        LoadCSVd(keys=['exp_keypnt', 'inp_keypnt']),
        EnsureChannelFirstd(keys=['exp', 'inp', 'exp_mask', 'inp_mask', 'exp_label', 'inp_label'],
                            channel_dim='no_channel'),
        MonaiScaleIntensityRanged(keys=['exp', 'inp'],
                                  a_min=-1000,
                                  a_max=500,
                                  b_min=0.0,
                                  b_max=1.0,
                                  clip=True),
        LungCTOneHotd(keys=['exp_label', 'inp_label']),
        ToTensord(keys=['exp', 'inp', 'exp_mask', 'inp_mask', 'exp_label', 'inp_label'], track_meta=False),
    ])
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)
    return dataset


def load_data_NLST_fg(cfg: CFG,
                      val: bool = False,
                      *args,
                      **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.data_indexs:
        data_dicts.append({
            'exp': os.path.join(data_dir, 'imagesTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp': os.path.join(data_dir, 'imagesTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_mask': os.path.join(data_dir, 'masksTr', f'NLST_{i:04d}_0000.nii.gz'),
            'inp_mask': os.path.join(data_dir, 'masksTr', f'NLST_{i:04d}_0001.nii.gz'),
            'exp_keypnt': os.path.join(data_dir, 'keypointsTr', f'NLST_{i:04d}_0000.csv'),
            'inp_keypnt': os.path.join(data_dir, 'keypointsTr', f'NLST_{i:04d}_0001.csv'),
        })

    if val:
        data_transforms = Compose([
            LoadImaged(keys=['exp', 'inp', 'exp_mask', 'inp_mask']),
            LoadCSVd(keys=['exp_keypnt', 'inp_keypnt']),
            EnsureChannelFirstd(keys=['exp', 'inp', 'exp_mask', 'inp_mask'],
                                channel_dim='no_channel'),
            MonaiScaleIntensityRanged(keys=['exp', 'inp'],
                                      a_min=-1000,
                                      a_max=500,
                                      b_min=0.0,
                                      b_max=1.0,
                                      clip=True),
            ForegroundMaskingd(keys=['exp', 'inp'],
                               mask_keys=['exp_mask', 'inp_mask']),
            ToTensord(keys=['exp', 'inp', 'exp_mask', 'inp_mask'], track_meta=False),
        ])
    else:
        data_transforms = Compose([
            LoadImaged(keys=['exp', 'inp', 'exp_mask', 'inp_mask']),
            EnsureChannelFirstd(keys=['exp', 'inp', 'exp_mask', 'inp_mask'],
                                channel_dim='no_channel'),
            MonaiScaleIntensityRanged(keys=['exp', 'inp'],
                                      a_min=-1000,
                                      a_max=500,
                                      b_min=0.0,
                                      b_max=1.0,
                                      clip=True),
            ForegroundMaskingd(keys=['exp', 'inp'],
                               mask_keys=['exp_mask', 'inp_mask']),
            # Orientationd
            # Cropping and Resize will mess up the keypoints
            ResizeWithPadOrCropd(keys=['exp', 'inp', 'exp_mask', 'inp_mask'], spatial_size=cfg.image_size,
                                 mode='constant'),
            ToTensord(keys=['exp', 'inp', 'exp_mask', 'inp_mask'], track_meta=False),
        ])
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)
    return dataset


def load_data_ThoraxCBCT(cfg: CFG,
                         val: bool = False,
                         *args,
                         **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.data_indexs:
        data_dicts.append({
            'FBCT': os.path.join(data_dir, 'imagesTr', f'ThoraxCBCT_{i:04d}_0000.nii.gz'),
            'begin_CBCT': os.path.join(data_dir, 'imagesTr', f'ThoraxCBCT_{i:04d}_0001.nii.gz'),
            'end_CBCT': os.path.join(data_dir, 'imagesTr', f'ThoraxCBCT_{i:04d}_0002.nii.gz'),
            'begin_FBCT_keypnt': os.path.join(data_dir, 'keypoints01Tr', f'ThoraxCBCT_{i:04d}_0000.csv'),
            'begin_CBCT_keypnt': os.path.join(data_dir, 'keypoints01Tr', f'ThoraxCBCT_{i:04d}_0001.csv'),
            'end_FBCT_keypnt': os.path.join(data_dir, 'keypoints02Tr', f'ThoraxCBCT_{i:04d}_0000.csv'),
            'end_CBCT_keypnt': os.path.join(data_dir, 'keypoints02Tr', f'ThoraxCBCT_{i:04d}_0002.csv'),
        })

    if val:
        data_transforms = Compose([
            LoadImaged(keys=['FBCT', 'begin_CBCT', 'end_CBCT']),
            LoadCSVd(keys=['begin_FBCT_keypnt', 'begin_CBCT_keypnt', 'end_FBCT_keypnt', 'end_CBCT_keypnt']),
            EnsureChannelFirstd(keys=['FBCT', 'begin_CBCT', 'end_CBCT'],
                                channel_dim='no_channel'),
            MonaiScaleIntensityRanged(keys=['FBCT', 'begin_CBCT', 'end_CBCT'],
                                      a_min=-1000,
                                      a_max=500,
                                      b_min=0.0,
                                      b_max=1.0,
                                      clip=True
                                      ),
            ToTensord(keys=['FBCT', 'begin_CBCT', 'end_CBCT'], track_meta=False),
        ])
    else:
        data_transforms = Compose([
            LoadImaged(keys=['FBCT', 'begin_CBCT', 'end_CBCT']),
            LoadCSVd(keys=['begin_FBCT_keypnt', 'begin_CBCT_keypnt', 'end_FBCT_keypnt', 'end_CBCT_keypnt']),
            EnsureChannelFirstd(keys=['FBCT', 'begin_CBCT', 'end_CBCT'],
                                channel_dim='no_channel'),
            MonaiScaleIntensityRanged(keys=['FBCT', 'begin_CBCT', 'end_CBCT'],
                                      a_min=-1000,
                                      a_max=500,
                                      b_min=0.0,
                                      b_max=1.0,
                                      clip=True
                                      ),
            # ResizeWithPadOrCropWithKeypointsd(
            #     image_keys=['FBCT', 'begin_CBCT', 'end_CBCT'],
            #     keypoint_keys=['begin_FBCT_keypnt', 'begin_CBCT_keypnt', 'end_FBCT_keypnt', 'end_CBCT_keypnt'],
            #     spatial_size=cfg.image_size,
            #     mode='constant'),
            ResizeWithInterpolationKeypointsd(
                image_keys=['FBCT', 'begin_CBCT', 'end_CBCT'],
                keypoint_keys=['begin_FBCT_keypnt', 'begin_CBCT_keypnt', 'end_FBCT_keypnt', 'end_CBCT_keypnt'],
                spatial_size=cfg.image_size,
                mode='linear',  # For 3D data
                align_corners=True
            ),
            ToTensord(keys=['FBCT', 'begin_CBCT', 'end_CBCT'], track_meta=False),
        ])
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)
    return dataset


class LoadCSV1d(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, *args, **kwargs) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            keypnt = pd.read_csv(d[key], delimiter=',', names=['x0', 'y0', 'z0', 'x1', 'y1', 'z1'])
            d[key] = keypnt.to_numpy()
            d[f'inp_{key}'] = keypnt.to_numpy()[:, :3]
            d[f'exp_{key}'] = keypnt.to_numpy()[:, 3:]
        return d


def load_data_Lung250M(cfg: CFG,
                       val: bool = False,
                       *args,
                       **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.data_indexs:
        data_dicts.append({
            'inp': os.path.join(data_dir, 'imagesTs', f'case_{i:03d}_1.nii.gz'),
            'exp': os.path.join(data_dir, 'imagesTs', f'case_{i:03d}_2.nii.gz'),
            'keypnt': os.path.join(data_dir, 'landmarkTs', f'case_{i:03d}.csv'),
        })

    if val:
        data_transforms = Compose([
            LoadImaged(keys=['exp', 'inp']),
            LoadCSV1d(keys=['keypnt']),
            EnsureChannelFirstd(keys=['exp', 'inp'],
                                channel_dim='no_channel'),
            MonaiScaleIntensityRanged(keys=['exp', 'inp'],
                                      a_min=-1000,
                                      a_max=500,
                                      b_min=0.0,
                                      b_max=1.0,
                                      clip=True
                                      ),
            ToTensord(keys=['exp', 'inp'], track_meta=False),
        ])
    else:
        data_transforms = Compose([
            LoadImaged(keys=['exp', 'inp']),
            LoadCSV1d(keys=['keypnt']),
            EnsureChannelFirstd(keys=['exp', 'inp'],
                                channel_dim='no_channel'),
            MonaiScaleIntensityRanged(keys=['exp', 'inp'],
                                      a_min=-1000,
                                      a_max=500,
                                      b_min=0.0,
                                      b_max=1.0,
                                      clip=True
                                      ),
            ResizeWithInterpolationKeypointsd(
                image_keys=['exp', 'inp'],
                keypoint_keys=['exp_keypnt', 'inp_keypnt'],
                spatial_size=cfg.image_size,
                as_scale_factor=False,
                mode='linear',  # For 3D data
                align_corners=True
            ),
            # ResizeWithPadOrCropWithKeypointsd(
            #     image_keys=['exp', 'inp'],
            #     keypoint_keys=['exp_keypnt', 'inp_keypnt'],
            #     spatial_size=cfg.image_size,
            #     mode='constant'),
            ToTensord(keys=['exp', 'inp'], track_meta=False),
        ])
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)
    return dataset
