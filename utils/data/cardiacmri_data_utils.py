import glob
import nibabel as nib
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
from itertools import combinations, permutations
from monai.config import KeysCollection
from monai.data import CacheDataset
from monai.transforms import (CastToTyped, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandSpatialCropSamplesd, Resized,
                              ResizeWithPadOrCropd, ScaleIntensityd,
                              Spacingd, ToTensord)
from monai.transforms import ScaleIntensityRanged as MonaiScaleIntensityRanged
from monai.transforms.transform import MapTransform
from monai.utils import first, set_determinism
from torch.utils.data import Dataset
from typing import (Any, Callable, Dict, Hashable, List, Mapping, Optional,
                    Sequence, Tuple, Union)

from ..transforms import ScaleIntensityRanged
from models import CFG
from ..path_utils import resolve_path

# def get_union_foreground(keys: KeysCollection) -> Callable:
#     """
#     Returns a function that computes the union of foreground masks from multiple segmentation images.
#     """
#     def _get_union(data):
#         # Initialize union mask with zeros
#         union_mask = np.zeros_like(data[keys[0]])

#         # Combine all segmentation masks
#         for key in keys:
#             # Convert segmentation to binary mask (any non-zero value is foreground)
#             mask = (data[key] > 0).astype(np.float32)
#             union_mask = np.logical_or(union_mask, mask)

#         return union_mask

#     return _get_union

# def load_data_ACDC(cfg: CFG,
#                    *args,
#                    **kwargs):
#     df_frames = pd.read_csv(os.path.join(cfg.data_dir, 'frames.csv'), index_col=0)
#     data_dicts = []
#     for i in cfg.data_indexs:
#         patient_dir = os.path.join(cfg.data_dir, f'patient{i:03d}')
#         fixed_frame = df_frames.loc[i]['fixed_frame']
#         moving_frame = df_frames.loc[i]['moving_frame']
#         data_dicts.append({
#             'fixed': os.path.join(patient_dir, f'patient{i:03d}_frame{fixed_frame:02d}.nii.gz'),
#             'fixed_seg': os.path.join(patient_dir, f'patient{i:03d}_frame{fixed_frame:02d}_gt.nii.gz'),
#             'moving': os.path.join(patient_dir, f'patient{i:03d}_frame{moving_frame:02d}.nii.gz'),
#             'moving_seg': os.path.join(patient_dir, f'patient{i:03d}_frame{moving_frame:02d}_gt.nii.gz'),
#         })

#     data_transforms = Compose([
#         LoadImaged(keys=['fixed', 'fixed_seg', 'moving', 'moving_seg']),
#         EnsureChannelFirstd(keys=['fixed', 'fixed_seg', 'moving', 'moving_seg']),
#         ScaleIntensityRanged(keys=['fixed', 'moving'],
#                              a_min=0.0,
#                              upper=99.9,
#                              b_min=0.0,
#                              b_max=1.0,
#                              clip=False),
#         Spacingd(keys=['fixed', 'moving'], pixdim=cfg.spacing, mode='bilinear'),
#         Spacingd(keys=['fixed_seg', 'moving_seg'], pixdim=cfg.spacing, mode='nearest'),
#         ResizeWithPadOrCropd(keys=['fixed', 'fixed_seg', 'moving', 'moving_seg'],
#                              spatial_size=cfg.image_size,
#                              mode='constant'),
#         ToTensord(keys=['fixed', 'moving', ], track_meta=False),
#         ToTensord(keys=['fixed_seg', 'moving_seg'], track_meta=False, dtype=torch.uint8),
#     ])

#     dataset = CacheDataset(data=data_dicts,
#                            transform=data_transforms,
#                            *args,
#                            **kwargs)
#     return dataset





def load_data_ACDC(cfg:CFG,
                   *args,
                   **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    df_path = resolve_path(cfg.df_path)
    df = pd.read_csv(df_path)
    data_dicts = []
    num_patients = len(df)
    for i in range(num_patients):
        patient_id = df.loc[i]['patient_id']
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.exists(patient_dir):
            continue
        ed_frame = df.loc[i]['ED']
        es_frame = df.loc[i]['ES']
        data_dicts.append({
            'ED': os.path.join(patient_dir, f'{patient_id}_frame{ed_frame:02d}.nii.gz'),
            'ES': os.path.join(patient_dir, f'{patient_id}_frame{es_frame:02d}.nii.gz'),
            'ED_seg': os.path.join(patient_dir, f'{patient_id}_frame{ed_frame:02d}_gt.nii.gz'),
            'ES_seg': os.path.join(patient_dir, f'{patient_id}_frame{es_frame:02d}_gt.nii.gz'),
            'patient_id': patient_id,
        })

    data_dicts = data_dicts[slice(*cfg.dataset_slice)]

    # First create a union mask
    def create_union_mask(data):
        ed_mask = (data['ED_seg'] > 0).astype(np.float32)
        es_mask = (data['ES_seg'] > 0).astype(np.float32)
        union_mask = np.logical_or(ed_mask, es_mask)
        data['union_mask'] = union_mask
        return data

    data_transforms = Compose([
        LoadImaged(keys=['ED', 'ES', 'ED_seg', 'ES_seg']),
        EnsureChannelFirstd(keys=['ED', 'ES', 'ED_seg', 'ES_seg']),
        Spacingd(keys=['ED', 'ES'], pixdim=cfg.spacing, mode='bilinear'),
        Spacingd(keys=['ED_seg', 'ES_seg'], pixdim=cfg.spacing, mode='nearest'),
        # Create union mask
        create_union_mask,
        # Crop based on union mask
        CropForegroundd(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            source_key='union_mask',
            k_divisible=32,
            margin=cfg.crop_margin if hasattr(cfg, 'crop_margin') else 10,
        ),
        ResizeWithPadOrCropd(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            spatial_size=cfg.image_size,
            mode='constant',
        ),
        ScaleIntensityRanged(keys=['ED', 'ES'],
                             a_min=0.0,
                             upper=100.0,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        ToTensord(keys=['ED', 'ES'], track_meta=False),
        ToTensord(keys=['ED_seg', 'ES_seg'], track_meta=False, dtype=torch.uint8),
    ])

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset

class StackValid2DSlicesAsChannel(MapTransform):
    def __init__(self, keys, seg_keys, required_labels=(1,2,3)):
        super().__init__(keys)
        self.seg_keys = seg_keys
        self.required_labels = set(required_labels)
    def __call__(self, data):
        # Keep input as (1, H, W, D)
        for key in self.keys:
            if data[key].ndim == 4 and data[key].shape[0] == 1:
                pass  # keep as (1, H, W, D)
        for key in self.seg_keys:
            if data[key].ndim == 4 and data[key].shape[0] == 1:
                pass
        seg1 = data[self.seg_keys[0]]  # (1, H, W, D)
        seg2 = data[self.seg_keys[1]]  # (1, H, W, D)
        D = seg1.shape[-1]
        required_labels = np.array(list(self.required_labels))
        # Remove channel dimension for label check
        seg1_ = seg1[0]  # (H, W, D)
        seg2_ = seg2[0]  # (H, W, D)
        def label_in_slice(seg, label):
            return np.any(seg == label, axis=(0, 1))  # (D,)
        valid_mask1 = np.ones(D, dtype=bool)
        valid_mask2 = np.ones(D, dtype=bool)
        for label in required_labels:
            valid_mask1 &= label_in_slice(seg1_, label)
            valid_mask2 &= label_in_slice(seg2_, label)
        valid_mask = valid_mask1 & valid_mask2
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid slices found with all required labels in both segmentations.")
        for key in self.keys:
            arr = data[key]  # (1, H, W, D)
            arr = arr[..., valid_indices]  # (1, H, W, D_valid)
            arr = np.moveaxis(arr, -1, 0)  # (D_valid, 1, H, W)
            arr = arr.squeeze(1)  # (D_valid, H, W)
            data[key] = arr
        return data


def load_data_ACDC_2d(cfg:CFG,
                      *args,
                      **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    df_path = resolve_path(cfg.df_path)
    df = pd.read_csv(df_path)
    data_dicts = []
    num_patients = len(df)
    for i in range(num_patients):
        patient_id = df.loc[i]['patient_id']
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.exists(patient_dir):
            continue
        ed_frame = df.loc[i]['ED']
        es_frame = df.loc[i]['ES']
        data_dicts.append({
            'ED': os.path.join(patient_dir, f'{patient_id}_frame{ed_frame:02d}.nii.gz'),
            'ES': os.path.join(patient_dir, f'{patient_id}_frame{es_frame:02d}.nii.gz'),
            'ED_seg': os.path.join(patient_dir, f'{patient_id}_frame{ed_frame:02d}_gt.nii.gz'),
            'ES_seg': os.path.join(patient_dir, f'{patient_id}_frame{es_frame:02d}_gt.nii.gz'),
            'patient_id': patient_id,
        })

    data_dicts = data_dicts[slice(*cfg.dataset_slice)]

    data_transforms = Compose([
        LoadImaged(keys=['ED', 'ES', 'ED_seg', 'ES_seg']),
        EnsureChannelFirstd(keys=['ED', 'ES', 'ED_seg', 'ES_seg']),
        StackValid2DSlicesAsChannel(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            seg_keys=['ED_seg', 'ES_seg'],
            required_labels=(1,2,3)
        ),
        CropForegroundd(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            source_key='ED_seg',  # Use one seg as crop source
            k_divisible=128,
            margin=cfg.crop_margin if hasattr(cfg, 'crop_margin') else 10,
        ),
        ResizeWithPadOrCropd(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            spatial_size=cfg.image_size,  # should be 2D, e.g., (H, W)
            mode='constant',
        ),
        ScaleIntensityRanged(keys=['ED', 'ES'],
                             a_min=0.0,
                             upper=100.0,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        ToTensord(keys=['ED', 'ES'], track_meta=False),
        ToTensord(keys=['ED_seg', 'ES_seg'], track_meta=False, dtype=torch.uint8),
    ])

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset

def load_data_MMs_2d(cfg:CFG,
                    *args,
                    **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    df_path = resolve_path(cfg.df_path)
    df = pd.read_csv(df_path)
    data_dicts = []
    num_patients = len(df)
    for i in range(num_patients):
        patient_id = df.loc[i]['External code']
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.exists(patient_dir):
            continue
        ed_frame = df.loc[i]['ED']
        es_frame = df.loc[i]['ES']
        data_dicts.append({
            'ED': os.path.join(patient_dir, f'{patient_id}_frame{ed_frame:02d}.nii.gz'),
            'ES': os.path.join(patient_dir, f'{patient_id}_frame{es_frame:02d}.nii.gz'),
            'ED_seg': os.path.join(patient_dir, f'{patient_id}_frame{ed_frame:02d}_gt.nii.gz'),
            'ES_seg': os.path.join(patient_dir, f'{patient_id}_frame{es_frame:02d}_gt.nii.gz'),
            'patient_id': patient_id,
        })

    data_dicts = data_dicts[slice(*cfg.dataset_slice)]
    print('num of subjects:', len(data_dicts))

    resize_to = getattr(cfg, 'resize_to', None)

    transform_list = [
        LoadImaged(keys=['ED', 'ES', 'ED_seg', 'ES_seg']),
        EnsureChannelFirstd(keys=['ED', 'ES', 'ED_seg', 'ES_seg']),
        StackValid2DSlicesAsChannel(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            seg_keys=['ED_seg', 'ES_seg'],
            required_labels=(1,2,3)
        ),
        CropForegroundd(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            source_key='ED_seg',
            k_divisible=128,
            margin=cfg.crop_margin if hasattr(cfg, 'crop_margin') else 32,
        ),
        ResizeWithPadOrCropd(
            keys=['ED', 'ES', 'ED_seg', 'ES_seg'],
            spatial_size=cfg.image_size,
            mode='constant',
        ),
    ]

    if resize_to is not None:
        transform_list.append(Resized(
            keys=['ED', 'ES'],
            spatial_size=resize_to,
            mode='bilinear',
        ))
        transform_list.append(Resized(
            keys=['ED_seg', 'ES_seg'],
            spatial_size=resize_to,
            mode='nearest',
        ))

    transform_list.extend([
        ScaleIntensityRanged(keys=['ED', 'ES'],
                             a_min=0.0,
                             upper=100.0,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        ToTensord(keys=['ED', 'ES'], track_meta=False),
        ToTensord(keys=['ED_seg', 'ES_seg'], track_meta=False, dtype=torch.uint8),
    ])

    data_transforms = Compose(transform_list)

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset
