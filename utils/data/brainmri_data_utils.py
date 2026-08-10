import json
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
from monai.transforms import (CastToTyped, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandSpatialCropSamplesd, Resized,
                              ResizeWithPadOrCropd, ScaleIntensityd,
                              NormalizeIntensityd,
                              Spacingd, ToTensord)
from monai.utils import first, set_determinism
from monai.transforms import ScaleIntensityRanged as MonaiScaleIntensityRanged
from ..transforms import ScaleIntensityRanged
from ..transforms import OasisBrainOneHotd, MindBoggleOneHotd, LesionOneHotd, ADNIOneHotd
from models import CFG
from itertools import combinations, permutations
from torch.utils.data import Dataset
import random
from .dataset_utils import IXIBrainDataset
from ..path_utils import resolve_path


def load_data_oasis(cfg: CFG,
                    val: bool = False,
                    *args, **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    scan_files = []
    seg_files = []

    for scan_dir in os.listdir(data_dir):
        if scan_dir.startswith('OASIS'):
            scan_file = os.path.join(data_dir, scan_dir, 'aligned_norm.nii.gz')
            seg_file = os.path.join(data_dir, scan_dir, 'aligned_seg35.nii.gz')
            if os.path.exists(seg_file):
                scan_files.append(scan_file)
                seg_files.append(seg_file)

    if val:
        scan_files = sorted(scan_files)
        seg_files = sorted(seg_files)

    scan_files = scan_files[slice(*cfg.dataset_slice)]
    seg_files = seg_files[slice(*cfg.dataset_slice)]
    print(len(scan_files), len(seg_files))
    # print(scan_files)
    data_dicts = [{
        'image': scan_file,
        'label': seg_file
    } for (scan_file, seg_file) in zip(scan_files, seg_files)]
    data_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        # ScaleIntensityRanged(keys=['image'],
        #                      a_min=0.0,
        #                      upper=99.9,
        #                      b_min=0.0,
        #                      b_max=1.0,
        #                      clip=False),
        OasisBrainOneHotd(keys=['label']),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])
    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_adni(cfg: CFG, *args, **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    df_path = resolve_path(cfg.df_path)
    df = pd.read_csv(df_path)
    scan_files = []
    seg_files = []
    image_uids = df['IMAGEUID'].unique()
    for uid in image_uids:
        scan_path = os.path.join(data_dir, f'{uid}/norm.nii.gz')
        seg_path = os.path.join(data_dir, f'{uid}/aseg.nii.gz')
        if os.path.exists(scan_path) and os.path.exists(seg_path):
            scan_files.append(scan_path)
            seg_files.append(seg_path)
    # print(len(scan_files))
    scan_files = scan_files[slice(*cfg.dataset_slice)]
    seg_files = seg_files[slice(*cfg.dataset_slice)]

    data_dicts = [{
        'image': scan_file,
        'label': seg_file
    } for (scan_file, seg_file) in zip(scan_files, seg_files)]

    print(len(data_dicts))
    data_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        Orientationd(keys=['image', 'label'], axcodes='LIA'),
        # ScaleIntensityRanged(keys=['image'],
        #                      a_min=0.0,
        #                      upper=99.99,
        #                      b_min=0.0,
        #                      b_max=1.0,
        #                      clip=False),
        ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ADNIOneHotd(keys=['label']),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])
    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_t1_t2(cfg: CFG, *args, **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    t1_files = []
    t2_files = []
    seg_files = []

    scan_dirs = sorted(os.listdir(data_dir))

    for scan_dir in scan_dirs:
        if os.path.isdir(os.path.join(data_dir, scan_dir)):
            t1_file = os.path.join(data_dir, scan_dir, 'norm_MNI.nii.gz')
            t2_file = os.path.join(data_dir, scan_dir, 'mri2.nii.gz')
            seg_file = os.path.join(data_dir, scan_dir, 'aseg.nii.gz')
            if os.path.exists(seg_file):
                t1_files.append(t1_file)
                t2_files.append(t2_file)
                seg_files.append(seg_file)

    t1_files = t1_files[slice(*cfg.dataset_slice)]
    t2_files = t2_files[slice(*cfg.dataset_slice)]
    seg_files = seg_files[slice(*cfg.dataset_slice)]

    data_dicts = [{
        't1': t1_file,
        't2': t2_file,
        'label': seg_file
    } for (t1_file, t2_file, seg_file) in zip(t1_files, t2_files, seg_files)]

    data_transforms = Compose([
        LoadImaged(keys=['t1', 't2', 'label']),
        EnsureChannelFirstd(keys=['t1', 't2', 'label'], channel_dim='no_channel'),
        # Model is trained with LIA orientated images
        Orientationd(keys=['t1', 't2', 'label'], axcodes='LIA'),
        ScaleIntensityRanged(keys=['t1', 't2'],
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        CropForegroundd(keys=['t1', 't2', 'label'],
                        source_key='t1',
                        k_divisible=2),
        ADNIOneHotd(keys=['label']),
        ResizeWithPadOrCropd(keys=['t1', 't2', 'label'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['t1', 't2', 'label'], track_meta=False)
    ])
    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_adni_tps(cfg: CFG, df_path: str, num_timepoints: int,
                       *args,
                       **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    df = pd.read_csv(df_path)
    df = df.sort_values(['PTID', 'Month'])
    rids = df['PTID'].unique()
    for p in rids:
        d_dict = {}
        subject_df = df.loc[df['PTID'] == p]
        for i in range(num_timepoints):
            image_uid = subject_df.iloc[i]['IMAGEUID']
            scan_path = os.path.join(data_dir,
                                     f'{image_uid}.long.{p}/mri/norm.mgz')
            d_dict.update({
                'subject': p,
                f't{i}_vis': subject_df.iloc[i]['VISCODE'],
                f't{i}_uid': image_uid,
                f't{i}': scan_path,
            })

        data_dicts.append(d_dict)

    print(f'tp{num_timepoints}:', len(data_dicts))
    data_dicts = data_dicts[slice(*cfg.dataset_slice)]
    # data_dicts = data_dicts[:1]

    scan_keys = [f't{i}' for i in range(num_timepoints)]

    pre_transforms = [
        LoadImaged(keys=scan_keys, reader='NibabelReader'),
        EnsureChannelFirstd(keys=scan_keys,
                            channel_dim='no_channel'),
        # ADNI long all oriented LIA
        # Orientationd(keys=scan_keys + seg_keys, axcodes='LIA'),
        ScaleIntensityRanged(keys=scan_keys,
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
    ]
    crop_fg_transforms = [
        CropForegroundd(keys=[f't{i}' for i in range(num_timepoints)],
                        source_key=f't0',
                        k_divisible=2)
    ]

    post_transforms = [
        ResizeWithPadOrCropd(keys=scan_keys,
                             spatial_size=cfg.image_size,
                             mode='constant'),
        # OneHotd(keys=seg_keys),
        ToTensord(keys=scan_keys, track_meta=False)
    ]

    data_transforms = Compose(pre_transforms + crop_fg_transforms +
                              post_transforms)

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_adni_tps_infer(cfg: CFG,
                             df_path: str,
                             num_timepoints: int,
                             *args,
                             **kwargs):
    '''
    Load ADNI data w/wo crop foreground transform
    '''
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    df = pd.read_csv(df_path)
    rids = df['PTID'].unique()
    for p in rids:
        d_dict = {}
        # each rids will only has three timepoints
        subject_df = df.loc[df['PTID'] == p]
        tps = []
        for i in range(num_timepoints):
            image_uid = subject_df.iloc[i]['IMAGEUID']
            scan_path = os.path.join(data_dir,
                                     f'{image_uid}.long.{p}/mri/norm.mgz')
            seg_path = os.path.join(data_dir,
                                    f'{image_uid}.long.{p}/mri/aseg.mgz')
            d_dict.update({
                'subject': p,
                f't{i}_vis': subject_df.iloc[i]['VISCODE'],
                f't{i}_uid': image_uid,
                f't{i}': scan_path,
                f't{i}_seg': seg_path,
                f't{i}_orig': scan_path,
                f't{i}_seg_orig': seg_path,
            })
        data_dicts.append(d_dict)

    print(f'tp{num_timepoints}:', len(data_dicts))
    # data_dicts = data_dicts[:20]

    scan_keys = [f't{i}' for i in range(num_timepoints)]
    seg_keys = [f't{i}_seg' for i in range(num_timepoints)]

    data_transforms = Compose([
        LoadImaged(keys=scan_keys, reader='NibabelReader'),
        EnsureChannelFirstd(keys=scan_keys,
                            channel_dim='no_channel'),
        # ADNI long all oriented LIA
        # Orientationd(keys=scan_keys + seg_keys, axcodes='LIA'),
        ScaleIntensityRanged(keys=scan_keys,
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        ToTensord(keys=scan_keys,
                  track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_Mindboggle(cfg: CFG,
                         # val,
                         *args,
                         **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    scan_files = []
    seg_files = []
    scan_dirs = sorted(os.listdir(data_dir))
    for scan_dir in scan_dirs:
        scan_file = os.path.join(data_dir, scan_dir,
                                 't1weighted_brain.MNI152.nii.gz')
        if os.path.exists(scan_file):
            # if ('Colin27-1' in scan_dir) \
            #        or ('OASIS' in scan_dir):
            if 'Colin27-1' in scan_dir:
                print(scan_dir)
                continue
            seg_file = os.path.join(data_dir, scan_dir,
                                    'labels.DKT31.manual+aseg.MNI152.nii.gz')
            if os.path.exists(seg_file):
                scan_files.append(scan_file)
                seg_files.append(seg_file)

    print(len(scan_files))

    data_dicts = [{
        'path': scan_file,
        'image': scan_file,
        'label': seg_file
    } for (scan_file, seg_file) in zip(scan_files, seg_files)]
    data_dicts = data_dicts[slice(*cfg.dataset_slice)]

    data_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        Orientationd(keys=['image', 'label'], axcodes='LIA'),
        ScaleIntensityRanged(keys=['image'],
                             a_min=0.0,
                             upper=99.99,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        # MindBoggleOneHotd(keys=['label']),
        ADNIOneHotd(keys=['label']),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])
    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_LPBA(cfg: CFG,
                   *args,
                   **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in range(1, 41):
        d_dict = {}
        scan_file = os.path.join(data_dir, f'S{i:02d}/Warped.nii.gz')
        seg_file = os.path.join(data_dir, f'S{i:02d}/S{i:02d}.delineation.structure.label.nii.gz')
        d_dict.update({
            'image': scan_file,
            'label': seg_file,
        })
        data_dicts.append(d_dict)

    data_dicts = data_dicts[slice(*cfg.dataset_slice)]

    data_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        # NOTE: if inference with model trained on OASIS, set the orientation to 'LIA'
        Orientationd(keys=['image', 'label'], axcodes='LIA'),
        # MonaiScaleIntensityRanged(keys=['image'],
        #                           a_min=0.0, a_max=256.0,
        #                           b_min=0.0, b_max=1.0),
        ScaleIntensityRanged(keys=['image'],
                             a_min=0.0,
                             upper=99.99,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        LesionOneHotd(keys=['label']),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_ixi(cfg: CFG,
                  *args,
                  **kwargs):
    '''
        ScaleIntensityRanged(upper=99.99),
        ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        both will fail the training
    '''
    data_dir = resolve_path(cfg.data_dir)
    df_path = resolve_path(cfg.df_path)
    with open(df_path, 'r') as f:
        file_list = json.load(f)
    data_paths = []
    for file in file_list:
        data_paths.append(os.path.join(data_dir, file))
    data_paths = data_paths[slice(*cfg.dataset_slice)]
    print(len(data_paths))
    data_transforms = Compose([
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        ADNIOneHotd(keys=['label']),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])
    dataset = CacheDataset(
        IXIBrainDataset(data_paths),
        transform=data_transforms,
        *args, **kwargs)
    return dataset


def load_data_LUMIR(cfg: CFG,
                    *args, **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in list(range(0, 3384)):
        if os.path.exists(os.path.join(data_dir, f'LUMIRMRI_{i:04d}_0000.nii.gz')):
            data_dicts.append(
                {'image': os.path.join(data_dir, f'LUMIRMRI_{i:04d}_0000.nii.gz')}
            )
    print(len(data_dicts))
    data_dicts = data_dicts[slice(*cfg.dataset_slice)]
    data_transforms = [LoadImaged(keys=['image']),
                       EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
                       # NOTE: if inference with model trained on OASIS, set the orientation to 'LIA'
                       Orientationd(keys=['image'], axcodes='LIA'), # 160, 192, 224
                       ScaleIntensityRanged(keys=['image'],
                                            a_min=0.0,
                                            upper=99.99,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=False),
                       ToTensord(keys=['image'], track_meta=False)]

    dataset = CacheDataset(data=data_dicts,
                           transform=Compose(data_transforms),
                           *args,
                           **kwargs)
    return dataset

def load_valdata_LUMIR(cfg:CFG,*args, **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in range(3454,4043):
        if os.path.exists(os.path.join(data_dir, 'simplelabelsVal', f'LUMIRMRI_{i:04d}_0000.nii.gz')):
            data_dicts.append(
                {
                'image': os.path.join(data_dir, 'imagesVal', f'LUMIRMRI_{i:04d}_0000.nii.gz'),
                'label': os.path.join(data_dir, 'simplelabelsVal', f'LUMIRMRI_{i:04d}_0000.nii.gz')
                }
            )
    print(len(data_dicts))
    data_dicts = data_dicts[slice(*cfg.dataset_slice)]
    data_transforms = [LoadImaged(keys=['image', 'label']),
                       EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
                       # NOTE: if inference with model trained on OASIS, set the orientation to 'LIA'
                       Orientationd(keys=['image', 'label'], axcodes='LIA'), # 160, 192, 224
                       ScaleIntensityRanged(keys=['image'],
                                            a_min=0.0,
                                            upper=99.99,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=False),
                        # ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                       ToTensord(keys=['image', 'label'], track_meta=False)]

    dataset = CacheDataset(data=data_dicts,
                           transform=Compose(data_transforms),
                           *args,
                           **kwargs)
    return dataset
