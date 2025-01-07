import json
import os
import pandas as pd
from typing import (Any, Callable, Dict, Hashable, List, Mapping, Optional,
                    Sequence, Tuple, Union)
from monai.transforms import (CastToTyped, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandSpatialCropSamplesd, Resized,
                              ResizeWithPadOrCropd, ScaleIntensityd,
                              NormalizeIntensityd,
                              Spacingd, ToTensord)
from monai.transforms import ScaleIntensityRanged as MonaiScaleIntensityRanged
from monai.data import CacheDataset
from torch.utils.data import Dataset
from models import CFG
from .one_hot_encoding import OASISOneHotd, OneHotd, FreesurferOneHotd
from .scale_intensity import ScaleIntensityRanged
from .dataset import IXIBrainDataset


def load_data_oasis(cfg: CFG,
                    val: bool = False,
                    *args, **kwargs):
    scan_files = []
    seg_files = []

    for scan_dir in os.listdir(cfg.data_dir):
        if scan_dir.startswith('OASIS'):
            scan_file = os.path.join(cfg.data_dir, scan_dir, 'aligned_norm.nii.gz')
            seg_file = os.path.join(cfg.data_dir, scan_dir, 'aligned_seg35.nii.gz')
            if os.path.exists(seg_file):
                scan_files.append(scan_file)
                seg_files.append(seg_file)

    if val:
        scan_files = sorted(scan_files)
        seg_files = sorted(seg_files)

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
        Orientationd(keys=['image', 'label'], axcodes='LIA'),  # OASIS is already 'LIA' oriented
        OASISOneHotd(keys=['label']),
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
    df = pd.read_csv(cfg.df_path)  # dataframe storing dataset info about ADNI
    scan_files = []
    seg_files = []
    image_uids = df['IMAGEUID'].unique()
    for uid in image_uids:
        scan_path = os.path.join(cfg.data_dir, f'{uid}/norm.nii.gz')
        seg_path = os.path.join(cfg.data_dir, f'{uid}/aseg.nii.gz')
        if os.path.exists(scan_path) and os.path.exists(seg_path):
            scan_files.append(scan_path)
            seg_files.append(seg_path)

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
        ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        FreesurferOneHotd(keys=['label']),
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
    with open(cfg.df_path, 'r') as f:
        file_list = json.load(f)
    data_paths = []
    for file in file_list:
        data_paths.append(os.path.join(cfg.data_dir, file))

    data_paths = data_paths[slice(*cfg.dataset_slice)]
    print(len(data_paths))

    data_transforms = Compose([
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        FreesurferOneHotd(keys=['label']),
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


def load_data_LUMIR(cfg: CFG, *args, **kwargs):
    data_dicts = []
    for i in range(0, 3494):
        data_dicts.append(
            {'image': os.path.join(cfg.data_dir, f'LUMIRMRI_{i:04d}_0000.nii.gz')}
        )

    print(len(data_dicts))
    data_dicts = data_dicts[slice(*cfg.dataset_slice)]

    data_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        # NOTE: if inference with model trained on OASIS, set the orientation to 'LIA'
        Orientationd(keys=['image'], axcodes='LIA'),
        ScaleIntensityRanged(keys=['image'],
                             a_min=0.0,
                             upper=99.99,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        ResizeWithPadOrCropd(keys=['image'],
                             spatial_size=cfg.image_size,
                             mode='constant'),
        ToTensord(keys=['image'], track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_LPBA(cfg: CFG,
                   *args,
                   **kwargs):
    data_dicts = []
    for i in range(1, 41):
        d_dict = {}
        scan_file = os.path.join(cfg.data_dir, f'S{i:02d}/Warped.nii.gz')
        seg_file = os.path.join(cfg.data_dir, f'S{i:02d}/S{i:02d}.delineation.structure.label.nii.gz')
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
        OneHotd(keys=['label']),
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


def load_data_Mindboggle(cfg: CFG,
                         # val,
                         *args,
                         **kwargs):
    scan_files = []
    seg_files = []
    scan_dirs = sorted(os.listdir(cfg.data_dir))
    for scan_dir in scan_dirs:
        scan_file = os.path.join(cfg.data_dir, scan_dir,
                                 't1weighted_brain.MNI152.nii.gz')
        if os.path.exists(scan_file):
            # if ('Colin27-1' in scan_dir) \
            #        or ('OASIS' in scan_dir):
            if 'Colin27-1' in scan_dir:
                print(scan_dir)
                continue
            seg_file = os.path.join(cfg.data_dir, scan_dir,
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
        FreesurferOneHotd(keys=['label']),
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
