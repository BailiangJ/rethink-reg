import os
import json
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union
import numpy as np
from monai.data import CacheDataset
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged, ToTensord, Transform)
from monai.transforms import ScaleIntensityRanged as MonaiScaleIntensityRanged

from ..transforms import ScaleIntensityRanged, SelectOneHotd
from models import CFG
from ..path_utils import resolve_path


# AbdomenMRCT (Learn2Reg) ships 16 MR/CT pairs, split in half. The two halves do
# not share a labelling scheme: 1-8 carry the four manual organ labels, while 9-16
# are labelled with TotalSegmentator IDs, so an index alone does not determine the
# label list. Training uses one half and evaluation the other -- see the --split
# flag on tasks/abdomenmrct/{train,evaluate}.py.
SPLITS = {
    'tr': dict(indices=list(range(1, 9)),
               image_dir='imagesTr',
               label_dir='labelsTr',
               # liver, spleen, right kidney, left kidney
               label_list=[1, 2, 3, 4]),
    'ts': dict(indices=list(range(9, 17)),
               image_dir='imagesTs',
               label_dir='TSlabelsTs',
               # same four organs, as TotalSegmentator IDs
               label_list=[5, 2, 3, 1]),
}

# label id -> organ name, per split. Evaluation reports metrics per organ.
SPLIT_ORGAN_NAMES = {
    'tr': {1: 'liver', 2: 'spleen', 3: 'right_kidney', 4: 'left_kidney'},
    # TotalSegmentator IDs: spleen=1, kidney_right=2, kidney_left=3, liver=5
    'ts': {5: 'liver', 2: 'right_kidney', 3: 'left_kidney', 1: 'spleen'},
}


def load_data_AbdomenMRCT_pair(cfg: CFG,
                               split: str = 'ts',
                               one_hot: bool = True,
                               *args,
                               **kwargs):
    """Load the MR/CT pairs of one AbdomenMRCT half.

    Args:
        cfg: needs only ``data_dir``; the index range, folders and label list
            come from :data:`SPLITS`.
        split: ``'tr'`` (indices 1-8) or ``'ts'`` (indices 9-16).
        one_hot: if True, ``mr_label``/``ct_label`` are one-hot stacks over the
            split's label list -- what training expects. If False they stay raw
            label maps, which lets evaluation intersect the labels actually
            present in each pair instead of scoring absent organs as zero.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLITS)}.")
    spec = SPLITS[split]

    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in spec['indices']:
        data_dicts.append({
            'mr': os.path.join(data_dir, spec['image_dir'], f'AbdomenMRCT_{i:04d}_0000.nii.gz'),
            'ct': os.path.join(data_dir, spec['image_dir'], f'AbdomenMRCT_{i:04d}_0001.nii.gz'),
            'mr_label': os.path.join(data_dir, spec['label_dir'], f'AbdomenMRCT_{i:04d}_0000.nii.gz'),
            'ct_label': os.path.join(data_dir, spec['label_dir'], f'AbdomenMRCT_{i:04d}_0001.nii.gz'),
            'id': i,
        })

    keys = ['mr', 'ct', 'mr_label', 'ct_label']
    data_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=['mr', 'ct']),
        MonaiScaleIntensityRanged(keys=['ct'],
                                  a_min=-450,
                                  a_max=450,
                                  b_min=0.0,
                                  b_max=1.0,
                                  clip=True),
        ScaleIntensityRanged(keys=['mr'],
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
    ]
    if one_hot:
        data_transforms.append(
            SelectOneHotd(keys=['mr_label', 'ct_label'], label_list=spec['label_list']))
    data_transforms.append(ToTensord(keys=keys, track_meta=False))

    dataset = CacheDataset(data=data_dicts, transform=Compose(data_transforms), *args, **kwargs)
    return dataset



def load_data_AbdomenMR(cfg: CFG,
                        *args,
                        **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.mr_data_indexs:
        if os.path.exists(os.path.join(data_dir, 'imagesTr', f'AbdomenMRCT_{i:04d}_0000.nii.gz')):
            data_dicts.append({
                'image': os.path.join(data_dir, 'imagesTr', f'AbdomenMRCT_{i:04d}_0000.nii.gz'),
                'label': os.path.join(data_dir, 'labelsTr', f'AbdomenMRCT_{i:04d}_0000.nii.gz')
            })
        else:
            print(f'subject {i} not found!')

    data_transform = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        ScaleIntensityRanged(keys=['image'],
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        SelectOneHotd(keys=['label'], label_list=cfg.label_list),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts, transform=data_transform, *args, **kwargs)
    return dataset


def load_data_AbdomenCT(cfg: CFG,
                        *args,
                        **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.ct_data_indexs:
        if os.path.exists(os.path.join(data_dir, 'imagesTr', f'AbdomenMRCT_{i:04d}_0001.nii.gz')):
            data_dicts.append({
                'image': os.path.join(data_dir, 'imagesTr', f'AbdomenMRCT_{i:04d}_0001.nii.gz'),
                'label': os.path.join(data_dir, 'labelsTr', f'AbdomenMRCT_{i:04d}_0001.nii.gz')
            })
        else:
            print(f'subject {i} not found!')

    data_transform = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        MonaiScaleIntensityRanged(keys=['image'],
                                  a_min=-450,
                                  a_max=450,
                                  b_min=0.0,
                                  b_max=1.0,
                                  clip=True),
        SelectOneHotd(keys=['label'], label_list=cfg.label_list),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts, transform=data_transform, *args, **kwargs)
    return dataset

def load_data_AmosMR(cfg:CFG,
                     *args,
                     **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.mr_data_indexs:
        if os.path.exists(os.path.join(data_dir, 'imagesTr_resampled', f'amos_{i:04d}.nii.gz')):
            data_dicts.append({
                'image': os.path.join(data_dir, 'imagesTr_resampled', f'amos_{i:04d}.nii.gz'),
                'label': os.path.join(data_dir, 'labelsTr_resampled', f'amos_{i:04d}.nii.gz')
            })
        else:
            print(f'subject {i} not found!')

    data_transform = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        ScaleIntensityRanged(keys=['image'],
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
        SelectOneHotd(keys=['label'], label_list=cfg.label_list),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts, transform=data_transform, *args, **kwargs)
    return dataset

class SliceTransform(Transform):
    def __init__(self, keys: Union[str, Sequence[str]], slice_obj: tuple):
        """
        Args:
            keys: keys of the corresponding items to be transformed
            slice_obj: tuple of slice objects for spatial dimensions only (e.g. (slice(64, None), slice(None)))
                      The channel dimension will be preserved automatically
        """
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.slice_obj = slice_obj

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Add slice(None) for channel dimension at the start
                full_slice = (slice(None),) + self.slice_obj
                d[key] = d[key][full_slice]
        return d

def load_data_AbdCTCT(cfg:CFG,
                      *args,
                      **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    data_dicts = []
    for i in cfg.ct_data_indexs:
        if os.path.exists(os.path.join(data_dir, 'imagesTr', f'AbdomenCTCT_{i:04d}_0000.nii.gz')):
            data_dicts.append({
                'image': os.path.join(data_dir, 'imagesTr', f'AbdomenCTCT_{i:04d}_0000.nii.gz'),
                'label': os.path.join(data_dir, 'labelsTr', f'AbdomenCTCT_{i:04d}_0000.nii.gz')
            })
        else:
            print(f'subject {i} not found!')

    data_transform = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        MonaiScaleIntensityRanged(keys=['image'],
                                  a_min=-450,
                                  a_max=450,
                                  b_min=0.0,
                                  b_max=1.0,
                                  clip=True),
        SliceTransform(keys=['image', 'label'], slice_obj=(slice(None), slice(None), slice(64, None))),
        SelectOneHotd(keys=['label'], label_list=cfg.label_list),
        ToTensord(keys=['image', 'label'], track_meta=False)
    ])

    dataset = CacheDataset(data=data_dicts, transform=data_transform, *args, **kwargs)
    return dataset
