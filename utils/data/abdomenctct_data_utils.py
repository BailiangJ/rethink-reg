import json
import os
from pathlib import Path

from monai.data import CacheDataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ResizeWithPadOrCropd, ToTensord
from monai.transforms import ScaleIntensityRanged as MonaiScaleIntensityRanged

from ..transforms import SelectOneHotd
from models import CFG

from ..path_utils import resolve_path

# 8 core organs (TotalSegmentator label IDs). One test pair is missing right kidney
# in one timepoint, so evaluation should treat absent organ pairs as NaN.
PSMAREG_ORGAN_LABELS = [5, 1, 2, 3, 6, 7, 52, 63]
PSMAREG_ORGAN_NAMES = ['liver', 'spleen', 'kidney_right', 'kidney_left',
                       'stomach', 'pancreas', 'aorta', 'inferior_vena_cava']
PSMAREG_LABEL_TO_ORGAN = dict(zip(PSMAREG_ORGAN_LABELS, PSMAREG_ORGAN_NAMES))


def get_PSMAReg_organ_names(label_list):
    return [PSMAREG_LABEL_TO_ORGAN.get(int(label), f'label_{int(label)}') for label in label_list]


def _resolve_split_json(split_json):
    """Locate the train/val/test split file.

    Configs name it relative to the task directory ('./dataset_split.json'), so
    the CWD hit is the normal case; the tasks/abdomenctct fallback covers being
    invoked from elsewhere.
    """
    split_json = resolve_path(split_json)
    if os.path.isabs(split_json):
        return split_json

    cwd_path = os.path.abspath(split_json)
    if os.path.exists(cwd_path):
        return cwd_path

    repo_root = Path(__file__).resolve().parents[2]
    task_path = repo_root / 'tasks' / 'abdomenctct' / split_json
    if task_path.exists():
        return str(task_path)

    return cwd_path


def load_data_PSMAReg_pair(cfg: CFG,
                           split: str = 'train',
                           *args,
                           **kwargs):
    data_dir = resolve_path(cfg.data_dir)
    with open(os.path.join(data_dir, 'PSMAReg_dataset.json')) as f:
        dataset_json = json.load(f)
    with open(_resolve_split_json(cfg.split_json)) as f:
        split_subjects = set(json.load(f)[split])

    entries = {e['subject']: e for e in dataset_json['training_paired']}

    label_list = getattr(cfg, 'label_list', PSMAREG_ORGAN_LABELS)

    data_dicts = []
    for subject in sorted(split_subjects):
        e = entries[subject]
        data_dicts.append({
            'fixed': os.path.join(data_dir, e['CT']),
            'moving': os.path.join(data_dir, e['Follow-up 01 CT']),
            'fixed_label': os.path.join(data_dir, e['CT Label']),
            'moving_label': os.path.join(data_dir, e['Follow-up 01 CT Label']),
            'subject': subject,
        })

    transforms = [
        LoadImaged(keys=['fixed', 'moving', 'fixed_label', 'moving_label']),
        EnsureChannelFirstd(keys=['fixed', 'moving'], channel_dim='no_channel'),
        MonaiScaleIntensityRanged(keys=['fixed', 'moving'],
                                  a_min=-300,
                                  a_max=300,
                                  b_min=0.0,
                                  b_max=1.0,
                                  clip=True),
        SelectOneHotd(keys=['fixed_label', 'moving_label'], label_list=label_list),
    ]
    image_size = getattr(cfg, 'image_size', None)
    if image_size is not None:
        transforms.append(ResizeWithPadOrCropd(
            keys=['fixed', 'moving', 'fixed_label', 'moving_label'],
            spatial_size=image_size))
    transforms.append(ToTensord(keys=['fixed', 'moving', 'fixed_label', 'moving_label'], track_meta=False))

    data_transforms = Compose(transforms)
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)
    return dataset
