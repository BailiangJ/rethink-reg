import os
import random
import nibabel as nib
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
from .path_utils import resolve_path
from .data import (PairDataset, IXIBrainDataset, pkload,
                   load_data_oasis, load_data_adni_tps,
                   load_data_adni_tps_infer, load_data_t1_t2,
                   load_data_Mindboggle, load_data_LPBA,
                   load_data_adni, load_data_ixi, load_data_LUMIR,
                   load_valdata_LUMIR,
                   load_data_NLST, load_data_NLST_fg, load_data_ThoraxCBCT,
                   load_data_Lung250M,
                   load_data_ACDC, load_data_ACDC_2d, load_data_MMs_2d,
                   load_data_AbdomenCT, load_data_AbdomenMR,
                   load_data_AbdomenMRCT_pair,
                   load_data_AmosMR, load_data_AbdCTCT,
                   ABDOMENMRCT_SPLITS, ABDOMENMRCT_ORGAN_NAMES,
                   load_data_PSMAReg_pair, PSMAREG_ORGAN_LABELS,
                   PSMAREG_ORGAN_NAMES, get_PSMAReg_organ_names)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


def worker_init_fn(worker_id):
    """Check https://github.com/Project-MONAI/MONAI/issues/1068."""
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(worker_info.seed %
                                                       (2 ** 32))
    except AttributeError:
        pass


@contextmanager
def optional_context(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def combined_dataloader(dataloaders, max_iterations=None):
    iters = [iter(cycle(d)) for d in dataloaders]
    if max_iterations is None:
        max_iterations = max(len(d) for d in dataloaders)

    for _ in range(max_iterations):
        for i in range(len(dataloaders)):
            yield next(iters[i])


from monai.transforms.utils_pytorch_numpy_unification import percentile


def rescale(x, a_min, upper, b_min, b_max):
    a_max: float = percentile(x, upper)
    x = ((x - a_min) / (a_max - a_min)) * (b_max - b_min) + b_min
    return x


def iterate_paired_dataloaders(dataloader_a, dataloader_b, reset_shorter=False, max_samples=None):
    """
    Iterate through two dataloaders of potentially different lengths, yielding paired samples.

    Args:
        dataloader_a: First dataloader
        dataloader_b: Second dataloader
        reset_shorter: If True, reset the shorter dataloader when exhausted.
                      If False, one epoch completes when either dataloader is exhausted.
        max_samples: Maximum number of sample pairs to yield.
                    If None: iterate for max(len(dataloader_a), len(dataloader_b)) samples
                    If int: iterate for specified number of samples or until dataloaders are exhausted

    Yields:
        Tuple of (sample_a, sample_b)
    """
    # Verify inputs are valid
    if len(dataloader_a) == 0 or len(dataloader_b) == 0:
        raise ValueError("One or both dataloaders are empty")

    # When max_samples is None, we want to iterate for the length of the longer dataloader
    if max_samples is None:
        max_samples = max(len(dataloader_a), len(dataloader_b))
        # When max_samples is set to longer dataloader length, we need to reset the shorter one
        reset_shorter = True

    iter_a = iter(dataloader_a)
    iter_b = iter(dataloader_b)
    samples_yielded = 0

    while True:
        # Check if we've reached the maximum number of samples
        if samples_yielded >= max_samples:
            break

        # Try to get next sample from dataloader A
        try:
            sample_a = next(iter_a)
        except StopIteration:
            if reset_shorter:
                # Reset iterator A
                iter_a = iter(dataloader_a)
                try:
                    sample_a = next(iter_a)
                except StopIteration:
                    raise RuntimeError("Dataloader A became empty during iteration")
            else:
                # If we don't reset shorter dataloaders, we're done with this epoch
                break

        # Try to get next sample from dataloader B
        try:
            sample_b = next(iter_b)
        except StopIteration:
            if reset_shorter:
                # Reset iterator B
                iter_b = iter(dataloader_b)
                try:
                    sample_b = next(iter_b)
                except StopIteration:
                    raise RuntimeError("Dataloader B became empty during iteration")
            else:
                # If we don't reset shorter dataloaders, we're done with this epoch
                break

        samples_yielded += 1
        yield sample_a, sample_b

    print(f"Completed iteration with {samples_yielded} samples (max_samples={max_samples})")


def save_nifti(arr, path, affine=None, header=None):
    if isinstance(arr, torch.Tensor):
        data = arr.float().detach().cpu().squeeze().numpy()
    elif isinstance(arr, np.ndarray):
        data = arr.squeeze().astype(np.float32)
    else:
        raise ValueError("arr should be either torch.Tensor or np.ndarray")

    nifti_img = nib.Nifti1Image(data, affine=affine, header=header)
    nib.save(nifti_img, path)
