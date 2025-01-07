import numpy as np
import os
import random
import torch
from contextlib import contextmanager
from typing import Sequence
from .dataset import PairDataset, IXIBrainDataset
from .data_loading import (load_data_oasis, load_data_adni, load_data_ixi, load_data_LUMIR, load_data_LPBA,
                           load_data_Mindboggle)


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
