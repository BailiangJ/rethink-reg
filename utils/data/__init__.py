"""Dataset loaders, one module per registration task.

Module names follow the task (`tasks/brainmri`, `tasks/lungct`, ...); the
loader functions keep their dataset names, since each one loads a specific
dataset. All of these are re-exported from `utils`, so task scripts import
them as `from utils import load_data_LUMIR`.
"""

from .dataset_utils import PairDataset, IXIBrainDataset, pkload
from .brainmri_data_utils import (load_data_oasis, load_data_adni_tps,
                                  load_data_adni_tps_infer, load_data_t1_t2,
                                  load_data_Mindboggle, load_data_LPBA,
                                  load_data_adni, load_data_ixi,
                                  load_data_LUMIR, load_valdata_LUMIR)
from .lungct_data_utils import (load_data_NLST, load_data_NLST_fg,
                                load_data_ThoraxCBCT, load_data_Lung250M)
from .cardiacmri_data_utils import (load_data_ACDC, load_data_ACDC_2d,
                                    load_data_MMs_2d)
from .abdomenmrct_data_utils import (load_data_AbdomenCT, load_data_AbdomenMR,
                                     load_data_AbdomenMRCT_pair,
                                     load_data_AmosMR, load_data_AbdCTCT,
                                     SPLITS as ABDOMENMRCT_SPLITS,
                                     SPLIT_ORGAN_NAMES as ABDOMENMRCT_ORGAN_NAMES)
from .abdomenctct_data_utils import (load_data_PSMAReg_pair,
                                     PSMAREG_ORGAN_LABELS,
                                     PSMAREG_ORGAN_NAMES,
                                     get_PSMAReg_organ_names)

__all__ = [
    'PairDataset', 'IXIBrainDataset', 'pkload',
    'load_data_oasis', 'load_data_adni_tps', 'load_data_adni_tps_infer',
    'load_data_t1_t2', 'load_data_Mindboggle', 'load_data_LPBA',
    'load_data_adni', 'load_data_ixi', 'load_data_LUMIR', 'load_valdata_LUMIR',
    'load_data_NLST', 'load_data_NLST_fg', 'load_data_ThoraxCBCT',
    'load_data_Lung250M',
    'load_data_ACDC', 'load_data_ACDC_2d', 'load_data_MMs_2d',
    'load_data_AbdomenCT', 'load_data_AbdomenMR', 'load_data_AbdomenMRCT_pair',
    'load_data_AmosMR', 'load_data_AbdCTCT',
    'ABDOMENMRCT_SPLITS', 'ABDOMENMRCT_ORGAN_NAMES',
    'load_data_PSMAReg_pair', 'PSMAREG_ORGAN_LABELS', 'PSMAREG_ORGAN_NAMES',
    'get_PSMAReg_organ_names',
]
