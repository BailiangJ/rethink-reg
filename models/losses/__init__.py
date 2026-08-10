from monai.losses import DiceLoss

from ..builder import LOSSES
from .diffusion_regularizer import GradientDiffusionLoss
from .flow_loss import FlowLoss
from .inverse_consistency import InverseConsistentLoss
from .lncc import LocalNormalizedCrossCorrelationLoss
from .long_constraint import LongitudinalConsistentLoss
from .group_consistency import GroupConsistencyLoss
from .mind import MINDSSCLoss
from .icon import ICONLoss, GradICONLoss
from .pyramid_self_flow import PyramidalDistillLoss
from .squared_lncc import SquaredNCC
from .np_jacdet import NonPositiveJacDetLoss

# LOSSES.register_module('lncc', module=LocalNormalizedCrossCorrelationLoss)
LOSSES.register_module('dice_loss', module=DiceLoss)
