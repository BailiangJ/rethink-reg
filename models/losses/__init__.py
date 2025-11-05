from monai.losses import DiceLoss, LocalNormalizedCrossCorrelationLoss

from ..builder import LOSSES
from .diffusion_regularizer import GradientDiffusionLoss
from .flow_loss import FlowLoss
from .inverse_consistency import InverseConsistentLoss
from .lncc import LocalNormalizedCrossCorrelationLoss
from .long_constraint import LongitudinalConsistentLoss
from .mind import MINDSSCLoss
from .icon import ICONLoss, GradICONLoss
from .np_jacdet import NonPositiveJacDetLoss

# LOSSES.register_module('lncc', module=LocalNormalizedCrossCorrelationLoss)
LOSSES.register_module('dice_loss', module=DiceLoss)
