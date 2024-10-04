from monai.losses import DiceLoss, LocalNormalizedCrossCorrelationLoss

from ..builder import LOSSES
from .diffusion_regularizer import GradientDiffusionLoss
from .flow_loss import FlowLoss
from .inverse_consistency import InverseConsistentLoss
from .lncc import LocalNormalizedCrossCorrelationLoss
from .mind import MINDSSCLoss
from .icon import ICONLoss, GradICONLoss

# LOSSES.register_module('lncc', module=LocalNormalizedCrossCorrelationLoss)
LOSSES.register_module('dice_loss', module=DiceLoss)

__all__ = [
    'GradientDiffusionLoss',
    'FlowLoss',
    'InverseConsistentLoss',
    'MINDSSCLoss',
    'LocalNormalizedCrossCorrelationLoss',
    'ICONLoss',
    'GradICONLoss'
]
