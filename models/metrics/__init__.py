from monai.metrics import (DiceMetric, HausdorffDistanceMetric, PSNRMetric,
                           SSIMMetric, SurfaceDistanceMetric)

from ..builder import METRICS
from .sdlogjac import SDlogDetJac
from .psnr import FgPSNR
from .tre import TargetRegistrationError

METRICS.register_module('dice', module=DiceMetric)
METRICS.register_module('haus_dist', module=HausdorffDistanceMetric)
METRICS.register_module('surf_dist', module=SurfaceDistanceMetric)
METRICS.register_module('ssim', module=SSIMMetric)
METRICS.register_module('psnr', module=PSNRMetric)

__all__ = ['SDlogDetJac', 'FgPSNR', 'TargetRegistrationError']
