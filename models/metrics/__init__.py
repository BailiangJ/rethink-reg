from monai.metrics import (DiceMetric, HausdorffDistanceMetric, PSNRMetric,
                           SSIMMetric, SurfaceDistanceMetric)

from ..builder import METRICS
from .sdlogjac import SDlogDetJac
from .sdlogjac_2d import SDlogDetJac2D, JacDet2D
from .psnr import FgPSNR
from .tre import TargetRegistrationError
from .digital_diffeomorphism import (calc_jac_dets, calc_measurements, get_identity_grid)

METRICS.register_module('dice', module=DiceMetric)
METRICS.register_module('haus_dist', module=HausdorffDistanceMetric)
METRICS.register_module('surf_dist', module=SurfaceDistanceMetric)
METRICS.register_module('ssim', module=SSIMMetric)
METRICS.register_module('psnr', module=PSNRMetric)
