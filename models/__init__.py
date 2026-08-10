from .builder import (
    CFG,
    MODELS,
    LOSSES,
    METRICS,
    ENCODERS,
    DECODERS,
    FLOW_ESTIMATORS,
    BACKBONES,
    REGISTRATION_HEAD,
    build,
    build_backbone,
    build_decoder,
    build_encoder,
    build_flow_estimator,
    build_loss,
    build_metrics,
    build_registration_head,
)
from .losses import (
    FlowLoss,
    GradientDiffusionLoss,
    InverseConsistentLoss,
    LongitudinalConsistentLoss,
    GroupConsistencyLoss,
    PyramidalDistillLoss,
    SquaredNCC,
    MINDSSCLoss,
    ICONLoss,
    GradICONLoss,
)
from .metrics import SDlogDetJac, JacDet, TargetRegistrationError
from .utils import (
    RegistrationHead,
    DownSizeRegistrationHead,
    SVFIntegrateHead,
    MultiScaleRegistrationHead,
    MultiScaleAdditionRegistrationHead,
    ResizeFlow,
    Warp,
    Composite,
)
from .backbones import UNet
from .flow_estimators import VoxelMorph
# Imported last, and for its side effect: every module under .networks decorates
# its class with @FLOW_ESTIMATORS.register_module(), so `import models` is what
# makes build_flow_estimator(cfg.model_cfg) able to resolve 'VXMPlus_PWC' & co.
# The submodules import from ..builder / ..utils directly (not `from .. import`)
# so that they do not re-enter this file while it is still executing.
from . import networks  # noqa: F401

__all__ = [
    'CFG',
    'MODELS', 'LOSSES', 'METRICS', 'ENCODERS', 'DECODERS', 'FLOW_ESTIMATORS',
    'BACKBONES', 'REGISTRATION_HEAD', 'build',
    'build_backbone', 'build_loss', 'build_metrics', 'build_flow_estimator',
    'build_decoder', 'build_encoder', 'build_registration_head',
    'FlowLoss', 'GradientDiffusionLoss', 'InverseConsistentLoss',
    'LongitudinalConsistentLoss', 'GroupConsistencyLoss', 'PyramidalDistillLoss',
    'SquaredNCC', 'MINDSSCLoss', 'ICONLoss', 'GradICONLoss',
    'SDlogDetJac', 'JacDet', 'TargetRegistrationError',
    'RegistrationHead', 'DownSizeRegistrationHead', 'SVFIntegrateHead',
    'MultiScaleRegistrationHead', 'MultiScaleAdditionRegistrationHead',
    'ResizeFlow', 'Warp', 'Composite',
    'UNet', 'VoxelMorph',
]
