# Effective objective: L_sim + reg_weight * L_diffusion.


device = "cuda"
amp_dtype = "bfloat16" if device == "cpu" else "float16"
use_amp = True
image_size = [160, 192, 224]

# wandb
project = "LUMIR"
group = "vfa"
name = "vfa-1lncc+0.5diffusion"

# output directory
out_path = "./vfa_outputs/exp0"
model_dir = "saved_models"

# load_model = "./"
load_model = None

# data
cache_rate = 1.0
num_workers = -1
batch_size = 2
dataset_slice = (None, 500, 1)
# dataset_slice = (None, None, 1)

# multiple datasets
lumir_cfg = dict(
    data_dir="$DATA_ROOT/LUMIR25/imagesTr",
    # data_dir="$DATA_ROOT/LUMIR25/imagesTr",
    data_indexs=list(range(0, 500)),
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)

lumir_val_cfg = dict(
    data_dir="$DATA_ROOT/LUMIR25",
    # data_dir="$DATA_ROOT/LUMIR25",
    image_size=image_size,
    dataset_slice=(None, None, 1),
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)

# registration head
registration_cfg = dict(
    type="RegistrationHead",
    image_size=image_size,
    spatial_scale=2.0,
    flow_scale=2.0,
    interp_mode="bilinear",
)

scale_pyramid = [16, 8, 4]
scale_loss_weights = [1 / 16, 1 / 8, 1 / 4]

sim_weight = 1.0
reg_weight = 0.5
dice_weight = 0.0
gradicon_weight = 0.0
np_jacdet_weight = 0.0

lr = 1e-4
lr_decay = 0.996
start_epoch = 0
max_epochs = 100
save_interval = 20
val_interval = 5

# common unsupervised losses
# similarity loss
# sim_loss_cfg = dict(type='IntensityLoss', penalty='l2', weight=10.0)
sim_loss_cfg = dict(
    type="ncc",
    spatial_dims=3,
    kernel_size=9,
    smooth_nr=0.0,
    smooth_dr=1e-5,
    weight=sim_weight,
)

# flow losses
flow_loss_cfg = dict(
    type="FlowLoss",
    penalty="l2",
    # penalty="charbonnier",
    # ch_cfg=dict(alpha=0.45, eps=0.001, truncate=None),
)

# smoothness regularization
reg_loss_cfg = dict(type="diffusion", penalty="l2", loss_mult=2.0, weight=reg_weight)

# dice loss
dice_loss_cfg = dict(
    type="dice_loss", include_background=False, reduction="mean", weight=dice_weight
)

# gradICON loss
gradicon_loss_cfg = dict(
    type="GradICONLoss",
    flow_loss_cfg=flow_loss_cfg,
    image_size=image_size,
    interp_mode="bilinear",
    compose_detach=False,
    delta=1e-3,
    weight=gradicon_weight,
)

np_jacdet_loss_cfg = dict(
    type="np_jacdet",
    weight=np_jacdet_weight,
)

model_cfg = dict(
    type='VXMVFA',
    config=dict(
        bidirectional=True,
        encoder_cfg=dict(
            spatial_dims=3,
            in_chan=1,
            down=True,
            out_channels=[16, 48, 64, 96, 128],
            out_indices=[1, 2, 3, 4],
            block_config=dict(
                kernel_size=3,
                res_skip=False,
                down_first=True,
                conv_down=True,
                bias=True,
                pool_name=('max', {'kernel_size': 2}),
                norm_name=('INSTANCE', {'affine': False}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                dropout=None,
            ),
        ),
        decoder_cfg=dict(
            image_size=image_size,
            skip_channels=[128, 96, 64, 48],
            proj_channels=[96, 64, 48, 32],
            corr_radius=[1, 1, 1, 1],
            out_indices=[4, 3, 2, 1],
            beta=[0.1, 0.125, 0.25, 0.5],
            norm_vectors=False,
            scale=True,
            corr_mode='einsum',
            composition='compose',
        )

    )
)
