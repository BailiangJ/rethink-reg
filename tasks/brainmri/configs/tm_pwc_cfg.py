# Effective objective: L_sim + reg_weight * L_diffusion.


# import torch

device = "cuda"
amp_dtype = "bfloat16" if device == "cpu" else "float16"
use_amp = True
image_size = [160, 192, 224]

# wandb
project = "LUMIR"
group = "tm-pwc"
name = "tm_pwc-1lncc+0.5diffusion"

# output directory
out_path = "./tm_pwc_outputs/exp0"
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
    data_indexs=list(range(0, 800)),
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)

lumir_val_cfg = dict(
    data_dir="$DATA_ROOT/LUMIR25",
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
    type='TM3D_PWC',
    config=dict(
        spatial_dims=3,
        bidirectional=True,
        if_transskip=True,
        if_convskip=True,
        patch_size=4,
        in_chans=1,
        embed_dim=48,
        depths=(2, 2, 4, 2),
        num_heads=(4, 4, 8, 8),
        window_size=(5, 6, 7),
        mlp_ratio=4,
        pat_merg_rf=4,
        qkv_bias=False,
        drop_rate=0,
        drop_path_rate=0.3,
        ape=False,
        spe=False,
        rpe=True,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(0, 1, 2, 3),
        reg_head_chan=16,
        pre_out_channels=128,
        decoder_cfg=dict(
            image_size=image_size,
            spatial_dims=3,
            skip_channels=[192, 96, 48, 24],  # 4E, 2E, E, E//2
            corr_radius=[1, 1, 1, 1],
            out_channels=[96, 64, 32, 16],
            out_indices=[4, 3, 2, 1],  # 1/16, 1/8, 1/4, 1/2 spatial size
            composition='compose',  # 'add'
            corr_mode='cuda',  # 'for', 'einsum', 'cuda'
            block_config=dict(
                kernel_size=3,
                res_skip=False,
                up_transp_conv=True,
                transp_bias=False,
                upsample_kernel_size=2,
                bias=True,
                norm_name=('INSTANCE', {'affine': False}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                dropout=None,
            ),
        )
    )
)
