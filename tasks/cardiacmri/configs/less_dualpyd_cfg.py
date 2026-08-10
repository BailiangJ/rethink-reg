# Effective objective: L_sim + reg_weight * L_diffusion. Every auxiliary loss
# weight (GradICON, Dice, non-positive Jacobian, TRE) is 0.0 in the paper setup.

# import torch

device = 'cuda'
amp_dtype = 'bfloat16' if device == 'cpu' else 'float16'
use_amp = True
image_size = [128, 128]

# wandb
project = 'ACDC'
group = 'less-dwp-2D'
name = 'less_dwp-2D-1mse+0.05diffusion+aug'

# output directory
out_path = './less_dwp_outputs/exp1'
model_dir = 'saved_models'

# data
df_path = "training.csv"
# data_dir = "$DATA_ROOT/ACDC/training"
data_dir = "$DATA_ROOT/ACDC/training"
cache_rate = 1.0
num_workers = -1
batch_size = 1
dataset_slice = (0, 80, None)

load_model = None

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=2.0,  # Adjusted for half resolution
                        flow_scale=2.0,     # Adjusted for half resolution
                        interp_mode='bilinear')

sim_weight = 1.0
reg_weight = 0.05
gradicon_weight = 0.0

scale_pyramid = [16, 8, 4]
scale_loss_weights = [1 / 16, 1 / 8, 1 / 4]

lr = 1e-4
lr_decay = 0.996
start_epoch = 0
max_epochs = 400
save_interval = 50

# common unsupervised losses
# similarity loss
sim_loss_cfg = dict(type='IntensityLoss', penalty='l2', weight=sim_weight)
# sim_loss_cfg = dict(type='ncc',
#                     spatial_dims=2,
#                     kernel_size=3,
#                     smooth_nr=0.0,
#                     smooth_dr=1e-5,
#                     fast=True,
#                     weight=sim_weight)


# smoothness regularization
reg_loss_cfg = dict(type='diffusion',
                    penalty='l2',
                    loss_mult=1.0,
                    weight=reg_weight)

gradicon_loss_cfg = dict(type='GradICONLoss',
                         flow_loss_cfg=dict(type='FlowLoss', penalty='l2'),
                         image_size=image_size,
                         interp_mode='bilinear',
                         compose_detach=True,
                         delta=1e-3,
                         weight=gradicon_weight,
                         )

# dice loss
dice_loss_cfg = dict(type='dice_loss',
                     include_background=False,
                     reduction='mean')
dice_weight = 0.0

model_cfg = dict(
    type='LessNet_DualWarpPy',
    config=dict(
        bidirectional=True,
        encoder_cfg=dict(
            spatial_dims=2,
            out_indices=[1, 2, 3, 4],
        ),
        decoder_cfg=dict(
            image_size=image_size,
            spatial_dims=2,
            skip_channels=[6, 6, 6, 6],
            out_channels=[128, 96, 64, 48],
            out_indices=[4, 3, 2, 1],
            composition='compose',  # 'add'
            block_config=dict(
                kernel_size=[3, 3],
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
