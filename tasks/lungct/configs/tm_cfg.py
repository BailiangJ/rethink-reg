# Effective objective: L_sim + reg_weight * L_diffusion. Every auxiliary loss
# weight (GradICON, Dice, non-positive Jacobian, TRE) is 0.0 in the paper setup.

import torch

device = 'cuda'
amp_dtype = 'bfloat16' if device == 'cpu' else 'float16'
use_amp = True
image_size = [224, 192, 224]

# wandb
project = 'NLST'
group = 'tm-half'
name = 'tm-flip_1lncc+1diffusion'

# output directory
out_path = './tm_outputs/exp2'
model_dir = 'saved_models'

# data
# data_dir = "$DATA_ROOT/NLST/"
data_dir = "$DATA_ROOT/NLST/"
cache_rate = 1.0
num_workers = -1
batch_size = 1
# data_indexs = list(range(1, 111)) + list(range(200, 250))
data_indexs = list(range(1, 111))

load_model = None

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=2.0,
                        flow_scale=2.0,
                        interp_mode='bilinear')

sim_weight = 1.0
reg_weight = 1.0
gradicon_weight = 0.0

lr = 1e-4
lr_decay = 0.996
start_epoch = 0
max_epochs = 200
save_interval = 50

# common unsupervised losses
# similarity loss
# sim_loss_cfg = dict(type='IntensityLoss', penalty='l2', weight=10.0)
sim_loss_cfg = dict(type='ncc',
                    spatial_dims=3,
                    kernel_size=9,
                    smooth_nr=0.0,
                    smooth_dr=1e-5,
                    fast=True,
                    weight=sim_weight)

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

# tre on keypoints
tre_cfg = dict(type='tre',
               image_size=image_size,
               spacing=(1.0, 1.0, 1.0),
               interp_mode='bilinear')
tre_weight = 0.0
# dice loss
dice_loss_cfg = dict(type='dice_loss',
                     include_background=False,
                     reduction='mean')
dice_weight = 0.0

model_cfg = dict(
    type='TransMorph3D',
    config=dict(
        spatial_dims=3,
        if_transskip=True,
        if_convskip=True,
        patch_size=4,
        in_chans=2,
        embed_dim=96,
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
    )
)
