# Effective objective: L_sim + reg_weight * L_diffusion.

import torch

device = 'cuda'
amp_dtype = 'bfloat16' if device == 'cpu' else 'float16'
use_amp = True
image_size = [160, 128, 160]

# wandb
project = 'PSMAReg'
group = 'lku-half-paired'
name = 'lku-pair-flip+1ncc+1diffusion'

# output directory
out_path = './lku_outputs/exp0'
model_dir = 'saved_models'
#
load_model = None

# data
cache_rate = 1.0
num_workers = -1
batch_size = 1
pairwise = True
data_dir = "$DATA_ROOT/PSMAReg/PSMAReg_CT_affine_crop160x128x160_FU01_no0359"
split_json = './dataset_split.json'
# organ_list = ['liver', 'spleen', 'kidney_right', 'kidney_left',
#               'stomach', 'pancreas', 'aorta', 'inferior_vena_cava']
label_list = [5, 1, 2, 3, 6, 7, 52, 63]

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=2.0,
                        flow_scale=2.0,
                        interp_mode='bilinear')

sim_weight = 1.0
reg_weight = 1.0
dice_weight = 0.0

lr = 1e-4
lr_decay = 0.996
start_epoch = 0
max_epochs = 200
save_interval = 50

# common unsupervised losses
# similarity loss
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

# dice loss
dice_loss_cfg = dict(type='dice_loss',
                     include_background=True,
                     reduction='mean',
                     weight=dice_weight)



model_cfg = dict(
    type='LKUNet',
    in_channel=2,
    out_channel=3,
    enc_feat_channel=8,
    dec_feat_channel=8,
    large_kernel_size=5,
    bias=True

)
