# Effective objective: L_sim + reg_weight * L_diffusion.

# import torch

device = 'cuda'
amp_dtype = 'bfloat16' if device == 'cpu' else 'float16'
use_amp = True
image_size = [160, 128, 160]

# wandb
project = 'PSMAReg'
group = 'vfa-half-paired'
name = 'vfa_cuda-pair-flip+1ncc+1diffusion'

# output directory
out_path = './vfa_outputs/exp0'
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

scale_pyramid = [16, 8, 4]
scale_loss_weights = [1 / 16, 1 / 8, 1 / 4]

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
            corr_mode='cuda',  # 'for', 'einsum', 'cuda'
            composition='compose',  # 'add'
        )
    )
)
