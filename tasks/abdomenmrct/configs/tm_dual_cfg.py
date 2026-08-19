# Effective objective: L_sim + reg_weight * L_diffusion.


# import torch

device = 'cuda'
amp_dtype = 'bfloat16' if device == 'cpu' else 'float16'
use_amp = True
image_size = [192, 160, 192]

# wandb
project = 'Abdomen'
group = 'tm-dual-half-paired'
name = 'tm_dual-pair-flip+1mind+1diffusion'

# output directory
out_path = './tm_dual_outputs/exp0'
model_dir = 'saved_models'
#
load_model = None

# data
cache_rate = 1.0
num_workers = -1
batch_size = 1
pairwise = True
# The MR/CT half used for training is chosen at run time by train.py --split
# {ts,tr}; evaluate.py must be given the other half. The index range, folder
# names and organ label list of each half live in utils.data.abdomenmrct_data_utils.SPLITS.
pair_cfg = dict(
    data_dir="$DATA_ROOT/AbdomenMRCT",
)

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=2.0,
                        flow_scale=2.0,
                        interp_mode='bilinear')

sim_weight = 1.0
reg_weight = 1.0
gradicon_weight = 0.0
dice_weight = 0.0

lr = 1e-4
lr_decay = 0.996
start_epoch = 0
max_epochs = 200
save_interval = 50

# common unsupervised losses
# similarity loss
# sim_loss_cfg = dict(type='ncc_squared',
#                     spatial_dims=3,
#                     kernel_size=3,
#                     smooth_nr=0.0,
#                     smooth_dr=1e-5,
#                     weight=sim_weight)
sim_loss_cfg = dict(type='mind',
                    radius=2,
                    dilation=2,
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
                         weight=gradicon_weight)
# dice loss
dice_loss_cfg = dict(type='dice_loss',
                     include_background=False,
                     reduction='mean',
                     weight=dice_weight)

model_cfg = dict(
    type='TransMorph3D_Dual',
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
    )
)
