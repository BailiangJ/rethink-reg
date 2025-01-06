device = 'cuda'
amp_dtype = 'float16'
image_size = [160, 192, 224]

sim_weight = 1.0
reg_weight = 0.5
dice_weight = 1.0

lr = 1e-4
lr_decay = 0.996
start_epoch = 0
max_epochs = 100
save_interval = 20

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=2.0,
                        flow_scale=2.0,
                        interp_mode='bilinear')

sim_loss_cfg = dict(type='ncc',
                    spatial_dims=3,
                    kernel_size=9,
                    smooth_nr=0.0,
                    smooth_dr=1e-5,
                    weight=sim_weight)

reg_loss_cfg = dict(type='diffusion',
                    penalty='l2',
                    loss_mult=2.0,
                    weight=reg_weight)

dice_loss_cfg = dict(type='dice_loss',
                     include_background=False,
                     reduction='mean',
                     weight=dice_weight)
