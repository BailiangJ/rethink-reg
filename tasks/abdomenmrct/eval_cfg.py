device = 'cuda'
image_size = [192, 160, 192]

data_dir = "$DATA_ROOT/AbdomenMRCT/"
cache_rate = 0.2
num_workers = -1
batch_size = 1

# Registration-head fallback for standalone config use. The merged evaluator
# replaces this with the saved training registration_cfg for released checkpoints.
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=1.0,
                        flow_scale=1.0,
                        interp_mode='bilinear')
