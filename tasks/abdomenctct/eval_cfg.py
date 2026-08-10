device = 'cuda'
image_size = [160, 128, 160]

data_dir = "$DATA_ROOT/PSMAReg/PSMAReg_CT_affine_crop160x128x160_FU01_no0359"
split_json = './dataset_split.json'
label_list = [5, 1, 2, 3, 6, 7, 52, 63]
cache_rate = 1.0
num_workers = -1
batch_size = 1

# Registration-head fallback for standalone config use. The merged evaluator
# replaces this with the saved training registration_cfg for released checkpoints.
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=1.0,
                        flow_scale=1.0,
                        interp_mode='bilinear')
