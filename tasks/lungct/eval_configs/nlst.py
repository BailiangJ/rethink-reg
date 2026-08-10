device = 'cuda'
image_size = [224, 192, 224]

# data
data_dir = "$DATA_ROOT/NLST"
cache_rate = 1.0
num_workers = -1
batch_size = 1

# NLST_0001-0100 are the training cases; the held-out cases are split into a
# validation and a test half. Swap the two blocks to score the validation half.
# prefix = 'val'
# data_indexs = list(range(200, 250))
prefix = 'test'
data_indexs = list(range(250, 300))

# tre on keypoints
tre_cfg = dict(type='tre',
               image_size=image_size,
               spacing=(1.5, 1.5, 1.5),
               interp_mode='bilinear')
