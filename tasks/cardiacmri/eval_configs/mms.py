device = 'cuda'
# M&Ms is the cross-dataset generalisation set: models trained on ACDC are
# applied to it unchanged. Slices are cropped around the segmentation and then
# padded/cropped to image_size, so 256 keeps M&Ms' native field of view while
# 128 matches the ACDC training crop. Both are reported in the appendix; use
# `--image-size 128` to score the latter without editing this file.
image_size = [256, 256]
spacing = (1.5, 1.5, 5.0)

# data
prefix = 'mms'
df_path = "$DATA_ROOT/MM-Cardiac/MMs_Dataset_info.csv"
data_dir = "$DATA_ROOT/MM-Cardiac/All"
# every M&Ms patient (the stride keeps the ordering explicit)
dataset_slice = (None, None, 1)

cache_rate = 1.0
num_workers = -1
batch_size = 1
