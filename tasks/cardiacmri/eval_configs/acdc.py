device = 'cuda'
image_size = [128, 128]
spacing = (1.5, 1.5, 5.0)

# data
# ACDC ships 100 training and 50 testing patients. Models are trained on
# patients 1-80; swap the two blocks below to score the 81-100 validation half
# instead of the official test set.
# prefix = 'val'
# df_path = "training.csv"
# data_dir = "$DATA_ROOT/ACDC/training"
# dataset_slice = (80, None, None)
prefix = 'test'
df_path = "testing.csv"
data_dir = "$DATA_ROOT/ACDC/testing"
dataset_slice = (None, None, None)

cache_rate = 1.0
num_workers = -1
batch_size = 1
