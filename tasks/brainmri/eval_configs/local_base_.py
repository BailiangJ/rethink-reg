import os

device = 'cuda'

image_size = [160, 192, 224]
dataset_slice = (None, None, 1)
cache_rate = 0.05
num_workers = -1
num_pairs = 200

# oasis
oasis_cfg = dict(
    data_dir="$DATA_ROOT/OASIS/test/",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
)
# adni
adni_cfg = dict(
    df_path="$DATA_ROOT/ADNI_SEG/splits/test_small.csv",
    data_dir="$DATA_ROOT/ADNI_SEG/registered_output_mni152/FS72",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
)
# ixi
ixi_cfg = dict(df_path="$DATA_ROOT/IXI_data/test.json",
               data_dir="$DATA_ROOT/IXI_data/Test",
               image_size=image_size,
               dataset_slice=dataset_slice,
               cache_rate=cache_rate,
               num_workers=num_workers, )
# lpba
lpba_cfg = dict(
    data_dir="$DATA_ROOT/LPBA/data/delineation_space/",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
)
# mindboggle
mindboggle_cfg = dict(
    data_dir="$DATA_ROOT/Mindboggle101",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
)

# registration head
registration_cfg = dict(
    type='RegistrationHead',
    image_size=image_size,
    spatial_scale=2.0,
    flow_scale=2.0,
    interp_mode='bilinear')
