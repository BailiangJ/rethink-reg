cache_rate = 1.0
num_workers = -1
image_size = [160, 192, 224]
batch_size = 2
dataset_slice = (None, None, 1)

# multiple datasets
oasis_cfg = dict(
    data_dir='/DATA/OASIS/train/',
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)

adni_cfg = dict(
    df_path="/ADNI_SEG/splits/train_small.csv",
    data_dir="/ADNI_SEG/registered_output_mni152/FS72/",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)

ixi_cfg = dict(
    df_path="/IXI_data/train.json",
    data_dir="/IXI_data/Train",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)

lumir_cfg = dict(
    data_dir="/LUMIR_L2R24_TrainVal/imagesTr",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
    batch_size=batch_size,
)
# lpba
lpba_cfg = dict(
    data_dir="/LPBA/data/delineation_space",
    image_size=image_size,
    dataset_slice=dataset_slice,
    cache_rate=cache_rate,
    num_workers=num_workers,
)
# mindboggle
mindboggle_cfg = dict(
    data_dir="/data/Mindboggle101",
    image_size=image_size,
    dataset_slice=(None,80,1),
    cache_rate=cache_rate,
    num_workers=num_workers,
)
