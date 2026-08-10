device = 'cuda'
image_size = [224, 192, 224]

# data
data_dir = "$DATA_ROOT/Lung250M-4B/"
cache_rate = 1.0
num_workers = -1
batch_size = 1
prefix = 'Lung250M'

# the 17 held-out inspiration/expiration pairs, pooled from four sources. The
# TCIA-NLST block is left out: those cases overlap with the NLST training set.
data_indexs = [2, 8,                                       # EMPIRE10
               54, 55, 56,                                 # L2R-LungCT
               94, 97,                                     # TCIA-Ventilation
               104, 105, 106, 107, 108, 109, 110, 111, 112, 113,  # COPDgene
               # 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,  # TCIA-NLST
               ]

# tre on keypoints
tre_cfg = dict(type='tre',
               image_size=image_size,
               spacing=(1.0, 1.0, 1.0),
               interp_mode='bilinear')
