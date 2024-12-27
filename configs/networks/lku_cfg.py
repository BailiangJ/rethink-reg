# Config file for Large-Kernel U-Net
model_cfg = dict(
    type='LKUNet',
    in_channel=2,
    out_channel=3,
    enc_feat_channel=16,
    dec_feat_channel=16,
    large_kernel_size=5,
    bias=True
)
