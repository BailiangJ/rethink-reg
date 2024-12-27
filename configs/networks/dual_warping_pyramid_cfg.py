image_size = [160, 192, 224]
bidirectional = False
model_cfg = dict(
    type='VXM_DualWarpPy',
    config=dict(
        bidirectional=bidirectional,
        encoder_cfg=dict(
            spatial_dims=3,
            in_chan=1,
            down=True,
            out_channels=[8, 16, 32, 48, 64],
            out_indices=[1, 2, 3, 4],
            block_config=dict(
                kernel_size=3,
                res_skip=False,
                down_first=True,
                conv_down=True,
                bias=True,
                pool_name=None,  # if conv_down==False: ('max', {'kernel_size': 2}),
                norm_name=('INSTANCE', {'affine': False}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                dropout=None,
            ),
        ),
        decoder_cfg=dict(
            image_size=image_size,
            spatial_dims=3,
            skip_channels=[128, 96, 64, 32],
            out_channels=[128, 96, 64, 32],
            out_indices=[4, 3, 2, 1],
            composition='compose',  # 'add'
            block_config=dict(
                kernel_size=3,
                res_skip=False,
                up_transp_conv=True,
                transp_bias=False,
                upsample_kernel_size=2,
                bias=True,
                norm_name=('INSTANCE', {'affine': False}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                dropout=None,
            ),
        ), )
)
