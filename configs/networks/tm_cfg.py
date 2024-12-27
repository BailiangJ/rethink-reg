# Config file for TransMorph
model_cfg = dict(
    type='TransMorph',
    config=dict(
        if_transskip=True,
        if_convskip=True,
        patch_size=4,
        in_chans=2,
        embed_dim=96,
        depths=(2, 2, 4, 2),
        num_heads=(4, 4, 8, 8),
        window_size=(5, 6, 7),
        mlp_ratio=4,
        pat_merg_rf=4,
        qkv_bias=False,
        drop_rate=0,
        drop_path_rate=0.3,
        ape=False,
        spe=False,
        rpe=True,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(0, 1, 2, 3),
        reg_head_chan=16,
    )
)