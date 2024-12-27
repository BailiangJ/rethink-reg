# Config file for Mamba-TransMorph (Replace Swin-Transformer with Mamba block)
model_cfg = dict(
    type='Mamba_TM',
    config=dict(
        if_transskip=True,
        if_convskip=True,
        patch_size=4,
        in_chans=2,
        embed_dim=96,
        depths=(2, 2, 4, 2),
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4,
        pat_merg_rf=4,
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