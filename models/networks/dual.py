from __future__ import annotations
from .voxelmorph import VoxelMorph

@FLOW_ESTIMATORS.register_module()
class VoxelMorph_Dual(VoxelMorph):
    def __init__(self, encoder_cfg, decoder_cfg, remain_cfg):
        super().__init__(encoder_cfg, decoder_cfg, remain_cfg)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        feats = [torch.cat([src_feat, tgt_feat], dim=1) for src_feat, tgt_feat in zip(src_feats, tgt_feats)]
        feats = feats[::-1]
        dec = self.decoder(feats)
        dec = self.remain(dec)[-1]
        flow = self.flow_conv(dec)
        return flow