from __future__ import annotations

import torch

from models import FLOW_ESTIMATORS
from .voxelmorph import VoxelMorph


@FLOW_ESTIMATORS.register_module()
@FLOW_ESTIMATORS.register_module(name='VXM_Dual')
class VoxelMorph_Dual(VoxelMorph):
    def __init__(self, encoder_cfg, decoder_cfg, remain_cfg, bidirectional: bool = False):
        super().__init__(encoder_cfg, decoder_cfg, remain_cfg)
        self.bidirectional = bidirectional

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        feats = [torch.cat([src_feat, tgt_feat], dim=1) for src_feat, tgt_feat in zip(src_feats, tgt_feats)]
        feats = feats[::-1]
        dec = self.decoder(feats)
        dec = self.remain(dec)[-1]
        flow = self.flow_conv(dec)
        if self.bidirectional:
            bck_feats = [torch.cat([tgt_feat, src_feat], dim=1) for src_feat, tgt_feat in zip(src_feats, tgt_feats)]
            bck_feats = bck_feats[::-1]
            bck_dec = self.decoder(bck_feats)
            bck_dec = self.remain(bck_dec)[-1]
            bck_flow = self.flow_conv(bck_dec)
        return (flow, bck_flow) if self.bidirectional else (flow, None)
