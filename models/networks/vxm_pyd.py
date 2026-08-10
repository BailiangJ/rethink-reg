from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..builder import FLOW_ESTIMATORS
from ..utils import ResizeFlow
from .vxm import CNNEncoder, CNNDecoder
from .utils import FlowConv


class CNNPyramidalDecoder(CNNDecoder):
    def __init__(self,
                 spatial_dims: int,
                 skip_channels: Sequence[int],
                 out_channels: int,
                 block_config: dict,
                 ):
        super().__init__(spatial_dims, skip_channels, out_channels, block_config)
        self.flow_convs = nn.ModuleList()
        for i in range(self.num_levels):
            self.flow_convs.append(
                FlowConv(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels[i] if i == 0 else out_channels[i] + 3,
                )
            )
        self.resize_flow = ResizeFlow(
            spatial_scale=2,
            flow_scale=2,
            ndim=spatial_dims,
        )

    def forward(self, skips: Sequence[torch.Tensor]):
        out_flows = []
        prev_dec = None
        prev_flow = None
        for i, (layer, flow_conv) in enumerate(zip(self.decoder, self.flow_convs)):
            prev_dec = layer(prev_dec, skips[i])
            if prev_flow is None:
                prev_flow = flow_conv(prev_dec)
            else:
                prev_flow = self.resize_flow(prev_flow)
                prev_flow = flow_conv(torch.cat([prev_dec, prev_flow], dim=1))
            out_flows.append(prev_flow)
        return out_flows


@FLOW_ESTIMATORS.register_module()
class VXMPlus_Py(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.encoder = CNNEncoder(**config.encoder_cfg)
        self.decoder = CNNPyramidalDecoder(**config.decoder_cfg)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(torch.cat([src, tgt], dim=1))
        feats = feats[::-1]
        flows = self.decoder(feats)
        if self.bidirectional:
            feats = self.encoder(torch.cat([tgt, src], dim=1))
            feats = feats[::-1]
            bck_flows = self.decoder(feats)
        return (flows, bck_flows) if self.bidirectional else (flows, None)


@FLOW_ESTIMATORS.register_module()
class VXMPlus_DualPy(VXMPlus_Py):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        feats = [torch.cat([src_feat, tgt_feat], dim=1) for src_feat, tgt_feat in zip(src_feats, tgt_feats)]
        feats = feats[::-1]
        flows = self.decoder(feats)
        if self.bidirectional:
            feats = [torch.cat([tgt_feat, src_feat], dim=1) for src_feat, tgt_feat in zip(src_feats, tgt_feats)]
            feats = feats[::-1]
            bck_flows = self.decoder(feats)
        return (flows, bck_flows) if self.bidirectional else (flows, None)