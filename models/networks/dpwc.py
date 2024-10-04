from __future__ import annotations
from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from monai.networks.layers.utils import get_act_layer, get_norm_layer, get_pool_layer
from monai.networks.blocks.dynunet_block import get_conv_layer
from models import FLOW_ESTIMATORS, ResizeFlow, Warp
from .transmorph import FlowConv
from .voxelmorph import CNNEncoder, CNNDecoder, UpBlock
from .correlation import WinCorrTorch3D, GlobalCorrTorch3D


class WarpPyramidalDecoder(CNNDecoder):
    def __init__(self,
                 image_size: Sequence[int],
                 spatial_dims: int,
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 composition: str = 'compose',  # 'compose' or 'sum'
                 *args, **kwargs,
                 ):
        super().__init__(spatial_dims, skip_channels, out_channels, block_config)
        self.out_indices = out_indices
        self.composition = composition
        self.flow_convs = nn.ModuleList()
        for i in range(self.num_levels):
            self.flow_convs.append(
                FlowConv(
                    in_channels=out_channels[i],
                    out_channels=3,
                )
            )
        self.resize_flow = ResizeFlow(
            spatial_scale=2,
            flow_scale=2,
            ndim=spatial_dims,
        )
        self.warp = nn.ModuleList()
        self.warp.append(None)
        for i in range(1, self.num_levels):
            self.warp.append(
                Warp(
                    image_size=np.array(image_size) // (2 ** self.out_indices[i]),
                    interp_mode='bilinear')
            )

    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_dec = None
        prev_flow = None

        for i in range(self.num_levels):
            if prev_flow == None:
                feats = torch.cat([src_feats[i], tgt_feats[i]], dim=1)
                prev_dec = self.decoder[i](prev_dec, feats)
                prev_flow = self.flow_convs[i](prev_dec)
            else:
                prev_flow = self.resize_flow(prev_flow)
                warp_src_feat = self.warp[i](src_feats[i], prev_flow)
                feats = torch.cat([warp_src_feat, tgt_feats[i]], dim=1)
                prev_dec = self.decoder[i](prev_dec, feats)
                flow = self.flow_convs[i](prev_dec)
                if self.composition == 'add':
                    prev_flow = prev_flow + flow
                elif self.composition == 'compose':
                    prev_flow = self.warp[i](prev_flow, flow.detach()) + flow
                else:
                    raise KeyError(f'Unsupported composition method: {self.composition}.')
            if prev_flow is not None:
                out_flows.append(prev_flow)

        return out_flows


class WarpCorrPyramidalDecoder(WarpPyramidalDecoder):
    def __init__(self,
                 image_size: Sequence[int],
                 spatial_dims: int,
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 corr_radius: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 composition: str = 'compose',  # 'compose' or 'sum'
                 *args, **kwargs,
                 ):
        super().__init__(image_size, spatial_dims,
                         skip_channels, out_channels,
                         out_indices, block_config, composition)
        self.corr_radius = corr_radius
        self.win_vol = []
        self.corr = nn.ModuleList()
        for k, r in zip(self.out_indices, self.corr_radius):
            self.corr.append(
                GlobalCorrTorch3D() if r == 0 else WinCorrTorch3D(radius=r)
            )
            self.win_vol.append(
                np.prod(np.array(image_size) // (2 ** k)) if r == 0 else (2 * r + 1) ** spatial_dims
            )
        self.decoder = nn.ModuleList()
        for i in range(self.num_levels):
            self.decoder.append(
                UpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=0 if i == 0 else out_channels[i - 1],
                    skip_channels=skip_channels[i] + self.win_vol[i],
                    out_channels=out_channels[i],
                    **block_config
                )
            )

    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_dec = None
        prev_flow = None

        for i in range(self.num_levels):
            if i == 0:
                corr = self.corr[i](src_feats[i], tgt_feats[i])
                skip = torch.cat([corr, tgt_feats[i]], dim=1)
                prev_dec = self.decoder[i](prev_dec, skip)
                prev_flow = self.flow_convs[i](prev_dec)
            else:
                prev_flow = self.resize_flow(prev_flow)
                warp_src_feat = self.warp[i](src_feats[i], prev_flow)
                corr = self.corr[i](warp_src_feat, tgt_feats[i])
                skip = torch.cat([corr, tgt_feats[i]], dim=1)
                prev_dec = self.decoder[i](prev_dec, skip)
                flow = self.flow_convs[i](prev_dec)
                if self.composition == 'add':
                    prev_flow = prev_flow + flow
                elif self.composition == 'compose':
                    prev_flow = self.warp[i](prev_flow, flow.detach()) + flow
                else:
                    raise KeyError(f'Unsupported composition method: {self.composition}.')
            if prev_flow is not None:
                out_flows.append(prev_flow)

        return out_flows


@FLOW_ESTIMATORS.register_module()
class VXM_DualWarpPy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.encoder = CNNEncoder(**config.encoder_cfg)
        self.decoder = WarpPyramidalDecoder(**config.decoder_cfg)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        src_feats = src_feats[::-1]
        tgt_feats = tgt_feats[::-1]
        flows = self.decoder(src_feats, tgt_feats)
        if self.bidirectional:
            bck_flows = self.decoder(tgt_feats, src_feats)
        return (flows, bck_flows) if self.bidirectional else (flows, None)


@FLOW_ESTIMATORS.register_module()
class VXM_DWPC(VXM_DualWarpPy):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = WarpCorrPyramidalDecoder(**config.decoder_cfg)
