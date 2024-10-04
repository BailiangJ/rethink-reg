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
from .dpwc import WarpPyramidalDecoder, WarpCorrPyramidalDecoder, VXM_DualWarpPy, VXM_DWPC


class IterWarpPyramidalDecoder(WarpPyramidalDecoder):
    def __init__(self,
                 image_size: Sequence[int],
                 spatial_dims: int,
                 num_iters: Sequence[int],
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 composition: str = 'compose',  # 'compose' or 'sum'
                 ):
        super().__init__(image_size, spatial_dims,
                         skip_channels, out_channels,
                         out_indices, block_config, composition)
        self.num_iters = num_iters
        self.warp = nn.ModuleList()
        for i in range(0, self.num_levels):
            self.warp.append(
                Warp(
                    image_size=np.array(image_size) // (2 ** self.out_indices[i]),
                    interp_mode='bilinear')
            )

    def forward_flow(self, warp: nn.Module, decoder: nn.Module, flow_conv: nn.Module,
                     src_feat: torch.Tensor, tgt_feat: torch.Tensor,
                     prev_dec: Optional[torch.Tensor], prev_flow: Optional[torch.Tensor],
                     composition: str = 'compose'):
        '''
        Args:
            warp: Resolution of current level.
            prev_flow: Resolution of current level.
        '''
        if prev_flow is not None:
            src_feat = warp(src_feat, prev_flow)
        feats = torch.cat([src_feat, tgt_feat], dim=1)
        dec = decoder(prev_dec, feats)
        flow = flow_conv(dec)
        if prev_flow is not None:
            if composition == 'add':
                flow = prev_flow + flow
            elif composition == 'compose':
                flow = warp(prev_flow, flow.detach()) + flow
            else:
                raise KeyError(f'Unsupported composition method: {composition}.')
        return flow, dec

    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_dec = None
        prev_flow = None

        for i in range(self.num_levels):
            if prev_flow is not None:
                prev_flow = self.resize_flow(prev_flow)
            for _ in range(self.num_iters[i]):
                prev_flow, dec = self.forward_flow(self.warp[i], self.decoder[i], self.flow_convs[i],
                                                   src_feats[i], tgt_feats[i], prev_dec, prev_flow,
                                                   self.composition)
            prev_dec = dec
            if prev_flow is not None:
                out_flows.append(prev_flow)
        return out_flows


class IterWarpCorrPyramidalDecoder(WarpCorrPyramidalDecoder):
    def __init__(self,
                 image_size: Sequence[int],
                 spatial_dims: int,
                 num_iters: Sequence[int],
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 corr_radius: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 composition: str = 'compose',  # 'compose' or 'sum'
                 ):
        super().__init__(image_size, spatial_dims,
                         skip_channels, out_channels, corr_radius,
                         out_indices, block_config, composition)
        self.num_iters = num_iters
        self.warp = nn.ModuleList()
        for i in range(0, self.num_levels):
            self.warp.append(
                Warp(
                    image_size=np.array(image_size) // (2 ** self.out_indices[i]),
                    interp_mode='bilinear')
            )

    def forward_flow(self, warp: nn.Module, compute_corr: nn.Module,
                     decoder: nn.Module, flow_conv: nn.Module,
                     src_feat: torch.Tensor, tgt_feat: torch.Tensor,
                     prev_dec: Optional[torch.Tensor], prev_flow: Optional[torch.Tensor],
                     composition: str = 'compose'):
        '''
        Args:
            warp: Resolution of current level.
            prev_flow: Resolution of current level.
        '''
        if prev_flow is not None:
            src_feat = warp(src_feat, prev_flow)
        corr = compute_corr(src_feat, tgt_feat)
        feats = torch.cat([corr, tgt_feat], dim=1)
        dec = decoder(prev_dec, feats)
        flow = flow_conv(dec)
        if prev_flow is not None:
            if composition == 'add':
                flow = prev_flow + flow
            elif composition == 'compose':
                flow = warp(prev_flow, flow.detach()) + flow
            else:
                raise KeyError(f'Unsupported composition method: {composition}.')
        return flow, dec

    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_dec = None
        prev_flow = None

        for i in range(self.num_levels):
            if prev_flow is not None:
                prev_flow = self.resize_flow(prev_flow)
            for _ in range(self.num_iters[i]):
                prev_flow, dec = self.forward_flow(self.warp[i], self.corr[i],
                                                   self.decoder[i], self.flow_convs[i],
                                                   src_feats[i], tgt_feats[i],
                                                   prev_dec, prev_flow,
                                                   self.composition)
            prev_dec = dec
            if prev_flow is not None:
                out_flows.append(prev_flow)
        return out_flows


@FLOW_ESTIMATORS.register_module()
class VXM_DualWarpPy_Iter(VXM_DualWarpPy):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpPyramidalDecoder(**config.decoder_cfg)


@FLOW_ESTIMATORS.register_module()
class VXM_DWPCI(VXM_DWPC):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpCorrPyramidalDecoder(**config.decoder_cfg)
