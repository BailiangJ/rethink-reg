from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.utils import get_act_layer, get_norm_layer, get_pool_layer
from monai.networks.blocks.dynunet_block import get_conv_layer
from models import FLOW_ESTIMATORS
from .transmorph import FlowConv
from .voxelmorph import CNNDecoder, CNNEncoder
from functools import partial
from .mamba_blocks import MambaLayer, MlpChannel, LayerNorm

class MambaEncoder(nn.Module):
    '''
    class MambaLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 residual: bool = False):
    '''

    def __init__(self,
                 spatial_dims: int,
                 in_chan: int,
                 depths: Sequence[int],
                 out_channels: Sequence[int],
                 out_indices: Sequence[int],
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 mlp_ratio=2,
                 res_skip: bool = True,
                 conv_down: bool = True,
                 bias: bool = True,
                 pool_name: tuple | str | None = ('max', {'kernel_size': 2}),
                 norm_name: tuple | str = ('INSTANCE', {'affine': False}),
                 ):
        super().__init__()
        self.num_levels = len(out_channels)
        self.downsample_layers = nn.ModuleList()
        stem = get_conv_layer(
            spatial_dims,
            in_chan,
            out_channels[0],
            kernel_size=7,
            stride=2,
            bias=bias,
            conv_only=True,
        )  # H/2, W/w, D/2
        self.downsample_layers.append(stem)
        for i in range(self.num_levels - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    # get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels[i]),
                    LayerNorm(out_channels[i], eps=1e-6, data_format="channels_first"),
                    get_conv_layer(
                        spatial_dims,
                        out_channels[i],
                        out_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                    )
                )
            )
        self.stages = nn.ModuleList()
        for i in range(self.num_levels):
            self.stages.append(
                nn.Sequential(
                    *[MambaLayer(out_channels[i],
                                 d_state, d_conv, expand,
                                 res_skip) for _ in range(depths[i])]
                )
            )

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        layer_norm = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i in range(self.num_levels):
            layer = layer_norm(out_channels[i])
            # layer = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(out_channels[i], mlp_ratio * out_channels[i]))

    def forward(self, x):
        outs = []
        for i in range(self.num_levels):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return outs


@FLOW_ESTIMATORS.register_module()
class Mamba_VXM(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, remain_cfg):
        super().__init__()
        self.encoder = MambaEncoder(**encoder_cfg)
        self.decoder = CNNDecoder(**decoder_cfg)
        self.remain = CNNEncoder(**remain_cfg)
        self.flow_conv = FlowConv(in_channels=remain_cfg.out_channels[-1], out_channels=3)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(torch.cat([src, tgt], dim=1))
        feats = feats[::-1]
        dec = self.decoder(feats)
        dec = self.remain(dec)[-1]
        flow = self.flow_conv(dec)
        return flow
