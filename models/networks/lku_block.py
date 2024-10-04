from __future__ import annotations
from typing import Sequence, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from models import FLOW_ESTIMATORS


class LKBlock(nn.Module):
    '''
    Large Kernel Convolutional Block

    Conv(k=1)+Norm + Conv(k=3)+Norm + Conv(k=larget_kernel_size)+Norm + Identity -> Act
    '''

    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 large_kernel_size: int,
                 stride: int = 1,
                 bias: bool = False,
                 norm_name=('INSTANCE', {'affine': True}),
                 act_name=('PRELU', {'init': 0.2}),
                 dropout: tuple | str | float | None = None,
                 ):
        super().__init__()
        self.conv_k1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            norm_name=norm_name,
            act_name=None,
            dropout=dropout,
        )
        self.conv_k3 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            bias=bias,
            norm_name=norm_name,
            act_name=None,
            dropout=dropout,
        )
        self.conv_lk = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=large_kernel_size,
            stride=stride,
            bias=bias,
            norm_name=norm_name,
            act_name=None,
            dropout=dropout,
        )
        self.act = get_act_layer(act_name)

    def forward(self, x: torch.Tesnor) -> torch.Tensor:
        x_k1 = self.conv_k1(x)
        x_k3 = self.conv_k3(x_k1)
        x_lk = self.conv_lk(x)
        x = self.act(x_k1 + x_k3 + x_lk + x)
        return x


class LKCNNEncoder(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 in_chan: int,
                 out_channels: Sequence[int],
                 out_indices: Sequence[int],
                 large_kernel_size: int = 5,
                 norm_name=('INSTANCE', {'affine': True}),
                 act_name=('PRELU', {'init': 0.2}),
                 bias: bool = True,
                 ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=in_chan,
                    out_channels=out_channels[0],
                    kernel_size=3,
                    stride=1,
                    dropout=None,
                    act=act_name,
                    norm=norm_name,
                    bias=bias,
                ),
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=3,
                    stride=1,
                    dropout=None,
                    act=act_name,
                    norm=norm_name,
                    bias=bias,
                )
            )
        )
        for i in range(len(out_channels) - 1):
            self.encoder.append(
                nn.Sequential(
                    get_conv_layer(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels[i],
                        out_channels=out_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        dropout=None,
                        act=act_name,
                        norm=norm_name,
                        bias=bias,
                    ),
                    LKBlock(
                        spatial_dims,
                        out_channels[i + 1],
                        out_channels[i + 1],
                        large_kernel_size,
                        stride=1,
                        bias=bias,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=None,
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        outs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
