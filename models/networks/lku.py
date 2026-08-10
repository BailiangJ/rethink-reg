from __future__ import annotations
from typing import Sequence, Optional, Dict
from ..builder import FLOW_ESTIMATORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from .utils import FlowConv


class LK_encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        bias: bool = False,
        batchnorm: bool = False,
    ):
        super(LK_encoder, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.batchnorm = batchnorm

        self.layer_regularKernel = self.encoder_LK_encoder(
            self.spatial_dims,
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        self.layer_largeKernel = self.encoder_LK_encoder(
            self.spatial_dims,
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        self.layer_oneKernel = self.encoder_LK_encoder(
            self.spatial_dims,
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        self.layer_nonlinearity = nn.PReLU(init=0.2)
        # Norm = getattr(nn, "BatchNorm%dd" % spatial_dims)
        # self.layer_batchnorm = Norm(num_features = self.out_channels)

    def encoder_LK_encoder(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        batchnorm: bool = False,
    ):
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        Norm = getattr(nn, "BatchNorm%dd" % spatial_dims)
        if batchnorm:
            layer = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                Norm(out_channels),
            )
        else:
            layer = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        return layer

    def forward(self, inputs):
        # print(self.layer_regularKernel)
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        # if self.layer_indentity:
        outputs = regularKernel + largeKernel + oneKernel + inputs
        # else:
        # outputs = regularKernel + largeKernel + oneKernel
        # if self.batchnorm:
        # outputs = self.layer_batchnorm(self.layer_batchnorm)
        return self.layer_nonlinearity(outputs)


@FLOW_ESTIMATORS.register_module()
class LKUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channel: int = 2,
        out_channel: int = 3,
        enc_feat_channel: int = 8,
        dec_feat_channel: int = 8,
        large_kernel_size: int = 5,
        bias: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.enc_feat_channel = enc_feat_channel
        self.dec_feat_channel = dec_feat_channel

        self.enc0 = nn.Sequential(
            self.encoder(
                self.spatial_dims, self.in_channel, self.enc_feat_channel, bias=bias
            ),
            self.encoder(
                self.spatial_dims,
                self.enc_feat_channel,
                self.enc_feat_channel,
                bias=bias,
            ),
        )

        self.enc1 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.enc_feat_channel,
                self.enc_feat_channel * 2,
                stride=2,
                bias=bias,
            ),
            LK_encoder(
                self.spatial_dims,
                self.enc_feat_channel * 2,
                self.enc_feat_channel * 2,
                kernel_size=large_kernel_size,
                stride=1,
                padding=2,
                bias=bias,
            ),
        )

        self.enc2 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.enc_feat_channel * 2,
                self.enc_feat_channel * 4,
                stride=2,
                bias=bias,
            ),
            LK_encoder(
                self.spatial_dims,
                self.enc_feat_channel * 4,
                self.enc_feat_channel * 4,
                kernel_size=large_kernel_size,
                stride=1,
                padding=2,
                bias=bias,
            ),
        )

        self.enc3 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.enc_feat_channel * 4,
                self.enc_feat_channel * 8,
                stride=2,
                bias=bias,
            ),
            LK_encoder(
                self.spatial_dims,
                self.enc_feat_channel * 8,
                self.enc_feat_channel * 8,
                kernel_size=large_kernel_size,
                stride=1,
                padding=2,
                bias=bias,
            ),
        )

        self.enc4 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.enc_feat_channel * 8,
                self.enc_feat_channel * 8,
                stride=2,
                bias=bias,
            ),
            LK_encoder(
                self.spatial_dims,
                self.enc_feat_channel * 8,
                self.enc_feat_channel * 8,
                kernel_size=large_kernel_size,
                stride=1,
                padding=2,
                bias=bias,
            ),
        )

        self.up4 = self.decoder(
            self.spatial_dims, self.dec_feat_channel * 8, self.dec_feat_channel * 8
        )
        self.dec4 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.dec_feat_channel * 8 + self.dec_feat_channel * 8,
                self.dec_feat_channel * 8,
                kernel_size=3,
                stride=1,
                bias=bias,
            ),
            self.encoder(
                self.spatial_dims,
                self.dec_feat_channel * 8,
                self.dec_feat_channel * 4,
                kernel_size=3,
                stride=1,
                bias=bias,
            ),
        )

        self.up3 = self.decoder(
            self.spatial_dims, self.dec_feat_channel * 4, self.dec_feat_channel * 4
        )
        self.dec3 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.dec_feat_channel * 4 + self.dec_feat_channel * 4,
                self.dec_feat_channel * 4,
                kernel_size=3,
                stride=1,
                bias=bias,
            ),
            self.encoder(
                self.spatial_dims,
                self.dec_feat_channel * 4,
                self.dec_feat_channel * 2,
                kernel_size=3,
                stride=1,
                bias=bias,
            ),
        )

        self.up2 = self.decoder(
            self.spatial_dims, self.dec_feat_channel * 2, self.dec_feat_channel * 2
        )
        self.dec2 = nn.Sequential(
            self.encoder(
                self.spatial_dims,
                self.dec_feat_channel * 2 + self.dec_feat_channel * 2,
                self.dec_feat_channel * 4,
                kernel_size=3,
                stride=1,
                bias=bias,
            ),
            self.encoder(
                self.spatial_dims,
                self.dec_feat_channel * 4,
                self.dec_feat_channel * 2,
                kernel_size=3,
                stride=1,
                bias=bias,
            ),
        )

        self.flow_conv = FlowConv(
            self.spatial_dims, self.dec_feat_channel * 2, kernel_size=3
        )

    def encoder(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        batchnorm: bool = False,
    ):
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        Norm = getattr(nn, "BatchNorm%dd" % spatial_dims)
        if batchnorm:
            layer = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                Norm(out_channels),
                nn.PReLU(init=0.2),
            )
        else:
            layer = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.PReLU(init=0.2),
            )
        return layer

    def decoder(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
    ):
        ConvTransposed = getattr(nn, "ConvTranspose%dd" % spatial_dims)
        layer = nn.Sequential(
            ConvTransposed(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.PReLU(init=0.2),
        )
        return layer

    def outputs(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        batchnorm: bool = False,
    ):
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        Norm = getattr(nn, "BatchNorm%dd" % spatial_dims)
        if batchnorm:
            layer = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                Norm(out_channels),
                nn.Tanh(),
            )
        else:
            layer = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.Softsign(),
            )
        return layer  # the output is in the range of [-1, 1]

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)

        e0 = self.enc0(x_in)  # B, feat_channel, H, W, D
        e1 = self.enc1(e0)  # B, feat_channel * 2, H/2, W/2, D/2
        e2 = self.enc2(e1)  # B, feat_channel * 4, H/4, W/4, D/4
        e3 = self.enc3(e2)  # B, feat_channel * 8, H/8, W/8, D/8
        e4 = self.enc4(e3)  # B, feat_channel * 8, H/16, W/16, D/16

        d4 = torch.cat((self.up4(e4), e3), 1)  # B, feat_channel * 16, H/8, W/8, D/8
        d4 = self.dec4(d4)  # B, feat_channel * 4, H/8, W/8, D/8

        d3 = torch.cat((self.up3(d4), e2), 1)  # B, feat_channel * 8, H/4, W/4, D/4
        d3 = self.dec3(d3)  # B, feat_channel * 2, H/4, W/4, D/4

        d2 = torch.cat((self.up2(d3), e1), 1)  # B, feat_channel * 4, H/2, W/2, D/2
        d2 = self.dec2(d2)  # B, feat_channel * 2, H/2, W/2, D/2

        out = self.flow_conv(d2)  # B, 3, H/2, W/2, D/2

        return out
