from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from functools import partial

from monai.networks.blocks.dynunet_block import get_conv_layer
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from mamba_ssm import Mamba
from .transmorph import Mlp


class MambaLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 res_skip: bool = False):
        super().__init__()
        self.dim = dim
        self.res_skip = res_skip
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba_type="v2",
        )

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = x_flat + x_mamba if self.res_skip else x_mamba
        out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class ResMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba_type="v2",
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(self.norm1(x_flat))
        x_mamba = x_flat + self.drop_path(x_mamba)
        x_out = self.mlp(self.norm2(x_mamba))
        x_out = x_mamba + self.drop_path(x_out)
        x_out = x_out.transpose(-1, -2).reshape(B, C, H, W, D)
        return x_out


class MlpChannel(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 bias=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        self.drop(x)
        x = self.fc2(x)
        self.drop(x)
        return x
