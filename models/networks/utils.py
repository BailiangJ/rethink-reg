import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mamba_ssm import Mamba
from typing import Optional


class ConvLReLU(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        negative_slope: float = 0.01,
        use_batchnorm: bool = True,
        bias: bool = False,
    ):
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        lrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        if not use_batchnorm:
            Norm = getattr(nn, "InstanceNorm%dd" % spatial_dims)
            norm = Norm(out_channels)
        else:
            Norm = getattr(nn, "BatchNorm%dd" % spatial_dims)
            norm = Norm(out_channels)

        super(ConvLReLU, self).__init__(conv, norm, lrelu)


class FlowConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        kernel_size: int = 3,
    ):
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        conv = Conv(
            in_channels,
            spatial_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        nn.init.normal_(conv.weight, mean=0.0, std=1e-5)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        super().__init__(conv)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TMDecoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.conv1 = ConvLReLU(
            spatial_dims,
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            bias=False,
        )
        self.conv2 = ConvLReLU(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            bias=False,
        )
        if spatial_dims == 2:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        elif spatial_dims == 3:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        else:
            raise KeyError(f"Unsupported spatial_dims for nn.Upsample:{spatial_dims}.")

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MambaMlpBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        d_state=16,
        d_conv=4,
        expand=2,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba_type="v2",
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B, n_tokens, C = x.shape
        assert C == self.dim
        x_mamba = self.mamba(self.norm1(x))
        x_mamba = x + self.drop_path(x_mamba)
        x_out = self.mlp(self.norm2(x_mamba))
        x_out = x_mamba + self.drop_path(x_out)
        return x_out


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            broadcast_shape = [1,-1] + [1] * (x.dim() -2)
            x = self.weight.view(*broadcast_shape) * x + self.bias.view(*broadcast_shape)

            return x


class MambaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        res_skip: bool = False,
    ):
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
        image_size = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = x_flat + x_mamba if self.res_skip else x_mamba
        out = x_out.transpose(-1, -2).reshape(B, C, *image_size)
        return out


class MlpChannel(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        self.fc1 = Conv(in_features, hidden_features, kernel_size=1, bias=bias)
        self.act = act_layer()
        self.fc2 = Conv(hidden_features, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
