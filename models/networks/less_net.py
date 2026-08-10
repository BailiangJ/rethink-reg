import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import get_conv_layer
import numpy as np
from typing import Sequence
from .vxm import CNNEncoder, CNNDecoder, UpBlock
from .vxm_pwc import WarpPyramidalDecoder, WarpCorrPyramidalDecoder, WarpCorrCatPyramidalDecoder
from .correlation import WinCorrTorch, GlobalCorrTorch
from .vxm_iter import IterWarpPyramidalDecoder, IterWarpCorrPyramidalDecoder, IterWarpCorrCatPyramidalDecoder
from .utils import FlowConv, MlpChannel
from ..builder import FLOW_ESTIMATORS
from ..utils import ResizeFlow, Warp

class PoolingEncoder(nn.Module):
    def __init__(self, spatial_dims: int, out_indices: Sequence[int], **kwargs):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        self.spatial_dims = spatial_dims
        self.out_indices = list(out_indices)
        self.num_levels = len(self.out_indices)
        self.avg_pool_fn = getattr(F, f"avg_pool{spatial_dims}d")
        self.max_pool_fn = getattr(F, f"max_pool{spatial_dims}d")


    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        outs = []
        for level in self.out_indices:
            k = 2 ** level
            x_avg = self.avg_pool_fn(x, kernel_size=k, stride=k)
            x_max = self.max_pool_fn(x, kernel_size=k, stride=k)
            x_min = -self.max_pool_fn(-x, kernel_size=k, stride=k)
            # Channel-wise features: [avg, max, min]
            out = torch.cat([x_avg, x_max, x_min], dim=1)
            outs.append(out)
        return outs


def build_projection_layer(spatial_dims: int,
                           in_channels: int,
                           out_channels: int,
                           proj_type: str = 'conv') -> nn.Module:
    if proj_type == 'conv':
        return get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            bias=True,
            norm=None,
            act=None,
            dropout=None,
        )
    if proj_type == 'mlp':
        return MlpChannel(
            spatial_dims=spatial_dims,
            in_features=in_channels,
            hidden_features=out_channels // 2,
            out_features=out_channels,
            act_layer=nn.LeakyReLU,
            drop=0.0,
            bias=True,
        )
    raise ValueError(f"Unsupported proj_type: {proj_type}. Expected 'conv' or 'mlp'.")


class PoolingProjEncoder(PoolingEncoder):
    def __init__(self,
                spatial_dims: int,
                out_indices: Sequence[int],
                proj_channels: Sequence[int],
                in_channels: int = 1,
                proj_type: str = 'conv'):
        super().__init__(spatial_dims, out_indices)
        assert len(proj_channels) == self.num_levels, "proj_channels length must match num_levels"
        pool_channels = 3 * in_channels  # avg, max, min per input channel
        self.proj_convs = nn.ModuleList()
        for i in range(self.num_levels):
            self.proj_convs.append(
                build_projection_layer(
                    spatial_dims,
                    pool_channels,
                    proj_channels[i],
                    proj_type=proj_type,
                )
            )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        outs = []
        for level, proj_conv in zip(self.out_indices, self.proj_convs):
            k = 2 ** level
            x_avg = self.avg_pool_fn(x, kernel_size=k, stride=k)
            x_max = self.max_pool_fn(x, kernel_size=k, stride=k)
            x_min = -self.max_pool_fn(-x, kernel_size=k, stride=k)
            out = torch.cat([x_avg, x_max, x_min], dim=1)
            out = proj_conv(out)
            outs.append(out)
        return outs



@FLOW_ESTIMATORS.register_module()
class LessNet(nn.Module):
    def __init__(self,
                encoder_cfg,
                decoder_cfg,
                remain_cfg):
        super().__init__()
        self.encoder = PoolingEncoder(**encoder_cfg)
        self.decoder = CNNDecoder(**decoder_cfg)
        self.remain = CNNEncoder(**remain_cfg)
        self.flow_conv = FlowConv(
            spatial_dims=remain_cfg.spatial_dims,
            in_channels=remain_cfg.out_channels[-1],
            kernel_size=3,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Encode source and target
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)

        feats = []
        for src_feat, tgt_feat in zip(src_feats, tgt_feats):
            feat = torch.cat([src_feat, tgt_feat], dim=1)
            feats.append(feat)
        feats = feats[::-1]
        dec = self.decoder(feats)
        dec = self.remain(dec)[-1]
        flow = self.flow_conv(dec)
        return flow

@FLOW_ESTIMATORS.register_module()
class LessNet_DualWarpPy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.encoder = PoolingEncoder(**config.encoder_cfg)
        self.decoder = WarpPyramidalDecoder(**config.decoder_cfg)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Encode source and target
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)

        src_feats = src_feats[::-1]
        tgt_feats = tgt_feats[::-1]
        flows = self.decoder(src_feats, tgt_feats)
        if self.bidirectional:
            bck_flows = self.decoder(tgt_feats, src_feats)
        return (flows, bck_flows) if self.bidirectional else (flows, None)

@FLOW_ESTIMATORS.register_module()
class LessNet_DWP_Iter(LessNet_DualWarpPy):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpPyramidalDecoder(**config.decoder_cfg)

    def forward_eval(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        src_feats = src_feats[::-1]
        tgt_feats = tgt_feats[::-1]
        fwd = self.decoder.forward_eval(src_feats, tgt_feats)
        bck = self.decoder.forward_eval(tgt_feats, src_feats) if self.bidirectional else None
        return {'fwd': fwd, 'bck': bck}


@FLOW_ESTIMATORS.register_module()
class LessNet_PWC(LessNet_DualWarpPy):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = WarpCorrPyramidalDecoder(**config.decoder_cfg)

@FLOW_ESTIMATORS.register_module()
class LessNet_PWC_Iter(LessNet_PWC):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpCorrPyramidalDecoder(**config.decoder_cfg)

    def forward_eval(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)

        src_feats = src_feats[::-1]
        tgt_feats = tgt_feats[::-1]
        fwd = self.decoder.forward_eval(src_feats, tgt_feats)
        bck = self.decoder.forward_eval(tgt_feats, src_feats) if self.bidirectional else None
        return {'fwd': fwd, 'bck': bck}

@FLOW_ESTIMATORS.register_module()
class LessNet_PWC_v1(LessNet_PWC):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PoolingProjEncoder(**config.encoder_cfg)

@FLOW_ESTIMATORS.register_module()
class LessNet_PWC_Iter_v1(LessNet_PWC_Iter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PoolingProjEncoder(**config.encoder_cfg)


@FLOW_ESTIMATORS.register_module()
class LessNet_PWC_Cat(LessNet_PWC):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = WarpCorrCatPyramidalDecoder(**config.decoder_cfg)

@FLOW_ESTIMATORS.register_module()
class LessNet_PWC_Cat_v1(LessNet_PWC_Cat):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PoolingProjEncoder(**config.encoder_cfg)

@FLOW_ESTIMATORS.register_module()
class LessNet_PWCC_Iter(LessNet_PWC_Cat):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpCorrCatPyramidalDecoder(**config.decoder_cfg)

    def forward_eval(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        src_feats = src_feats[::-1]
        tgt_feats = tgt_feats[::-1]
        fwd = self.decoder.forward_eval(src_feats, tgt_feats)
        bck = self.decoder.forward_eval(tgt_feats, src_feats) if self.bidirectional else None
        return {'fwd': fwd, 'bck': bck}

@FLOW_ESTIMATORS.register_module()
class LessNet_PWCC_Iter_v1(LessNet_PWCC_Iter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PoolingProjEncoder(**config.encoder_cfg)


class LessWarpCorrPyramidalDecoder(WarpPyramidalDecoder):
    def __init__(self,
                 image_size: Sequence[int],
                 spatial_dims: int,
                 proj_channels: Sequence[int],
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 corr_radius: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 composition: str = 'compose',  # 'compose' or 'sum'
                 corr_mode: str = 'for',
                 proj_type: str = 'mlp',
                 *args, **kwargs,
                 ):
        super().__init__(image_size, spatial_dims,
                         skip_channels, out_channels,
                         out_indices, block_config, composition)
        self.corr_radius = corr_radius
        self.corr_mode = corr_mode
        self.win_vol = []
        self.corr = nn.ModuleList()
        for k, r in zip(self.out_indices, self.corr_radius):
            self.corr.append(
                GlobalCorrTorch(ndim=spatial_dims)
                if r == 0
                else WinCorrTorch(radius=r, ndim=spatial_dims, mode=self.corr_mode)
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
        self.proj = nn.ModuleList()
        for i in range(self.num_levels):
            self.proj.append(
                build_projection_layer(
                    spatial_dims,
                    skip_channels[i],
                    proj_channels[i],
                    proj_type=proj_type,
                )
            )


    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_dec = None
        prev_flow = None

        for i in range(self.num_levels):
            if i == 0:
                proj_src_feat = self.proj[i](src_feats[i])
                proj_tgt_feat = self.proj[i](tgt_feats[i])
                corr = self.corr[i](proj_tgt_feat, proj_src_feat)
                skip = torch.cat([corr, tgt_feats[i]], dim=1)
                prev_dec = self.decoder[i](prev_dec, skip)
                prev_flow = self.flow_convs[i](prev_dec)
            else:
                prev_flow = self.resize_flow(prev_flow)
                warp_src_feat = self.warp[i](src_feats[i], prev_flow)
                proj_warp_src_feat = self.proj[i](warp_src_feat)
                proj_tgt_feat = self.proj[i](tgt_feats[i])
                corr = self.corr[i](proj_tgt_feat, proj_warp_src_feat)
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
class LessNet_PWC_v2(LessNet_PWC):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = LessWarpCorrPyramidalDecoder(**config.decoder_cfg)


class WarpProjCorrPyramidalDecoder(WarpPyramidalDecoder):
    def __init__(self,
                 image_size: Sequence[int],
                 spatial_dims: int,
                 proj_channels: int,
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 corr_radius: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 composition: str = 'compose',  # 'compose' or 'sum'
                 corr_mode: str = 'for',
                 *args, **kwargs,
                 ):
        assert len(set(skip_channels)) == 1, \
            f"WarpProjCorrPyramidalDecoder requires equal skip_channels across all levels, got {list(skip_channels)}"
        super().__init__(image_size, spatial_dims,
                         skip_channels, out_channels,
                         out_indices, block_config, composition)
        self.corr_radius = corr_radius
        self.corr_mode = corr_mode
        self.win_vol = []
        self.corr = nn.ModuleList()
        for k, r in zip(self.out_indices, self.corr_radius):
            self.corr.append(
                GlobalCorrTorch(ndim=spatial_dims)
                if r == 0
                else WinCorrTorch(radius=r, ndim=spatial_dims, mode=self.corr_mode)
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
        self.proj = MlpChannel(
            spatial_dims=spatial_dims,
            in_features=skip_channels[0],
            hidden_features=proj_channels//2,
            out_features=proj_channels,
            act_layer=nn.LeakyReLU,
            drop=0.0,
            bias=True,
        )


    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_dec = None
        prev_flow = None

        for i in range(self.num_levels):
            if i == 0:
                proj_src_feat = self.proj(src_feats[i])
                proj_tgt_feat = self.proj(tgt_feats[i])
                corr = self.corr[i](proj_tgt_feat, proj_src_feat)
                skip = torch.cat([corr, tgt_feats[i]], dim=1)
                prev_dec = self.decoder[i](prev_dec, skip)
                prev_flow = self.flow_convs[i](prev_dec)
            else:
                prev_flow = self.resize_flow(prev_flow)
                warp_src_feat = self.warp[i](src_feats[i], prev_flow)
                proj_warp_src_feat = self.proj(warp_src_feat)
                proj_tgt_feat = self.proj(tgt_feats[i])
                corr = self.corr[i](proj_tgt_feat, proj_warp_src_feat)
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
class LessNet_PWC_v3(LessNet_PWC):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = WarpProjCorrPyramidalDecoder(**config.decoder_cfg)
