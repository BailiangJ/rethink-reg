from __future__ import annotations
from typing import Sequence, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import get_conv_layer
from .matching import CorrelationMatching
from .vxm import CNNEncoder
from ..builder import FLOW_ESTIMATORS
from ..utils import ResizeFlow, Warp


class VectorFieldAttention(nn.Module):
    """Single-scale VFA block: optionally projects features, computes a correlation-based
    displacement field via softmax expectation, and scales the result by a learnable beta.

    Args:
        image_size:    Spatial dimensions of the input feature map (D, H, W) or (H, W).
        radius:        Correlation window half-size. ``radius > 0`` → window correlation;
                       ``radius = 0`` → global (all-pairs) correlation.
        corr_mode:     Correlation computation backend: ``'for'`` (memory-efficient loop)
                       or ``'einsum'`` (faster, higher memory).
        norm_vectors:  If True, L2-normalise features before correlation (cosine similarity).
        scale:         If True, divide correlation logits by sqrt(C) (scaled dot-product).
        beta:          Initial value of the learnable output scale factor.
        in_channels:   Number of input feature channels. If provided together with
                       ``proj_channels``, a 3×3 conv projection is inserted before matching.
        proj_channels: Number of channels in the projection space for matching.
    """
    def __init__(self,
                 image_size: Sequence[int],
                 radius: int = 3,
                 corr_mode: str = 'for',
                 norm_vectors: bool = True,
                 scale: bool = True,
                 beta: float = 0.1,
                 in_channels: Optional[int] = None,
                 proj_channels: Optional[int] = None):
        super().__init__()

        self.matching = CorrelationMatching(
            image_size=image_size,
            radius=radius,
            corr_mode=corr_mode,
            norm_vectors=norm_vectors,
            scale=scale,
        )

        if in_channels is not None and proj_channels is not None:
            self.proj = get_conv_layer(
                len(image_size),
                in_channels,
                proj_channels,
                kernel_size=3,
                stride=1,
                bias=True,
                norm=None,
                act=None,
                dropout=None,
            )
        else:
            self.proj = None

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            src_feat = self.proj(src_feat)
            tgt_feat = self.proj(tgt_feat)

        flow, _ = self.matching(src_feat, tgt_feat)
        return flow * self.beta


class VFAPyramidalDecoder(nn.Module):
    def __init__(self,
                 image_size: Sequence[int],
                 skip_channels: Sequence[int],
                 proj_channels: Sequence[int],
                 corr_radius: Sequence[int],
                 out_indices: Sequence[int],
                 beta: Union[float, Sequence[float]] = 1.0,
                 norm_vectors: bool = True,
                 scale: bool = True,
                 corr_mode: str = 'for',
                 composition: str = 'compose',  # 'compose' or 'add'
                 ):
        super().__init__()
        self.out_indices = out_indices
        self.composition = composition
        self.num_levels = len(out_indices)
        if isinstance(beta, float):
            beta_list = [beta] * self.num_levels
        else:
            assert len(beta) == self.num_levels, "Length of beta must match number of levels"
            beta_list = beta
        self.decoder = nn.ModuleList()
        for i in range(self.num_levels):
            # NOTE: proj_channels[i] controls the matching dimensionality per level.
            # The original VFA used min(max_channels, matching_channels * 2**i).
            self.decoder.append(
                VectorFieldAttention(
                    image_size=np.array(image_size) // (2 ** self.out_indices[i]),
                    radius=corr_radius[i],
                    corr_mode=corr_mode,
                    in_channels=skip_channels[i],
                    proj_channels=proj_channels[i],
                    beta=beta_list[i],
                    norm_vectors=norm_vectors,
                    scale=scale,
                )
            )

        self.resize_flow = ResizeFlow(
            spatial_scale=2,
            flow_scale=2,
            ndim=len(image_size)
        )
        self.warp = nn.ModuleList()
        self.warp.append(None)  # No warping at the coarsest level
        for i in range(1, self.num_levels):
            self.warp.append(
                Warp(
                    image_size=np.array(image_size) // (2 ** self.out_indices[i]),
                    interp_mode='bilinear')
            )

    def forward(self, src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        out_flows = []
        prev_flow = None

        for i in range(self.num_levels):
            if prev_flow is None:
                prev_flow = self.decoder[i](src_feats[i], tgt_feats[i])
            else:
                prev_flow = self.resize_flow(prev_flow)
                warp_src_feat = self.warp[i](src_feats[i], prev_flow)
                curr_flow = self.decoder[i](warp_src_feat, tgt_feats[i])
                if self.composition == 'compose':
                    prev_flow = self.warp[i](prev_flow, curr_flow.detach()) + curr_flow
                elif self.composition == 'add':
                    prev_flow = curr_flow + prev_flow
                else:
                    raise ValueError(f"Unknown composition mode: {self.composition}")
            if prev_flow is not None:
                out_flows.append(prev_flow)

        return out_flows


@FLOW_ESTIMATORS.register_module()
class VXMVFA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.encoder = CNNEncoder(**config.encoder_cfg)
        self.decoder = VFAPyramidalDecoder(**config.decoder_cfg)

    def forward(self, src:torch.Tensor, tgt:torch.Tensor)->torch.Tensor:
        src_feats = self.encoder(src)
        tgt_feats = self.encoder(tgt)
        src_feats = src_feats[::-1]
        tgt_feats = tgt_feats[::-1]
        flows = self.decoder(src_feats, tgt_feats)
        if self.bidirectional:
            bck_flows = self.decoder(tgt_feats, src_feats)
        return (flows, bck_flows) if self.bidirectional else (flows, None)
