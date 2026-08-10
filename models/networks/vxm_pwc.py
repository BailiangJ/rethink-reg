"""Pyramidal flow decoders with warp (and optional correlation).

Decoder hierarchy
-----------------
CNNDecoder  (vxm.py)
└── WarpPyramidalDecoder          – warp-only coarse-to-fine
    └── WarpCorrPyramidalDecoder  – adds correlation as additional skip input
        └── WarpCorrCatPyramidalDecoder – also concatenates warped src feat

Key design notes
----------------
* ``_compose_flow`` is a shared helper that implements the two supported
  composition modes ('add' and 'compose') in one place.
* ``WarpCorrPyramidalDecoder`` avoids the previous double-build of
  ``self.decoder`` by overriding ``_build_decoder`` instead of running
  ``super().__init__`` and then rebuilding.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .utils import FlowConv
from ..builder import FLOW_ESTIMATORS
from ..utils import ResizeFlow, Warp
from .correlation import WinCorrTorch, GlobalCorrTorch
from .vxm import CNNEncoder, CNNDecoder, UpBlock


class WarpPyramidalDecoder(CNNDecoder):
    """Coarse-to-fine pyramidal decoder with warp-based feature alignment.

    At each level the previous flow is upsampled, the source feature is warped
    with it, and a new flow increment is predicted.  The increment is composed
    with the previous flow via ``composition`` ('add' or 'compose').

    Args:
        image_size:   full-resolution spatial size.
        spatial_dims: 2 or 3.
        skip_channels: encoder feature channels per level (finest first after
                       reversal in the parent model).
        out_channels:  decoder output channels per level.
        out_indices:   encoder pyramid indices corresponding to each level
                       (used to compute the spatial size of each warp module).
        block_config:  kwargs forwarded to every ``UpBlock``.
        composition:   'add' or 'compose' (default).
    """

    def __init__(
        self,
        image_size: Sequence[int],
        spatial_dims: int,
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
        out_indices: Sequence[int],
        block_config: dict,
        composition: str = 'compose',
        *args, **kwargs,
    ):
        super().__init__(spatial_dims, skip_channels, out_channels, block_config)
        self.out_indices = out_indices
        self.composition = composition

        self.flow_convs = nn.ModuleList([
            FlowConv(spatial_dims=spatial_dims, in_channels=c)
            for c in out_channels
        ])
        self.resize_flow = ResizeFlow(spatial_scale=2, flow_scale=2, ndim=spatial_dims)

        # Level 0 has no previous flow to warp with, so we use None as a
        # sentinel; subsequent levels get a proper Warp module.
        self.warp = nn.ModuleList([None] + [
            Warp(
                image_size=np.array(image_size) // (2 ** out_indices[i]),
                interp_mode='bilinear',
            )
            for i in range(1, self.num_levels)
        ])

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _compose_flow(
        self,
        warp: Warp,
        prev_flow: torch.Tensor,
        delta_flow: torch.Tensor,
    ) -> torch.Tensor:
        """Compose ``prev_flow`` with ``delta_flow`` using ``self.composition``."""
        if self.composition == 'add':
            return prev_flow + delta_flow
        if self.composition == 'compose':
            return warp(prev_flow, delta_flow.detach()) + delta_flow
        raise KeyError(f'Unsupported composition method: {self.composition!r}.')

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        src_feats: Sequence[torch.Tensor],
        tgt_feats: Sequence[torch.Tensor],
        init_dec: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        out_flows = []
        prev_dec  = init_dec
        prev_flow = None

        for i in range(self.num_levels):
            if i == 0:
                feats    = torch.cat([src_feats[i], tgt_feats[i]], dim=1)
                prev_dec  = self.decoder[i](prev_dec, feats)
                prev_flow = self.flow_convs[i](prev_dec)
            else:
                prev_flow     = self.resize_flow(prev_flow)
                warp_src      = self.warp[i](src_feats[i], prev_flow)
                feats         = torch.cat([warp_src, tgt_feats[i]], dim=1)
                prev_dec      = self.decoder[i](prev_dec, feats)
                delta_flow    = self.flow_convs[i](prev_dec)
                prev_flow     = self._compose_flow(self.warp[i], prev_flow, delta_flow)
            out_flows.append(prev_flow)

        return out_flows


class WarpCorrPyramidalDecoder(WarpPyramidalDecoder):
    """Warp-pyramidal decoder that also computes a correlation volume as an
    additional skip input at every level.

    The decoder blocks are rebuilt to accept the extra correlation channels;
    this is done in ``_build_decoder`` to avoid building them twice.

    Args:
        corr_radius:  per-level search radii; 0 means global correlation.
        corr_mode:    'for' (default) or other modes supported by
                      ``WinCorrTorch``.
        norm_vectors: if True, L2-normalise feature vectors before correlation.
        scale:        if True, divide correlation by sqrt(C).
        (other args inherited from WarpPyramidalDecoder)
    """

    def __init__(
        self,
        image_size: Sequence[int],
        spatial_dims: int,
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
        corr_radius: Sequence[int],
        out_indices: Sequence[int],
        block_config: dict,
        composition: str = 'compose',
        corr_mode: str = 'for',
        norm_vectors: bool = False,
        scale: bool = True,
        *args, **kwargs,
    ):
        super().__init__(
            image_size, spatial_dims,
            skip_channels, out_channels,
            out_indices, block_config, composition,
        )
        self.corr_radius  = corr_radius
        self.corr_mode    = corr_mode
        self.norm_vectors = norm_vectors
        self.scale        = scale

        # Correlation volume sizes (channels) per level
        self.win_vol: list[int] = []
        self.corr = nn.ModuleList()
        for k, r in zip(out_indices, corr_radius):
            self.corr.append(
                GlobalCorrTorch(
                    ndim=spatial_dims,
                    norm_vectors=norm_vectors,
                    scale=scale,
                )
                if r == 0
                else WinCorrTorch(
                    radius=r,
                    ndim=spatial_dims,
                    mode=corr_mode,
                    norm_vectors=norm_vectors,
                    scale=scale,
                )
            )
            self.win_vol.append(
                int(np.prod(np.array(image_size) // (2 ** k)))
                if r == 0
                else (2 * r + 1) ** spatial_dims
            )

        # Rebuild decoder blocks with the wider skip (skip + corr_vol)
        self.decoder = self._build_decoder(spatial_dims, skip_channels, out_channels, block_config)

    def _build_decoder(
        self,
        spatial_dims: int,
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
        block_config: dict,
    ) -> nn.ModuleList:
        """Build UpBlocks whose skip dimension includes the correlation volume."""
        return nn.ModuleList([
            UpBlock(
                spatial_dims=spatial_dims,
                in_channels=0 if i == 0 else out_channels[i - 1],
                skip_channels=self._skip_channels(i, skip_channels),
                out_channels=out_channels[i],
                **block_config,
            )
            for i in range(self.num_levels)
        ])

    def _skip_channels(self, level: int, skip_channels: Sequence[int]) -> int:
        """Total skip width at ``level`` = encoder feat + corr vol."""
        return skip_channels[level] + self.win_vol[level]

    def _build_skip(
        self,
        i: int,
        tgt_feat: torch.Tensor,
        src_feat: torch.Tensor,  # already warped if i > 0
    ) -> torch.Tensor:
        """Assemble the skip tensor for level ``i``."""
        corr = self.corr[i](tgt_feat, src_feat)
        return torch.cat([corr, tgt_feat], dim=1)

    def forward(
        self,
        src_feats: Sequence[torch.Tensor],
        tgt_feats: Sequence[torch.Tensor],
        init_dec: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        out_flows = []
        prev_dec  = init_dec
        prev_flow = None

        for i in range(self.num_levels):
            if i == 0:
                skip      = self._build_skip(i, tgt_feats[i], src_feats[i])
                prev_dec  = self.decoder[i](prev_dec, skip)
                prev_flow = self.flow_convs[i](prev_dec)
            else:
                prev_flow  = self.resize_flow(prev_flow)
                warp_src   = self.warp[i](src_feats[i], prev_flow)
                skip       = self._build_skip(i, tgt_feats[i], warp_src)
                prev_dec   = self.decoder[i](prev_dec, skip)
                delta_flow = self.flow_convs[i](prev_dec)
                prev_flow  = self._compose_flow(self.warp[i], prev_flow, delta_flow)
            out_flows.append(prev_flow)

        return out_flows


class WarpCorrCatPyramidalDecoder(WarpCorrPyramidalDecoder):
    """Like ``WarpCorrPyramidalDecoder`` but also concatenates the (warped)
    source feature into the skip connection: skip = [corr | tgt | src]."""

    def _skip_channels(self, level: int, skip_channels: Sequence[int]) -> int:
        return 2 * skip_channels[level] + self.win_vol[level]

    def _build_skip(
        self,
        i: int,
        tgt_feat: torch.Tensor,
        src_feat: torch.Tensor,
    ) -> torch.Tensor:
        corr = self.corr[i](tgt_feat, src_feat)
        return torch.cat([corr, tgt_feat, src_feat], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Top-level registered models
# ══════════════════════════════════════════════════════════════════════════════

@FLOW_ESTIMATORS.register_module()
class VXMPlus_DualWarpPy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.encoder = CNNEncoder(**config.encoder_cfg)
        self.decoder = WarpPyramidalDecoder(**config.decoder_cfg)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_feats = self.encoder(src)[::-1]
        tgt_feats = self.encoder(tgt)[::-1]
        flows     = self.decoder(src_feats, tgt_feats)
        bck_flows = self.decoder(tgt_feats, src_feats) if self.bidirectional else None
        return flows, bck_flows


@FLOW_ESTIMATORS.register_module()
class VXMPlus_PWC(VXMPlus_DualWarpPy):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = WarpCorrPyramidalDecoder(**config.decoder_cfg)
