"""Iterative pyramidal flow decoders.

These extend the decoders in ``vxm_pwc.py`` by running multiple refinement
iterations at each pyramid level instead of a single pass.

Design
------
``_IterPyramidalMixin`` is a pure-logic mixin (not an ``nn.Module``) that:

  1. ``_setup_iter`` – saves ``num_iters`` and rebuilds ``self.warp`` so
     that *every* level (including level 0) has a warp module.  In the
     non-iter base, ``self.warp[0]`` is ``None`` because there is no
     previous flow at the first level; here iterations after the first can
     produce a flow that is then used within the same level.

  2. ``forward_flow(i, src_feat, tgt_feat, prev_dec, prev_flow)`` –
     abstract; subclasses implement the per-level, per-iteration logic.
     Modules are accessed via ``self`` (indexed by ``i``), avoiding the
     verbose explicit-module-passing used in the previous version.

  3. ``forward`` / ``forward_eval`` – shared loop bodies.  ``forward``
     returns only the final flow per level; ``forward_eval`` collects every
     intermediate iteration's flow for analysis / visualisation.

MRO note
--------
Mixin is listed *before* the decoder base so its ``forward`` /
``forward_eval`` shadow the base class versions:

    class IterWarpPyramidalDecoder(_IterPyramidalMixin, WarpPyramidalDecoder)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..builder import FLOW_ESTIMATORS
from ..utils import Warp
from .vxm_pwc import (
    WarpPyramidalDecoder,
    WarpCorrPyramidalDecoder,
    WarpCorrCatPyramidalDecoder,
    VXMPlus_DualWarpPy,
    VXMPlus_PWC,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared mixin
# ══════════════════════════════════════════════════════════════════════════════

class _IterPyramidalMixin:
    """Mixin that adds iterative per-level refinement to any
    ``WarpPyramidalDecoder`` subclass.

    Must be listed *before* the decoder base in the MRO so that ``forward``
    and ``forward_eval`` defined here take precedence.
    """

    # ------------------------------------------------------------------
    # Setup (called from __init__ of each concrete class)
    # ------------------------------------------------------------------

    def _setup_iter(self, image_size: Sequence[int], num_iters: Sequence[int]) -> None:
        """Save ``num_iters`` and rebuild ``self.warp`` for *all* levels.

        The non-iter base sets ``self.warp[0] = None`` because there is no
        previous flow at the very first level.  In the iterative variant,
        after the first iteration of level 0 there *is* a flow, so all
        levels need a proper Warp module.
        """
        self.num_iters = num_iters
        self.warp = nn.ModuleList([
            Warp(
                image_size=np.array(image_size) // (2 ** self.out_indices[i]),
                interp_mode='bilinear',
            )
            for i in range(self.num_levels)
        ])

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    def forward_flow(
        self,
        i: int,
        src_feat: torch.Tensor,
        tgt_feat: torch.Tensor,
        prev_dec: Optional[torch.Tensor],
        prev_flow: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single iteration at pyramid level ``i``.

        Args:
            i:         level index (0 = coarsest).
            src_feat:  source feature at level ``i`` (not yet warped).
            tgt_feat:  target feature at level ``i``.
            prev_dec:  decoder feature from the previous iteration / level.
            prev_flow: accumulated flow at the *current* level resolution
                       (already upsampled from the coarser level before the
                       level loop starts).

        Returns:
            (flow, dec): updated accumulated flow and decoder feature.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared forward / forward_eval
    # ------------------------------------------------------------------

    def forward(
        self,
        src_feats: Sequence[torch.Tensor],
        tgt_feats: Sequence[torch.Tensor],
        init_dec: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Returns one flow per pyramid level (final iteration only)."""
        out_flows: list[torch.Tensor] = []
        prev_dec  = init_dec
        prev_flow = None

        for i in range(self.num_levels):
            if prev_flow is not None:
                prev_flow = self.resize_flow(prev_flow)
            for _ in range(self.num_iters[i]):
                prev_flow, dec = self.forward_flow(
                    i, src_feats[i], tgt_feats[i], prev_dec, prev_flow,
                )
            prev_dec = dec
            if prev_flow is not None:
                out_flows.append(prev_flow)

        return out_flows

    def forward_eval(
        self,
        src_feats: Sequence[torch.Tensor],
        tgt_feats: Sequence[torch.Tensor],
        init_dec: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Returns *every* intermediate flow (all levels × all iterations)."""
        iter_flows: list[torch.Tensor] = []
        prev_dec  = init_dec
        prev_flow = None

        for i in range(self.num_levels):
            if prev_flow is not None:
                prev_flow = self.resize_flow(prev_flow)
            for _ in range(self.num_iters[i]):
                prev_flow, dec = self.forward_flow(
                    i, src_feats[i], tgt_feats[i], prev_dec, prev_flow,
                )
                iter_flows.append(prev_flow)
            prev_dec = dec

        return iter_flows


# ══════════════════════════════════════════════════════════════════════════════
# Concrete iterative decoders
# ══════════════════════════════════════════════════════════════════════════════

class IterWarpPyramidalDecoder(_IterPyramidalMixin, WarpPyramidalDecoder):
    """Iterative warp-only pyramidal decoder."""

    def __init__(
        self,
        image_size: Sequence[int],
        spatial_dims: int,
        num_iters: Sequence[int],
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
        out_indices: Sequence[int],
        block_config: dict,
        composition: str = 'compose',
    ):
        super().__init__(
            image_size, spatial_dims,
            skip_channels, out_channels,
            out_indices, block_config, composition,
        )
        self._setup_iter(image_size, num_iters)

    def forward_flow(self, i, src_feat, tgt_feat, prev_dec, prev_flow):
        if prev_flow is not None:
            src_feat = self.warp[i](src_feat, prev_flow)
        feats    = torch.cat([src_feat, tgt_feat], dim=1)
        dec      = self.decoder[i](prev_dec, feats)
        flow     = self.flow_convs[i](dec)
        if prev_flow is not None:
            flow = self._compose_flow(self.warp[i], prev_flow, flow)
        return flow, dec


class IterWarpCorrPyramidalDecoder(_IterPyramidalMixin, WarpCorrPyramidalDecoder):
    """Iterative warp + correlation pyramidal decoder."""

    def __init__(
        self,
        image_size: Sequence[int],
        spatial_dims: int,
        num_iters: Sequence[int],
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
        corr_radius: Sequence[int],
        out_indices: Sequence[int],
        block_config: dict,
        composition: str = 'compose',
        corr_mode: str = 'for',
        norm_vectors: bool = False,
        scale: bool = True,
    ):
        super().__init__(
            image_size=image_size,
            spatial_dims=spatial_dims,
            skip_channels=skip_channels,
            out_channels=out_channels,
            corr_radius=corr_radius,
            out_indices=out_indices,
            block_config=block_config,
            composition=composition,
            corr_mode=corr_mode,
            norm_vectors=norm_vectors,
            scale=scale,
        )
        self._setup_iter(image_size, num_iters)

    def forward_flow(self, i, src_feat, tgt_feat, prev_dec, prev_flow):
        if prev_flow is not None:
            src_feat = self.warp[i](src_feat, prev_flow)
        skip     = self._build_skip(i, tgt_feat, src_feat)
        dec      = self.decoder[i](prev_dec, skip)
        flow     = self.flow_convs[i](dec)
        if prev_flow is not None:
            flow = self._compose_flow(self.warp[i], prev_flow, flow)
        return flow, dec


class IterWarpCorrCatPyramidalDecoder(_IterPyramidalMixin, WarpCorrCatPyramidalDecoder):
    """Iterative warp + correlation + source-cat pyramidal decoder."""

    def __init__(
        self,
        image_size: Sequence[int],
        spatial_dims: int,
        num_iters: Sequence[int],
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
        corr_radius: Sequence[int],
        out_indices: Sequence[int],
        block_config: dict,
        composition: str = 'compose',
        corr_mode: str = 'for',
        norm_vectors: bool = False,
        scale: bool = True,
    ):
        super().__init__(
            image_size=image_size,
            spatial_dims=spatial_dims,
            skip_channels=skip_channels,
            out_channels=out_channels,
            corr_radius=corr_radius,
            out_indices=out_indices,
            block_config=block_config,
            composition=composition,
            corr_mode=corr_mode,
            norm_vectors=norm_vectors,
            scale=scale,
        )
        self._setup_iter(image_size, num_iters)

    def forward_flow(self, i, src_feat, tgt_feat, prev_dec, prev_flow):
        if prev_flow is not None:
            src_feat = self.warp[i](src_feat, prev_flow)
        skip     = self._build_skip(i, tgt_feat, src_feat)  # [corr | tgt | src]
        dec      = self.decoder[i](prev_dec, skip)
        flow     = self.flow_convs[i](dec)
        if prev_flow is not None:
            flow = self._compose_flow(self.warp[i], prev_flow, flow)
        return flow, dec


# ══════════════════════════════════════════════════════════════════════════════
# Top-level registered models
# ══════════════════════════════════════════════════════════════════════════════

@FLOW_ESTIMATORS.register_module()
class VXMPlus_DualWarpPy_Iter(VXMPlus_DualWarpPy):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpPyramidalDecoder(**config.decoder_cfg)

    def forward_eval(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        src_feats = self.encoder(src)[::-1]
        tgt_feats = self.encoder(tgt)[::-1]
        fwd = self.decoder.forward_eval(src_feats, tgt_feats)
        bck = self.decoder.forward_eval(tgt_feats, src_feats) if self.bidirectional else None
        return {'fwd': fwd, 'bck': bck}


@FLOW_ESTIMATORS.register_module()
class VXMPlus_PWC_Iter(VXMPlus_PWC):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = IterWarpCorrPyramidalDecoder(**config.decoder_cfg)

    def forward_eval(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        src_feats = self.encoder(src)[::-1]
        tgt_feats = self.encoder(tgt)[::-1]
        fwd = self.decoder.forward_eval(src_feats, tgt_feats)
        bck = self.decoder.forward_eval(tgt_feats, src_feats) if self.bidirectional else None
        return {'fwd': fwd, 'bck': bck}
