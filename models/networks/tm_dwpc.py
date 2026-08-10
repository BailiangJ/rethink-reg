"""TransMorph variants with pyramidal flow decoders (Dual-Warp and PWC).

Architecture overview
---------------------
Two abstract base families, each parameterised by a ``_TM_BASE`` class
variable (either ``TransMorph2D`` or ``TransMorph3D``):

  _TM_DualBase          – keeps the full TM UNet decoder; runs src/tgt
                          through the transformer separately and concatenates
                          their features at every decoder level.
      _TM_DualBase2D / _TM_DualBase3D

  _TM_PyramidalBase     – drops the TM UNet decoder; keeps only the
                          transformer encoder + c1 + avg_pool, feeding
                          features into a replaceable pyramidal flow decoder.
      _TM_PyramidalBase2D / _TM_PyramidalBase3D

"""

import torch
import torch.nn as nn
import copy

from ..builder import FLOW_ESTIMATORS
from .transmorph_2d import TransMorph2D
from .transmorph_3d import TransMorph3D
from .vxm import UpBlock
from .vxm_pwc import WarpPyramidalDecoder, WarpCorrPyramidalDecoder
from .vxm_iter import (
    IterWarpPyramidalDecoder,
    IterWarpCorrPyramidalDecoder,
)


# ══════════════════════════════════════════════════════════════════════════════
# Family 1: Dual variant of TransMorph
# ══════════════════════════════════════════════════════════════════════════════

class _TM_DualBase(nn.Module):
    """
    Channel accounting  (E = config.embed_dim, the actual transformer width)
    ------------------
    Encoder built with embed_dim = E, outputs per branch:
        [E, 2E, 4E, 8E]   (levels 0 → 3, finest → coarsest)

    After src + tgt cat:
        bottleneck  8E + 8E = 16E   → decoder needs base width 2E
        f1 skip     4E + 4E =  8E
        f2 skip     2E + 2E =  4E
        f3 skip      E +  E =  2E

    Convskip: c1 outputs E//2 per branch → cat → E  (2E base → up3 skip = E) ✓

    Decoder is taken from a double-embed base (embed_dim = 2E) so its
    block sizes match the concatenated features exactly.
    """
    _TM_BASE: type  # set by _TM_DualBase2D / _TM_DualBase3D

    def __init__(self, config):
        super().__init__()

        # Normal encoder at embed_dim = E
        base = self._TM_BASE(config)

        # Double-embed decoder: built with embed_dim = 2E so its block
        # channel sizes match the concatenated (src + tgt) feature widths
        double_config = copy.copy(config)
        double_config.embed_dim = config.embed_dim * 2
        double_base = self._TM_BASE(double_config)

        self.transformer  = base.transformer
        self.c1           = base.c1
        self.avg_pool     = base.avg_pool
        self.up0          = double_base.up0
        self.up1          = double_base.up1
        self.up2          = double_base.up2
        self.up3          = double_base.up3
        self.flow_conv    = double_base.flow_conv
        self.if_convskip  = config.if_convskip
        self.if_transskip = config.if_transskip
        self.bidirectional = getattr(config, 'bidirectional', False)

    def _decode(self, src_feats, tgt_feats, source, target):
        if self.if_convskip:
            f4 = torch.cat([self.c1(self.avg_pool(source)),
                            self.c1(self.avg_pool(target))], dim=1)
        else:
            f4 = None

        if self.if_transskip:
            f1 = torch.cat([src_feats[-2], tgt_feats[-2]], dim=1)
            f2 = torch.cat([src_feats[-3], tgt_feats[-3]], dim=1)
            f3 = torch.cat([src_feats[-4], tgt_feats[-4]], dim=1)
        else:
            f1, f2, f3 = None, None, None

        x = self.up0(torch.cat([src_feats[-1], tgt_feats[-1]], dim=1), f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        return self.flow_conv(x)

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        src_feats = self.transformer(source)
        tgt_feats = self.transformer(target)

        fwd_flow = self._decode(src_feats, tgt_feats, source, target)
        bck_flow = (self._decode(tgt_feats, src_feats, target, source)
                    if self.bidirectional else None)
        return fwd_flow, bck_flow


class _TM_DualBase2D(_TM_DualBase):
    _TM_BASE = TransMorph2D

class _TM_DualBase3D(_TM_DualBase):
    _TM_BASE = TransMorph3D


# ══════════════════════════════════════════════════════════════════════════════
# Family 2: Pyramidal variants of TransMorph
# ══════════════════════════════════════════════════════════════════════════════

class _TM_PyramidalBase(nn.Module):
    """Runs src/tgt through the TM encoder (at real ``embed_dim``) separately,
    then feeds the per-branch feature pyramids into a replaceable pyramidal
    flow decoder.

    The encoder is the normal base (not halved): each branch produces features
    at channels [E, 2E, 4E, 8E] (E = embed_dim).  These are passed separately
    to the pyramidal decoder, so ``decoder_cfg.skip_channels`` should reflect
    the actual per-branch sizes.

    Args:
        config:        model config; ``embed_dim`` is the actual transformer
                       embedding dimension. Must also have ``bidirectional``
                       and ``if_convskip``.
        decoder_class: a ``WarpPyramidalDecoder`` subclass instantiated with
                       ``config.decoder_cfg``.
    """
    _TM_BASE: type  # set by _TM_PyramidalBase2D / _TM_PyramidalBase3D

    def __init__(self, config, decoder_class):
        super().__init__()
        base = self._TM_BASE(config)
        self.transformer   = base.transformer
        self.c1            = base.c1
        self.avg_pool      = base.avg_pool
        self.if_convskip   = base.if_convskip
        self.bidirectional = config.bidirectional
        self.decoder = decoder_class(**config.decoder_cfg)

        # Auto-detect whether the encoder produces more feature levels than
        # the decoder expects.  If so, build a pre_block that decodes the
        # extra coarsest level(s) without producing flow.
        n_encoder = len(config.out_indices) + (1 if config.if_convskip else 0)
        n_decoder = self.decoder.num_levels
        n_pre = n_encoder - n_decoder

        pre_out = getattr(config, 'pre_out_channels', 0)
        if n_pre > 0 and pre_out > 0:
            E = config.embed_dim
            coarsest_ch = E * (2 ** (len(config.out_indices) - 1))
            pre_skip = 2 * coarsest_ch  # src + tgt concat

            self.pre_block = UpBlock(
                spatial_dims=config.spatial_dims,
                in_channels=0,
                skip_channels=pre_skip,
                out_channels=pre_out,
                **config.decoder_cfg['block_config'],
            )

            # Rebuild decoder level 0 to accept pre_block output as prev_dec
            dcfg = config.decoder_cfg
            if hasattr(self.decoder, '_skip_channels'):
                skip_0 = self.decoder._skip_channels(0, dcfg['skip_channels'])
            else:
                skip_0 = dcfg['skip_channels'][0]

            self.decoder.decoder[0] = UpBlock(
                spatial_dims=config.spatial_dims,
                in_channels=pre_out,
                skip_channels=skip_0,
                out_channels=dcfg['out_channels'][0],
                **dcfg['block_config'],
            )
        else:
            self.pre_block = None

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        """
        Args:
            source: [B, C, (D,) H, W]
            target: [B, C, (D,) H, W]
        Returns:
            (fwd_flows, bck_flows): lists of flow tensors at each pyramid
            level; bck_flows is None when not bidirectional.
        """
        src_feats = self.transformer(source)[::-1]  # coarse → fine
        tgt_feats = self.transformer(target)[::-1]

        if self.if_convskip:
            src_feats = src_feats + [self.c1(self.avg_pool(source))]
            tgt_feats = tgt_feats + [self.c1(self.avg_pool(target))]

        init_dec, bck_init_dec = None, None
        if self.pre_block is not None:
            init_dec = self.pre_block(
                None, torch.cat([src_feats[0], tgt_feats[0]], dim=1))
            if self.bidirectional:
                bck_init_dec = self.pre_block(
                    None, torch.cat([tgt_feats[0], src_feats[0]], dim=1))
            src_feats = src_feats[1:]
            tgt_feats = tgt_feats[1:]

        fwd_flows = self.decoder(src_feats, tgt_feats, init_dec=init_dec)
        bck_flows = (self.decoder(tgt_feats, src_feats, init_dec=bck_init_dec)
                     if self.bidirectional else None)
        return fwd_flows, bck_flows

    def forward_eval(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        """Return decoder evaluation flows for pyramid/intermediate analysis.

        Iterative decoders expose every intermediate refinement through
        ``decoder.forward_eval``; non-iterative decoders fall back to their
        normal forward pass, which returns one flow per pyramid level.
        """
        src_feats = self.transformer(source)[::-1]  # coarse → fine
        tgt_feats = self.transformer(target)[::-1]

        if self.if_convskip:
            src_feats = src_feats + [self.c1(self.avg_pool(source))]
            tgt_feats = tgt_feats + [self.c1(self.avg_pool(target))]

        init_dec, bck_init_dec = None, None
        if self.pre_block is not None:
            init_dec = self.pre_block(
                None, torch.cat([src_feats[0], tgt_feats[0]], dim=1))
            if self.bidirectional:
                bck_init_dec = self.pre_block(
                    None, torch.cat([tgt_feats[0], src_feats[0]], dim=1))
            src_feats = src_feats[1:]
            tgt_feats = tgt_feats[1:]

        decoder_eval = getattr(self.decoder, 'forward_eval', None)
        if decoder_eval is None:
            fwd_flows = self.decoder(src_feats, tgt_feats, init_dec=init_dec)
            bck_flows = (self.decoder(tgt_feats, src_feats, init_dec=bck_init_dec)
                         if self.bidirectional else None)
        else:
            fwd_flows = decoder_eval(src_feats, tgt_feats, init_dec=init_dec)
            bck_flows = (decoder_eval(tgt_feats, src_feats, init_dec=bck_init_dec)
                         if self.bidirectional else None)

        return {'fwd': fwd_flows, 'bck': bck_flows}


class _TM_PyramidalBase2D(_TM_PyramidalBase):
    _TM_BASE = TransMorph2D

class _TM_PyramidalBase3D(_TM_PyramidalBase):
    _TM_BASE = TransMorph3D


# ══════════════════════════════════════════════════════════════════════════════
# Registered models
# Naming: TransMorph{2D|3D}_<variant>  /  TM{2D|3D}_<variant>
# ══════════════════════════════════════════════════════════════════════════════

# ── Dual UNet decoder ─────────────────────────────────────────────────────────

@FLOW_ESTIMATORS.register_module()
class TransMorph2D_Dual(_TM_DualBase2D):
    def __init__(self, config):
        super().__init__(config)

@FLOW_ESTIMATORS.register_module()
class TransMorph3D_Dual(_TM_DualBase3D):
    def __init__(self, config):
        super().__init__(config)

# ── DualWarp pyramidal decoder ────────────────────────────────────────────────

@FLOW_ESTIMATORS.register_module()
class TM2D_DualWarpPy(_TM_PyramidalBase2D):
    def __init__(self, config):
        super().__init__(config, WarpPyramidalDecoder)

@FLOW_ESTIMATORS.register_module()
class TM3D_DualWarpPy(_TM_PyramidalBase3D):
    def __init__(self, config):
        super().__init__(config, WarpPyramidalDecoder)

@FLOW_ESTIMATORS.register_module()
class TM2D_DualWarpPy_Iter(_TM_PyramidalBase2D):
    def __init__(self, config):
        super().__init__(config, IterWarpPyramidalDecoder)

@FLOW_ESTIMATORS.register_module()
class TM3D_DualWarpPy_Iter(_TM_PyramidalBase3D):
    def __init__(self, config):
        super().__init__(config, IterWarpPyramidalDecoder)

# ── PWC (correlation) pyramidal decoder ──────────────────────────────────────

@FLOW_ESTIMATORS.register_module()
class TM2D_PWC(_TM_PyramidalBase2D):
    def __init__(self, config):
        super().__init__(config, WarpCorrPyramidalDecoder)

@FLOW_ESTIMATORS.register_module()
class TM3D_PWC(_TM_PyramidalBase3D):
    def __init__(self, config):
        super().__init__(config, WarpCorrPyramidalDecoder)

@FLOW_ESTIMATORS.register_module()
class TM2D_PWC_Iter(_TM_PyramidalBase2D):
    def __init__(self, config):
        super().__init__(config, IterWarpCorrPyramidalDecoder)

@FLOW_ESTIMATORS.register_module()
class TM3D_PWC_Iter(_TM_PyramidalBase3D):
    def __init__(self, config):
        super().__init__(config, IterWarpCorrPyramidalDecoder)
