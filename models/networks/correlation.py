from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Window-based Correlation
# ──────────────────────────────────────────────────────────────────────

class WinCorrTorch(nn.Module):
    """Unified window correlation for 2D and 3D.

    Supports three modes:
    - 'for': Iterates over window offsets one at a time, keeping peak GPU memory low.
             Trade-off: slower than the einsum variant but uses ~15× less memory.
    - 'einsum': Extracts all patches via unfold, then computes correlation in one einsum.
                Trade-off: faster than the for-loop variant but uses much more memory.
    - 'cuda': Uses the optional standalone corr_cuda extension. This backend is loaded
              lazily and must be explicitly requested.

    Args:
        radius:       half-window size; window = 2*radius+1
        ndim:         number of spatial dimensions (2 or 3)
        mode:         computation mode ('for', 'einsum', or 'cuda')
        norm_vectors: if True, L2-normalise feature vectors before correlation;
                      makes inner product equivalent to cosine similarity
        scale:        if True, divide output by sqrt(C) (scaled dot-product attention);
                      disable when the calling module applies its own scaling
        padding:      padding mode passed to ``F.pad`` for the search tensor.
                      'constant' (zero) matches the backup reference and is used for testing;
                      'replicate' avoids boundary artefacts and is recommended for production.
    """

    def __init__(self,
                radius: int = 3,
                ndim: int = 3,
                mode: str = 'for',
                norm_vectors: bool = False,
                scale: bool = True,
                padding: str = 'constant'):
        super().__init__()
        self.radius = radius
        self.ndim = ndim
        self.mode = mode
        self.norm_vectors = norm_vectors
        self.scale = scale
        self.padding = padding
        self.win_size = 2 * radius + 1

        if self.mode == 'for':
            # Pre-compute all offsets once: list of ndim-tuples
            self.offsets = list(product(range(self.win_size), repeat=ndim))
        elif self.mode == 'cuda':
            try:
                from corr_cuda import WinCorrCuda
            except ImportError as exc:
                raise ImportError(
                    "WinCorrTorch(mode='cuda') requires the standalone corr_cuda package. "
                    "Build it with: pip install -e ./corr_cuda --no-build-isolation"
                ) from exc
            self.cuda_corr = WinCorrCuda(
                radius=self.radius,
                ndim=self.ndim,
                norm_vectors=False,
                scale=self.scale,
                padding=self.padding,
                auto_contiguous=True,
            )
        elif self.mode != 'einsum':
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.norm_vectors:
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)

        if self.mode == 'for':
            return self._forward_for(x, y)
        elif self.mode == 'einsum':
            return self._forward_einsum(x, y)
        elif self.mode == 'cuda':
            return self._forward_cuda(x, y)

    def _forward_for(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        b, c = x.shape[:2]
        spatial = x.shape[2:]  # (D, H, W) or (H, W)

        # 'constant' (zero) padding matches the backup reference for testing;
        # swap to 'replicate' to avoid boundary artefacts in production.
        y_padded = F.pad(y, [self.radius] * (self.ndim * 2), mode=self.padding)

        # Each iteration: slice y_padded at one offset, dot with x over C
        corr = torch.cat(
            [
                (x * y_padded[(..., *[slice(o, o + s) for o, s in zip(off, spatial)])])
                .sum(dim=1, keepdim=True)
                for off in self.offsets
            ],
            dim=1,
        )
        return corr * (c ** -0.5) if self.scale else corr

    def _forward_einsum(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        b, c = x.shape[:2]
        spatial = x.shape[2:]

        # 'constant' (zero) padding matches the backup reference for testing;
        # swap to 'replicate' to avoid boundary artefacts in production.
        y_padded = F.pad(y, [self.radius] * (self.ndim * 2), mode=self.padding)

        # Chained unfold: (B, C, *spatial, K, K, [K])  — strided view, no copy
        y_patches = y_padded
        for dim in range(2, 2 + self.ndim):
            y_patches = y_patches.unfold(dim, self.win_size, 1)

        # Einsum directly on the unfold view (avoids bad reshape on strided tensor)
        if self.ndim == 3:
            corr = torch.einsum('bcdhw, bcdhwijk -> bdhwijk', x, y_patches)
            corr = corr.flatten(start_dim=-3).permute(0, 4, 1, 2, 3)
        else:
            corr = torch.einsum('bchw, bchwij -> bhwij', x, y_patches)
            corr = corr.flatten(start_dim=-2).permute(0, 3, 1, 2)

        return corr * (c ** -0.5) if self.scale else corr

    def _forward_cuda(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.cuda_corr(x, y)


class WinCorrTorchFast(WinCorrTorch):
    """Legacy alias for backward compatibility. Uses mode='einsum'."""

    def __init__(self, radius: int = 3, ndim: int = 3, norm_vectors: bool = False):
        super().__init__(radius=radius, ndim=ndim, mode='einsum', norm_vectors=norm_vectors)


# ──────────────────────────────────────────────────────────────────────
# Global Correlation
# ──────────────────────────────────────────────────────────────────────

class GlobalCorrTorch(nn.Module):
    """Unified global correlation for 2D and 3D.

    Computes all-pairs dot product between every pair of spatial locations.

    Note:
        This operation is **not commutative**: ``forward(x, y)`` and ``forward(y, x)``
        produce transposed correlation matrices. Concretely, ``forward(x, y)`` outputs
        shape ``(B, y_locs, *x_spatial)``, placing y's locations as the candidate
        dimension and x's spatial layout as the query dimensions. The caller must
        therefore pass arguments in the correct role order.

    Args:
        ndim:         number of spatial dimensions (2 or 3), used only for documentation
        norm_vectors: if True, L2-normalise feature vectors before computing the dot
                      product, making the result equivalent to cosine similarity
        scale:        if True, divide output by sqrt(C) (scaled dot-product attention);
                      disable when the calling module applies its own scaling
    """

    def __init__(self, ndim: int = 3, norm_vectors: bool = False, scale: bool = True):
        super().__init__()
        self.ndim = ndim
        self.norm_vectors = norm_vectors
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        if self.norm_vectors:
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)

        b, c = x.shape[:2]
        spatial = x.shape[2:]

        x_flat = x.view(b, c, -1)
        y_flat = y.view(b, c, -1)
        corr = torch.einsum('bci, bcj -> bji', x_flat, y_flat).view(b, -1, *spatial)
        return corr * (c ** -0.5) if self.scale else corr


# ──────────────────────────────────────────────────────────────────────
# Legacy aliases (backward compatibility)
# ──────────────────────────────────────────────────────────────────────

class WinCorrTorch3D(WinCorrTorch):
    def __init__(self, radius: int = 3, mode: str = 'for', norm_vectors: bool = False):
        super().__init__(radius=radius, ndim=3, mode=mode, norm_vectors=norm_vectors)

class WinCorrTorch2D(WinCorrTorch):
    def __init__(self, radius: int = 3, mode: str = 'for', norm_vectors: bool = False):
        super().__init__(radius=radius, ndim=2, mode=mode, norm_vectors=norm_vectors)

class GlobalCorrTorch3D(GlobalCorrTorch):
    def __init__(self, norm_vectors: bool = False):
        super().__init__(ndim=3, norm_vectors=norm_vectors)

class GlobalCorrTorch2D(GlobalCorrTorch):
    def __init__(self, norm_vectors: bool = False):
        super().__init__(ndim=2, norm_vectors=norm_vectors)
