from __future__ import annotations

from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import win_corr_cuda


def ref_win_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    radius: int,
    ndim: int,
    norm_vectors: bool = False,
    scale: bool = True,
    padding: str = "constant",
) -> torch.Tensor:
    if norm_vectors:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
    if padding not in {"constant", "replicate"}:
        raise ValueError(f"unsupported padding: {padding}")
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    spatial = x.shape[2:]
    y_padded = F.pad(y, [radius] * (ndim * 2), mode=padding)
    win_size = 2 * radius + 1
    corr = torch.cat(
        [
            (x * y_padded[(..., *[slice(o, o + s) for o, s in zip(off, spatial)])]).sum(dim=1, keepdim=True)
            for off in product(range(win_size), repeat=ndim)
        ],
        dim=1,
    )
    return corr * (x.shape[1] ** -0.5) if scale else corr


class RefWinCorr(nn.Module):
    def __init__(
        self,
        radius: int = 3,
        ndim: int = 3,
        norm_vectors: bool = False,
        scale: bool = True,
        padding: str = "constant",
    ) -> None:
        super().__init__()
        self.radius = radius
        self.ndim = ndim
        self.norm_vectors = norm_vectors
        self.scale = scale
        self.padding = padding

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ref_win_corr(x, y, self.radius, self.ndim, self.norm_vectors, self.scale, self.padding)


class WinCorrCuda(nn.Module):
    def __init__(
        self,
        radius: int = 3,
        ndim: int = 3,
        norm_vectors: bool = False,
        scale: bool = True,
        padding: str = "constant",
        auto_contiguous: bool = False,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.ndim = ndim
        self.norm_vectors = norm_vectors
        self.scale = scale
        self.padding = padding
        self.auto_contiguous = auto_contiguous

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return win_corr_cuda(
            x,
            y,
            radius=self.radius,
            ndim=self.ndim,
            norm_vectors=self.norm_vectors,
            scale=self.scale,
            padding=self.padding,
            auto_contiguous=self.auto_contiguous,
        )
