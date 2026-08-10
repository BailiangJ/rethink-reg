import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple
from .correlation import WinCorrTorch, GlobalCorrTorch

class CorrelationMatching(nn.Module):
    """Computes a dense displacement field via softmax expectation over a correlation volume.

    ``radius > 0``: window-based correlation with window size ``2*radius+1``.
    ``radius = 0``: global (all-pairs) correlation.
    """

    def __init__(self,
                 image_size: Sequence[int],
                 radius: int = 3,
                 corr_mode: str = 'for',
                 norm_vectors: bool = True,
                 scale: bool = True):
        super().__init__()
        self.image_size = image_size
        self.radius = radius
        self.ndim = len(image_size)
        self.corr_mode = corr_mode
        self.mode = 'window' if radius > 0 else 'global'
        self.win_size = 2 * radius + 1

        # 1. Initialize correlation engine
        if self.mode == 'window':
            self.corr_fn = WinCorrTorch(radius=self.radius, ndim=self.ndim, mode=self.corr_mode, norm_vectors=norm_vectors, scale=scale)
        else:
            self.corr_fn = GlobalCorrTorch(ndim=self.ndim, norm_vectors=norm_vectors, scale=scale)

        # 2. Setup Reference Grid
        ref_grid = self._get_reference_grid(self.image_size)
        self.register_buffer('ref_grid', ref_grid, persistent=False)

        # 3. Setup Layout-Specific Mask and Samples Grid
        if self.mode == 'window':
            invalid_mask, sample_grid = self._setup_window_grids(ref_grid)
            self.register_buffer('invalid_mask', invalid_mask, persistent=False)
            self.register_buffer('sample_grid', sample_grid, persistent=False)
        else:
            # For global mode, prepare a flattened view of the reference grid
            # ref_grid: (1, ndim, D, H, W) -> (1, ndim, D*H*W)
            global_grid_flat = ref_grid.view(1, self.ndim, -1)
            self.register_buffer('global_grid_flat', global_grid_flat, persistent=False)

    def _get_reference_grid(self, image_size: Sequence[int]) -> torch.Tensor:
        """Constructs unnormalized (D, H, W) reference grid."""
        mesh_points = [torch.arange(0, dim, dtype=torch.float) for dim in image_size]
        # Important: indexing='ij' ensures shapes remain explicitly (D, H, W)
        grid = torch.stack(torch.meshgrid(*mesh_points, indexing='ij'), dim=0) # (ndim, D, H, W)
        return grid.unsqueeze(0) # (1, ndim, D, H, W)

    def _setup_window_grids(self, ref_grid: torch.Tensor):
        """Constructs sample grid and invalid bounds mask for memory-efficient computation.

        Target shapes:
        - invalid_mask: (1, win**ndim, D, H, W)
        - sample_grid:  (1, ndim, win**ndim, D, H, W)
        """
        # Construct raw relative offsets -> (ndim, win**ndim)
        mesh_points = [torch.arange(-self.radius, self.radius + 1, dtype=torch.float) for _ in range(self.ndim)]
        offsets = torch.stack(torch.meshgrid(*mesh_points, indexing='ij'), dim=0) # (ndim, win_size, ..., win_size)
        offsets = offsets.reshape(self.ndim, -1) # (ndim, win**ndim)

        # Shape offsets to broadcast: (1, ndim, win**ndim, 1, 1, 1)
        # Note: ref_grid is (1, ndim, D, H, W), we unsqueeze it to (1, ndim, 1, D, H, W)
        offsets_bc = offsets.view(1, self.ndim, -1, *[1 for _ in range(self.ndim)])
        ref_grid_bc = ref_grid.unsqueeze(2)

        # Compute absolute source coordinates layout directly via broadcasting
        sample_grid = ref_grid_bc + offsets_bc # (1, ndim, win**ndim, D, H, W)

        # Mask out coordinates out of physical bounds
        invalid_mask = torch.zeros_like(sample_grid, dtype=torch.bool)
        for i, dim_size in enumerate(self.image_size):
            invalid_mask[:, i, ...] = (sample_grid[:, i, ...] < 0) | (sample_grid[:, i, ...] >= dim_size)

        # Any coordinate spanning out of bounds across its dimensional vector marks the patch as invalid
        invalid_mask = invalid_mask.any(dim=1) # (1, win**ndim, D, H, W)
        invalid_mask = invalid_mask.type(torch.float) * -1e9

        return invalid_mask, sample_grid

    def _compute_corr_logits(self, src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
        # Default matcher uses correlation operators (window/global).
        # WinCorrTorch.forward(anchor, search) expects (tgt, src) for tgt-to-src matching.
        return self.corr_fn(tgt_feat, src_feat)

    def forward(self, src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a dense displacement field via softmax expectation over a correlation volume.

        Args:
            src_feat: Source feature map, shape ``(B, C, *spatial)``.
            tgt_feat: Target feature map, shape ``(B, C, *spatial)``.

        Returns:
            flow: Displacement field (tgt → src), shape ``(B, ndim, *spatial)``.
            prob: Softmax probability map over candidates, shape ``(B, candidates, *spatial)``.
                  Temporary return value used during testing; will be removed.
        """
        corr_logits = self._compute_corr_logits(src_feat, tgt_feat)

        if self.mode == 'window':
            # corr_logits: (B, win**ndim, D, H, W)
            corr_logits = corr_logits + self.invalid_mask
            prob = F.softmax(corr_logits, dim=1) # Softmax across window dim

            # Expected correspondence: sum of spatial vectors evaluated against patch probability density
            # prob: (B, win**ndim, D, H, W) -> unsqueezed -> (B, 1, win**ndim, D, H, W)
            # sample_grid: (1, ndim, win**ndim, D, H, W)
            correspondence = torch.sum(prob.unsqueeze(1) * self.sample_grid, dim=2) # (B, ndim, D, H, W)

        elif self.mode == 'global':
            # GlobalCorrTorch.forward(anchor, search) computes 'bji' — output dim 1 indexes
            # search locations and spatial dims follow anchor's layout. For tgt-to-src matching,
            # tgt_feat is the anchor (spatial layout) and src_feat fills the candidate dimension.
            # corr_logits: (B, src_locs, *tgt_spatial)
            prob = F.softmax(corr_logits, dim=1) # Softmax across src candidates

            # Flat operations:
            # prob: (B, src_locs, D, H, W) -> view -> (B, src_locs, tgt_locs)
            prob_flat = prob.view(prob.shape[0], prob.shape[1], -1)

            # global_grid_flat: (1, ndim, src_locs)
            # Find expected coordinate over 's' (src_locs) for each 't' (tgt_locs)
            correspondence_flat = torch.einsum('bst, ks -> bkt', prob_flat, self.global_grid_flat.squeeze(0)) # (B, ndim, tgt_locs)
            correspondence = correspondence_flat.view(-1, self.ndim, *self.image_size) # (B, ndim, D, H, W)

        flow = correspondence - self.ref_grid
        return flow, prob
