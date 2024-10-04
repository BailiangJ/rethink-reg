import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, List
from ..builder import METRICS


@METRICS.register_module('tre')
class TargetRegistrationError:
    def __init__(self,
                 image_size: Sequence[int],
                 spacing: Sequence[float],
                 interp_mode='bilinear') -> None:
        self.ndim = len(image_size)
        self.image_size = image_size
        self.spacing = torch.tensor(spacing).view(1, 1, -1)
        self.spacing.type(torch.FloatTensor)
        self.interp_mode = interp_mode

    def sample_displacement_flow(self,
                                 sample_grid: torch.Tensor,
                                 flow: torch.Tensor,
                                 mode: str = 'bilinear') -> torch.Tensor:
        """Sample 3D displacement flow at certain locations.
        TODO: adapt it to be compatiable with 2D images
        reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        Args:
            sample_grid: torch.Tensor, shape (B, num_samples, 3), coordinate of sampling locations
            flow: torch.Tensor, shape (B,3,HWD), dense displacement field
            image_size: shape of input image
            mode: str, interpolation mode, ['nearest', 'bilinear', 'bicubic']
        Returns:
            sample_flow: torch.Tensor, sampled displacement vectors
        """
        # normalize
        # F.grid_sample takes normalized grid with range at [-1,1]
        for i, dim in enumerate(self.image_size):
            sample_grid[..., i] = sample_grid[..., i] * 2 / (dim - 1) - 1

        index_ordering: List[int] = list(range(self.ndim - 1, -1, -1))
        # F.grid_sample takes grid in a different order
        sample_grid = sample_grid[..., index_ordering]  # x,y,z -> z,y,x
        # reshape to (1,1,B,num_samples,3) for grid_sample
        sample_grid = sample_grid[None, None, ...]

        sample_flow = F.grid_sample(flow,
                                    sample_grid,
                                    mode=mode,
                                    padding_mode='zeros',
                                    align_corners=True)
        return sample_flow

    def __call__(self,
                 flow: torch.Tensor,
                 fixed_keypnts: torch.Tensor,
                 moving_keypnts: torch.Tensor) -> torch.Tensor:
        """Compute the target registration error (TRE) between fixed and moved key points.
        Args:
            flow: torch.Tensor, shape (B,3,H,W,D), dense displacement field
            fixed_keypnts: torch.Tensor, shape (B,N,3), fixed key points
            moving_keypnts: torch.Tensor, shape (B,N,3), moving key points
        """
        assert list(self.image_size) == list(flow.shape[2:])
        assert fixed_keypnts.shape == moving_keypnts.shape

        moving_keypnts_flow = self.sample_displacement_flow(moving_keypnts, flow, self.interp_mode)
        moved_keypnts = moving_keypnts + moving_keypnts_flow.squeeze((0, 1))

        tre = torch.linalg.norm((moved_keypnts - fixed_keypnts) * self.spacing, dim=-1)
        print(tre.shape)
        return tre
