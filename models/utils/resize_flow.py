from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ResizeFlow(nn.Module):
    """Resize and rescale a flow field.

    Args:
        spatial_scale (float): scaling factor of spatial resizing.
        flow_scale (float): scaling factor of flow components.
        ndim (int): number of dimensions.
    """
    def __init__(self, spatial_scale: float, flow_scale: float, ndim: int):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.flow_scale = flow_scale
        if ndim == 2:
            self.interp_mode = 'bilinear'
        elif ndim == 3:
            self.interp_mode = 'trilinear'
        else:
            raise KeyError(f'Unsupported ndim for ResizeFlow:{ndim}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flow_scale < 1:
            # resize first to save memory
            x = F.interpolate(x,
                              align_corners=True,
                              scale_factor=self.spatial_scale,
                              recompute_scale_factor=True,
                              mode=self.interp_mode)
            x = self.flow_scale * x

        elif self.flow_scale > 1:
            # multiply first to save memory
            x = self.flow_scale * x
            x = F.interpolate(x,
                              align_corners=True,
                              scale_factor=self.spatial_scale,
                              recompute_scale_factor=True,
                              mode=self.interp_mode)
        else:  # self.flow_scale = 1
            x = F.interpolate(x,
                              align_corners=True,
                              scale_factor=self.spatial_scale,
                              recompute_scale_factor=True,
                              mode=self.interp_mode)

        return x

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(spatial_scale={self.spatial_scale}, '
                     f'flow_scale={self.flow_scale})')
        return repr_str
