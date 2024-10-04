from typing import List, Sequence

import torch
from torch import nn

from .warp import Warp


class Composite(nn.Module):
    """Composite two flow fields.

    Args:
        image_size (Sequence[int]): size of input image.
        interp_mode (str): interpolation mode. Options are: ["nearest", "bilinear", "bicubic"]
    """
    def __init__(self,
                 image_size: Sequence[int],
                 interp_mode: str = 'bilinear') -> None:
        super().__init__()
        self.warp = Warp(image_size, interp_mode)

    def forward(
        self,
        flow_12: torch.Tensor,
        flow_23: torch.Tensor,
    ):
        """
        Args:
            flow_12 (torch.Tensor): flow field mapping from image space 2 to image space 1, tensor of shape [B3HWD].
            flow_23 (torch.Tensor): flow field mapping from image space 3 to image space 2, tensor of shape [B3HWD].
        """

        return (self.warp(flow_12, flow_23) + flow_23)
