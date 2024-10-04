from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CFG, LOSSES
from ..utils.warp import Warp
from .flow_loss import FlowLoss


@LOSSES.register_module()
class InverseConsistentLoss(nn.Module):
    def __init__(
        self,
        flow_loss_cfg: CFG,
        image_size: Sequence[int] = (160, 192, 224),
        interp_mode: str = 'bilinear',
    ):
        """
        Compute the inverse consistency loss of forward and backward flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__()
        flow_loss_cfg.pop('type', None)
        self.flow_loss = FlowLoss(**flow_loss_cfg)
        self.image_size = image_size
        self.interp_mode = interp_mode
        self.warp = Warp(self.image_size, self.interp_mode)

    def forward(
        self,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        target_fg: Optional[torch.Tensor] = None,
        source_fg: Optional[torch.Tensor] = None,
        val: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            forward_flow: in TARGET space, mapping from TARGET space to SOURCE space. Tensor of shape [B3HWD].
            backward_flow: in SOURCE spacce, mapping from SOURCE space to TARGET space. Tensor of shape [B3HWD].
        """

        # backward_flow in TARGET space
        backward_flow_ = self.warp(backward_flow, forward_flow)
        # forward_flow in SOURCE space
        forward_flow_ = self.warp(forward_flow, backward_flow)

        zero_flow = torch.zeros_like(forward_flow)

        # forward_flow + backward_flow_ = 0
        # backward_flow + forward_flow_ = 0
        loss = (self.flow_loss(forward_flow + backward_flow_, zero_flow,
                               target_fg, val) +
                self.flow_loss(backward_flow + forward_flow_, zero_flow,
                               source_fg, val))

        return loss, forward_flow_, backward_flow_

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode})')
        return repr_str
