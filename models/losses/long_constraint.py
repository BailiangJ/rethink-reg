from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CFG, LOSSES
from ..utils.warp import Warp
from .flow_loss import FlowLoss


@LOSSES.register_module()
class LongitudinalConsistentLoss(nn.Module):
    def __init__(self,
                 flow_loss_cfg: CFG,
                 image_size: Sequence[int] = (160, 192, 224),
                 interp_mode: str = 'bilinear',
                 compose_detach: bool = False,
                 ):
        """
        Compute the longitudinal consistency loss of triplet flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__()
        flow_loss_cfg.pop('type', None)
        self.flow_loss = FlowLoss(**flow_loss_cfg)
        self.image_size = image_size
        self.interp_mode = interp_mode
        self.warp = Warp(self.image_size, self.interp_mode)
        self.compose_detach = compose_detach
        self.auto_detach = lambda x: x.detach() if self.compose_detach else x

    def forward(
        self,
        triplet_flow: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        compute_sim_loss: Optional[nn.Module] = None,
        fg_mask: Optional[torch.Tensor] = None,
        val: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            triplet_flow: Tensor of shape [B3HWD], B=3. It contains:
                flow_12: in Scan2 space, mapping from Scan2 space to Scan1 space. Tensor of shape [13HWD].
                flow_23: in Scan3 space, mapping from Scan3 space to Scan2 space. Tensor of shape [13HWD].
                flow_13: in Scan3 space, mapping from Scan3 space to Scan1 space. Tensor of shape [13HWD].
            source: Tensor of shape [BCHWD], B=1. Source image.
            target: Tensor of shape [BCHWD], B=1. Target image.
            compute_sim_loss: nn.Module to compute similarity loss.
            fg_mask: None|Tensor of shape [BHWD]. Foreground binary mask.
        """
        flow_12 = triplet_flow[[0]]
        flow_23 = triplet_flow[[1]]
        flow_13 = triplet_flow[[2]]

        # Composition of flow_12 and flow_23
        # flow_12 in Scan3 space
        flow_12_ = self.warp(flow_12, self.auto_detach(flow_23))
        flow_13_ = flow_12_ + flow_23

        if compute_sim_loss is None:
            sim_loss = None
        else:
            # compute similarity loss
            y_source_ = self.warp(source, flow_13_)
            with torch.autocast(device_type='cuda' if source.is_cuda else 'cpu', dtype=torch.float16, enabled=False):
                sim_loss = compute_sim_loss(y_source_, target)


        # flow_12_ + flow_23 in Scan3 space should equal to flow_13
        flow_loss = self.flow_loss(flow_13_, flow_13, fg_mask, val)

        return sim_loss, flow_loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode}), ' 
                     f'compose_detach={self.compose_detach})')
        return repr_str
