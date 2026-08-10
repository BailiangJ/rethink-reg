from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CFG, LOSSES
from ..utils.warp import Warp
from .flow_loss import FlowLoss

@LOSSES.register_module()
class GroupConsistencyLoss(nn.Module):
    def __init__(self,
                 flow_loss_cfg: CFG,
                 image_size: Sequence[int] = (160, 192, 224),
                 interp_mode: str = 'bilinear',
                 compose_detach: bool = False,
                 **kwargs,
                 ):
        """
        Compute the groupwise flow consistency loss of triplet flows
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
        flow_12:torch.Tensor,
        flow_23:torch.Tensor,
        flow_13:torch.Tensor,
        source:Optional[torch.Tensor]=None,
        target:Optional[torch.Tensor]=None,
        compute_sim_loss:Optional[nn.Module]=None,
        fg_mask: Optional[torch.Tensor]=None,
    )->torch.Tensor:
        """
        Args:
            flow_12: in Scan2 space, mapping from Scan2 space to Scan1 space. Tensor of shape [B3HWD].
            flow_23: in Scan3 space, mapping from Scan3 space to Scan2 space. Tensor of shape [B3HWD].
            flow_13: in Scan3 space, mapping from Scan3 space to Scan1 space. Tensor of shape [B3HWD].
            source: Tensor of shape [BCHWD]. Source image.
            target: Tensor of shape [BCHWD]. Target image.
            compute_sim_loss: nn.Module to compute similarity loss.
            fg_mask: None|Tensor of shape [B1HWD]. Foreground binary mask.
        """
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
        flow_loss = self.flow_loss(flow_13_, flow_13.detach(), fg_mask, val=False)

        return sim_loss, flow_loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode}), '
                     f'compose_detach={self.compose_detach})')
        return repr_str


# @LOSSES.register_module()
# class GroupConsistencyLoss(nn.Module):
#     def __init__(self,
#                  flow_loss_cfg: CFG,
#                  image_size: Sequence[int] = (160, 192, 224),
#                  interp_mode: str = 'bilinear',
#                  compose_detach: bool = False,
#                  **kwargs,
#                  ):
#         """
#         Compute the groupwise flow consistency loss of triplet flows
#         Args:
#             image_size (Sequence[int]): shape of input flow field.
#         """
#         super().__init__()
#         flow_loss_cfg.pop('type', None)
#         self.flow_loss = FlowLoss(**flow_loss_cfg)
#         self.image_size = image_size
#         self.interp_mode = interp_mode
#         self.warp = Warp(self.image_size, self.interp_mode)
#         self.compose_detach = compose_detach
#         self.auto_detach = lambda x: x.detach() if self.compose_detach else x

#     def forward(
#         self,
#         flow_12:torch.Tensor,
#         flow_23:torch.Tensor,
#         flow_31:torch.Tensor,
#         source:Optional[torch.Tensor]=None,
#         target:Optional[torch.Tensor]=None,
#         compute_sim_loss:Optional[nn.Module]=None,
#         fg_mask: Optional[torch.Tensor]=None,
#     )->torch.Tensor:
#         """
#         Args:
#             flow_12: in Scan2 space, mapping from Scan2 space to Scan1 space. Tensor of shape [B3HWD].
#             flow_23: in Scan3 space, mapping from Scan3 space to Scan2 space. Tensor of shape [B3HWD].
#             flow_31: in Scan3 space, mapping from Scan1 space to Scan3 space. Tensor of shape [B3HWD].
#             source: Tensor of shape [BCHWD]. Source image.
#             target: Tensor of shape [BCHWD]. Target image.
#             compute_sim_loss: nn.Module to compute similarity loss.
#             fg_mask: None|Tensor of shape [B1HWD]. Foreground binary mask.
#         """
#         # Composition of flow_12 and flow_23
#         # flow_12 in Scan3 space
#         flow_12_ = self.warp(flow_12, self.auto_detach(flow_23))
#         flow_13_ = flow_12_ + flow_23

#         if compute_sim_loss is None:
#             sim_loss = None
#         else:
#             # compute similarity loss
#             y_source_ = self.warp(source, flow_13_)
#             with torch.autocast(device_type='cuda' if source.is_cuda else 'cpu', dtype=torch.float16, enabled=False):
#                 sim_loss = compute_sim_loss(y_source_, target)

#         # Composition of flow_13_ and flow_31 should be zero displacement
#         flow_11_ = self.warp(flow_13_, self.auto_detach(flow_31)) + flow_31
#         flow_loss = self.flow_loss(flow_11_, torch.zeros_like(flow_11_), fg_mask, val=False)

#         return sim_loss, flow_loss

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += (f'(flow_loss={self.flow_loss}, '
#                      f'image_size={self.image_size}, '
#                      f'interp_mode={self.interp_mode}), '
#                      f'compose_detach={self.compose_detach})')
#         return repr_str
