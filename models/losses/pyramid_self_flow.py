from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CFG, LOSSES
from ..utils.resize_flow import ResizeFlow
from .flow_loss import FlowLoss


@LOSSES.register_module()
class PyramidalDistillLoss(nn.Module):
    '''
    Self-supervised flow distillation loss for pyramidal flows output.
    '''

    def __init__(self,
                 flow_loss_cfg: CFG,
                 ndim: int = 3):
        super().__init__()
        flow_loss_cfg.pop('type', None)
        self.flow_loss = FlowLoss(**flow_loss_cfg)
        self.resize_flow = ResizeFlow(0.5,
                                      0.5,
                                      ndim=ndim)

    def forward(self, flows: Sequence[torch.Tensor], target_fgs: Sequence[torch.Tensor]) -> torch.Tensor:
        '''
        Args:
            flows: List of torch.tensor of shape [B,ndim,H,W,D]. Pyramidal flow fields from scale (1/16, 1/8, 1/4, 1/2).
               target_fgs: List of torch.tensor of shape [B,1,H,W,D]. Target foreground masks from scale (1/16, 1/8, 1/4).
        '''
        loss = 0.0
        hr_flow = flows[-1]
        for i in range(len(flows) - 2, -1, -1):
            lr_flow = flows[i]
            hr_flow = self.resize_flow(hr_flow)
            loss += self.flow_loss(lr_flow, hr_flow.detach(), target_fgs[i])
        # for i in range(len(flows) - 1):
        #     lr_flow = flows[i]
        #     hr_flow = self.resize_flow(flows[i + 1])
        #     loss += self.flow_loss(lr_flow, hr_flow.detach(), target_fgs[i])
        return loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss},'
                     f'resize_flow={self.resize_flow})')
        return repr_str
