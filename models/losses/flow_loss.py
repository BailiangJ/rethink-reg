from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CFG, LOSSES


def charbonnier_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     alpha: float = 0.45,
                     eps: float = 0.001,
                     truncate: Optional[float] = None) -> torch.Tensor:
    """Compute the generalized Charbonnier loss between the predicted flow and the
    ground truth flow.

    Args:
        pred (torch.Tensor): The predicted flow. Tensor of shape [B3HWD].
        target (torch.Tensor): The ground truth flow. Tensor of shape [B3HWD].
    Returns:
        loss (torch.Tensor): The Charbonnier loss between the predicted flow and the
            ground truth flow. Tensor of shape [B1HWD].
    """
    diff = torch.add(pred, -target)
    loss = (torch.sum(diff**2, dim=1, keepdim=True) + eps)**alpha  # [B1HWD]
    if truncate is not None:
        loss = torch.minimum(loss, truncate)
    return loss


def charbonnier_penalty(x: torch.Tensor,
                        alpha: float = 0.45,
                        beta=1.0,
                        epsilon=0.001,
                        truncate: Optional[float] = None):
    """Compute the generalized Charbonnier loss of the difference tensor x.

    Ref: Sun D, Roth S, Black MJ. A quantitative analysis of current practices in optical flow estimation
    and the principles behind them. International Journal of Computer Vision. 2014 Jan;106:115-37.
    Args:
        x:
        alpha:
        beta:
        epsilon:
        truncate:
    """
    error = torch.sum(x**2, dim=1)
    error = torch.pow((beta**2) * error + epsilon**2, alpha)
    # error = torch.pow(beta * (x ** 2) + epsilon ** 2, alpha)
    if truncate is not None:
        error = torch.minimum(error, truncate)
    return error


@LOSSES.register_module()
@LOSSES.register_module('IntensityLoss')
class FlowLoss(nn.Module):
    """Compute the flow loss between the predicted flow and the ground truth flow.

    Args:
        penalty (str): The penalty norm to use. Options: ['l1', 'l2', 'rmse', 'charbonnier'].
        ch_cfg (CFG): The config for the Charbonnier penalty.
    """
    def __init__(self, penalty: str = 'l2', ch_cfg: Optional[CFG] = None):
        super().__init__()
        self.penalty = penalty
        self.ch_cfg = {} if ch_cfg is None else ch_cfg

    def forward(self,
                pred_flow: torch.Tensor,
                gt_flow: torch.Tensor,
                fg_mask: Optional[torch.Tensor] = None,
                val: bool=False) -> torch.Tensor:
        """
        Args:
            pred_flow (torch.Tensor): The predicted flow. Tensor of shape [B3HWD].
            gt_flow (torch.Tensor): The ground truth flow. Tensor of shape [B3HWD].
            fg_mask (torch.Tensor): The foreground mask in target space. Tensor of shape [B1HWD].
            val (bool): If True, keep the batch dimension of the computed loss.
        """
        if self.penalty == 'l1':
            dist = torch.sum(torch.abs(pred_flow - gt_flow),
                             dim=1,
                             keepdim=True)
        elif self.penalty == 'l2':
            dist = torch.sum((pred_flow - gt_flow)**2, dim=1, keepdim=True)
        elif self.penalty == 'rmse':
            dist = torch.linalg.norm((pred_flow - gt_flow),
                                     dim=1,
                                     keepdim=True)
        elif self.penalty == 'charbonnier':
            dist = charbonnier_loss(pred_flow, gt_flow, **self.ch_cfg)
        else:
            raise ValueError(
                f'Unsupported norm: {self.penalty}, available options are ["l1","l2", "rmse", "charbonnier"].'
            )

        # dist: (B1HWD)
        # fg_mask: (B1HWD)
        if fg_mask is not None:
            if dist.shape[-3:] != fg_mask.shape[:-3]:
                output_size = dist.shape[-3:]
                fg_mask = F.interpolate(fg_mask,
                                        align_corners=True,
                                        size=output_size,
                                        mode='trilinear')

            if dist.shape[0] != fg_mask.shape[0]:
                fg_mask = fg_mask.repeat(dist.shape[0], 1, 1, 1, 1)

            assert dist.shape == fg_mask.shape

            if not val:
                loss = torch.sum(dist * fg_mask) / torch.sum(fg_mask)
            else:
                loss = (dist*fg_mask).sum(dim=(2,3,4)) / fg_mask.sum(dim=(2,3,4))
        else:
            if not val:
                loss = torch.mean(dist)
            else:
                loss = dist.mean(dim=(2,3,4))

        return loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(penalty=\'{self.penalty}\',' f'ch_cfg={self.ch_cfg})')
        return repr_str
