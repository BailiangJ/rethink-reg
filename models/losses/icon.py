from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CFG, LOSSES
from ..utils.warp import Warp_off_grid
from .flow_loss import FlowLoss
from .inverse_consistency import InverseConsistentLoss


@LOSSES.register_module()
class ICONLoss(InverseConsistentLoss):
    def __init__(self,
                 flow_loss_cfg: CFG,
                 image_size: Sequence[int] = (160, 192, 224),
                 interp_mode: str = 'bilinear',
                 ):
        """
        Compute the inverse consistency loss of forward and backward flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__(flow_loss_cfg, image_size, interp_mode)
        self.warp_off = Warp_off_grid(self.image_size, self.interp_mode)

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

        # Gaussian noise for off-grid sampling
        epsilon = torch.randn_like(forward_flow) * (1.0 / self.image_size[-1])

        # off_grid forward_flow
        # phi_AB(I + epsilon)
        fwd_flow_eps = self.warp(forward_flow, epsilon)
        # off_grid backward_flow
        # phi_BA(I + epsilon)
        bck_flow_eps = self.warp(backward_flow, epsilon)

        # off_grid backward_flow in TARGET space
        # phi_BA(phi_AB(I + epsilon)+I+epsilon)
        bck_flow_eps_ = self.warp_off(backward_flow, fwd_flow_eps, epsilon)
        # off_grid forward_flow in SOURCE space
        # phi_AB(phi_BA(I + epsilon)+I+epsilon)
        fwd_flow_eps_ = self.warp_off(forward_flow, bck_flow_eps, epsilon)

        zero_flow = torch.zeros_like(forward_flow)

        # fwd_flow_eps + bck_flow_eps_ = 0
        # phi_AB(I + epsilon) + phi_BA(phi_AB(I + epsilon)+I+epsilon) + (I+epsilon) - (I+epsilon) = 0
        # bck_flow_eps + fwd_flow_eps_ = 0
        # phi_BA(I + epsilon) + phi_AB(phi_BA(I + epsilon)+I+epsilon) + (I+epsilon) - (I+epsilon) = 0
        loss = (self.flow_loss(fwd_flow_eps + bck_flow_eps_, zero_flow,
                               target_fg, val) +
                self.flow_loss(bck_flow_eps + fwd_flow_eps_, zero_flow,
                               source_fg, val))

        return loss#, fwd_flow_eps, bck_flow_eps, fwd_flow_eps_, bck_flow_eps_

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode})')
        return repr_str


@LOSSES.register_module()
class GradICONLoss(ICONLoss):
    def __init__(self,
                 flow_loss_cfg: CFG,
                 image_size: Sequence[int] = (160, 192, 224),
                 interp_mode: str = 'bilinear',
                 delta: float = 0.001,
                 ):
        """
        Compute the inverse consistency loss of forward and backward flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__(flow_loss_cfg, image_size, interp_mode)
        self.ndim = len(image_size)
        self.delta = delta

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

        # Gaussian noise for off-grid sampling
        epsilon = torch.randn_like(forward_flow) * (1.0 / self.image_size[-1])

        # off_grid forward_flow
        # phi_AB(I + epsilon)
        fwd_flow_eps = self.warp(forward_flow, epsilon)
        # off_grid backward_flow
        # phi_BA(I + epsilon)
        bck_flow_eps = self.warp(backward_flow, epsilon)

        # off_grid backward_flow in TARGET space
        # phi_BA(phi_AB(I + epsilon)+I+epsilon)
        bck_flow_eps_ = self.warp_off(backward_flow, fwd_flow_eps, epsilon)
        # off_grid forward_flow in SOURCE space
        # phi_AB(phi_BA(I + epsilon)+I+epsilon)
        fwd_flow_eps_ = self.warp_off(forward_flow, bck_flow_eps, epsilon)

        # inverse consistency error in TARGET space
        # fwd_flow_eps + bck_flow_eps_ = 0
        # phi_AB(I + epsilon) + phi_BA(phi_AB(I + epsilon)+I+epsilon) = 0
        tgt_ic_err = fwd_flow_eps + bck_flow_eps_
        # inverse consistency error in SOURCE space
        src_ic_err = bck_flow_eps + fwd_flow_eps_

        loss = 0.0

        for i in range(self.ndim):
            d = torch.zeros([1] + [self.ndim] + [1] * self.ndim)
            d[:, i, ...] = self.delta

            fwd_flow_eps_d = self.warp(forward_flow, epsilon + d)
            bck_flow_eps_d = self.warp(backward_flow, epsilon + d)

            bck_flow_eps_d_ = self.warp_off(backward_flow, fwd_flow_eps_d, epsilon + d)
            fwd_flow_eps_d_ = self.warp_off(forward_flow, bck_flow_eps_d, epsilon + d)

            # inverse consistency error (with delta) in TARGET space
            tgt_ic_err_d = fwd_flow_eps_d + bck_flow_eps_d_
            # inverse consistency error (with delta) in SOURCE space
            src_ic_err_d = bck_flow_eps_d + fwd_flow_eps_d_

            tgt_gradicon_err = (tgt_ic_err - tgt_ic_err_d) / self.delta
            src_gradicon_err = (src_ic_err - src_ic_err_d) / self.delta

            loss += torch.mean(tgt_gradicon_err ** 2) + torch.mean(src_gradicon_err ** 2)

        return loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode}, '
                     f'delta={self.delta})')
