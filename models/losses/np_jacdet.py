from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .diffusion_regularizer import spatial_gradient


@LOSSES.register_module('np_jacdet')
class NonPositiveJacDetLoss(nn.Module):
    """Regularization loss that penalizes non-positive Jacobian determinants
        in deformation fields to enforce topology preservation."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_jacobian_determinant(disp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disp: the shape should be BCH(WD)
        """
        first_order_gradient = [
            spatial_gradient(disp, dim, mode='forward')
            for dim in range(2, disp.ndim)
        ]

        if disp.ndim - 2 == 2:  # 2D case
            # consistent with shape (B, H-1, W-1)
            dx_x = first_order_gradient[0][:, 0, :, :-1]
            dx_y = first_order_gradient[0][:, 1, :, :-1]

            dy_x = first_order_gradient[1][:, 0, :-1, :]
            dy_y = first_order_gradient[1][:, 1, :-1, :]

            # construct Jacobian matrix (B, 2, 2, H-1, W-1)
            jacobian = torch.stack(
                [
                    torch.stack([dx_x, dx_y], dim=1),
                    torch.stack([dy_x, dy_y], dim=1)
                ], dim=1
            )
            # add identity matrix to compute the Jacobian matrix of the deformation
            jacobian = jacobian + torch.eye(2, 2).reshape(1, 2, 2, 1, 1).to(jacobian.device)
            jacobian = jacobian[:, :, :, 2:-2, 2:-2]
            # compute the determinant of the Jacobian matrix
            jac_det = jacobian[:, 0, 0, :, :] * jacobian[:, 1, 1, :, :] - \
                      jacobian[:, 0, 1, :, :] * jacobian[:, 1, 0, :, :]

        elif disp.ndim - 2 == 3:  # 3D case
            dx_x = first_order_gradient[0][:, 0, :, :-1, :-1]
            dx_y = first_order_gradient[0][:, 1, :, :-1, :-1]
            dx_z = first_order_gradient[0][:, 2, :, :-1, :-1]

            dy_x = first_order_gradient[1][:, 0, :-1, :, :-1]
            dy_y = first_order_gradient[1][:, 1, :-1, :, :-1]
            dy_z = first_order_gradient[1][:, 2, :-1, :, :-1]

            dz_x = first_order_gradient[2][:, 0, :-1, :-1, :]
            dz_y = first_order_gradient[2][:, 1, :-1, :-1, :]
            dz_z = first_order_gradient[2][:, 2, :-1, :-1, :]

            # construct Jacobian matrix (B, 3, 3, H-1, W-1, D-1)
            jacobian = torch.stack(
                [
                    torch.stack([dx_x, dx_y, dx_z], dim=1),
                    torch.stack([dy_x, dy_y, dy_z], dim=1),
                    torch.stack([dz_x, dz_y, dz_z], dim=1)
                ], dim=1
            )
            # add identity matrix to compute the Jacobian matrix of the deformation
            jacobian = jacobian + torch.eye(3, 3).reshape(1, 3, 3, 1, 1, 1).to(jacobian.device)
            jacobian = jacobian[:, :, :, 2:-2, 2:-2, 2:-2]
            # compute the determinant of the Jacobian matrix
            jac_det = jacobian[:, 0, 0, ...] * \
                      (jacobian[:, 1, 1, ...] * jacobian[:, 2, 2, ...] -
                       jacobian[:, 1, 2, ...] * jacobian[:, 2, 1, ...]) - \
                      jacobian[:, 1, 0, ...] * \
                      (jacobian[:, 0, 1, ...] * jacobian[:, 2, 2, ...] -
                       jacobian[:, 0, 2, ...] * jacobian[:, 2, 1, ...]) + \
                      jacobian[:, 2, 0, ...] * \
                      (jacobian[:, 0, 1, ...] * jacobian[:, 1, 2, ...] -
                       jacobian[:, 0, 2, ...] * jacobian[:, 1, 1, ...])
        else:
            raise ValueError(f'Expecting 2-d, 3-d displacement field, instead got {disp.ndim - 2}')

        return jac_det

    def forward(self, pred: torch.Tensor, fg_mask: Optional[torch.Tensor] = None):
        """
        Args:
            pred: the shape should be BCH(WD)
        """
        if pred.ndim not in [3, 4, 5]:
            raise ValueError(
                f'Expecting 3-d, 4-d or 5-d pred, instead got pred of shape {pred.shape}'
            )
        for i in range(pred.ndim - 2):
            if pred.shape[-i - 1] <= 4:
                raise ValueError(
                    f'All spatial dimensions must be > 4, got spatial dimensions {pred.shape[2:]}'
                )
        if pred.shape[1] != pred.ndim - 2:
            raise ValueError(
                f'Number of vector components, {pred.shape[1]}, does not match number of spatial dimensions, {pred.ndim - 2}'
            )

        jac_det = self.compute_jacobian_determinant(pred)
        neg_jac_det = F.relu(-1.0 * jac_det)
        if fg_mask is not None:
            if pred.ndim - 2 == 2:
                fg_mask = fg_mask[:, :, 2:-3, 2:-3]
            elif pred.ndim - 2 == 3:
                fg_mask = fg_mask[:, :, 2:-3, 2:-3, 2:-3]
            else:
                pass
            return (neg_jac_det * fg_mask).sum() / fg_mask.sum()
        else:
            return neg_jac_det.mean()
