"""Rigid pre-alignment of one-hot segmentations, by gradient descent.

`get_closest_rigid` parameterises a rigid transform by three Euler angles and a
translation, initialises the translation from the mass centres of the two label
maps, and minimises the squared difference between the resampled source and the
target with Adam. It is only reachable through `RigidPreAlign`; the paper's
AbdomenMRCT results use translation-only pre-alignment (see prealign.py).
"""

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def get_reference_grid(image_size: Sequence[int]) -> torch.Tensor:
    """
    Generate a unnormalized coordinate grid
    Args:
        image_size: shape of input image, e.g. (64,128,128)
    Return:
        grid: torch.Tensor, shape (spatial_dims, ...)
    """
    mesh_points = [torch.arange(0, dim) for dim in image_size]
    grid = torch.stack(torch.meshgrid(*mesh_points, indexing='ij'),
                       dim=0).to(dtype=torch.float)  # (spatial_dims, ...)
    return grid


def get_mass_center(label_map: torch.Tensor, grid: torch.Tensor,
                    dim: int) -> torch.Tensor:
    """
    Get the mass center of one-hot mask
    Args:
        label_map: tensor of shape N x D x H x W, one channel per label
        grid: tensor of shape 3 x D x H x W, reference grid of image
        dim: int, number of dimensions, 3 in our usage
    Returns:
        center_mass: tensor of shape (3, N)

    """
    intensity_sum = torch.sum(label_map, dim=list(range(1, dim + 1)))
    # center_mass_i shape N
    center_mass_x = torch.sum(label_map * grid[0, ...],
                              dim=list(range(1, dim + 1))) / intensity_sum
    center_mass_y = torch.sum(label_map * grid[1, ...],
                              dim=list(range(1, dim + 1))) / intensity_sum
    center_mass_z = torch.sum(label_map * grid[2, ...],
                              dim=list(range(1, dim + 1))) / intensity_sum
    center_mass = torch.stack([center_mass_x, center_mass_y, center_mass_z],
                              dim=0)
    return center_mass


class RigidTransformation(nn.Module):
    """Rigid transformation for 3D rigid registration.

    Args:
        moving_oh (torch.Tensor): one-hot encoded segmentation tensor of shape BNHWD, with B=1
    """
    def __init__(self,
                 moving_oh: torch.Tensor,
                 dtype=torch.float32,
                 device: str = 'cpu') -> None:
        super().__init__()
        moving_oh = moving_oh.squeeze(0)  # Remove batch dimension
        self._image_size = moving_oh.shape[1:]  # Spatial dimensions
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        self.register_buffer('grid', grid)

        # Compute center of mass for each organ
        self.center_mass_x, self.center_mass_y, self.center_mass_z = get_mass_center(
            moving_oh, self.grid, self._dim)

        # Single set of parameters for all organs
        self.phi_x = Parameter(torch.tensor([0.0]))
        self.phi_y = Parameter(torch.tensor([0.0]))
        self.phi_z = Parameter(torch.tensor([0.0]))

        self.t_x = Parameter(torch.tensor([0.0]))
        self.t_y = Parameter(torch.tensor([0.0]))
        self.t_z = Parameter(torch.tensor([0.0]))

    def init_translation(self, fixed_oh: torch.Tensor):
        """
        Initialize translation using mass centers of one-hot segmentations.

        Args:
            fixed_oh: Fixed one-hot segmentation tensor (B, C, D, H, W)
        """
        fixed_oh = fixed_oh.squeeze(0)  # Remove batch dimension
        assert fixed_oh.shape[1:] == self._image_size

        # Compute mass centers for both segmentations
        fixed_cm_x, fixed_cm_y, fixed_cm_z = get_mass_center(fixed_oh, self.grid, self._dim)

        # Compute translation (moving_cm - fixed_cm)
        self.t_x = Parameter(self.center_mass_x.mean() - fixed_cm_x.mean())
        self.t_y = Parameter(self.center_mass_y.mean() - fixed_cm_y.mean())
        self.t_z = Parameter(self.center_mass_z.mean() - fixed_cm_z.mean())

    def _compute_transformation_3d(self):
        self.trans_matrix_pos = torch.diag(
            torch.ones(self._dim + 1, dtype=self._dtype,
                       device=self._device))
        rotation_matrix = torch.zeros(self._dim + 1,
                                      self._dim + 1,
                                      dtype=self._dtype,
                                      device=self._device)
        rotation_matrix[-1, -1] = 1
        self.rotation_matrix = rotation_matrix.to(
            dtype=self._dtype, device=self._device)

        self.trans_matrix_pos[0, 3] = self.t_x
        self.trans_matrix_pos[1, 3] = self.t_y
        self.trans_matrix_pos[2, 3] = self.t_z

        R_x = torch.diag(
            torch.ones(self._dim + 1, dtype=self._dtype,
                       device=self._device))
        R_x[1, 1] = torch.cos(self.phi_x)
        R_x[1, 2] = -torch.sin(self.phi_x)
        R_x[2, 1] = torch.sin(self.phi_x)
        R_x[2, 2] = torch.cos(self.phi_x)

        R_y = torch.diag(
            torch.ones(self._dim + 1, dtype=self._dtype,
                       device=self._device))
        R_y[0, 0] = torch.cos(self.phi_y)
        R_y[0, 2] = torch.sin(self.phi_y)
        R_y[2, 0] = -torch.sin(self.phi_y)
        R_y[2, 2] = torch.cos(self.phi_y)

        R_z = torch.diag(
            torch.ones(self._dim + 1, dtype=self._dtype,
                       device=self._device))
        R_z[0, 0] = torch.cos(self.phi_z)
        R_z[0, 1] = -torch.sin(self.phi_z)
        R_z[1, 0] = torch.sin(self.phi_z)
        R_z[1, 1] = torch.cos(self.phi_z)

        self.rotation_matrix = torch.einsum(
            'ij,jk->ik', torch.einsum('ij,jk->ik', R_z, R_y), R_x)

    def _compute_transformation_matrix(self):
        transformation_matrix = torch.einsum(
            'ij,jk->ik', self.trans_matrix_pos,
            self.rotation_matrix)[0:self._dim, :]
        return transformation_matrix

    def _compute_dense_flow(self, transformation_matrix, return_orig=False):
        # (3, HWD)
        flow = torch.einsum('qijk,pq->pijk', self.grid,
                            transformation_matrix.reshape(3, 4))
        flow = flow.unsqueeze(0)
        if not return_orig:
            # normalize flow values to [-1, 1] for grid_sample
            for i in range(self._dim):
                flow[:, i, ...] = 2 * (flow[:, i, ...] /
                                       (self._image_size[i] - 1) - 0.5)

            # [X, Y, Z, [x,y,z]]
            flow = flow.permute([0] + list(range(2, 2 + self._dim)) + [1])
            index_ordering: List[int] = list(range(self._dim - 1, -1, -1))
            flow = flow[..., index_ordering]  # x,y,z -> z,y,x
        else:
            flow = flow - self.grid[None, :3, ...]
        return flow

    @property
    def transformation_matrix(self):
        self._compute_transformation_3d()
        return self._compute_transformation_matrix()

    @property
    def dense_flow(self):
        """Normalized sampling grid, ready for F.grid_sample."""
        self._compute_transformation_3d()
        transformation_matrix = self._compute_transformation_matrix()
        return self._compute_dense_flow(transformation_matrix,
                                        return_orig=False)

    @property
    def orig_flow(self):
        """Unnormalized displacement field, in voxels, ready for Warp."""
        self._compute_transformation_3d()
        transformation_matrix = self._compute_transformation_matrix()
        return self._compute_dense_flow(transformation_matrix,
                                        return_orig=True)


def get_closest_rigid(source_oh: torch.Tensor,
                      target_oh: torch.Tensor,
                      lr: float = 1e-2,
                      num_iteration: int = 50,
                      dtype=torch.float32):
    """
    Rigid registration of one-hot encoded segmentations
    Args:
        source_oh: one-hot format, the shape should be BNHW[D], the background is excluded
        target_oh: one-hot format, the shape should be BNHW[D], the background is excluded
        lr: learning rate of Adam optimizer
        num_iteration: number of iterations in the rigid registration
    Return:
        resample_source: (soft) one-hot format, the shape should be BNHW[D]
        orig_flow: displacement field in voxels, applicable to the source image
    """
    try:
        device = torch.device('cuda', source_oh.get_device())
    except RuntimeError:
        device = 'cpu'

    # Initialize rigid transformation with source segmentation
    rigid_transform = RigidTransformation(source_oh,
                                          dtype=dtype,
                                          device=device)

    # Initialize translation using centroids
    rigid_transform.init_translation(target_oh)

    optimizer = torch.optim.Adam(rigid_transform.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)

    for i in range(num_iteration):
        optimizer.zero_grad()
        flow = rigid_transform.dense_flow
        resample_source = F.grid_sample(source_oh,
                                        grid=flow,
                                        mode='bilinear',
                                        align_corners=True)

        loss = torch.sum((target_oh - resample_source)**2)

        loss.backward()
        optimizer.step()
        scheduler.step()

    flow = rigid_transform.dense_flow
    # since we are using bilinear resampling in the model
    # we also use bilinear resampling here
    resample_source = F.grid_sample(source_oh,
                                    grid=flow,
                                    mode='bilinear',
                                    align_corners=True)

    for param in rigid_transform.parameters():
        param.requires_grad = False

    return resample_source.detach(), rigid_transform.orig_flow.detach()
