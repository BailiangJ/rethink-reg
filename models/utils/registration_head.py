from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from ..builder import REGISTRATION_HEAD
from .integrate import VecIntegrate
from .resize_flow import ResizeFlow
from .warp import Warp


@REGISTRATION_HEAD.register_module()
class RegistrationHead(nn.Module):
    """Registration head for voxelmorph.

    Args:
        image_size (Sequence[int]): size of input image.
        spatial_scale (int): the spatial rescale factor of flow field.
        flow_scale (int): the flow magnitude rescale factor of flow field.
        interp_mode (str): interpolation mode. Options are: ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 spatial_scale: float = 2.0,
                 flow_scale: float = 1.0,
                 interp_mode: str = 'bilinear') -> None:
        super().__init__()
        self.warp = Warp(image_size, interp_mode)
        self.resize_flow = ResizeFlow(spatial_scale,
                                      flow_scale,
                                      ndim=len(image_size))

    def forward(
            self,
            vec_flow: torch.Tensor,
            source: Optional[torch.Tensor] = None,
            source_oh: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        """
        Args:
            vec_flow (torch.Tensor): flow field predicted by network. [B3 Hf Wf Df]
            source (torch.Tensor): source image, tensor of shape [BCHWD].
            source_oh (torch.Tensor|None): one-hot segmentation label of source image,
                tensor of shape [BCHWD]. (Default: None)
        """
        # resize flow field
        # TransMorph: align_corners=False
        # disp_flow = nn.Upsample(self.spatial_scale,
        #                         mode='trilinear',
        #                         align_corners=False)(vec_flow)
        # ResizeFlow: align_corners=True
        disp_flow = self.resize_flow(vec_flow)

        # warp source image with displacement field
        y_source = self.warp(source, disp_flow) if source is not None else None

        # warp source one-hot segmentation label with displacement field
        y_source_oh = self.warp(source_oh,
                                disp_flow) if source_oh is not None else None

        return disp_flow, y_source, y_source_oh

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(warp={self.warp}, resize_flow={self.resize_flow})'
        return repr_str


@REGISTRATION_HEAD.register_module()
class DownSizeRegistrationHead(nn.Module):
    def __init__(self,
                 image_size: Sequence[int],
                 scale: int,
                 interp_mode: str = 'bilinear') -> None:
        super().__init__()
        self.image_size = np.array(image_size) // scale
        self.warp = Warp(self.image_size, interp_mode)

    def forward(
            self,
            disp_flow: torch.Tensor,
            source: Optional[torch.Tensor] = None,
            source_oh: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        # warp source image with displacement field
        y_source = self.warp(source, disp_flow) if source is not None else None

        # warp source one-hot segmentation label with displacement field
        y_source_oh = self.warp(source_oh,
                                disp_flow) if source_oh is not None else None

        return y_source, y_source_oh


@REGISTRATION_HEAD.register_module()
class SVFIntegrateHead(nn.Module):
    """Registration head for voxelmorph.

    Args:
        image_size (Sequence[int]): size of input image.
        int_steps (int): number of steps in scaling and squaring integration. (Default: 0, which means no integration.)
        resize_scale (int): scale factor of flow field. (The output flow field might be half resolution.
            Implicit requirement, image size should be divisible by resize scale factor.)
        resize_first (bool): whether to resize before integration, only matters when int_steps>0. (Default: False)
        bidir (bool): whether to run bidirectional registration, only matters when int_steps>0. (Default: False)
        interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 int_steps: int = 0,
                 resize_scale: int = 2,
                 resize_first: bool = False,
                 bidir: bool = False,
                 interp_mode: str = 'bilinear') -> None:
        super().__init__()
        if int_steps > 0:
            flow_size = [s / resize_scale for s in image_size]
            self.integrate = VecIntegrate(flow_size, int_steps, interp_mode)
        else:
            self.integrate = None

        self.resize_flow = ResizeFlow(spatial_scale=resize_scale,
                                      flow_scale=resize_scale,
                                      ndim=len(image_size))
        self.resize_first = resize_first
        self.bidir = bidir if int_steps > 0 else False
        self.warp = Warp(image_size, interp_mode)

    def forward(
            self,
            vec_flow: torch.Tensor,
            source: Optional[torch.Tensor] = None,
            target: Optional[torch.Tensor] = None,
            source_oh: Optional[torch.Tensor] = None,
            target_oh: Optional[torch.Tensor] = None
    ) -> Sequence[torch.Tensor]:
        """
        Args:
            vec_flow (torch.Tensor): flow field predicted by network.
            source (torch.Tensor): source image, tensor of shape [BCHWD].
            target (torch.Tensor): target image, tensor of shape [BCHWD].
            source_oh (torch.Tensor|None): one-hot segmentation label of source image,
                tensor of shape [BCHWD]. (Default: None)
            target_oh (torch.Tensor|None): one-hot segmentation label of target image,
                tensor of shape [BCHWD]. (Default: None)

        """
        if self.resize_first:
            vec_flow = self.resize_flow(vec_flow)

        # integrate by scaling and squaring to generate diffeomorphic flow
        fwd_flow = self.integrate(vec_flow) if self.integrate else vec_flow
        bck_flow = self.integrate(-vec_flow) if self.bidir else None

        if not self.resize_first:
            fwd_flow = self.resize_flow(fwd_flow)
            bck_flow = self.resize_flow(
                bck_flow) if bck_flow is not None else None

        # warp image with displacement field
        y_source = self.warp(source, fwd_flow) if source is not None else None
        y_target = self.warp(
            target, bck_flow) if (self.bidir and target is not None) else None

        # warp one-hot label with displacement field
        y_source_oh = self.warp(source_oh,
                                fwd_flow) if source_oh is not None else None
        y_target_oh = self.warp(target_oh, bck_flow) if (
                self.bidir and target_oh is not None) else None

        return fwd_flow, bck_flow, y_source, y_target, y_source_oh, y_target_oh

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(image_size={self.image_size},'
                     f'int_steps={self.int_steps},'
                     f'resize_scale={self.resize_scale},'
                     f'resize_first={self.resize_first},'
                     f'bidir={self.bidir},'
                     f'interp_mode={self.interp_mode})')
        return repr_str


@REGISTRATION_HEAD.register_module()
class MultiScaleRegistrationHead(nn.Module):
    """Multi-Scale Registration head.

    Args:
        image_size (Sequence[int]): size of input image. shape has to be divisible by scale factor.
        scale_pyramid (Sequence[int]): scale factors for multi-scale registration. (Default: (2, 4, 8))
        interp_mode (str): interpolation mode. Options are: ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 scale_pyramid: Sequence[int] = [4, 8],
                 interp_mode: str = 'bilinear',
                 device: str = 'cuda') -> None:
        super().__init__()
        self.image_size = np.array(image_size)
        self.scale_pyramid = scale_pyramid
        self.avg_pools = [nn.AvgPool3d(kernel_size=scale,
                                       stride=scale,
                                       padding=0)
                          for scale in self.scale_pyramid
                          ]
        self.warps = [Warp(self.image_size // scale, interp_mode).to(device)
                      for scale in self.scale_pyramid]

    def forward(
            self,
            flows: Sequence[torch.Tensor],
            source: torch.Tensor
    ) -> Sequence[torch.Tensor]:
        """
        Args:
            flows (Sequence[torch.Tensor]): flow fields predicted by network.
            source (torch.Tensor): source image, tensor of shape [BCHWD].
        """
        assert len(self.scale_pyramid) == len(flows)
        y_source_list = []

        for i, scale in enumerate(self.scale_pyramid):
            source_down = self.avg_pools[i](source)
            disp_flow = flows[i]
            y_source = self.warps[i](source_down, disp_flow)
            y_source_list.append(y_source)

        return y_source_list


@REGISTRATION_HEAD.register_module()
class MultiScaleAdditionRegistrationHead(nn.Module):
    """Multi-Scale Addition Registration head.

    Args:
        image_size (Sequence[int]): size of input image. shape has to be divisible by scale factor.
        scale_pyramid (Sequence[int]): scale factors for multi-scale registration. (Default: (4, 8))
        interp_mode (str): interpolation mode. Options are: ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 scale_pyramid: Sequence[int] = [8, 4],
                 interp_mode: str = 'bilinear',
                 device: str = 'cuda') -> None:
        super().__init__()
        self.image_size = np.array(image_size)
        self.scale_pyramid = scale_pyramid
        self.avg_pools = [nn.AvgPool3d(kernel_size=scale,
                                       stride=scale,
                                       padding=0)
                          for scale in self.scale_pyramid
                          ]
        self.warps = [Warp(self.image_size // scale, interp_mode).to(device)
                      for scale in self.scale_pyramid]

        self.resize_flow = ResizeFlow(spatial_scale=2,
                                      flow_scale=2,
                                      ndim=len(image_size))

        self.orig_warp = Warp(self.image_size, interp_mode).to(device)

    def forward(
            self,
            flows: Sequence[torch.Tensor],
            source: torch.Tensor,
            source_oh: Optional[torch.Tensor] = None
    ) -> Sequence[torch.Tensor]:
        """
        Args:
            flows (Sequence[torch.Tensor]): flow fields predicted by network. multi-scale - [2, 4, 8]
            source (torch.Tensor): source image, tensor of shape [BCHWD].
        """
        assert len(self.scale_pyramid) + 1 == len(flows)
        y_sources_list = []
        add_flows = []

        flows = flows[::-1]  # [8, 4, 2]

        prev_flow = torch.zeros_like(flows[0])
        for i in range(len(self.scale_pyramid)):
            source_down = self.avg_pools[i](source)
            curr_flow = flows[i]
            disp_flow = curr_flow + prev_flow
            add_flows.append(disp_flow)
            y_source = self.warps[i](source_down, disp_flow)
            y_sources_list.append(y_source)
            prev_flow = self.resize_flow(disp_flow)

        # half_size
        disp_flow = flows[-1] + prev_flow
        add_flows.append(disp_flow)
        disp_flow = self.resize_flow(disp_flow)
        y_source = self.orig_warp(source, disp_flow)
        y_sources_list.append(y_source)

        y_source_oh = self.orig_warp(source_oh,
                                     disp_flow) if source_oh is not None else None

        return disp_flow, add_flows, y_sources_list, y_source_oh
