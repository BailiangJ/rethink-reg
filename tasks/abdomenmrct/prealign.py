"""Pre-alignment of an abdominal MR/CT pair before deformable registration.

MR and CT are acquired separately, so the two volumes can be tens of voxels
apart before any deformable model sees them. Both `train.py` and `evaluate.py`
therefore run one of these pre-aligners first:

    TranslationPreAlign  centroid translation of the organ labels  (paper default)
    RigidPreAlign        rotation + translation, optimised by gradient descent

Both share the same pipeline -- pad the two volumes to a common shape, align the
source to the target, crop back to the target's shape -- and differ only in how
the transform is produced, which is the `_align` hook.

Segmentations are expected to be one-hot encoded already; the caller builds them
from the labels common to both modalities.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
import torch.nn.functional as F

from models import Warp
from rigid_solver import get_closest_rigid


def compute_centroids(label: torch.Tensor) -> torch.Tensor:
    """
    Compute centroids of one-hot encoded labels

    Args:
        label: PyTorch tensor of shape (B, C, H, W, D) with one-hot encoded labels

    Returns:
        Tensor of shape (B, C, 3) containing centroids (x, y, z coordinates).
        Channels with no foreground get a zero centroid.
    """
    B, C, H, W, D = label.shape

    # Create coordinate grids
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(H, device=label.device),
        torch.arange(W, device=label.device),
        torch.arange(D, device=label.device),
        indexing='ij',
    )  # Each is (H, W, D)

    # Expand to (1, 1, H, W, D) for broadcasting
    x_coords = x_coords[None, None, ...]
    y_coords = y_coords[None, None, ...]
    z_coords = z_coords[None, None, ...]

    # Compute sums
    weight_sum = label.sum(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
    # Avoid division by zero
    weight_sum_safe = weight_sum + (weight_sum == 0)

    x_centroid = (label * x_coords).sum(dim=(2, 3, 4), keepdim=True) / weight_sum_safe
    y_centroid = (label * y_coords).sum(dim=(2, 3, 4), keepdim=True) / weight_sum_safe
    z_centroid = (label * z_coords).sum(dim=(2, 3, 4), keepdim=True) / weight_sum_safe

    # Concatenate and remove extra dims
    centroids = torch.cat([x_centroid, y_centroid, z_centroid], dim=-1)  # (B, C, 1, 1, 3)
    centroids = centroids.squeeze(3).squeeze(2)  # (B, C, 3)

    # Find where class is missing
    mask_zero = (weight_sum.squeeze() == 0)  # (B, C)

    # Set centroids to zero where class is missing
    centroids = centroids.masked_fill(mask_zero.unsqueeze(-1), 0.0)

    return centroids


def compute_translation(source_label: torch.Tensor,
                        target_label: torch.Tensor) -> torch.Tensor:
    """
    Compute translation vector between centroids of two one-hot encoded labels

    Organ channels missing from either modality are masked before the
    channel-wise centroid offsets are averaged.

    Args:
        source_label: Source label tensor (B, C, H, W, D)
        target_label: Target label tensor (B, C, H, W, D)

    Returns:
        translation: Translation vectors (B, 3)
    """
    # Compute centroids for both label sets
    src_cnt = compute_centroids(source_label)  # (B, C, 3)
    tgt_cnt = compute_centroids(target_label)  # (B, C, 3)

    # Find where source or target is all zeros for a channel
    src_zero = (source_label.sum(dim=(2, 3, 4)) == 0)  # (B, C)
    tgt_zero = (target_label.sum(dim=(2, 3, 4)) == 0)  # (B, C)
    mask = (src_zero | tgt_zero).unsqueeze(-1)          # (B, C, 1)

    # Set centroids to zero where mask is True
    src_cnt = src_cnt.masked_fill(mask, 0.0)
    tgt_cnt = tgt_cnt.masked_fill(mask, 0.0)

    # Compute translation (src_cnt - tgt_cnt)
    translation = src_cnt.mean(dim=1) - tgt_cnt.mean(dim=1)

    return translation


class PreAlign:
    """Pad -> align -> crop, for one (moving MR, fixed CT) pair.

    Subclasses implement `_align`; `process_pair` is shared.

    Attributes:
        device (str): Device to use for computations ('cuda' or 'cpu')
        compute_dice (Optional[Callable]): Optional function to compute Dice score
        img_warp (Optional[Warp]): Warping module for images
        seg_warp (Optional[Warp]): Warping module for segmentations
    """

    #: wording of the Dice line printed by process_pair
    transform_name = 'alignment'

    def __init__(self,
                 compute_dice: Optional[Callable] = None,
                 device: str = 'cuda'):
        """
        Args:
            compute_dice: Optional function to compute Dice score between
                segmentations. If provided, prints Dice before and after
                alignment.
            device: Device to use for computations ('cuda' or 'cpu').
        """
        self.device = device
        self.compute_dice = compute_dice
        # Initialize warping modules
        self.img_warp = None
        self.seg_warp = None

    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        """Ensure input is a tensor on the correct device."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        return x.to(self.device)

    def _ensure_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has correct shape (B, C, D, H, W)."""
        if len(x.shape) == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(x.shape) == 4:
            x = x.unsqueeze(0)  # Add batch dimension
        return x

    def _get_warp_modules(self, image_size: Tuple[int, int, int]) -> Tuple[Warp, Warp]:
        """Get or create warping modules for the given image size."""
        if self.img_warp is None or list(self.img_warp.image_size) != list(image_size):
            self.img_warp = Warp(image_size=image_size, interp_mode='bilinear').to(self.device)
            self.seg_warp = Warp(image_size=image_size, interp_mode='nearest').to(self.device)
        return self.img_warp, self.seg_warp

    def _pad_to_largest_shape(self, src_img: torch.Tensor, tgt_img: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Pad both images to the largest shape and return padding information.

        Args:
            src_img: Source image tensor
            tgt_img: Target image tensor

        Returns:
            Tuple containing:
            - Padded source image
            - Padded target image
            - Dictionary with padding information for both images
        """
        src_img = self._ensure_shape(self._ensure_tensor(src_img))
        tgt_img = self._ensure_shape(self._ensure_tensor(tgt_img))

        # Get current sizes
        src_size = src_img.shape[2:]
        tgt_size = tgt_img.shape[2:]

        # Calculate target size (maximum of both)
        target_size = tuple(max(s, t) for s, t in zip(src_size, tgt_size))

        # Calculate padding for each image, centred
        paddings = {}
        infos = {}
        for name, size in (('src', src_size), ('tgt', tgt_size)):
            padding = []
            info = {}
            for i, (curr, targ) in enumerate(zip(size, target_size)):
                pad = max(0, targ - curr)
                pad_before = pad // 2
                pad_after = pad - pad_before
                padding.extend([pad_before, pad_after])
                info[f'dim_{i}'] = {'before': pad_before, 'after': pad_after}
            paddings[name] = padding
            infos[name] = info

        # Apply padding if needed
        if any(paddings['src']):
            src_img = F.pad(src_img, paddings['src'][::-1])
        if any(paddings['tgt']):
            tgt_img = F.pad(tgt_img, paddings['tgt'][::-1])

        padding_info = {
            'src_padding': infos['src'],
            'tgt_padding': infos['tgt'],
            'target_size': target_size
        }

        return src_img, tgt_img, padding_info

    def _crop_to_shape(self, tensor: torch.Tensor,
                       target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Crop tensor to match target shape.

        Args:
            tensor: Input tensor to crop
            target_shape: Target shape (D, H, W) to crop to

        Returns:
            Cropped tensor matching target shape
        """
        current_shape = tensor.shape[2:]

        # Calculate cropping
        slices = []
        for curr, targ in zip(current_shape, target_shape):
            if curr > targ:
                start = (curr - targ) // 2
                end = start + targ
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))

        # Apply cropping if needed
        if any(curr > targ for curr, targ in zip(current_shape, target_shape)):
            tensor = tensor[:, :, slices[0], slices[1], slices[2]]

        return tensor

    def _align(self,
               src_img: torch.Tensor,
               src_oh: torch.Tensor,
               tgt_img: torch.Tensor,
               tgt_oh: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (aligned image, aligned segmentation, transform).

        Subclass hook. Inputs are already padded to a common shape; the returned
        tensors keep that shape.
        """
        raise NotImplementedError

    def process_pair(self,
                     src_img: torch.Tensor,
                     src_oh: torch.Tensor,
                     tgt_img: torch.Tensor,
                     tgt_oh: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a pair of source and target images through the complete workflow.

        1. Pads both images to the largest shape
        2. Computes and applies the alignment
        3. Crops the aligned source image back to target shape
        4. Optionally computes and prints Dice scores if compute_dice is provided

        Args:
            src_img: Source image tensor (B, C, D, H, W)
            src_oh: Source one-hot segmentation tensor (B, num_classes, D, H, W)
            tgt_img: Target image tensor (B, C, D, H, W)
            tgt_oh: Target one-hot segmentation tensor (B, num_classes, D, H, W)

        Returns:
            Dictionary containing:
            - src_image: Processed source image
            - src_seg: Processed source segmentation
            - tgt_image: Target image (unchanged)
            - tgt_seg: Target segmentation (unchanged)
            - transform: the translation vector or rigid flow that was applied
            - padding_info: Information about padding applied
        """
        # Ensure all inputs are on the correct device
        src_img = self._ensure_tensor(src_img)
        src_oh = self._ensure_tensor(src_oh)
        tgt_img = self._ensure_tensor(tgt_img)
        tgt_oh = self._ensure_tensor(tgt_oh)

        # Pad to largest shape
        padded_src_img, padded_tgt_img, padding_info = self._pad_to_largest_shape(src_img, tgt_img)
        padded_src_oh, padded_tgt_oh, _ = self._pad_to_largest_shape(src_oh, tgt_oh)

        # Compute and apply the alignment
        aligned_src_img, aligned_src_oh, transform = self._align(
            padded_src_img, padded_src_oh, padded_tgt_img, padded_tgt_oh
        )

        # Crop source images to match target shape
        final_src_img = self._crop_to_shape(aligned_src_img, tgt_img.shape[2:])
        final_src_oh = self._crop_to_shape(aligned_src_oh, tgt_img.shape[2:])

        # Compute and print Dice scores if requested
        if self.compute_dice is not None:
            with torch.no_grad():
                # Ensure all tensors are on the same device for Dice computation
                padded_src_oh = padded_src_oh.to(self.device)
                padded_tgt_oh = padded_tgt_oh.to(self.device)
                final_src_oh = final_src_oh.to(self.device)
                tgt_oh = tgt_oh.to(self.device)

                init_dice, = self.compute_dice(padded_src_oh, padded_tgt_oh)
                final_dice, = self.compute_dice(final_src_oh, tgt_oh)
                print(f"Init Dice: {init_dice.nanmean().item():.4f}, "
                      f"Dice after {self.transform_name}: {final_dice.nanmean().item():.4f}")

        return {
            'src_image': final_src_img,
            'src_seg': final_src_oh,
            'tgt_image': tgt_img,  # Target image is unchanged
            'tgt_seg': tgt_oh,     # Target segmentation is unchanged
            'transform': transform,
            'padding_info': padding_info
        }


class TranslationPreAlign(PreAlign):
    """Centroid translation. The pre-alignment used for the paper's results."""

    transform_name = 'translation'

    def __init__(self,
                 compute_dice: Optional[Callable] = None,
                 device: str = 'cuda',
                 randomize: bool = False):
        """
        Args:
            compute_dice: see PreAlign.
            device: see PreAlign.
            randomize: scale the translation by a uniform factor in [0, 1], as a
                training-time augmentation. The paper's runs did not use it.
        """
        super().__init__(compute_dice=compute_dice, device=device)
        self.randomize = randomize

    def _align(self, src_img, src_oh, tgt_img, tgt_oh):
        # Ensure tensors are on the correct device and have correct shape
        src_img = self._ensure_shape(self._ensure_tensor(src_img))
        src_oh = self._ensure_shape(self._ensure_tensor(src_oh)).float()
        tgt_oh = self._ensure_shape(self._ensure_tensor(tgt_oh)).float()

        image_size = src_img.shape[2:]
        img_warp, seg_warp = self._get_warp_modules(image_size)

        # Compute translation using one-hot segmentations
        translation = compute_translation(src_oh, tgt_oh)

        if self.randomize:
            translation = np.random.uniform() * translation

        # Broadcast the translation into a dense flow field
        pre_translation = torch.ones(
            (1, 3, *image_size),
            device=self.device) * translation.view((1, 3, 1, 1, 1))

        return (img_warp(src_img, pre_translation),
                seg_warp(src_oh, pre_translation),
                translation)


class RigidPreAlign(PreAlign):
    """Rotation + translation, fitted to the one-hot labels by gradient descent."""

    transform_name = 'rigid alignment'

    def _align(self, src_img, src_oh, tgt_img, tgt_oh):
        # Ensure tensors are on the correct device and have correct shape
        src_img = self._ensure_shape(self._ensure_tensor(src_img))
        src_oh = self._ensure_shape(self._ensure_tensor(src_oh)).float()
        tgt_oh = self._ensure_shape(self._ensure_tensor(tgt_oh)).float()

        image_size = src_img.shape[2:]
        img_warp, _ = self._get_warp_modules(image_size)

        # get_closest_rigid returns the resampled segmentation alongside the flow
        aligned_src_oh, rigid_flow = get_closest_rigid(src_oh,
                                                       tgt_oh,
                                                       lr=1e-2,
                                                       num_iteration=50)

        return img_warp(src_img, rigid_flow), aligned_src_oh, rigid_flow
