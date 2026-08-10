from monai.transforms import MapTransform
from monai.transforms.utils import fall_back_tuple
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Hashable, List, Tuple, Union, Sequence


class ResizeWithPadOrCropWithKeypointsd(MapTransform):
    """
    Dictionary-based transform to resize images with padding or cropping while
    adjusting corresponding keypoints.

    Args:
        image_keys: Keys for the images to be transformed
        keypoint_keys: Keys for the keypoints to be adjusted
        spatial_size: Target spatial size for images
        method: Method for handling padding ("symmetric" or "end")
        mode: Padding mode for torch.nn.functional.pad
    """

    def __init__(
            self,
            image_keys: List[str],
            keypoint_keys: List[str],
            spatial_size: Union[Sequence[int], int],
            method: str = "symmetric",
            mode: str = "constant",
    ):
        super().__init__(image_keys + keypoint_keys)
        self.image_keys = image_keys
        self.keypoint_keys = keypoint_keys
        self.spatial_size = spatial_size
        self.method = method
        self.mode = mode

        if method not in ["symmetric", "end"]:
            raise ValueError(f"Method must be 'symmetric' or 'end', got {method}")

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        # Process each image and store transformation parameters
        for key in self.image_keys:
            img = d[key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)

            spatial_shape = img.shape[1:]  # assuming channel-first
            spatial_ndim = len(spatial_shape)
            target_size = fall_back_tuple(self.spatial_size, spatial_shape)

            if len(target_size) != spatial_ndim:
                raise ValueError(
                    f"Target size {target_size} does not match image dimensions {spatial_ndim}"
                )

            # Calculate crop
            crop_slices = []
            crop_offsets = []
            for i, (orig, target) in enumerate(zip(spatial_shape, target_size)):
                if orig > target:
                    start = (orig - target) // 2
                    crop_slices.append(slice(start, start + target))
                    crop_offsets.append(start)
                else:
                    crop_slices.append(slice(0, orig))
                    crop_offsets.append(0)

            cropped = img[(...,) + tuple(crop_slices)]

            # Calculate pad
            pad_width = []
            pad_offsets = []
            for i, (size, target) in enumerate(zip(cropped.shape[1:], target_size)):
                diff = target - size
                if diff < 0:
                    raise ValueError(f"Something went wrong: cropped size {size} > target size {target}")

                if self.method == "symmetric":
                    pre = diff // 2
                    post = diff - pre
                else:  # end padding
                    pre = 0
                    post = diff
                pad_width.append((pre, post))
                pad_offsets.append(pre)

            # Calculate total offsets for keypoint adjustment
            # (positive means add to keypoint coordinates, negative means subtract)
            total_offsets = [po - co for po, co in zip(pad_offsets, crop_offsets)]

            # Store the transformed image
            needs_pad = any(pre != 0 or post != 0 for pre, post in pad_width)
            if needs_pad:
                # Convert padding format for torch functional pad: (padding_left, padding_right, padding_top, padding_bottom, ...)
                torch_pad_width = sum(pad_width[::-1], ()) # concatenate list of tuples
                d[key] = torch.nn.functional.pad(cropped, torch_pad_width, mode=self.mode)
            else:
                d[key] = cropped

            # Store the offset information for keypoints
            d[f"{key}_offsets"] = total_offsets

        # Adjust keypoints based on the corresponding image transformations
        for key in self.keypoint_keys:
            # Extract the relevant image name from the keypoint key
            keypoint_parts = key.split('_')
            if len(keypoint_parts) >= 3 and keypoint_parts[-1] == "keypnt":
                # Handle cases like "begin_FBCT_keypnt" -> "FBCT" or "end_FBCT_keypnt" -> "FBCT"
                corresponding_img_key = keypoint_parts[1]  # Extract FBCT part
                if corresponding_img_key == "CBCT":
                    corresponding_img_key = '_'.join(keypoint_parts[:2])  # "begin_CBCT" or "end_CBCT"
            else:
                corresponding_img_key = keypoint_parts[0]

            if corresponding_img_key is None:
                raise ValueError(f"Could not determine corresponding image key for keypoint key {key}")

            total_offsets = d[f"{corresponding_img_key}_offsets"]
            spatial_ndim = len(total_offsets)

            keypoints = np.asarray(d[key])

            if len(keypoints.shape) != 2 or keypoints.shape[1] != spatial_ndim:
                raise ValueError(f"Keypoints with key {key} should be of shape (N, {spatial_ndim}), got {keypoints.shape}")

            # Apply offsets to coordinates (z,y,x or y,x)
            for i in range(spatial_ndim):
                keypoints[:, i] += total_offsets[i]

            d[key] = keypoints

        return d


class ResizeWithInterpolationKeypointsd(MapTransform):
    """
    Dictionary-based transform to resize images with interpolation while
    adjusting corresponding keypoints by the same scale factor.

    Args:
        image_keys: Keys for the images to be transformed
        keypoint_keys: Keys for the keypoints to be adjusted
        spatial_size: Target spatial size for images (absolute size or scale factors)
        mode: Interpolation mode ("nearest", "linear", "bilinear", "bicubic", "trilinear", etc.)
        as_scale_factor: If True, spatial_size is treated as scale factors rather than absolute size
    """

    def __init__(
            self,
            image_keys: List[str],
            keypoint_keys: List[str],
            spatial_size: Union[Sequence[Union[int, float]], int, float],
            mode: str = "bilinear",
            as_scale_factor: bool = False,
            align_corners: bool = False,
    ):
        super().__init__(image_keys + keypoint_keys)
        self.image_keys = image_keys
        self.keypoint_keys = keypoint_keys
        self.spatial_size = spatial_size
        self.mode = mode
        self.as_scale_factor = as_scale_factor
        self.align_corners = align_corners

        # Define valid interpolation modes
        self.valid_modes = ["nearest", "linear", "bilinear", "bicubic", "trilinear"]
        if mode not in self.valid_modes:
            raise ValueError(f"Mode must be one of {self.valid_modes}, got {mode}")

    def _get_scale_factors(self, img_shape: Sequence[int], target_size: Sequence[int]) -> Tuple[float, ...]:
        """Calculate scale factors between original and target size."""
        return tuple(float(ts) / float(s) for s, ts in zip(img_shape, target_size))

    def _get_target_size(self, img_shape: Sequence[int]) -> Tuple[int, ...]:
        """Determine target size from spatial_size parameter."""
        if self.as_scale_factor:
            if isinstance(self.spatial_size, (int, float)):
                scale_factor = float(self.spatial_size)
                return tuple(int(scale_factor * s) for s in img_shape)
            else:
                return tuple(int(f * s) for f, s in zip(self.spatial_size, img_shape))
        else:
            if isinstance(self.spatial_size, (int, float)):
                size = int(self.spatial_size)
                return (size,) * len(img_shape)
            else:
                return tuple(int(s) for s in self.spatial_size)

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        # Process each image and store transformation parameters
        for key in self.image_keys:
            img = d[key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)

            spatial_shape = img.shape[1:] # assuming channel-first
            spatial_ndim = len(spatial_shape)

            # Determine target size
            target_size = self._get_target_size(spatial_shape)
            if len(target_size) != spatial_ndim:
                raise ValueError(
                    f"Target size {target_size} does not match image dimensions {spatial_ndim}"
                )

            # Calculate scale factors for keypoint adjustment
            scale_factors = self._get_scale_factors(spatial_shape, target_size)
            d[f"{key}_scale_factors"] = scale_factors

            # Resize image using interpolation
            if spatial_ndim == 2:
                # Add batch dimension if missing
                need_batch = len(img.shape) == 3  # [C, H, W]
                if need_batch:
                    img = img.unsqueeze(0)  # [1, C, H, W]

                resized_img = F.interpolate(
                    img,
                    size=target_size,
                    mode=self.mode if self.mode != "linear" else "bilinear",
                    align_corners=self.align_corners if self.mode != "nearest" else None
                )

                # Remove batch dimension if it was added
                if need_batch:
                    resized_img = resized_img.squeeze(0)

            elif spatial_ndim == 3:
                # Add batch dimension if missing
                need_batch = len(img.shape) == 4  # [C, D, H, W]
                if need_batch:
                    img = img.unsqueeze(0)  # [1, C, D, H, W]

                resized_img = F.interpolate(
                    img,
                    size=target_size,
                    mode=self.mode if self.mode != "linear" else "trilinear",
                    align_corners=self.align_corners if self.mode != "nearest" else None
                )

                # Remove batch dimension if it was added
                if need_batch:
                    resized_img = resized_img.squeeze(0)
            else:
                raise ValueError(f"Unsupported spatial dimensions: {spatial_ndim}")

            d[key] = resized_img

        # Adjust keypoints based on the corresponding image transformations
        for key in self.keypoint_keys:
            # Extract the relevant image name from the keypoint key
            keypoint_parts = key.split('_')
            if len(keypoint_parts) >= 3 and keypoint_parts[-1] == "keypnt":
                # Handle cases like "begin_FBCT_keypnt" -> "FBCT" or "end_FBCT_keypnt" -> "FBCT"
                corresponding_img_key = keypoint_parts[1]  # Extract FBCT part
                if corresponding_img_key == "CBCT":
                    corresponding_img_key = '_'.join(keypoint_parts[:2])  # "begin_CBCT" or "end_CBCT"
            else:
                corresponding_img_key = keypoint_parts[0]

            if corresponding_img_key is None:
                raise ValueError(f"Could not determine corresponding image key for keypoint key {key}")

            scale_factors = d[f"{corresponding_img_key}_scale_factors"]
            spatial_ndim = len(scale_factors)

            keypoints = np.asarray(d[key])

            if len(keypoints.shape) != 2 or keypoints.shape[1] != spatial_ndim:
                raise ValueError(f"Keypoints with key {key} should be of shape (N, {spatial_ndim}), got {keypoints.shape}")

            # Determine which image's scale factors to use based on naming convention
            corresponding_img_key = None

            # Apply scale factors to coordinates (z,y,x or y,x)
            for i in range(spatial_ndim):
                keypoints[:, i] = keypoints[:, i] * scale_factors[i]

            d[key] = keypoints

        return d
