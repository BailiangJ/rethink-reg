from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models import Warp
from models.utils.integrate import VecIntegrate


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated CLI value into floats."""
    if value is None or value == "":
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_translation_vectors(
        magnitudes_mm: Sequence[float],
        spacing: Sequence[float],
        num_cases: int,
        seed: int = 2023,
) -> torch.Tensor:
    """Create deterministic per-case translation vectors in voxel units.

    The same case gets the same random direction for every magnitude, making the
    stress curve isolate displacement size rather than direction changes.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn(num_cases, 3, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    spacing_t = torch.tensor(spacing, dtype=torch.float32).view(1, 1, 3)
    magnitudes_t = torch.tensor(magnitudes_mm, dtype=torch.float32).view(-1, 1, 1)
    return magnitudes_t * directions.unsqueeze(0) / spacing_t


def dense_translation_flow(
        image_size: Sequence[int],
        translation_vox: torch.Tensor,
        device: torch.device | str,
        batch_size: int = 1,
) -> torch.Tensor:
    """Return the sampling flow that shifts image content by translation_vox.

    Warp(image, flow) samples image at grid + flow. To move image content by +t,
    each output location samples input at x - t, so the dense sampling flow is -t.
    """
    flow = -translation_vox.to(device=device, dtype=torch.float32).view(1, 3, 1, 1, 1)
    return flow.expand(batch_size, 3, *image_size).contiguous()


def translate_image_and_keypoints(
        image: torch.Tensor,
        keypoints: torch.Tensor,
        translation_vox: torch.Tensor,
        warp: Warp,
        mask: Optional[torch.Tensor] = None,
        mask_warp: Optional[Warp] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Translate image content and keypoints by a known voxel displacement."""
    flow = dense_translation_flow(image.shape[2:], translation_vox, image.device, image.shape[0])
    translated_image = warp(image, flow)
    translated_keypoints = keypoints + translation_vox.to(image.device).view(1, 1, 3)
    translated_mask = None
    if mask is not None:
        translated_mask = (mask_warp or warp)(mask, flow)
    return translated_image, translated_keypoints, translated_mask, flow


def foreground_overlap_fraction(
        fixed: torch.Tensor,
        moving: torch.Tensor,
        threshold: float = 0.0,
) -> torch.Tensor:
    """Compute foreground overlap / fixed foreground for each batch item."""
    fixed_fg = fixed > threshold
    moving_fg = moving > threshold
    dims = tuple(range(1, fixed.ndim))
    denom = fixed_fg.sum(dim=dims).clamp_min(1)
    overlap = (fixed_fg & moving_fg).sum(dim=dims)
    return overlap.float() / denom.float()


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_smooth3d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian smoothing to a [B, C, H, W, D] tensor."""
    if sigma <= 0:
        return field
    kernel = _gaussian_kernel1d(sigma, field.device, field.dtype)
    channels = field.shape[1]
    padding = kernel.numel() // 2

    kernel_x = kernel.view(1, 1, -1, 1, 1).expand(channels, 1, -1, 1, 1)
    kernel_y = kernel.view(1, 1, 1, -1, 1).expand(channels, 1, 1, -1, 1)
    kernel_z = kernel.view(1, 1, 1, 1, -1).expand(channels, 1, 1, 1, -1)

    field = F.conv3d(field, kernel_x, padding=(padding, 0, 0), groups=channels)
    field = F.conv3d(field, kernel_y, padding=(0, padding, 0), groups=channels)
    field = F.conv3d(field, kernel_z, padding=(0, 0, padding), groups=channels)
    return field


def _cubic_bspline_weights(frac: torch.Tensor) -> torch.Tensor:
    """Cubic uniform B-spline basis weights for offsets [-1, 0, 1, 2]."""
    frac2 = frac * frac
    frac3 = frac2 * frac
    return torch.stack([
        (1 - frac) ** 3 / 6.0,
        (3 * frac3 - 6 * frac2 + 4) / 6.0,
        (-3 * frac3 + 3 * frac2 + 3 * frac + 1) / 6.0,
        frac3 / 6.0,
    ], dim=0)


def _bspline_interpolate_along(field: torch.Tensor, out_size: int, dim: int) -> torch.Tensor:
    """Cubic B-spline interpolate a 5D field along one spatial dimension."""
    in_size = field.shape[dim]
    if in_size < 2:
        return field.expand(*field.shape[:dim], out_size, *field.shape[dim + 1:]).contiguous()

    coords = torch.linspace(0, in_size - 1, out_size, device=field.device, dtype=field.dtype)
    base = torch.floor(coords).long().clamp(0, in_size - 1)
    frac = (coords - base.to(coords.dtype)).clamp(0, 1)
    weights = _cubic_bspline_weights(frac)

    pad = [0, 0, 0, 0, 0, 0]
    # F.pad order is D, W, H for 5D tensors.
    if dim == 2:
        pad[4], pad[5] = 2, 2
    elif dim == 3:
        pad[2], pad[3] = 2, 2
    elif dim == 4:
        pad[0], pad[1] = 2, 2
    else:
        raise ValueError(f'Unsupported interpolation dim: {dim}')
    padded = F.pad(field, pad, mode='replicate')

    pieces = []
    for basis_idx, offset in enumerate((-1, 0, 1, 2)):
        indices = (base + offset + 2).clamp(0, in_size + 3)
        part = torch.index_select(padded, dim, indices)
        weight_shape = [1] * field.ndim
        weight_shape[dim] = out_size
        pieces.append(part * weights[basis_idx].view(weight_shape))
    return sum(pieces)


def bspline_upsample3d(control: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    """Upsample sparse control-point vectors with separable cubic B-spline basis."""
    field = _bspline_interpolate_along(control, int(image_size[0]), dim=2)
    field = _bspline_interpolate_along(field, int(image_size[1]), dim=3)
    field = _bspline_interpolate_along(field, int(image_size[2]), dim=4)
    return field.contiguous()


def displacement_magnitude_mm(flow: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
    spacing_t = torch.tensor(spacing, dtype=flow.dtype, device=flow.device).view(1, 3, 1, 1, 1)
    return torch.linalg.norm(flow * spacing_t, dim=1)


def summarize_displacement(
        flow: torch.Tensor,
        spacing: Sequence[float],
        fg_mask: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    disp = displacement_magnitude_mm(flow, spacing).detach().float().cpu()  # (B, H, W, D)
    B = disp.shape[0]
    if fg_mask is not None:
        fg = (fg_mask.squeeze(1).detach().cpu() > 0.5)  # (B, H, W, D)
        results: Dict[str, list] = {"mean": [], "std": [], "max": [], "p95": []}
        for b in range(B):
            vals = disp[b][fg[b]].numpy()
            if vals.size == 0:
                vals = disp[b].numpy().ravel()
            results["mean"].append(vals.mean())
            results["std"].append(vals.std())
            results["max"].append(vals.max())
            results["p95"].append(np.percentile(vals, 95))
        return {k: np.array(v) for k, v in results.items()}
    flat = disp.numpy().reshape(B, -1)
    return {
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
        "max": flat.max(axis=1),
        "p95": np.percentile(flat, 95, axis=1),
    }


def generate_svf_flows(
        image_size: Sequence[int],
        spacing: Sequence[float],
        target_mean_mm: float,
        device: torch.device | str,
        seed: int,
        coarse_size: int = 5,
        smooth_sigma: float = 0.0,
        int_steps: int = 5,
        calibration_iters: int = 3,
        calibration_stat: str = "p95",
        fg_mask: Optional[torch.Tensor] = None,
        fg_smooth_sigma: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    """Generate paired forward/inverse SVF flows for synthetic stress tests.

    The velocity is sampled on a sparse B-spline control lattice and evaluated
    densely with a cubic B-spline basis. This gives a much smoother deformation
    than upsampling voxelwise noise, especially at larger displacement levels.

    If ``fg_mask`` is provided (shape (1,1,H,W,D), values 0/1), the dense
    velocity field is multiplied by a Gaussian-smoothed version of the mask
    (sigma = ``fg_smooth_sigma`` voxels). This confines deformations to the
    foreground anatomy and prevents large-amplitude folding at tissue boundaries.
    Calibration and the returned displacement summary are then computed over
    foreground voxels only, so the reported statistics match the requested level.

    The forward flow maps fixed/target coordinates to synthetic moving/source
    coordinates, matching this codebase's TRE and Warp convention. The inverse
    flow is generated by integrating -velocity and is used to synthesize the
    moving image from the fixed image.
    """
    if target_mean_mm == 0:
        zeros = torch.zeros(1, 3, *image_size, dtype=torch.float32, device=device)
        return zeros, zeros, summarize_displacement(zeros, spacing, fg_mask)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    control_size = max(4, int(coarse_size))
    control_shape = (control_size, control_size, control_size)
    control = torch.randn(1, 3, *control_shape, generator=generator, dtype=torch.float32, device=device)

    if smooth_sigma > 0:
        control = gaussian_smooth3d(control, smooth_sigma)

    # Dampen boundary control points so large fields do not aggressively pull
    # anatomy out of the field-of-view. This is important for visually plausible
    # synthetic CT pairs at >=10 mm displacement levels.
    axes = [torch.linspace(0, 1, control_size, device=device, dtype=torch.float32) for _ in range(3)]
    ramps = [0.25 + 0.75 * torch.sin(np.pi * axis) for axis in axes]
    boundary_weight = ramps[0].view(1, 1, -1, 1, 1) * ramps[1].view(1, 1, 1, -1, 1) * ramps[2].view(1, 1, 1, 1, -1)
    control = control * boundary_weight

    velocity = bspline_upsample3d(control, image_size)

    # Optionally mask the velocity to the foreground region. Gaussian-smoothing
    # the binary mask gives a soft, differentiable boundary so integration does
    # not produce sharp velocity discontinuities at the tissue edge.
    if fg_mask is not None:
        fg_weight = gaussian_smooth3d(
            fg_mask.to(device=device, dtype=torch.float32).clamp(0, 1),
            fg_smooth_sigma,
        ).clamp(0, 1)
        velocity = velocity * fg_weight

    integrator = VecIntegrate(image_size=image_size, num_steps=int_steps).to(device)
    scale = 1.0
    fwd_flow = None
    for _ in range(max(1, calibration_iters)):
        fwd_flow = integrator(velocity * scale)
        disp_spatial = displacement_magnitude_mm(fwd_flow, spacing)  # (B, H, W, D)
        if fg_mask is not None:
            fg = fg_mask.to(device=device).squeeze(1) > 0.5  # (B, H, W, D)
            fg_vals = disp_spatial[fg]
            if fg_vals.numel() == 0:
                fg_vals = disp_spatial.reshape(fwd_flow.shape[0], -1).flatten()
            if calibration_stat == "mean":
                stat_mm = fg_vals.mean()
            elif calibration_stat == "p95":
                stat_mm = torch.quantile(fg_vals, 0.95)
            elif calibration_stat == "max":
                stat_mm = fg_vals.max()
            else:
                raise ValueError(f"Unsupported calibration_stat: {calibration_stat!r}")
        else:
            disp_mm = disp_spatial.reshape(fwd_flow.shape[0], -1)
            if calibration_stat == "mean":
                stat_mm = disp_mm.mean()
            elif calibration_stat == "p95":
                stat_mm = torch.quantile(disp_mm, 0.95)
            elif calibration_stat == "max":
                stat_mm = disp_mm.max()
            else:
                raise ValueError(f"Unsupported calibration_stat: {calibration_stat!r}")
        stat_mm = stat_mm.clamp_min(1e-6)
        scale *= float(target_mean_mm / stat_mm.detach().cpu())

    fwd_flow = integrator(velocity * scale)
    inv_flow = integrator(-velocity * scale)
    return fwd_flow, inv_flow, summarize_displacement(fwd_flow, spacing, fg_mask)


def sample_flow_at_keypoints(
        flow: torch.Tensor,
        keypoints: torch.Tensor,
        image_size: Sequence[int],
        mode: str = "bilinear",
) -> torch.Tensor:
    """Sample a dense flow at keypoint coordinates."""
    sample_grid = keypoints.clone()
    for dim_idx, dim in enumerate(image_size):
        sample_grid[..., dim_idx] = sample_grid[..., dim_idx] * 2 / (dim - 1) - 1
    sample_grid = sample_grid[..., list(range(len(image_size) - 1, -1, -1))]
    sample_grid = sample_grid[None, None, ...]
    sampled = F.grid_sample(flow, sample_grid, mode=mode, padding_mode="zeros", align_corners=True)
    sampled = sampled.squeeze((2, 3)).permute(0, 2, 1)
    return sampled


def synthesize_from_svf(
        fixed_image: torch.Tensor,
        fixed_keypoints: torch.Tensor,
        fwd_flow: torch.Tensor,
        inv_flow: torch.Tensor,
        warp: Warp,
        fixed_mask: Optional[torch.Tensor] = None,
        mask_warp: Optional[Warp] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Create a synthetic moving image/mask/keypoints from a fixed image."""
    moving_image = warp(fixed_image, inv_flow)
    moving_keypoints = fixed_keypoints + sample_flow_at_keypoints(
        fwd_flow, fixed_keypoints, fixed_image.shape[2:])
    moving_mask = None
    if fixed_mask is not None:
        moving_mask = (mask_warp or warp)(fixed_mask, inv_flow)
    return moving_image, moving_keypoints, moving_mask


def compute_epe_stats_mm(
        pred_flow: torch.Tensor,
        gt_flow: torch.Tensor,
        spacing: Sequence[float],
        mask: Optional[torch.Tensor] = None,
        percentiles: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Return per-batch endpoint-error statistics in mm."""
    if percentiles is None:
        percentiles_arr = np.arange(5, 101, 5, dtype=np.float32)
    else:
        percentiles_arr = np.asarray(percentiles, dtype=np.float32)

    epe = displacement_magnitude_mm(pred_flow - gt_flow, spacing).detach().float().cpu()
    B = epe.shape[0]

    if mask is not None:
        mask_bool = mask.squeeze(1).detach().cpu() > 0.5
        means = []
        percentile_rows = []
        for batch_idx in range(B):
            vals = epe[batch_idx][mask_bool[batch_idx]]
            if vals.numel() == 0:
                vals = epe[batch_idx].reshape(-1)
            vals_np = vals.numpy()
            means.append(vals_np.mean())
            percentile_rows.append(np.percentile(vals_np, percentiles_arr))
        return {
            "mean": np.asarray(means),
            "percentiles": np.asarray(percentile_rows),
            "percentile_levels": percentiles_arr,
        }

    flat = epe.numpy().reshape(B, -1)
    return {
        "mean": flat.mean(axis=1),
        "percentiles": np.percentile(flat, percentiles_arr, axis=1).T,
        "percentile_levels": percentiles_arr,
    }


def compute_epe_mm(
        pred_flow: torch.Tensor,
        gt_flow: torch.Tensor,
        spacing: Sequence[float],
        mask: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-batch mean and 95th-percentile endpoint error in mm."""
    stats = compute_epe_stats_mm(pred_flow, gt_flow, spacing, mask=mask, percentiles=[95])
    return stats["mean"], stats["percentiles"][:, 0]


def _final_flow(flow_like: Any) -> torch.Tensor:
    if hasattr(flow_like, "iter_flows"):
        return flow_like.iter_flows[-1]
    if hasattr(flow_like, "pyramid_flows"):
        return flow_like.pyramid_flows[-1]
    if isinstance(flow_like, dict):
        if "iter_flows" in flow_like:
            return flow_like["iter_flows"][-1]
        if "pyramid_flows" in flow_like:
            return flow_like["pyramid_flows"][-1]
        if "fwd" in flow_like:
            return _final_flow(flow_like["fwd"])
    if isinstance(flow_like, (list, tuple)):
        return flow_like[-1]
    return flow_like


def _split_forward_output(output: Any) -> Tuple[Any, Any]:
    if isinstance(output, dict):
        return output.get("fwd", output), output.get("bck", None)
    if hasattr(output, "fwd"):
        return output.fwd, getattr(output, "bck", None)
    if isinstance(output, (list, tuple)) and len(output) == 2:
        first, second = output
        if second is None:
            return first, None
        if torch.is_tensor(first) and torch.is_tensor(second):
            return first, second
        if not torch.is_tensor(first):
            return first, second
    return output, None


def infer_raw_flows(model: torch.nn.Module, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a model and return final raw forward/backward flows.

    This handles single-flow, bidirectional, pyramid, and iterative-pyramid return
    conventions found in the NLST/Lung250M experiment folders.
    """
    output = model(source, target)
    fwd_like, bck_like = _split_forward_output(output)
    fwd_flow = _final_flow(fwd_like)

    if bck_like is None:
        reverse_output = model(target, source)
        bck_fwd_like, _ = _split_forward_output(reverse_output)
        bck_flow = _final_flow(bck_fwd_like)
    else:
        bck_flow = _final_flow(bck_like)

    return fwd_flow, bck_flow
