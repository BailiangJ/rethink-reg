"""NumPy metrics operating on integer label maps and displacement fields.

These are the Learn2Reg-style reference implementations, kept alongside the
tensor-based metrics in this package. The shipped evaluators use MONAI's
``DiceHelper``/``HausdorffDistanceMetric`` for the reported numbers; these
functions are provided for scoring saved ``.nii.gz`` displacement fields
offline, where no GPU or MONAI pipeline is available.
"""

import numpy as np
import scipy.ndimage
from scipy.ndimage import map_coordinates

from .surface_distance import (compute_dice_coefficient,
                               compute_robust_hausdorff,
                               compute_surface_distances)

__all__ = ['jacobian_determinant', 'compute_tre', 'calc_TRE',
           'compute_dice', 'compute_hd95']


def jacobian_determinant(disp):
    """Jacobian determinant of a dense 3D displacement field.

    Args:
        disp: displacement field of shape (1, 3, H, W, D), in voxels.

    Returns:
        Jacobian determinant of shape (H-4, W-4, D-4) — a 2-voxel border is
        cropped on each side, where the finite-difference stencil is invalid.
    """
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

    return jacdet


def compute_tre(fix_lms, mov_lms, disp, spacing_fix, spacing_mov):
    """Target registration error over corresponding landmark pairs.

    Args:
        fix_lms: fixed-image landmarks, (N, 3) in voxels.
        mov_lms: moving-image landmarks, (N, 3) in voxels.
        disp: displacement field of shape (H, W, D, 3), in voxels.
        spacing_fix: voxel spacing of the fixed image (unused; kept for
            signature compatibility with the Learn2Reg reference).
        spacing_mov: voxel spacing of the moving image, used to convert the
            residual to millimetres.

    Returns:
        Per-landmark TRE of shape (N,), in millimetres.
    """
    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp

    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


def calc_TRE(dfm_lms, fx_lms, spacing_mov=1):
    """TRE between landmarks given as *label maps* rather than coordinates.

    Each non-zero label value marks one landmark; its position is taken as the
    centroid of the voxels carrying that value.

    Args:
        dfm_lms: warped moving landmark map, (H, W, D).
        fx_lms: fixed landmark map, (H, W, D), same label values.
        spacing_mov: voxel spacing, to convert the result to millimetres.

    Returns:
        Mean TRE over all landmarks, as a scalar.
    """
    x = np.linspace(0, fx_lms.shape[0] - 1, fx_lms.shape[0])
    y = np.linspace(0, fx_lms.shape[1] - 1, fx_lms.shape[1])
    z = np.linspace(0, fx_lms.shape[2] - 1, fx_lms.shape[2])
    yv, xv, zv = np.meshgrid(y, x, z)
    unique = np.unique(fx_lms)

    dfm_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (dfm_lms == unique[i]).astype('float32')
        dfm_pos[i - 1, 0] = np.sum(label * xv) / np.sum(label)
        dfm_pos[i - 1, 1] = np.sum(label * yv) / np.sum(label)
        dfm_pos[i - 1, 2] = np.sum(label * zv) / np.sum(label)

    fx_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (fx_lms == unique[i]).astype('float32')
        fx_pos[i - 1, 0] = np.sum(label * xv) / np.sum(label)
        fx_pos[i - 1, 1] = np.sum(label * yv) / np.sum(label)
        fx_pos[i - 1, 2] = np.sum(label * zv) / np.sum(label)

    return np.mean(np.sqrt(np.sum(np.power((dfm_pos - fx_pos) * spacing_mov, 2), 1)))


def compute_dice(fixed, moving, moving_warped, labels):
    """Per-label Dice, skipping labels absent from either input.

    Args:
        fixed, moving, moving_warped: integer label maps of equal shape.
        labels: label values to score.

    Returns:
        (mean_dice, per_label_dice) — labels missing from `fixed` or `moving`
        contribute NaN and are excluded from the mean.
    """
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            dice.append(np.nan)
        else:
            dice.append(compute_dice_coefficient((fixed == i), (moving_warped == i)))
    return np.nanmean(dice), dice


def compute_hd95(fixed, moving, moving_warped, labels):
    """Per-label 95th-percentile Hausdorff distance, in voxels.

    Args & returns follow :func:`compute_dice`. Isotropic unit spacing is
    assumed.
    """
    hd95 = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            hd95.append(np.nan)
        else:
            hd95.append(compute_robust_hausdorff(
                compute_surface_distances((fixed == i), (moving_warped == i), np.ones(3)), 95.))
    return np.nanmean(hd95), hd95
