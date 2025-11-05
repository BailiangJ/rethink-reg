import numpy as np
import scipy
from typing import Optional
from ..builder import METRICS


@METRICS.register_module('sdlogjac2d')
class SDlogDetJac2D:
    def __call__(self, disp: np.ndarray, fg_mask: Optional[np.ndarray] = None):
        '''
        Args:
            disp: displacement field of shape (B, 2, H, W)
            fg_mask: foreground mask of shape (1,1,H,W) or (B,1,H,W)
        '''
        B, _, H, W = disp.shape

        if fg_mask is None:
            fg_mask = np.ones((B, 1, H, W), dtype=np.float32)
        else:
            if fg_mask.shape[0] == 1:
                fg_mask = fg_mask.repeat(B, axis=0)
            fg_mask = fg_mask.astype(np.float32)
        fg_mask = fg_mask.squeeze(1)
        assert fg_mask.shape == (B, H, W)

        gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1)
        grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3)

        # Compute the gradient of the displacement field
        # gradx_disp: (B, 2, H, W)
        gradx_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :], gradx, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :], gradx, mode='constant', cval=0.0)],
            axis=1)

        # grady_disp: (B, 2, H, W)
        grady_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :], grady, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :], grady, mode='constant', cval=0.0)],
            axis=1)

        # grad_disp: (B, 2, 2, H, W)
        grad_disp = np.stack([gradx_disp, grady_disp], 1)

        # jacobian: (B, 2, 2, H, W)
        jacobian = grad_disp + np.eye(2, 2).reshape(1, 2, 2, 1, 1)
        jacobian = jacobian[:, :, :, 2:-2, 2:-2]
        if fg_mask is not None:
            fg_mask = fg_mask[:, 2:-2, 2:-2]
        # jacdet: (B, H, W)
        jacdet = jacobian[:, 0, 0, ...] * jacobian[:, 1, 1, ...] - jacobian[:, 0, 1, ...] * jacobian[:, 1, 0, ...]

        non_pos_jacdet = np.sum((jacdet <= 0) * fg_mask, axis=(1, 2))

        log_jacdet = np.log((jacdet + 3).clip(0.000000001, 1000000000))

        return np.std(log_jacdet, axis=(1, 2)), non_pos_jacdet


@METRICS.register_module('jacdet2d')
class JacDet2D:
    def __call__(self, disp: np.ndarray):
        '''
        Args:
            disp: displacement field of shape (B, 2, H, W)
        '''
        B, _, H, W = disp.shape

        gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1)
        grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3)

        # Compute the gradient of the displacement field
        # gradx_disp: (B, 2, H, W)
        gradx_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :], gradx, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :], gradx, mode='constant', cval=0.0)],
            axis=1)

        # grady_disp: (B, 2, H, W)
        grady_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :], grady, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :], grady, mode='constant', cval=0.0)],
            axis=1)

        # grad_disp: (B, 2, 2, H, W)
        grad_disp = np.stack([gradx_disp, grady_disp], 1)

        # jacobian: (B, 2, 2, H, W)
        jacobian = grad_disp + np.eye(2, 2).reshape(1, 2, 2, 1, 1)
        jacobian = jacobian[:, :, :, 2:-2, 2:-2]
        # jacdet: (B, H, W)
        jacdet = jacobian[:, 0, 0, ...] * jacobian[:, 1, 1, ...] - jacobian[:, 0, 1, ...] * jacobian[:, 1, 0, ...]

        return jacdet
