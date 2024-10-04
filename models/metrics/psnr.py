import torch
import math
from typing import Union, Any, Optional
from ..builder import METRICS


@METRICS.register_module('fg_psnr')
class FgPSNR:
    def __init__(self,
                 max_val: Union[int, float] = 1.0):
        self.max_val = max_val

    def _compute_metric(self,
                        y_pred: torch.Tensor,
                        y: torch.Tensor,
                        fg_mask: Optional[torch.Tensor] = None) -> Any:
        B = y_pred.shape[0]
        if fg_mask is None:
            fg_mask = torch.ones_like(y_pred, dtype=torch.float)
        else:
            if fg_mask.shape[0] == 1:
                fg_mask = fg_mask.repeat(B, 1, 1, 1, 1)
        assert fg_mask.shape == y_pred.shape

        mse_out = (fg_mask * ((y_pred - y) ** 2)).sum(dim=(2, 3, 4)) \
                  / fg_mask.sum(dim=(2, 3, 4))
        return 20 * math.log10(self.max_val) - 10 * torch.log10(mse_out)
