import torch
import torch.nn.functional as F

from . import _C


_SUPPORTED_PADDING = {"constant", "replicate"}


def _check_common(x: torch.Tensor, y: torch.Tensor, radius: int, ndim: int, padding: str) -> None:
    if not isinstance(radius, int) or radius < 0:
        raise ValueError(f"radius must be a non-negative int, got {radius!r}")
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    if padding not in _SUPPORTED_PADDING:
        raise ValueError(f"padding must be one of {_SUPPORTED_PADDING}, got {padding!r}")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.dim() != ndim + 2:
        raise ValueError(f"expected {ndim + 2}D tensors for ndim={ndim}, got x.dim()={x.dim()}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("raw_win_corr_cuda currently supports CUDA tensors only")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous for raw_win_corr_cuda; call x.contiguous() before dispatch")
    if not y.is_contiguous():
        raise ValueError("y must be contiguous for raw_win_corr_cuda; call y.contiguous() before dispatch")


class _RawWinCorrCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, radius: int, ndim: int, padding: str) -> torch.Tensor:
        _check_common(x, y, radius, ndim, padding)
        out = _C.forward(x, y, radius, ndim, padding)
        ctx.save_for_backward(x, y)
        ctx.radius = radius
        ctx.ndim = ndim
        ctx.padding = padding
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, y = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        need_x, need_y = ctx.needs_input_grad[:2]
        grad_x, grad_y = _C.backward(grad_out, x, y, ctx.radius, ctx.ndim, ctx.padding, need_x, need_y)
        return grad_x, grad_y, None, None, None


def raw_win_corr_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    radius: int,
    ndim: int,
    padding: str = "constant",
) -> torch.Tensor:
    return _RawWinCorrCuda.apply(x, y, radius, ndim, padding)


def win_corr_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    radius: int,
    ndim: int,
    norm_vectors: bool = False,
    scale: bool = True,
    padding: str = "constant",
    auto_contiguous: bool = False,
) -> torch.Tensor:
    if norm_vectors:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
    if x.dtype != y.dtype:
        common_dtype = torch.promote_types(x.dtype, y.dtype)
        x = x.to(common_dtype)
        y = y.to(common_dtype)
    if auto_contiguous:
        x = x.contiguous()
        y = y.contiguous()
    corr = raw_win_corr_cuda(x, y, radius, ndim, padding)
    return corr * (x.shape[1] ** -0.5) if scale else corr
