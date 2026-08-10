import pytest
import torch

from corr_cuda import WinCorrCuda, ref_win_corr


def _compare_grads(x, y, radius, ndim, padding, scale, norm_vectors=False):
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    x_cuda = x.detach().clone().requires_grad_(True)
    y_cuda = y.detach().clone().requires_grad_(True)

    out_ref = ref_win_corr(x_ref, y_ref, radius=radius, ndim=ndim, norm_vectors=norm_vectors, scale=scale, padding=padding)
    out_cuda = WinCorrCuda(radius=radius, ndim=ndim, norm_vectors=norm_vectors, scale=scale, padding=padding)(x_cuda, y_cuda)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_cuda.backward(grad)

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x_cuda.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(y_cuda.grad, y_ref.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("padding", ["constant", "replicate"])
@pytest.mark.parametrize("scale", [False, True])
def test_backward_3d_float32(padding, scale):
    torch.manual_seed(29)
    x = torch.randn(1, 3, 5, 6, 7, device="cuda", requires_grad=True)
    y = torch.randn(1, 3, 5, 6, 7, device="cuda", requires_grad=True)
    _compare_grads(x, y, radius=2, ndim=3, padding=padding, scale=scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backward_3d_norm_vectors():
    torch.manual_seed(31)
    x = torch.randn(1, 4, 4, 5, 6, device="cuda", requires_grad=True)
    y = torch.randn(1, 4, 4, 5, 6, device="cuda", requires_grad=True)
    _compare_grads(x, y, radius=1, ndim=3, padding="replicate", scale=True, norm_vectors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("x_requires_grad,y_requires_grad", [(True, False), (False, True)])
def test_backward_3d_skips_unused_input_grad(x_requires_grad, y_requires_grad):
    torch.manual_seed(71)
    x = torch.randn(1, 3, 4, 5, 6, device="cuda", requires_grad=x_requires_grad)
    y = torch.randn(1, 3, 4, 5, 6, device="cuda", requires_grad=y_requires_grad)
    xr = x.detach().clone().requires_grad_(x_requires_grad)
    yr = y.detach().clone().requires_grad_(y_requires_grad)

    out_cuda = WinCorrCuda(radius=2, ndim=3, padding="replicate", scale=True)(x, y)
    out_ref = ref_win_corr(xr, yr, radius=2, ndim=3, padding="replicate", scale=True)
    grad = torch.randn_like(out_cuda)
    out_cuda.backward(grad)
    out_ref.backward(grad)

    if x_requires_grad:
        torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=1e-5)
    else:
        assert x.grad is None
    if y_requires_grad:
        torch.testing.assert_close(y.grad, yr.grad, rtol=1e-4, atol=1e-5)
    else:
        assert y.grad is None
