import pytest
import torch

from corr_cuda import WinCorrCuda, ref_win_corr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ndim,shape", [(2, (1, 8, 9, 11)), (3, (1, 8, 5, 6, 7))])
@pytest.mark.parametrize("padding", ["constant", "replicate"])
def test_float16_forward_backward(ndim, shape, padding):
    torch.manual_seed(53)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
    y = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
    xr = x.detach().clone().float().requires_grad_(True)
    yr = y.detach().clone().float().requires_grad_(True)

    out = WinCorrCuda(radius=1, ndim=ndim, padding=padding, scale=True)(x, y)
    ref = ref_win_corr(xr, yr, radius=1, ndim=ndim, padding=padding, scale=True).half()
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref_win_corr(xr, yr, radius=1, ndim=ndim, padding=padding, scale=True).backward(grad.float())
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(y.grad.float(), yr.grad, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autocast_float16_output():
    torch.manual_seed(59)
    x = torch.randn(1, 8, 7, 9, device="cuda", dtype=torch.float32, requires_grad=True)
    y = torch.randn(1, 8, 7, 9, device="cuda", dtype=torch.float32, requires_grad=True)
    mod = WinCorrCuda(radius=1, ndim=2, padding="constant")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = mod(x, y)
        loss = out.square().mean()
    loss.backward()

    assert out.dtype == torch.float32
    assert x.grad is not None
    assert y.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autocast_with_half_inputs():
    torch.manual_seed(61)
    x = torch.randn(1, 8, 7, 9, device="cuda", dtype=torch.float16, requires_grad=True)
    y = torch.randn(1, 8, 7, 9, device="cuda", dtype=torch.float16, requires_grad=True)
    mod = WinCorrCuda(radius=1, ndim=2, padding="replicate")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = mod(x, y)
        loss = out.float().square().mean()
    loss.backward()

    assert out.dtype == torch.float16
    assert x.grad is not None
    assert y.grad is not None
