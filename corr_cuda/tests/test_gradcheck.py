import pytest
import torch

from corr_cuda import raw_win_corr_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("padding", ["constant", "replicate"])
def test_gradcheck_2d(padding):
    torch.manual_seed(43)
    x = torch.randn(1, 2, 4, 5, device="cuda", dtype=torch.double, requires_grad=True)
    y = torch.randn(1, 2, 4, 5, device="cuda", dtype=torch.double, requires_grad=True)

    def fn(a, b):
        return raw_win_corr_cuda(a, b, radius=1, ndim=2, padding=padding)

    assert torch.autograd.gradcheck(fn, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("padding", ["constant", "replicate"])
def test_gradcheck_3d(padding):
    torch.manual_seed(47)
    x = torch.randn(1, 2, 3, 4, 5, device="cuda", dtype=torch.double, requires_grad=True)
    y = torch.randn(1, 2, 3, 4, 5, device="cuda", dtype=torch.double, requires_grad=True)

    def fn(a, b):
        return raw_win_corr_cuda(a, b, radius=1, ndim=3, padding=padding)

    assert torch.autograd.gradcheck(fn, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)
