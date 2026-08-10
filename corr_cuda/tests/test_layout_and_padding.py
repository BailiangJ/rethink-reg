import pytest
import torch

from corr_cuda import WinCorrCuda, raw_win_corr_cuda, ref_win_corr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_raw_rejects_non_contiguous_x():
    x = torch.randn(1, 3, 5, 6, device="cuda").transpose(2, 3)
    y = torch.randn(1, 3, 6, 5, device="cuda")
    with pytest.raises(ValueError, match="x must be contiguous"):
        raw_win_corr_cuda(x, y, radius=1, ndim=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_raw_rejects_non_contiguous_y():
    x = torch.randn(1, 3, 6, 5, device="cuda")
    y = torch.randn(1, 3, 5, 6, device="cuda").transpose(2, 3)
    with pytest.raises(ValueError, match="y must be contiguous"):
        raw_win_corr_cuda(x, y, radius=1, ndim=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_module_auto_contiguous_matches_reference():
    x_base = torch.randn(1, 3, 5, 6, device="cuda")
    y_base = torch.randn(1, 3, 5, 6, device="cuda")
    x = x_base.transpose(2, 3)
    y = y_base.transpose(2, 3)

    out_ref = ref_win_corr(x, y, radius=1, ndim=2, padding="constant")
    out_cuda = WinCorrCuda(radius=1, ndim=2, padding="constant", auto_contiguous=True)(x, y)

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_module_default_rejects_non_contiguous():
    x = torch.randn(1, 3, 5, 6, device="cuda").transpose(2, 3)
    y = torch.randn(1, 3, 6, 5, device="cuda")
    with pytest.raises(ValueError, match="x must be contiguous"):
        WinCorrCuda(radius=1, ndim=2)(x, y)
