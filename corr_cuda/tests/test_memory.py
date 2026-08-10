import pytest
import torch

from corr_cuda import WinCorrCuda, ref_win_corr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_smoke_3d_less_than_reference_unfold_like_intermediate():
    torch.manual_seed(37)
    x = torch.randn(1, 8, 8, 9, 10, device="cuda")
    y = torch.randn(1, 8, 8, 9, 10, device="cuda")
    corr = WinCorrCuda(radius=2, ndim=3)(x, y)
    assert corr.shape == (1, 125, 8, 9, 10)
    torch.testing.assert_close(corr, ref_win_corr(x, y, radius=2, ndim=3), rtol=1e-4, atol=1e-5)
