import pytest
import torch

from corr_cuda import WinCorrCuda, ref_win_corr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 3, 8, 32])
@pytest.mark.parametrize("padding", ["constant", "replicate"])
@pytest.mark.parametrize("scale", [False, True])
def test_forward_2d(radius, channels, padding, scale):
    torch.manual_seed(7)
    x = torch.randn(2, channels, 17, 19, device="cuda")
    y = torch.randn(2, channels, 17, 19, device="cuda")

    out_ref = ref_win_corr(x, y, radius=radius, ndim=2, scale=scale, padding=padding)
    out_cuda = WinCorrCuda(radius=radius, ndim=2, scale=scale, padding=padding)(x, y)

    assert out_cuda.shape == (2, (2 * radius + 1) ** 2, 17, 19)
    assert out_cuda.is_contiguous()
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_2d_norm_vectors():
    torch.manual_seed(11)
    x = torch.randn(1, 8, 11, 13, device="cuda")
    y = torch.randn(1, 8, 11, 13, device="cuda")

    out_ref = ref_win_corr(x, y, radius=2, ndim=2, norm_vectors=True, scale=True, padding="constant")
    out_cuda = WinCorrCuda(radius=2, ndim=2, norm_vectors=True, scale=True, padding="constant")(x, y)

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-5)
