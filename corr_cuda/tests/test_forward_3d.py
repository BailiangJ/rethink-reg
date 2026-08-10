import pytest
import torch

from corr_cuda import WinCorrCuda, ref_win_corr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 3, 8])
@pytest.mark.parametrize("padding", ["constant", "replicate"])
@pytest.mark.parametrize("scale", [False, True])
def test_forward_3d(radius, channels, padding, scale):
    torch.manual_seed(13)
    x = torch.randn(1, channels, 7, 9, 11, device="cuda")
    y = torch.randn(1, channels, 7, 9, 11, device="cuda")

    out_ref = ref_win_corr(x, y, radius=radius, ndim=3, scale=scale, padding=padding)
    out_cuda = WinCorrCuda(radius=radius, ndim=3, scale=scale, padding=padding)(x, y)

    assert out_cuda.shape == (1, (2 * radius + 1) ** 3, 7, 9, 11)
    assert out_cuda.is_contiguous()
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_3d_norm_vectors():
    torch.manual_seed(17)
    x = torch.randn(1, 8, 5, 6, 7, device="cuda")
    y = torch.randn(1, 8, 5, 6, 7, device="cuda")

    out_ref = ref_win_corr(x, y, radius=2, ndim=3, norm_vectors=True, scale=True, padding="replicate")
    out_cuda = WinCorrCuda(radius=2, ndim=3, norm_vectors=True, scale=True, padding="replicate")(x, y)

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-5)
