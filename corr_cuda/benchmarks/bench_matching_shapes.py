from __future__ import annotations

import torch
import torch.nn as nn

from corr_cuda import WinCorrCuda, ref_win_corr


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(41)
    radius = 3
    ndim = 3
    tgt_feat = torch.randn(1, 32, 12, 14, 16, device="cuda", requires_grad=True)
    src_feat = torch.randn(1, 32, 12, 14, 16, device="cuda", requires_grad=True)
    conv = nn.Conv3d((2 * radius + 1) ** 3 + 32, 16, kernel_size=1).cuda()

    corr_ref = ref_win_corr(tgt_feat, src_feat, radius=radius, ndim=ndim)
    corr_cuda = WinCorrCuda(radius=radius, ndim=ndim)(tgt_feat, src_feat)
    skip_ref = torch.cat([corr_ref, tgt_feat], dim=1)
    skip_cuda = torch.cat([corr_cuda, tgt_feat], dim=1)

    torch.testing.assert_close(skip_cuda, skip_ref, rtol=1e-4, atol=1e-5)
    loss = conv(skip_cuda).square().mean()
    loss.backward()
    print("PWC-style skip smoke passed")


if __name__ == "__main__":
    main()
