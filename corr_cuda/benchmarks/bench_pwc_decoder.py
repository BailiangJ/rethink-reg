from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.networks.vxm_pwc import WarpCorrPyramidalDecoder


def make_decoder(mode: str, dtype: torch.dtype) -> WarpCorrPyramidalDecoder:
    decoder = WarpCorrPyramidalDecoder(
        image_size=[32, 40, 48],
        spatial_dims=3,
        skip_channels=[16, 16],
        out_channels=[16, 16],
        corr_radius=[3, 3],
        out_indices=[2, 1],
        corr_mode=mode,
        block_config=dict(
            kernel_size=3,
            res_skip=False,
            up_transp_conv=True,
            transp_bias=False,
            upsample_kernel_size=2,
            bias=True,
            norm_name=("INSTANCE", {"affine": False}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    ).cuda()
    return decoder.to(dtype=dtype)


def measure(fn: Callable[[], list[torch.Tensor]], n: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    outs = fn()
    loss = sum(o.float().square().mean() for o in outs)
    loss.backward()
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()

    for _ in range(3):
        outs = fn()
        loss = sum(o.float().square().mean() for o in outs)
        loss.backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        outs = fn()
        loss = sum(o.float().square().mean() for o in outs)
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n, (mem_peak - mem_before) / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    torch.manual_seed(79)
    src_feats = [
        torch.randn(1, 16, 8, 10, 12, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(1, 16, 16, 20, 24, device="cuda", dtype=dtype, requires_grad=True),
    ]
    tgt_feats = [
        torch.randn(1, 16, 8, 10, 12, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(1, 16, 16, 20, 24, device="cuda", dtype=dtype, requires_grad=True),
    ]
    dec_for = make_decoder("for", dtype)
    dec_cuda = make_decoder("cuda", dtype)
    dec_cuda.load_state_dict(dec_for.state_dict())

    outs_for = dec_for(src_feats, tgt_feats)
    outs_cuda = dec_cuda(src_feats, tgt_feats)
    for i, (a, b) in enumerate(zip(outs_for, outs_cuda)):
        diff = (a - b).abs().max().item()
        print(f"flow[{i}] max_abs_diff={diff:.6g}")
        torch.testing.assert_close(b, a, rtol=3e-3, atol=3e-3)

    def run_for():
        dec_for.zero_grad(set_to_none=True)
        return dec_for(src_feats, tgt_feats)

    def run_cuda():
        dec_cuda.zero_grad(set_to_none=True)
        return dec_cuda(src_feats, tgt_feats)

    t_for, m_for = measure(run_for, args.n)
    t_cuda, m_cuda = measure(run_cuda, args.n)
    print(f"{'method':<12} {'fwd_bwd_ms':>12} {'peak_mb':>10}")
    print(f"{'for':<12} {t_for:>12.3f} {m_for:>10.1f}")
    print(f"{'cuda':<12} {t_cuda:>12.3f} {m_cuda:>10.1f}")


if __name__ == "__main__":
    main()
