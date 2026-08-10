from __future__ import annotations

import argparse
import time

import torch

from corr_cuda import WinCorrCuda, ref_win_corr


def bench(fn, n=20):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n, (mem_peak - mem_before) / 1024**2


def run_case(label, shape, radius, ndim, padding):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    cuda_mod = WinCorrCuda(radius=radius, ndim=ndim, padding=padding)

    t_ref, m_ref = bench(lambda: ref_win_corr(x, y, radius=radius, ndim=ndim, padding=padding), n=5)
    torch.cuda.empty_cache()
    t_cuda, m_cuda = bench(lambda: cuda_mod(x, y), n=20)
    out_ref = ref_win_corr(x, y, radius=radius, ndim=ndim, padding=padding)
    out_cuda = cuda_mod(x, y)
    max_diff = (out_ref - out_cuda).abs().max().item()

    print(f"{label}: shape={shape}, radius={radius}, padding={padding}")
    print(f"  {'method':<16} {'time_ms':>10} {'peak_mb':>10}")
    print(f"  {'reference':<16} {t_ref:>10.2f} {m_ref:>10.1f}")
    print(f"  {'cuda':<16} {t_cuda:>10.2f} {m_cuda:>10.1f}")
    print(f"  max_abs_diff={max_diff:.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", default="constant", choices=["constant", "replicate"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    run_case("3D small", (1, 32, 24, 28, 32), 3, 3, args.padding)
    run_case("3D radius1", (1, 32, 48, 56, 64), 1, 3, args.padding)
    run_case("2D ACDC", (1, 32, 128, 128), 1, 2, args.padding)


if __name__ == "__main__":
    main()
