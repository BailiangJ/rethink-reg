from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from corr_cuda import WinCorrCuda, ref_win_corr
from models.networks.correlation import WinCorrTorch


def measure(fn: Callable[[], torch.Tensor], n: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()

    for _ in range(3):
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


def make_forward_backward(fn_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape: tuple[int, ...]):
    def run():
        x = torch.randn(*shape, device="cuda", requires_grad=True)
        y = torch.randn(*shape, device="cuda", requires_grad=True)
        out = fn_factory(x, y)
        loss = out.square().mean()
        loss.backward()
        return out

    return run


def run_case(label: str, shape: tuple[int, ...], radius: int, ndim: int, padding: str, n: int) -> None:
    methods = [
        ("corr_cuda", lambda x, y: WinCorrCuda(radius=radius, ndim=ndim, padding=padding)(x, y)),
        ("local_for_ref", lambda x, y: ref_win_corr(x, y, radius=radius, ndim=ndim, padding=padding)),
    ]
    if padding == "constant":
        methods.extend(
            [
                ("networks_for", lambda x, y: WinCorrTorch(radius=radius, ndim=ndim, mode="for", padding="constant").cuda()(x, y)),
                ("networks_einsum", lambda x, y: WinCorrTorch(radius=radius, ndim=ndim, mode="einsum", padding="constant").cuda()(x, y)),
            ]
        )

    print(f"\n{label}: shape={shape}, radius={radius}, padding={padding}")
    print(f"{'method':<18} {'fwd_bwd_ms':>12} {'peak_mb':>10}")
    print("-" * 44)
    for name, fn_factory in methods:
        time_ms, peak_mb = measure(make_forward_backward(fn_factory, shape), n)
        print(f"{name:<18} {time_ms:>12.3f} {peak_mb:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", default="constant", choices=["constant", "replicate"])
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    run_case("3D small", (1, 16, 16, 18, 20), 3, 3, args.padding, args.n)
    run_case("3D boundary", (1, 3, 5, 6, 7), 2, 3, args.padding, args.n)
    run_case("2D boundary", (1, 3, 5, 6), 2, 2, args.padding, args.n)


if __name__ == "__main__":
    main()
