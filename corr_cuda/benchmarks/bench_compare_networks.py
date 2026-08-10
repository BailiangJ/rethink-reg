from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from corr_cuda import WinCorrCuda, ref_win_corr
from models.networks.correlation import WinCorrTorch


@dataclass
class Result:
    name: str
    time_ms: float
    peak_mb: float
    output_mb: float
    max_abs_diff: float


def measure(fn: Callable[[], torch.Tensor], ref: torch.Tensor, n: int) -> Result:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    peak_mb = (mem_peak - mem_before) / 1024**2
    output_mb = out.numel() * out.element_size() / 1024**2
    max_abs_diff = (out - ref).abs().max().item()

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
    time_ms = start.elapsed_time(end) / n
    return time_ms, peak_mb, output_mb, max_abs_diff


def run_case(label: str, shape: tuple[int, ...], radius: int, ndim: int, padding: str, n: int) -> None:
    torch.manual_seed(101)
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    ref = ref_win_corr(x, y, radius=radius, ndim=ndim, scale=True, padding=padding)

    methods: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("corr_cuda", lambda: WinCorrCuda(radius=radius, ndim=ndim, padding=padding)(x, y)),
        ("local_for_ref", lambda: ref_win_corr(x, y, radius=radius, ndim=ndim, scale=True, padding=padding)),
    ]
    if padding == "constant":
        methods.extend(
            [
                ("networks_for", lambda: WinCorrTorch(radius=radius, ndim=ndim, mode="for", padding="constant").cuda()(x, y)),
                ("networks_einsum", lambda: WinCorrTorch(radius=radius, ndim=ndim, mode="einsum", padding="constant").cuda()(x, y)),
            ]
        )

    print(f"\n{label}: shape={shape}, radius={radius}, padding={padding}")
    print(f"Reference for max_abs_diff: local_for_ref / current WinCorrTorch for-loop semantics")
    print(f"{'method':<18} {'time_ms':>10} {'peak_mb':>10} {'output_mb':>10} {'max_abs_diff':>14}")
    print("-" * 78)
    for name, fn in methods:
        time_ms, peak_mb, output_mb, max_abs_diff = measure(fn, ref, n)
        print(f"{name:<18} {time_ms:>10.3f} {peak_mb:>10.1f} {output_mb:>10.1f} {max_abs_diff:>14.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", default="constant", choices=["constant", "replicate"])
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    run_case("3D small", (1, 32, 24, 28, 32), 3, 3, args.padding, args.n)
    run_case("3D radius1", (1, 32, 48, 56, 64), 1, 3, args.padding, args.n)
    run_case("2D ACDC", (1, 32, 128, 128), 1, 2, args.padding, args.n)


if __name__ == "__main__":
    main()
