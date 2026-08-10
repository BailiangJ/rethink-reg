from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from corr_cuda import WinCorrCuda, ref_win_corr
from models.networks.correlation import WinCorrTorch


@dataclass(frozen=True)
class ShapeCase:
    name: str
    spatial: tuple[int, int, int]


SHAPE_SETS = {
    "quick": [
        ShapeCase("small", (10, 12, 14)),
        ShapeCase("mid", (20, 24, 28)),
    ],
    "training": [
        ShapeCase("lvl4", (10, 12, 14)),
        ShapeCase("lvl3", (20, 24, 28)),
        ShapeCase("lvl2", (40, 48, 56)),
        ShapeCase("lvl1", (80, 96, 112)),
    ],
}


def parse_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_strings(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def sync_measure(fn: Callable[[], torch.Tensor], n: int) -> tuple[float, float, torch.Tensor]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n, (mem_peak - mem_before) / 1024**2, out


def make_fwd(method: str, x: torch.Tensor, y: torch.Tensor, radius: int, ndim: int, padding: str) -> Callable[[], torch.Tensor]:
    if method == "corr_cuda":
        mod = WinCorrCuda(radius=radius, ndim=ndim, padding=padding).cuda()
        return lambda: mod(x, y)
    if method == "ref_for":
        return lambda: ref_win_corr(x, y, radius=radius, ndim=ndim, padding=padding)
    if method == "networks_for":
        mod = WinCorrTorch(radius=radius, ndim=ndim, mode="for", padding=padding).cuda()
        return lambda: mod(x, y)
    if method == "networks_einsum":
        mod = WinCorrTorch(radius=radius, ndim=ndim, mode="einsum", padding=padding).cuda()
        return lambda: mod(x, y)
    raise ValueError(f"unknown method: {method}")


def make_fwd_bwd(
    method: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    radius: int,
    ndim: int,
    padding: str,
    pass_type: str,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        x_requires_grad = pass_type in {"fwd_bwd_both", "fwd_bwd_x_only"}
        y_requires_grad = pass_type in {"fwd_bwd_both", "fwd_bwd_y_only"}
        x = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=x_requires_grad)
        y = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=y_requires_grad)
        out = make_fwd(method, x, y, radius, ndim, padding)()
        out.float().square().mean().backward()
        return out

    return run


def maybe_methods(padding: str, include_ref: bool, include_einsum: bool) -> list[str]:
    methods = ["corr_cuda"]
    if include_ref:
        methods.extend(["ref_for", "networks_for"])
    if include_einsum:
        methods.append("networks_einsum")
    if padding == "replicate":
        methods = [m for m in methods if m != "networks_einsum"]
    return methods


def write_row(writer: csv.DictWriter | None, row: dict) -> None:
    if writer is not None:
        writer.writerow(row)


def print_row(row: dict) -> None:
    print(
        f"{row['shape_name']:<6} C={row['channels']:<3} R={row['radius']} "
        f"{row['dtype']:<7} {row['padding']:<9} {row['pass_type']:<13} {row['method']:<15} "
        f"time={row['time_ms']:>8.3f}ms peak={row['peak_mem_mb']:>8.1f}MB "
        f"out/s={row['output_elems_per_sec']:>10.3e} dot/s={row['effective_dot_products_per_sec']:>10.3e}"
    )


def run_matrix(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    channels = parse_ints(args.channels)
    radii = parse_ints(args.radii)
    dtypes = parse_strings(args.dtypes)
    paddings = parse_strings(args.paddings)
    pass_types = parse_strings(args.pass_types)
    shapes = SHAPE_SETS[args.shape_set]
    methods_by_padding = {
        padding: maybe_methods(padding, args.include_ref, args.include_einsum) for padding in paddings
    }

    csv_file = None
    writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        fieldnames = [
            "shape_name",
            "batch",
            "channels",
            "depth",
            "height",
            "width",
            "radius",
            "ndim",
            "dtype",
            "padding",
            "pass_type",
            "method",
            "time_ms",
            "peak_mem_mb",
            "output_mb",
            "output_elems_per_sec",
            "effective_dot_products_per_sec",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for shape_case in shapes:
            for c in channels:
                for radius in radii:
                    offsets = (2 * radius + 1) ** 3
                    spatial_numel = shape_case.spatial[0] * shape_case.spatial[1] * shape_case.spatial[2]
                    output_elems = args.batch * offsets * spatial_numel
                    effective_dots = output_elems * c
                    for dtype_name in dtypes:
                        dtype = torch.float16 if dtype_name == "float16" else torch.float32
                        shape = (args.batch, c, *shape_case.spatial)
                        for padding in paddings:
                            for pass_type in pass_types:
                                for method in methods_by_padding[padding]:
                                    if method == "networks_einsum" and shape_case.name == "lvl1":
                                        continue
                                    torch.manual_seed(args.seed)
                                    if pass_type == "fwd":
                                        x = torch.randn(*shape, device="cuda", dtype=dtype)
                                        y = torch.randn(*shape, device="cuda", dtype=dtype)
                                        fn = make_fwd(method, x, y, radius, 3, padding)
                                    else:
                                        fn = make_fwd_bwd(method, shape, dtype, radius, 3, padding, pass_type)
                                    try:
                                        time_ms, peak_mem_mb, out = sync_measure(fn, args.iters)
                                    except torch.cuda.OutOfMemoryError:
                                        torch.cuda.empty_cache()
                                        row = {
                                            "shape_name": shape_case.name,
                                            "batch": args.batch,
                                            "channels": c,
                                            "depth": shape_case.spatial[0],
                                            "height": shape_case.spatial[1],
                                            "width": shape_case.spatial[2],
                                            "radius": radius,
                                            "ndim": 3,
                                            "dtype": dtype_name,
                                            "padding": padding,
                                            "pass_type": pass_type,
                                            "method": method,
                                            "time_ms": float("nan"),
                                            "peak_mem_mb": float("nan"),
                                            "output_mb": float("nan"),
                                            "output_elems_per_sec": float("nan"),
                                            "effective_dot_products_per_sec": float("nan"),
                                        }
                                        print_row(row)
                                        write_row(writer, row)
                                        continue
                                    time_sec = time_ms / 1000.0
                                    row = {
                                        "shape_name": shape_case.name,
                                        "batch": args.batch,
                                        "channels": c,
                                        "depth": shape_case.spatial[0],
                                        "height": shape_case.spatial[1],
                                        "width": shape_case.spatial[2],
                                        "radius": radius,
                                        "ndim": 3,
                                        "dtype": dtype_name,
                                        "padding": padding,
                                        "pass_type": pass_type,
                                        "method": method,
                                        "time_ms": time_ms,
                                        "peak_mem_mb": peak_mem_mb,
                                        "output_mb": out.numel() * out.element_size() / 1024**2,
                                        "output_elems_per_sec": output_elems / time_sec,
                                        "effective_dot_products_per_sec": effective_dots / time_sec,
                                    }
                                    print_row(row)
                                    write_row(writer, row)
    finally:
        if csv_file is not None:
            csv_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-set", choices=sorted(SHAPE_SETS), default="quick")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", default="8,16,32,64,96,128")
    parser.add_argument("--radii", default="1,2,3")
    parser.add_argument("--dtypes", default="float16,float32")
    parser.add_argument("--paddings", default="constant,replicate")
    parser.add_argument("--pass-types", default="fwd,fwd_bwd_both,fwd_bwd_x_only,fwd_bwd_y_only")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--include-ref", action="store_true")
    parser.add_argument("--include-einsum", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    run_matrix(args)


if __name__ == "__main__":
    main()
