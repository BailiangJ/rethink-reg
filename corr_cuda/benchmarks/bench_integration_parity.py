from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.networks.matching import CorrelationMatching
from models.networks.correlation import WinCorrTorch


def compare_outputs(name: str, out_cuda, out_ref, rtol=2e-3, atol=2e-3):
    if isinstance(out_cuda, tuple):
        for i, (a, b) in enumerate(zip(out_cuda, out_ref)):
            diff = (a - b).abs().max().item()
            print(f"{name}[{i}] max_abs_diff={diff:.6g}")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    else:
        diff = (out_cuda - out_ref).abs().max().item()
        print(f"{name} max_abs_diff={diff:.6g}")
        torch.testing.assert_close(out_cuda, out_ref, rtol=rtol, atol=atol)


def compare_grads(fn_ref: Callable, fn_cuda: Callable, x: torch.Tensor, y: torch.Tensor, label: str) -> None:
    xr = x.detach().clone().requires_grad_(True)
    yr = y.detach().clone().requires_grad_(True)
    xc = x.detach().clone().requires_grad_(True)
    yc = y.detach().clone().requires_grad_(True)
    out_ref = fn_ref(xr, yr)
    out_cuda = fn_cuda(xc, yc)
    compare_outputs(label, out_cuda, out_ref)
    loss_ref = sum(o.float().square().mean() for o in out_ref) if isinstance(out_ref, tuple) else out_ref.float().square().mean()
    loss_cuda = sum(o.float().square().mean() for o in out_cuda) if isinstance(out_cuda, tuple) else out_cuda.float().square().mean()
    loss_ref.backward()
    loss_cuda.backward()
    compare_outputs(f"{label}.grad_x", xc.grad, xr.grad)
    compare_outputs(f"{label}.grad_y", yc.grad, yr.grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--ndim", type=int, choices=[2, 3], default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    shape = (1, 8, 9, 11) if args.ndim == 2 else (1, 8, 5, 6, 7)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    y = torch.randn(*shape, device="cuda", dtype=dtype)
    spatial = shape[2:]

    compare_grads(
        lambda a, b: WinCorrTorch(radius=1, ndim=args.ndim, mode="for").cuda()(a, b),
        lambda a, b: WinCorrTorch(radius=1, ndim=args.ndim, mode="cuda").cuda()(a, b),
        x,
        y,
        "WinCorrTorch",
    )
    compare_grads(
        lambda a, b: CorrelationMatching(spatial, radius=1, corr_mode="for", norm_vectors=True, scale=True).cuda()(a, b),
        lambda a, b: CorrelationMatching(spatial, radius=1, corr_mode="cuda", norm_vectors=True, scale=True).cuda()(a, b),
        x,
        y,
        "CorrelationMatching",
    )


if __name__ == "__main__":
    main()
