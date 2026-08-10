# CUDA window correlation

Standalone PyTorch CUDA extension for local window correlation. It computes the raw local cost volume used by correlation-based registration decoders without materializing an unfolded `B*C*N*K^ndim` patch tensor.

## Build

Build from this directory with the active PyTorch environment:

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` is important in this repo because the active PyTorch/CUDA build must match the local CUDA toolkit.

## Basic usage

```python
from corr_cuda import WinCorrCuda, win_corr_cuda

corr = WinCorrCuda(radius=3, ndim=3, padding="constant")(x, y)
```

Low-level API:

- `raw_win_corr_cuda(...)` calls the custom autograd op directly and requires contiguous CUDA tensors.
- `win_corr_cuda(...)` applies optional normalization/scaling and can promote mixed dtypes.
- `WinCorrCuda(auto_contiguous=True)` can be used for drop-in experiments with non-contiguous tensors.

Supported dtypes:

- `torch.float16`
- `torch.float32`
- `torch.float64`

`float16` is intended for AMP/autocast training. The kernels accumulate half inputs in `float` before casting back to half output.

## Optional integration

The main codebase does not depend on this package by default. To enable it after building:

```python
WinCorrTorch(radius=1, ndim=3, mode="cuda")
```

For PWC configs, set the decoder correlation mode:

```python
decoder_cfg=dict(
    ...,
    corr_mode="cuda",
)
```

Defaults remain unchanged (`mode="for"` / `corr_mode="for"`) unless CUDA mode is explicitly requested.

## Debugging

Set `CORR_CUDA_DEBUG=1` to print one debug line per CUDA correlation module:

```bash
CORR_CUDA_DEBUG=1 python tasks/brainmri/train.py --train-config CONFIG.py --random-seed 2023
```

The debug line includes input shape, dtype, radius, padding, scaling, and current max CUDA memory. This is useful for confirming that AMP training is actually reaching the CUDA backend.

## Validation commands

Run these commands from the repository root:

```bash
python -m compileall -q models/networks/correlation.py models/networks/vxm_pwc.py corr_cuda

pip install -e ./corr_cuda --no-build-isolation

python corr_cuda/benchmarks/bench_integration_parity.py --dtype float32 --ndim 3
python corr_cuda/benchmarks/bench_integration_parity.py --dtype float16 --ndim 3
python corr_cuda/benchmarks/bench_pwc_decoder.py --dtype float16
python corr_cuda/benchmarks/bench_matrix.py --shape-set quick --channels 8,32,96 --radii 1,3 --dtypes float16
```

Direct `torch.autograd.gradcheck` passed for 2D/3D `float64` inputs with both `constant` and `replicate` padding.

## Training smoke commands

Run these commands from the repository root after setting `DATA_ROOT`:

PWC:

```bash
CORR_CUDA_DEBUG=1 python tasks/brainmri/train.py \
  --train-config corr_cuda/benchmarks/lumir_pwc_cuda_smoke_cfg.py \
  --random-seed 2023
```

These smoke configurations use reduced-channel LUMIR inputs; run them after setting `DATA_ROOT` and installing the task dependencies.

## Current limitations

- CUDA tensors only.
- Optional package: must be built before `mode="cuda"` is used.
- No custom autocast registration yet; inputs already in `float16` run as half, while `float32` inputs remain float unless caller/autocast produced half tensors.
- Mixed dtypes are promoted in the Python wrapper.
- The kernel uses a correctness-first one-thread-per-output design; it does not use shared memory, vectorized loads, or cooperative channel reductions yet.
- Replicate-padding backward, especially `grad_y`, is much slower than constant-padding backward.
