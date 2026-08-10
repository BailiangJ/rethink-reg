#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
constexpr int kPaddingConstant = 0;
constexpr int kPaddingReplicate = 1;

template <typename scalar_t>
struct acc_type {
  using type = float;
};

template <>
struct acc_type<double> {
  using type = double;
};

__device__ __forceinline__ int64_t clamp_index(int64_t v, int64_t size) {
  return v < 0 ? 0 : (v >= size ? size - 1 : v);
}

template <typename scalar_t>
__global__ void corr2d_forward_kernel(const scalar_t* __restrict__ x,
                                      const scalar_t* __restrict__ y,
                                      scalar_t* __restrict__ out,
                                      int64_t total,
                                      int64_t bsz,
                                      int64_t channels,
                                      int64_t height,
                                      int64_t width,
                                      int64_t radius,
                                      int64_t win,
                                      int padding_code) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int64_t tmp = index;
    const int64_t w = tmp % width;
    tmp /= width;
    const int64_t h = tmp % height;
    tmp /= height;
    const int64_t offset_id = tmp % (win * win);
    const int64_t b = tmp / (win * win);

    const int64_t oy = offset_id / win;
    const int64_t ox = offset_id % win;
    int64_t yy = h + oy - radius;
    int64_t xx = w + ox - radius;

    bool valid = yy >= 0 && yy < height && xx >= 0 && xx < width;
    if (!valid && padding_code == kPaddingConstant) {
      out[index] = static_cast<scalar_t>(0);
      continue;
    }
    if (!valid) {
      yy = clamp_index(yy, height);
      xx = clamp_index(xx, width);
    }

    typename acc_type<scalar_t>::type acc = 0;
    const int64_t spatial = height * width;
    const int64_t x_base = b * channels * spatial + h * width + w;
    const int64_t y_base = b * channels * spatial + yy * width + xx;
    for (int64_t c = 0; c < channels; ++c) {
      acc += static_cast<typename acc_type<scalar_t>::type>(x[x_base + c * spatial]) * static_cast<typename acc_type<scalar_t>::type>(y[y_base + c * spatial]);
    }
    out[index] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void corr3d_forward_kernel(const scalar_t* __restrict__ x,
                                      const scalar_t* __restrict__ y,
                                      scalar_t* __restrict__ out,
                                      int64_t total,
                                      int64_t bsz,
                                      int64_t channels,
                                      int64_t depth,
                                      int64_t height,
                                      int64_t width,
                                      int64_t radius,
                                      int64_t win,
                                      int padding_code) {
  const int64_t offsets = win * win * win;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int64_t tmp = index;
    const int64_t w = tmp % width;
    tmp /= width;
    const int64_t h = tmp % height;
    tmp /= height;
    const int64_t d = tmp % depth;
    tmp /= depth;
    const int64_t offset_id = tmp % offsets;
    const int64_t b = tmp / offsets;

    const int64_t oz = offset_id / (win * win);
    const int64_t oy = (offset_id / win) % win;
    const int64_t ox = offset_id % win;
    int64_t zz = d + oz - radius;
    int64_t yy = h + oy - radius;
    int64_t xx = w + ox - radius;

    bool valid = zz >= 0 && zz < depth && yy >= 0 && yy < height && xx >= 0 && xx < width;
    if (!valid && padding_code == kPaddingConstant) {
      out[index] = static_cast<scalar_t>(0);
      continue;
    }
    if (!valid) {
      zz = clamp_index(zz, depth);
      yy = clamp_index(yy, height);
      xx = clamp_index(xx, width);
    }

    typename acc_type<scalar_t>::type acc = 0;
    const int64_t spatial = depth * height * width;
    const int64_t x_base = b * channels * spatial + d * height * width + h * width + w;
    const int64_t y_base = b * channels * spatial + zz * height * width + yy * width + xx;
    for (int64_t c = 0; c < channels; ++c) {
      acc += static_cast<typename acc_type<scalar_t>::type>(x[x_base + c * spatial]) * static_cast<typename acc_type<scalar_t>::type>(y[y_base + c * spatial]);
    }
    out[index] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void corr2d_backward_x_kernel(const scalar_t* __restrict__ grad_out,
                                         const scalar_t* __restrict__ y,
                                         scalar_t* __restrict__ grad_x,
                                         int64_t total,
                                         int64_t channels,
                                         int64_t height,
                                         int64_t width,
                                         int64_t radius,
                                         int64_t win,
                                         int padding_code) {
  const int64_t offsets = win * win;
  const int64_t spatial = height * width;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int64_t tmp = index;
    const int64_t w = tmp % width;
    tmp /= width;
    const int64_t h = tmp % height;
    tmp /= height;
    const int64_t c = tmp % channels;
    const int64_t b = tmp / channels;

    typename acc_type<scalar_t>::type acc = 0;
    for (int64_t offset_id = 0; offset_id < offsets; ++offset_id) {
      const int64_t oy = offset_id / win;
      const int64_t ox = offset_id % win;
      int64_t yy = h + oy - radius;
      int64_t xx = w + ox - radius;
      bool valid = yy >= 0 && yy < height && xx >= 0 && xx < width;
      if (!valid && padding_code == kPaddingConstant) {
        continue;
      }
      if (!valid) {
        yy = clamp_index(yy, height);
        xx = clamp_index(xx, width);
      }
      const int64_t go_idx = ((b * offsets + offset_id) * height + h) * width + w;
      const int64_t y_idx = (b * channels + c) * spatial + yy * width + xx;
      acc += static_cast<typename acc_type<scalar_t>::type>(grad_out[go_idx]) * static_cast<typename acc_type<scalar_t>::type>(y[y_idx]);
    }
    grad_x[index] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void corr2d_backward_y_kernel(const scalar_t* __restrict__ grad_out,
                                         const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ grad_y,
                                         int64_t total,
                                         int64_t channels,
                                         int64_t height,
                                         int64_t width,
                                         int64_t radius,
                                         int64_t win,
                                         int padding_code) {
  const int64_t offsets = win * win;
  const int64_t spatial = height * width;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int64_t tmp = index;
    const int64_t qx = tmp % width;
    tmp /= width;
    const int64_t qy = tmp % height;
    tmp /= height;
    const int64_t c = tmp % channels;
    const int64_t b = tmp / channels;

    typename acc_type<scalar_t>::type acc = 0;
    for (int64_t offset_id = 0; offset_id < offsets; ++offset_id) {
      const int64_t oy = offset_id / win;
      const int64_t ox = offset_id % win;
      if (padding_code == kPaddingConstant) {
        const int64_t h = qy - oy + radius;
        const int64_t w = qx - ox + radius;
        if (h < 0 || h >= height || w < 0 || w >= width) {
          continue;
        }
        const int64_t go_idx = ((b * offsets + offset_id) * height + h) * width + w;
        const int64_t x_idx = (b * channels + c) * spatial + h * width + w;
        acc += static_cast<typename acc_type<scalar_t>::type>(grad_out[go_idx]) * static_cast<typename acc_type<scalar_t>::type>(x[x_idx]);
      } else {
        for (int64_t h = 0; h < height; ++h) {
          const int64_t yy = clamp_index(h + oy - radius, height);
          if (yy != qy) continue;
          for (int64_t w = 0; w < width; ++w) {
            const int64_t xx = clamp_index(w + ox - radius, width);
            if (xx != qx) continue;
            const int64_t go_idx = ((b * offsets + offset_id) * height + h) * width + w;
            const int64_t x_idx = (b * channels + c) * spatial + h * width + w;
            acc += static_cast<typename acc_type<scalar_t>::type>(grad_out[go_idx]) * static_cast<typename acc_type<scalar_t>::type>(x[x_idx]);
          }
        }
      }
    }
    grad_y[index] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void corr3d_backward_x_kernel(const scalar_t* __restrict__ grad_out,
                                         const scalar_t* __restrict__ y,
                                         scalar_t* __restrict__ grad_x,
                                         int64_t total,
                                         int64_t channels,
                                         int64_t depth,
                                         int64_t height,
                                         int64_t width,
                                         int64_t radius,
                                         int64_t win,
                                         int padding_code) {
  const int64_t offsets = win * win * win;
  const int64_t spatial = depth * height * width;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int64_t tmp = index;
    const int64_t w = tmp % width;
    tmp /= width;
    const int64_t h = tmp % height;
    tmp /= height;
    const int64_t d = tmp % depth;
    tmp /= depth;
    const int64_t c = tmp % channels;
    const int64_t b = tmp / channels;

    typename acc_type<scalar_t>::type acc = 0;
    for (int64_t offset_id = 0; offset_id < offsets; ++offset_id) {
      const int64_t oz = offset_id / (win * win);
      const int64_t oy = (offset_id / win) % win;
      const int64_t ox = offset_id % win;
      int64_t zz = d + oz - radius;
      int64_t yy = h + oy - radius;
      int64_t xx = w + ox - radius;
      bool valid = zz >= 0 && zz < depth && yy >= 0 && yy < height && xx >= 0 && xx < width;
      if (!valid && padding_code == kPaddingConstant) continue;
      if (!valid) {
        zz = clamp_index(zz, depth);
        yy = clamp_index(yy, height);
        xx = clamp_index(xx, width);
      }
      const int64_t go_idx = (((b * offsets + offset_id) * depth + d) * height + h) * width + w;
      const int64_t y_idx = (b * channels + c) * spatial + zz * height * width + yy * width + xx;
      acc += static_cast<typename acc_type<scalar_t>::type>(grad_out[go_idx]) * static_cast<typename acc_type<scalar_t>::type>(y[y_idx]);
    }
    grad_x[index] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void corr3d_backward_y_kernel(const scalar_t* __restrict__ grad_out,
                                         const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ grad_y,
                                         int64_t total,
                                         int64_t channels,
                                         int64_t depth,
                                         int64_t height,
                                         int64_t width,
                                         int64_t radius,
                                         int64_t win,
                                         int padding_code) {
  const int64_t offsets = win * win * win;
  const int64_t spatial = depth * height * width;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int64_t tmp = index;
    const int64_t qx = tmp % width;
    tmp /= width;
    const int64_t qy = tmp % height;
    tmp /= height;
    const int64_t qz = tmp % depth;
    tmp /= depth;
    const int64_t c = tmp % channels;
    const int64_t b = tmp / channels;

    typename acc_type<scalar_t>::type acc = 0;
    for (int64_t offset_id = 0; offset_id < offsets; ++offset_id) {
      const int64_t oz = offset_id / (win * win);
      const int64_t oy = (offset_id / win) % win;
      const int64_t ox = offset_id % win;
      if (padding_code == kPaddingConstant) {
        const int64_t d = qz - oz + radius;
        const int64_t h = qy - oy + radius;
        const int64_t w = qx - ox + radius;
        if (d < 0 || d >= depth || h < 0 || h >= height || w < 0 || w >= width) continue;
        const int64_t go_idx = (((b * offsets + offset_id) * depth + d) * height + h) * width + w;
        const int64_t x_idx = (b * channels + c) * spatial + d * height * width + h * width + w;
        acc += static_cast<typename acc_type<scalar_t>::type>(grad_out[go_idx]) * static_cast<typename acc_type<scalar_t>::type>(x[x_idx]);
      } else {
        for (int64_t d = 0; d < depth; ++d) {
          const int64_t zz = clamp_index(d + oz - radius, depth);
          if (zz != qz) continue;
          for (int64_t h = 0; h < height; ++h) {
            const int64_t yy = clamp_index(h + oy - radius, height);
            if (yy != qy) continue;
            for (int64_t w = 0; w < width; ++w) {
              const int64_t xx = clamp_index(w + ox - radius, width);
              if (xx != qx) continue;
              const int64_t go_idx = (((b * offsets + offset_id) * depth + d) * height + h) * width + w;
              const int64_t x_idx = (b * channels + c) * spatial + d * height * width + h * width + w;
              acc += static_cast<typename acc_type<scalar_t>::type>(grad_out[go_idx]) * static_cast<typename acc_type<scalar_t>::type>(x[x_idx]);
            }
          }
        }
      }
    }
    grad_y[index] = static_cast<scalar_t>(acc);
  }
}

int64_t blocks_for(int64_t total, int threads) {
  constexpr int64_t max_blocks = 65535;
  const int64_t blocks = (total + threads - 1) / threads;
  return blocks < max_blocks ? blocks : max_blocks;
}
}  // namespace

torch::Tensor corr_forward_cuda(torch::Tensor x, torch::Tensor y, int64_t radius, int64_t ndim, int padding_code) {
  const int64_t bsz = x.size(0);
  const int64_t channels = x.size(1);
  const int64_t win = 2 * radius + 1;
  const int64_t offsets = ndim == 2 ? win * win : win * win * win;
  std::vector<int64_t> out_shape;
  if (ndim == 2) {
    out_shape = {bsz, offsets, x.size(2), x.size(3)};
  } else {
    out_shape = {bsz, offsets, x.size(2), x.size(3), x.size(4)};
  }
  auto out = torch::empty(out_shape, x.options());
  const int threads = 256;
  const int64_t total = out.numel();
  if (total == 0) return out;
  const dim3 blocks(blocks_for(total, threads));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "corr_forward_cuda", [&] {
    if (ndim == 2) {
      corr2d_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), total,
          bsz, channels, x.size(2), x.size(3), radius, win, padding_code);
    } else {
      corr3d_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), total,
          bsz, channels, x.size(2), x.size(3), x.size(4), radius, win, padding_code);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> corr_backward_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor y,
                                              int64_t radius, int64_t ndim, int padding_code,
                                              bool need_grad_x, bool need_grad_y) {
  torch::Tensor grad_x;
  torch::Tensor grad_y;
  if (need_grad_x) {
    grad_x = torch::empty_like(x);
  }
  if (need_grad_y) {
    grad_y = torch::empty_like(y);
  }
  const int threads = 256;
  const int64_t total = x.numel();
  if (total == 0) return {grad_x, grad_y};
  const dim3 blocks(blocks_for(total, threads));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t channels = x.size(1);
  const int64_t win = 2 * radius + 1;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "corr_backward_cuda", [&] {
    if (ndim == 2) {
      if (need_grad_x) {
        corr2d_backward_x_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(),
            total, channels, x.size(2), x.size(3), radius, win, padding_code);
      }
      if (need_grad_y) {
        corr2d_backward_y_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), grad_y.data_ptr<scalar_t>(),
            total, channels, x.size(2), x.size(3), radius, win, padding_code);
      }
    } else {
      if (need_grad_x) {
        corr3d_backward_x_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(),
            total, channels, x.size(2), x.size(3), x.size(4), radius, win, padding_code);
      }
      if (need_grad_y) {
        corr3d_backward_y_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), grad_y.data_ptr<scalar_t>(),
            total, channels, x.size(2), x.size(3), x.size(4), radius, win, padding_code);
      }
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_y};
}
