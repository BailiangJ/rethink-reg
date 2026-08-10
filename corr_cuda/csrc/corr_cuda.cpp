#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "checks.h"

namespace {
constexpr int kPaddingConstant = 0;
constexpr int kPaddingReplicate = 1;

void check_common(const torch::Tensor& x, const torch::Tensor& y, int64_t radius, int64_t ndim, const std::string& padding) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  TORCH_CHECK(x.device() == y.device(), "x and y must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have identical shapes");
  TORCH_CHECK(ndim == 2 || ndim == 3, "ndim must be 2 or 3, got ", ndim);
  TORCH_CHECK(x.dim() == ndim + 2, "expected input rank ", ndim + 2, " for ndim=", ndim, ", got ", x.dim());
  TORCH_CHECK(radius >= 0, "radius must be non-negative, got ", radius);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble || x.scalar_type() == at::kHalf,
              "only float16, float32, and float64 are currently supported");
  padding_to_code(padding);
}
}  // namespace

torch::Tensor corr_forward_cuda(torch::Tensor x, torch::Tensor y, int64_t radius, int64_t ndim, int padding_code);
std::vector<torch::Tensor> corr_backward_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor y,
                                              int64_t radius, int64_t ndim, int padding_code,
                                              bool need_grad_x, bool need_grad_y);

torch::Tensor forward(torch::Tensor x, torch::Tensor y, int64_t radius, int64_t ndim, const std::string& padding) {
  check_common(x, y, radius, ndim, padding);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int padding_code = padding_to_code(padding);
  return corr_forward_cuda(x, y, radius, ndim, padding_code);
}

std::vector<torch::Tensor> backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor y,
                                    int64_t radius, int64_t ndim, const std::string& padding,
                                    bool need_grad_x, bool need_grad_y) {
  check_common(x, y, radius, ndim, padding);
  CHECK_INPUT(grad_out);
  TORCH_CHECK(grad_out.device() == x.device(), "grad_out must be on the same CUDA device as x and y");
  TORCH_CHECK(grad_out.scalar_type() == x.scalar_type(), "grad_out must have the same dtype as x and y");
  const int64_t k = 2 * radius + 1;
  const int64_t offsets = ndim == 2 ? k * k : k * k * k;
  TORCH_CHECK(grad_out.size(0) == x.size(0), "grad_out batch size mismatch");
  TORCH_CHECK(grad_out.size(1) == offsets, "grad_out offset channel mismatch");
  for (int64_t i = 0; i < ndim; ++i) {
    TORCH_CHECK(grad_out.size(i + 2) == x.size(i + 2), "grad_out spatial shape mismatch");
  }
  if (!need_grad_x && !need_grad_y) {
    return {torch::Tensor(), torch::Tensor()};
  }
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int padding_code = padding_to_code(padding);
  return corr_backward_cuda(grad_out, x, y, radius, ndim, padding_code, need_grad_x, need_grad_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "local window correlation forward");
  m.def("backward", &backward, "local window correlation backward");
}
