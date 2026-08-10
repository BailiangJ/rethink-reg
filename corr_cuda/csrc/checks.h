#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous; call .contiguous() before dispatch")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

inline int padding_to_code(const std::string& padding) {
  if (padding == "constant") {
    return 0;
  }
  if (padding == "replicate") {
    return 1;
  }
  TORCH_CHECK(false, "padding must be 'constant' or 'replicate', got '", padding, "'");
}
