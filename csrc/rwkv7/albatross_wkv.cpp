// SPDX-License-Identifier: Apache-2.0
// Derived from BlinkDL/Albatross faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cpp
// at 545dd1870ea6cf424d7f7d4da16b7f87491405db.
// Modified for BF16 I/O and the uniform one-token decode entry point.

#include <torch/extension.h>

void rwkv7_albatross_decode_cuda(
    int64_t B,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y);

namespace {

void check_bf16_cuda_contiguous(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be BF16");
}

void forward_decode(
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  TORCH_CHECK(state.is_cuda(), "state must be CUDA");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be FP32");
  TORCH_CHECK(
      state.dim() == 4 && state.size(2) == 64 && state.size(3) == 64,
      "state must have shape [B,H,64,64]");
  const int64_t B = state.size(0);
  const int64_t H = state.size(1);
  TORCH_CHECK(B > 0 && B <= 65535, "decode batch must be in [1,65535]");
  TORCH_CHECK(H > 0 && H <= 65535, "head count must be in [1,65535]");
  check_bf16_cuda_contiguous(r, "r");
  check_bf16_cuda_contiguous(w, "w");
  check_bf16_cuda_contiguous(k, "k");
  check_bf16_cuda_contiguous(v, "v");
  check_bf16_cuda_contiguous(a, "a");
  check_bf16_cuda_contiguous(b, "b");
  check_bf16_cuda_contiguous(y, "y");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) == B && r.size(1) == H && r.size(2) == 64,
      "inputs must have shape [B,H,64]");
  TORCH_CHECK(
      r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == y.sizes(),
      "r,w,k,v,a,b,y shape mismatch");
  rwkv7_albatross_decode_cuda(B, H * 64, H, state, r, w, k, v, a, b, y);
}

}  // namespace

TORCH_LIBRARY(irwkv7_albatross, m) {
  m.def(
      "forward_decode(Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, "
      "Tensor a, Tensor b, Tensor(a!) y) -> ()");
}

TORCH_LIBRARY_IMPL(irwkv7_albatross, CUDA, m) {
  m.impl("forward_decode", &forward_decode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
