// SPDX-License-Identifier: Apache-2.0
// Derived from BlinkDL/Albatross faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cpp
// at 545dd1870ea6cf424d7f7d4da16b7f87491405db.
// Modified for indexed state-pool access and inference-only decode fusion.

#include <torch/extension.h>

#include <tuple>

void rwkv7_albatross_decode_cuda(
    int64_t B,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor state_indices,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y);

void rwkv7_albatross_prefill_cuda(
    int64_t B,
    int64_t H,
    torch::Tensor state,
    torch::Tensor state_indices,
    torch::Tensor cu_seqlens,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rwkv7_albatross_kk_pre_decode_cuda(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a);

torch::Tensor rwkv7_albatross_post_decode_cuda(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);

torch::Tensor rwkv7_albatross_layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps);

torch::Tensor rwkv7_albatross_shift_decode_cuda(
    torch::Tensor x,
    torch::Tensor previous,
    torch::Tensor state_indices,
    torch::Tensor shift);

std::tuple<torch::Tensor, torch::Tensor>
rwkv7_albatross_attention_prepare_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor positions,
    torch::Tensor slot_mapping,
    torch::Tensor cache);

namespace {

void check_bf16_cuda_contiguous(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be BF16");
}

void check_head_parameters(
    const torch::Tensor& x,
    int64_t H,
    const char* name) {
  check_bf16_cuda_contiguous(x, name);
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) == H && x.size(1) == 64,
      name, " must have shape [H,64]");
}

void forward_decode(
    torch::Tensor state,
    torch::Tensor state_indices,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  TORCH_CHECK(state.is_cuda(), "state must be CUDA");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be FP32");
  TORCH_CHECK(
      state.dim() == 4 && state.size(2) == 64 && state.size(3) == 64,
      "state must have shape [P,H,64,64]");
  TORCH_CHECK(
      state.stride(3) == 1 && state.stride(2) == 64 &&
          state.stride(1) == 64 * 64 &&
          state.stride(0) >= state.size(1) * 64 * 64,
      "state must have contiguous [H,64,64] pages");
  const int64_t H = state.size(1);
  check_bf16_cuda_contiguous(r, "r");
  TORCH_CHECK(
      r.dim() == 3 && r.size(1) == H && r.size(2) == 64,
      "inputs must have shape [B,H,64]");
  const int64_t B = r.size(0);
  TORCH_CHECK(B > 0 && B <= 65535, "decode batch must be in [1,65535]");
  TORCH_CHECK(H > 0 && H <= 65535, "head count must be in [1,65535]");
  TORCH_CHECK(
      state_indices.is_cuda() && state_indices.is_contiguous() &&
          state_indices.scalar_type() == torch::kInt64 &&
          state_indices.dim() == 1 && state_indices.size(0) == B,
      "state_indices must be contiguous CUDA int64 with shape [B]");
  check_bf16_cuda_contiguous(w, "w");
  check_bf16_cuda_contiguous(k, "k");
  check_bf16_cuda_contiguous(v, "v");
  check_bf16_cuda_contiguous(a, "a");
  check_bf16_cuda_contiguous(b, "b");
  check_bf16_cuda_contiguous(y, "y");
  TORCH_CHECK(
      r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == y.sizes(),
      "r,w,k,v,a,b,y shape mismatch");
  rwkv7_albatross_decode_cuda(
      B, H * 64, H, state, state_indices, r, w, k, v, a, b, y);
}

void forward_prefill(
    torch::Tensor state,
    torch::Tensor state_indices,
    torch::Tensor cu_seqlens,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  TORCH_CHECK(state.is_cuda(), "state must be CUDA");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be FP32");
  TORCH_CHECK(
      state.dim() == 4 && state.size(2) == 64 && state.size(3) == 64,
      "state must have shape [P,H,64,64]");
  TORCH_CHECK(
      state.stride(3) == 1 && state.stride(2) == 64 &&
          state.stride(1) == 64 * 64 &&
          state.stride(0) >= state.size(1) * 64 * 64,
      "state must have contiguous [H,64,64] pages");
  const int64_t H = state.size(1);
  TORCH_CHECK(H > 0 && H <= 65535, "head count must be in [1,65535]");
  TORCH_CHECK(
      state_indices.is_cuda() && state_indices.is_contiguous() &&
          state_indices.scalar_type() == torch::kInt64 &&
          state_indices.dim() == 1,
      "state_indices must be contiguous CUDA int64 with shape [B]");
  const int64_t B = state_indices.size(0);
  TORCH_CHECK(B > 0 && B <= 65535, "prefill batch must be in [1,65535]");
  TORCH_CHECK(
      cu_seqlens.is_cuda() && cu_seqlens.is_contiguous() &&
          (cu_seqlens.scalar_type() == torch::kInt32 ||
           cu_seqlens.scalar_type() == torch::kInt64) &&
          cu_seqlens.dim() == 1 && cu_seqlens.size(0) == B + 1,
      "cu_seqlens must be contiguous CUDA int32/int64 with shape [B+1]");
  check_bf16_cuda_contiguous(r, "r");
  TORCH_CHECK(
      r.dim() == 3 && r.size(1) == H && r.size(2) == 64,
      "inputs must have shape [T,H,64]");
  check_bf16_cuda_contiguous(w, "w");
  check_bf16_cuda_contiguous(k, "k");
  check_bf16_cuda_contiguous(v, "v");
  check_bf16_cuda_contiguous(a, "a");
  check_bf16_cuda_contiguous(b, "b");
  check_bf16_cuda_contiguous(y, "y");
  TORCH_CHECK(
      r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == y.sizes(),
      "r,w,k,v,a,b,y shape mismatch");
  rwkv7_albatross_prefill_cuda(
      B, H, state, state_indices, cu_seqlens, r, w, k, v, a, b, y);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kk_pre_decode(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a) {
  check_bf16_cuda_contiguous(k, "k");
  check_bf16_cuda_contiguous(a, "a");
  TORCH_CHECK(
      k.dim() == 2 && k.size(1) % 64 == 0,
      "k must have shape [B,H*64]");
  TORCH_CHECK(k.sizes() == a.sizes(), "k and a shape mismatch");
  const int64_t C = k.size(1);
  for (const auto* item : {&k_k, &k_a}) {
    check_bf16_cuda_contiguous(*item, "head parameter");
    TORCH_CHECK(
        item->dim() == 1 && item->size(0) == C,
        "head parameter must have shape [H*64]");
  }
  return rwkv7_albatross_kk_pre_decode_cuda(k, k_k, a, k_a);
}

torch::Tensor post_decode(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  check_bf16_cuda_contiguous(x, "x");
  TORCH_CHECK(
      x.dim() == 3 && x.size(2) == 64,
      "x must have shape [B,H,64]");
  const int64_t B = x.size(0);
  const int64_t H = x.size(1);
  for (const auto* item : {&r, &k, &v}) {
    check_bf16_cuda_contiguous(*item, "post input");
    TORCH_CHECK(item->sizes() == x.sizes(), "post input shape mismatch");
  }
  check_head_parameters(r_k, H, "r_k");
  check_head_parameters(weight, H, "weight");
  check_head_parameters(bias, H, "bias");
  check_bf16_cuda_contiguous(g, "g");
  TORCH_CHECK(
      g.dim() == 2 && g.size(0) == B && g.size(1) == H * 64,
      "g must have shape [B,H*64]");
  return rwkv7_albatross_post_decode_cuda(x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor layer_norm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
  check_bf16_cuda_contiguous(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must have shape [B,C]");
  const int64_t C = x.size(1);
  for (const auto* item : {&weight, &bias}) {
    check_bf16_cuda_contiguous(*item, "layer_norm parameter");
    TORCH_CHECK(
        item->dim() == 1 && item->size(0) == C,
        "layer_norm parameters must have shape [C]");
  }
  return rwkv7_albatross_layer_norm_cuda(x, weight, bias, eps);
}

torch::Tensor shift_decode(
    torch::Tensor x,
    torch::Tensor previous,
    torch::Tensor state_indices,
    torch::Tensor shift) {
  check_bf16_cuda_contiguous(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must have shape [B,C]");
  const int64_t B = x.size(0);
  const int64_t C = x.size(1);
  TORCH_CHECK(
      previous.is_cuda() && previous.scalar_type() == torch::kBFloat16 &&
          previous.dim() == 2 && previous.size(1) == C &&
          previous.stride(1) == 1,
      "previous must be a CUDA BF16 [P,C] pool with contiguous channels");
  TORCH_CHECK(
      state_indices.is_cuda() && state_indices.is_contiguous() &&
          state_indices.scalar_type() == torch::kInt64 &&
          state_indices.dim() == 1 && state_indices.size(0) == B,
      "state_indices must be contiguous CUDA int64 with shape [B]");
  check_bf16_cuda_contiguous(shift, "shift");
  TORCH_CHECK(
      shift.dim() == 2 && shift.size(0) == 6 && shift.size(1) == C,
      "shift must have shape [6,C]");
  return rwkv7_albatross_shift_decode_cuda(x, previous, state_indices, shift);
}

std::tuple<torch::Tensor, torch::Tensor> attention_prepare_decode(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor positions,
    torch::Tensor slot_mapping,
    torch::Tensor cache) {
  for (const auto* item : {&q, &k, &v}) {
    TORCH_CHECK(
        item->is_cuda() && item->scalar_type() == torch::kBFloat16 &&
            item->dim() == 3 && item->size(2) == 64 && item->stride(2) == 1,
        "q,k,v must be CUDA BF16 [B,H,64] with contiguous head channels");
  }
  TORCH_CHECK(
      q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q,k,v shape mismatch");
  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  for (const auto* item : {&positions, &slot_mapping}) {
    TORCH_CHECK(
        item->is_cuda() && item->is_contiguous() && item->dim() == 1 &&
            item->size(0) == B &&
            (item->scalar_type() == torch::kInt32 ||
             item->scalar_type() == torch::kInt64),
        "positions and slot_mapping must be contiguous CUDA int32/int64 [B]");
  }
  TORCH_CHECK(
      cache.is_cuda() &&
          (cache.scalar_type() == torch::kBFloat16 ||
           cache.scalar_type() == torch::kFloat8_e4m3fn) &&
          cache.dim() == 4 && cache.size(1) == H && cache.size(3) == 128 &&
          cache.stride(3) == 1 && cache.size(2) > 0,
      "cache must be CUDA BF16/FP8-E4M3 [P,H,S,128] with contiguous channels");
  return rwkv7_albatross_attention_prepare_decode_cuda(
      q, k, v, positions, slot_mapping, cache);
}

}  // namespace

TORCH_LIBRARY(irwkv7_albatross, m) {
  m.def(
      "forward_decode(Tensor(a!) state, Tensor state_indices, Tensor r, "
      "Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(b!) y) -> ()");
  m.def(
      "forward_prefill(Tensor(a!) state, Tensor state_indices, "
      "Tensor cu_seqlens, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, "
      "Tensor b, Tensor(b!) y) -> ()");
  m.def(
      "kk_pre_decode(Tensor k, Tensor k_k, Tensor a, Tensor k_a) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "post_decode(Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, "
      "Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "layer_norm(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor");
  m.def(
      "shift_decode(Tensor x, Tensor(a!) previous, Tensor state_indices, "
      "Tensor shift) -> Tensor");
  m.def(
      "attention_prepare_decode(Tensor q, Tensor k, Tensor v, "
      "Tensor positions, Tensor slot_mapping, Tensor(a!) cache) "
      "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(irwkv7_albatross, CUDA, m) {
  m.impl("forward_decode", &forward_decode);
  m.impl("forward_prefill", &forward_prefill);
  m.impl("kk_pre_decode", &kk_pre_decode);
  m.impl("post_decode", &post_decode);
  m.impl("layer_norm", &layer_norm);
  m.impl("shift_decode", &shift_decode);
  m.impl("attention_prepare_decode", &attention_prepare_decode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
