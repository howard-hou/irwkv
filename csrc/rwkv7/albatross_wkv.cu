// SPDX-License-Identifier: Apache-2.0
// Derived from BlinkDL/Albatross faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cu
// at 545dd1870ea6cf424d7f7d4da16b7f87491405db.
// kk-pre and output fusion follow BlinkDL/RWKV-LM train_temp inference formulas.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <tuple>
#include <type_traits>

namespace {

constexpr int kHeadSize = 64;
constexpr int kWarpThreads = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kValueTile = 4;
constexpr int kKeysPerWarp = kWarpThreads / kValueTile;
constexpr float kWScaleLog2E = -0.8750387749145276f;
constexpr float kNLog2E = -1.4426950408889634f;
constexpr float kNormalizeEps = 1.0e-12f;
constexpr float kLnXEps = 64e-5f;

__device__ __forceinline__ float load_bf16(
    const __nv_bfloat16* ptr, int64_t index) {
  return __bfloat162float(__ldg(ptr + index));
}

__device__ __forceinline__ __nv_bfloat16 to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float bf16_round(float value) {
  return __bfloat162float(to_bf16(value));
}

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float retention(float w) {
  return exp2f(kWScaleLog2E / (1.0f + exp2f(kNLog2E * w)));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// One warp owns four adjacent V columns in the FP32 [B,H,K,V] state.
__global__ __launch_bounds__(kWarpThreads, 4) void wkv_decode_kernel(
    int C,
    int H,
    int64_t state_pool_stride,
    float* __restrict__ state_ptr,
    const int64_t* __restrict__ state_indices_ptr,
    const __nv_bfloat16* __restrict__ r_ptr,
    const __nv_bfloat16* __restrict__ w_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const __nv_bfloat16* __restrict__ b_ptr,
    __nv_bfloat16* __restrict__ y_ptr) {
  const int head = static_cast<int>(blockIdx.x);
  const int batch = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int value_lane = lane & (kValueTile - 1);
  const int key_lane = lane / kValueTile;
  const int value_index = static_cast<int>(blockIdx.z) * kValueTile + value_lane;
  const int64_t token_base = static_cast<int64_t>(batch) * C + head * kHeadSize;
  const int64_t state_slot = __ldg(state_indices_ptr + batch);
  float* state_base = state_ptr + state_slot * state_pool_stride +
      static_cast<int64_t>(head) * kHeadSize * kHeadSize;

  float a_state = 0.0f;
#pragma unroll
  for (int key_base = 0; key_base < kHeadSize; key_base += kKeysPerWarp) {
    const int key_index = key_base + key_lane;
    float contribution =
        state_base[key_index * kHeadSize + value_index] *
        load_bf16(a_ptr, token_base + key_index);
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= kValueTile; offset >>= 1) {
      contribution += __shfl_down_sync(0xffffffffu, contribution, offset);
    }
    if (lane < kValueTile) {
      a_state += contribution;
    }
  }
  a_state = __shfl_sync(0xffffffffu, a_state, value_lane);
  float value = lane < kValueTile
      ? load_bf16(v_ptr, token_base + value_index)
      : 0.0f;
  value = __shfl_sync(0xffffffffu, value, value_lane);

  float result = 0.0f;
#pragma unroll
  for (int key_base = 0; key_base < kHeadSize; key_base += kKeysPerWarp) {
    const int key_index = key_base + key_lane;
    const int64_t input_index = token_base + key_index;
    const int state_index = key_index * kHeadSize + value_index;
    const float updated = retention(load_bf16(w_ptr, input_index)) *
            state_base[state_index] +
        load_bf16(b_ptr, input_index) * a_state +
        load_bf16(k_ptr, input_index) * value;
    state_base[state_index] = updated;
    float contribution = load_bf16(r_ptr, input_index) * updated;
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= kValueTile; offset >>= 1) {
      contribution += __shfl_down_sync(0xffffffffu, contribution, offset);
    }
    if (lane < kValueTile) {
      result += contribution;
    }
  }
  if (lane < kValueTile) {
    y_ptr[token_base + value_index] = to_bf16(result);
  }
}

// Fuse a=sigmoid(a), k scaling, kk normalization, -kk and kk*a.
// Intermediate BF16 rounding intentionally matches the eager serving path.
__global__ __launch_bounds__(kWarpsPerBlock * kWarpThreads) void
rwkv7_fused_kk_pre_decode_kernel(
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ k_k_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const __nv_bfloat16* __restrict__ k_a_ptr,
    __nv_bfloat16* __restrict__ new_k_ptr,
    __nv_bfloat16* __restrict__ neg_kk_ptr,
    __nv_bfloat16* __restrict__ kka_ptr,
    int64_t head_rows,
    int H) {
  const int warp = threadIdx.x / kWarpThreads;
  const int lane = threadIdx.x % kWarpThreads;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= head_rows) {
    return;
  }
  const int head = static_cast<int>(row % H);
  const int64_t input_base = row * kHeadSize + lane * 2;
  const int parameter_base = head * kHeadSize + lane * 2;

  const float k0 = load_bf16(k_ptr, input_base);
  const float k1 = load_bf16(k_ptr, input_base + 1);
  const float u0 = bf16_round(k0 * load_bf16(k_k_ptr, parameter_base));
  const float u1 = bf16_round(k1 * load_bf16(k_k_ptr, parameter_base + 1));
  const float sum_sq = warp_sum(u0 * u0 + u1 * u1);
  const float inv_d = rsqrtf(
      __shfl_sync(0xffffffffu, sum_sq, 0) + kNormalizeEps);
  const float kk0 = bf16_round(u0 * inv_d);
  const float kk1 = bf16_round(u1 * inv_d);
  const float a0 = bf16_round(sigmoid(load_bf16(a_ptr, input_base)));
  const float a1 = bf16_round(sigmoid(load_bf16(a_ptr, input_base + 1)));
  const float ka0 = load_bf16(k_a_ptr, parameter_base);
  const float ka1 = load_bf16(k_a_ptr, parameter_base + 1);
  const float scale0 = bf16_round(
      1.0f + bf16_round(bf16_round(a0 - 1.0f) * ka0));
  const float scale1 = bf16_round(
      1.0f + bf16_round(bf16_round(a1 - 1.0f) * ka1));

  new_k_ptr[input_base] = to_bf16(k0 * scale0);
  new_k_ptr[input_base + 1] = to_bf16(k1 * scale1);
  neg_kk_ptr[input_base] = to_bf16(-kk0);
  neg_kk_ptr[input_base + 1] = to_bf16(-kk1);
  kka_ptr[input_base] = to_bf16(kk0 * a0);
  kka_ptr[input_base + 1] = to_bf16(kk1 * a1);
}

// Fuse GroupNorm, r*k*r_k residual and output gating for one decode token.
__global__ __launch_bounds__(kWarpsPerBlock * kWarpThreads) void
rwkv7_fused_post_decode_kernel(
    const __nv_bfloat16* __restrict__ x_ptr,
    const __nv_bfloat16* __restrict__ r_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const __nv_bfloat16* __restrict__ r_k_ptr,
    const __nv_bfloat16* __restrict__ weight_ptr,
    const __nv_bfloat16* __restrict__ bias_ptr,
    const __nv_bfloat16* __restrict__ g_ptr,
    __nv_bfloat16* __restrict__ y_ptr,
    int64_t head_rows,
    int H) {
  const int warp = threadIdx.x / kWarpThreads;
  const int lane = threadIdx.x % kWarpThreads;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= head_rows) {
    return;
  }
  const int head = static_cast<int>(row % H);
  const int64_t input_base = row * kHeadSize + lane * 2;
  const int parameter_base = head * kHeadSize + lane * 2;

  const float x0 = load_bf16(x_ptr, input_base);
  const float x1 = load_bf16(x_ptr, input_base + 1);
  const float mean = __shfl_sync(0xffffffffu, warp_sum(x0 + x1), 0) /
      static_cast<float>(kHeadSize);
  const float d0 = x0 - mean;
  const float d1 = x1 - mean;
  const float variance = __shfl_sync(
      0xffffffffu, warp_sum(d0 * d0 + d1 * d1), 0) /
      static_cast<float>(kHeadSize);
  const float rstd = rsqrtf(variance + kLnXEps);
  const float normalized0 = bf16_round(d0 * rstd);
  const float normalized1 = bf16_round(d1 * rstd);
  const float norm0 = bf16_round(
      bf16_round(normalized0 * load_bf16(weight_ptr, parameter_base)) +
      load_bf16(bias_ptr, parameter_base));
  const float norm1 = bf16_round(
      bf16_round(normalized1 * load_bf16(weight_ptr, parameter_base + 1)) +
      load_bf16(bias_ptr, parameter_base + 1));

  const float rk0 = bf16_round(
      bf16_round(load_bf16(r_ptr, input_base) * load_bf16(k_ptr, input_base)) *
      load_bf16(r_k_ptr, parameter_base));
  const float rk1 = bf16_round(
      bf16_round(
          load_bf16(r_ptr, input_base + 1) * load_bf16(k_ptr, input_base + 1)) *
      load_bf16(r_k_ptr, parameter_base + 1));
  const float scale = bf16_round(
      __shfl_sync(0xffffffffu, warp_sum(rk0 + rk1), 0));
  const float residual0 = bf16_round(
      scale * load_bf16(v_ptr, input_base));
  const float residual1 = bf16_round(
      scale * load_bf16(v_ptr, input_base + 1));
  const float gated0 = bf16_round(norm0 + residual0);
  const float gated1 = bf16_round(norm1 + residual1);
  y_ptr[input_base] = to_bf16(gated0 * load_bf16(g_ptr, input_base));
  y_ptr[input_base + 1] = to_bf16(gated1 * load_bf16(g_ptr, input_base + 1));
}

constexpr int kLayerNormWarps = 16;
constexpr int kLayerNormThreads = kLayerNormWarps * kWarpThreads;

__device__ __forceinline__ float layer_norm_block_sum(
    float value, float* warp_sums) {
  const int warp = threadIdx.x / kWarpThreads;
  const int lane = threadIdx.x % kWarpThreads;
  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < kLayerNormWarps ? warp_sums[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  const float total = warp_sums[0];
  __syncthreads();
  return total;
}

// Preserve the eager path's BF16 rounding after normalization and affine steps.
__global__ __launch_bounds__(kLayerNormThreads) void
rwkv7_fused_layer_norm_kernel(
    const __nv_bfloat16* __restrict__ x_ptr,
    const __nv_bfloat16* __restrict__ weight_ptr,
    const __nv_bfloat16* __restrict__ bias_ptr,
    __nv_bfloat16* __restrict__ output_ptr,
    int C,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int channel = threadIdx.x; channel < C; channel += blockDim.x) {
    sum += load_bf16(x_ptr, base + channel);
  }
  __shared__ float warp_sums[kLayerNormWarps];
  const float mean = layer_norm_block_sum(sum, warp_sums) / static_cast<float>(C);
  float square_sum = 0.0f;
  for (int channel = threadIdx.x; channel < C; channel += blockDim.x) {
    const float centered = __fsub_rn(load_bf16(x_ptr, base + channel), mean);
    square_sum = __fadd_rn(square_sum, __fmul_rn(centered, centered));
  }
  const float variance =
      layer_norm_block_sum(square_sum, warp_sums) / static_cast<float>(C);
  const float rstd = rsqrtf(variance + eps);
  for (int channel = threadIdx.x; channel < C; channel += blockDim.x) {
    const float centered = __fsub_rn(load_bf16(x_ptr, base + channel), mean);
    const float normalized = bf16_round(__fmul_rn(centered, rstd));
    const float scaled = bf16_round(__fmul_rn(
        normalized, load_bf16(weight_ptr, channel)));
    output_ptr[base + channel] = to_bf16(__fadd_rn(
        scaled, load_bf16(bias_ptr, channel)));
  }
}

__global__ __launch_bounds__(kWarpsPerBlock * kWarpThreads) void
rwkv7_fused_shift_decode_kernel(
    const __nv_bfloat16* __restrict__ x_ptr,
    __nv_bfloat16* __restrict__ previous_ptr,
    const int64_t* __restrict__ state_indices_ptr,
    const __nv_bfloat16* __restrict__ shift_ptr,
    __nv_bfloat16* __restrict__ output_ptr,
    int64_t previous_stride,
    int C,
    int64_t elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int channel = static_cast<int>(index % C);
  const int batch = static_cast<int>(index / C);
  const int64_t state_slot = __ldg(state_indices_ptr + batch);
  const int64_t previous_index = state_slot * previous_stride + channel;
  const float x = load_bf16(x_ptr, index);
  const float previous = load_bf16(previous_ptr, previous_index);
  const float delta = bf16_round(__fsub_rn(previous, x));
#pragma unroll
  for (int component = 0; component < 6; ++component) {
    const float product = bf16_round(__fmul_rn(
        delta, load_bf16(shift_ptr, component * C + channel)));
    const int64_t output_index =
        (static_cast<int64_t>(component) * (elements / C) + batch) * C + channel;
    output_ptr[output_index] = to_bf16(__fadd_rn(x, product));
  }
  previous_ptr[previous_index] = to_bf16(x);
}

__device__ __forceinline__ float rope_component(
    const __nv_bfloat16* __restrict__ input,
    int channel,
    float cosine,
    float sine) {
  const bool first_half = channel < kHeadSize / 2;
  const int pair_channel = first_half ? channel + kHeadSize / 2
                                      : channel - kHeadSize / 2;
  const float value = load_bf16(input, channel);
  const float pair = load_bf16(input, pair_channel);
  return first_half
      ? __fsub_rn(__fmul_rn(value, cosine), __fmul_rn(pair, sine))
      : __fadd_rn(__fmul_rn(value, cosine), __fmul_rn(pair, sine));
}

template <typename CacheT, typename PositionT, typename SlotT>
__global__ __launch_bounds__(kWarpsPerBlock * kWarpThreads) void
rwkv7_fused_attention_prepare_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const PositionT* __restrict__ positions_ptr,
    const SlotT* __restrict__ slots_ptr,
    CacheT* __restrict__ cache_ptr,
    __nv_bfloat16* __restrict__ output_q_ptr,
    __nv_bfloat16* __restrict__ output_k_ptr,
    int64_t q_token_stride,
    int64_t q_head_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    int64_t cache_page_stride,
    int64_t cache_head_stride,
    int64_t cache_slot_stride,
    int block_size,
    int num_pages,
    int H,
    int64_t elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int channel = static_cast<int>(index % kHeadSize);
  const int64_t token_head = index / kHeadSize;
  const int head = static_cast<int>(token_head % H);
  const int batch = static_cast<int>(token_head / H);
  const int pair = channel % (kHeadSize / 2);
  const float frequency = powf(10000.0f, -static_cast<float>(2 * pair) / 64.0f);
  const float angle = __fmul_rn(
      static_cast<float>(__ldg(positions_ptr + batch)), frequency);
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const __nv_bfloat16* q = q_ptr +
      static_cast<int64_t>(batch) * q_token_stride + head * q_head_stride;
  const __nv_bfloat16* k = k_ptr +
      static_cast<int64_t>(batch) * k_token_stride + head * k_head_stride;
  const __nv_bfloat16* v = v_ptr +
      static_cast<int64_t>(batch) * v_token_stride + head * v_head_stride;
  const float rotated_q = rope_component(q, channel, cosine, sine);
  const float rotated_k = rope_component(k, channel, cosine, sine);
  output_q_ptr[index] = to_bf16(rotated_q);
  output_k_ptr[index] = to_bf16(rotated_k);

  const int64_t slot = static_cast<int64_t>(__ldg(slots_ptr + batch));
  int64_t block = slot / block_size;
  int64_t offset = slot % block_size;
  if (offset < 0) {
    offset += block_size;
    --block;
  }
  if (block < 0) {
    block += num_pages;
  }
  CacheT* cache = cache_ptr + block * cache_page_stride +
      head * cache_head_stride + offset * cache_slot_stride;
  if constexpr (std::is_same_v<CacheT, __nv_bfloat16>) {
    cache[channel] = to_bf16(rotated_k);
    cache[kHeadSize + channel] = v[channel];
  } else {
    cache[channel] = __nv_cvt_float_to_fp8(
        __bfloat162float(to_bf16(rotated_k)), __NV_SATFINITE, __NV_E4M3);
    cache[kHeadSize + channel] = __nv_cvt_float_to_fp8(
        __bfloat162float(v[channel]), __NV_SATFINITE, __NV_E4M3);
  }
}

inline int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

}  // namespace

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
    torch::Tensor y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_decode_kernel<<<dim3(H, B, kHeadSize / kValueTile), kWarpThreads, 0, stream>>>(
      static_cast<int>(C),
      static_cast<int>(H),
      state.stride(0),
      state.data_ptr<float>(),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(r.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rwkv7_albatross_kk_pre_decode_cuda(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a) {
  auto new_k = torch::empty_like(k);
  auto neg_kk = torch::empty_like(k);
  auto kka = torch::empty_like(k);
  const int64_t B = k.size(0);
  const int64_t H = k.size(1) / kHeadSize;
  const int64_t rows = B * H;
  const int blocks = static_cast<int>(ceil_div(rows, kWarpsPerBlock));
  auto stream = at::cuda::getCurrentCUDAStream();
  rwkv7_fused_kk_pre_decode_kernel<<<
      blocks, kWarpsPerBlock * kWarpThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k_k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k_a.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(new_k.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(neg_kk.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(kka.data_ptr<at::BFloat16>()),
      rows,
      static_cast<int>(H));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_k, neg_kk, kka};
}

torch::Tensor rwkv7_albatross_post_decode_cuda(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  auto y = torch::empty_like(x);
  const int64_t B = x.size(0);
  const int64_t H = x.size(1);
  const int64_t rows = B * H;
  const int blocks = static_cast<int>(ceil_div(rows, kWarpsPerBlock));
  auto stream = at::cuda::getCurrentCUDAStream();
  rwkv7_fused_post_decode_kernel<<<
      blocks, kWarpsPerBlock * kWarpThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(r.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(r_k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      rows,
      static_cast<int>(H));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor rwkv7_albatross_layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
  auto output = torch::empty_like(x);
  const int64_t rows = x.size(0);
  const int C = static_cast<int>(x.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  rwkv7_fused_layer_norm_kernel<<<
      static_cast<int>(rows), kLayerNormThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      C,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_albatross_shift_decode_cuda(
    torch::Tensor x,
    torch::Tensor previous,
    torch::Tensor state_indices,
    torch::Tensor shift) {
  const int64_t B = x.size(0);
  const int64_t C = x.size(1);
  auto output = torch::empty({6, B, C}, x.options());
  const int64_t elements = B * C;
  const int threads = kWarpsPerBlock * kWarpThreads;
  const int blocks = static_cast<int>(ceil_div(elements, threads));
  auto stream = at::cuda::getCurrentCUDAStream();
  rwkv7_fused_shift_decode_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(previous.data_ptr<at::BFloat16>()),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      previous.stride(0),
      static_cast<int>(C),
      elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
rwkv7_albatross_attention_prepare_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor positions,
    torch::Tensor slot_mapping,
    torch::Tensor cache) {
  auto output_q = torch::empty(q.sizes(), q.options());
  auto output_k = torch::empty(k.sizes(), k.options());
  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  const int64_t elements = B * H * kHeadSize;
  const int threads = kWarpsPerBlock * kWarpThreads;
  const int blocks = static_cast<int>(ceil_div(elements, threads));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch_cache = [&](auto cache_marker) {
    using CacheT = decltype(cache_marker);
    auto launch = [&](auto position_marker, auto slot_marker) {
      using PositionT = decltype(position_marker);
      using SlotT = decltype(slot_marker);
      rwkv7_fused_attention_prepare_decode_kernel<CacheT, PositionT, SlotT>
          <<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
              positions.data_ptr<PositionT>(),
              slot_mapping.data_ptr<SlotT>(),
              reinterpret_cast<CacheT*>(cache.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(output_q.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output_k.data_ptr<at::BFloat16>()),
              q.stride(0),
              q.stride(1),
              k.stride(0),
              k.stride(1),
              v.stride(0),
              v.stride(1),
              cache.stride(0),
              cache.stride(1),
              cache.stride(2),
              static_cast<int>(cache.size(2)),
              static_cast<int>(cache.size(0)),
              static_cast<int>(H),
              elements);
    };
    if (positions.scalar_type() == torch::kInt32) {
      if (slot_mapping.scalar_type() == torch::kInt32) {
        launch(int32_t{}, int32_t{});
      } else {
        launch(int32_t{}, int64_t{});
      }
    } else {
      if (slot_mapping.scalar_type() == torch::kInt32) {
        launch(int64_t{}, int32_t{});
      } else {
        launch(int64_t{}, int64_t{});
      }
    }
  };
  if (cache.scalar_type() == torch::kBFloat16) {
    launch_cache(__nv_bfloat16{});
  } else {
    launch_cache(__nv_fp8_storage_t{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_q, output_k};
}
