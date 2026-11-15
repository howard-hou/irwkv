// SPDX-License-Identifier: Apache-2.0
// Derived from BlinkDL/Albatross faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cu
// at 545dd1870ea6cf424d7f7d4da16b7f87491405db.
// Modified for BF16 I/O and the uniform one-token decode entry point.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadSize = 64;
constexpr int kWarpThreads = 32;
constexpr int kValueTile = 4;
constexpr int kKeysPerWarp = kWarpThreads / kValueTile;
constexpr float kWScaleLog2E = -0.8750387749145276f;
constexpr float kNLog2E = -1.4426950408889634f;

__device__ __forceinline__ float load_bf16(
    const __nv_bfloat16* ptr, int64_t index) {
  return __bfloat162float(__ldg(ptr + index));
}

__device__ __forceinline__ float retention(float w) {
  return exp2f(kWScaleLog2E / (1.0f + exp2f(kNLog2E * w)));
}

// One warp owns four adjacent V columns in the FP32 [B,H,K,V] state.
__global__ __launch_bounds__(kWarpThreads, 4) void wkv_decode_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
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
  float* state_base = state_ptr +
      (static_cast<int64_t>(batch) * H + head) * kHeadSize * kHeadSize;

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
    y_ptr[token_base + value_index] = __float2bfloat16_rn(result);
  }
}

}  // namespace

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
    torch::Tensor y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_decode_kernel<<<dim3(H, B, kHeadSize / kValueTile), kWarpThreads, 0, stream>>>(
      static_cast<int>(C),
      static_cast<int>(H),
      state.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(r.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
