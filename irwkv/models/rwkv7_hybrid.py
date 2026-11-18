# SPDX-License-Identifier: Apache-2.0
"""The bux timeline RWKV-7 hybrid's tensor operations.

Architecture source: pilot7747/bux@03d5031fcc842ae4a4992921356af31b1bceaee0.
Cache ownership and recurrence dispatch belong to the serving adapter.
"""

import math
from collections.abc import Callable
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_QUANT_CHUNK_ELEMS = 16 * 1024 * 1024


@cache
def _load_paged_fa2():
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
        )
    except ImportError as error:
        raise RuntimeError(
            "CUDA hybrid attention requires vllm 0.32.0's paged FA2 binary"
        ) from error
    return flash_attn_varlen_func


@cache
def _load_fp8_paged_attention():
    major = torch.cuda.get_device_capability()[0]
    if major == 12:
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        return unified_attention, None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
            is_fa_version_supported,
        )
    except ImportError as error:
        raise RuntimeError(
            "FP8 hybrid attention requires vllm 0.32.0's attention backends"
        ) from error
    version = 3 if major == 9 else 4
    if major not in (9, 10) or not is_fa_version_supported(version):
        raise RuntimeError(
            "FP8 paged attention requires FA3 on SM90, FA4 on SM100, or Triton on SM120"
        )
    return flash_attn_varlen_func, version


def _layer_norm_fake(x, weight, bias, eps):
    return torch.empty_like(x)


@cache
def _load_albatross():
    from irwkv import rwkv7_albatross_ops  # noqa: F401

    return torch.ops.irwkv7_albatross


@cache
def _register_albatross_fakes():
    _load_albatross()
    torch.library.register_fake("irwkv7_albatross::layer_norm")(_layer_norm_fake)


@cache
def _load_cutlass_scaled_mm():
    from vllm._custom_ops import cutlass_scaled_mm

    return cutlass_scaled_mm


def _fp8_quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.detach().float().abs().amax(-1, keepdim=True) / _FP8_MAX).clamp_min(
        1e-12
    )
    quantized = (x.float() / scale).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
    return quantized.contiguous(), scale


def _fp8_quantize_unit_scale(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)


class Rwkv7HybridLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias, quantize=False):
        super().__init__(in_features, out_features, bias=bias)
        self.quantize = quantize
        self.register_buffer("weight_scale", torch.empty(0), persistent=False)
        self.register_buffer("weight_scale_rowwise", torch.empty(0), persistent=False)

    def process_weights_after_loading(self):
        if not self.quantize or self.weight.dtype == _FP8_DTYPE:
            return
        weight = self.weight.detach()
        self.orig_dtype = weight.dtype
        scale = (weight.abs().max().float() / _FP8_MAX).clamp_min(1e-12).reshape(1)
        quantized = torch.empty_like(weight, dtype=_FP8_DTYPE)
        rows = max(1, _QUANT_CHUNK_ELEMS // weight.shape[-1])
        for start in range(0, weight.shape[0], rows):
            chunk = slice(start, start + rows)
            quantized[chunk] = (
                (weight[chunk].float() / scale).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
            )
        self.weight.data = quantized
        self.weight_scale = scale
        self.weight_scale_rowwise = scale.expand(1, weight.shape[0]).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != _FP8_DTYPE:
            return F.linear(x, self.weight, self.bias)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        quantized, scale = _fp8_quantize_per_token(x)
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            if major < 8 or (major == 8 and minor < 9):
                raise RuntimeError("Hybrid FP8 W8A8 requires an SM89+ CUDA device")
            output = _load_cutlass_scaled_mm()(
                quantized,
                self.weight.t(),
                scale,
                self.weight_scale_rowwise,
                out_dtype=self.orig_dtype,
                bias=self.bias,
            )
        else:
            dequantized_x = (quantized.float() * scale).to(self.orig_dtype)
            dequantized_weight = (self.weight.float() * self.weight_scale).to(
                self.orig_dtype
            )
            output = F.linear(dequantized_x, dequantized_weight, self.bias)
        return output.unflatten(0, shape[:-1])


class Rwkv7HybridLayerNorm(nn.LayerNorm):
    def forward(self, x):
        if x.is_cuda and x.dtype == torch.bfloat16:
            return _load_albatross().layer_norm(x, self.weight, self.bias, self.eps)
        centered = x.float() - x.float().mean(-1, keepdim=True)
        normalized = centered * torch.rsqrt(
            centered.square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight + self.bias


class Rwkv7HybridLowRank(nn.Module):
    def __init__(
        self, width: int, rank: int, activation: str | None, bias: bool, quantize=False
    ):
        super().__init__()
        self.down = Rwkv7HybridLinear(width, rank, False, quantize)
        self.up = Rwkv7HybridLinear(rank, width, bias, quantize)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        if self.activation == "tanh":
            x = x.tanh()
        elif self.activation == "sigmoid":
            x = x.sigmoid()
        return self.up(x)


class Rwkv7HybridMixer(nn.Module):
    def __init__(self, config, quantize=False):
        super().__init__()
        width = config.hidden_size
        self.num_heads = width // 64
        self.shift = nn.Parameter(torch.empty(6, width))
        self.r = Rwkv7HybridLinear(width, width, False, quantize)
        self.k = Rwkv7HybridLinear(width, width, False, quantize)
        self.v = Rwkv7HybridLinear(width, width, False, quantize)
        self.o = Rwkv7HybridLinear(width, width, False, quantize)
        self.w = Rwkv7HybridLowRank(width, 64, "tanh", True, quantize)
        self.a = Rwkv7HybridLowRank(width, 64, None, True, quantize)
        self.g = Rwkv7HybridLowRank(width, 128, "sigmoid", False, quantize)
        self.k_k = nn.Parameter(torch.empty(width))
        self.k_a = nn.Parameter(torch.empty(width))
        self.r_k = nn.Parameter(torch.empty(self.num_heads, 64))
        self.norm_weight = nn.Parameter(torch.empty(self.num_heads, 64))
        self.norm_bias = nn.Parameter(torch.empty(self.num_heads, 64))

    def prepare(self, shifted, fused_decode=False):
        xr, xw, xk, xv, xa, xg = shifted.unbind(0)
        r = self.r(xr).unflatten(-1, (self.num_heads, 64))
        k = self.k(xk)
        v = self.v(xv).unflatten(-1, (self.num_heads, 64))
        decay = self.w(xw).unflatten(-1, (self.num_heads, 64))
        a = self.a(xa)
        g = self.g(xg)
        if fused_decode:
            k, a, b = _load_albatross().kk_pre_decode(k, self.k_k, a, self.k_a)
            k = k.unflatten(-1, (self.num_heads, 64))
            a = a.unflatten(-1, (self.num_heads, 64))
            b = b.unflatten(-1, (self.num_heads, 64))
        else:
            a = a.sigmoid()
            kk = (k * self.k_k).unflatten(-1, (self.num_heads, 64)).float()
            kk = (kk * torch.rsqrt(kk.square().sum(-1, keepdim=True) + 1e-12)).to(
                shifted.dtype
            )
            k = (k * (1 + (a - 1) * self.k_a)).unflatten(-1, (self.num_heads, 64))
            a = a.unflatten(-1, (self.num_heads, 64))
            a, b = -kk, kk * a
        return r, decay, k, v, a, b, g

    def post(self, out, r, k, v, g, fused_decode=False):
        if fused_decode:
            out = _load_albatross().post_decode(
                out,
                r,
                k,
                v,
                self.r_k,
                self.norm_weight,
                self.norm_bias,
                g,
            )
            return self.o(out.flatten(1))
        centered = out.float() - out.float().mean(-1, keepdim=True)
        out = (
            centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 64e-5)
        ).to(r.dtype)
        out = out * self.norm_weight + self.norm_bias
        out = out + (r * k * self.r_k).sum(-1, keepdim=True) * v
        return self.o(out.flatten(1) * g)

    def forward(
        self, x, previous, recurrence: Callable, fused_decode=False, shifted=None
    ):
        if shifted is None:
            shifted = x + (previous - x) * self.shift[:, None]
        r, decay, k, v, a, b, g = self.prepare(shifted, fused_decode)
        out = recurrence(r, decay, k, v, a, b)
        return self.post(out, r, k, v, g, fused_decode)


def _load_fused_qkv(param, tensor):
    shard = getattr(tensor, "shard_id", None)
    if shard is None:
        if param.shape != tensor.shape:
            raise ValueError(f"QKV shape mismatch: {param.shape} != {tensor.shape}")
        param.data.copy_(tensor)
        return
    if shard not in (0, 1, 2) or tensor.ndim not in (1, 2):
        raise ValueError(f"Invalid QKV shard {shard} with shape {tensor.shape}")
    target = param.data.narrow(0, shard * tensor.shape[0], tensor.shape[0])
    if target.shape != tensor.shape:
        raise ValueError(f"QKV shard shape mismatch: {target.shape} != {tensor.shape}")
    target.copy_(tensor)


class Rwkv7HybridAttention(nn.Module):
    def __init__(self, config, quantize=False):
        super().__init__()
        width = config.hidden_size
        self.num_heads = width // 64
        self.qkv = Rwkv7HybridLinear(width, 3 * width, True, quantize)
        self.qkv.weight.weight_loader = _load_fused_qkv
        self.qkv.bias.weight_loader = _load_fused_qkv
        self.o = Rwkv7HybridLinear(width, width, True, quantize)

    @staticmethod
    def rotary(x, positions):
        frequency = 10000.0 ** (-torch.arange(0, 64, 2, device=x.device).float() / 64)
        angles = positions.float()[:, None] * frequency
        cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), -1
        ).to(x.dtype)

    def forward(self, x, positions, attention: Callable, fused_decode=False):
        q, k, v = (
            projection.unflatten(-1, (self.num_heads, 64))
            for projection in self.qkv(x).chunk(3, dim=-1)
        )
        if not fused_decode:
            q, k = self.rotary(q, positions), self.rotary(k, positions)
        return self.o(attention(q, k, v).flatten(1))


class Rwkv7HybridDecoderLayer(nn.Module):
    def __init__(self, config, index: int, quantize=False):
        super().__init__()
        width = config.hidden_size
        self.input_layernorm = Rwkv7HybridLayerNorm(width, eps=1e-5)
        self.post_attention_layernorm = Rwkv7HybridLayerNorm(width, eps=1e-5)
        self.mixer = (
            Rwkv7HybridAttention(config, quantize)
            if index % 4 == 3
            else Rwkv7HybridMixer(config, quantize)
        )
        self.up = Rwkv7HybridLinear(width, 4 * width, True, quantize)
        self.down = Rwkv7HybridLinear(4 * width, width, True, quantize)

    def forward(self, x, run_mixer: Callable):
        x = x + run_mixer(self.mixer, self.input_layernorm(x))
        return x + self.down(F.gelu(self.up(self.post_attention_layernorm(x))))


class Rwkv7HybridModel(nn.Module):
    def __init__(self, config, quantize=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Rwkv7HybridDecoderLayer(config, i, quantize)
            for i in range(config.num_hidden_layers)
        )
        self.norm = Rwkv7HybridLayerNorm(config.hidden_size, eps=1e-5)

    def process_weights_after_loading(self):
        for module in self.modules():
            if isinstance(module, Rwkv7HybridLinear):
                module.process_weights_after_loading()

    def forward(self, input_ids, run_mixer: Callable, inputs_embeds=None):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for index, layer in enumerate(self.layers):
            x = layer(x, lambda mixer, value, i=index: run_mixer(i, mixer, value))
        return self.norm(x)


def reference_recurrence(r, decay, k, v, a, b, state):
    """Slow FP32 oracle for full-model correctness tests, never serving dispatch."""
    outputs = []
    for index in range(r.shape[0]):
        read = torch.einsum("hvk,hk->hv", state, a[index].float())
        weight = torch.exp(-math.exp(-0.5) * decay[index].float().sigmoid())
        state = state * weight[:, None]
        state = state + read[:, :, None] * b[index].float()[:, None]
        state = state + v[index].float()[:, :, None] * k[index].float()[:, None]
        outputs.append(torch.einsum("hvk,hk->hv", state, r[index].float()))
    return torch.stack(outputs).to(r.dtype), state


class Rwkv7HybridCacheStorage:
    def state_views(self):
        raw = self.kv_cache[:, 0, 0]
        shift_bytes = self.num_heads * 64 * 2
        previous = raw[:, :shift_bytes].view(torch.bfloat16)
        state = (
            raw[:, shift_bytes:]
            .view(torch.float32)
            .unflatten(1, (self.num_heads, 64, 64))
        )
        return previous, state

    def reset_state_slots(self, indices):
        if len(indices) == 0:
            return
        self.kv_cache[indices] = 0

    def copy_state_slots(self, sources, destinations):
        if len(sources) == len(destinations) == 0:
            return
        self.kv_cache[destinations] = self.kv_cache[sources]

    def prepare_recurrent_metadata(self, metadata, flash, *, retain_ticket=False):
        indices = metadata.state_indices
        slots = torch.arange(len(indices), dtype=torch.int32, device=indices.device)
        max_length = max(b - a for a, b in zip(metadata.starts, metadata.starts[1:]))
        ticket = flash.prepare_tmix_wkv7_recurrent_metadata(
            metadata.cu_seqlens,
            slots,
            state_pool_size=len(indices),
            total_tokens=getattr(metadata, "token_capacity", metadata.starts[-1]),
            max_seqlen=max_length,
        )
        metadata.recurrent_slots = slots
        metadata.max_recurrent_seq_len = max_length
        metadata.validated_metadata = ticket
        if retain_ticket:
            self._live_graph_tickets.append(ticket)
        return ticket

    def run_recurrent(self, mixer, x, metadata):
        indices = metadata.state_indices
        previous_pool, state_pool = self.state_views()
        albatross_decode = metadata.max_query_len == 1
        previous = shifted = None
        if albatross_decode:
            shifted = _load_albatross().shift_decode(
                x, previous_pool, indices, mixer.shift
            )
        else:
            previous = torch.roll(x, 1, 0)
            starts = metadata.cu_seqlens[:-1].long()
            previous[starts] = previous_pool[indices]

        def recur(r, decay, k, v, a, b):
            tensors = tuple(tensor.contiguous() for tensor in (r, decay, k, v, a, b))
            output = torch.empty_like(r)
            ops = _load_albatross()
            if albatross_decode:
                ops.forward_decode(state_pool, indices, *tensors, output)
            else:
                ops.forward_prefill(
                    state_pool, indices, metadata.cu_seqlens, *tensors, output
                )
            return output

        output = mixer(
            x, previous, recur, fused_decode=albatross_decode, shifted=shifted
        )
        if not albatross_decode:
            previous_pool[indices] = x[metadata.cu_seqlens[1:].long() - 1]
        return output

    def run_attention(self, q, k, v, metadata, positions=None, rotary=False, out=None):
        cache = self.kv_cache
        block_size = cache.shape[2]
        fp8 = cache.dtype == _FP8_DTYPE
        fused_decode = (
            q.is_cuda and metadata.max_query_len == 1 and positions is not None
        )
        if fused_decode:
            q, k = _load_albatross().attention_prepare_decode(
                q, k, v, positions, metadata.slot_mapping, cache
            )
        else:
            if rotary:
                q, k = (
                    Rwkv7HybridAttention.rotary(projection, positions)
                    for projection in (q, k)
                )
            slots = metadata.slot_mapping.long()
            blocks, offsets = slots // block_size, slots % block_size
            if fp8:
                k, v = _fp8_quantize_unit_scale(k), _fp8_quantize_unit_scale(v)
            cache[blocks, :, offsets, :64] = k
            cache[blocks, :, offsets, 64:] = v
        output_dtype = q.dtype
        if fp8:
            q = _fp8_quantize_unit_scale(q)
        if q.is_cuda:
            paged = cache.transpose(1, 2)
            key_cache, value_cache = paged.split(64, dim=-1)
            output = (
                torch.empty(q.shape, dtype=output_dtype, device=q.device)
                if out is None
                else out
            )
            kwargs = {}
            if fp8:
                attention, version = _load_fp8_paged_attention()
                descale_shape = (len(metadata.seq_lens), q.shape[1])
                kwargs = dict(
                    q_descale=self.q_scale.float().expand(descale_shape),
                    k_descale=self.k_scale.float().expand(descale_shape),
                    v_descale=self.v_scale.float().expand(descale_shape),
                )
            else:
                attention, version = _load_paged_fa2(), 2
            if version is None:
                from vllm.v1.kv_cache_interface import KVQuantMode

                kwargs.update(
                    window_size=(-1, -1),
                    softcap=0.0,
                    seq_threshold_3D=getattr(metadata, "seq_threshold_3D", None),
                    num_par_softmax_segments=getattr(
                        metadata, "num_par_softmax_segments", None
                    ),
                    softmax_segm_output=getattr(metadata, "softmax_segm_output", None),
                    softmax_segm_max=getattr(metadata, "softmax_segm_max", None),
                    softmax_segm_expsum=getattr(metadata, "softmax_segm_expsum", None),
                    kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
                )
            else:
                kwargs["fa_version"] = version
            attention(
                q,
                key_cache,
                value_cache,
                max_seqlen_q=metadata.max_query_len,
                cu_seqlens_q=metadata.cu_seqlens,
                max_seqlen_k=metadata.max_seq_len,
                seqused_k=metadata.seq_lens,
                block_table=metadata.block_table,
                causal=True,
                softmax_scale=64**-0.5,
                out=output,
                **kwargs,
            )
            return output

        output = torch.empty(q.shape, dtype=output_dtype, device=q.device)
        if fp8:
            q = q.to(output_dtype)
        for row, (start, end, length) in enumerate(
            zip(metadata.starts, metadata.starts[1:], metadata.lengths)
        ):
            positions = torch.arange(length, device=q.device)
            pages = metadata.block_table[row, positions // block_size].long()
            history = cache[pages, :, positions % block_size].to(output_dtype)
            key, value = history.split(64, dim=-1)
            causal = causal_lower_right(end - start, length)
            output[start:end] = (
                F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1).unsqueeze(0),
                    key.transpose(0, 1).unsqueeze(0),
                    value.transpose(0, 1).unsqueeze(0),
                    attn_mask=causal,
                    scale=64**-0.5,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return output
