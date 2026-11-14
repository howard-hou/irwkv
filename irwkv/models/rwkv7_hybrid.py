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


class Rwkv7HybridLayerNorm(nn.LayerNorm):
    def forward(self, x):
        centered = x.float() - x.float().mean(-1, keepdim=True)
        normalized = centered * torch.rsqrt(
            centered.square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight + self.bias


class Rwkv7HybridLowRank(nn.Module):
    def __init__(self, width: int, rank: int, activation: str | None, bias: bool):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        if self.activation == "tanh":
            x = x.tanh()
        elif self.activation == "sigmoid":
            x = x.sigmoid()
        return self.up(x)


class Rwkv7HybridMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.num_heads = width // 64
        self.shift = nn.Parameter(torch.empty(6, width))
        self.r = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.w = Rwkv7HybridLowRank(width, 64, "tanh", True)
        self.a = Rwkv7HybridLowRank(width, 64, None, True)
        self.g = Rwkv7HybridLowRank(width, 128, "sigmoid", False)
        self.k_k = nn.Parameter(torch.empty(width))
        self.k_a = nn.Parameter(torch.empty(width))
        self.r_k = nn.Parameter(torch.empty(self.num_heads, 64))
        self.norm_weight = nn.Parameter(torch.empty(self.num_heads, 64))
        self.norm_bias = nn.Parameter(torch.empty(self.num_heads, 64))

    def forward(self, x, previous, recurrence: Callable):
        xr, xw, xk, xv, xa, xg = (x + (previous - x) * self.shift[:, None]).unbind(0)
        r = self.r(xr).unflatten(-1, (self.num_heads, 64))
        k = self.k(xk)
        v = self.v(xv).unflatten(-1, (self.num_heads, 64))
        decay = self.w(xw).unflatten(-1, (self.num_heads, 64))
        a = self.a(xa).sigmoid()
        kk = (k * self.k_k).unflatten(-1, (self.num_heads, 64)).float()
        kk = (kk * torch.rsqrt(kk.square().sum(-1, keepdim=True) + 1e-12)).to(x.dtype)
        k = (k * (1 + (a - 1) * self.k_a)).unflatten(-1, (self.num_heads, 64))
        a = a.unflatten(-1, (self.num_heads, 64))
        out = recurrence(r, decay, k, v, -kk, kk * a)
        centered = out.float() - out.float().mean(-1, keepdim=True)
        out = (
            centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 64e-5)
        ).to(x.dtype)
        out = out * self.norm_weight + self.norm_bias
        out = out + (r * k * self.r_k).sum(-1, keepdim=True) * v
        return self.o(out.flatten(1) * self.g(xg))


class Rwkv7HybridAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.num_heads = width // 64
        self.q = nn.Linear(width, width, bias=True)
        self.k = nn.Linear(width, width, bias=True)
        self.v = nn.Linear(width, width, bias=True)
        self.o = nn.Linear(width, width, bias=True)

    @staticmethod
    def rotary(x, positions):
        frequency = 10000.0 ** (-torch.arange(0, 64, 2, device=x.device).float() / 64)
        angles = positions.float()[:, None] * frequency
        cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), -1
        ).to(x.dtype)

    def forward(self, x, positions, attention: Callable):
        q, k, v = (
            projection(x).unflatten(-1, (self.num_heads, 64))
            for projection in (self.q, self.k, self.v)
        )
        q, k = self.rotary(q, positions), self.rotary(k, positions)
        return self.o(attention(q, k, v).flatten(1))


class Rwkv7HybridDecoderLayer(nn.Module):
    def __init__(self, config, index: int):
        super().__init__()
        width = config.hidden_size
        self.input_layernorm = Rwkv7HybridLayerNorm(width, eps=1e-5)
        self.post_attention_layernorm = Rwkv7HybridLayerNorm(width, eps=1e-5)
        self.mixer = (
            Rwkv7HybridAttention(config) if index % 4 == 3 else Rwkv7HybridMixer(config)
        )
        self.up = nn.Linear(width, 4 * width, bias=True)
        self.down = nn.Linear(4 * width, width, bias=True)

    def forward(self, x, run_mixer: Callable):
        x = x + run_mixer(self.mixer, self.input_layernorm(x))
        return x + self.down(F.gelu(self.up(self.post_attention_layernorm(x))))


class Rwkv7HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Rwkv7HybridDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Rwkv7HybridLayerNorm(config.hidden_size, eps=1e-5)

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
        self.kv_cache[indices] = 0

    def copy_state_slots(self, sources, destinations):
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

    def run_recurrent(self, mixer, x, metadata, flash):
        indices = metadata.state_indices
        previous_pool, state_pool = self.state_views()
        previous = torch.roll(x, 1, 0)
        starts = metadata.cu_seqlens[:-1].long()
        previous[starts] = previous_pool[indices]
        # FlashRWKV2 requires contiguous state. The scheduler's strided page
        # remains authoritative, including copy-on-write and prefix snapshots.
        state = state_pool.index_select(0, indices).contiguous()
        handle = flash.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state)
        if getattr(metadata, "validated_metadata", None) is None:
            self.prepare_recurrent_metadata(metadata, flash)
        slots = metadata.recurrent_slots
        max_length = metadata.max_recurrent_seq_len
        ticket = metadata.validated_metadata

        def recur(r, decay, k, v, a, b):
            return flash.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                *(tensor.contiguous() for tensor in (r, decay, k, v, a, b)),
                state=handle,
                cu_seqlens=metadata.cu_seqlens,
                state_indices=slots,
                max_seqlen=max_length,
                validated_metadata=ticket,
            )

        output = mixer(x, previous, recur)
        state_pool[indices] = state
        previous_pool[indices] = x[metadata.cu_seqlens[1:].long() - 1]
        return output

    def run_attention(self, q, k, v, metadata):
        cache = self.kv_cache
        block_size = cache.shape[2]
        slots = metadata.slot_mapping.long()
        blocks, offsets = slots // block_size, slots % block_size
        cache[blocks, :, offsets, :64] = k
        cache[blocks, :, offsets, 64:] = v
        if q.is_cuda:
            paged = cache.transpose(1, 2)
            key_cache, value_cache = paged.split(64, dim=-1)
            output = torch.empty_like(q)
            _load_paged_fa2()(
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
                fa_version=2,
                out=output,
            )
            return output

        output = torch.empty_like(q)
        for row, (start, end, length) in enumerate(
            zip(metadata.starts, metadata.starts[1:], metadata.lengths)
        ):
            positions = torch.arange(length, device=q.device)
            pages = metadata.block_table[row, positions // block_size].long()
            history = cache[pages, :, positions % block_size]
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
