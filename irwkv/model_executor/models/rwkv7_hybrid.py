# SPDX-License-Identifier: Apache-2.0
"""bux timeline hybrid served by irwkv with scheduler-owned KV and RWKV state."""

import torch
from torch import nn

from irwkv import envs
from irwkv.compilation.decorators import support_torch_compile
from irwkv.config.compilation import CompilationMode
from irwkv.forward_context import get_forward_context
from irwkv.model_executor.layers.attention_layer_base import AttentionLayerBase
from irwkv.model_executor.models.interfaces import HasInnerState, IsHybrid
from irwkv.model_executor.models.rwkv import _load_flashrwkv2
from irwkv.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from irwkv.models.rwkv7_hybrid import (
    Rwkv7HybridCacheStorage,
    Rwkv7HybridModel,
    _load_albatross,
    _register_albatross_fakes,
)
from irwkv.utils.torch_utils import direct_register_custom_op
from irwkv.v1.attention.backends.rwkv7_hybrid import (
    Rwkv7HybridAttentionBackend,
    Rwkv7HybridRecurrentBackend,
)
from irwkv.v1.attention.backends.rwkv_attn import RwkvStateSpec
from irwkv.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode


def _layer_and_metadata(layer_name: str):
    context = get_forward_context()
    return (
        context.no_compile_layers[layer_name],
        context.attn_metadata[layer_name],
    )


def rwkv7_hybrid_shift(
    x: torch.Tensor, out: torch.Tensor, shift: torch.Tensor, layer_name: str
) -> None:
    layer, metadata = _layer_and_metadata(layer_name)
    indices = metadata.state_indices
    previous_pool, _ = layer.state_views()
    previous = torch.roll(x, 1, 0)
    starts = metadata.cu_seqlens[:-1].long()
    previous[starts] = previous_pool[indices]
    out.copy_(x + (previous - x) * shift[:, None])
    previous_pool[indices] = x[metadata.cu_seqlens[1:].long() - 1]


def rwkv7_hybrid_recur(
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer, metadata = _layer_and_metadata(layer_name)
    indices = metadata.state_indices
    _, state_pool = layer.state_views()
    tensors = tuple(tensor.contiguous() for tensor in (r, decay, k, v, a, b))
    if envs.IRWKV_RWKV_HYBRID_RECURRENT_BACKEND == "flashrwkv2":
        state = state_pool.index_select(0, indices).contiguous()
        flash = _load_flashrwkv2()
        handle = flash.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state)
        if metadata.validated_metadata is None:
            layer.prepare_recurrent_metadata(metadata, flash)
        out.copy_(
            flash.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                *tensors,
                state=handle,
                cu_seqlens=metadata.cu_seqlens,
                state_indices=metadata.recurrent_slots,
                max_seqlen=metadata.max_recurrent_seq_len,
                validated_metadata=metadata.validated_metadata,
            )
        )
        state_pool[indices] = state
        return
    ops = _load_albatross()
    if metadata.max_query_len == 1:
        ops.forward_decode(state_pool, indices, *tensors, out)
    else:
        ops.forward_prefill(state_pool, indices, metadata.cu_seqlens, *tensors, out)


def rwkv7_hybrid_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer, metadata = _layer_and_metadata(layer_name)
    layer.run_attention(q, k, v, metadata, positions, rotary=True, out=out)


direct_register_custom_op(
    op_name="rwkv7_hybrid_shift",
    op_func=rwkv7_hybrid_shift,
    mutates_args=["out"],
)
direct_register_custom_op(
    op_name="rwkv7_hybrid_recur",
    op_func=rwkv7_hybrid_recur,
    mutates_args=["out"],
)
direct_register_custom_op(
    op_name="rwkv7_hybrid_attention",
    op_func=rwkv7_hybrid_attention,
    mutates_args=["out"],
)


class Rwkv7HybridCache(nn.Module, Rwkv7HybridCacheStorage, AttentionLayerBase):
    def __init__(self, irwkv_config, prefix, recurrent):
        super().__init__()
        self.irwkv_config = irwkv_config
        self.prefix = prefix
        self.recurrent = recurrent
        self.num_heads = irwkv_config.model_config.hf_config.num_attention_heads
        self._live_graph_tickets = []
        self.register_buffer("kv_cache", torch.empty(0), persistent=False)
        for name in ("q_scale", "k_scale", "v_scale"):
            self.register_buffer(name, torch.ones(1), persistent=False)
        compilation = irwkv_config.compilation_config
        self.use_compiled_ops = compilation.mode != CompilationMode.NONE
        if self.use_compiled_ops:
            _register_albatross_fakes()
        context = compilation.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate cache layer {prefix}")
        context[prefix] = self
        if self.use_compiled_ops and compilation.splitting_ops is not None:
            names = (
                ("rwkv7_hybrid_shift", "rwkv7_hybrid_recur")
                if recurrent
                else ("rwkv7_hybrid_attention",)
            )
            for name in names:
                op = f"irwkv::{name}"
                if op not in compilation.splitting_ops:
                    compilation.splitting_ops.append(op)

    def get_attn_backend(self):
        return (
            Rwkv7HybridRecurrentBackend
            if self.recurrent
            else Rwkv7HybridAttentionBackend
        )

    def prepare_metadata_ticket(self, metadata, *, retain_ticket=False):
        if (
            not self.use_compiled_ops
            or envs.IRWKV_RWKV_HYBRID_RECURRENT_BACKEND != "flashrwkv2"
        ):
            return None
        return self.prepare_recurrent_metadata(
            metadata, _load_flashrwkv2(), retain_ticket=retain_ticket
        )

    def run_recurrent_compiled(self, mixer, x):
        shifted = x.new_empty((6, *x.shape))
        torch.ops.irwkv.rwkv7_hybrid_shift(x, shifted, mixer.shift, self.prefix)
        r, decay, k, v, a, b, g = mixer.prepare(shifted)
        out = torch.empty_like(r)
        torch.ops.irwkv.rwkv7_hybrid_recur(r, decay, k, v, a, b, out, self.prefix)
        return mixer.post(out, r, k, v, g)

    def run_attention_compiled(self, mixer, x, positions):
        return mixer(
            x,
            positions,
            lambda q, k, v: self._attention_op(q, k, v, positions),
            fused_decode=True,
        )

    def _attention_op(self, q, k, v, positions):
        out = torch.empty_like(q)
        torch.ops.irwkv.rwkv7_hybrid_attention(q, k, v, positions, out, self.prefix)
        return out

    def get_kv_cache_spec(self, irwkv_config):
        cache = irwkv_config.cache_config
        width = self.num_heads * 64
        if self.recurrent:
            return RwkvStateSpec(
                block_size=cache.mamba_block_size,
                shapes=((width,), (self.num_heads, 64, 64)),
                dtypes=(torch.bfloat16, torch.float32),
                page_size_padded=cache.mamba_page_size_padded,
                mamba_cache_mode=cache.mamba_cache_mode,
            )
        fp8 = cache.cache_dtype == "fp8"
        return FullAttentionSpec(
            block_size=cache.block_size,
            num_kv_heads=self.num_heads,
            head_size=64,
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR if fp8 else KVQuantMode.NONE,
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Rwkv7HybridForCausalLM(nn.Module, HasInnerState, IsHybrid):
    supports_mamba_prefix_caching = True

    def __init__(self, *, irwkv_config, prefix=""):
        super().__init__()
        self.config = irwkv_config.model_config.hf_config
        quantize = irwkv_config.model_config.quantization == "fp8"
        self.model = Rwkv7HybridModel(self.config, quantize=quantize)
        self.caches = nn.ModuleList(
            Rwkv7HybridCache(
                irwkv_config,
                f"{prefix + '.' if prefix else ''}model.layers.{i}.cache",
                i % 4 != 3,
            )
            for i in range(self.config.num_hidden_layers)
        )
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def get_mamba_state_shape_from_config(cls, irwkv_config):
        width = irwkv_config.model_config.hf_config.hidden_size
        return ((width,), (width // 64, 64, 64))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, irwkv_config):
        return (torch.bfloat16, torch.float32)

    @staticmethod
    def get_model_state_cls():
        from irwkv.v1.worker.gpu.model_states.rwkv7_hybrid import Rwkv7HybridModelState

        return Rwkv7HybridModelState

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        metadata = get_forward_context().attn_metadata

        def run(index, mixer, x):
            cache = self.caches[index]
            if metadata is None:
                # Memory profiling executes projections without touching unbound caches.
                if cache.recurrent:
                    return mixer(
                        x, torch.zeros_like(x), lambda r, *args: torch.zeros_like(r)
                    )
                return mixer(x, positions, lambda q, k, v: torch.zeros_like(q))
            if cache.use_compiled_ops:
                if cache.recurrent:
                    return cache.run_recurrent_compiled(mixer, x)
                return cache.run_attention_compiled(mixer, x, positions)
            batch = metadata[cache.prefix]
            if cache.recurrent:
                return cache.run_recurrent(mixer, x, batch)
            return mixer(
                x,
                positions,
                lambda q, k, v: cache.run_attention(
                    q, k, v, batch, positions=positions
                ),
                fused_decode=batch.max_query_len == 1,
            )

        return self.model(input_ids, run, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states).float()

    def process_weights_after_loading(self):
        self.model.process_weights_after_loading()

    def load_weights(self, weights):
        mapper = WeightsMapper(
            orig_to_new_stacked={
                ".mixer.q.": (".mixer.qkv.", 0),
                ".mixer.k.": (".mixer.qkv.", 1),
                ".mixer.v.": (".mixer.qkv.", 2),
            }
        )
        return AutoWeightsLoader(self, skip_prefixes=["lm_head."]).load_weights(
            weights, mapper=mapper
        )
