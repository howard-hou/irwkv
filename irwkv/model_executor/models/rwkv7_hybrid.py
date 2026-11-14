# SPDX-License-Identifier: Apache-2.0
"""bux timeline hybrid served by irwkv with scheduler-owned KV and RWKV state."""

import torch
from torch import nn

from irwkv.forward_context import get_forward_context
from irwkv.model_executor.layers.attention_layer_base import AttentionLayerBase
from irwkv.model_executor.models.interfaces import HasInnerState, IsHybrid
from irwkv.model_executor.models.rwkv import _load_flashrwkv2
from irwkv.model_executor.models.utils import AutoWeightsLoader
from irwkv.models.rwkv7_hybrid import Rwkv7HybridCacheStorage, Rwkv7HybridModel
from irwkv.v1.attention.backends.rwkv7_hybrid import (
    Rwkv7HybridAttentionBackend,
    Rwkv7HybridRecurrentBackend,
)
from irwkv.v1.attention.backends.rwkv_attn import RwkvStateSpec
from irwkv.v1.kv_cache_interface import FullAttentionSpec


class Rwkv7HybridCache(nn.Module, Rwkv7HybridCacheStorage, AttentionLayerBase):
    def __init__(self, irwkv_config, prefix, recurrent):
        super().__init__()
        self.irwkv_config = irwkv_config
        self.prefix = prefix
        self.recurrent = recurrent
        self.num_heads = irwkv_config.model_config.hf_config.num_attention_heads
        self._live_graph_tickets = []
        self.register_buffer("kv_cache", torch.empty(0), persistent=False)
        context = irwkv_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate cache layer {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return (
            Rwkv7HybridRecurrentBackend
            if self.recurrent
            else Rwkv7HybridAttentionBackend
        )

    def prepare_metadata_ticket(self, metadata, *, retain_ticket=False):
        return self.prepare_recurrent_metadata(
            metadata, _load_flashrwkv2(), retain_ticket=retain_ticket
        )

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
        return FullAttentionSpec(
            block_size=cache.block_size,
            num_kv_heads=self.num_heads,
            head_size=64,
            dtype=torch.bfloat16,
        )


class Rwkv7HybridForCausalLM(nn.Module, HasInnerState, IsHybrid):
    supports_mamba_prefix_caching = True

    def __init__(self, *, irwkv_config, prefix=""):
        super().__init__()
        self.config = irwkv_config.model_config.hf_config
        self.model = Rwkv7HybridModel(self.config)
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
            batch = metadata[cache.prefix]
            if cache.recurrent:
                return cache.run_recurrent(mixer, x, batch, _load_flashrwkv2())
            return mixer(
                x, positions, lambda q, k, v: cache.run_attention(q, k, v, batch)
            )

        return self.model(input_ids, run, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states).float()

    def load_weights(self, weights):
        return AutoWeightsLoader(self, skip_prefixes=["lm_head."]).load_weights(weights)
