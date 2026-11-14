# SPDX-License-Identifier: Apache-2.0
"""Eager metadata for full RoPE attention and RWKV state in a shared block pool."""

from dataclasses import dataclass

import torch

from irwkv.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from irwkv.v1.attention.backends.utils import mamba_get_block_table_tensor
from irwkv.v1.kv_cache_interface import KVCacheLayout, MambaSpec


@dataclass
class Rwkv7HybridMetadata:
    starts: tuple[int, ...]
    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    state_indices: torch.Tensor | None
    token_capacity: int
    recurrent_slots: torch.Tensor | None = None
    max_recurrent_seq_len: int = 1
    validated_metadata: object | None = None


class Rwkv7HybridMetadataBuilder(AttentionMetadataBuilder[Rwkv7HybridMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names, irwkv_config, device):
        super().__init__(kv_cache_spec, layer_names, irwkv_config, device)
        self._state_layer = None
        if isinstance(kv_cache_spec, MambaSpec):
            self._state_layer = irwkv_config.compilation_config.static_forward_context[
                layer_names[0]
            ]
        self._live_state_indices = {}
        self._captured_descriptors = set()
        cudagraph_mode = irwkv_config.compilation_config.cudagraph_mode
        self._use_live_metadata = bool(
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )

    def _build(self, common_attn_metadata, *, for_capture):
        m = common_attn_metadata
        indices = None
        if isinstance(self.kv_cache_spec, MambaSpec):
            indices = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.irwkv_config.cache_config.mamba_cache_mode,
            )[:, 0].long()
            if self._use_live_metadata:
                live = self._live_state_indices.get(m.num_reqs)
                if live is None:
                    live = torch.empty_like(indices)
                    self._live_state_indices[m.num_reqs] = live
                live.copy_(indices)
                indices = live
        block_table = m.block_table_tensor
        seq_lens = m.seq_lens
        if block_table.dtype != torch.int32:
            block_table = block_table.int()
        if seq_lens.dtype != torch.int32:
            seq_lens = seq_lens.int()
        metadata = Rwkv7HybridMetadata(
            starts=tuple(m.query_start_loc_cpu.tolist()),
            cu_seqlens=m.query_start_loc,
            seq_lens=seq_lens,
            max_query_len=m.max_query_len,
            max_seq_len=m.max_seq_len,
            block_table=block_table,
            slot_mapping=m.slot_mapping,
            state_indices=indices,
            token_capacity=m.num_actual_tokens,
        )
        if self._state_layer is not None:
            descriptor = (m.num_reqs, m.num_actual_tokens, m.max_query_len)
            if for_capture:
                self._captured_descriptors.add(descriptor)
                self._state_layer.prepare_metadata_ticket(
                    metadata, retain_ticket=True
                )
            elif descriptor not in self._captured_descriptors:
                self._state_layer.prepare_metadata_ticket(metadata)
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self._build(common_attn_metadata, for_capture=True)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        return self._build(common_attn_metadata, for_capture=False)


class Rwkv7HybridAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name():
        return "RWKV7_HYBRID_FA2"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("The hybrid model executes paged FA2 directly")

    @staticmethod
    def get_builder_cls():
        return Rwkv7HybridMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def supports_kv_connector(cls):
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls):
        return False


class Rwkv7HybridRecurrentBackend(Rwkv7HybridAttentionBackend):
    @staticmethod
    def get_name():
        return "RWKV7_HYBRID_RECURRENT"

    @classmethod
    def is_ssm(cls):
        return True
