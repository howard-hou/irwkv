# SPDX-License-Identifier: Apache-2.0
"""Eager metadata for full RoPE attention and RWKV state in a shared block pool."""

from dataclasses import dataclass

import torch

from irwkv.v1.attention.backend import AttentionBackend, AttentionMetadataBuilder
from irwkv.v1.attention.backends.utils import mamba_get_block_table_tensor
from irwkv.v1.kv_cache_interface import KVCacheLayout, MambaSpec


@dataclass
class Rwkv7HybridMetadata:
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    cu_seqlens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    state_indices: torch.Tensor | None


class Rwkv7HybridMetadataBuilder(AttentionMetadataBuilder[Rwkv7HybridMetadata]):
    def __init__(self, kv_cache_spec, layer_names, irwkv_config, device):
        super().__init__(kv_cache_spec, layer_names, irwkv_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        m = common_attn_metadata
        indices = None
        if isinstance(self.kv_cache_spec, MambaSpec):
            indices = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.irwkv_config.cache_config.mamba_cache_mode,
            )[:, 0].long()
        return Rwkv7HybridMetadata(
            tuple(m.query_start_loc_cpu.tolist()),
            tuple(m.seq_lens.cpu().tolist()),
            m.query_start_loc,
            m.block_table_tensor,
            m.slot_mapping,
            indices,
        )


class Rwkv7HybridAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name():
        return "RWKV7_HYBRID_SDPA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("The hybrid model executes PyTorch SDPA directly")

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
