# SPDX-License-Identifier: Apache-2.0
"""Eager metadata for full RoPE attention and RWKV state in a shared block pool."""

from dataclasses import dataclass

import torch

from irwkv import envs
from irwkv.config.compilation import CompilationMode
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
    seq_threshold_3D: int | None = None
    num_par_softmax_segments: int | None = None
    softmax_segm_output: torch.Tensor | None = None
    softmax_segm_max: torch.Tensor | None = None
    softmax_segm_expsum: torch.Tensor | None = None


class Rwkv7HybridMetadataBuilder(AttentionMetadataBuilder[Rwkv7HybridMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names, irwkv_config, device):
        super().__init__(kv_cache_spec, layer_names, irwkv_config, device)
        self._state_layer = None
        compilation = irwkv_config.compilation_config
        if (
            isinstance(kv_cache_spec, MambaSpec)
            and compilation.mode != CompilationMode.NONE
            and envs.IRWKV_RWKV_HYBRID_RECURRENT_BACKEND == "flashrwkv2"
        ):
            self._state_layer = compilation.static_forward_context[layer_names[0]]
        self._captured_descriptors = set()
        self._live_state_indices = {}
        cudagraph_mode = compilation.cudagraph_mode
        self._use_live_metadata = bool(
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )
        self._triton_fp8 = False
        self._max_seqs_64_segments = 0
        if not isinstance(kv_cache_spec, MambaSpec):
            from irwkv.platforms import current_platform

            capability = current_platform.get_device_capability()
            self._triton_fp8 = (
                kv_cache_spec.dtype == torch.float8_e4m3fn
                and capability is not None
                and capability.major == 12
            )
        if self._triton_fp8:
            from irwkv.utils.math_utils import cdiv, next_power_of_2

            num_heads = kv_cache_spec.num_kv_heads
            self._seq_threshold_3D = 128 // num_heads
            if self._use_live_metadata:
                capture_sizes = irwkv_config.compilation_config.cudagraph_capture_sizes
                assert capture_sizes
                self._seq_threshold_3D = min(
                    capture_sizes,
                    key=lambda x: abs(x - self._seq_threshold_3D),
                )
            self._num_par_softmax_segments = 16
            self._max_seqs_64_segments = (current_platform.num_compute_units() - 1) // (
                16 * num_heads
            )
            max_num_tokens_3d = self._seq_threshold_3D
            if self._max_seqs_64_segments > 0:
                self._num_par_softmax_segments = 64
                max_num_tokens_3d = max(
                    min(self._max_seqs_64_segments, max_num_tokens_3d),
                    cdiv(max_num_tokens_3d, self._num_par_softmax_segments // 16),
                )
            head_size_padded = next_power_of_2(kv_cache_spec.head_size)
            self._softmax_segm_output = torch.empty(
                (
                    max_num_tokens_3d,
                    num_heads,
                    self._num_par_softmax_segments,
                    head_size_padded,
                ),
                dtype=torch.float32,
                device=device,
            )
            self._softmax_segm_max = torch.empty(
                (max_num_tokens_3d, num_heads, self._num_par_softmax_segments),
                dtype=torch.float32,
                device=device,
            )
            self._softmax_segm_expsum = torch.empty_like(self._softmax_segm_max)

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
        triton_metadata = {}
        if self._triton_fp8:
            triton_metadata = dict(
                seq_threshold_3D=self._seq_threshold_3D,
                num_par_softmax_segments=self._num_par_softmax_segments,
                softmax_segm_output=self._softmax_segm_output,
                softmax_segm_max=self._softmax_segm_max,
                softmax_segm_expsum=self._softmax_segm_expsum,
            )
            if self._max_seqs_64_segments > 0 and (
                self._max_seqs_64_segments < m.num_reqs
                or self._softmax_segm_max.shape[0] < m.num_actual_tokens
            ):
                segments = 16
                triton_metadata.update(
                    num_par_softmax_segments=segments,
                    softmax_segm_output=self._softmax_segm_output.view(
                        -1,
                        self.kv_cache_spec.num_kv_heads,
                        segments,
                        self._softmax_segm_output.shape[-1],
                    ),
                    softmax_segm_max=self._softmax_segm_max.view(
                        -1, self.kv_cache_spec.num_kv_heads, segments
                    ),
                    softmax_segm_expsum=self._softmax_segm_expsum.view(
                        -1, self.kv_cache_spec.num_kv_heads, segments
                    ),
                )
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
            **triton_metadata,
        )
        if self._state_layer is not None:
            descriptor = (m.num_reqs, m.num_actual_tokens, m.max_query_len)
            if for_capture:
                self._captured_descriptors.add(descriptor)
                self._state_layer.prepare_metadata_ticket(metadata, retain_ticket=True)
            elif descriptor not in self._captured_descriptors:
                self._state_layer.prepare_metadata_ticket(metadata)
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self._build(common_attn_metadata, for_capture=True)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        return self._build(common_attn_metadata, for_capture=False)


class Rwkv7HybridAttentionBackend(AttentionBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8"]

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
