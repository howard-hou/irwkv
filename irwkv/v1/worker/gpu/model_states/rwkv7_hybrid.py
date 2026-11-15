# SPDX-License-Identifier: Apache-2.0
"""Joint block lifecycle for the bux hybrid's attention and recurrent layers."""

import numpy as np
import torch

from irwkv.v1.worker.gpu.model_states.default import DefaultModelState


class Rwkv7HybridModelState(DefaultModelState):
    def __init__(self, irwkv_config, model, encoder_cache, device):
        super().__init__(irwkv_config, model, encoder_cache, device)
        self._computed = np.zeros(self.max_num_reqs, dtype=np.int64)
        self._groups = []

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        self._groups = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            caches = [
                cache
                for cache in self.model.caches
                if cache.recurrent and cache.prefix in group.layer_names
            ]
            if caches:
                self._groups.append((group_id, group.kv_cache_spec.block_size, caches))

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self._computed[req_index] = new_req_data.num_computed_tokens

    def reset_kv_cache_blocks(self, block_ids):
        for cache in self.model.caches:
            if cache.recurrent:
                cache.reset_state_slots(block_ids)
        # The runner must also zero full-attention pages.
        return False

    def preprocess_state(
        self, input_batch, block_tables, kv_cache_config, num_computed_tokens
    ):
        count = input_batch.num_reqs
        requests = input_batch.idx_mapping_np[:count]
        computed = self._computed[requests]
        end = computed + input_batch.num_scheduled_tokens[:count]
        if self.irwkv_config.cache_config.mamba_cache_mode == "align":
            for group_id, block_size, caches in self._groups:
                old, new = (computed - 1) // block_size, (end - 1) // block_size
                rows = np.flatnonzero((old >= 0) & (old != new))
                fresh = np.flatnonzero(old < 0)
                table = block_tables[group_id]
                if fresh.size:
                    fresh_gpu = torch.as_tensor(fresh, device=self.device)
                    fresh_columns = torch.as_tensor(new[fresh], device=self.device)
                    fresh_slots = table[fresh_gpu, fresh_columns].long()
                    for cache in caches:
                        cache.reset_state_slots(fresh_slots)
                if rows.size:
                    rows_gpu = torch.as_tensor(rows, device=self.device)
                    old_gpu = torch.as_tensor(old[rows], device=self.device)
                    new_gpu = torch.as_tensor(new[rows], device=self.device)
                    sources = table[rows_gpu, old_gpu].long()
                    destinations = table[rows_gpu, new_gpu].long()
                    for cache in caches:
                        cache.copy_state_slots(sources, destinations)
        self._computed[requests] = end
