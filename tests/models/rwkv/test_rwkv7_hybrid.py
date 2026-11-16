# SPDX-License-Identifier: Apache-2.0
"""Full hybrid parity and scheduler-page storage regressions.

The tensor-only tests also run with pytest --noconftest on Windows.
"""

import copy
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from irwkv.models.rwkv7_hybrid import (
    Rwkv7HybridAttention,
    Rwkv7HybridCacheStorage,
    Rwkv7HybridLayerNorm,
    Rwkv7HybridModel,
    reference_recurrence,
)
from irwkv.transformers_utils.rwkv7_hybrid import Rwkv7HybridConfig


@pytest.mark.parametrize(
    "width,layers,total,active",
    [
        (3328, 52, 7148454144, 6981042432),
        (3584, 56, 8893208576, 8712919040),
        (3840, 60, 10903031040, 10709863680),
    ],
)
def test_full_model_matches_published_parameter_counts(width, layers, total, active):
    config = Rwkv7HybridConfig(hidden_size=width)
    with torch.device("meta"):
        model = Rwkv7HybridModel(config)
    assert len(model.layers) == layers
    assert sum(p.numel() for p in model.parameters()) == total
    assert config.parameter_count() == {"total": total, "active": active}
    assert not any("v_mix" in name for name, _ in model.named_parameters())
    assert sum(kind == "attention" for kind in config.layers_block_type) == layers // 4


def test_config_round_trip_and_reject_architecture_drift(tmp_path):
    config = Rwkv7HybridConfig(hidden_size=3584)
    config.save_pretrained(tmp_path)
    loaded = Rwkv7HybridConfig.from_pretrained(tmp_path)
    assert loaded.parameter_count() == config.parameter_count()
    for override in (
        {"head_size": 128},
        {"tie_word_embeddings": False},
        {"intermediate_size": 1024},
        {"num_hidden_layers": 7},
    ):
        with pytest.raises(ValueError):
            Rwkv7HybridConfig(**override)


def initialized_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(71)
    config = Rwkv7HybridConfig(hidden_size=64, num_hidden_layers=8, vocab_size=128)
    model = Rwkv7HybridModel(config).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and (
                name.endswith("weight") or name.endswith("norm_weight")
            ):
                parameter.fill_(1)
            else:
                parameter.uniform_(-0.1, 0.1)
    return model


@torch.inference_mode()
def test_fused_qkv_matches_separate_projection_and_shard_loading():
    model = initialized_model()
    mixer = model.layers[3].mixer
    weight = mixer.qkv.weight.detach().clone()
    bias = mixer.qkv.bias.detach().clone()
    x = torch.randn(5, 64)
    expected = [F.linear(x, w, b) for w, b in zip(weight.chunk(3), bias.chunk(3))]
    torch.testing.assert_close(list(mixer.qkv(x).chunk(3, dim=-1)), expected)

    mixer.qkv.weight.zero_()
    mixer.qkv.bias.zero_()
    for shard, (w, b) in enumerate(zip(weight.chunk(3), bias.chunk(3))):
        for param, tensor in ((mixer.qkv.weight, w), (mixer.qkv.bias, b)):
            tensor.shard_id = shard
            param.weight_loader(param, tensor)
    torch.testing.assert_close(mixer.qkv.weight, weight)
    torch.testing.assert_close(mixer.qkv.bias, bias)
    torch.testing.assert_close(list(mixer.qkv(x).chunk(3, dim=-1)), expected)


def run_reference(model, tokens, state, offset):
    positions = torch.arange(offset, offset + len(tokens), device=tokens.device)

    def run(index, mixer, x):
        if index % 4 != 3:
            previous, matrix = state.get(
                index,
                (
                    torch.zeros_like(x[0]),
                    torch.zeros(1, 64, 64, device=x.device),
                ),
            )
            shifted = torch.cat((previous[None], x[:-1]))

            def recur(*arguments):
                output, final = reference_recurrence(*arguments, matrix)
                state[index] = (x[-1].clone(), final)
                return output

            return mixer(x, shifted, recur)

        def attention(q, k, v):
            if index in state:
                old_k, old_v = state[index]
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
            state[index] = (k.clone(), v.clone())
            mask = torch.arange(len(k), device=x.device)[None] <= positions[:, None]
            return F.scaled_dot_product_attention(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                attn_mask=mask,
            ).transpose(0, 1)

        return mixer(x, positions, attention)

    return F.linear(model(tokens, run), model.embed_tokens.weight)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
@torch.inference_mode()
def test_full_model_chunked_prefill_decode_and_forked_prefix(device):
    model = initialized_model(device)
    tokens = torch.arange(19, device=device)
    expected = run_reference(model, tokens, {}, 0)
    state = {}
    chunks = []
    for start, end in ((0, 5), (5, 11), (11, 18), (18, 19)):
        chunks.append(run_reference(model, tokens[start:end], state, start))
    torch.testing.assert_close(torch.cat(chunks), expected, atol=2e-5, rtol=2e-5)
    prefix_state = {}
    run_reference(model, tokens[:11], prefix_state, 0)
    snapshot = copy.deepcopy(prefix_state)
    branch_a, branch_b = copy.deepcopy(prefix_state), copy.deepcopy(prefix_state)
    a = run_reference(model, tokens[11:], branch_a, 11)
    other = tokens[11:].flip(0)
    b = run_reference(model, other, branch_b, 11)
    torch.testing.assert_close(a, expected[11:], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        b,
        run_reference(model, torch.cat((tokens[:11], other)), {}, 0)[11:],
        atol=2e-5,
        rtol=2e-5,
    )
    for index in snapshot:
        for actual, saved in zip(prefix_state[index], snapshot[index]):
            torch.testing.assert_close(actual, saved, atol=0, rtol=0)


@torch.inference_mode()
def test_paged_attention_isolates_requests_and_offsets_chunk_causality():
    torch.manual_seed(9)
    storage = Rwkv7HybridCacheStorage()
    storage.kv_cache = torch.zeros(9, 2, 4, 128)
    # Two disjoint requests, deliberately nonsequential page IDs.
    tables = torch.tensor([[3, 1, 7], [4, 6, 8]])
    q, k, v = [torch.randn(2, 9, 2, 64) for _ in range(3)]
    for begin, end in ((0, 3), (3, 8), (8, 9)):
        positions = torch.arange(begin, end)
        slots = (tables[:, positions // 4] * 4 + positions % 4).flatten()
        batch = SimpleNamespace(
            starts=(0, end - begin, 2 * (end - begin)),
            lengths=(end, end),
            block_table=tables,
            slot_mapping=slots,
        )
        inputs = [value[:, begin:end].flatten(0, 1) for value in (q, k, v)]
        output = storage.run_attention(*inputs, batch).unflatten(0, (2, end - begin))
        mask = torch.arange(end)[None] <= positions[:, None]
        expected = F.scaled_dot_product_attention(
            q[:, begin:end].transpose(1, 2),
            k[:, :end].transpose(1, 2),
            v[:, :end].transpose(1, 2),
            attn_mask=mask,
        ).transpose(1, 2)
        torch.testing.assert_close(output, expected)


def test_recurrent_page_padding_copy_and_reset_preserve_other_slots():
    storage = Rwkv7HybridCacheStorage()
    storage.num_heads = 2
    content_bytes = 2 * 64 * 2 + 2 * 64 * 64 * 4
    page_bytes = content_bytes + 512
    raw = torch.zeros(5 * page_bytes, dtype=torch.int8)
    storage.kv_cache = raw.as_strided(
        (5, 1, 1, content_bytes), (page_bytes, content_bytes, content_bytes, 1)
    )
    shift, state = storage.state_views()
    shift[1].fill_(2)
    state[1].fill_(3)
    storage.copy_state_slots(torch.tensor([1]), torch.tensor([3]))
    storage.reset_state_slots([1])
    assert shift[1].count_nonzero() == state[1].count_nonzero() == 0
    assert torch.all(shift[3] == 2) and torch.all(state[3] == 3)
    assert raw.view(5, page_bytes)[:, content_bytes:].count_nonzero() == 0


def test_empty_recurrent_state_operations_do_not_touch_cache():
    class ForbiddenCache:
        def __getitem__(self, indices):
            raise AssertionError("empty state operation accessed the cache")

        def __setitem__(self, indices, value):
            raise AssertionError("empty state operation wrote the cache")

    storage = Rwkv7HybridCacheStorage()
    storage.kv_cache = ForbiddenCache()
    empty = torch.empty(0, dtype=torch.long)
    storage.reset_state_slots(empty)
    storage.copy_state_slots(empty, empty)


def test_recurrent_preprocess_skips_steady_decode_and_preserves_page_transitions():
    np = pytest.importorskip("numpy")
    from irwkv.v1.worker.gpu.model_states.rwkv7_hybrid import Rwkv7HybridModelState

    class RecordingCache:
        def __init__(self):
            self.resets = []
            self.copies = []

        def reset_state_slots(self, indices):
            self.resets.append(indices.tolist())

        def copy_state_slots(self, sources, destinations):
            self.copies.append((sources.tolist(), destinations.tolist()))

    caches = [RecordingCache(), RecordingCache()]
    state = object.__new__(Rwkv7HybridModelState)
    state.irwkv_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align")
    )
    state.device = torch.device("cpu")
    state._computed = np.array([0], dtype=np.int64)
    state._groups = [(0, 80, caches)]
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        num_scheduled_tokens=np.array([1]),
    )
    tables = [torch.tensor([[4, 5]])]

    state.preprocess_state(batch, tables, None, None)
    for cache in caches:
        assert cache.resets == [[4]]
        assert cache.copies == []

    state.preprocess_state(batch, tables, None, None)
    for cache in caches:
        assert cache.resets == [[4]]
        assert cache.copies == []

    batch.num_scheduled_tokens[0] = 79
    state.preprocess_state(batch, tables, None, None)
    for cache in caches:
        assert cache.resets == [[4]]
        assert cache.copies == [([4], [5])]


@pytest.mark.skipif(
    os.environ.get("IRWKV_TEST_FLASH_HYBRID") != "1",
    reason="Requires the installed Linux CUDA engine environment",
)
def test_engine_config_preserves_hybrid_geometry_and_cache_mode(tmp_path):
    from irwkv.engine.arg_utils import EngineArgs

    Rwkv7HybridConfig(
        hidden_size=64, num_hidden_layers=4, vocab_size=128
    ).save_pretrained(tmp_path)
    config = EngineArgs(
        model=str(tmp_path),
        load_format="dummy",
        dtype="bfloat16",
        skip_tokenizer_init=True,
        enable_prefix_caching=True,
        max_model_len=1024,
        max_num_batched_tokens=128,
    ).create_engine_config()
    model, parallel, cache = (
        config.model_config,
        config.parallel_config,
        config.cache_config,
    )
    assert model.get_num_layers_by_block_type(parallel, "attention") == 1
    assert model.get_num_layers_by_block_type(parallel, "mamba") == 3
    assert model.get_num_kv_heads(parallel) == 1
    assert not model.enforce_eager
    assert config.compilation_config.mode.name == "NONE"
    assert config.compilation_config.cudagraph_mode.name == "FULL"
    assert config.compilation_config.cudagraph_capture_sizes[-1] == 128
    assert cache.mamba_cache_mode == "align"
    assert cache.mamba_ssm_cache_dtype == "float32"


@pytest.mark.skipif(
    os.environ.get("IRWKV_TEST_FLASH_HYBRID") != "1",
    reason="Set IRWKV_TEST_FLASH_HYBRID=1 on Linux with RWKV CUDA backends",
)
@torch.inference_mode()
def test_fused_layernorm_matches_eager_bf16_rounding():
    module = Rwkv7HybridLayerNorm(128).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.uniform_(-0.2, 0.2)
        module.bias.uniform_(-0.1, 0.1)
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    centered = x.float() - x.float().mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(
        centered.square().mean(-1, keepdim=True) + module.eps
    )
    expected = normalized.to(x.dtype) * module.weight + module.bias
    torch.testing.assert_close(module(x), expected, atol=0.01, rtol=0.02)


@pytest.mark.skipif(
    os.environ.get("IRWKV_TEST_FLASH_HYBRID") != "1",
    reason="Set IRWKV_TEST_FLASH_HYBRID=1 on Linux with RWKV CUDA backends",
)
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.inference_mode()
def test_shift_decode_matches_eager_and_updates_nonsequential_pages(batch_size):
    from irwkv import rwkv7_albatross_ops  # noqa: F401

    x = torch.randn(batch_size, 64, device="cuda", dtype=torch.bfloat16)
    previous = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    expected_previous = previous.clone()
    indices = torch.tensor([5, 2][:batch_size], device="cuda", dtype=torch.int64)
    shift = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    old = expected_previous[indices].clone()
    expected = x + (old - x) * shift[:, None]
    expected_previous[indices] = x
    actual = torch.ops.irwkv7_albatross.shift_decode(x, previous, indices, shift)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(previous, expected_previous, atol=0, rtol=0)


@pytest.mark.skipif(
    os.environ.get("IRWKV_TEST_FLASH_HYBRID") != "1",
    reason="Set IRWKV_TEST_FLASH_HYBRID=1 on Linux with RWKV CUDA backends",
)
@torch.inference_mode()
def test_attention_prepare_decode_matches_strided_eager_views_and_pages():
    from irwkv import rwkv7_albatross_ops  # noqa: F401

    batch_size, heads, block_size = 2, 2, 8
    projection = torch.randn(
        batch_size, 3 * heads * 64, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = [
        value.unflatten(-1, (heads, 64))
        for value in projection.chunk(3, dim=-1)
    ]
    positions = torch.tensor([17, 23], device="cuda", dtype=torch.int64)
    slots = torch.tensor([47, -1], device="cuda", dtype=torch.int32)
    cache = torch.randn(
        11, heads, block_size, 128, device="cuda", dtype=torch.bfloat16
    )
    expected_cache = cache.clone()
    blocks, offsets = slots.long() // block_size, slots.long() % block_size
    expected_cache[blocks, :, offsets, :64] = Rwkv7HybridAttention.rotary(
        k, positions
    )
    expected_cache[blocks, :, offsets, 64:] = v
    actual_q, actual_k = torch.ops.irwkv7_albatross.attention_prepare_decode(
        q, k, v, positions, slots, cache
    )
    torch.testing.assert_close(
        actual_q, Rwkv7HybridAttention.rotary(q, positions), atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(actual_k, expected_cache[blocks, :, offsets, :64])
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)


@pytest.mark.skipif(
    os.environ.get("IRWKV_TEST_FLASH_HYBRID") != "1",
    reason="Set IRWKV_TEST_FLASH_HYBRID=1 on Linux with RWKV CUDA backends",
)
@torch.inference_mode()
def test_albatross_decode_fusion_matches_eager():
    model = initialized_model("cuda", torch.bfloat16)
    mixer = model.layers[0].mixer
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    previous = torch.randn_like(x)
    recurrence_output = torch.randn(
        3, 1, 64, device="cuda", dtype=torch.bfloat16
    )

    def run(fused_decode):
        captured = {}

        def recur(r, decay, k, v, a, b):
            values = (r, decay, k, v, a, b)
            for name, value in zip(("r", "decay", "k", "v", "a", "b"), values):
                captured[name] = value.clone()
            return recurrence_output

        output = mixer(x, previous, recur, fused_decode=fused_decode)
        return output, captured

    expected_output, expected = run(False)
    actual_output, actual = run(True)
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], atol=0.002, rtol=0.02
        )
    torch.testing.assert_close(actual_output, expected_output, atol=0.002, rtol=0.02)


@pytest.mark.skipif(
    os.environ.get("IRWKV_TEST_FLASH_HYBRID") != "1",
    reason="Set IRWKV_TEST_FLASH_HYBRID=1 on Linux with RWKV CUDA backends",
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("recurrent_backend", ["flashrwkv2", "albatross"])
@torch.inference_mode()
def test_full_hybrid_recurrence_and_paged_attention(batch_size, recurrent_backend):
    """Exercise real serving storage and recurrent backends through decode."""
    import flashrwkv2

    model = initialized_model("cuda", torch.bfloat16)
    bounds = [(0, 5, 11, 18, 19), (0, 4, 9, 16, 17)][:batch_size]
    tokens = [
        torch.arange(row[-1], device="cuda") + 23 * i for i, row in enumerate(bounds)
    ]
    expected = [run_reference(model, row, {}, 0) for row in tokens]
    caches = []
    for index in range(8):
        cache = Rwkv7HybridCacheStorage()
        cache.num_heads = 1
        cache.use_albatross = recurrent_backend == "albatross"
        if index % 4 != 3:
            cache.kv_cache = torch.zeros(
                12, 1, 1, 128 + 64 * 64 * 4, dtype=torch.int8, device="cuda"
            )
        else:
            cache.kv_cache = torch.zeros(
                12, 1, 16, 128, dtype=torch.bfloat16, device="cuda"
            )
        caches.append(cache)
    pages = torch.tensor(
        [[3, 1], [8, 6]][:batch_size], dtype=torch.int32, device="cuda"
    )
    outputs = [[] for _ in bounds]
    for step in range(4):
        ranges = [
            torch.arange(row[step], row[step + 1], device="cuda") for row in bounds
        ]
        positions = torch.cat(ranges)
        starts = [0]
        for row in ranges:
            starts.append(starts[-1] + len(row))
        metadata = SimpleNamespace(
            starts=tuple(starts),
            lengths=tuple(row[step + 1] for row in bounds),
            cu_seqlens=torch.tensor(starts, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor(
                [row[step + 1] for row in bounds], dtype=torch.int32, device="cuda"
            ),
            max_query_len=max(len(row) for row in ranges),
            max_seq_len=max(row[step + 1] for row in bounds),
            state_indices=torch.tensor([2, 4][:batch_size], device="cuda"),
            block_table=pages,
            slot_mapping=torch.cat(
                [pages[i, row // 16] * 16 + row % 16 for i, row in enumerate(ranges)]
            ),
        )

        def run(index, mixer, x, metadata=metadata, positions=positions):
            cache = caches[index]
            if index % 4 != 3:
                return cache.run_recurrent(mixer, x, metadata, flashrwkv2)
            return mixer(
                x,
                positions,
                lambda q, k, v: cache.run_attention(
                    q, k, v, metadata, positions=positions
                ),
                fused_decode=metadata.max_query_len == 1,
            )

        packed = torch.cat([row[indices] for row, indices in zip(tokens, ranges)])
        logits = F.linear(model(packed, run), model.embed_tokens.weight)
        for i in range(batch_size):
            outputs[i].append(logits[starts[i] : starts[i + 1]])
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(torch.cat(actual), reference, atol=0.04, rtol=0.04)


@pytest.mark.skipif(
    not os.environ.get("BUX_HF_SOURCE"),
    reason="Set BUX_HF_SOURCE to the pinned bux src/bux/export/hf directory",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Known BF16 discrepancy against bux HF export; "
                "low-rank bias and log-decay rounding paths differ",
            ),
        ),
    ],
)
@torch.inference_mode()
def test_full_model_against_original_bux_modules(dtype, monkeypatch):
    """Compare the full stack to original bux RWKV, norms, FFN and RoPE.

    Attention uses an explicit dense softmax oracle. No bux source is copied or
    patched, and this test needs neither its JAX training package nor a GPU.
    """
    directory = Path(os.environ["BUX_HF_SOURCE"])
    name = "bux_hf_reference"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py")
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    common = importlib.import_module(f"{name}.common")
    basic = importlib.import_module(f"{name}.basic")
    recurrent = importlib.import_module(f"{name}.rwkv7")
    rope = importlib.import_module(f"{name}.rope").Rope(
        {"fraction": 1.0, "base": 10000.0, "layout": "half", "scaling": None}, 64
    )
    model = initialized_model(dtype=dtype)
    tokens = torch.arange(73)
    expected = run_reference(model, tokens, {}, 0)
    ctx = common.Ctx(
        tokens[None],
        tokens[None],
        None,
        None,
        torch.nn.ModuleDict(),
        lambda *args: None,
    )
    x = model.embed_tokens(tokens)[None]
    norm_config = common.norm_config("layernorm", 1e-5, bias=True)
    for index, layer in enumerate(model.layers):
        norm = common.Norm(norm_config, 64).to(dtype=dtype)
        norm.load_state_dict(layer.input_layernorm.state_dict())
        value = norm(x)
        source = layer.mixer
        if index % 4 != 3:
            mixer = recurrent.RWKV7(
                dict(
                    conv_size=None,
                    num_heads=1,
                    num_v_heads=None,
                    head_dim=64,
                    v_head_dim=None,
                    decay_rank=64,
                    a_rank=64,
                    v_rank=32,
                    output_gate="sigmoid",
                    gate_rank=128,
                    norm_eps=1e-5,
                ),
                64,
                common.Site(None, f"layer.{index}"),
            ).to(dtype=dtype)
            weights = source.state_dict()
            converted = {}
            for key in mixer.state_dict():
                if key.startswith("norm."):
                    converted[key] = weights[key.replace("norm.", "norm_")]
                elif key in ("w.down", "a.down", "g.down", "w.up", "a.up", "g.up"):
                    converted[key] = weights[key + ".weight"].T
                elif key in ("w.bias", "a.bias"):
                    converted[key] = weights[key.replace(".bias", ".up.bias")]
                else:
                    converted[key] = weights[key]
            mixer.load_state_dict(converted)
            mixed = mixer(value, ctx)
        else:
            q, k, v = [
                p.unflatten(-1, (1, 64))
                for p in source.qkv(value).chunk(3, dim=-1)
            ]
            q, k = rope(q, ctx.positions), rope(k, ctx.positions)
            scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) / 8
            mask = torch.ones(73, 73, dtype=torch.bool).tril()
            probabilities = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            mixed = source.o(
                torch.einsum("bhts,bshd->bthd", probabilities, v.float())
                .to(dtype)
                .flatten(-2)
            )
        x = x + mixed
        norm.load_state_dict(layer.post_attention_layernorm.state_dict())
        ffn = basic.MLP(
            dict(
                type="mlp",
                intermediate_size=256,
                bias=True,
                activation="gelu",
                limit=None,
                up_softcap=None,
            ),
            64,
        )
        ffn = ffn.to(dtype=dtype)
        ffn.up.load_state_dict(layer.up.state_dict())
        ffn.down.load_state_dict(layer.down.state_dict())
        x = x + ffn(norm(x))
    norm.load_state_dict(model.norm.state_dict())
    actual = F.linear(norm(x), model.embed_tokens.weight).squeeze(0)
    tolerance = 2e-5 if dtype == torch.float32 else 0.04
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
