# bux RWKV-7 / RoPE Hybrid

Experimental adapter for [pilot7747/bux](https://github.com/pilot7747/bux/tree/03d5031fcc842ae4a4992921356af31b1bceaee0).
The complete FlashRWKV2 path, paged attention, full-decode CUDA Graphs and engine
generation have been exercised on Colab A100. Inductor compilation remains disabled.

## Architecture

Sources: examples/experiments/timeline/mixers.bux, gpt3_2020.bux,
gpt3_rope.bux, and the RWKV7 mixer at the revision above.
Each group is three RWKV-7 layers followed by one full causal attention layer.
Each layer has pre-LayerNorm mixer and biased 4D GELU FFN residuals.
Norm epsilon: 1e-5. Attention: biased Q/K/V/O, D/64 heads, full head-size-64
half-layout RoPE, base 10000. No learned positional embeddings.
Vocabulary: 50304; tied output embedding. RWKV: head size 64,
w/a/g ranks 64/64/128, six token shifts, per-head norm epsilon 64e-5.
There is no cross-layer value residual.

| Width | Layers | RWKV / attention | Total parameters | Active parameters |
| --- | --- | --- | --- | --- |
| 3328 | 52 | 39 / 13 | 7,148,454,144 | 6,981,042,432 |
| 3584 | 56 | 42 / 14 | 8,893,208,576 | 8,712,919,040 |
| 3840 | 60 | 45 / 15 | 10,903,031,040 | 10,709,863,680 |

Counts match the author's [dataset](https://huggingface.co/datasets/pavlichenko/bux-timeline/blob/main/serving/lanes.csv).
Active excludes the shared embedding matrix. Tests count actual parameters on
the meta device, including biases and normalization.

## Cache and execution

Scheduler-owned attention pages store BF16 K/V. Each recurrent layer has a
scheduler-owned page containing BF16 token shift and FP32 recurrent state.
The adapter gathers contiguous state for FlashRWKV2 and writes it back after
each batch. Prefix-boundary transitions copy recurrent state; generic block
copies and resets cover both cache types.

Supported scope: BF16 weights/KV, FP32 recurrent state, TP=PP=1, chunked prefill,
recurrent cache modes none and align. CUDA attention uses vLLM 0.32.0's paged FA2
with lower-right causal bias and reads scheduler-owned pages directly, without
materializing each request's complete KV history. The attention block size must be
divisible by 16. CPU tensor tests retain a PyTorch SDPA reference path.
Quantization, FP8 KV, distributed execution, cache transfer and offload are
rejected. Recurrent-state gathering and eager prefill dispatch still introduce
overhead.

CUDA defaults to `CompilationMode.NONE` with full-decode CUDA Graphs. The engine
reports this as `FULL_DECODE_ONLY`; prefill remains eager. Paged FA2 uses live GPU
sequence lengths and block tables during replay. FlashRWKV2 metadata tickets and
recurrent slot tensors are prepared outside capture, retained for graph reuse and
shared by all recurrent layers in a batch. `enforce_eager=True` remains available
for debugging. No Inductor or `torch.compile` result is claimed.

Native irwkv parameter names are accepted. A bux checkpoint-name conversion is
not supplied. Dummy weights support timing, not meaningful generated text or
model-quality evaluation.

## Verification

Use Linux CUDA and the repository's pinned installation instructions in AGENTS.md.
Run from the repository root in its dedicated environment:

```bash
.venv/bin/python -m pytest --noconftest tests/models/rwkv/test_rwkv7_hybrid.py -q
IRWKV_TEST_FLASH_HYBRID=1 .venv/bin/python -m pytest --noconftest \
  tests/models/rwkv/test_rwkv7_hybrid.py -q
```

The explicit GPU test fails if FlashRWKV2/CUDA is missing. It compares a complete
eight-layer BF16 model using actual FlashRWKV2 and paged attention against the
reference through prefill and decode. Default tests cover complete-model
chunking/prefix forks, page isolation/reset/copy/padding, configuration and
parameter counts. They do not exhaustively certify scheduler preemption or
trained-model quality. No single-layer microbenchmark is used.

For independent source comparison, clone the pinned bux revision and set
`BUX_HF_SOURCE` to its `src/bux/export/hf` directory. The additional full-model
test uses the original RWKV7, Norm, MLP and RoPE modules with an explicit attention
softmax oracle. FP32 passes at atol=rtol=2e-5. BF16 currently has a strict expected
failure at atol=rtol=0.04 (observed maximum error 0.0859375 on the deterministic
73-token test). The HF export and serving paths have different low-rank-bias and
decay rounding; the cause of the full discrepancy has not been isolated. This is
not a claim of exact BF16 numerical parity with bux or of parity with its FP8 path.

Generate a small configuration and run a short engine workload first:

```bash
.venv/bin/python -m tools.export_rwkv7_hybrid exported/hybrid-smoke \
  --width 64 --layers 4 --vocab-size 4096
.venv/bin/python -m benchmarks.benchmark_rwkv7_hybrid_agentic \
  exported/hybrid-smoke --concurrency 2 --sessions 2 \
  --system 64 --context 128 --tool 32 --output 8 --turns 4 \
  --max-model-len 1024 --max-num-batched-tokens 128
```

## Author's Agent workload

benchmarks/benchmark_rwkv7_hybrid_agentic.py preserves the author's
scripts/old_vs_new/agentic.py workload, adapting engine imports and configuration
attribute only. Defaults: shared prompt 2048, private context 14336, four turns,
512 output tokens/turn, 2048 tool tokens, sessions=2*concurrency, max context
32768, batch-token limit 16384, seed 0, dummy BF16 weights, TP=1, prefix caching.
It warms up and resets prefix caching between concurrency runs. JSON reports
sessions/hour, TTFT, TPOT and cached tokens.

```bash
for width in 3328 3584 3840; do
  .venv/bin/python -m tools.export_rwkv7_hybrid exported/hybrid-$width --width $width
  .venv/bin/python -m benchmarks.benchmark_rwkv7_hybrid_agentic \
    exported/hybrid-$width --concurrency 1,2,4,8,16,32,40 \
    --slo 30 --tag irwkv-bf16-hybrid-$width > hybrid-$width.jsonl
done
```

Run the original script and this adapter on the same GPU with matching BF16
weight/KV and recurrent precision, model configuration, and workload. Record
GPU, software/source revisions, cache capacity and options.
Published target-width rows use B300 with FP8 weights and FP8 KV. This adapter
cannot reproduce that precision setting: BF16 results are not directly comparable.

## Validation record (2026-11-14)

Windows, RTX 3090 24 GiB, PyTorch 2.11.0+cu128: eight default tests passed,
including complete small-model CUDA reference execution. The opt-in Flash test
was skipped. FlashRWKV2 0.1.0a19 installed but import fails on Linux-only fcntl.
The Windows engine environment remains incomplete. Its tensor-test environment
differs from pinned dependencies.

Colab A100-SXM4-40GB, Python 3.12.3, PyTorch 2.13.0+cu132, system CUDA Toolkit
13.0, FlashRWKV2 0.1.0a19, and the pinned Transformers/tokenizers forks:

- irwkv built successfully from the source snapshot.
- Eleven Linux tests passed, including actual FlashRWKV2 full-model execution
  with one and two independent requests and engine configuration. No GPU test
  was skipped. Two optional original-bux-source comparisons were skipped on Colab;
  those comparisons were run separately on local CPU as described below.
- Small-model async engine smoke passed with concurrency 2, four sessions,
  four turns, prefix caching and chunked prefill (cache hit 55.56%).
- All three full target widths passed four-turn dummy-weight generation with
  concurrency 1 and one timed session. Smoke prompts were shortened to
  system/context/tool/output = 64/128/32/8; these are correctness runs, not the
  author's full workload or comparable throughput results.
- Independently loading original bux modules on local CPU: the FP32 full-model
  comparison passed; the BF16 discrepancy above remains a strict expected failure.

The 3328 model completed the author's original Agent workload with concurrency 1
and the default two timed sessions plus warmup. All prompt, turn, output, tool,
batch-token, context and precision defaults were unchanged. These are individual
completed runs on the same A100 model, not a concurrency sweep:

| Metric | irwkv eager KV gather | irwkv paged FA2 eager | irwkv FA2 full-decode graph | Author bux/vLLM |
| --- | ---: | ---: | ---: | ---: |
| Timed duration | 540.04 s | 490.21 s | 140.06 s | 104.64 s |
| Sessions/hour | 13.3 | 14.7 | 51.4 | 68.8 |
| Output tokens/s | 7.6 | 8.4 | 29.2 | 39.1 |
| Median TPOT | 129.18 ms | 117.12 ms | 31.84 ms | 22.20 ms |
| p90 TPOT | 133.43 ms | 117.81 ms | 32.52 ms | 22.86 ms |
| Median first-turn TTFT | 3.397 s | 3.390 s | 3.242 s | 3.815 s |
| Prefix cache hit | 72.09% | 72.09% | 72.09% | 70.02% |

The graph run used `compilation_config={"max_cudagraph_capture_size":4}` and tag
`colab-a100-bf16-3328-fa2-graph-author-workload`. It completed full-decode capture
and replay with Inductor disabled. Median TPOT improved 4.06x over the original
irwkv implementation and 3.68x over paged FA2 without graphs. It remains 1.43x
the author's 22.20 ms and is just above the 30 ms SLO. The author run used vLLM
0.32.0 with Inductor plus full/piecewise graphs; irwkv used its pinned fork,
FlashRWKV2 and no Inductor, so this is a matched hardware/precision/workload A/B,
not a claim of identical software or operator stacks. The original-length workload
was not run for widths 3584 and 3840.

A pre-optimization trace of 32 unique decode steps averaged 234.26 ms/token, with
50.97 ms GPU-active and 183.29 ms GPU-idle time. These instrumented measurements
located host/dispatch overhead and are not throughput baseline results.

The engine configuration check exposed and fixed a reference to the removed
ModelConfig.quantization field. SDPA now uses four-dimensional inputs and
lower-right causal bias. Colab setup needs the virtual environment bin directory
on PATH for Ninja, Rust for the pinned tokenizer, and a setuptools-scm version
override when building from a ZIP without Git metadata. The notebook includes
these settings. FlashRWKV2 uses its untuned default configuration in these runs.

Target BF16 weights alone take approximately 13.3 / 16.6 / 20.3 GiB, before caches
and workspace. The 24 GiB card cannot support the author's high-concurrency target
workload. Exhaustive preemption tests, trained checkpoint conversion/evaluation,
concurrency sweeps and Inductor/kernel-level parity work remain outstanding.

