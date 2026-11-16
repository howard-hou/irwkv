# bux RWKV-7 / RoPE Hybrid

Experimental adapter for [pilot7747/bux](https://github.com/pilot7747/bux/tree/03d5031fcc842ae4a4992921356af31b1bceaee0).
The FlashRWKV2 and optional Albatross recurrent paths, paged attention,
full-decode CUDA Graphs and engine generation have been exercised on Colab A100.
Inductor compilation remains disabled.

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
The adapter gathers contiguous state for the selected recurrent backend and
writes it back after each batch. Prefix-boundary transitions copy recurrent
state; generic block copies and resets cover both cache types.

Supported scope: BF16 weights/KV, FP32 recurrent state, TP=PP=1, chunked prefill,
recurrent cache modes none and align. CUDA attention uses vLLM 0.32.0's paged FA2
with lower-right causal bias and reads scheduler-owned pages directly, without
materializing each request's complete KV history. The attention block size must be
divisible by 16. CPU tensor tests retain a PyTorch SDPA reference path.
Quantization, FP8 KV, distributed execution, cache transfer and offload are
rejected. Recurrent-state gathering and eager prefill dispatch still introduce
overhead.

`IRWKV_RWKV_HYBRID_RECURRENT_BACKEND=flashrwkv2` is the default. Setting it to
`albatross` replaces only uniform one-token-per-request decode with the CUDA
kernel derived from BlinkDL/Albatross; variable-length prefill continues to use
FlashRWKV2 without padding. The derived kernel accepts BF16 inputs and updates
FP32 state pages in place through `state_indices` on the current CUDA stream.
The decode-only kk-pre and post kernels additionally fuse the small normalization,
scaling, residual and gating operations while retaining eager BF16 intermediate
rounding. Albatross pages retain its native `[K,V]` state layout, while
FlashRWKV2 pages use `[V,K]`, so decode needs no layout conversion. Select the
backend before engine startup and build the `rwkv7_albatross_ops` extension.

CUDA defaults to `CompilationMode.NONE` with full-decode CUDA Graphs. The engine
reports this as `FULL_DECODE_ONLY`; prefill remains eager. Paged FA2 uses live GPU
sequence lengths and block tables during replay. FlashRWKV2 metadata tickets and
recurrent slot tensors are prepared outside capture, retained for graph reuse and
shared by all recurrent layers in a batch. Albatross decode does not prepare
FlashRWKV2 metadata tickets. `enforce_eager=True` remains available for debugging.
No Inductor or `torch.compile` result is claimed.

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

The explicit GPU test fails if the required CUDA backend is missing. It compares
a complete eight-layer BF16 model using FlashRWKV2 and Albatross, with one and
two requests, against the reference through prefill and decode. Default tests
cover complete-model chunking/prefix forks, page isolation/reset/copy/padding,
configuration and parameter counts. They do not exhaustively certify scheduler
preemption or trained-model quality. No single-layer microbenchmark is used.

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

### Albatross recurrent decode A/B (2026-11-15)

The optional decode kernel was derived from BlinkDL/Albatross at
`545dd1870ea6cf424d7f7d4da16b7f87491405db`. Thirteen tests passed and the two
optional original-bux-source comparisons were skipped. The full-model cases
covered both recurrent backends and batch sizes one and two. Five consecutive
CUDA Graph replays also matched the FP32 reference; the largest state error was
1.23e-7. A small asynchronous engine completed full-decode graph capture and
generation with concurrency two.

The original 3328 Agent workload then ran sequentially in one A100 session with
identical BF16 precision, paged FA2, full-decode graphs and workload defaults:

| Metric | FlashRWKV2 repeat | Albatross decode |
| --- | ---: | ---: |
| Timed duration | 139.11 s | 138.91 s |
| Output tokens/s | 29.4 | 29.5 |
| Median TPOT | 31.61 ms | 31.56 ms |
| p90 TPOT | 32.30 ms | 32.28 ms |
| Median first-turn TTFT | 3.229 s | 3.227 s |
| Prefix cache hit | 72.09% | 72.09% |

The 0.05 ms median TPOT difference is negligible in these single runs. The
Albatross backend therefore remains optional; FlashRWKV2 is still the default.
This A/B does not claim that the remaining gap to the author's 22.20 ms can be
closed by replacing the recurrent kernel alone.

A 32-step full-decode graph trace attributes only 7.219 ms in total to the
Albatross kernel, or 0.226 ms/token. The three largest GEMM kernel groups account
for 10.78 ms/token and paged FA2 for 3.04 ms/token. GPU activity occupies 24.17
ms/token in this instrumented trace. The profiler-inflated 44.34 ms mean iteration
is not a throughput measurement. Further decode work must address the GEMMs,
attention and copy/synchronization overhead rather than recurrent substitution
alone.

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

### Fused Albatross decode A/B (2026-11-15)

Following the inference formulas in BlinkDL/RWKV-LM `train_temp`, the Albatross
decode path now fuses a-sigmoid, k scaling, kk normalization, `-kk` and `kk*a`
into one kk-pre kernel, and output GroupNorm, the `r*k*r_k` residual and g gate
into one post kernel. Decay sigmoid/exp remain inside the recurrent kernel. The
recurrent kernel also updates indexed FP32 state pages directly, eliminating the
per-layer `index_select(...).contiguous()` gather and `state_pool[indices] = state`
writeback. These changes do not alter eager prefill, CPU, FlashRWKV2 or other
non-Albatross paths.

The hybrid test file passed 17 tests with two optional bux-source tests skipped.
Coverage includes fused/eager A/B, complete models on both recurrent backends,
batch sizes one and two, non-contiguous state indices, chunked prefill/decode,
paged attention and CUDA Graph replay.

Uninstrumented 3328 BF16 Agent workloads ran sequentially in the same A100
session at concurrency one, with full-decode CUDA Graphs and otherwise unchanged
defaults. These are single-run A/B results:

| Metric | Pre-fusion | Post-fusion | Change |
| --- | ---: | ---: | ---: |
| Timed duration | 116.64 s | 104.92 s | -10.0% |
| Output tokens/s | 35.1 | 39.0 | +11.1% |
| TPOT p50 | 26.15 ms | 23.29 ms | -2.86 ms (-10.9%) |
| TPOT p90 | 26.81 ms | 23.95 ms | -2.86 ms (-10.7%) |
| First-turn TTFT p50 | 3.207 s | 3.205 s | -0.002 s |

Matching 32-step profiles used the first request's second turn. Excluding the
first incomplete window leaves 30 start-to-start decode windows:

| Trace category | Pre-fusion ms/token | Post-fusion ms/token | Delta |
| --- | ---: | ---: | ---: |
| GEMM | 13.604 | 13.497 | -0.107 |
| Flash attention | 3.397 | 3.398 | +0.001 |
| Recurrent main kernel | 0.252 | 0.257 | +0.005 |
| Norm/reduction | 1.439 | 0.832 | -0.607 |
| Elementwise, including new fused kernels | 4.356 | 2.975 | -1.380 |
| Index/gather/scatter | 1.404 | 1.009 | -0.395 |
| Copy/cat/roll | 2.027 | 1.466 | -0.561 |
| Other GPU total | 9.281 | 6.338 | -2.943 |
| All GPU activity sum | 26.535 | 23.491 | -3.044 |
| Start-to-start cadence | 27.259 | 24.166 | -3.093 |
| GPU busy union | 26.497 | 23.452 | -3.045 |
| Exposed non-GPU gap | 0.761 | 0.713 | -0.048 |

The FP32 state gather and 4D state writeback each removed about 38.93 calls per
token. The new kk-pre and post kernels each ran about 38.9 times per token and
took 0.1115 and 0.0942 ms/token respectively. Total GPU event calls fell from
4688.97 to 3402.83 per token. GEMM, flash attention and the recurrent main kernel
were effectively unchanged; the improvement came from fused small operators and
state-management removal. Category values are event-duration sums, while GPU busy
uses their interval union. The exposed gap is wall time minus that union, not
pure CPU work. Profiler cadence explains the direction and attribution of the
uninstrumented gain but is not itself a throughput baseline.

### Second-round fusion and LayerNorm launch A/B (2026-11-16)

The second decode-fusion round combined the six token shifts and previous-token
state writeback into one kernel, and combined RoPE with paged KV-cache writes.
The latter supports non-contiguous QKV inputs and the existing negative-slot
semantics. A custom LayerNorm kernel preserved the eager path's BF16 rounding
after normalization and affine operations. Its initial 128-thread configuration
was inefficient for 3328 channels at batch one: each thread serially processed
about 26 channels across three passes and two block reductions. The third round
changed only this LayerNorm launch and reduction to 512 threads and 16 warps;
it did not switch to native `F.layer_norm` or relax numerical tolerances.

Both rounds passed 21 hybrid tests with two optional bux-source comparisons
skipped, plus the full-decode-graph engine smoke. The second-round same-session
A/B reduced median TPOT from 23.34 to 20.77 ms and timed duration from 105.16 to
93.88 seconds. The third-round A/B then used the validated second-round source
as its baseline, with the same A100, 3328 BF16 model, concurrency one, two
sessions, four turns and 512 output tokens per turn:

| Metric | Second round | Third round | Change |
| --- | ---: | ---: | ---: |
| Timed duration | 93.72 s | 87.41 s | -6.31 s (-6.73%) |
| Output tokens/s | 43.7 | 46.9 | +7.32% |
| TPOT p50 | 20.74 ms | 19.19 ms | -1.55 ms (-7.47%) |
| TPOT p90 | 21.40 ms | 19.85 ms | -1.55 ms (-7.24%) |
| First-turn TTFT p50 | 2.965 s | 2.965 s | unchanged |

Matching 32-step profiles again used the first request's second turn, excluding
the first incomplete window and averaging 30 start-to-start decode windows:

| Trace category | Second round ms/token | Third round ms/token | Delta |
| --- | ---: | ---: | ---: |
| GEMM | 13.474 | 13.474 | approximately 0 |
| Flash attention | 3.401 | 3.400 | -0.001 |
| Recurrent main kernel | 0.258 | 0.259 | +0.001 |
| Norm/reduction | 2.390 | 0.864 | -1.527 |
| Elementwise | 0.883 | 0.883 | approximately 0 |
| Index/gather/scatter | 0.085 | 0.085 | approximately 0 |
| Copy/cat/roll | 0.020 | 0.020 | approximately 0 |
| Other GPU total | 3.435 | 1.909 | -1.527 |
| GPU busy union | 20.528 | 19.000 | -1.528 |
| Exposed non-GPU gap | 0.577 | 0.614 | +0.037 |
| Start-to-start cadence | 21.105 | 19.614 | -1.491 |

The fused LayerNorm ran 104.633 times per token in both traces. Its total time
fell from 2.384 to 0.858 ms/token, or from 22.79 to 8.20 microseconds per call:
a 64.0% reduction and 2.78x speedup. The cadence improvement closely matches the
LayerNorm reduction, while GEMM, flash attention, recurrent and the other small
operator categories remained stable. These are single-run A/B measurements;
profiler cadence supports attribution but is not an uninstrumented throughput
baseline.

