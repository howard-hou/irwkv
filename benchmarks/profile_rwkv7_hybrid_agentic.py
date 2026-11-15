# SPDX-License-Identifier: Apache-2.0
"""Run the original agentic workload and profile one complete-model decode segment."""

import json
import os
import time

from benchmarks import benchmark_rwkv7_hybrid_agentic as base
from irwkv import SamplingParams
from irwkv.inputs import TokensPrompt
from irwkv.sampling_params import RequestOutputKind


async def _session(
    engine,
    a,
    head,
    rng,
    vocab,
    name,
    totals,
    turns=None,
):
    """Preserve the benchmark request stream, with one optional profiled turn."""
    prompt = head + base._tokens(rng, vocab, a.context)
    params = SamplingParams(
        max_tokens=a.output,
        ignore_eos=True,
        temperature=0.0,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    target = os.environ.get("IRWKV_PROFILE_REQUEST", "c1-s0")
    target_turn = int(os.environ.get("IRWKV_PROFILE_TURN", "1"))
    for turn in range(turns or a.turns):
        start, first, out, cached = time.perf_counter(), None, [], 0
        profiled = name == target and turn == target_turn
        if profiled:
            await engine.start_profile(
                "irwkv-3328-full-decode",
                delay_iterations=int(os.environ.get("IRWKV_PROFILE_DELAY", "8")),
                max_iterations=int(os.environ.get("IRWKV_PROFILE_STEPS", "32")),
            )
        try:
            request = TokensPrompt(prompt_token_ids=prompt)
            async for output in engine.generate(
                request, params, request_id=f"{name}-{turn}"
            ):
                first = first or time.perf_counter()
                out += output.outputs[0].token_ids
                cached = output.num_cached_tokens or cached
        finally:
            if profiled:
                await engine.stop_profile()
        now = time.perf_counter()
        ttft = (first or now) - start
        totals.turns.append(
            base.Turn(turn, len(prompt), cached, ttft, now - start, len(out))
        )
        if profiled:
            print(
                json.dumps(
                    {
                        "profiled_turn": name,
                        "turn": turn,
                        "prompt_tokens": len(prompt),
                        "cached_tokens": cached,
                        "ttft_seconds": ttft,
                        "seconds": now - start,
                        "output_tokens": len(out),
                        "tpot_ms": (now - start - ttft) / max(len(out) - 1, 1) * 1e3,
                    }
                ),
                flush=True,
            )
        prompt = prompt + out + base._tokens(rng, vocab, a.tool)


base._session = _session


if __name__ == "__main__":
    base.main()
