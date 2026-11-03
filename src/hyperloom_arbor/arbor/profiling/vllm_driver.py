#!/usr/bin/env python3
"""Phase-isolated per-op profile of a vLLM engine, in-process, no server.

Replaces the pack-local ``profile_decode_vllm.py``. Same idea — drive vLLM's
offline ``LLM`` class and read the workers' torch traces — with the four things
that script could not do:

1. **It starts.** ``moe_backend="aiter"`` is not an engine argument; it falls
   through to ``EngineArgs(**kwargs)`` and raises ``TypeError`` before any GPU
   work. The aiter path is selected by environment variable, and DSv4.1-Flash
   additionally needs ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` or engine init aborts.
   See :mod:`arbor.profiling.capability`.
2. **It adapts to the build.** Profiler options are filtered against the
   installed ``ProfilerConfig`` rather than assumed, and the ones that were
   dropped are printed.
3. **It isolates decode with the engine when it can**, via
   ``delay_iterations``, and falls back to two-pass subtraction when the build
   has no such field — rather than assuming one strategy works everywhere.
4. **It verifies the isolation held.** The estimate of how many iterations
   prefill takes is an estimate; the trace says whether it was right.

Run inside the container with the GPUs free:

    python3 -m arbor.profiling.vllm_driver --json /workspace/sessions/decode.json
    python3 -m arbor.profiling.vllm_driver --isl 131072 --conc 32 --probe-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

from arbor.profiling.capability import (
    build_profiler_config,
    check_engine_kwargs,
    describe,
    rocm_profiling_env,
)
from arbor.profiling.record import ProfileRecord
from arbor.profiling.trace import KernelTotals, aggregate_kernels, load_events, phase_report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", "/model_weights"))
    p.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    p.add_argument("--conc", type=int, default=int(os.environ.get("CONC", "32")),
                   help="batch size; decode rows = conc x (1 + num_spec_tokens)")
    p.add_argument("--isl", type=int, default=int(os.environ.get("ISL", "32768")),
                   help="live context per request. The campaign's traces reach ~996k; "
                        "a share measured at 8k understates every context-scaling component")
    p.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    p.add_argument("--steps", type=int, default=16,
                   help="engine iterations to profile once started")
    p.add_argument("--warmup-iterations", type=int, default=2,
                   help="iterations profiled then discarded, on builds that support it; "
                        "drops one-time JIT and allocator cost from the first step")
    p.add_argument("--delay-iterations", type=int, default=-1,
                   help="iterations to skip before profiling; -1 estimates from isl/chunk")
    p.add_argument("--chunked-prefill-size", type=int, default=16384)
    p.add_argument("--max-model-len", type=int,
                   default=int(os.environ.get("SIKL_MAX_MODEL_LEN", "1048576")))
    p.add_argument("--num-spec-tokens", type=int,
                   default=int(os.environ.get("SIKL_NUM_SPEC_TOKENS", "5")),
                   help="0 disables speculative decoding")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--cudagraph-capture-size", type=int, default=1024)
    p.add_argument("--tokenizer-mode", default="deepseek_v41")
    p.add_argument("--trace-dir", default=os.environ.get(
        "ARBOR_TRACE_DIR", "/workspace/sessions/torch_profile"))
    p.add_argument("--json", default=os.environ.get("SIKL_OUTPUT",
                                                    "/tmp/sikl_decode_result.json"))
    p.add_argument("--top", type=int, default=25, help="ops to print")
    p.add_argument("--probe-only", action="store_true",
                   help="report engine profiling capabilities and exit; no GPU needed")
    return p.parse_args(argv)


def estimate_prefill_iterations(conc: int, isl: int, chunk: int) -> int:
    """Rough number of engine iterations spent on prefill for ``conc`` prompts.

    With chunked prefill the scheduler admits ``chunk`` prompt tokens per
    iteration across the whole batch, so the total is the token count over the
    chunk size. Deliberately rounded **up** and padded: overshooting costs a few
    decode steps of delay, undershooting silently contaminates the profile with
    prefill, which is the error this whole module exists to catch.
    """
    if chunk <= 0:
        return 0
    return int(math.ceil(conc * isl / chunk)) + 4


def clear_dir(path: str) -> None:
    if os.path.isdir(path):
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) else os.remove(full)
    os.makedirs(path, exist_ok=True)


def wait_for_flush(trace_dir: str, timeout: float = 300.0) -> None:
    """Workers flush traces asynchronously on stop; wait for the bytes to settle."""
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        time.sleep(5)
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(trace_dir)
            for f in fs
        )
        if total > 0 and total == last:
            return
        last = total
    print(f"[profiling] trace size still changing after {timeout:.0f}s — reading anyway")


def capture(llm, prompts, max_tokens: int, trace_dir: str, label: str) -> tuple[KernelTotals, list[dict]]:
    """Profile one generate() through vLLM's own profiler RPC.

    ``torch.profiler`` in this process captures nothing: with TP>1 the model
    runs in worker subprocesses and the driver sees no kernels.
    ``start_profile``/``stop_profile`` broadcast to every worker — the same
    mechanism the HTTP route drives, minus the server.

    Workers write to the directory given to ``profiler_config`` at engine
    construction; it cannot be changed per pass. Passes are isolated by
    emptying that one directory, not by pointing at subdirectories.
    """
    from vllm import SamplingParams

    clear_dir(trace_dir)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    llm.start_profile()
    llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    llm.stop_profile()
    wait_for_flush(trace_dir)

    events, n_files = load_events(trace_dir)
    totals = aggregate_kernels(events, n_files=n_files)
    print(f"[profiling] {label}: {n_files} trace file(s), "
          f"{len(totals.by_name)} distinct kernels, {totals.total_us / 1000:.1f} ms GPU")
    return totals, events


def subtract(b: KernelTotals, a: KernelTotals, n_steps: int) -> KernelTotals:
    """Per-step decode cost as ``(passB - passA) / (N - 1)``, op by op.

    Fallback for builds without ``delay_iterations``. Ops whose delta is <= 0
    are prefill-only or noise and are dropped rather than clamped, so they
    cannot inflate the denominator.
    """
    out: dict[str, tuple[float, int]] = {}
    for name in set(a.by_name) | set(b.by_name):
        us_b, cnt_b = b.by_name.get(name, (0.0, 0))
        us_a, cnt_a = a.by_name.get(name, (0.0, 0))
        delta = (us_b - us_a) / n_steps
        if delta > 0:
            out[name] = (delta, max(cnt_b - cnt_a, 0) / n_steps)
    return KernelTotals(by_name=out, n_files=b.n_files, n_events=b.n_events)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Environment must be set BEFORE vllm is imported — envs.py snapshots some
    # of these at import time.
    os.environ.update(rocm_profiling_env())

    caps = describe()
    print("[profiling] engine capabilities:")
    print(json.dumps(caps, indent=2))
    if caps.get("schedule_fields_absent_here"):
        print("[profiling] NOTE: absent on THIS build — "
              + ", ".join(caps["schedule_fields_absent_here"])
              + ". That is a fact about this build only; other vLLM builds do "
                "have them. Re-run --probe-only in the campaign container "
                "before writing anything down.")
    if args.probe_only:
        return 0

    delay = args.delay_iterations
    if delay < 0:
        delay = (
            estimate_prefill_iterations(args.conc, args.isl, args.chunked_prefill_size)
            if args.phase == "decode" else 0
        )

    profiler_cfg, cfg_report = build_profiler_config(
        args.trace_dir,
        delay_iterations=delay,
        warmup_iterations=args.warmup_iterations,
        active_iterations=args.steps,
        max_iterations=args.steps,
    )
    print("[profiling] profiler_config: " + json.dumps(cfg_report, indent=2))

    engine_isolates = cfg_report.get("phase_isolation_by_engine", False)
    if engine_isolates:
        shape = cfg_report["schedule"]["shape"]
        print(f"[profiling] engine-side phase isolation: skipping {delay} iterations "
              f"(estimated prefill length), profiling {args.steps} "
              f"(schedule shape: {shape})")
        if shape == "delay_and_cap" and args.warmup_iterations:
            print("[profiling] this build has no warmup_iterations — the first "
                  "profiled step still carries one-time JIT/allocator cost. "
                  "Profiles taken with and without a warmup are not directly "
                  "comparable; record which you got.")
    else:
        print("[profiling] build has no delay_iterations — falling back to two-pass "
              "subtraction. The profile will be noisier and the phase label unverified "
              "by construction.")

    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
    except Exception as exc:
        json.dump({"status": "unavailable", "reason": f"import failed: {exc}", "ops": []},
                  open(args.json, "w"), indent=2)
        print(f"[profiling] unavailable: {exc}", file=sys.stderr)
        return 1

    spec = None
    if args.num_spec_tokens > 0:
        spec = {
            "method": "dspark",
            "num_speculative_tokens": args.num_spec_tokens,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": 3.51,
            # Must stay False on ROCm: DeepseekV4IndexerBackend does not
            # support on-device verification trimming and the engine refuses
            # to start with it enabled.
            "enable_adaptive_verification": False,
        }

    rows = args.conc * (1 + args.num_spec_tokens)
    engine_meta = {
        "model": args.model,
        "tp": args.tp,
        "conc": args.conc,
        "isl": args.isl,
        "max_model_len": args.max_model_len,
        "num_spec_tokens": args.num_spec_tokens,
        "decode_rows": rows,
        "vllm_version": caps.get("vllm_version"),
        "profiler_config": cfg_report,
    }
    print(f"[profiling] decode rows = conc x (1 + next_n) = {rows}")

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.conc, 32),
        "gpu_memory_utilization": args.gpu_mem_util,
        "tokenizer_mode": args.tokenizer_mode,
        # Required for DSv4.1-Flash and missing from the original C8 config:
        # the recipe derives it as the first power of two >= max_num_seqs x
        # (1 + num_speculative_tokens). Top-level EngineArgs field that
        # forwards into CompilationConfig.
        "max_cudagraph_capture_size": args.cudagraph_capture_size,
        "speculative_config": spec,
        "profiler_config": profiler_cfg or None,
    }
    bad = check_engine_kwargs(llm_kwargs)
    if bad:
        print(f"[profiling] ABORT: this vLLM build does not accept these engine "
              f"arguments: {', '.join(bad)}. Passing them raises TypeError from "
              f"EngineArgs(**kwargs), which presents as 'engine init failed'.",
              file=sys.stderr)
        return 1

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[profiling] engine ready in {time.time() - t0:.0f}s")

    prompts = [
        TokensPrompt(prompt_token_ids=[(i * 7 + j) % 60000 + 100 for j in range(args.isl)])
        for i in range(args.conc)
    ]
    llm.generate(prompts,
                 sampling_params=SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
                 use_tqdm=False)
    print("[profiling] warmup done")

    if engine_isolates:
        # max_tokens must be large enough to guarantee `steps` DECODE iterations
        # after the delay. An iteration emits at most (1 + next_n) tokens when
        # every draft is accepted, so that is the worst case to budget for.
        # Sizing this from `delay` instead would under-run: delay counts prefill
        # iterations, which emit no tokens at all.
        applied = cfg_report["schedule"]["applied"]
        window = args.steps + applied.get("warmup_iterations", 0)
        budget = window * (1 + args.num_spec_tokens) + 8
        totals, events = capture(llm, prompts, budget, args.trace_dir, "single pass")
        totals = totals.scaled(args.steps)
        verification = phase_report(events, expected=args.phase,
                                    spec_decode=args.num_spec_tokens > 0)
        method = f"engine-isolated: delay_iterations={delay}, max_iterations={args.steps}"
        steps_profiled = args.steps
    else:
        a, _ = capture(llm, prompts, 1, args.trace_dir, "pass A (prefill + 1 step)")
        b, events = capture(llm, prompts, args.steps, args.trace_dir,
                            f"pass B (prefill + {args.steps} steps)")
        n = args.steps - 1
        if n < 1:
            print("[profiling] --steps must be >= 2 for the subtraction", file=sys.stderr)
            return 1
        totals = subtract(b, a, n)
        verification = {
            "isolation": "unverified",
            "reason": "two-pass subtraction leaves no per-step boundary to check",
            "expected": args.phase,
        }
        method = f"subtraction: (passB - passA) / {n}"
        steps_profiled = n

    record = ProfileRecord.build(
        totals,
        phase=args.phase,
        phase_verification=verification,
        steps_profiled=steps_profiled,
        method=method,
        profiler="vllm-offline-torch-profiler",
        engine=engine_meta,
    )
    data = record.write(args.json, top=args.top)

    attr = data["attribution"]
    print(f"\n[profiling] {args.phase} step total: "
          f"{data['total_gpu_us_per_step']:.1f} us GPU across "
          f"{data['distinct_kernels']} kernels")
    print(f"[profiling] phase isolation: {verification['isolation']}"
          + (f" — {verification['reason']}" if verification.get("reason") else ""))
    print(f"[profiling] attribution coverage: {100 * attr['coverage']:.1f}% "
          f"({attr['unattributed_pct']:.1f}% unattributed)")
    print("[profiling] components:")
    for comp in attr["components"]:
        print(f"  {comp['name']:20s} {comp['us']:10.1f} us  {comp['pct']:5.1f}%  "
              f"({comp['kernels']} kernels)")
    if attr["rest_top"]:
        print("[profiling] largest unattributed kernels (extend the component patterns):")
        for k in attr["rest_top"]:
            print(f"  {k['name'][:70]:70s} {k['us']:10.1f} us")
    print(f"[profiling] written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
