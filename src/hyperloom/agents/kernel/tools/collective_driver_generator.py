###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Generate a torchrun driver + task brief for a multi-GPU collective kernel.

``harness_generator`` cannot serve this lane: every template it emits is
single-device (``device="cuda"``, no process group), and a collective measured
on one rank produces no inter-GPU traffic at all.

What is generated:

* ``driver.py`` -- the complete measurement rig: self-launch under torchrun,
  process-group setup, rank-distinct inputs, an SNR parity gate against
  ``torch.distributed``, a warmup+median benchmark, and a single JSON result
  line. Everything except one function is derived from the candidate.
* ``program.md`` -- the task brief forge's author agent reads.

The one function left blank is ``run_candidate``. The trace tells us *where* the
launcher lives (file, line, function) and *what shapes* flow through it, but not
the callee's parameter names or order. Guessing that yields a driver that fails
inside the forge loop, where the failure is expensive to diagnose; the brief
points the author at the exact call site instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template
from typing import Any

# Reference implementations per collective op, expressed against
# ``torch.distributed``. The candidate is always compared to one of these.
_REFERENCE_CALLS: dict[str, str] = {
    "all_reduce": "dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)",
    "all_gather": "dist.all_gather_into_tensor(gathered, out, group=group); out = gathered",
    "reduce_scatter": "dist.reduce_scatter_tensor(scattered, out, group=group); out = scattered",
    "all_to_all": "dist.all_to_all_single(shuffled, out, group=group); out = shuffled",
    "broadcast": "dist.broadcast(out, src=0, group=group)",
    "reduce": "dist.reduce(out, dst=0, op=dist.ReduceOp.SUM, group=group)",
}

# Output-tensor allocation each reference needs before the collective call.
_REFERENCE_PREALLOC: dict[str, str] = {
    "all_gather": "gathered = out.new_empty((out.shape[0] * ctx.world_size,) + tuple(out.shape[1:]))",
    "reduce_scatter": "scattered = out.new_empty((out.shape[0] // ctx.world_size,) + tuple(out.shape[1:]))",
    "all_to_all": "shuffled = torch.empty_like(out)",
}

_DEFAULT_OP = "all_reduce"

# Parity gate. 30 dB is the same floor the fusion lane applies: bf16 with fp32
# accumulation is never bit-exact, so allclose would reject correct kernels.
_SNR_FLOOR_DB = 30.0

_SHAPE_RE = re.compile(r"(\d+)")


def _parse_shapes(candidate: dict[str, Any], *, limit: int = 6) -> list[tuple[int, ...]]:
    """Extract operand shapes from a TraceLens candidate.

    Accepts the several shapes TraceLens uses (``input_shapes`` records, plain
    ``shapes`` strings) and returns deduplicated integer tuples.
    """
    raw: list[str] = []
    for record in candidate.get("input_shapes") or []:
        if isinstance(record, dict):
            raw.append(str(record.get("shape") or ""))
        else:
            raw.append(str(record))
    for item in candidate.get("shapes") or []:
        raw.append(str(item))

    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for text in raw:
        for group in re.findall(r"[\[(]([0-9,\s]+)[\])]", text):
            dims = tuple(int(d) for d in _SHAPE_RE.findall(group))
            if len(dims) < 2 or any(d <= 0 for d in dims):
                continue
            dims = dims[:2]
            if dims in seen:
                continue
            seen.add(dims)
            out.append(dims)
            if len(out) >= limit:
                return out
    return out


def _fallback_shapes(hidden: int = 7168) -> list[tuple[int, ...]]:
    """Token-count sweep used when the trace carried no usable shape.

    Spans the decode-to-prefill range so a candidate cannot win on one regime
    while regressing the other.
    """
    return [(rows, hidden) for rows in (1, 8, 64, 512, 4096)]


def _dtype_of(candidate: dict[str, Any]) -> str:
    """Element dtype for the driver's tensors ("bf16" unless stated otherwise)."""
    for value in candidate.get("input_dtypes") or []:
        text = str(value).lower()
        for name in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"):
            if name in text:
                return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(name, name)
    return "bf16"


def _collective_op(candidate: dict[str, Any]) -> str:
    """Collective op from the enriched kernel contract."""
    contract = candidate.get("kernel_contract") or {}
    op = str(contract.get("collective_op") or "").strip().lower()
    return op if op in _REFERENCE_CALLS else _DEFAULT_OP


def _world_size(candidate: dict[str, Any], default: int) -> int:
    """World size for the driver, preferring the caller's TP degree.

    ``kernel_contract.world_size`` is not always a real parallelism degree:
    TraceLens stamps ``num_gpus_recommended = 2`` on every multi-GPU kernel as a
    "needs at least two ranks" marker, and the contract falls back to that value
    whenever model_params carries no TP. Benchmarking an 8-rank all-reduce on 2
    ranks measures a different regime, so the caller's TP -- which comes from the
    session's actual launch configuration -- wins whenever it is usable.
    """
    try:
        tp = int(default or 0)
    except (TypeError, ValueError):
        tp = 0
    if tp > 1:
        return tp
    contract = candidate.get("kernel_contract") or {}
    for key in ("world_size", "tp_size"):
        try:
            value = int(contract.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 1:
            return value
    return 2


_DRIVER_TEMPLATE = Template('''#!/usr/bin/env python3
"""Forge driver for $kernel_name ($collective_op, world_size=$world_size).

Generated by Hyperloom's collective lane. Self-launches under torchrun, so it
can be invoked as a plain script.

Scoring: median latency per case, compared against the pre-edit baseline. A
candidate must also clear the parity gate (SNR >= $snr_floor dB) against
torch.distributed.$collective_op.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist

WORLD_SIZE = $world_size
SNR_FLOOR_DB = $snr_floor
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
CASES = $cases
DTYPE = "$dtype"


# --------------------------------------------------------------------------
# Self-launch: one process per rank, otherwise the collective is a no-op.
# --------------------------------------------------------------------------


def self_launch(argv: list[str]) -> int:
    """Re-exec this script under torchrun when not already in a rank."""
    if os.environ.get("RANK") is not None:
        return -1
    visible = torch.cuda.device_count()
    if visible < WORLD_SIZE:
        # Oversubscribing would bind several ranks to one device: the collective
        # would then measure intra-device copies, and a "speedup" would not
        # transfer to the real multi-GPU path. Fail loudly instead.
        print(
            f"ERROR: need {WORLD_SIZE} visible GPUs for this collective, found {visible}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc-per-node={WORLD_SIZE}",
        os.path.abspath(__file__), *argv,
    ]
    return subprocess.call(cmd)


@dataclass
class WorkerCtx:
    rank: int
    world_size: int
    device: torch.device
    group: Any


def init_worker() -> WorkerCtx:
    """Initialise the process group and pin this rank to its own device."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    visible = torch.cuda.device_count()
    if world_size > visible:
        raise RuntimeError(
            f"world_size={world_size} exceeds {visible} visible GPUs; "
            "ranks would share a device and the collective would not be measured"
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return WorkerCtx(rank, world_size, torch.device("cuda", torch.cuda.current_device()), dist.group.WORLD)


# --------------------------------------------------------------------------
# Inputs and parity
# --------------------------------------------------------------------------


def make_inputs(shape: tuple[int, ...], ctx: WorkerCtx, seed: int = 0) -> dict:
    """Rank-distinct inputs, so a collective that drops ranks cannot pass."""
    gen = torch.Generator(device="cuda").manual_seed(seed + ctx.rank)
    dtype = DTYPES[DTYPE]
    x = torch.randn(shape, generator=gen, device=ctx.device, dtype=torch.float32).to(dtype)
    return {"x": x}


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB; inf when bit-exact."""
    ref32, got32 = ref.float(), got.float()
    noise = (ref32 - got32).pow(2).sum().item()
    if noise == 0.0:
        return float("inf")
    signal = ref32.pow(2).sum().item()
    if signal == 0.0:
        return float("-inf")
    return 10.0 * torch.log10(torch.tensor(signal / noise)).item()


# --------------------------------------------------------------------------
# Candidate vs reference
# --------------------------------------------------------------------------


def run_candidate(inputs: dict, ctx: WorkerCtx):
    """Run the implementation under optimisation.

    FILL THIS IN. The trace attributes this kernel to:

        $launcher_hint

    Import that entry point and call it with ``inputs["x"]``. Keep the call in
    this function only -- everything else in this driver is already wired.

    Returns:
        The collective's output tensor.
    """
    raise NotImplementedError(
        "run_candidate is unimplemented; call $launcher_hint"
    )


def run_reference(inputs: dict, ctx: WorkerCtx):
    """Independent reference via torch.distributed."""
    group = ctx.group
    out = inputs["x"].clone()
    $reference_prealloc
    $reference_call
    return out


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def check_case(
    shape: tuple[int, ...],
    ctx: WorkerCtx,
    snr_threshold: float = SNR_FLOOR_DB,
    seed: int = 0,
) -> dict:
    """Parity of candidate vs reference for one shape."""
    ref = run_reference(make_inputs(shape, ctx, seed), ctx)
    got = run_candidate(make_inputs(shape, ctx, seed), ctx)
    if isinstance(got, (tuple, list)):
        got = got[0]
    value = snr_db(ref, got)
    return {"shape": list(shape), "snr_db": value, "passed": value >= snr_threshold}


def capture_chain(inputs: dict, ctx: WorkerCtx, chain: int):
    """Capture ``chain`` back-to-back candidate calls into one graph.

    A custom all-reduce has to be captured inside the framework's
    ``graph_capture()`` context: it wraps the communicator's own ``capture()``,
    which flushes the IPC buffer registrations on exit, and capturing outside it
    raises from the extension. Falls back to a plain capture when that hook is
    absent, and to ``None`` when the collective cannot be captured at all.

    Capturing a chain rather than a single call is deliberate. Timing one
    collective per barrier carries several percent of run-to-run spread, because
    a rank leaving the barrier late makes the collective wait and that jitter
    lands in the sample; timing an eager burst instead measures Python dispatch
    cost, reporting the same figure for every payload size.
    """
    try:
        from aiter.dist.parallel_state import graph_capture
    except ImportError:
        graph_capture = None
    try:
        graph = torch.cuda.CUDAGraph()
        if graph_capture is not None:
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=getattr(gc, "stream", None)):
                    for _ in range(chain):
                        run_candidate(inputs, ctx)
        else:
            with torch.cuda.graph(graph):
                for _ in range(chain):
                    run_candidate(inputs, ctx)
        return graph
    except Exception:  # noqa: BLE001 - any capture failure falls back to eager
        return None


def bench_case(
    shape: tuple[int, ...],
    ctx: WorkerCtx,
    warmup: int = 20,
    iters: int = 100,
    repeat: int = 1,
) -> float:
    """Median latency in ms for one shape, maximised across ranks.

    The slowest rank bounds the collective, so per-rank medians are reduced with
    MAX rather than averaged. ``repeat`` re-measures in-process and takes the
    median of medians, which matters when the true speedup sits near the noise
    floor. Timing replays a captured graph, keeping per-call CPU dispatch out of
    the sample; the eager path below is only a fallback for collectives that
    refuse capture.
    """
    inputs = make_inputs(shape, ctx)
    for _ in range(warmup):
        run_candidate(inputs, ctx)
    torch.cuda.synchronize()

    chain = max(1, min(int(iters), 100))
    graph = capture_chain(inputs, ctx, chain)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rounds = []

    if graph is not None:
        for _ in range(max(1, warmup // 10)):
            graph.replay()
        torch.cuda.synchronize()
        for _ in range(max(1, repeat)):
            samples = []
            for _ in range(5):
                dist.barrier(group=ctx.group)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / chain)
            rounds.append(statistics.median(samples))
        del graph
    else:
        dist.barrier(group=ctx.group)
        for _ in range(max(1, repeat)):
            samples = []
            for _ in range(iters):
                start.record()
                run_candidate(inputs, ctx)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            rounds.append(statistics.median(samples))

    median = torch.tensor([statistics.median(rounds)], device=ctx.device)
    dist.all_reduce(median, op=dist.ReduceOp.MAX, group=ctx.group)
    return float(median.item())


def profile_case(ctx: WorkerCtx) -> None:
    """A few warm dispatches of one case, for a hardware profiler to replay.

    Deliberately not a timing loop: rocprof-compute replays the process once per
    counter group, so extra iterations only multiply collection time.
    """
    inputs = make_inputs(tuple(CASES[0]), ctx)
    for _ in range(3):
        run_candidate(inputs, ctx)
    torch.cuda.synchronize()


def build_parser() -> argparse.ArgumentParser:
    """CLI contract expected by forge-loop.

    With no flags the driver runs correctness; ``--bench-mode`` and
    ``--profile-run`` select the other two modes.
    """
    p = argparse.ArgumentParser(description="Collective kernel driver ($collective_op)")
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--profile-run", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeat", type=int, default=1, help="in-process repeats of the sweep")
    p.add_argument("--snr-threshold", type=float, default=SNR_FLOOR_DB)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str]) -> int:
    # Permissive: forge-loop may pass flags a future driver revision adds, and an
    # unknown one must not fail the run.
    args, _unknown = build_parser().parse_known_args(argv)

    rc = self_launch(argv)
    if rc >= 0:
        return rc

    ctx = init_worker()
    result = {"kernel": "$kernel_name", "collective_op": "$collective_op", "world_size": ctx.world_size}
    try:
        if args.profile_run:
            profile_case(ctx)
            result["status"] = "ok"
        elif args.bench_mode:
            timings = {
                str(tuple(s)): bench_case(tuple(s), ctx, args.warmup, args.iters, args.repeat)
                for s in CASES
            }
            result["latency_ms"] = timings
            result["latency_ms_total"] = sum(timings.values())
            result["status"] = "ok"
        else:
            checks = [check_case(tuple(s), ctx, args.snr_threshold, args.seed) for s in CASES]
            result["correctness"] = checks
            result["correctness_passed"] = all(c["passed"] for c in checks)
            result["status"] = "ok" if result["correctness_passed"] else "failed"
    except Exception as exc:  # noqa: BLE001 - report, never traceback-only
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dist.is_initialized():
            dist.barrier(group=ctx.group)
            dist.destroy_process_group()

    if ctx.rank == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
''')


_PROGRAM_TEMPLATE = Template('''# Optimize `$kernel_name`

## Target

| | |
|---|---|
| Collective op | `$collective_op` |
| World size | $world_size |
| GPU time share | $gpu_pct% |
| Source | `$source_file` |
| Launch site | `$launcher_hint` |

## What to do

1. **Implement `run_candidate` in `driver.py` first.** It is the only blank in
   the rig. Import the entry point at the launch site above and call it with
   `inputs["x"]`. Verify with `--correctness` before optimising anything.
2. Optimise the kernel source at `$source_file`.
3. Re-measure. A candidate is kept only when it is faster on the median case
   **and** still clears the parity gate.

## Gates

- **Parity**: SNR >= $snr_floor dB against `torch.distributed.$collective_op`.
  bf16 with fp32 accumulation is not bit-exact, so this is an SNR gate, not
  `allclose`.
- **Inputs are rank-distinct**: a collective that silently drops ranks fails
  parity rather than passing trivially.
- **Latency is reduced with MAX across ranks**: the slowest rank bounds a
  collective, so improving one rank at another's expense does not score.
- **Timing replays a captured graph**: `bench_case` captures a chain of calls
  via `capture_chain` and replays it, so the sample carries device latency
  rather than Python dispatch cost. The eager branch is a fallback for
  collectives that refuse capture -- if you land there, fix capture instead of
  settling for it.

## Shapes under test

$cases_md

## Constraints

- Every rank runs the same code path; do not branch on rank in the fast path.
- Do not weaken or bypass the parity gate.
- Do not replace graph replay with eager timing to make a number look better.
- Keep the process-group setup as generated; the driver self-launches under
  torchrun with `--nproc-per-node=$world_size`.
''')


def generate_collective_driver(
    candidate: dict[str, Any],
    out_dir: Path | str,
    *,
    tp: int = 8,
    overwrite_driver: bool = True,
) -> dict[str, str]:
    """Write ``driver.py`` and ``program.md`` for one collective candidate.

    Args:
        candidate: A finalized TraceLens hot-kernel candidate. Uses
            ``kernel_contract`` (collective op, world size), ``input_shapes`` /
            ``shapes``, ``input_dtypes``, ``source_file`` and the trace-resolved
            ``source_file`` / ``source_line`` / ``source_function``.
        out_dir: Directory to write into (created when absent).
        tp: Tensor-parallel degree, used when the contract carries no world size.
        overwrite_driver: When False, an existing ``driver.py`` is left in place.
            Set this to keep a rig whose ``run_candidate`` has already been
            authored; regenerating would restore the placeholder and discard the
            entry point somebody already resolved.

    Returns:
        ``{"driver": <path>, "program": <path>, "collective_op": ..., "world_size": ...}``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    op = _collective_op(candidate)
    world_size = _world_size(candidate, tp)
    shapes = _parse_shapes(candidate) or _fallback_shapes()
    dtype = _dtype_of(candidate)
    kernel_name = str(candidate.get("device_kernel_name") or candidate.get("name") or "kernel")
    source_file = str(candidate.get("source_file") or "")

    line = candidate.get("source_line")
    function = candidate.get("source_function")
    if source_file and line and function:
        launcher_hint = f"{source_file}({line}): {function}"
    elif source_file:
        launcher_hint = source_file
    else:
        launcher_hint = "unresolved -- locate the collective entry point in the framework source"

    driver_src = _DRIVER_TEMPLATE.substitute(
        kernel_name=kernel_name,
        collective_op=op,
        world_size=world_size,
        snr_floor=_SNR_FLOOR_DB,
        cases=json.dumps([list(s) for s in shapes]),
        dtype=dtype,
        launcher_hint=launcher_hint,
        reference_prealloc=_REFERENCE_PREALLOC.get(op, ""),
        reference_call=_REFERENCE_CALLS[op],
    )
    program_src = _PROGRAM_TEMPLATE.substitute(
        kernel_name=kernel_name,
        collective_op=op,
        world_size=world_size,
        gpu_pct=candidate.get("gpu_pct") or 0.0,
        source_file=source_file or "unresolved",
        launcher_hint=launcher_hint,
        snr_floor=_SNR_FLOOR_DB,
        cases_md="\n".join(f"- `{tuple(s)}` ({dtype})" for s in shapes),
    )

    driver_path = out_path / "driver.py"
    program_path = out_path / "program.md"
    if overwrite_driver or not driver_path.is_file():
        driver_path.write_text(driver_src, encoding="utf-8")
    program_path.write_text(program_src, encoding="utf-8")

    return {
        "driver": str(driver_path),
        "program": str(program_path),
        "collective_op": op,
        "world_size": str(world_size),
    }


__all__ = ["generate_collective_driver"]
