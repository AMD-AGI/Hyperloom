###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Generate the torchrun rig used to optimize a traced collective kernel.

The parity gate lives in the same file the optimizing agent edits; its integrity
depends on forge-loop pinning the driver digest during task preparation, which
``forge_collective`` asserts stays enabled. Nothing on the Hyperloom side
re-checks the gate afterwards, so if that upstream behaviour changes this rig
loses its protection silently.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from string import Template
from typing import Any

#: Per-op ``run_reference`` body. Scatter and gather change the dim-0 extent, so
#: each op allocates its own output instead of cloning the input.
_REFERENCE_BODIES: dict[str, str] = {
    "all_reduce": """    out = inputs["x"].clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
    return out""",
    "reduce_scatter": """    src = inputs["x"].contiguous()
    out = torch.empty(
        (src.shape[0] // ctx.world_size, *src.shape[1:]),
        dtype=src.dtype,
        device=src.device,
    )
    dist.reduce_scatter_tensor(out, src, op=dist.ReduceOp.SUM, group=ctx.group)
    return out""",
    "all_gather": """    src = inputs["x"].contiguous()
    out = torch.empty(
        (src.shape[0] * ctx.world_size, *src.shape[1:]),
        dtype=src.dtype,
        device=src.device,
    )
    dist.all_gather_into_tensor(out, src, group=ctx.group)
    return out""",
}
#: Bus-bandwidth correction numerator per op, matching the nccl-tests
#: convention: an all-reduce crosses the link twice, a scatter or gather once.
_BUSBW_NUMERATORS: dict[str, int] = {
    "all_reduce": 2,
    "reduce_scatter": 1,
    "all_gather": 1,
}
SNR_FLOOR_DB = 30.0

_SHAPE_RE = re.compile(r"(\d+)")


def _parse_shapes(candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    """Return distinct traced first-input tensor shapes."""
    records = candidate.get("input_shapes")
    if not isinstance(records, list) or not records:
        raise ValueError("collective candidate has no traced input shape")
    shapes: list[tuple[int, ...]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"collective input_shapes[{index}] must be an object")
        text = str(record.get("shape") or "").strip()
        match = re.search(r"[\[(]([0-9,\s]+)[\])]", text)
        if match is None:
            raise ValueError(f"invalid collective input shape: {text!r}")
        dims = tuple(int(d) for d in _SHAPE_RE.findall(match.group(1)))
        if not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"invalid collective input shape: {text!r}")
        if dims not in shapes:
            shapes.append(dims)
    return shapes


def _dtype_of(candidate: dict[str, Any]) -> str:
    """Return the single traced input dtype accepted by the driver."""
    values = candidate.get("input_dtypes")
    if not isinstance(values, list) or not values:
        raise ValueError("collective candidate has no traced input dtype")
    resolved: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid collective input dtype: {value!r}")
        text = value.lower()
        if "bfloat16" in text or "bf16" in text:
            resolved.add("bf16")
        elif "float16" in text or "fp16" in text:
            resolved.add("fp16")
        elif "float32" in text or "fp32" in text:
            resolved.add("fp32")
        else:
            raise ValueError(f"unsupported collective input dtype: {value!r}")
    if len(resolved) != 1:
        raise ValueError(f"collective candidate has conflicting input dtypes: {values!r}")
    return resolved.pop()


def _collective_op(candidate: dict[str, Any]) -> str:
    """Validate and return the supported collective operation.

    References reduce with ``SUM``, so an explicit non-sum reduction is refused.
    """
    contract = candidate.get("kernel_contract")
    if not isinstance(contract, dict) or contract.get("kind") != "collective":
        raise ValueError("candidate kernel_contract.kind must be 'collective'")
    op = str(contract.get("collective_op") or "").strip().lower()
    if op not in _REFERENCE_BODIES:
        raise ValueError(f"unsupported collective operation: {op or '<missing>'}")
    reduce_op = str(contract.get("reduce_op") or "sum").strip().lower()
    if op != "all_gather" and reduce_op != "sum":
        raise ValueError(f"unsupported collective reduction: {reduce_op} (references reduce with sum)")
    return op


def _world_size(tp: int) -> int:
    """Return an explicit tensor-parallel world size."""
    if isinstance(tp, bool) or not isinstance(tp, int):
        raise ValueError(f"invalid collective tp: {tp!r}")
    requested = tp
    if requested <= 1:
        raise ValueError("collective tp must be greater than one")
    return requested


_DRIVER_TEMPLATE = Template('''#!/usr/bin/env python3
"""Measure a traced $collective_op implementation under torchrun."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

WORLD_SIZE = $world_size
SNR_FLOOR_DB = $snr_floor
SNR_LIMIT_DB = 300.0
DIST_TIMEOUT_SEC = 120
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
CASES = $cases
DTYPE = "$dtype"
#: Bus-bandwidth correction numerator for this op: an all-reduce crosses the
#: link twice (reduce then broadcast), a scatter or gather once.
BUSBW_NUMERATOR = $busbw_numerator
#: True when the measured payload is the gathered output rather than the input.
GATHERED_OUTPUT = $gathered_output
#: Below this payload the pair of cross-device barriers dominates the kernel;
#: above it the fabric does. 1 MiB is what a barrier pair's worth of time moves
#: at fabric bandwidth.
REGIME_CUT_BYTES = 1048576
_DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def self_launch(argv: list[str]) -> int:
    """Launch the configured rank count unless torchrun already did."""
    distributed_env = [bool(os.environ.get(key)) for key in _DIST_ENV]
    if any(distributed_env) and not all(distributed_env):
        raise RuntimeError("incomplete torchrun environment")
    if all(distributed_env):
        actual_world_size = int(os.environ["WORLD_SIZE"])
        if actual_world_size != WORLD_SIZE:
            raise RuntimeError(
                f"expected world_size={WORLD_SIZE}, got {actual_world_size}"
            )
        return -1
    visible = torch.cuda.device_count()
    if visible < WORLD_SIZE:
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
    env = os.environ.copy()
    for key in _DIST_ENV:
        env.pop(key, None)
    return subprocess.call(cmd, env=env)


@dataclass
class WorkerCtx:
    """Distributed state owned by one driver rank."""

    rank: int
    world_size: int
    device: torch.device
    group: Any


def init_worker() -> WorkerCtx:
    """Initialise the process group and pin this rank to its own device."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected world_size={WORLD_SIZE}, got {world_size}")
    visible = torch.cuda.device_count()
    if world_size > visible:
        raise RuntimeError(
            f"world_size={world_size} exceeds {visible} visible GPUs; "
            "ranks would share a device and the collective would not be measured"
        )
    if local_rank < 0 or local_rank >= visible:
        raise RuntimeError(f"LOCAL_RANK={local_rank} is outside {visible} visible GPUs")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=DIST_TIMEOUT_SEC),
        )
    return WorkerCtx(rank, world_size, torch.device("cuda", torch.cuda.current_device()), dist.group.WORLD)


def make_inputs(shape: tuple[int, ...], ctx: WorkerCtx, seed: int = 0) -> dict:
    """Rank-distinct inputs, so a collective that drops ranks cannot pass."""
    gen = torch.Generator(device="cuda").manual_seed(seed + ctx.rank)
    dtype = DTYPES[DTYPE]
    x = torch.randn(shape, generator=gen, device=ctx.device, dtype=torch.float32).to(dtype)
    return {"x": x}


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Return a finite signal-to-noise ratio in dB."""
    ref32, got32 = ref.float(), got.float()
    noise = (ref32 - got32).pow(2).sum().item()
    if not math.isfinite(noise) or noise < 0.0:
        return -SNR_LIMIT_DB
    if noise == 0.0:
        return SNR_LIMIT_DB
    signal = ref32.pow(2).sum().item()
    if not math.isfinite(signal) or signal <= 0.0:
        return -SNR_LIMIT_DB
    value = 10.0 * math.log10(signal / noise)
    if not math.isfinite(value):
        return -SNR_LIMIT_DB
    return max(-SNR_LIMIT_DB, min(SNR_LIMIT_DB, value))


def run_candidate(inputs: dict, ctx: WorkerCtx):
    """Run the implementation under optimisation."""
    raise NotImplementedError(
        "run_candidate must invoke the launcher for " + $launcher_hint_literal
    )


def run_reference(inputs: dict, ctx: WorkerCtx):
    """Independent reference via torch.distributed."""
$reference_body


def check_case(
    shape: tuple[int, ...],
    ctx: WorkerCtx,
    snr_threshold: float = SNR_FLOOR_DB,
    seed: int = 0,
) -> dict:
    """Parity of candidate vs reference for one shape.

    Two candidate calls are issued back to back and validated only afterwards.
    Scratch reused between consecutive collectives races exactly when the next
    call lands before the previous output is read, so a single validated call
    can never expose it -- and dropping a barrier is the change most likely to
    introduce that race.
    """
    seeds = (seed, seed + 1)
    refs = [run_reference(make_inputs(shape, ctx, item), ctx) for item in seeds]
    pending = [make_inputs(shape, ctx, item) for item in seeds]
    got = [run_candidate(inputs, ctx) for inputs in pending]
    for produced in got:
        if not isinstance(produced, torch.Tensor):
            raise TypeError("run_candidate must return a torch.Tensor")
    value = min(snr_db(ref, produced) for ref, produced in zip(refs, got))
    score = torch.tensor([value], dtype=torch.float32, device=ctx.device)
    dist.all_reduce(score, op=dist.ReduceOp.MIN, group=ctx.group)
    value = float(score.item())
    return {"shape": list(shape), "snr_db": value, "passed": value >= snr_threshold}


def capture_chain(inputs: dict, ctx: WorkerCtx, chain: int):
    """Capture a chain of candidate calls for device-side timing."""
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(chain):
                run_candidate(inputs, ctx)
    current_stream.wait_stream(capture_stream)
    return graph


def bench_case(
    shape: tuple[int, ...],
    ctx: WorkerCtx,
    warmup: int = 20,
    iters: int = 100,
    repeat: int = 1,
) -> float:
    """Return graph-replay latency, maximised across ranks.

    Every iteration is captured into one graph and replayed as a unit, with a
    single barrier before the timed region. Re-synchronising the ranks before
    each call would hide the arrival skew a collective actually pays, and it
    systematically flatters any change that removes an internal barrier.
    """
    inputs = make_inputs(shape, ctx)
    for _ in range(warmup):
        run_candidate(inputs, ctx)
    torch.cuda.synchronize()

    chain = max(1, iters)
    graph = capture_chain(inputs, ctx, chain)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rounds = []

    graph.replay()
    torch.cuda.synchronize()
    for _ in range(max(1, repeat)):
        dist.barrier(group=ctx.group)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latency_ms = start.elapsed_time(end) / chain
        if not math.isfinite(latency_ms) or latency_ms <= 0.0:
            raise RuntimeError("graph replay produced invalid latency")
        rounds.append(latency_ms)
    del graph

    median = torch.tensor([statistics.median(rounds)], device=ctx.device)
    dist.all_reduce(median, op=dist.ReduceOp.MAX, group=ctx.group)
    value = float(median.item())
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("cross-rank latency is invalid")
    return value


def payload_bytes(shape: tuple[int, ...], ctx: WorkerCtx) -> int:
    """Return the bytes one case moves across the fabric."""
    elements = 1
    for dim in shape:
        elements *= dim
    if GATHERED_OUTPUT:
        elements *= ctx.world_size
    return elements * torch.empty((), dtype=DTYPES[DTYPE]).element_size()


def case_group(shape: tuple[int, ...], ctx: WorkerCtx) -> str:
    """Return the scoring group a case belongs to.

    The two regimes respond to different edits, so a gain in one says nothing
    about the other and forge-loop scores them apart.
    """
    if payload_bytes(shape, ctx) >= REGIME_CUT_BYTES:
        return "fabric_bound"
    return "barrier_bound"


def case_bandwidth(shape: tuple[int, ...], ctx: WorkerCtx, latency_ms: float) -> dict:
    """Return payload bytes plus algorithm and bus bandwidth for one case.

    Latency alone cannot distinguish a faster transfer from a cheaper barrier.
    ``busbw`` applies the standard per-op correction so the number stays
    comparable across rank counts: a collective moving the same payload over
    more ranks does strictly more link work for the same wall time.
    """
    payload = payload_bytes(shape, ctx)
    algbw_gbps = payload / (latency_ms * 1.0e6)
    ranks = max(1, ctx.world_size)
    correction = BUSBW_NUMERATOR * (ranks - 1) / ranks
    return {
        "bytes": payload,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": algbw_gbps * correction,
    }


def profile_case(ctx: WorkerCtx) -> None:
    """Dispatch the first case for hardware profiling."""
    inputs = make_inputs(tuple(CASES[0]), ctx)
    for _ in range(3):
        run_candidate(inputs, ctx)
    torch.cuda.synchronize()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI accepted by forge-loop."""
    p = argparse.ArgumentParser(description="Collective kernel driver ($collective_op)")
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--profile-run", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeat", type=int, default=1, help="in-process repeats of the sweep")
    return p


def case_id(shape: tuple[int, ...]) -> str:
    """Return a stable identifier for one measured shape."""
    return "x".join(str(dim) for dim in shape)


def main(argv: list[str]) -> int:
    """Run correctness, benchmark, or profile mode on every rank."""
    ctx = None
    result = {
        "kernel": $kernel_name_literal,
        "collective_op": "$collective_op",
        "world_size": WORLD_SIZE,
    }
    try:
        args = build_parser().parse_args(argv)
        if args.warmup < 0 or args.iters <= 0 or args.repeat <= 0:
            raise ValueError(
                "warmup must be non-negative; iters and repeat must be positive"
            )

        rc = self_launch(argv)
        if rc >= 0:
            if rc != 0:
                result.update(
                    {
                        "status": "failed",
                        "error_class": "self_launch_failed",
                        "error": f"torchrun exited with rc={rc}",
                    }
                )
                print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            return rc

        ctx = init_worker()
        result["world_size"] = ctx.world_size
        if args.profile_run:
            profile_case(ctx)
            result["status"] = "ok"
        elif args.bench_mode:
            timings = {
                case_id(tuple(s)): bench_case(tuple(s), ctx, args.warmup, args.iters, args.repeat)
                for s in CASES
            }
            result["latency_ms"] = timings
            result["mean_ms"] = statistics.fmean(timings.values())
            if not math.isfinite(result["mean_ms"]) or result["mean_ms"] <= 0.0:
                raise RuntimeError("benchmark mean latency is invalid")
            result["bandwidth"] = {
                case_id(tuple(s)): case_bandwidth(
                    tuple(s), ctx, timings[case_id(tuple(s))]
                )
                for s in CASES
            }
            groups: dict = {}
            for s in CASES:
                groups.setdefault(case_group(tuple(s), ctx), []).append(
                    timings[case_id(tuple(s))]
                )
            result["group_ms"] = {
                name: math.exp(statistics.fmean(math.log(v) for v in values))
                for name, values in groups.items()
            }
            result["status"] = "ok"
            if ctx.rank == 0:
                for name, latency_ms in timings.items():
                    band = result["bandwidth"][name]
                    print(f"case_ms: {name} {latency_ms:.9f}", flush=True)
                    print(
                        f"case_bw: {name} bytes={band['bytes']} "
                        f"algbw={band['algbw_gbps']:.3f}GB/s "
                        f"busbw={band['busbw_gbps']:.3f}GB/s",
                        flush=True,
                    )
                for name, score_ms in sorted(result["group_ms"].items()):
                    print(f"group_ms: {name} {score_ms:.9f}", flush=True)
                print(f"mean_ms: {result['mean_ms']:.9f}", flush=True)
        else:
            checks = [check_case(tuple(s), ctx) for s in CASES]
            result["correctness"] = checks
            result["correctness_passed"] = all(c["passed"] for c in checks)
            result["status"] = "ok" if result["correctness_passed"] else "failed"
            if ctx.rank == 0:
                minimum = min(c["snr_db"] for c in checks)
                print(f"SNR: {minimum:.6f} dB", flush=True)
                print(f"allclose: {str(result['correctness_passed']).lower()}", flush=True)
        if ctx.rank == 0:
            print(
                json.dumps(result, sort_keys=True, allow_nan=False),
                flush=True,
            )
        return 0 if result.get("status") == "ok" else 1
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_class": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        rank_zero = (
            ctx.rank == 0
            if ctx is not None
            else os.environ.get("RANK", "0") == "0"
        )
        if rank_zero:
            print("allclose: false", flush=True)
            print(
                json.dumps(result, sort_keys=True, allow_nan=False),
                flush=True,
            )
        return 1
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                print(
                    f"process-group cleanup failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
''')


_PROGRAM_TEMPLATE = Template("""# Optimize `$kernel_name`

| | |
|---|---|
| Collective op | `$collective_op` |
| World size | $world_size |
| GPU time share | $gpu_pct% |
| Source | `$source_file` |
| Device symbol | `$launcher_hint` |

## What to do

1. Resolve the framework callable that launches the device symbol above and
   implement only `run_candidate` in `driver.py`.
2. Optimise the kernel source at `$source_file`.
3. Keep only changes that improve the traced cases and preserve parity.

## Gates

- SNR must remain at least $snr_floor dB against
  `torch.distributed.$collective_op`.
- SNR permits expected bf16 differences from fp32 reduction accumulation.
- Inputs remain rank-distinct.
- Parity issues two calls back to back before validating either, so scratch
  reused across consecutive collectives is exercised.
- Latency replays every iteration as one captured chain behind a single
  barrier, and takes the slowest rank.
- Cases are scored per regime, not blended: `group_ms: fabric_bound` and
  `group_ms: barrier_bound` split the suite at a 1 MiB payload. A win needs one
  clear gain and no clear loss across groups, so speeding up the small payloads
  cannot pay for a regression on the large ones.

## Shapes under test

$cases_md

## Reading the bench output

`--bench-mode` also prints `case_bw: <case> bytes=… algbw=… busbw=…`. Latency
alone cannot tell a faster transfer from a cheaper barrier; `busbw` against the
fabric peak tells you which of the two a change actually bought, and which
regime still has headroom.

## Constraints

- Every rank runs the same code path; do not branch on rank in the fast path.
- Do not weaken or bypass the parity gate.
- Do not add a barrier inside the timed region to stabilise the measurement.
- Keep graph replay and `--nproc-per-node=$world_size`.
""")


def generate_collective_driver(
    candidate: dict[str, Any],
    out_dir: Path | str,
    *,
    tp: int,
) -> dict[str, str]:
    """Write a strict collective driver and Forge task brief."""
    if not isinstance(candidate, dict):
        raise TypeError("collective candidate must be a mapping")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    op = _collective_op(candidate)
    world_size = _world_size(tp)
    shapes = _parse_shapes(candidate)
    if op == "reduce_scatter":
        # The reference scatters dim 0 across ranks, so an indivisible extent
        # would compare against a truncated output rather than fail loudly.
        ragged = [s for s in shapes if s[0] % world_size]
        if ragged:
            raise ValueError(f"reduce_scatter shapes must divide across {world_size} ranks: {ragged}")
    dtype = _dtype_of(candidate)
    kernel_name_raw = candidate.get("device_kernel_name") or candidate.get("name")
    source_file_raw = candidate.get("source_file")
    function_raw = candidate.get("source_function")
    if not isinstance(kernel_name_raw, str) or not kernel_name_raw.strip():
        raise ValueError("collective candidate has no kernel name")
    if (
        not isinstance(source_file_raw, str)
        or not source_file_raw.strip()
        or not isinstance(function_raw, str)
        or not function_raw.strip()
    ):
        raise ValueError("collective candidate requires source_file and source_function")
    kernel_name = kernel_name_raw.strip()
    source_file = source_file_raw.strip()
    function = function_raw.strip()
    gpu_pct_raw = candidate.get("gpu_pct")
    if isinstance(gpu_pct_raw, bool) or not isinstance(gpu_pct_raw, (int, float)):
        raise ValueError("collective candidate has invalid gpu_pct")
    gpu_pct = float(gpu_pct_raw)
    if not math.isfinite(gpu_pct) or gpu_pct < 0:
        raise ValueError("collective candidate gpu_pct must be finite and non-negative")

    line = candidate.get("source_line")
    if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line <= 0):
        raise ValueError("collective candidate source_line must be positive")
    if line is not None:
        launcher_hint = f"{source_file}({line}): {function}"
    else:
        launcher_hint = f"{source_file}: {function}"

    driver_src = _DRIVER_TEMPLATE.substitute(
        collective_op=op,
        world_size=world_size,
        snr_floor=SNR_FLOOR_DB,
        cases=json.dumps([list(s) for s in shapes]),
        dtype=dtype,
        launcher_hint_literal=repr(launcher_hint),
        kernel_name_literal=repr(kernel_name),
        reference_body=_REFERENCE_BODIES[op],
        busbw_numerator=_BUSBW_NUMERATORS[op],
        gathered_output=(op == "all_gather"),
    )
    program_src = _PROGRAM_TEMPLATE.substitute(
        kernel_name=kernel_name,
        collective_op=op,
        world_size=world_size,
        gpu_pct=gpu_pct,
        source_file=source_file,
        launcher_hint=launcher_hint,
        snr_floor=SNR_FLOOR_DB,
        cases_md="\n".join(f"- `{tuple(s)}` ({dtype})" for s in shapes),
    )

    driver_path = out_path / "driver.py"
    program_path = out_path / "program.md"
    driver_path.write_text(driver_src, encoding="utf-8")
    program_path.write_text(program_src, encoding="utf-8")

    return {
        "driver": str(driver_path),
        "program": str(program_path),
        "collective_op": op,
        "world_size": str(world_size),
    }


__all__ = ["SNR_FLOOR_DB", "generate_collective_driver"]
