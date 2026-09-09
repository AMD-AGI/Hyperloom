# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measurement harness a multi-rank driver runs inside.

A collective is only measured honestly when a list of properties all hold at
once: the ranks really launched, each owns its own device, inputs differ per
rank, the timed region carries no synchronization, parity is judged over two
back-to-back calls, the reported time is the slowest rank's, and the process
group is torn down. Left to a driver author, each one is a separate chance to
produce a plausible number that describes something other than the collective.

So the driver does not own them. This module owns the run -- torchrun bootstrap,
device binding, process-group lifetime, per-rank seeding, warmup, graph capture,
barrier placement, the cross-rank reduction and the stdout contract -- and the
driver supplies only what is specific to its operator: how to build inputs, how
to call the candidate, and what the answer should be. The properties above stop
being requirements a driver is asked to remember and become code it does not
write.

Two of them are worth naming because they are enforced by construction rather
than by checking. Inputs cannot be identical across ranks or across the two
parity calls, because this module reseeds between them rather than trusting
``build_inputs`` to vary. And no barrier can land inside the timed region,
because the driver never sees the timed region at all.

Copied next to the driver during task preparation, imported as ``dist_harness``.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

#: Offset between the two parity calls' seeds. Any fixed non-zero value works;
#: it only has to make the second call's inputs differ from the first's.
_PARITY_SEED_STRIDE = 7919

#: Replays issued before timing so the graph reaches a steady state.
_REPLAY_WARMUP = 3

#: How many cases this process measured inside the harness. Read by the graph
#: probe's ``sitecustomize`` to establish that the ranks ran here rather than in
#: a driver that reimplemented the launch: every property this module enforces
#: is worth nothing on a driver that did not use it, and that single fact is
#: what the probe checks instead of re-deriving the properties from the run.
MEASURED = [0]


@dataclass(frozen=True)
class RankContext:
    """What a driver callback is allowed to know about the rank it runs on."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    case_id: str


@dataclass(frozen=True)
class Case:
    """One measurable shape, and the three operator-specific callables for it.

    ``reference`` must compute the answer with the matching ``torch.distributed``
    collective. It is the one requirement here that this module cannot enforce:
    a single-GPU reference is still a function that returns a tensor, and a
    candidate that quietly drops a rank's contribution matches it.
    """

    case_id: str
    build_inputs: Callable[[RankContext], Any]
    call_candidate: Callable[[RankContext, Any], Any]
    reference: Callable[[RankContext, Any], Any]


class HarnessError(RuntimeError):
    """A refusal to measure, as opposed to a measurement that came out badly."""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--bench-case", default="")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default="")
    # Accepted and ignored: the loop passes them to every driver, and refusing
    # an argument the caller always sends would fail the run for its shape.
    parser.add_argument("--shape", default="default")
    parser.add_argument("--mode", default="full")
    known, _unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    return known


def _relaunch_under_torchrun(world_size: int, argv: Sequence[str]) -> int:
    """Re-exec this driver under torchrun, once, from the process that has no rank.

    ``start_new_session`` is deliberately not used: the caller kills this
    process group when the task runs out of time, and a detached torchrun would
    outlive it still holding its GPUs.
    """
    visible = torch.cuda.device_count()
    if visible < world_size:
        print(
            f"error: this task needs {world_size} GPUs and {visible} are visible",
            file=sys.stderr,
        )
        return 2
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        os.path.abspath(sys.argv[0]),
        *argv,
    ]
    return subprocess.run(command, check=False).returncode


def _init_worker(world_size: int) -> RankContext:
    """Bind this rank's device and join the process group, or refuse to run."""
    local_rank = int(os.environ["LOCAL_RANK"])
    # WORLD_SIZE, not RANK: job launchers routinely preset RANK=0/WORLD_SIZE=1 in
    # the environment forge-loop inherits, and a rank count inherited from one of
    # them would let a one-rank run pass for the real thing.
    launched = int(os.environ.get("WORLD_SIZE") or 0)
    if launched != world_size:
        raise HarnessError(f"torchrun launched {launched} ranks, but this task is measured on {world_size}")
    if torch.cuda.device_count() < world_size:
        raise HarnessError(
            f"rank {local_rank} sees {torch.cuda.device_count()} GPUs, fewer than the {world_size} this task needs"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    return RankContext(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        case_id="",
    )


def _seeded_inputs(case: Case, context: RankContext, seed: int) -> Any:
    """Build one call's inputs under a seed this module, not the driver, chose.

    Every rank and every parity call gets a different seed, so inputs that are
    identical across ranks -- which let a collective that drops a rank still
    pass parity -- cannot be produced by a ``build_inputs`` that simply draws
    from the ambient generator.
    """
    torch.manual_seed(seed + context.rank)
    torch.cuda.manual_seed_all(seed + context.rank)
    return case.build_inputs(context)


def _snr_db(candidate: Any, expected: Any) -> float:
    """Signal-to-noise ratio in dB over every tensor in the two answers.

    The worst tensor decides, because an operator whose output is right in one
    field and wrong in another is wrong.
    """
    pairs = list(zip(_tensors(candidate), _tensors(expected), strict=False))
    if not pairs:
        raise HarnessError("the candidate and the reference produced no tensors to compare")
    worst = math.inf
    for got, want in pairs:
        got_f = got.detach().float()
        want_f = want.detach().float()
        if got_f.shape != want_f.shape:
            raise HarnessError(f"candidate returned shape {tuple(got_f.shape)}, reference {tuple(want_f.shape)}")
        noise = torch.sum((want_f - got_f) ** 2).item()
        signal = torch.sum(want_f**2).item()
        if noise <= 0.0:
            continue
        if signal <= 0.0:
            worst = -math.inf
            continue
        worst = min(worst, 10.0 * math.log10(signal / noise))
    return worst


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def _worst_across_ranks(value: float, context: RankContext) -> float:
    """The worst rank's value, because one wrong rank is a wrong collective."""
    tensor = torch.tensor([value], device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def _slowest_across_ranks(value: float, context: RankContext) -> float:
    """The slowest rank's time, because a collective is as fast as its laggard."""
    tensor = torch.tensor([value], device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _check_case(case: Case, context: RankContext, seed: int) -> float:
    """Validate the candidate over two back-to-back calls, and return worst SNR.

    Both candidate calls are issued before either is compared. A compiled
    collective usually writes into a registered scratch buffer, so a second call
    can overwrite the first call's output before anything has read it --
    comparing each call the moment it is issued cannot see that, because nothing
    has overwritten anything yet. Removing a synchronization is the change this
    lane's optimizer is most likely to make, and it is exactly what opens that
    race.
    """
    first_inputs = _seeded_inputs(case, context, seed)
    second_inputs = _seeded_inputs(case, context, seed + _PARITY_SEED_STRIDE)
    first_got = case.call_candidate(context, first_inputs)
    second_got = case.call_candidate(context, second_inputs)
    first_want = case.reference(context, first_inputs)
    second_want = case.reference(context, second_inputs)
    torch.cuda.synchronize()
    local = min(_snr_db(first_got, first_want), _snr_db(second_got, second_want))
    return _worst_across_ranks(local, context)


def _bench_case(case: Case, context: RankContext, seed: int, warmup: int, iters: int) -> list[float]:
    """Time one case under graph replay, returning the slowest rank per iteration.

    The barrier is issued once, before the first ``start.record()``, and never
    again. A barrier between the timing brackets returns the ranks to a fully
    synchronised state on every sample, which is the condition that makes
    deleting an internal barrier look free: it turns a slower implementation
    into an apparent speedup.
    """
    inputs = _seeded_inputs(case, context, seed)

    def step() -> None:
        case.call_candidate(context, inputs)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(_REPLAY_WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(_slowest_across_ranks(start.elapsed_time(end), context))
    return samples


def _select(cases: Sequence[Case], case_id: str) -> list[Case]:
    if not case_id:
        return list(cases)
    chosen = [case for case in cases if case.case_id == case_id]
    if not chosen:
        raise HarnessError(f"unknown case {case_id!r}; this driver declares {[case.case_id for case in cases]}")
    return chosen


def _context_for(context: RankContext, case: Case) -> RankContext:
    return RankContext(
        rank=context.rank,
        local_rank=context.local_rank,
        world_size=context.world_size,
        device=context.device,
        case_id=case.case_id,
    )


def _run_worker(cases: Sequence[Case], context: RankContext, seed: int, args: argparse.Namespace) -> int:
    speak = context.rank == 0
    MEASURED[0] += 1
    if args.profile_run:
        for case in _select(cases, args.profile_case)[:1]:
            scoped = _context_for(context, case)
            inputs = _seeded_inputs(case, scoped, seed)
            for _ in range(3):
                case.call_candidate(scoped, inputs)
            torch.cuda.synchronize()
        return 0

    if args.bench_mode:
        for case in _select(cases, args.bench_case):
            scoped = _context_for(context, case)
            samples = _bench_case(case, scoped, seed, args.warmup, args.iters)
            if speak:
                for sample in samples:
                    print(f"wall_ms: {sample:.6f}")
                print(f"case_ms: {case.case_id} {statistics.median(samples):.6f}")
        return 0

    worst = math.inf
    for case in _select(cases, ""):
        scoped = _context_for(context, case)
        worst = min(worst, _check_case(case, scoped, seed))
    if speak:
        if worst == math.inf:
            print("allclose: True")
        else:
            print(f"SNR: {worst:.2f} dB")
    return 0


def run(cases: Sequence[Case], *, world_size: int, seed: int = 1234, argv: Sequence[str] | None = None) -> int:
    """Run ``cases`` on ``world_size`` ranks and print the forge-loop contract.

    Called from the driver's ``__main__``. Returns the process exit code, so a
    refusal to measure is a non-zero exit rather than a plausible number.
    """
    if world_size < 2:
        raise HarnessError(f"dist_harness measures collectives; world_size {world_size} needs no harness")
    if not cases:
        raise HarnessError("no cases to measure")
    passthrough = list(argv) if argv is not None else sys.argv[1:]
    if os.environ.get("LOCAL_RANK") is None:
        return _relaunch_under_torchrun(world_size, passthrough)

    args = _parse_args(passthrough)
    context = _init_worker(world_size)
    try:
        return _run_worker(cases, context, seed, args)
    finally:
        # Before returning either way: a rank that exits with its group standing
        # leaves the next stage to inherit a wedged communicator.
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "Case",
    "HarnessError",
    "RankContext",
    "run",
]
