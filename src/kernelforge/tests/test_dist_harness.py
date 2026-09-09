# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The guarantees a multi-rank driver gets by not writing them.

The harness exists so that a list of properties stops being requirements a
driver author is asked to remember. These tests hold it to the two that are
enforced by construction rather than by checking -- inputs that differ across
ranks and across the two parity calls, and both candidate calls issued before
either is compared -- plus the refusals that keep a run from being reported at
all.

The cross-rank collectives are replaced with a single-process stand-in so the
capture, timing and comparison paths run for real on the GPU that is present.
What that leaves untested is the part that needs more than one device: the
device binding, and any behaviour of a real NCCL/RCCL collective.
"""

from __future__ import annotations

import math

import pytest
import torch

from kernelforge.loop import dist_harness
from kernelforge.loop.dist_harness import Case, HarnessError, RankContext

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class _SoloDist:
    """torch.distributed for one rank: every reduction is already the answer."""

    class ReduceOp:
        MIN = "min"
        MAX = "max"

    def __init__(self) -> None:
        self.barriers = 0
        self.reductions: list[str] = []

    def all_reduce(self, tensor, op):  # noqa: ARG002 - one rank needs no reduction
        self.reductions.append(str(op))

    def barrier(self):
        self.barriers += 1

    def is_initialized(self):
        return True

    def destroy_process_group(self):
        return None


@pytest.fixture
def solo(monkeypatch) -> _SoloDist:
    fake = _SoloDist()
    monkeypatch.setattr(dist_harness, "dist", fake)
    return fake


@pytest.fixture
def context() -> RankContext:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return RankContext(rank=0, local_rank=0, world_size=2, device=device, case_id="default")


def _case(**overrides) -> Case:
    def build_inputs(ctx):
        return torch.randn(64, device=ctx.device)

    def call_candidate(ctx, inputs):  # noqa: ARG001 - the shape under test needs no ctx
        return inputs * 2

    def reference(ctx, inputs):  # noqa: ARG001
        return inputs * 2

    fields = {
        "case_id": "default",
        "build_inputs": build_inputs,
        "call_candidate": call_candidate,
        "reference": reference,
    }
    fields.update(overrides)
    return Case(**fields)


# --- what the harness refuses outright ---------------------------------------


def test_a_single_rank_task_has_no_business_here():
    with pytest.raises(HarnessError, match="needs no harness"):
        dist_harness.run([_case()], world_size=1)


def test_a_driver_with_no_cases_is_refused():
    with pytest.raises(HarnessError, match="no cases"):
        dist_harness.run([], world_size=2)


def test_an_unknown_case_id_names_the_ones_that_exist():
    with pytest.raises(HarnessError, match="unknown case 'absent'"):
        dist_harness._select([_case()], "absent")


def test_a_rank_count_torchrun_disagrees_with_is_refused(monkeypatch):
    """An inherited WORLD_SIZE must not pass for the count that was launched."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    with pytest.raises(HarnessError, match="launched 1 ranks"):
        dist_harness._init_worker(4)


def test_fewer_visible_gpus_than_ranks_refuses_before_launching(monkeypatch, capsys):
    """The machine is wrong, so nothing is measured and nothing is printed as a time."""
    monkeypatch.setattr(dist_harness.torch.cuda, "device_count", lambda: 2)

    code = dist_harness._relaunch_under_torchrun(8, [])

    assert code == 2
    assert "8 GPUs and 2 are visible" in capsys.readouterr().err


def test_the_relaunch_forwards_the_arguments_it_was_given(monkeypatch):
    """Re-exec is the same driver, same request -- only now under torchrun."""
    seen: dict = {}

    def _fake_run(command, check):  # noqa: ARG001
        seen["command"] = command
        return type("_Completed", (), {"returncode": 0})()

    monkeypatch.setattr(dist_harness.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(dist_harness.subprocess, "run", _fake_run)

    assert dist_harness._relaunch_under_torchrun(8, ["--bench-mode", "--iters", "5"]) == 0
    command = seen["command"]
    assert "torch.distributed.run" in command
    assert "--nproc-per-node=8" in command
    assert command[-3:] == ["--bench-mode", "--iters", "5"]
    # A detached torchrun would outlive the timeout kill still holding its GPUs.
    assert "start_new_session" not in seen


# --- the guarantees the driver gets for free ---------------------------------


@requires_gpu
def test_the_two_parity_calls_cannot_receive_the_same_inputs(solo, context):
    """Identical inputs let a collective that drops a rank still pass parity.

    The harness reseeds between the calls rather than trusting ``build_inputs``
    to vary, so a driver drawing from the ambient generator cannot produce them.
    """
    seen: list[torch.Tensor] = []

    def build_inputs(ctx):
        drawn = torch.randn(32, device=ctx.device)
        seen.append(drawn)
        return drawn

    dist_harness._check_case(_case(build_inputs=build_inputs), context, seed=11)

    assert len(seen) == 2
    assert not torch.equal(seen[0], seen[1])


@requires_gpu
def test_ranks_do_not_receive_the_same_inputs_as_each_other(context):
    """The seed carries the rank, so no two ranks draw the same tensor."""
    drawn = []
    for rank in (0, 1):
        scoped = RankContext(
            rank=rank,
            local_rank=rank,
            world_size=2,
            device=context.device,
            case_id="default",
        )
        drawn.append(dist_harness._seeded_inputs(_case(), scoped, seed=11))

    assert not torch.equal(drawn[0], drawn[1])


@requires_gpu
def test_both_candidate_calls_are_issued_before_either_is_compared(solo, context):
    """A second call overwriting the first's scratch buffer is only visible this way.

    Comparing each call the moment it is issued cannot see the overwrite,
    because nothing has overwritten anything yet -- and removing a
    synchronization, the change this lane's optimizer is most likely to make, is
    exactly what opens that race.
    """
    order: list[str] = []

    def call_candidate(ctx, inputs):  # noqa: ARG001
        order.append("candidate")
        return inputs * 2

    def reference(ctx, inputs):  # noqa: ARG001
        order.append("reference")
        return inputs * 2

    dist_harness._check_case(
        _case(call_candidate=call_candidate, reference=reference),
        context,
        seed=11,
    )

    assert order == ["candidate", "candidate", "reference", "reference"]


@requires_gpu
def test_correctness_is_scored_by_the_worst_rank(solo, context):
    """One wrong rank is a wrong collective, so the reduction takes the minimum."""
    dist_harness._check_case(_case(), context, seed=11)

    assert solo.reductions == ["min"]


@requires_gpu
def test_a_candidate_that_disagrees_scores_a_finite_snr(solo, context):
    def call_candidate(ctx, inputs):  # noqa: ARG001
        return inputs * 2 + 0.5

    snr = dist_harness._check_case(_case(call_candidate=call_candidate), context, seed=11)

    assert math.isfinite(snr)


@requires_gpu
def test_an_exact_candidate_is_not_penalised(solo, context):
    assert dist_harness._check_case(_case(), context, seed=11) == math.inf


# --- timing ------------------------------------------------------------------


@requires_gpu
def test_the_benchmark_replays_a_graph_once_per_timed_iteration(solo, context):
    """Preflight counts replays, so eager timing is not a style choice here."""
    replays = []
    original = torch.cuda.CUDAGraph.replay

    def counted(self, *args, **kwargs):
        replays.append(1)
        return original(self, *args, **kwargs)

    torch.cuda.CUDAGraph.replay = counted
    try:
        samples = dist_harness._bench_case(_case(), context, seed=11, warmup=2, iters=6)
    finally:
        torch.cuda.CUDAGraph.replay = original

    assert len(samples) == 6
    assert len(replays) >= 6


@requires_gpu
def test_the_timed_region_carries_no_barrier(solo, context):
    """One barrier, before the first sample, and never between the brackets.

    A barrier inside the timed region returns the ranks to a fully synchronised
    state on every sample, which is what makes deleting an internal barrier look
    free.
    """
    dist_harness._bench_case(_case(), context, seed=11, warmup=2, iters=5)

    assert solo.barriers == 1


@requires_gpu
def test_each_sample_reports_the_slowest_rank(solo, context):
    """A collective is as fast as its laggard, so the reduction takes the maximum."""
    dist_harness._bench_case(_case(), context, seed=11, warmup=2, iters=4)

    assert solo.reductions == ["max"] * 4


# --- the scoring helpers -----------------------------------------------------


def test_the_worst_tensor_decides_the_snr():
    """Right in one field and wrong in another is wrong."""
    want = [torch.ones(8), torch.ones(8)]
    got = [torch.ones(8), torch.ones(8) + 1.0]

    assert math.isfinite(dist_harness._snr_db(got, want))
    assert dist_harness._snr_db(want, want) == math.inf


def test_a_shape_disagreement_is_a_refusal_not_a_score():
    with pytest.raises(HarnessError, match="shape"):
        dist_harness._snr_db(torch.ones(4), torch.ones(8))


def test_nothing_to_compare_is_a_refusal():
    with pytest.raises(HarnessError, match="no tensors"):
        dist_harness._snr_db("not a tensor", "not a tensor")


def test_tensors_are_found_in_the_shapes_a_driver_returns():
    single = torch.ones(2)
    assert dist_harness._tensors(single) == [single]
    assert dist_harness._tensors((single, "skip")) == [single]
    assert dist_harness._tensors({"out": single}) == [single]
    assert dist_harness._tensors(None) == []


def test_the_loops_arguments_are_accepted_and_the_rest_ignored():
    """forge-loop passes flags this harness has no use for; refusing them would
    fail the run for its shape rather than for its measurement."""
    args = dist_harness._parse_args(["--warmup", "4", "--iters", "9", "--bench-mode", "--shape", "big", "--extra"])

    assert (args.warmup, args.iters, args.bench_mode) == (4, 9, True)
