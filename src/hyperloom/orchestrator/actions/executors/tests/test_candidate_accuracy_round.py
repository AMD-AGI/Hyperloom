# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate whose throughput cleared the KEEP threshold runs the D9 accuracy round into ``<slot>/accuracy``.

The gates still grade AgentX accuracy through ``parse_eval_results``; these tests pin only that the round runs, where,
under which lease and cap, and that its failure never turns into a verdict.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors._grid_base import VARIANT_CONFIG_FILE, variant_config_of
from hyperloom.orchestrator.actions.executors._suite_config import ACCURACY_DIR, AccuracyRound
from hyperloom.orchestrator.actions.executors.benchmark_run import accuracy
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import (
    ACCURACY_ROUND_KEY,
    AccuracyRoundOutcome,
    accuracy_round_due,
    record_accuracy_round,
    run_accuracy_round,
    run_candidate_accuracy_round,
)
from hyperloom.orchestrator.state.shared_state import SharedState

_AGENTX = "agentx"


class _Lease:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def suite_rounds(tmp_path, monkeypatch) -> list[AccuracyRound]:
    """The accuracy rounds handed to inference_suite, each completing in ``<slot>/accuracy``."""
    requests: list[AccuracyRound] = []

    def run_accuracy_suite_round(route, request, *, on_poll):
        requests.append(request)
        return SimpleNamespace(report=SimpleNamespace(failed_step=None), layout=SimpleNamespace(root=request.slot))

    monkeypatch.setattr(accuracy, "session_route", lambda session: SimpleNamespace(session=session))
    monkeypatch.setattr(accuracy, "run_accuracy_suite_round", run_accuracy_suite_round)
    monkeypatch.setattr(accuracy, "_num_gpus_for_config", lambda config_path: 1)
    return requests


@pytest.mark.parametrize(
    ("mode", "switch", "mlperf", "due"),
    [
        (_AGENTX, True, False, True),
        (_AGENTX, False, False, False),
        ("", True, False, False),
        (_AGENTX, True, True, False),
    ],
    ids=["native_agentx", "switched_off", "fixed_workload", "mlperf"],
)
def test_only_a_native_agentx_session_that_kept_the_switch_runs_the_round(monkeypatch, mode, switch, mlperf, due):
    monkeypatch.setattr(accuracy, "is_mlperf_backend", lambda: mlperf)
    for name in ("HYPERLOOM_AGENTX", "HYPERLOOM_BENCHMARK_CONFIG", "HYPERLOOM_AGENTIC_BACKEND"):
        monkeypatch.delenv(name, raising=False)

    assert bool(accuracy_round_due(SharedState(benchmark_mode=mode, agentx_accuracy=switch))) is due


def test_a_candidate_round_runs_on_the_lease_its_gate_holds_and_leaves_it_open(tmp_path, suite_rounds, monkeypatch):
    lease = _Lease()
    monkeypatch.setattr(accuracy, "resolve_benchmark_timeouts", lambda: (60.0, 1800.0))
    monkeypatch.setattr(accuracy, "maybe_serving_lease", lambda num_gpus: pytest.fail("took a second lease"))
    slot = tmp_path / "v00"

    outcome = asyncio.run(
        run_candidate_accuracy_round(slot, slot / "point.yaml", SharedState(), session=tmp_path, lease=lease)
    )

    (request,) = suite_rounds
    assert (request.slot, request.config_path, request.timeout_sec) == (slot, slot / "point.yaml", 1800.0)
    assert request.serving_lease is lease and not lease.closed
    assert outcome == AccuracyRoundOutcome(slot, "")


def test_a_round_without_a_lease_runs_on_one_of_its_own_and_releases_it(tmp_path, suite_rounds, monkeypatch):
    owned = _Lease()
    monkeypatch.setattr(accuracy, "maybe_serving_lease", lambda num_gpus: owned)
    slot = tmp_path / "slot"

    asyncio.run(run_accuracy_round(AccuracyRound(slot / "point.yaml", slot, 900.0), session=tmp_path))

    assert suite_rounds[0].serving_lease is owned and owned.closed


def test_a_failed_round_is_recorded_with_a_warning_and_no_verdict(tmp_path) -> None:
    result: dict = {"status": "succeeded"}

    record_accuracy_round(result, AccuracyRoundOutcome(tmp_path / ACCURACY_DIR, "eval_failed"))

    assert result == {
        "status": "succeeded",
        ACCURACY_ROUND_KEY: {"run_dir": str(tmp_path / ACCURACY_DIR), "error_class": "eval_failed"},
        "nonfatal_warnings": ["agentx_accuracy_round_failed"],
    }


def test_a_grid_rounds_point_is_the_variant_config_beside_its_workspace(tmp_path) -> None:
    workspace = tmp_path / "variant_00_x" / "suite_run_0"

    assert variant_config_of(str(workspace)) == tmp_path / "variant_00_x" / VARIANT_CONFIG_FILE
