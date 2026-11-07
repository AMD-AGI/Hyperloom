# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate whose throughput cleared the KEEP threshold runs the D9 accuracy round into ``<slot>/accuracy``.

The gates still grade AgentX accuracy through ``parse_eval_results``; these tests pin only that the round runs, where,
under which lease and cap, and that its failure never turns into a verdict.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.tests._helpers import init_git_repo, integrate_extra

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


async def _integrate_gate(tmp_path, monkeypatch, *, tput: float, regraded_pass: bool | None = True):
    """Weigh an integrate bench at ``tput`` against a base of 1000 tok/s, the round regrading accuracy to
    ``regraded_pass``."""
    from hyperloom.orchestrator.actions.executors import _integrate_bench, _integrate_perf_gate
    from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor

    calls = []

    async def run_round(slot, launch, state, *, session):
        calls.append((slot, launch, session))
        return AccuracyRoundOutcome(slot / ACCURACY_DIR, "")

    def regrade(bench, bench_leg, result_dir, gate_evidence):
        gate_evidence["accuracy_pass"] = regraded_pass

    monkeypatch.setattr(_integrate_perf_gate, "accuracy_round_due", lambda state: True)
    monkeypatch.setattr(_integrate_bench, "run_candidate_accuracy_round", run_round)
    monkeypatch.setattr(_integrate_bench, "_grade_bench_accuracy", regrade)
    attempt = IntegrateAttempt(
        task_id="t-int",
        specialist_task_id="t-spec",
        shared_state=SharedState(baseline_tput=1000.0, current_best={"tput": 1000.0}),
        done_payload={},
        output_root=tmp_path / "run",
    )
    workspace = tmp_path / "run" / "variant_00_patch" / "benchmark_x"
    out = await IntegratePatchExecutor(session_dir=tmp_path / "session")._gate_perf(
        attempt=attempt,
        params={"base_tput": 1000.0, "keep_threshold_pct": 1.0, "require_accuracy_for_keep": True},
        bench_result={"status": "succeeded", "output_throughput": tput, "workspace": str(workspace)},
        gate_evidence={"accuracy_pass": True},
    )
    return out, calls, workspace.parent


@pytest.mark.asyncio
async def test_an_integrate_bench_that_cleared_the_threshold_is_weighed_on_the_grade_after_its_round(
    tmp_path, monkeypatch
):
    out, calls, slot = await _integrate_gate(tmp_path, monkeypatch, tput=1100.0, regraded_pass=False)

    assert calls == [(slot, slot / VARIANT_CONFIG_FILE, tmp_path / "session")]
    assert out["status"] != "kept" and out["accuracy_pass"] is False
    assert out["bench_result"][ACCURACY_ROUND_KEY] == {"run_dir": str(slot / ACCURACY_DIR), "error_class": ""}


@pytest.mark.asyncio
async def test_an_integrate_bench_below_the_threshold_runs_no_accuracy_round(tmp_path, monkeypatch) -> None:
    out, calls, _ = await _integrate_gate(tmp_path, monkeypatch, tput=1005.0)

    assert calls == []
    assert out["status"] == "reverted"


_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""
_SPECIALIST = "t-spec-acc"


def _enablement_bench(slot: Path, *, served: bool) -> dict:
    return {
        "status": "succeeded" if served else "failed",
        "output_throughput": 137.0 if served else 0.0,
        "completed_requests": 12 if served else 0,
        "workspace": str(slot / "benchmark_1"),
        "materialized_config": "",
        "effective_config": {},
    }


def _enablement_round(tmp_path, monkeypatch, *, served=True, agentx=True, agentx_accuracy=True, ended=""):
    """Gate one enablement round whose bench served or not; the gated result, the rounds it ran, and its slot."""
    from hyperloom.orchestrator.actions.executors import _integrate_bench

    session = tmp_path / "session"
    specialist = session / "runs" / "specialist" / _SPECIALIST
    (specialist / "worktree" / "patches").mkdir(parents=True)
    (specialist / "worktree" / "patches" / "001.patch").write_text(_PATCH, encoding="utf-8")
    (specialist / "specialist_done.json").write_text(
        json.dumps({"proposal_set": [], "patches_written": ["patches/001.patch"]}), encoding="utf-8"
    )
    repo = tmp_path / "framework"
    init_git_repo(repo)
    slot = tmp_path / "variant_00"
    calls = []

    async def run_round(slot, launch, state, *, session):
        calls.append((slot, launch))
        return AccuracyRoundOutcome(slot / ACCURACY_DIR, ended)

    async def bench_patch(*_args, **_kwargs):
        return _enablement_bench(slot, served=served), {"accuracy_pass": None, "enablement_accuracy": None}

    async def no_kb(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_integrate_bench, "run_candidate_accuracy_round", run_round)
    monkeypatch.setattr(_integrate_bench, "_grade_bench_accuracy", lambda *args: None)
    monkeypatch.setattr(accuracy, "is_mlperf_backend", lambda: False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    executor = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(executor, "_bench_patch", bench_patch)
    monkeypatch.setattr(executor, "_maybe_write_framework_kb_record", no_kb)
    params = {"specialist_task_id": _SPECIALIST, "framework_source_root": str(repo), "enablement": True}
    extra = integrate_extra(params)
    extra["shared_state"].benchmark_mode = _AGENTX if agentx else ""
    extra["shared_state"].agentx_accuracy = agentx_accuracy
    task = Task(task_id="t-int-acc", kind="integrate_patch", state="queued", params=params, idempotency_key="t")
    return asyncio.run(executor(RunnerContext(task=task, lease=None, extra=extra))), calls, slot


def test_a_served_agentx_enablement_round_runs_its_accuracy_round(tmp_path, monkeypatch) -> None:
    result, calls, slot = _enablement_round(tmp_path, monkeypatch)

    assert result["status"] == "kept"
    assert calls == [(slot, slot / VARIANT_CONFIG_FILE)]
    assert result["bench_result"][ACCURACY_ROUND_KEY] == {"run_dir": str(slot / ACCURACY_DIR), "error_class": ""}


@pytest.mark.parametrize(
    "case",
    [{"served": False}, {"agentx_accuracy": False}, {"agentx": False}],
    ids=["not_served", "switched_off", "fixed_workload"],
)
def test_an_enablement_round_that_did_not_serve_or_is_off_the_switch_runs_no_accuracy_round(
    tmp_path, monkeypatch, case
) -> None:
    result, calls, _ = _enablement_round(tmp_path, monkeypatch, **case)

    assert calls == []
    assert ACCURACY_ROUND_KEY not in result["bench_result"]


def test_a_failed_accuracy_round_keeps_the_enablement_verdict_with_a_warning(tmp_path, monkeypatch) -> None:
    result, _, _ = _enablement_round(tmp_path, monkeypatch, ended="eval_failed")

    assert result["status"] == "kept"
    assert result["bench_result"][ACCURACY_ROUND_KEY]["error_class"] == "eval_failed"
    assert "agentx_accuracy_round_failed" in result["bench_result"]["nonfatal_warnings"]
