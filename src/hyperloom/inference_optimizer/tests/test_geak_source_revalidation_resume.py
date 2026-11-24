# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A GEAK source measurement that was cut short leaves a session that resumes on the tree it started from."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_env
from hyperloom.orchestrator.actions.executors import integrate_patch as integrate_patch_module
from hyperloom.orchestrator.actions.executors.integrate import _integrate_bench as integrate_bench
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.enablement.recipe import keep_probe
from hyperloom.orchestrator.loop._promote_outcome import IntegrateRecoveryIncomplete
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.tests._helpers import init_git_repo, patch_integrate_patch_roots, variant_result

from .conftest import make_mock_critic_coordinator
from .test_geak_revalidation_dispatch import coordinator as coordinator

_PATCH = "\n".join(
    [
        "diff --git a/src.py b/src.py",
        "index 0000000..1111111 100644",
        "--- a/src.py",
        "+++ b/src.py",
        "@@ -1,2 +1,2 @@",
        " def f():",
        "-    return 1",
        "+    return 2",
        "",
    ]
)
_ORIGINAL = "def f():\n    return 1\n"
_PATCHED = "def f():\n    return 2\n"
_INCUMBENT = {
    "tput": 100.0,
    "total_throughput": 20000.0,
    "e2e_norm_intvty_p90": 450.0,
    "e2e_norm_intvty_p50": 450.0,
    "duration_seconds": 900.0,
    "request_error_rate": 0.0,
    "workload_identity": "agentx-workload",
    "extra_server_args": "",
    "extra_envs": {},
}
_REFUSAL = "pending integrate recovery is incomplete; refusing resume before measurement"


class _Bench:
    """The stubbed benchmark round: how it ends, and the task id of the journal each round ran under."""

    def __init__(self, session: Path, killed: Path):
        self.session, self.killed = session, killed
        self.ending = "reverted"
        self.task_ids: list[str] = []

    def snapshot(self) -> None:
        shutil.copytree(self.session, self.killed / "session", symlinks=True)

    def _measured(self, output_throughput: float):
        workspace = self.session / f"grid-{len(self.task_ids)}" / "benchmark_test"
        workspace.mkdir(parents=True)
        return [
            variant_result(
                name="v",
                status="succeeded",
                input_throughput=24800.0,
                output_throughput=output_throughput,
                total_token_throughput=output_throughput * 125,
                intvty_p90=459.0,
                intvty_p50=459.0,
                duration_seconds=900.0,
                request_error_rate=0.0,
                tpot_p90_ms=3.0,
                valid_measurement=True,
                diagnostic_only=False,
                workload_identity="agentx-workload",
                workspace=str(workspace),
            )
        ]

    async def run_grid(self, **_kwargs):
        journal = json.loads((self.session / "state.json").read_text())["pending_integrate"]
        self.task_ids.append(journal["task_id"])
        if self.ending == "killed_mid_bench":
            self.snapshot()
        if self.ending == "raised":
            raise RuntimeError("the benchmark round broke")
        if self.ending == "cancelled":
            raise asyncio.CancelledError
        if self.ending == "budget_stopped":
            return [
                variant_result(
                    name="v",
                    status="failed",
                    output_throughput=None,
                    error_class="session_time_exhausted",
                    error="stopped",
                )
            ]
        return self._measured(200.0 if self.ending == "killed_after_keep" else 50.0)

    def die_at_snapshot(self, coord) -> None:
        """Put the session directory, which holds the tree, back as the snapshot saw it: the process died there."""
        coord.db.raw.close()
        shutil.rmtree(self.session)
        shutil.copytree(self.killed / "session", self.session, symlinks=True)


def _register_executor(coord) -> None:
    coord.sub.register_executor("integrate_patch", IntegratePatchExecutor(session_dir=coord.session_dir))


@pytest.fixture
def bench(coordinator, tmp_path, tmp_path_factory, monkeypatch):
    """A GEAK product whose patch the real integration applies to a git tree; only the benchmark round is stubbed."""
    state = coordinator.shared_state
    state.benchmark_mode = "agentx"
    state.agentx_backend = "inferencex"
    config = tmp_path / "native.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    state.baseline_config_path = str(config)
    state.baseline_tput = 100.0
    state.current_best = dict(_INCUMBENT)
    framework = tmp_path / "framework"
    init_git_repo(framework)
    assert framework.is_relative_to(coordinator.session_dir)  # a copy of the session directory holds the tree too
    patch_integrate_patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    monkeypatch.setattr(keep_probe, "probe_keep_environment", lambda *_a, **_k: ({}, {}))
    directory = tmp_path / "geak-eval"
    directory.mkdir()
    patch = directory / "final.patch"
    patch.write_text(_PATCH)
    state.geak_result = {
        "status": "ok",
        "eval_dir": str(directory),
        "final_patch": str(patch),
        "final_throughput_tok_s": 99999.0,
        "accepted_config": {"flags": "", "env_map": {}},
        "accepted_kernels": [{"short_name": "kernel", "e2e_delta_pct": 10, "final_patch": str(patch)}],
    }
    state.save(coordinator.session_dir)
    stub = _Bench(coordinator.session_dir, tmp_path_factory.mktemp("killed"))
    monkeypatch.setattr(integrate_bench, "run_grid", stub.run_grid)
    monkeypatch.setattr(integrate_bench, "prepare_grid_benchmark_config", lambda *_a, **_k: config)
    _register_executor(coordinator)
    return stub


async def _measure(coord) -> None:
    try:
        await coord.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    except (RuntimeError, asyncio.CancelledError):
        pass


class _Resumed(NamedTuple):
    """A resume pass in a new Coordinator: the fixes it made, or why it refused."""

    coord: Any
    fixes: list[str]
    refusal: str = ""


@pytest.fixture
def resume():
    """Resume a session in a new Coordinator on the same session directory."""
    opened = []

    async def _resume(session: Path) -> _Resumed:
        coord = make_mock_critic_coordinator(session)
        opened.append(coord)
        await coord.writeback.replay_for_resume()
        try:
            report = await coord.writeback._resume_consistency_pass()
        except IntegrateRecoveryIncomplete as exc:
            return _Resumed(coord, [], str(exc))
        return _Resumed(coord, [fix["kind"] for fix in report["fixes"]])

    yield _resume
    for coord in opened:
        coord.db.raw.close()


def _source(session: Path) -> str:
    return (session / "framework" / "src.py").read_text()


def _uncommitted(session: Path) -> str:
    done = subprocess.run(
        ["git", "-C", str(session / "framework"), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return done.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["budget_stopped", "raised", "cancelled"])
async def test_source_measurement_that_ends_unmeasured_leaves_a_session_that_resumes(
    coordinator, bench, resume, ending
):
    session = coordinator.session_dir
    bench.ending = ending
    await _measure(coordinator)
    assert _source(session) == _ORIGINAL
    coordinator.db.raw.close()

    resumed = await resume(session)
    assert resumed.fixes == ["cleared_stale_pending_integrate"]
    on_disk = SharedState.load_or_init(session)
    assert on_disk.pending_integrate == {}
    assert on_disk.stop_reason == ""
    assert _uncommitted(session) == ""


@pytest.mark.asyncio
async def test_source_measurement_killed_mid_bench_is_rolled_back_on_resume_and_measured_again(
    coordinator, bench, resume
):
    session = coordinator.session_dir
    bench.ending = "killed_mid_bench"
    await _measure(coordinator)
    bench.die_at_snapshot(coordinator)
    assert SharedState.load_or_init(session).pending_integrate["recovery"]["phase"] == "ready"
    assert _source(session) == _PATCHED

    resumed = await resume(session)
    assert resumed.fixes == ["rolled_back_pending_integrate"]
    assert await resumed.coord.tasks.find(bench.task_ids[0], kind="integrate_patch") is None
    on_disk = SharedState.load_or_init(session)
    assert on_disk.pending_integrate == {}
    assert on_disk.stop_reason == ""
    assert _source(session) == _ORIGINAL
    assert _uncommitted(session) == ""

    bench.ending = "reverted"
    _register_executor(resumed.coord)
    await _measure(resumed.coord)
    assert bench.task_ids[1] == bench.task_ids[0]
    assert resumed.coord.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert resumed.coord.shared_state.pending_integrate == {}
    assert _source(session) == _ORIGINAL


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["budget_stopped", "reverted"])
async def test_source_measurement_whose_restore_failed_is_restored_on_resume(
    coordinator, bench, resume, monkeypatch, ending
):
    session = coordinator.session_dir
    bench.ending = ending
    with monkeypatch.context() as live:
        live.setattr(
            integrate_patch_module,
            "restore_pending_integrate",
            lambda _pending, *, keep=False: {"reversed": [], "artifacts_reverted": [], "failed": ["tree busy"]},
        )
        await _measure(coordinator)
    assert _source(session) == _PATCHED
    assert coordinator.shared_state.geak_result["revalidation_status"] == "failed"
    assert coordinator.shared_state.pending_integrate["recovery"]["phase"] == "ready"
    coordinator.db.raw.close()

    resumed = await resume(session)
    assert resumed.fixes == ["rolled_back_pending_integrate"]
    assert SharedState.load_or_init(session).pending_integrate == {}
    assert _source(session) == _ORIGINAL
    assert _uncommitted(session) == ""


@pytest.mark.asyncio
async def test_journal_of_an_integration_that_was_not_a_covered_step_is_still_held(coordinator, bench, resume):
    session = coordinator.session_dir
    bench.ending = "killed_mid_bench"
    await _measure(coordinator)
    bench.die_at_snapshot(coordinator)
    path = session / "state.json"
    on_disk = json.loads(path.read_text())
    on_disk["pending_integrate"]["covered_step"] = False
    path.write_text(json.dumps(on_disk))

    assert (await resume(session)).refusal == _REFUSAL
    assert SharedState.load_or_init(session).pending_integrate["recovery"]["phase"] == "ready"
    assert _source(session) == _PATCHED


@pytest.mark.asyncio
async def test_kept_candidate_killed_before_promotion_is_held_not_rolled_back(coordinator, bench, resume, monkeypatch):
    session = coordinator.session_dir
    bench.ending = "killed_after_keep"
    promote = coordinator.writeback.promote_to_shared_state

    async def snapshot_then_promote(kind, result, *, task):
        assert result["status"] == "kept"
        bench.snapshot()
        return await promote(kind, result, task=task)

    monkeypatch.setattr(coordinator.writeback, "promote_to_shared_state", snapshot_then_promote)
    await _measure(coordinator)
    bench.die_at_snapshot(coordinator)

    assert (await resume(session)).refusal == _REFUSAL
    assert SharedState.load_or_init(session).pending_integrate["recovery"]["phase"] == "accepted"
    assert _source(session) == _PATCHED
