# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grid's callers on a session that froze a base: their rounds run on inference_suite, played by ScriptedRounds.

The contract: a conc sweep arm boots one server on its first rung, measures that rung on it, reuses it for every rung
after, and every rung serves the same launch; the arm's server and an explore variant's server are reaped with
``stop`` on the directory their rounds shared it through, however the rounds ended; explore launches one server once
per round and starts no variant its session or phase budget cannot fit; a specialist rebench is one round that owns its
server on the port it reports.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.deadline import Deadline
from hyperloom.inference_optimizer.tests.test_conc_sweep import _make_state
from hyperloom.orchestrator.actions.executors import _server_lifecycle
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors import explore as explore_module
from hyperloom.orchestrator.actions.executors.explore import ExploreExecutor
from hyperloom.orchestrator.actions.executors.tests.test_grid_suite_round import _HOT, suite  # noqa: F401
from hyperloom.orchestrator.kernel import conc_sweep
from hyperloom.orchestrator.kernel.conc_sweep import run_conc_sweep
from hyperloom.orchestrator.rehearsal import READY, LaunchAttempt, LaunchScenario, ScriptedRounds, VirtualClock
from hyperloom.orchestrator.rehearsal import installed_clock
from hyperloom.orchestrator.specialists.rebench import RebenchBenchmark, RebenchServer, run_specialist_rebench
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite.contracts import LifecycleMode, WorkloadKind
from inference_suite.tests.test_compose import _launch

_PORT = 30123


@pytest.fixture
def eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every point lifecycle-eligible on ``_PORT``."""
    monkeypatch.setattr(
        _server_lifecycle,
        "resolve_lifecycle_params",
        lambda _path: {"eligible": True, "framework": "sglang", "port": _PORT, "reason": ""},
    )
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._ray_serving.maybe_serving_lease", lambda **_k: None)


def _rounds(monkeypatch: pytest.MonkeyPatch, count: int) -> ScriptedRounds:
    attempts = tuple(LaunchAttempt(outcome=READY, metrics=_HOT) for _ in range(count))
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    return rounds


def _session(suite: SimpleNamespace) -> Path:
    return suite.point.parent / "session"


def _sweep(monkeypatch, suite: SimpleNamespace, concs: list[int]) -> list:
    """Sweep both arms over ``concs`` and return the config of every round played."""
    rounds = _rounds(monkeypatch, 2 * len(concs))
    state = _make_state(baseline_config_path=str(suite.point), tp=2, current_best={"extra_envs": {"SGLANG_X": "1"}})
    monkeypatch.setattr(
        conc_sweep,
        "prepare_grid_benchmark_config",
        lambda src, out_dir, **_kw: Path(shutil.copy(src, Path(out_dir) / "conc_sweep_base.with_envs.yaml")),
    )
    with installed_clock(rounds.clock):
        payload = asyncio.run(run_conc_sweep(state, _session(suite), concs=concs, total_budget_sec=9000))
    assert payload["summary"]["successful_pairs"] == len(concs), payload
    return [call.config for call in rounds.calls]


def test_a_conc_sweep_arm_boots_on_its_first_rung_reuses_one_launch_and_is_reaped(
    monkeypatch,
    suite,
    eligible,
):
    configs = _sweep(monkeypatch, suite, [8, 16])

    arms: dict[str, list] = {}
    for config in configs:
        arms.setdefault(config.lifecycle.pid_dir, []).append(config)
    assert len(arms) == 2
    for pid_dir, (boot, reuse) in arms.items():
        assert Path(pid_dir).name.startswith("server_")
        assert (boot.lifecycle.mode, reuse.lifecycle.mode) == (LifecycleMode.BOOT, LifecycleMode.REUSE)
        assert boot.workload is WorkloadKind.FIXED and boot.fixed.concurrency == 8
        assert reuse.fixed.concurrency == 16
        assert boot.server == reuse.server
        assert Path(pid_dir) in suite.stopped


def test_a_conc_sweep_arm_whose_only_rung_booted_its_server_is_reaped_when_the_arm_ends(
    monkeypatch,
    suite,
    eligible,
):
    configs = _sweep(monkeypatch, suite, [8])

    assert [config.lifecycle.mode for config in configs] == [LifecycleMode.BOOT] * 2
    assert {Path(config.lifecycle.pid_dir) for config in configs} <= set(suite.stopped)


@pytest.mark.parametrize("outcome", ["measured", "raised"])
def test_an_explore_variants_server_is_reaped_through_its_slot(monkeypatch, suite, eligible, tmp_path, outcome):
    async def grid(**_kw):
        if outcome == "raised":
            raise RuntimeError("the grid broke")
        return []

    monkeypatch.setattr(explore_module, "run_grid", grid)
    state = SharedState()
    state.baseline_tput = 800.0
    task = SimpleNamespace(
        task_id="explore-1",
        params={
            "config_path": str(suite.point),
            "output_dir": str(tmp_path / "explore"),
            "base_tput": 800.0,
            "keep_threshold_pct": 1.0,
            "grid": [{"name": "v0", "extra_args": "--max-running-requests 64", "extra_envs": {}}],
        },
    )

    run = ExploreExecutor(session_dir=_session(suite))(SimpleNamespace(task=task, extra={"shared_state": state}))
    if outcome == "raised":
        with pytest.raises(RuntimeError, match="the grid broke"):
            asyncio.run(run)
    else:
        asyncio.run(run)

    slot = tmp_path / "explore" / "v00_v0"
    assert suite.stopped == [slot]


def test_a_specialist_rebench_runs_one_round_that_owns_its_server_on_the_port_it_reports(monkeypatch, suite, tmp_path):
    rounds = _rounds(monkeypatch, 1)

    result = asyncio.run(
        run_specialist_rebench(
            output_dir=tmp_path / "rebench",
            benchmark=RebenchBenchmark(config_path=suite.point),
            server=RebenchServer(extra_envs={"RUN_EVAL": "false"}, port=_PORT),
        )
    )

    assert result["ok"] is True, result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    (config,) = (call.config for call in rounds.calls)
    assert config.lifecycle.mode is LifecycleMode.OWN
    assert config.server.port == result["port"] == _PORT


def _explore(
    monkeypatch,
    suite: SimpleNamespace,
    tmp_path: Path,
    rows: list[dict],
    state: SharedState,
    *,
    rounds: ScriptedRounds | None = None,
    phase_deadline: Deadline | None = None,
    **params,
):
    """Run explore over ``rows`` on played rounds, which play no eval; return its result and the rounds launched."""
    rounds = rounds or _rounds(monkeypatch, 2 * len(rows))
    task = SimpleNamespace(
        task_id="explore-1",
        params={
            "config_path": str(suite.point),
            "output_dir": str(tmp_path / "explore"),
            "base_tput": 800.0,
            "keep_threshold_pct": 1.0,
            "baseline_double_run": False,
            "grid": [{"extra_envs": {"RUN_EVAL": "false"}, **row} for row in rows],
            **params,
        },
    )
    executor = ExploreExecutor(session_dir=_session(suite))
    with installed_clock(rounds.clock):
        extra = {"shared_state": state, "phase_deadline": phase_deadline}
        result = asyncio.run(executor(SimpleNamespace(task=task, extra=extra)))
    return result, rounds.calls


def test_explore_benches_two_rows_that_launch_one_server_once(monkeypatch, suite, tmp_path):
    """The second row spells the first row's change twice, and the server applies the last spelling."""
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = _HOT["output_throughput"]
    rows = [
        {"name": "v_a", "extra_args": "--max-running-requests 64"},
        {"name": "v_b", "extra_args": "--max-running-requests 32 --max-running-requests 64"},
    ]

    result, calls = _explore(monkeypatch, suite, tmp_path, rows, state, base_tput=_HOT["output_throughput"])

    assert len(calls) == 1, result
    assert [(row["name"], row["reason"]) for row in result["skipped_dup"]] == [("v_b", "round_dup")]


@pytest.mark.parametrize(
    ("session_minutes", "phase_sec", "error_class"),
    [(3.0, None, "session_time_exhausted"), (600.0, 10.0, "phase_time_exhausted")],
    ids=["session", "phase"],
)
def test_explore_starts_no_variant_the_budget_cannot_fit(
    monkeypatch, suite, tmp_path, session_minutes, phase_sec, error_class
):
    """A variant expected to take 600s starts neither with about 60s of the session left nor with 10s of the phase."""
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    state.max_minutes = session_minutes
    rows = [{"name": "v_too_long", "extra_args": "--max-running-requests 64"}]
    rounds = _rounds(monkeypatch, 1)
    phase = None if phase_sec is None else Deadline.after(phase_sec, now=rounds.clock.monotonic())

    result, calls = _explore(
        monkeypatch, suite, tmp_path, rows, state, rounds=rounds, phase_deadline=phase, baseline_runtime_sec=600.0
    )

    assert calls == []
    assert result["error_class"] == error_class
    assert "1 variant(s) went unmeasured" in result["error"]
    assert result["explore_search_update"]["tested"] == {}
