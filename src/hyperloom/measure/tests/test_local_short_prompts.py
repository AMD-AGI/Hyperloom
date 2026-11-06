# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend's short-prompt check: the test's own server answers after its measured rounds, before its teardown,
and the check boots no server of its own.

``run_grid`` and the server lifecycle are stubbed as the other local-backend tests stub them; the server the check talks
to is a real HTTP server that the test's held round brings up on the port its lifecycle names. Nothing about the test's
rounds, its output directory or its outcome may depend on the check.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.phases import SHORT_PROMPTS
from hyperloom.measure.ports import DEV_SANDBOX_PORTS
from hyperloom.measure.socket import RoundPlan
from hyperloom.measure.long_context.held import FLUSHED
from hyperloom.measure.short_prompts.outputs import FAILED, OK, read_outputs
from hyperloom.measure.short_prompts.plan import SHORT_PROMPTS_DIR
from hyperloom.measure.short_prompts.tests.support import (
    FakeServer,
    free_port,
    staged_plan,
)
from hyperloom.measure.tests.test_local_backend import _agentx_point, _call, _Grid, _point


SKIPPED = "short_prompt_check skipped: test failed"
NOT_HELD = "the test's server was not held up for the check"


class _Servers:
    """The servers rounds bring up, each a real HTTP server; ``started`` names the rounds that booted one.

    ``port`` is the lifecycle's own pick (a free port by default). The backend may not keep it, as it does not keep one a
    dev sandbox serves on, so a server listens where the round's ``server_lifecycle`` says, as a real one would."""

    def __init__(
        self, monkeypatch, *, eligible: bool = True, framework: str = "sglang", port: int | None = None
    ) -> None:
        self.port = free_port() if port is None else port
        self.live: FakeServer | None = None
        self.started: list[str] = []
        self.teardowns = 0
        self._lock = threading.Lock()
        monkeypatch.setattr(
            local,
            "resolve_lifecycle_params",
            lambda path: {"eligible": eligible, "port": self.port, "framework": framework, "reason": "not eligible"},
        )
        monkeypatch.setattr(local, "teardown_lifecycle_server", lambda **kw: self.stop())

        self.flushes = 0

    def start(self, name: str, port: int) -> None:
        self.live = FakeServer(port=port).__enter__()
        self.started.append(name)

    def stop(self) -> None:
        with self._lock:
            self.teardowns += 1
            if self.live is not None:
                self.flushes += self.live.flushes
                self.live.__exit__(None, None, None)
                self.live = None


class _CheckGrid(_Grid):
    """``run_grid`` that brings a server up for a round that holds one when none is up; a round with a scripted answer
    brings none up."""

    def __init__(self, tmp_path: Path, servers: _Servers) -> None:
        super().__init__(tmp_path)
        self.servers = servers

    async def __call__(self, **kwargs: Any):
        name = kwargs["grid"][0].name
        lifecycle = kwargs["launch"].server_lifecycle
        if lifecycle and self.servers.live is None and name not in self.answers:
            self.servers.start(name, lifecycle["port"])
        return await super().__call__(**kwargs)


@pytest.fixture(autouse=True)
def magpie_backend(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")


@pytest.fixture
def cache(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "runner-cache"
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(root))
    return root


def _grid(monkeypatch, tmp_path: Path, **servers: Any) -> _CheckGrid:
    grid = _CheckGrid(tmp_path, _Servers(monkeypatch, **servers))
    monkeypatch.setattr(local, "run_grid", grid)
    return grid


def _run(tmp_path: Path, cache: Path, **plan: Any):
    call = _call(tmp_path, short_prompts=staged_plan(cache, **plan), result_dir=str(tmp_path / "out" / "results"))
    return LocalMeasurementBackend().run_test(call)


def _names(grid: _CheckGrid) -> list[str]:
    return [kwargs["grid"][0].name for kwargs in grid.calls]


def test_a_planned_test_answers_on_its_own_server_after_its_measured_round_and_boots_no_other(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    assert _names(grid) == ["test"], "no second boot"
    assert grid.calls[0]["launch"].server_lifecycle["cleanup"] is False, "Magpie leaves the server up for the check"
    assert (grid.servers.started, grid.servers.teardowns) == (["test"], 1), "one server, torn down after the answers"
    found = read_outputs(tmp_path / "out")
    assert [(e.index, e.status, e.boot_seconds) for e in found.processes] == [(0, OK, 0.0)]
    assert len(found.prompts) == 64
    assert grid.servers.flushes == 1 and FLUSHED in outcome.stderr, "the prefix cache is flushed before the check"
    assert (outcome.returncode, outcome.failure_class) == (0, None)


def test_a_lifecycle_port_a_dev_sandbox_serves_on_is_replaced_and_the_check_asks_the_server_on_the_replacement(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path, port=DEV_SANDBOX_PORTS.start)

    outcome = _run(tmp_path, cache)

    ports = [c["launch"].server_lifecycle["port"] for c in grid.calls]
    assert len(ports) == 1 and not set(ports) & set(DEV_SANDBOX_PORTS), "the server is given a replacement"
    assert [e.status for e in read_outputs(tmp_path / "out").processes] == [OK]
    assert grid.servers.flushes == 1 and outcome.returncode == 0


def test_the_check_is_apart_from_the_test_its_rounds_its_files_and_its_result_directory_are_the_tests_own(
    monkeypatch, tmp_path, cache
):
    plain = _grid(monkeypatch, tmp_path / "plain")
    (tmp_path / "plain").mkdir(exist_ok=True)
    baseline = LocalMeasurementBackend().run_test(_call(tmp_path / "plain"))
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [(r.phase, r.returncode) for r in baseline.rounds]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["server_arm", SHORT_PROMPTS_DIR]
    assert _names(grid) == ["test"] and grid.calls[0]["output_root"] == tmp_path / "out"
    assert grid.calls[0]["launch"].result_dir == str(tmp_path / "out" / "results"), (
        "the test's round keeps its directory"
    )
    assert plain.calls and len(plain.calls) == 1


def test_the_check_adds_a_phase_for_its_questions_and_the_phases_still_tile_the_test(monkeypatch, tmp_path, cache):
    _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    names = [p.name for p in outcome.phases]
    assert names.count(SHORT_PROMPTS) == 1
    assert names.index(SHORT_PROMPTS) < names.index("teardown"), "the own server's answers come before its teardown"
    for earlier, later in zip(outcome.phases, outcome.phases[1:]):
        assert earlier.started_at + earlier.seconds == pytest.approx(later.started_at, abs=1e-6), "no time is lost"


def test_a_test_whose_round_failed_runs_no_check(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    grid.answers["test"] = grid.result("test", status="failed", returncode=1, error="server died")

    outcome = _run(tmp_path, cache)

    assert _names(grid) == ["test"] and outcome.returncode == 1
    assert not (tmp_path / "out" / SHORT_PROMPTS_DIR).exists()
    assert SKIPPED in outcome.stderr, "the test says why the check did not run"
    assert "prefix cache" not in outcome.stderr, "its server was not asked to flush either"
    assert SHORT_PROMPTS not in [p.name for p in outcome.phases], "nor asked the prompts"


def test_a_graded_test_answers_after_its_measured_round_on_the_server_its_warmup_booted(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = LocalMeasurementBackend().run_test(
        _call(
            tmp_path,
            point,
            accuracy=True,
            short_prompts=staged_plan(cache),
            result_dir=str(tmp_path / "res"),
            rounds=RoundPlan(warmup_rounds=1),
        )
    )

    assert _names(grid) == ["warmup", "test"]
    assert [c["launch"].server_lifecycle["cleanup"] for c in grid.calls] == [False, False]
    assert (
        grid.calls[1]["recipe"].base.extra_envs == {"RUN_EVAL": "false"}
        and grid.calls[1]["launch"].server_already_ready is True
    )
    assert grid.servers.started == ["warmup"] and [e.status for e in read_outputs(tmp_path / "out").processes] == [OK]
    assert [r.phase for r in outcome.rounds] == ["warmup", "accuracy", "measure"]


def test_an_agentx_test_holds_its_server_for_the_check_and_reports_no_accuracy_round(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    point = _agentx_point(tmp_path, monkeypatch)

    outcome = LocalMeasurementBackend().run_test(
        _call(tmp_path, point, short_prompts=staged_plan(cache), rounds=RoundPlan(warmup_rounds=1))
    )

    assert _names(grid) == ["warmup", "test"] and grid.calls[1]["launch"].server_already_ready is True
    assert grid.calls[1]["recipe"].base.extra_envs is None, "an AgentX round carries nothing of an evaluation"
    assert [r.phase for r in outcome.rounds] == ["warmup", "measure"]
    assert [e.status for e in read_outputs(tmp_path / "out").processes] == [OK]


def test_a_test_whose_evaluation_failed_runs_no_check_although_its_round_kept_its_status(monkeypatch, tmp_path, cache):
    """A benchmark that served its whole protocol is ``succeeded`` when only its accuracy evaluation fails: the evaluation's
    return code is what says the test failed, and a test that failed spends no time on the check."""
    grid = _grid(monkeypatch, tmp_path)
    result = grid.result("test", returncode=0)
    (Path(result.workspace) / "eval_returncode").write_text("3\n", encoding="utf-8")
    grid.answers["test"] = result
    call = _call(
        tmp_path,
        _point(tmp_path, "acc", run_eval="true"),
        accuracy=True,
        kind="accuracy",
        short_prompts=staged_plan(cache),
        result_dir=str(tmp_path / "out" / "results"),
        rounds=RoundPlan(warmup_rounds=0),
    )

    outcome = LocalMeasurementBackend().run_test(call)

    assert _names(grid) == ["test"], "no server of the check was started"
    assert outcome.returncode == 3 and SKIPPED in outcome.stderr
    assert not (tmp_path / "out" / SHORT_PROMPTS_DIR).exists()


def test_a_test_that_ran_clean_records_no_skip(monkeypatch, tmp_path, cache):
    _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    assert SKIPPED not in outcome.stderr


def test_a_framework_that_cannot_hold_a_server_up_fails_the_check_with_that_reason_and_not_the_test(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path, eligible=False)

    outcome = _run(tmp_path, cache)

    found = read_outputs(tmp_path / "out")
    assert [(e.index, e.status) for e in found.processes] == [(0, FAILED)] and NOT_HELD in found.processes[0].reason
    assert _names(grid) == ["test"] and outcome.returncode == 0


def test_a_server_whose_framework_has_no_flush_keeps_its_cache_and_the_test_says_so(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path, framework="vllm")

    outcome = _run(tmp_path, cache)

    assert (
        grid.servers.flushes == 0 and "not flushed before the accuracy checks (vllm exposes no flush)" in outcome.stderr
    )
    assert [e.status for e in read_outputs(tmp_path / "out").processes] == [OK]


def test_a_sweep_answers_on_its_arms_server_after_its_last_level(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    points = (_point(tmp_path, "c4", conc=4), _point(tmp_path, "c2", conc=2))

    LocalMeasurementBackend().run_test(_call(tmp_path, *points, prep="conc_sweep", short_prompts=staged_plan(cache)))

    assert _names(grid) == ["c4", "c4", "c2"], "no boot for the check"
    assert grid.calls[2]["launch"].server_lifecycle["cleanup"] is False, (
        "the last level leaves the server up for the check"
    )
    assert grid.servers.started == ["c4"] and list(read_outputs(tmp_path / "out").completed) == [0]


def test_a_call_naming_a_prompt_set_that_is_not_staged_is_refused_before_any_server_starts(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)
    plan = staged_plan(tmp_path / "elsewhere")

    with pytest.raises(TestCallRefused, match=f"short-prompt set {plan.prompt_set} cannot be used: .*not staged"):
        LocalMeasurementBackend().run_test(_call(tmp_path, short_prompts=plan))
    assert grid.calls == []


def test_a_prompt_set_whose_file_differs_from_the_plans_hash_is_refused_naming_both_hashes(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)
    plan = staged_plan(cache)

    with pytest.raises(TestCallRefused, match="the sheet names " + "1" * 64):
        LocalMeasurementBackend().run_test(_call(tmp_path, short_prompts=type(plan)(plan.prompt_set, "1" * 64)))
    assert grid.calls == []


@pytest.mark.parametrize("kind", ["profile", "repro"])
def test_a_profile_or_repro_test_does_not_run_the_check(tmp_path, cache, kind):
    call = _call(tmp_path, kind=kind, short_prompts=staged_plan(cache))

    with pytest.raises(TestCallRefused, match=f"a {kind} test does not run the short-prompt consistency check"):
        local._validate_checks(call)


def test_a_call_with_no_plan_is_run_exactly_as_before(monkeypatch, tmp_path):
    grid = _grid(monkeypatch, tmp_path)

    LocalMeasurementBackend().run_test(_call(tmp_path))

    assert _names(grid) == ["test"] and grid.calls[0]["launch"].server_lifecycle is None
    assert not (tmp_path / "out" / SHORT_PROMPTS_DIR).exists()
