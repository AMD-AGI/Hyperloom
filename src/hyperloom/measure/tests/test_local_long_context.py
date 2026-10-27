# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend's long-context check: on the test's own server after its measured rounds, before its teardown, with
no second boot.

``run_grid`` and the server lifecycle are stubbed as the other local-backend tests stub them; the server the check talks to
is a real HTTP server on a free local port that the test's held round brings up. Nothing about the test's rounds, its
output directory or its outcome may depend on the check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.long_context import PROMPT_NOT_RUN, PROMPT_OK
from hyperloom.inference_optimizer import framework_registry
from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.long_context.outputs import read_outputs
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR
from hyperloom.measure.long_context.tests.support import FakeServer, staged_plan
from hyperloom.measure.phases import LONG_CONTEXT, SHORT_PROMPTS
from hyperloom.measure.short_prompts.tests.support import staged_plan as short_plan
from hyperloom.measure.tests.test_local_backend import _call, _point
from hyperloom.measure.tests.test_local_short_prompts import _CheckGrid as _ShortGrid
from hyperloom.measure.tests.test_local_short_prompts import _Servers as _ShortServers
from hyperloom.measure.tests.test_local_short_prompts import cache, magpie_backend  # noqa: F401 - fixtures

SKIPPED = "long_context_check skipped: test failed"
CONTEXT_FLAG_ENV = framework_registry.FRAMEWORKS["sglang"].extra_args_env


class _Servers(_ShortServers):
    """The test's server: a real HTTP server the held round brings up (the lifecycle stubs are the short-prompt tests')."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.asked: list[int] = []  # the requests each server had when it was stopped

    def start(self, index: int) -> None:
        self.live = FakeServer(port=self.port).__enter__()
        self.started.append(index)

    def stop(self) -> None:
        if self.live is not None:
            self.asked.append(len(self.live.requests))
        super().stop()


class _CheckGrid(_ShortGrid):
    """``run_grid`` that brings the test's server up for the round that holds it."""

    prefix = "long"


def _grid(monkeypatch, tmp_path: Path, **servers: Any) -> _CheckGrid:
    grid = _CheckGrid(tmp_path, _Servers(monkeypatch, **servers))
    monkeypatch.setattr(local, "run_grid", grid)
    return grid


def _run(tmp_path: Path, cache: Path, *, served: int = 4096, **call: Any):
    point = _point(tmp_path, "served", MAX_MODEL_LEN=served)
    plan = call.pop("long_context", None) or staged_plan(cache)
    call = {"long_context": plan, "result_dir": str(tmp_path / "out" / "results"), **call}
    return LocalMeasurementBackend().run_test(_call(tmp_path, point, **call))


def test_a_planned_test_asks_its_own_server_after_its_measured_round_and_boots_no_other(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    assert [kwargs["grid"][0].name for kwargs in grid.calls] == ["test"], "no second boot"
    assert grid.calls[0]["server_lifecycle"]["cleanup"] is False, "Magpie leaves the server up for the check"
    assert (grid.servers.started, grid.servers.teardowns) == ([0], 1), "one server, torn down after the check"
    found = read_outputs(tmp_path / "out")
    assert [e.status for e in found.prompts] == [PROMPT_OK] * 64 and found.served_context == 4096
    assert found.boot_seconds == pytest.approx(0.0, abs=0.5), "no server was booted for the check"
    assert grid.servers.flushes == 1, "the prefix cache was flushed before the check"
    assert (outcome.returncode, outcome.failure_class) == (0, None)


def test_the_check_is_apart_from_the_test_its_files_and_its_result_directory_are_the_tests_own(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)

    _run(tmp_path, cache)

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [LONG_CONTEXT_DIR, "server_arm"]
    assert grid.calls[0]["result_dir"] == str(tmp_path / "out" / "results")


def test_the_check_adds_a_phase_for_its_questions_and_the_phases_still_tile_the_test(monkeypatch, tmp_path, cache):
    _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    names = [p.name for p in outcome.phases]
    assert names.count(LONG_CONTEXT) == 1 and names.index("kernel_tables") < names.index(LONG_CONTEXT)
    assert names.index(LONG_CONTEXT) < names.index("teardown"), "the check runs before the server's teardown"
    for earlier, later in zip(outcome.phases, outcome.phases[1:]):
        assert earlier.started_at + earlier.seconds == pytest.approx(later.started_at, abs=1e-6), "no time is lost"


def test_both_checks_share_the_held_server_and_one_flush(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache, short_prompts=short_plan(cache))

    assert [kwargs["grid"][0].name for kwargs in grid.calls] == ["test"] and grid.servers.flushes == 1
    names = [p.name for p in outcome.phases]
    assert names.index(SHORT_PROMPTS) < names.index(LONG_CONTEXT) < names.index("teardown")
    assert [e.status for e in read_outputs(tmp_path / "out").prompts] == [PROMPT_OK] * 64


def test_a_test_whose_round_failed_runs_no_check(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    grid.answers["test"] = grid.result(
        "test", status="failed", returncode=2, error="the server died", error_class="crash"
    )

    outcome = _run(tmp_path, cache)

    assert [kwargs["grid"][0].name for kwargs in grid.calls] == ["test"]
    assert (outcome.returncode, SKIPPED in outcome.stderr) == (2, True)
    assert not (tmp_path / "out" / LONG_CONTEXT_DIR).exists()


def test_a_launch_that_serves_too_little_context_sends_no_prompt_and_the_test_still_succeeds(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache, served=2048)

    assert [kwargs["grid"][0].name for kwargs in grid.calls] == ["test"] and grid.servers.asked == [0]
    found = read_outputs(tmp_path / "out")
    assert found.stopped.startswith("served context too short: the launch serves 2048 tokens, the check needs 2176")
    assert {e.status for e in found.prompts} == {PROMPT_NOT_RUN} and outcome.returncode == 0


def test_a_context_flag_in_the_server_arguments_wins_over_the_points_max_model_len(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    point = _point(tmp_path, "flagged", MAX_MODEL_LEN=100000, **{CONTEXT_FLAG_ENV: "--context-length 1500"})
    call = _call(tmp_path, point, long_context=staged_plan(cache), result_dir=str(tmp_path / "out" / "results"))

    LocalMeasurementBackend().run_test(call)

    assert "the launch serves 1500 tokens" in read_outputs(tmp_path / "out").stopped


def test_a_framework_that_cannot_hold_a_server_up_leaves_the_prompts_unrun_with_that_reason(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path, eligible=False)

    outcome = _run(tmp_path, cache)

    found = read_outputs(tmp_path / "out")
    assert "the test's server was not held up for the check" in found.stopped
    assert {e.status for e in found.prompts} == {PROMPT_NOT_RUN}
    assert len(grid.calls) == 1 and grid.servers.started == [] and outcome.returncode == 0


def test_a_call_naming_a_set_that_is_not_staged_is_refused_before_any_server_starts(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    plan = staged_plan(tmp_path / "elsewhere")

    with pytest.raises(TestCallRefused, match=f"long-context set {plan.prompt_set} cannot be used: .*not staged"):
        _run(tmp_path, cache, long_context=plan)

    assert grid.calls == []


def test_a_set_whose_file_differs_from_the_plans_hash_is_refused_naming_both_hashes(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    plan = staged_plan(cache)

    with pytest.raises(TestCallRefused, match="the sheet names " + "1" * 64):
        _run(tmp_path, cache, long_context=type(plan)(plan.prompt_set, "1" * 64, plan.length))

    assert grid.calls == []


@pytest.mark.parametrize("kind", ["profile", "repro"])
def test_a_profile_or_repro_test_does_not_run_the_check(tmp_path, cache, kind):
    call = _call(tmp_path, kind=kind, long_context=staged_plan(cache))

    with pytest.raises(TestCallRefused, match=f"a {kind} test does not run the long-context check"):
        local._validate_checks(call)
