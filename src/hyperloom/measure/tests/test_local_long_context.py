# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend's long-context check: a fresh server after the test's own rounds, apart from them.

``run_grid`` and the server lifecycle are stubbed as the other local-backend tests stub them; the server the check talks to
is a real HTTP server on a free local port, one for each boot-only round the check asks for. Nothing about the test's rounds,
its output directory or its outcome may depend on the check.
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
from hyperloom.measure.phases import LONG_CONTEXT
from hyperloom.measure.tests.test_local_backend import BOOT_ENVS, _call, _point
from hyperloom.measure.tests.test_local_short_prompts import _CheckGrid as _ShortGrid
from hyperloom.measure.tests.test_local_short_prompts import _Servers as _ShortServers
from hyperloom.measure.tests.test_local_short_prompts import cache, magpie_backend  # noqa: F401 - fixtures

SKIPPED = "long_context_check skipped: test failed"
CONTEXT_FLAG_ENV = framework_registry.FRAMEWORKS["sglang"].extra_args_env


class _Servers(_ShortServers):
    """The server of the check: one real HTTP server for each boot-only round (the lifecycle stubs are the short-prompt
    tests')."""

    def start(self, index: int) -> None:
        self.live = FakeServer(port=self.port).__enter__()
        self.started.append(index)


class _CheckGrid(_ShortGrid):
    """``run_grid`` that brings the check's server up when it is asked for a boot-only round called ``long<index>``."""

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


def test_a_planned_test_runs_its_rounds_then_one_fresh_server_and_leaves_the_outputs(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    assert [kwargs["grid"][0].name for kwargs in grid.calls] == ["test", "long0"]
    boot = grid.calls[1]
    assert boot["lifecycle_boot_only"] is True and boot["base_extra_envs"] == BOOT_ENVS
    assert boot["server_lifecycle"]["cleanup"] is False and boot["warmup_before_measure"] is False
    assert (grid.servers.started, grid.servers.teardowns) == ([0], 1)
    found = read_outputs(tmp_path / "out")
    assert [e.status for e in found.prompts] == [PROMPT_OK] * 64 and found.served_context == 4096
    assert (outcome.returncode, outcome.failure_class) == (0, None)


def test_the_check_is_apart_from_the_test_its_files_and_its_result_directory_are_the_tests_own(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)

    _run(tmp_path, cache)

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [LONG_CONTEXT_DIR], "only the outputs file is added"
    assert grid.calls[1]["result_dir"] is None and grid.calls[1]["output_root"] != tmp_path / "out"
    assert grid.calls[0]["result_dir"] == str(tmp_path / "out" / "results")


def test_the_check_adds_a_phase_for_its_questions_and_the_phases_still_tile_the_test(monkeypatch, tmp_path, cache):
    _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache)

    names = [p.name for p in outcome.phases]
    assert names.count(LONG_CONTEXT) == 1 and names.index(LONG_CONTEXT) > names.index("kernel_tables")
    for earlier, later in zip(outcome.phases, outcome.phases[1:]):
        assert earlier.started_at + earlier.seconds == pytest.approx(later.started_at, abs=1e-6), "no time is lost"


def test_a_test_whose_round_failed_runs_no_check(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    grid.answers["test"] = grid.result(
        "test", status="failed", returncode=2, error="the server died", error_class="crash"
    )

    outcome = _run(tmp_path, cache)

    started = [kwargs["grid"][0].name for kwargs in grid.calls]
    assert started == ["test"], "no boot-only round of the check"
    assert (outcome.returncode, SKIPPED in outcome.stderr) == (2, True)
    assert not (tmp_path / "out" / LONG_CONTEXT_DIR).exists()


def test_a_launch_that_serves_too_little_context_starts_no_server_and_the_test_still_succeeds(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path)

    outcome = _run(tmp_path, cache, served=2048)

    assert [kwargs["grid"][0].name for kwargs in grid.calls] == ["test"] and grid.servers.started == []
    found = read_outputs(tmp_path / "out")
    assert found.stopped.startswith("served context too short: the launch serves 2048 tokens, the check needs 2176")
    assert {e.status for e in found.prompts} == {PROMPT_NOT_RUN} and outcome.returncode == 0


def test_a_context_flag_in_the_server_arguments_wins_over_the_points_max_model_len(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    point = _point(tmp_path, "flagged", MAX_MODEL_LEN=100000, **{CONTEXT_FLAG_ENV: "--context-length 1500"})
    call = _call(tmp_path, point, long_context=staged_plan(cache), result_dir=str(tmp_path / "out" / "results"))

    LocalMeasurementBackend().run_test(call)

    assert grid.servers.started == []
    assert "the launch serves 1500 tokens" in read_outputs(tmp_path / "out").stopped


def test_a_server_that_does_not_boot_is_the_checks_incomplete_and_not_the_tests_failure(monkeypatch, tmp_path, cache):
    grid = _grid(monkeypatch, tmp_path)
    grid.answers["long0"] = grid.result("long0", status="failed", returncode=1, error="oom", error_class="oom")

    outcome = _run(tmp_path, cache)

    found = read_outputs(tmp_path / "out")
    assert "the server did not come up (oom)" in found.stopped and grid.servers.teardowns == 1
    assert (outcome.returncode, outcome.failure_class) == (0, None)


def test_a_framework_that_cannot_hold_a_server_up_leaves_the_prompts_unrun_with_that_reason(
    monkeypatch, tmp_path, cache
):
    grid = _grid(monkeypatch, tmp_path, eligible=False)

    outcome = _run(tmp_path, cache)

    assert "cannot be held up for the check (not eligible)" in read_outputs(tmp_path / "out").stopped
    assert len(grid.calls) == 1 and outcome.returncode == 0


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
        local._validate_long_context(call)
