# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round with a Ray serving lease runs in the lease's actor on the GPUs Ray placed it on.

The contract: the actor runs ``python -m inference_suite run`` on a config pinned to its placed GPUs, the host reads
the report the round wrote, and a leased round that left no report is refused under the class of why it never ran.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _ray_serving, _suite_config
from hyperloom.orchestrator.actions.executors._grid_runner import GridLaunch
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._ray_backend import SessionKillRound, SuiteLeasedRound
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
)
from hyperloom.orchestrator.actions.executors._suite_config import (
    LEASED_CONFIG_SUFFIX,
    RAY_ROUND_UNPLACED_CLASS,
    SuiteRound,
    SuiteRoundRefused,
    round_lifecycle,
    run_suite_round,
    suite_route,
)
from hyperloom.orchestrator.actions.executors.tests.test_baseline_suite_round import _COLD, _HOT, suite  # noqa: F401
from hyperloom.orchestrator.actions.executors.tests.test_baseline_suite_round import _play as baseline_play
from hyperloom.orchestrator.actions.executors.tests.test_grid_suite_round import _play as grid_play
from hyperloom.orchestrator.actions.executors.tests.test_grid_suite_round import suite as grid_suite  # noqa: F401
from hyperloom.orchestrator.actions.executors.tests.test_suite_round import _rounds, _write_point, session  # noqa: F401
from hyperloom.orchestrator.rehearsal import READY, LaunchAttempt
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS
from inference_suite import EXIT_REFUSED
from inference_suite.contracts import FailedStep, LifecycleMode, RunStatus, SuiteConfig
from inference_suite.tests.test_config_contract import _config


class _Lease:
    """Plays the actor: runs the leased round's config with ``ScriptedRounds`` into its run_dir, or exits as told."""

    def __init__(self, rounds=None, exit_code: int = 0, raised: Exception | None = None) -> None:
        self.rounds = rounds
        self.exit_code = exit_code
        self.raised = raised
        self.leased: list[SuiteLeasedRound] = []

    def run_session_kill(self, leased: SuiteLeasedRound) -> tuple[int, str, str]:
        self.leased.append(leased)
        if self.raised is not None:
            raise self.raised
        if self.rounds is None:
            return self.exit_code, "", "the actor said why"
        config = SuiteConfig.from_document(yaml.safe_load(Path(leased.config_path).read_text()))
        self.rounds.run_round(replace(config, paths=replace(config.paths, run_dir=leased.run_dir)))
        return 0, "", ""


def _request(point: Path, lease: _Lease) -> SuiteRound:
    return SuiteRound(point, point.parent, round_lifecycle(point.parent, LifecycleMode.OWN), serving_lease=lease)


def test_a_leased_round_runs_in_the_lease_and_its_report_is_read_back(session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_suite_config, "run_round", lambda *args, **kwargs: pytest.fail("ran outside the lease"))
    lease = _Lease(_rounds())
    point = _write_point(tmp_path / "slot" / "point.yaml")

    result = run_suite_round(suite_route(session, point), _request(point, lease), on_poll=lambda: None)

    assert result.report.run_status is RunStatus.COMPLETED
    (leased,) = lease.leased
    assert leased.run_dir == str(result.layout.root)
    assert leased.config_path == f"{result.layout.root}{LEASED_CONFIG_SUFFIX}"
    assert leased.cmd[1:] == [
        "-m",
        "inference_suite",
        "run",
        "--config",
        leased.config_path,
        "--run-dir",
        leased.run_dir,
    ]


@pytest.mark.parametrize(
    ("lease", "error_class"),
    [
        (_Lease(exit_code=EXIT_REFUSED), failed_step_error_class(FailedStep.CONFIG_INVALID)),
        (_Lease(exit_code=ORCHESTRATOR_CANCELLED_RETURNCODE), ORCHESTRATOR_CANCELLED_CLASS),
        (_Lease(exit_code=SESSION_TIME_EXHAUSTED_RETURNCODE), SESSION_TIME_EXHAUSTED_CLASS),
        (_Lease(exit_code=1), RAY_ROUND_UNPLACED_CLASS),
        (_Lease(raised=subprocess.TimeoutExpired(["round"], 60)), RAY_ROUND_UNPLACED_CLASS),
    ],
)
def test_a_leased_round_that_left_no_report_is_refused_under_why_it_never_ran(session, tmp_path, lease, error_class):
    point = _write_point(tmp_path / "slot" / "point.yaml")

    with pytest.raises(SuiteRoundRefused) as refused:
        run_suite_round(suite_route(session, point), _request(point, lease), on_poll=lambda: None)

    assert refused.value.error_class == error_class


def test_the_leased_round_is_reaped_only_past_its_own_session_deadline(session, tmp_path) -> None:
    lease = _Lease(exit_code=SESSION_TIME_EXHAUSTED_RETURNCODE)
    point = _write_point(tmp_path / "slot" / "point.yaml")
    request = replace(_request(point, lease), deadline_unix=int(time.time()) + 600)

    with pytest.raises(SuiteRoundRefused):
        run_suite_round(suite_route(session, point), request, on_poll=lambda: None)

    (leased,) = lease.leased
    assert leased.session_remaining_sec == pytest.approx(600 + _suite_config._LEASED_BACKSTOP_SLACK_SEC, abs=5)


def test_a_leased_round_sizes_its_server_by_tp_not_by_the_gpu_mask_the_point_carries(
    session, tmp_path, monkeypatch
) -> None:
    lease = _Lease(_rounds())
    point = _write_point(tmp_path / "slot" / "point.yaml", ROCR_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")

    run_suite_round(suite_route(session, point), _request(point, lease), on_poll=lambda: None)

    (leased,) = lease.leased
    assert _gpus(leased) == (0, 1)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "6,7")
    monkeypatch.setattr(SessionKillRound, "run", lambda self: (0, "", ""))
    assert leased.run() == (0, "", "")
    assert _gpus(leased) == (6, 7)


def _leased(tmp_path: Path, gpus: tuple[int, ...]) -> SuiteLeasedRound:
    config = _config()
    config = replace(config, server=replace(config.server, gpus=gpus))
    path = tmp_path / "suite_run_0.config.yaml"
    path.write_text(yaml.safe_dump(config.to_document()))
    return SuiteLeasedRound(str(path), str(tmp_path / "suite_run_0"), env={}, timeout=60.0)


def _ran(monkeypatch) -> list[SessionKillRound]:
    ran: list[SessionKillRound] = []
    monkeypatch.setattr(SessionKillRound, "run", lambda self: ran.append(self) or (0, "", ""))
    return ran


def _gpus(leased: SuiteLeasedRound) -> tuple[int, ...]:
    return SuiteConfig.from_document(yaml.safe_load(Path(leased.config_path).read_text())).server.gpus


def test_the_actor_pins_the_round_to_the_gpus_ray_placed_it_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    ran = _ran(monkeypatch)
    leased = _leased(tmp_path, (0, 1))

    assert leased.run() == (0, "", "")

    assert _gpus(leased) == (4, 5)
    (process,) = ran
    assert (process.cmd, process.timeout) == (leased.cmd, leased.timeout)
    assert process.server_log_path is None and process.silence_timeout_sec is None


def test_an_actor_ray_narrowed_nothing_for_runs_the_config_as_written(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    ran = _ran(monkeypatch)
    leased = _leased(tmp_path, (0, 1))

    leased.run()

    assert _gpus(leased) == (0, 1) and len(ran) == 1


def test_an_actor_placed_on_another_gpu_count_refuses_the_round(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4")
    ran = _ran(monkeypatch)
    leased = _leased(tmp_path, (0, 1))

    returncode, _out, stderr = leased.run()

    assert returncode == EXIT_REFUSED and "config_invalid" in stderr
    assert _gpus(leased) == (0, 1) and not ran


class _Actor:
    """A lease whose actor plays each round with whatever ``run_round`` the test installed in ``_suite_config``."""

    def __init__(self) -> None:
        self.leased: list[SuiteLeasedRound] = []
        self.closed = False

    def run_session_kill(self, leased: SuiteLeasedRound) -> tuple[int, str, str]:
        self.leased.append(leased)
        config = SuiteConfig.from_document(yaml.safe_load(Path(leased.config_path).read_text()))
        _suite_config.run_round(replace(config, paths=replace(config.paths, run_dir=leased.run_dir)))
        return 0, "", ""

    def close(self) -> None:
        self.closed = True


def test_a_baseline_with_a_serving_lease_runs_both_rounds_of_its_pair_in_the_lease(monkeypatch, suite) -> None:
    actor = _Actor()
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda num_gpus: actor)

    result, rounds = baseline_play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT),
    )

    assert result["status"] == "succeeded", result
    assert len(actor.leased) == len(rounds.calls) == 2 and actor.closed
    assert [call.config.lifecycle.mode for call in rounds.calls] == [LifecycleMode.BOOT, LifecycleMode.REUSE]


def test_a_grid_with_a_serving_lease_runs_its_rounds_in_the_lease(monkeypatch, grid_suite) -> None:
    actor = _Actor()

    row, calls, _slot = grid_play(
        monkeypatch,
        grid_suite,
        LaunchAttempt(outcome=READY, metrics=_HOT),
        launch=GridLaunch(warmup_before_measure=False, serving_lease=actor),
    )

    assert row.status == "succeeded", row
    assert len(actor.leased) == len(calls) == 1
