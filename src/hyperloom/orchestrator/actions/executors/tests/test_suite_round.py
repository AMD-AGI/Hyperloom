# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which points run on inference_suite, and the round each runs as: its run_dir, lifecycle, seed and refusals."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.state.tests._session_base import freeze_base
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors import _agentx_checkpoint, _suite_config
from hyperloom.orchestrator.actions.executors._agentx_checkpoint import (
    CHECKPOINT_ARTIFACT_NAME,
    AgentxCheckpoint,
    checkpoint_summary,
    read_checkpoint,
)
from hyperloom.orchestrator.actions.executors._grid_base import GridRound, run_grid_suite_round
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    AGENTX_EARLY_STOP_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
)
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._suite_config import MEASURED_SEED, WARMUP_SEED
from hyperloom.orchestrator.actions.executors._suite_config import (
    RUN_DIR_PREFIX,
    TOOL_CACHE_ENV,
    SuiteRound,
    SuiteRoundRefused,
    inherited_env_names,
    round_lifecycle,
    run_suite_round,
    session_route,
    suite_route,
)
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _base, _point
from hyperloom.orchestrator.rehearsal import READY, LaunchAttempt, LaunchScenario, ScriptedRounds, VirtualClock
from inference_suite.contracts import FailedStep, LifecycleMode, RunStatus

_STAGED = b'{"schema": "inference-suite-staging/1", "tools": {}}\n'


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    freeze_base(session_dir, _base("sglang"))
    cache = tmp_path / "tools"
    cache.mkdir()
    (cache / "staging_manifest.json").write_bytes(_STAGED)
    monkeypatch.setenv(TOOL_CACHE_ENV, str(cache))
    monkeypatch.setattr(_suite_config, "_resolve_probe_python", lambda framework: f"/venv/{framework}/python")
    monkeypatch.setattr(_suite_config, "client_only_server_url", lambda: None)
    return session_dir


def _write_point(path: Path, **envs: str) -> Path:
    point = _point("sglang", RUN_EVAL="false", **envs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(point), encoding="utf-8")
    return path


def _rounds(count: int = 1) -> ScriptedRounds:
    attempts = tuple(LaunchAttempt(outcome=READY, duration_sec=10.0) for _ in range(count))
    return ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock())


def test_a_point_of_a_session_with_a_frozen_base_runs_on_the_suite(session: Path, tmp_path: Path) -> None:
    route = suite_route(session)

    assert route is not None
    assert route.base_path == session / "base_launch.json"
    assert route.framework_python == "/venv/sglang/python"
    assert route.manifest_sha256 is not None


def test_a_session_without_a_frozen_base_stays_on_the_backend(session: Path, tmp_path: Path) -> None:
    unfrozen = tmp_path / "unfrozen"
    unfrozen.mkdir()

    assert suite_route(unfrozen) is None
    assert suite_route(None) is None


def test_every_point_of_a_session_with_a_frozen_base_runs_on_the_suite(session) -> None:
    """The route is the session's: no point, scriptable or serving, falls back to the benchmark backend."""
    assert suite_route(session) == session_route(session)


def test_a_client_only_session_runs_external_rounds_against_its_server(session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_suite_config, "client_only_server_url", lambda: "http://10.0.0.2:30000")
    rounds = _rounds()
    monkeypatch.setattr(_suite_config, "run_round", rounds.run_round)
    point = _write_point(tmp_path / "slot" / "point.yaml")
    route = suite_route(session)
    assert route is not None and route.server_url == "http://10.0.0.2:30000"

    ran = run_suite_round(
        route, SuiteRound(point, point.parent, round_lifecycle(point.parent, LifecycleMode.OWN)), on_poll=lambda: None
    )

    assert ran.report.run_status is RunStatus.COMPLETED
    (config,) = [call.config for call in rounds.calls]
    assert config.lifecycle.mode is LifecycleMode.EXTERNAL
    assert (config.server.host, config.server.port) == ("10.0.0.2", 30000)


def test_a_lone_round_owns_its_server_and_a_cold_start_pair_shares_one_through_the_slot(tmp_path: Path) -> None:
    slot = tmp_path / "slot"

    assert round_lifecycle(slot, LifecycleMode.OWN).mode is LifecycleMode.OWN
    boot = round_lifecycle(slot / "warmup_round", LifecycleMode.BOOT)
    reuse = round_lifecycle(slot / "measure_round", LifecycleMode.REUSE)
    assert (boot.mode, boot.pid_dir) == (LifecycleMode.BOOT, str(slot))
    assert (reuse.mode, reuse.pid_dir) == (LifecycleMode.REUSE, str(slot))


def test_each_round_writes_a_fresh_run_dir_so_a_retry_in_the_same_directory_runs(session, tmp_path, monkeypatch):
    rounds = _rounds(2)
    monkeypatch.setattr(_suite_config, "run_round", rounds.run_round)
    route = suite_route(session)
    point = _write_point(tmp_path / "slot" / "point.yaml")
    request = SuiteRound(point, tmp_path / "slot", round_lifecycle(tmp_path, LifecycleMode.OWN))

    first = run_suite_round(route, request, on_poll=lambda: None)
    second = run_suite_round(route, request, on_poll=lambda: None)

    assert [first.layout.root.name, second.layout.root.name] == [f"{RUN_DIR_PREFIX}0", f"{RUN_DIR_PREFIX}1"]
    assert second.report.run_status is RunStatus.COMPLETED


@pytest.mark.parametrize(("warmup", "seed"), [(False, MEASURED_SEED), (True, WARMUP_SEED)])
def test_a_discarded_warmup_round_draws_its_prompts_with_the_warmup_seed(session, tmp_path, monkeypatch, warmup, seed):
    rounds = _rounds()
    monkeypatch.setattr(_suite_config, "run_round", rounds.run_round)
    point = _write_point(tmp_path / "slot" / "point.yaml")
    lifecycle = round_lifecycle(point.parent / "warmup_round", LifecycleMode.BOOT if warmup else LifecycleMode.OWN)

    run_suite_round(suite_route(session), SuiteRound(point, point.parent, lifecycle, warmup), on_poll=lambda: None)

    (config,) = [call.config for call in rounds.calls]
    assert config.fixed.seed == seed
    assert config.lifecycle == lifecycle


def test_the_round_sees_the_host_environment_but_not_its_gpu_masks() -> None:
    names = inherited_env_names({"PATH": "/bin", "HF_HOME": "/hf", "ROCR_VISIBLE_DEVICES": "0", "BASH_FUNC_x%%": "()"})

    assert names == ("HF_HOME", "PATH")


def test_a_cache_with_nothing_staged_refuses_the_round_as_staging_missing(session, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(TOOL_CACHE_ENV, str(tmp_path / "empty"))
    point = _write_point(tmp_path / "slot" / "point.yaml")
    lifecycle = round_lifecycle(point.parent, LifecycleMode.OWN)

    with pytest.raises(SuiteRoundRefused) as refused:
        run_suite_round(suite_route(session), SuiteRound(point, point.parent, lifecycle), on_poll=lambda: None)

    assert refused.value.error_class == failed_step_error_class(FailedStep.STAGING_MISSING)


def test_a_point_that_does_not_translate_refuses_the_round_as_config_invalid(session, tmp_path) -> None:
    point = _write_point(tmp_path / "slot" / "point.yaml")
    route = suite_route(session)
    _write_point(point, TP="4")
    lifecycle = round_lifecycle(point.parent, LifecycleMode.OWN)

    with pytest.raises(SuiteRoundRefused) as refused:
        run_suite_round(route, SuiteRound(point, point.parent, lifecycle), on_poll=lambda: None)

    assert refused.value.error_class == failed_step_error_class(FailedStep.CONFIG_INVALID)
    assert "TP 4" in refused.value.detail


def test_the_actions_cancel_scope_cancels_the_round_and_each_poll_is_heard(session, tmp_path, monkeypatch) -> None:
    rounds = _rounds()
    monkeypatch.setattr(_suite_config, "run_round", rounds.run_round)
    point = _write_point(tmp_path / "slot" / "point.yaml")
    lifecycle = round_lifecycle(point.parent, LifecycleMode.OWN)
    scope = CancelScope()
    scope.cancel(reason="the coordinator stopped the action")
    polls: list[None] = []

    with use_cancel_scope(scope):
        result = run_suite_round(
            suite_route(session), SuiteRound(point, point.parent, lifecycle), on_poll=lambda: polls.append(None)
        )

    assert result.report.failed_step is FailedStep.CANCELLED
    assert result.report.failure_detail == "the coordinator stopped the action"
    assert polls


def test_a_point_naming_an_unknown_framework_runs_on_the_suite_and_is_refused_there(session, tmp_path) -> None:
    point = _write_point(tmp_path / "slot" / "point.yaml")
    document = yaml.safe_load(point.read_text(encoding="utf-8"))
    document["benchmark"]["framework"] = "tgi"
    point.write_text(yaml.safe_dump(document), encoding="utf-8")
    route = suite_route(session)
    lifecycle = round_lifecycle(point.parent, LifecycleMode.OWN)

    with pytest.raises(SuiteRoundRefused) as refused:
        run_suite_round(route, SuiteRound(point, point.parent, lifecycle), on_poll=lambda: None)

    assert refused.value.error_class == failed_step_error_class(FailedStep.CONFIG_INVALID)


def _grid_round(point: Path, checkpoint: AgentxCheckpoint) -> GridRound:
    return GridRound(
        magpie_python="",
        config_path=point,
        output_dir=point.parent,
        timeout_sec=600,
        cwd=str(point.parent),
        checkpoint=checkpoint,
    )


@pytest.fixture
def due_checkpoint(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    """A checkpoint that is due as the round starts and reads ``candidate`` as the round's median."""
    monkeypatch.setattr(_agentx_checkpoint, "AGENTX_CHECKPOINT_SEC", 0.0)

    def reads(candidate: float) -> None:
        monkeypatch.setattr(_agentx_checkpoint, "workspace_checkpoint_p50", lambda root: candidate)

    return reads


def test_a_round_trailing_its_checkpoint_is_stopped_and_exits_as_an_early_stop(
    session, tmp_path, monkeypatch, due_checkpoint
) -> None:
    monkeypatch.setattr(_suite_config, "run_round", _rounds().run_round)
    due_checkpoint(90.0)
    point = _write_point(tmp_path / "slot" / "point.yaml")

    ended = run_grid_suite_round(
        suite_route(session),
        _grid_round(point, AgentxCheckpoint(100.0, enforce=True)),
        env_drop=frozenset(),
        serving_lease=None,
        on_poll=lambda: None,
    )

    ruling = read_checkpoint(point.parent)
    assert ended.returncode == AGENTX_EARLY_STOP_RETURNCODE
    assert ended.detail == checkpoint_summary(ruling) and ruling["stopped"] is True


def test_a_shadow_checkpoint_records_the_ruling_and_lets_the_round_finish(
    session, tmp_path, monkeypatch, due_checkpoint
) -> None:
    monkeypatch.setattr(_suite_config, "run_round", _rounds().run_round)
    due_checkpoint(90.0)
    point = _write_point(tmp_path / "slot" / "point.yaml")

    ended = run_grid_suite_round(
        suite_route(session),
        _grid_round(point, AgentxCheckpoint(100.0, enforce=False)),
        env_drop=frozenset(),
        serving_lease=None,
        on_poll=lambda: None,
    )

    ruling = read_checkpoint(point.parent)
    assert ended.returncode == 0
    assert (ruling["trails"], ruling["stopped"]) == (True, False)


def test_an_earlier_rounds_stop_ruling_does_not_turn_a_cancel_into_an_early_stop(
    session, tmp_path, monkeypatch, due_checkpoint
) -> None:
    monkeypatch.setattr(_suite_config, "run_round", _rounds().run_round)
    due_checkpoint(100.0)
    point = _write_point(tmp_path / "slot" / "point.yaml")
    (point.parent / CHECKPOINT_ARTIFACT_NAME).write_text(json.dumps({"stopped": True}), encoding="utf-8")
    scope = CancelScope()
    scope.cancel(reason="the coordinator stopped the action")

    with use_cancel_scope(scope):
        ended = run_grid_suite_round(
            suite_route(session),
            _grid_round(point, AgentxCheckpoint(100.0, enforce=True)),
            env_drop=frozenset(),
            serving_lease=None,
            on_poll=lambda: None,
        )

    assert ended.returncode == ORCHESTRATOR_CANCELLED_RETURNCODE
    assert read_checkpoint(point.parent)["stopped"] is False
