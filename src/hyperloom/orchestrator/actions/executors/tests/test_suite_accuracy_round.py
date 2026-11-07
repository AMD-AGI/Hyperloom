# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The D9 accuracy round of an AgentX slot: the slot's launch on a server of its own, without golden, eval only.

The points are composed the way the baseline composes them: the session base and its golden layer seeded in place of
the recipe's argv, the slot's own layer applied above.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _suite_config
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._suite_config import (
    ACCURACY_DIR,
    TOOL_CACHE_ENV,
    AccuracyRound,
    SuiteRoundRefused,
    accuracy_config,
    run_accuracy_suite_round,
    session_route,
)
from hyperloom.orchestrator.actions.executors.magpie_agentx import LaunchChanges, apply_launch_changes
from hyperloom.orchestrator.actions.executors.magpie_agentx import seed_session_base
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import AccuracyRoundOutcome, run_accuracy_round
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _settings
from hyperloom.orchestrator.state.base_launch import ServerBase
from hyperloom.orchestrator.state.tests._session_base import freeze_base, session_base
from inference_suite.compose import RoundFacts, compose
from inference_suite.contracts import (
    BaseLaunch,
    ConfigError,
    FailedStep,
    GoldenAcceptance,
    GoldenCurve,
    GoldenUse,
    LaunchDelta,
    LifecycleMode,
    Workload,
    WorkloadKind,
)
from inference_suite.contracts.launch import Launch, VersionCheck

_MODEL_PATH = "/models/glm"
_BASE_ARGS = ("--kv-cache-dtype", "fp8_e4m3", "--mem-fraction-static", "0.8")
_BASE_ENV = {"SGLANG_USE_AITER": "1"}
_GOLDEN = GoldenAcceptance(
    value=3.48,
    method="match-expected",
    token_mode="real-draft-token",
    spec_method="EAGLE",
    draft_length=3,
    curve=GoldenCurve(repo="https://github.com/org/curves", commit="e" * 40, path="al.json", sha256="f" * 64),
    args=("--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3"),
    env={"SGLANG_SIMULATE_ACC_LEN": "3.48"},
)
_SLOT_LAYER = LaunchChanges(
    args="--cuda-graph-max-bs 64 --mem-fraction-static 0.85",
    env={"AITER_LOG_TUNED_CONFIG": "1"},
    remove_args=["--kv-cache-dtype"],
    unset_env=["SGLANG_USE_AITER"],
)


def _agentx_base(golden: GoldenAcceptance | None = _GOLDEN) -> BaseLaunch:
    base = session_base(_BASE_ARGS, _BASE_ENV, golden)
    return replace(base, query=replace(base.query, workload=Workload.AGENTX, isl=None, osl=None))


def _agentx_point(base: BaseLaunch, layer: LaunchChanges = _SLOT_LAYER) -> dict:
    golden = base.golden_al
    server_base = ServerBase(
        args=" ".join((*base.args, *(golden.args if golden else ()))),
        env={**base.env, **(golden.env if golden else {})},
    )
    bench = {
        "framework": "sglang",
        "model": "org/model",
        "agentx": {"enabled": True},
        "envs": {"TP": "1", "CONC": "8", "MODEL_PATH": _MODEL_PATH, "RUN_EVAL": "false"},
    }
    seed_session_base(bench, server_base, recipe=SimpleNamespace(argv=(), env={}))
    apply_launch_changes(bench, layer, accepted=True)
    return {"benchmark": bench}


def test_the_round_serves_the_slots_launch_on_its_own_server_without_golden_and_runs_only_eval() -> None:
    base = _agentx_base()

    config = accuracy_config(_agentx_point(base), base, _settings(LifecycleMode.REUSE), 900.0)

    assert (config.workload, config.server.golden, config.eval.enabled) == (WorkloadKind.NONE, GoldenUse.EXCLUDE, True)
    assert config.lifecycle.mode is LifecycleMode.OWN
    assert config.eval.concurrency == 8
    assert (config.deadlines.boot_sec, config.deadlines.eval_sec) == (900.0, 900.0)
    assert config.server.model_path == _MODEL_PATH
    assert config.server.user == LaunchDelta(
        args_add=("--mem-fraction-static", "0.85", "--cuda-graph-max-bs", "64"),
        args_remove=("--kv-cache-dtype",),
        env_set={"AITER_LOG_TUNED_CONFIG": "1"},
        env_unset=("SGLANG_USE_AITER",),
    )
    launch = compose(config, base, RoundFacts("0" * 64, "0.0.0", VersionCheck.NOT_APPLICABLE))
    assert isinstance(launch, Launch), launch
    flags = [arg.flag for arg in launch.args]
    assert "--speculative-algorithm" not in flags and "--speculative-num-steps" not in flags
    assert "--kv-cache-dtype" not in flags and "--cuda-graph-max-bs" in flags
    assert "SGLANG_SIMULATE_ACC_LEN" not in {env.name for env in launch.env_sources}


def test_a_slot_running_the_base_unchanged_adds_no_layer_of_its_own() -> None:
    base = _agentx_base()

    config = accuracy_config(_agentx_point(base, LaunchChanges()), base, _settings(), 900.0)

    assert config.server.user == LaunchDelta()


def test_a_base_without_a_golden_layer_has_nothing_to_exclude() -> None:
    base = _agentx_base(golden=None)

    config = accuracy_config(_agentx_point(base), base, _settings(), 900.0)

    assert config.server.golden is GoldenUse.APPLY
    assert config.workload is WorkloadKind.NONE and config.eval.enabled


def _fixed(point: dict) -> dict:
    point["benchmark"]["agentx"] = {"enabled": False}
    return point


def _unseeded(point: dict) -> dict:
    point["benchmark"]["agentx"]["launch_overrides"]["replace_args"] = False
    return point


@pytest.mark.parametrize(
    ("change", "settings", "message"),
    [
        (_fixed, {}, "serves an AgentX point"),
        (_unseeded, {}, "does not run the session base"),
        (lambda point: point, {"server_url": "http://10.0.0.2:30000"}, "client-only session"),
    ],
)
def test_a_slot_with_no_accuracy_round_is_refused(change, settings, message) -> None:
    base = _agentx_base()

    with pytest.raises(ConfigError, match=message):
        accuracy_config(change(_agentx_point(base)), base, replace(_settings(), **settings), 900.0)


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    freeze_base(session_dir, _agentx_base())
    cache = tmp_path / "tools"
    cache.mkdir()
    (cache / "staging_manifest.json").write_text('{"schema": "inference-suite-staging/1", "tools": {}}\n')
    monkeypatch.setenv(TOOL_CACHE_ENV, str(cache))
    monkeypatch.setattr(_suite_config, "_resolve_probe_python", lambda framework: f"/venv/{framework}/python")
    monkeypatch.setattr(_suite_config, "client_only_server_url", lambda: None)
    return session_dir


def _write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_the_round_writes_its_run_dir_into_the_slots_accuracy_dir(session, tmp_path, monkeypatch) -> None:
    configs = []
    monkeypatch.setattr(_suite_config, "run_round", lambda config, cancel: configs.append(config) or "ran")
    slot = tmp_path / "runs" / "baseline" / "t1"
    point = _write(slot / "point.yaml", _agentx_point(_agentx_base()))
    route = session_route(session)
    assert route is not None

    ran = run_accuracy_suite_round(route, AccuracyRound(point, slot, 900.0), on_poll=lambda: None)

    assert ran == "ran"
    (config,) = configs
    assert config.paths.run_dir == str(slot / ACCURACY_DIR)
    assert config.server.golden is GoldenUse.EXCLUDE


def test_a_fixed_point_refuses_the_accuracy_round_as_config_invalid(session, tmp_path) -> None:
    slot = tmp_path / "slot"
    point = _write(slot / "point.yaml", _fixed(_agentx_point(_agentx_base())))
    route = session_route(session)
    assert route is not None

    with pytest.raises(SuiteRoundRefused) as refused:
        run_accuracy_suite_round(route, AccuracyRound(point, slot, 900.0), on_poll=lambda: None)

    assert refused.value.error_class == failed_step_error_class(FailedStep.CONFIG_INVALID)


def test_the_accuracy_round_entry_reports_the_step_the_round_failed_at(session, tmp_path, monkeypatch) -> None:
    slot = tmp_path / "slot"
    point = _write(slot / "point.yaml", _agentx_point(_agentx_base()))
    failed = SimpleNamespace(
        report=SimpleNamespace(failed_step=FailedStep.EVAL_FAILED), layout=SimpleNamespace(root=slot)
    )
    monkeypatch.setattr(_suite_config, "run_round", lambda config, cancel: failed)

    outcome = asyncio.run(run_accuracy_round(slot, point, session=session, timeout_sec=900.0, deadline=None))

    assert outcome == AccuracyRoundOutcome(slot, failed_step_error_class(FailedStep.EVAL_FAILED))


def test_a_session_that_froze_no_base_has_no_accuracy_round(tmp_path) -> None:
    unfrozen = tmp_path / "unfrozen"
    unfrozen.mkdir()
    slot = tmp_path / "slot"
    point = _write(slot / "point.yaml", _agentx_point(_agentx_base()))

    outcome = asyncio.run(run_accuracy_round(slot, point, session=unfrozen, timeout_sec=900.0, deadline=None))

    assert outcome == AccuracyRoundOutcome(slot, failed_step_error_class(FailedStep.CONFIG_INVALID))
