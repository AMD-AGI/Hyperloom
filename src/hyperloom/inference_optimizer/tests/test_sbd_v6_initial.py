"""Focused coverage for the additive SBD V6 bootstrap fields and stages."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.critic_reviews import normalize_framework_reviews
from hyperloom.inference_optimizer.session.sbd_v6 import (
    SCHEMA_VERSION_V6,
    read_timeline_event,
    read_timeline_events,
    write_timeline_event_at,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _framework_events(timeline: list[dict]) -> list[dict]:
    """Keep only the ``framework_agent`` events.

    A state that walks the macro loop also produces ``kernel`` events (and a
    baseline throughput produces a ``baseline`` one). The tests below are
    scoped to the Framework Agent projection, so they filter rather than
    assert over the whole timeline.
    """
    return [event for event in timeline if event["type"] == "framework_agent"]


def _gate_args(model: Path, **overrides) -> argparse.Namespace:
    values = {
        "model": str(model),
        "model_display_name": model.name,
        "framework": "sglang",
        "gpu_type": "MI300X",
        "isl": 1024,
        "osl": 1024,
        "allow_mm_text_fallback": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _seed_state(session_dir: Path, monkeypatch, model: Path) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir))
    from hyperloom.orchestrator.state.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="sbd-v6-test", model_name=model.name, model_path=str(model)).save(session_dir)


def _write_model_config(model: Path, payload: dict) -> None:
    _write_json(model / "config.json", payload)
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _model_gate_from_breakdown(session_dir: Path) -> dict:
    breakdown = json.loads((session_dir / "session_breakdown.json").read_text(encoding="utf-8"))
    assert breakdown["schema_version"] == SCHEMA_VERSION_V6
    assert breakdown["metadata"]["versions"]["schema_version"] == SCHEMA_VERSION_V6
    assert breakdown["outcome"]["status"] == "failed"
    assert breakdown["outcome"]["stage_reached"] == "model_gate"
    return next(event for event in breakdown["timeline"] if event["type"] == "model_gate")


def test_v6_blocks_are_additive_to_the_rest_of_the_document(tmp_path):
    state = {
        "session_id": "session-v6",
        "model_name": "Qwen-Test",
        "model_path": "/models/qwen-test",
        "framework": "sglang",
        "gpu_type": "MI300X",
        "phase": "CLOSE",
        "start_ts": "2026-08-27T01:00:00+00:00",
        "stop_ts": "2026-08-27T02:00:00+00:00",
        "stop_reason": "target_reached",
        "tick": 9,
        "baseline_tput": 100.0,
        "baseline_accuracy": 0.75,
        "current_best": {
            "tput": 125.0,
            "extra_envs": {"SGLANG_USE_AITER": "1"},
            "extra_server_args": "--watchdog-timeout 1800",
        },
        "cumulative_gain_validated": 25.0,
        "operator_extra_env": {"TP": "8"},
        "operator_server_args": "--context-length 11264",
        "model_info": {
            "model_type": "qwen3",
            "num_hidden_layers": 36,
            "attention_type": "GQA",
            "num_experts": None,
        },
    }
    manifest = {
        "session_id": "session-v6",
        "created_at_utc": "2026-08-27T01:00:00+00:00",
        "host": "test-host",
        "code_revision": "abc1234",
        "pid": 42,
        "max_minutes": 180,
        "model_name": "Qwen-Test",
        "model_path": "/models/qwen-test",
        "framework": "sglang",
        "framework_version": "0.5.17",
        "gpu_type": "MI300X",
        "tp": 8,
        "workload": {
            "conc": 64,
            "isl": 1024,
            "osl": 1024,
            "precision": "bf16",
            "max_model_len": 11264,
        },
        "objective": {"kind": "throughput", "value": 120.0},
    }
    _write_json(tmp_path / "state.json", state)
    _write_json(tmp_path / "manifest.json", manifest)

    before = exporter.build(tmp_path)
    write_timeline_event_at(
        tmp_path,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T00:58:00+00:00",
            "end_time": "2026-08-27T00:59:00+00:00",
            "ext": {"run_kind": "fresh", "hard_fail_step_id": None, "runtime_snapshot": {}, "steps": []},
        },
    )
    write_timeline_event_at(
        tmp_path,
        {
            "type": "model_gate",
            "kind": "model_gate",
            "status": "succeeded",
            "start_time": "2026-08-27T00:59:00+00:00",
            "end_time": "2026-08-27T01:00:00+00:00",
            "ext": {"run_kind": "fresh", "checks": []},
        },
    )

    after = exporter.build(tmp_path)

    assert after["schema_version"] == SCHEMA_VERSION_V6
    v6_keys = {"exported_at_utc", "metadata", "outcome", "timeline", "close"}
    assert {key: value for key, value in after.items() if key not in v6_keys} == {
        key: value for key, value in before.items() if key not in v6_keys
    }
    assert after["metadata"]["versions"]["schema_version"] == SCHEMA_VERSION_V6
    assert after["metadata"]["versions"]["hyperloom"] == "abc1234"
    assert after["metadata"]["task_config"]["launch_env"] == {"TP": "8"}
    assert after["outcome"]["status"] == "completed"
    assert after["outcome"]["stage_reached"] == "close"
    assert "token_usage" not in after["outcome"]
    # Only the durable events. ``state.baseline_tput`` is a real measurement,
    # but baseline is recorded by the action that runs it rather than projected
    # from the section, and this session recorded none.
    assert [event["type"] for event in after["timeline"]] == ["install", "model_gate"]
    # No CLOSE step was ever recorded, so the close-out has no evidence.
    assert after["close"]["status"] == "failed"
    assert after["close"]["steps"] == []
    # ``close`` is a top-level key, never a timeline event.
    assert all(event["type"] != "close" for event in after["timeline"])


def test_invalid_v6_event_does_not_change_v5_warnings(tmp_path):
    before = exporter.build(tmp_path)
    path = tmp_path / "reports" / "sbd_v6" / "timeline" / "000001-install.json"
    path.parent.mkdir(parents=True)
    path.write_text("{invalid", encoding="utf-8")

    after = exporter.build(tmp_path)

    assert after["warnings"] == before["warnings"]
    assert any("timeline.install" in warning for warning in after["metadata"]["warnings"])
    assert after["timeline"] == []


def test_install_event_stays_pending_until_session_creation(tmp_path):
    from hyperloom.inference_optimizer.cli import preflight

    args = argparse.Namespace(resume_from=None, no_kernel=True, enable_roofline=False)
    event = preflight._begin_install_event(args)
    preflight._run_install_step(
        event,
        step_id="load_dotenv",
        category="normalize",
        action=lambda: {
            "status": "already_present",
            "skip_reason": None,
            "detail": {"vars_loaded": 0, "source": "/repo/.env"},
        },
    )
    preflight._run_install_step(
        event,
        step_id="check_shm_disk",
        category="check",
        action=lambda: {
            "status": "warned",
            "skip_reason": None,
            "detail": {"shm_free_gib": 8.0, "min_gib": 16},
        },
    )
    preflight._finish_install_event(
        event,
        args=args,
        benchmark_backend="bypass",
        benchmark_python="python",
        magpie_python="python",
        inferencex_path="/opt/InferenceX",
        resolved_urls=("", "https://api.openai.com/v1"),
    )

    assert not (tmp_path / "reports" / "sbd_v6" / "timeline").exists()
    preflight._persist_install_event(args, tmp_path)

    persisted = read_timeline_event(tmp_path, "install")
    assert persisted is not None
    assert persisted["status"] == "degraded"
    assert [step["step_id"] for step in persisted["ext"]["steps"]] == [
        "load_dotenv",
        "check_shm_disk",
    ]
    assert persisted["ext"]["runtime_snapshot"]["provider_mode"] == "openai"


@pytest.mark.parametrize("failure", [SystemExit(2), KeyboardInterrupt()])
def test_expected_preflight_exit_does_not_create_session(tmp_path, monkeypatch, failure):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    workspace = tmp_path / "sessions"
    model = tmp_path / "Qwen-Test"
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )

    def fail_preflight(args):
        event = preflight._begin_install_event(args)

        def reject_credentials():
            raise failure

        preflight._run_install_step(
            event,
            step_id="validate_credentials",
            category="check",
            action=reject_credentials,
        )

    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--model", str(model)])

    with pytest.raises(type(failure)) as exc:
        asyncio.run(optimizer_cli._run_optimize(args))

    if isinstance(failure, SystemExit):
        assert exc.value.code == 2
    assert ENV_CURRENT_SESSION_DIR not in os.environ
    assert not workspace.exists() or not list(workspace.rglob("session_breakdown.json"))


@pytest.mark.parametrize("failure", [SystemExit(2), KeyboardInterrupt()])
def test_expected_resume_preflight_exit_does_not_mutate_session(tmp_path, monkeypatch, failure):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    workspace = tmp_path / "sessions"
    resume_dir = workspace / "Qwen-Test" / "existing-session"
    _write_json(resume_dir / "session_breakdown.json", {"sentinel": "unchanged"})
    before = {
        path.relative_to(resume_dir).as_posix(): path.read_bytes() for path in resume_dir.rglob("*") if path.is_file()
    }
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )

    def fail_preflight(args):
        event = preflight._begin_install_event(args)

        def reject_environment():
            raise failure

        preflight._run_install_step(
            event,
            step_id="validate_environment",
            category="check",
            action=reject_environment,
        )

    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--resume-from", str(resume_dir)])

    with pytest.raises(type(failure)):
        asyncio.run(optimizer_cli._run_optimize(args))

    after = {
        path.relative_to(resume_dir).as_posix(): path.read_bytes() for path in resume_dir.rglob("*") if path.is_file()
    }
    assert after == before
    assert ENV_CURRENT_SESSION_DIR not in os.environ
    assert not list(workspace.rglob("*failed-attempt*"))


def test_unwrapped_preflight_failure_is_persisted_as_failed(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    workspace = tmp_path / "sessions"
    model = tmp_path / "Qwen-Test"
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )
    monkeypatch.setattr(preflight, "_load_dotenv_fallback", lambda: None)
    monkeypatch.setattr(preflight, "_provider_only_mode", lambda: "")
    monkeypatch.setattr(preflight, "_load_kernel_agent_env_fallback", lambda: None)

    def fail_runtime_paths():
        raise RuntimeError("runtime path resolution failed")

    monkeypatch.setattr(preflight, "_derive_runtime_paths", fail_runtime_paths)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--model", str(model)])

    with pytest.raises(RuntimeError, match="runtime path resolution failed"):
        asyncio.run(optimizer_cli._run_optimize(args))

    session_dir = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    install = read_timeline_event(session_dir, "install")
    assert install is not None
    assert install["status"] == "failed"
    assert install["end_time"]
    assert install["ext"]["hard_fail_step_id"] == "unhandled_preflight"
    failure = install["ext"]["steps"][-1]
    assert failure["step_id"] == "unhandled_preflight"
    assert failure["error_class"] == "RuntimeError"
    assert failure["message"] == "runtime path resolution failed"
    breakdown = json.loads((session_dir / "session_breakdown.json").read_text(encoding="utf-8"))
    assert breakdown["outcome"]["status"] == "failed"


def test_resume_preflight_failure_uses_isolated_session(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    workspace = tmp_path / "sessions"
    resume_dir = workspace / "Qwen-Test" / "active-session"
    model = tmp_path / "Qwen-Test"
    _seed_state(resume_dir, monkeypatch, model)
    _write_json(resume_dir / "manifest.json", {"schema_version": 4, "session_id": "sbd-v6-test"})
    original_install = {
        "type": "install",
        "kind": "install",
        "status": "succeeded",
        "start_time": "2026-08-27T01:00:00+00:00",
        "end_time": "2026-08-27T01:01:00+00:00",
        "ext": {"run_kind": "fresh", "steps": []},
    }
    original_install_public = json.loads(json.dumps(original_install))
    write_timeline_event_at(resume_dir, original_install)
    _write_json(resume_dir / "session_breakdown.json", {"sentinel": "active"})
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )

    def fail_preflight(args):
        preflight._begin_install_event(args)
        raise RuntimeError("resume preflight failed")

    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--resume-from", str(resume_dir)])

    with pytest.raises(RuntimeError, match="resume preflight failed"):
        asyncio.run(optimizer_cli._run_optimize(args))

    assert read_timeline_events(resume_dir) == [original_install_public]
    assert json.loads((resume_dir / "session_breakdown.json").read_text(encoding="utf-8")) == {"sentinel": "active"}
    failed_session = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    assert failed_session != resume_dir
    install = read_timeline_event(failed_session, "install")
    assert install is not None
    assert install["status"] == "failed"
    assert install["ext"]["run_kind"] == "resume"


def test_resume_preflight_failure_does_not_overwrite_completed_outcome(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
    from hyperloom.orchestrator.state.shared_state import SharedState

    workspace = tmp_path / "sessions"
    resume_dir = workspace / "Qwen-Test" / "completed-session"
    model = tmp_path / "Qwen-Test"
    _seed_state(resume_dir, monkeypatch, model)
    state = SharedState.load_or_init(resume_dir)
    state.phase = "CLOSE"
    state.stop_reason = "target_reached"
    state.save(resume_dir)
    _write_json(resume_dir / "manifest.json", {"schema_version": 4, "session_id": "sbd-v6-test"})
    write_timeline_event_at(
        resume_dir,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T01:00:00+00:00",
            "end_time": "2026-08-27T01:01:00+00:00",
            "ext": {"run_kind": "fresh", "steps": []},
        },
    )
    exporter.write_breakdown_json(resume_dir)
    original_breakdown = (resume_dir / "session_breakdown.json").read_bytes()
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )

    def fail_preflight(args):
        preflight._begin_install_event(args)
        raise RuntimeError("resume preflight failed")

    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--resume-from", str(resume_dir)])

    with pytest.raises(RuntimeError, match="resume preflight failed"):
        asyncio.run(optimizer_cli._run_optimize(args))

    assert (resume_dir / "session_breakdown.json").read_bytes() == original_breakdown
    assert [(event["type"], event["status"]) for event in read_timeline_events(resume_dir)] == [
        ("install", "succeeded")
    ]
    failed_session = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    assert failed_session != resume_dir
    breakdown = json.loads((failed_session / "session_breakdown.json").read_text(encoding="utf-8"))
    assert breakdown["outcome"]["status"] == "failed"
    assert breakdown["timeline"][-1]["type"] == "install"
    assert breakdown["timeline"][-1]["status"] == "failed"


@pytest.mark.parametrize("invalid_artifact", ["missing", "manifest", "state"])
def test_invalid_resume_preflight_failure_does_not_mutate_requested_directory(
    tmp_path,
    monkeypatch,
    invalid_artifact,
):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    workspace = tmp_path / "sessions"
    resume_dir = workspace / "not-a-session"
    _write_json(resume_dir / "session_breakdown.json", {"sentinel": "unchanged"})
    if invalid_artifact != "missing":
        manifest = resume_dir / "manifest.json"
        state = resume_dir / "state.json"
        manifest.write_text(
            "{invalid" if invalid_artifact == "manifest" else json.dumps({"session_id": "not-a-session"}),
            encoding="utf-8",
        )
        state.write_text(
            "{invalid" if invalid_artifact == "state" else json.dumps({"session_id": "not-a-session"}),
            encoding="utf-8",
        )
    before = {
        path.relative_to(resume_dir).as_posix(): path.read_bytes() for path in resume_dir.rglob("*") if path.is_file()
    }
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )

    def fail_preflight(args):
        preflight._begin_install_event(args)
        raise RuntimeError("invalid resume preflight failed")

    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--resume-from", str(resume_dir)])

    with pytest.raises(RuntimeError, match="invalid resume preflight failed"):
        asyncio.run(optimizer_cli._run_optimize(args))

    after = {
        path.relative_to(resume_dir).as_posix(): path.read_bytes() for path in resume_dir.rglob("*") if path.is_file()
    }
    assert after == before
    assert not (resume_dir / "runtime").exists()
    assert not (resume_dir / "reports").exists()
    failed_session = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    assert failed_session != resume_dir
    assert "failed-attempt" in failed_session.parent.name
    install = read_timeline_event(failed_session, "install")
    assert install is not None
    assert install["status"] == "failed"


def test_timeline_history_retains_fresh_and_resume_events(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import model_gate

    write_timeline_event_at(
        tmp_path,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T01:00:00+00:00",
            "end_time": "2026-08-27T01:01:00+00:00",
            "ext": {"run_kind": "fresh", "steps": []},
        },
    )
    timestamps = iter(
        (
            "2026-08-27T01:02:00+00:00",
            "2026-08-27T01:03:00+00:00",
            "2026-08-27T02:02:00+00:00",
            "2026-08-27T02:03:00+00:00",
            "2026-08-27T02:04:00+00:00",
        )
    )
    monkeypatch.setattr(model_gate, "now_iso", lambda **_kwargs: next(timestamps))
    fresh_args = _gate_args(tmp_path / "model")
    model_gate._start_model_gate(fresh_args, tmp_path)
    model_gate._finish_model_gate(fresh_args, tmp_path)
    write_timeline_event_at(
        tmp_path,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T02:00:00+00:00",
            "end_time": "2026-08-27T02:01:00+00:00",
            "ext": {"run_kind": "resume", "steps": []},
        },
    )
    resume_args = _gate_args(tmp_path / "model", resume_from=str(tmp_path))
    model_gate._record_resumed_model_gate(resume_args, tmp_path)

    expected = [
        ("install", "succeeded", "fresh"),
        ("model_gate", "succeeded", "fresh"),
        ("install", "succeeded", "resume"),
        ("model_gate", "skipped", "resume"),
    ]
    assert [
        (event["type"], event["status"], event["ext"]["run_kind"]) for event in read_timeline_events(tmp_path)
    ] == expected
    assert [
        (event["type"], event["status"], event["ext"]["run_kind"])
        for event in collect_v6_timeline(tmp_path, [], state={}, recorded_operations=[])
    ] == expected
    latest_gate = read_timeline_event(tmp_path, "model_gate")
    assert latest_gate is not None
    assert latest_gate["ext"]["run_kind"] == "resume"


def test_timeline_reads_ignore_flat_files_without_mutating_session(tmp_path):
    _write_json(
        tmp_path / "reports" / "sbd_v6" / "install.json",
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T01:00:00+00:00",
            "end_time": "2026-08-27T01:01:00+00:00",
            "ext": {"run_kind": "fresh", "steps": []},
        },
    )
    _write_json(
        tmp_path / "reports" / "sbd_v6" / "model_gate.json",
        {
            "type": "model_gate",
            "kind": "model_gate",
            "status": "succeeded",
            "start_time": "2026-08-27T01:02:00+00:00",
            "end_time": "2026-08-27T01:03:00+00:00",
            "ext": {"run_kind": "fresh", "checks": []},
        },
    )

    before = {path.name: path.read_bytes() for path in (tmp_path / "reports" / "sbd_v6").iterdir()}

    assert read_timeline_events(tmp_path) == []
    assert read_timeline_event(tmp_path, "install") is None
    assert not (tmp_path / "reports" / "sbd_v6" / "timeline").exists()
    after = {path.name: path.read_bytes() for path in (tmp_path / "reports" / "sbd_v6").iterdir()}
    assert after == before


def test_timeline_writer_does_not_migrate_flat_files(tmp_path):
    fresh_install = {
        "type": "install",
        "kind": "install",
        "status": "succeeded",
        "start_time": "2026-08-27T01:00:00+00:00",
        "end_time": "2026-08-27T01:01:00+00:00",
        "ext": {"run_kind": "fresh", "steps": []},
    }
    _write_json(tmp_path / "reports" / "sbd_v6" / "install.json", fresh_install)
    _write_json(
        tmp_path / "reports" / "sbd_v6" / "model_gate.json",
        {
            "type": "model_gate",
            "kind": "model_gate",
            "status": "succeeded",
            "start_time": "2026-08-27T01:02:00+00:00",
            "end_time": "2026-08-27T01:03:00+00:00",
            "ext": {"run_kind": "fresh", "checks": []},
        },
    )

    write_timeline_event_at(
        tmp_path,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T02:00:00+00:00",
            "end_time": "2026-08-27T02:01:00+00:00",
            "ext": {"run_kind": "resume", "steps": []},
        },
    )

    assert [(event["type"], event["ext"]["run_kind"]) for event in read_timeline_events(tmp_path)] == [
        ("install", "resume")
    ]
    assert (tmp_path / "reports" / "sbd_v6" / "timeline" / "000001-install.json").is_file()
    assert json.loads((tmp_path / "reports" / "sbd_v6" / "install.json").read_text(encoding="utf-8")) == (fresh_install)


def test_preflight_records_install_steps_in_execution_order(tmp_path, monkeypatch):
    from hyperloom.agents.framework import kb as framework_kb
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.sbd_v6 import pending_install_event
    from hyperloom.orchestrator.actions.executors import benchmark_backend

    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex))
    monkeypatch.setattr(benchmark_backend, "resolve_backend_name", lambda: "bypass")
    monkeypatch.setattr(benchmark_backend, "resolve_benchmark_interpreter", lambda: "python")
    monkeypatch.setattr(preflight, "_provider_only_mode", lambda: "")
    monkeypatch.setattr(preflight, "_load_dotenv_fallback", lambda: None)
    monkeypatch.setattr(preflight, "_load_kernel_agent_env_fallback", lambda: None)
    monkeypatch.setattr(preflight, "_derive_runtime_paths", lambda: None)
    monkeypatch.setattr(preflight, "_restore_provider_only_mode", lambda *_args: None)
    monkeypatch.setattr(preflight, "_normalize_legacy_deepseek_env", lambda: None)
    monkeypatch.setattr(preflight, "_validate_credentials", lambda: None)
    monkeypatch.setattr(framework_kb, "prepare_kb_environment", lambda: None)
    monkeypatch.setattr(preflight, "_ensure_python_sdks", lambda *_args: None)
    monkeypatch.setattr(preflight, "_resolve_llm_endpoints", lambda: ("", ""))
    monkeypatch.setattr(preflight, "_unset_hip_visible_devices", lambda: None)
    monkeypatch.setattr(preflight, "_check_gpu_visibility", lambda: None)
    monkeypatch.setattr(preflight, "_check_shm_disk", lambda: None)
    monkeypatch.setattr(preflight, "_check_platform_tuning", lambda: None)
    monkeypatch.setattr(preflight, "_ensure_ray", lambda *_args: None)
    monkeypatch.setattr(preflight, "_ensure_bench_serving_deps", lambda *_args: None)
    monkeypatch.setattr(preflight, "_ensure_lm_eval_dep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preflight, "_ensure_framework_deps", lambda *_args: None)
    monkeypatch.setattr(preflight, "_check_serving_framework", lambda *_args: None)
    monkeypatch.setattr(preflight, "_inferencex_checkout_ok", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(preflight, "_inferencex_head_sha", lambda *_args: "abc123")
    monkeypatch.setattr(preflight, "_report_inferencex_patch_anchors", lambda *_args: True)
    monkeypatch.setattr(preflight, "_check_node_claude_cli", lambda: None)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        preflight,
        "_run_ir3_preflight",
        lambda _args: {"status": "applied", "skip_reason": None},
    )
    monkeypatch.setattr(
        preflight,
        "_emit_preflight_diagnostics",
        lambda **_kwargs: {"status": "applied", "skip_reason": None},
    )
    args = argparse.Namespace(
        resume_from=None,
        degraded_kb=True,
        framework="sglang",
        no_eval=True,
        no_kernel=True,
        enable_roofline=False,
    )

    preflight._preflight(args)

    event = pending_install_event(args)
    assert event is not None
    assert [step["step_id"] for step in event["ext"]["steps"]] == [
        "load_dotenv",
        "load_kernel_agent_env",
        "normalize_legacy_deepseek_env",
        "validate_credentials",
        "prepare_kb_environment",
        "ensure_python_sdks",
        "check_gpu_visibility",
        "check_shm_disk",
        "check_platform_tuning",
        "ensure_ray",
        "ensure_bench_serving_deps",
        "ensure_lm_eval",
        "framework_deps",
        "check_serving_framework",
        "ensure_magpie",
        "clone_inferencex",
        "patch_magpie_eval_concurrency",
        "check_tracelens_cli",
        "check_tracelens_root",
        "ir3_pr_monitor_probe",
        "diagnostics_snapshot",
    ]
    steps = {step["step_id"]: step for step in event["ext"]["steps"]}
    assert steps["prepare_kb_environment"]["status"] == "skipped"
    assert steps["prepare_kb_environment"]["skip_reason"] == "explicit_flag"
    assert steps["ensure_magpie"]["message"] == "benchmark backend is 'bypass'"


def test_ir3_unreachable_uses_v6_reason_without_changing_v5_state_reason(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setattr(preflight, "_workspace_root_resolve", lambda: tmp_path)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(degraded_kb=False, degraded_pr=False, pr_monitor_url="")

    outcome = preflight._run_ir3_preflight(args)

    assert args.pr_degraded_reason == "ir3_auto"
    assert outcome["status"] == "warned"
    assert outcome["skip_reason"] == "ir3_unreachable"
    assert outcome["detail"]["pr_monitor"] == {
        "enabled": False,
        "reason": "ir3_unreachable",
    }


@pytest.mark.parametrize(
    ("scenario", "failed_gate_id", "expected_statuses"),
    [
        ("unsupported", "unsupported_model_arch", ["failed", "skipped", "skipped"]),
        ("config", "model_config_compat", ["passed", "failed", "skipped"]),
        ("context", "context_window", ["passed", "passed", "failed"]),
    ],
)
def test_each_model_gate_failure_is_written_to_final_sbd(
    tmp_path,
    monkeypatch,
    scenario,
    failed_gate_id,
    expected_statuses,
):
    from hyperloom.inference_optimizer.cli import model_gate

    monkeypatch.setattr(model_gate, "_emit_breakdown_to_langfuse", lambda _session_dir: None)
    model = tmp_path / scenario
    if scenario == "unsupported":
        _write_model_config(
            model,
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
            },
        )
    elif scenario == "config":
        _write_model_config(
            model,
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "rope_scaling": {"factor": 2.0},
            },
        )
    else:
        _write_model_config(
            model,
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 2048,
            },
        )
    session_dir = tmp_path / f"session-{scenario}"
    _seed_state(session_dir, monkeypatch, model)
    args = _gate_args(model)
    model_gate._start_model_gate(args, session_dir)

    if scenario == "unsupported":
        assert model_gate._preflight_unsupported_model_arch(args, session_dir) is True
    else:
        assert model_gate._preflight_unsupported_model_arch(args, session_dir) is False
        if scenario == "config":
            assert model_gate._preflight_model_config_compat(args, session_dir) is True
        else:
            assert model_gate._preflight_model_config_compat(args, session_dir) is False
            assert model_gate._preflight_context_window(args, session_dir) is True

    event = _model_gate_from_breakdown(session_dir)
    assert event["status"] == "failed"
    assert event["ext"]["failed_gate_id"] == failed_gate_id
    assert [check["gate_id"] for check in event["ext"]["checks"]] == [
        "unsupported_model_arch",
        "model_config_compat",
        "context_window",
    ]
    assert [check["status"] for check in event["ext"]["checks"]] == expected_statuses
    assert "breakdown_written" not in event["ext"]["failure"]["artifacts"]


def test_resume_model_gate_records_three_explicit_skips(tmp_path):
    from hyperloom.inference_optimizer.cli import model_gate

    args = _gate_args(tmp_path / "model")
    model_gate._record_resumed_model_gate(
        args,
        tmp_path,
        workload_overrides={
            "model_path": "/models/resumed",
            "model_name": "resumed-model",
            "framework": "vllm",
            "gpu_type": "MI355X",
        },
    )

    event = read_timeline_event(tmp_path, "model_gate")
    assert event is not None
    assert event["status"] == "skipped"
    assert event["ext"]["run_kind"] == "resume"
    assert event["ext"]["skip_reason"] == "resume"
    assert event["ext"]["workload"]["model_path"] == "/models/resumed"
    assert event["ext"]["workload"]["model_name"] == "resumed-model"
    assert event["ext"]["workload"]["framework"] == "vllm"
    assert event["ext"]["workload"]["gpu_type"] == "MI355X"
    assert [check["skip_reason"] for check in event["ext"]["checks"]] == [
        "resume",
        "resume",
        "resume",
    ]


def test_model_gate_event_write_failure_does_not_change_gate_result(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import model_gate
    from hyperloom.inference_optimizer.session import sbd_v6

    model = tmp_path / "healthy"
    _write_model_config(
        model,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 8192,
        },
    )
    monkeypatch.setattr(
        sbd_v6,
        "write_timeline_event_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    args = _gate_args(model)
    model_gate._start_model_gate(args, tmp_path)

    assert model_gate._preflight_unsupported_model_arch(args, tmp_path) is False
    breakdown = exporter.build(tmp_path)
    assert any("sbd_v6.write.model_gate.event" in warning for warning in breakdown["metadata"]["warnings"])


def test_model_gate_fail_fast_writes_breakdown_once(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer import breakdown
    from hyperloom.inference_optimizer.cli import model_gate

    writes: list[Path] = []
    monkeypatch.setattr(breakdown, "write_breakdown_json", lambda session_dir: writes.append(Path(session_dir)))

    model_gate._write_model_gate_breakdown(tmp_path, failure_label="test")

    assert writes == [tmp_path]


def test_model_gate_projection_failure_does_not_change_gate_result(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import model_gate

    model = tmp_path / "healthy"
    _write_model_config(
        model,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 8192,
        },
    )
    monkeypatch.setattr(
        model_gate,
        "_load_model_gate_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad V6 projection")),
    )

    assert model_gate._preflight_unsupported_model_arch(_gate_args(model), tmp_path) is False


def test_install_projection_failure_does_not_change_step_result(monkeypatch):
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setattr(
        preflight,
        "_record_install_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad V6 projection")),
    )

    assert (
        preflight._run_install_step(
            {"ext": {"steps": []}},
            step_id="unchanged",
            category="check",
            action=lambda: "original-result",
        )
        == "original-result"
    )


def test_install_event_write_failure_is_exported_as_v6_warning(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session import sbd_v6

    args = argparse.Namespace(resume_from=None, no_kernel=True, enable_roofline=False)
    preflight._begin_install_event(args)
    monkeypatch.setattr(
        sbd_v6,
        "write_timeline_event_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("install disk unavailable")),
    )

    preflight._persist_install_event(args, tmp_path)

    breakdown = exporter.build(tmp_path)
    assert any("sbd_v6.write.install.event" in warning for warning in breakdown["metadata"]["warnings"])


def test_missing_pending_install_event_is_exported_as_v6_warning(tmp_path):
    from hyperloom.inference_optimizer.cli import preflight

    preflight._persist_install_event(argparse.Namespace(), tmp_path)

    breakdown = exporter.build(tmp_path)
    assert any("pending install event is unavailable" in warning for warning in breakdown["metadata"]["warnings"])


def test_corrupt_model_gate_event_is_safely_normalized(tmp_path):
    from hyperloom.inference_optimizer.cli import model_gate

    path = tmp_path / "reports" / "sbd_v6" / "timeline" / "000001-model_gate.json"
    _write_json(
        path,
        {
            "type": "model_gate",
            "ext": {
                "checks": [{"gate_id": "legacy", "order": "invalid", "status": "unknown"}],
                "degraded": [],
            },
        },
    )
    args = _gate_args(tmp_path / "model")

    model_gate._record_model_gate_check(
        args,
        tmp_path,
        {
            "gate_id": "unsupported_model_arch",
            "order": 1,
            "status": "passed",
            "skip_reason": None,
            "detail": {},
        },
    )

    event = read_timeline_event(tmp_path, "model_gate")
    assert event is not None
    assert event["status"] == "degraded"
    assert event["ext"]["checks"][0]["gate_id"] == "unsupported_model_arch"
    assert event["ext"]["degraded"] == {"active": False, "warnings": []}


def test_fresh_model_gate_with_only_soft_skips_succeeds(tmp_path):
    from hyperloom.inference_optimizer.cli import model_gate

    args = _gate_args(tmp_path / "model")
    model_gate._start_model_gate(args, tmp_path)
    for order, gate_id in enumerate(model_gate._MODEL_GATE_ORDER, start=1):
        model_gate._record_model_gate_check(
            args,
            tmp_path,
            {
                "gate_id": gate_id,
                "order": order,
                "status": "skipped",
                "skip_reason": "soft_pass",
                "detail": {},
            },
        )
    model_gate._finish_model_gate(args, tmp_path)

    event = read_timeline_event(tmp_path, "model_gate")
    assert event is not None
    assert event["status"] == "succeeded"
    assert event["ext"]["skip_reason"] is None


def test_framework_timeline_merges_legacy_framework_and_explore(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "framework_agent_phase_done": True,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "prelude_complete",
                "ts": "2026-08-27T01:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "EXPLORE",
                "reason": "framework_agent_phase_done",
                "ts": "2026-08-27T01:10:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "EXPLORE",
                "to_phase": "KERNEL_AGENT",
                "reason": "explore_no_more_leverage",
                "ts": "2026-08-27T01:20:00+00:00",
                "cycle": 0,
                "evidence": {
                    "recent_keep_gain_pct": 5.0,
                    "keep_gain_threshold_pct": 6.0,
                    "empty_streak": 2,
                    "empty_streak_threshold": 2,
                    "lookback": 6,
                    "tested_this_cycle": 1,
                    "config_arm_plateaued": True,
                    "source_consecutive_no_keep": 1,
                    "source_threshold": 3,
                    "source_candidates_exhausted": True,
                    "source_arm_plateaued": True,
                    "switch_bottleneck": True,
                    "evidence": "both_arms_plateaued",
                },
            },
        ],
        "specialist_rounds": [
            {
                "round_id": "spec-config-1",
                "task_id": "spec-config-task",
                "domain": "serving_specialist",
                "cycle": 0,
                "completed_at": "2026-08-27T01:04:00+00:00",
                "proposal_set": [
                    {
                        "name": "chunked-prefill",
                        "fingerprint": "fp-config-1",
                    }
                ],
            }
        ],
        "explore_search": {
            "tested": {
                "fp-config-1": {
                    "fingerprint": "fp-config-1",
                    "name": "chunked-prefill",
                    "outcome": "KEEP",
                    "tput": 105.0,
                    "base_tput": 100.0,
                    "gain_pct": 5.0,
                    "round_id": "config-round-1",
                    "cycle": 0,
                    "workload_signature": "qwen-tp8-c64",
                    "framework": "sglang",
                    "stack_rebench_tput": 104.0,
                    "stack_rebench_workspace": "runs/config-round-1/rebench",
                }
            },
            "winners_history": [{"gain_pct": 5.0, "cycle": 0}],
        },
        "framework_agent_batches": [
            {
                "batch_id": "legacy-batch",
                "candidates": [
                    {
                        "pr_url": "https://example.test/pr/7",
                        "route": "direct_framework",
                        "audit": {"verdict": "worth_a_bench"},
                    }
                ],
            }
        ],
        "framework_agent_phase_progress": [
            {
                "candidate_id": "https://example.test/pr/7",
                "status": "kept",
                "kept": True,
                "pre_tput": 105.0,
                "post_tput": 108.0,
                "gain_pct": 2.857,
                "cycle": 0,
                "ts": "2026-08-27T01:08:00+00:00",
            }
        ],
    }
    operations = [
        {
            "operation_id": "op-source-1",
            "name": "framework_agent",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 0,
            "status": "succeeded",
            "ended_at": "2026-08-27T01:08:00+00:00",
            "outputs": {
                "status": "kept",
                "candidate": {
                    "pr_url": "https://example.test/pr/7",
                    "route": "direct_framework",
                    "changed_files": ["python/server.py"],
                },
                "base_tput": 105.0,
                "output_throughput": 108.0,
                "delta_pct": 2.857,
                "accuracy_pass": True,
                "keep_threshold_pct": 1.0,
                "patches_applied": ["patches/pr-7.patch"],
                "target_files": ["python/server.py"],
                "workspace": "runs/framework/pr-7",
            },
            "extensions": {"task_id": "source-task-1"},
        },
        {
            "operation_id": "op-config-1",
            "name": "explore",
            "phase": "EXPLORE",
            "macro_cycle": 0,
            "status": "succeeded",
            "ended_at": "2026-08-27T01:16:00+00:00",
            "outputs": {
                "status": "succeeded",
                "round_id": "config-round-1",
                "framework": "sglang",
                "base_tput": 100.0,
                "per_variant_outcomes": [
                    {
                        "variant_name": "chunked-prefill",
                        "outcome": "KEEP",
                        "fingerprint": "fp-config-1",
                        "provenance": "specialist:serving_specialist",
                        "scope": "domain",
                        "metrics": {"tput": 105.0, "gain_pct": 5.0},
                        "variant": {
                            "extra_server_args": "--enable-chunked-prefill",
                            "extra_envs": {"SGLANG_CHUNKED_PREFILL": "1"},
                        },
                    }
                ],
                "explore_search_update": {
                    "tested": state["explore_search"]["tested"],
                    "last_round": {
                        "round_id": "config-round-1",
                        "base_tput": 100.0,
                        "base_extra_args": "--base-flag",
                    },
                },
            },
            "extensions": {"task_id": "config-task-1"},
        },
    ]

    timeline = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)

    events = _framework_events(timeline)
    assert [event["type"] for event in events] == ["framework_agent"]
    event = events[0]
    assert event["start_time"] == "2026-08-27T01:00:00+00:00"
    assert event["end_time"] == "2026-08-27T01:20:00+00:00"
    assert "summary" not in event
    assert "stack_rebench_enabled" not in event["ext"]["policy"]
    assert event["ext"]["config_arm"]["rounds"][0]["workload_signature"] == "qwen-tp8-c64"
    assert event["ext"]["config_arm"]["rounds"][0]["input_stack"]["extra_server_args"] == "--base-flag"
    variant = event["ext"]["config_arm"]["rounds"][0]["variants"][0]
    assert "stack_rebench" not in variant
    attempt = event["ext"]["source_arm"]["attempts"][0]
    assert attempt["patch_source"] == "upstream_pr"
    assert attempt["lever_kind"] == "upstream_pr"
    assert attempt["route"] == "direct_framework"
    assert attempt["status"] == "KEEP"
    assert event["ext"]["exit"] == {
        "reason": "optimize_no_more_leverage",
        "trigger": "both_arms_plateaued",
        "hint": None,
        "switch_bottleneck": True,
    }


def test_framework_timeline_projects_pr1301_source_and_critic_data(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 2,
        "framework_agent_authoring_enabled": True,
        "framework_agent_phase_done": False,
        "phase_history": [
            {
                "from_phase": "SWEEP",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "cycle_reloop",
                "ts": "2026-08-27T02:00:00+00:00",
                "cycle": 2,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "reason": "optimize_phase_budget_exhausted",
                "ts": "2026-08-27T02:20:00+00:00",
                "cycle": 2,
                "evidence": {
                    "source_consecutive_no_keep": 0,
                    "source_threshold": 3,
                    "source_candidates_exhausted": False,
                    "source_arm_plateaued": False,
                    "recent_keep_gain_pct": 0.0,
                    "keep_gain_threshold_pct": 1.0,
                    "empty_streak": 0,
                    "empty_streak_threshold": 3,
                    "lookback": 6,
                    "tested_this_cycle": 0,
                    "config_arm_plateaued": False,
                    "switch_bottleneck": False,
                },
            },
        ],
        "specialist_rounds": [
            {
                "round_id": "discover-round",
                "task_id": "discover-task-1234",
                "domain": "candidate_discovery_specialist",
                "cycle": 2,
                "completed_at": "2026-08-27T02:04:00+00:00",
                "proposal_set": [
                    {
                        "pr_url": "https://example.test/pr/9",
                        "title": "Fuse host copies",
                        "verdict": "worth_a_bench",
                        "route": "author_via_specialist",
                    },
                    {
                        "pr_url": "https://example.test/pr/10",
                        "title": "Unrelated backend",
                        "verdict": "not_applicable",
                        "reason": "wrong framework",
                    },
                ],
            },
            {
                "round_id": "author-round",
                "task_id": "author-task-1",
                "domain": "serving_specialist",
                "cycle": 2,
                "completed_at": "2026-08-27T02:08:00+00:00",
                "proposal_set": [{"patches_written": ["patches/pr-9.patch"]}],
                "task_kind": "framework_authoring",
                "framework_agent_authoring": True,
                "framework_agent_candidate_id": "https://example.test/pr/9",
                "reauthor_attempt": 1,
            },
        ],
        "framework_agent_batches": [
            {
                "batch_id": "discovery-0-discover",
                "candidates": [
                    {
                        "pr_url": "https://example.test/pr/9",
                        "route": "author_via_specialist",
                        "audit": {"verdict": "worth_a_bench"},
                    }
                ],
            }
        ],
        "framework_agent_specialist_candidate_map": {
            "author-task-1": "https://example.test/pr/9",
        },
        "framework_agent_phase_progress": [
            {
                "candidate_id": "https://example.test/pr/9",
                "batch_id": "discovery-0-discover",
                "status": "kept",
                "kept": True,
                "gain_pct": 4.0,
                "pre_tput": 100.0,
                "post_tput": 104.0,
                "specialist_task_id": "author-task-1",
                "integrate_task_id": "integrate-task-1",
                "reauthor_attempt": 1,
                "cycle": 2,
                "ts": "2026-08-27T02:15:00+00:00",
            }
        ],
    }
    operations = [
        {
            "operation_id": "op-integrate-1",
            "name": "integrate_patch",
            "phase": "FRAMEWORK_AGENT",
            "agent": "framework_agent",
            "macro_cycle": 2,
            "status": "succeeded",
            "ended_at": "2026-08-27T02:15:00+00:00",
            "outputs": {
                "status": "kept",
                "framework_agent_authoring": True,
                "specialist_task_id": "author-task-1",
                "reauthor_attempt": 1,
                "base_tput": 100.0,
                "output_throughput": 104.0,
                "delta_pct": 4.0,
                "accuracy_pass": True,
                "keep_threshold_pct": 1.0,
                "patches_applied": ["patches/pr-9.patch"],
                "target_files": ["python/worker.py"],
                "source_snapshot": "optimization_stack/src/author-task-1",
                "source_manifest": "optimization_stack/src/author-task-1/manifest.json",
                "workspace": "runs/integrate-task-1",
                "switch_off_parity": {"ran": True, "ok": True},
                "stack_rebench": {"stable": True},
                "framework_levers": [{"switch": "SGLANG_FAST_COPY", "default_on": True}],
            },
            "extensions": {"task_id": "integrate-task-1"},
        }
    ]
    critic_dir = tmp_path / "critic-workdir" / "000000"
    _write_json(
        critic_dir / "judge_bundle.json",
        {
            "merged_context": {"macro_cycle": 2},
            "proposals": [
                {
                    "msg_id": "proposal-1",
                    "action_name": "integrate_patch",
                    "payload": {
                        "params": {
                            "framework_agent_candidate_id": "https://example.test/pr/9",
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        critic_dir / "review.json",
        {
            "review_verdicts": [
                {
                    "target_proposal_msg_id": "proposal-1",
                    "verdict": "needs_review",
                    "source": "critic",
                    "reasoning": "needs parity evidence",
                    "confidence": "high",
                    "required_evidence": ["switch-off parity"],
                    "risks": [{"severity": "major", "risk": "default behavior may change"}],
                }
            ]
        },
    )
    _write_json(
        critic_dir / "emit.json",
        {
            "intent_envelope": {
                "intents": [
                    {
                        "intent_type": "review_verdict",
                        "payload": {
                            "target_proposal_msg_id": "proposal-1",
                            "verdict": "approve",
                            "advice_text": "retain the switch-off check",
                        },
                    }
                ]
            }
        },
    )

    timeline = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)

    event = timeline[0]
    discovery = event["ext"]["source_arm"]["candidate_discovery_runs"][0]
    assert discovery["task_id"] == "discover-task-1234"
    assert [candidate["verdict"] for candidate in discovery["candidates"]] == [
        "worth_a_bench",
        "not_applicable",
    ]
    authoring = event["ext"]["source_arm"]["authoring_runs"][0]
    assert authoring["candidate_id"] == "https://example.test/pr/9"
    assert authoring["kind"] == "reauthor"
    assert authoring["reauthor_attempt"] == 1
    assert authoring["patch_refs"] == ["patches/pr-9.patch"]
    attempt = event["ext"]["source_arm"]["attempts"][0]
    assert attempt["patch_source"] == "specialist_authored"
    assert attempt["lever_kind"] == "source_patch"
    assert attempt["route"] == "author_via_specialist"
    assert attempt["status"] == "KEEP"
    assert attempt["gates"] == {
        "accuracy_passed": True,
        "keep_threshold_pct": 1.0,
        "switch_off_parity_passed": True,
    }
    review = event["ext"]["critic_reviews"][0]
    assert review["arm"] == "source"
    assert review["target_action"] == "integrate_patch"
    assert review["verdict"] == "needs_review"
    assert review["effective_verdict"] == "approve"
    assert "token" not in json.dumps(event).lower()


def test_framework_timeline_keeps_config_serving_specialist_out_of_source_arm(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "ts": "2026-08-27T03:10:00+00:00",
                "cycle": 0,
            },
        ],
        "specialist_rounds": [
            {
                "round_id": "config-round",
                "task_id": "config-task",
                "domain": "serving_specialist",
                "source_phase": "FRAMEWORK_AGENT",
                "cycle": 0,
                "proposal_set": [
                    {
                        "name": "larger-page-size",
                        "extra_server_args": "--page-size 32",
                    }
                ],
            }
        ],
    }
    critic_dir = tmp_path / "critic-workdir" / "000000"
    _write_json(
        critic_dir / "request.json",
        {
            "context": {"phase": "FRAMEWORK_AGENT"},
            "raw_prompt": "=== Shared session state ===\nmacro_cycle=0\n",
        },
    )
    _write_json(
        critic_dir / "judge_bundle.json",
        {
            "phase": "FRAMEWORK_AGENT",
            "proposals": [
                {
                    "msg_id": "config-proposal",
                    "action_name": "specialist",
                    "payload": {
                        "params": {
                            "domain": "serving_specialist",
                            "source_phase": "FRAMEWORK_AGENT",
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        critic_dir / "review.json",
        {
            "review_verdicts": [
                {
                    "target_proposal_msg_id": "config-proposal",
                    "verdict": "approve",
                }
            ]
        },
    )
    _write_json(
        critic_dir / "emit.json",
        {
            "intent_envelope": {
                "intents": [
                    {
                        "intent_type": "review_verdict",
                        "payload": {
                            "target_proposal_msg_id": "config-proposal",
                            "verdict": "approve",
                        },
                    }
                ]
            }
        },
    )

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=[])[0]

    assert [row["task_id"] for row in event["ext"]["config_arm"]["specialist_runs"]] == ["config-task"]
    assert event["ext"]["source_arm"]["authoring_runs"] == []
    assert event["ext"]["critic_reviews"][0]["arm"] == "config"


def test_framework_timeline_ignores_kernel_specialist_without_framework_evidence(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 4,
        "specialist_rounds": [
            {
                "round_id": "kernel-specialist",
                "task_id": "kernel-specialist",
                "domain": "kernel_specialist",
                "cycle": 4,
                "completed_at": "2026-08-27T04:00:00+00:00",
                "proposal_set": [{"name": "kernel-rewrite"}],
            }
        ],
    }

    operations = [
        {
            "operation_id": "op-kernel-specialist",
            "kind": "specialist",
            "name": "specialist round kernel-specialist",
            "phase": "EXPLORE",
            "agent": "explore",
            "source": "specialist_recorder_hook",
            "macro_cycle": 4,
            "status": "succeeded",
            "outputs": state["specialist_rounds"][0],
        }
    ]

    timeline = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)

    assert _framework_events(timeline) == []


def test_framework_timeline_recovers_direct_upstream_patch_source(tmp_path):
    candidate_id = "https://example.test/pr/11"
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "ts": "2026-08-27T03:10:00+00:00",
                "cycle": 0,
            },
        ],
        "framework_agent_batches": [
            {
                "batch_id": "discovery-0",
                "candidates": [
                    {
                        "pr_url": candidate_id,
                        "route": "direct_framework",
                    }
                ],
            }
        ],
        "framework_agent_phase_progress": [
            {
                "candidate_id": candidate_id,
                "integrate_task_id": "integrate-direct-1",
                "status": "kept",
                "kept": True,
                "cycle": 0,
            }
        ],
    }
    operations = [
        {
            "operation_id": "op-integrate-direct",
            "name": "integrate_patch",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 0,
            "status": "succeeded",
            "outputs": {
                "status": "kept",
                "framework_agent_authoring": True,
                "specialist_task_id": "integrate-direct-1",
            },
            "extensions": {"task_id": "integrate-direct-1"},
        }
    ]

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)[0]
    attempt = event["ext"]["source_arm"]["attempts"][0]

    assert attempt["candidate_id"] == candidate_id
    assert attempt["patch_source"] == "upstream_pr"
    assert attempt["lever_kind"] == "upstream_pr"
    assert attempt["route"] == "direct_framework"


def test_framework_timeline_keeps_macro_cycles_isolated(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 1,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "prelude_complete",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "reason": "optimize_no_more_leverage",
                "ts": "2026-08-27T03:10:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "SWEEP",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "cycle_reloop",
                "ts": "2026-08-27T04:00:00+00:00",
                "cycle": 1,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "reason": "optimize_no_more_leverage",
                "ts": "2026-08-27T04:10:00+00:00",
                "cycle": 1,
            },
        ],
    }
    operations = [
        {
            "operation_id": "cycle-0",
            "name": "explore",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 0,
            "status": "succeeded",
            "outputs": {"status": "succeeded", "round_id": "round-0"},
        },
        {
            "operation_id": "cycle-1",
            "name": "explore",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 1,
            "status": "succeeded",
            "outputs": {"status": "succeeded", "round_id": "round-1"},
        },
    ]

    timeline = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)

    events = _framework_events(timeline)
    assert [event["ext"]["macro_cycle"] for event in events] == [0, 1]
    assert [event["ext"]["config_arm"]["rounds"][0]["round_id"] for event in events] == [
        "round-0",
        "round-1",
    ]


def test_framework_timeline_excludes_kernel_phase_explore_rebench(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "ts": "2026-08-27T03:10:00+00:00",
                "cycle": 0,
            },
        ],
    }
    operations = [
        {
            "operation_id": "framework-round",
            "name": "explore",
            "phase": "FRAMEWORK_AGENT",
            "agent": "explore",
            "macro_cycle": 0,
            "status": "succeeded",
            "ended_at": "2026-08-27T03:05:00+00:00",
            "outputs": {"status": "succeeded", "round_id": "framework-round"},
        },
        {
            "operation_id": "kernel-rebench",
            "name": "explore",
            "phase": "KERNEL_AGENT",
            "agent": "explore",
            "macro_cycle": 0,
            "status": "succeeded",
            "ended_at": "2026-08-27T03:06:00+00:00",
            "outputs": {"status": "succeeded", "round_id": "kernel-rebench"},
        },
    ]

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)[0]

    assert [row["round_id"] for row in event["ext"]["config_arm"]["rounds"]] == ["framework-round"]


def test_framework_timeline_projects_discovery_history_outcomes_in_order(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "framework_agent_discover_failed",
                "evidence": {
                    "event": "framework_agent_discover_failed",
                    "attempt": 1,
                    "limit": 3,
                    "error": "TimeoutError('upstream unavailable')",
                },
                "ts": "2026-08-27T03:01:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "discover_empty_payload",
                "evidence": {
                    "event": "framework_agent_phase_done",
                    "failure_count": 0,
                    "retry_limit": 3,
                },
                "ts": "2026-08-27T03:03:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "reason": "framework_agent_phase_done",
                "ts": "2026-08-27T03:04:00+00:00",
                "cycle": 0,
            },
        ],
        "framework_agent_batches": [
            {
                "batch_id": "discovery-0",
                "ts": "2026-08-27T03:02:00+00:00",
                "cycle": 0,
                "candidates": [
                    {
                        "pr_url": "https://example.test/pr/12",
                        "route": "direct_framework",
                    }
                ],
            }
        ],
    }

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=[])[0]
    runs = event["ext"]["source_arm"]["candidate_discovery_runs"]

    assert [run["status"] for run in runs] == ["failed", "succeeded", "empty"]
    assert runs[0]["reason"] == "TimeoutError('upstream unavailable')"
    assert runs[1]["batch_id"] == "discovery-0"
    assert runs[1]["candidates"][0]["candidate_id"] == "https://example.test/pr/12"
    assert runs[2]["reason"] == "discover_empty_payload"
    assert event["status"] == "succeeded"
    assert event["ext"]["failure"] == {
        "failed_task_id": None,
        "error_class": None,
        "error": None,
    }


def test_framework_timeline_marks_exhausted_discovery_retries_failed(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "framework_agent_discover_failed",
                "evidence": {
                    "event": "framework_agent_discover_failed",
                    "attempt": 1,
                    "limit": 3,
                    "error": "TimeoutError('first')",
                },
                "ts": "2026-08-27T03:01:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "framework_agent_discover_failed",
                "evidence": {
                    "event": "framework_agent_discover_failed",
                    "attempt": 3,
                    "limit": 3,
                    "error": "TimeoutError('last')",
                },
                "ts": "2026-08-27T03:02:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "FRAMEWORK_AGENT",
                "reason": "discover_retries_exhausted",
                "evidence": {
                    "event": "framework_agent_phase_done",
                    "failure_count": 3,
                    "retry_limit": 3,
                },
                "ts": "2026-08-27T03:03:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "reason": "framework_agent_phase_done",
                "ts": "2026-08-27T03:04:00+00:00",
                "cycle": 0,
            },
        ],
    }

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=[])[0]
    runs = event["ext"]["source_arm"]["candidate_discovery_runs"]

    assert [run["status"] for run in runs] == ["failed", "failed"]
    assert event["status"] == "failed"
    assert event["ext"]["failure"] == {
        "failed_task_id": None,
        "error_class": "candidate_discovery_failed",
        "error": "TimeoutError('last')",
    }


def test_framework_timeline_assigns_critic_reviews_from_request_cycle(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 1,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T03:00:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "ts": "2026-08-27T03:10:00+00:00",
                "cycle": 0,
            },
            {
                "from_phase": "SWEEP",
                "to_phase": "FRAMEWORK_AGENT",
                "ts": "2026-08-27T04:00:00+00:00",
                "cycle": 1,
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "ts": "2026-08-27T04:10:00+00:00",
                "cycle": 1,
            },
        ],
    }
    for cycle in (0, 1):
        proposal_id = f"proposal-cycle-{cycle}"
        critic_dir = tmp_path / "critic-workdir" / f"{cycle:06d}"
        _write_json(
            critic_dir / "request.json",
            {
                "context": {"phase": "FRAMEWORK_AGENT", "macro_cycle": cycle},
                "raw_prompt": "=== Shared session state ===\nmacro_cycle=99\n",
            },
        )
        _write_json(
            critic_dir / "judge_bundle.json",
            {
                "phase": "FRAMEWORK_AGENT",
                "proposals": [
                    {
                        "msg_id": proposal_id,
                        "action_name": "integrate_patch",
                        "payload": {
                            "framework_agent_candidate_id": f"candidate-{cycle}",
                        },
                    }
                ],
            },
        )
        _write_json(
            critic_dir / "review.json",
            {
                "review_verdicts": [
                    {
                        "target_proposal_msg_id": proposal_id,
                        "verdict": "approve",
                    }
                ]
            },
        )
        _write_json(
            critic_dir / "emit.json",
            {
                "intent_envelope": {
                    "intents": [
                        {
                            "intent_type": "review_verdict",
                            "payload": {
                                "target_proposal_msg_id": proposal_id,
                                "verdict": "approve",
                            },
                        }
                    ]
                }
            },
        )

    _write_json(
        tmp_path / "reports" / "trace" / "proposal_task_map.jsonl",
        {
            "proposal_msg_id": "proposal-cycle-1",
            "task_id": "integrate-cycle-1",
        },
    )
    operations = [
        {
            "operation_id": "op-cycle-1",
            "name": "integrate_patch",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 1,
            "extensions": {"task_id": "integrate-cycle-1"},
            "outputs": {"status": "reverted"},
        }
    ]

    timeline = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)

    assert [
        [review["proposal_msg_id"] for review in event["ext"]["critic_reviews"]]
        for event in _framework_events(timeline)
    ] == [
        ["proposal-cycle-0"],
        ["proposal-cycle-1"],
    ]


def test_framework_review_does_not_parse_macro_cycle_from_prompt():
    reviews = normalize_framework_reviews(
        request={
            "context": {"phase": "FRAMEWORK_AGENT"},
            "raw_prompt": "=== Shared session state ===\nmacro_cycle=7\n",
        },
        judge_bundle={
            "phase": "FRAMEWORK_AGENT",
            "proposals": [
                {
                    "msg_id": "proposal-1",
                    "action_name": "integrate_patch",
                    "payload": {"framework_agent_candidate_id": "candidate-1"},
                }
            ],
        },
        review={
            "review_verdicts": [
                {
                    "target_proposal_msg_id": "proposal-1",
                    "verdict": "approve",
                }
            ]
        },
        emit={"intent_envelope": {"intents": []}},
        review_path=None,
    )

    assert reviews[0]["macro_cycle"] is None


def test_specialist_recorder_preserves_runtime_phase_when_entry_has_no_source_phase(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.breakdown.recorder import instrument

    captured: dict[str, dict] = {}

    class Recorder:
        def record_item(self, stream, item, *, key=None):
            captured["ledger"] = {"stream": stream, "item": item, "key": key}

    monkeypatch.setattr(instrument, "_recorder", lambda *_args, **_kwargs: Recorder())
    monkeypatch.setattr(instrument, "record_subject", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        instrument,
        "record_operation",
        lambda *_args, **kwargs: captured.setdefault("operation", kwargs),
    )
    monkeypatch.setattr(instrument, "record_trace_event", lambda *_args, **_kwargs: None)

    instrument.record_specialist_round(
        tmp_path,
        {
            "round_id": "kernel-specialist",
            "task_id": "kernel-task",
            "completed_at": "2026-08-28T01:00:00+00:00",
        },
        phase="KERNEL_AGENT",
    )

    assert captured["ledger"]["item"]["source_phase"] == "KERNEL_AGENT"
    assert captured["operation"]["phase"] == "KERNEL_AGENT"
    assert captured["operation"]["outputs"]["source_phase"] == "KERNEL_AGENT"


def test_framework_timeline_treats_empty_discovery_as_executed_work(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 1,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "cycle": 1,
                "ts": "2026-08-28T01:00:00+00:00",
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "cycle": 1,
                "ts": "2026-08-28T01:05:00+00:00",
                "reason": "optimize_no_more_leverage",
            },
        ],
        "specialist_rounds": [
            {
                "round_id": "discovery-empty",
                "task_id": "discovery-task",
                "source_phase": "FRAMEWORK_AGENT",
                "cycle": 1,
                "task_kind": "candidate_discovery",
                "domain": "candidate_discovery_specialist",
                "proposal_set": [],
                "empty": True,
                "completed_at": "2026-08-28T01:03:00+00:00",
            }
        ],
    }

    events = _framework_events(collect_v6_timeline(tmp_path, [], state=state, recorded_operations=[]))

    assert len(events) == 1
    assert events[0]["status"] == "succeeded"
    assert events[0]["ext"]["source_arm"]["candidate_discovery_runs"] == [
        {
            "task_id": "discovery-task",
            "status": "empty",
            "batch_id": None,
            "gap_canonical_id": None,
            "reason": None,
            "candidates": [],
        }
    ]


def test_framework_timeline_does_not_copy_final_progress_into_earlier_retry(tmp_path):
    state = {
        "phase": "KERNEL_AGENT",
        "macro_cycle": 1,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "cycle": 1,
                "ts": "2026-08-28T01:00:00+00:00",
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "cycle": 1,
                "ts": "2026-08-28T01:10:00+00:00",
            },
        ],
        "framework_agent_phase_progress": [
            {
                "candidate_id": "candidate-1",
                "integrate_task_id": "integrate-2",
                "status": "kept",
                "kept": True,
                "pre_tput": 100.0,
                "post_tput": 120.0,
                "gain_pct": 20.0,
                "cycle": 1,
                "ts": "2026-08-28T01:08:00+00:00",
            }
        ],
    }
    operations = [
        {
            "operation_id": "operation-1",
            "name": "integrate_patch",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 1,
            "extensions": {"task_id": "integrate-1"},
            "outputs": {
                "framework_agent_candidate_id": "candidate-1",
                "specialist_task_id": "specialist-1",
                "status": "apply_failed",
            },
            "ended_at": "2026-08-28T01:04:00+00:00",
        },
        {
            "operation_id": "operation-2",
            "name": "integrate_patch",
            "phase": "FRAMEWORK_AGENT",
            "macro_cycle": 1,
            "extensions": {"task_id": "integrate-2"},
            "outputs": {
                "framework_agent_candidate_id": "candidate-1",
                "specialist_task_id": "specialist-1",
                "status": "kept",
            },
            "ended_at": "2026-08-28T01:08:00+00:00",
        },
    ]

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=operations)[0]
    attempts = event["ext"]["source_arm"]["attempts"]

    assert attempts[0]["status"] == "FAILED"
    assert attempts[0]["before_tput"] is None
    assert attempts[0]["after_tput"] is None
    assert attempts[0]["local_gain_pct"] is None
    assert attempts[1]["status"] == "KEEP"
    assert attempts[1]["before_tput"] == 100.0
    assert attempts[1]["after_tput"] == 120.0
    assert attempts[1]["local_gain_pct"] == 20.0


def test_failed_discovery_uses_task_params_and_actual_terminal_reason(tmp_path):
    from hyperloom.orchestrator.phases.explore import ExplorePhase

    phase = object.__new__(ExplorePhase)
    phase.shared_state = SimpleNamespace(phase="KERNEL_AGENT")
    task = SimpleNamespace(
        task_id="discovery-task",
        params={
            "source_phase": "FRAMEWORK_AGENT",
            "domain": "candidate_discovery_specialist",
            "task_kind": "candidate_discovery",
            "candidate_discovery": True,
            "gap_canonical_id": "gap.framework.candidate_discovery.sglang",
        },
    )
    entry = phase._build_specialist_round_entry(
        task=task,
        done_payload={},
        source="specialist:discovery-task",
        run_error="TimeoutError('upstream unavailable')",
    )
    state = {
        "phase": "FRAMEWORK_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "cycle": 0,
                "ts": "2026-08-28T00:00:00+00:00",
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "FRAMEWORK_AGENT",
                "cycle": 0,
                "reason": "no_candidates_and_discovery_exhausted",
                "evidence": {
                    "event": "framework_agent_phase_done",
                    "failure_count": 1,
                    "retry_limit": 3,
                },
                "ts": "2026-08-28T00:05:00+00:00",
            },
        ],
        "specialist_rounds": [entry],
    }

    event = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=[])[0]

    assert entry["domain"] == "candidate_discovery_specialist"
    assert entry["task_kind"] == "candidate_discovery"
    assert entry["candidate_discovery"] is True
    assert entry["status"] == "failed"
    assert entry["run_error"] == "TimeoutError('upstream unavailable')"
    assert event["ext"]["source_arm"]["candidate_discovery_runs"] == [
        {
            "task_id": "discovery-task",
            "status": "failed",
            "batch_id": None,
            "gap_canonical_id": "gap.framework.candidate_discovery.sglang",
            "reason": "TimeoutError('upstream unavailable')",
            "candidates": [],
        }
    ]
    assert event["status"] == "failed"
    assert event["ext"]["failure"] == {
        "failed_task_id": "discovery-task",
        "error_class": "candidate_discovery_failed",
        "error": "TimeoutError('upstream unavailable')",
    }


def test_framework_critic_reviews_survive_pruning_and_reused_iteration_number(tmp_path):
    from hyperloom.inference_optimizer.breakdown.recorder import instrument
    from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts

    state = {
        "session_id": "durable-critic",
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "phase_history": [
            {
                "from_phase": "PRELUDE",
                "to_phase": "FRAMEWORK_AGENT",
                "cycle": 0,
                "ts": "2026-08-28T01:00:00+00:00",
            },
            {
                "from_phase": "FRAMEWORK_AGENT",
                "to_phase": "KERNEL_AGENT",
                "cycle": 0,
                "ts": "2026-08-28T01:10:00+00:00",
            },
        ],
    }
    _write_json(tmp_path / "state.json", state)
    _write_json(tmp_path / "manifest.json", {"session_id": "durable-critic"})
    workdir = tmp_path / "critic-workdir" / "000000"

    for index in (1, 2):
        proposal_id = f"proposal-{index}"
        timestamp = f"2026-08-28T01:0{index}:00+00:00"
        request = {"context": {"phase": "FRAMEWORK_AGENT", "macro_cycle": 0}}
        judge_bundle = {
            "phase": "FRAMEWORK_AGENT",
            "proposals": [
                {
                    "msg_id": proposal_id,
                    "action_name": "integrate_patch",
                    "payload": {"framework_agent_candidate_id": f"candidate-{index}"},
                }
            ],
        }
        review = {
            "ts": timestamp,
            "review_verdicts": [
                {
                    "target_proposal_msg_id": proposal_id,
                    "verdict": "approve",
                    "reasoning": f"review {index}",
                }
            ],
        }
        emit = {
            "ts": timestamp,
            "intent_envelope": {
                "intents": [
                    {
                        "intent_type": "review_verdict",
                        "payload": {
                            "target_proposal_msg_id": proposal_id,
                            "verdict": "approve",
                        },
                    }
                ]
            },
        }
        for name, payload in (
            ("request", request),
            ("judge_bundle", judge_bundle),
            ("review", review),
            ("emit", emit),
        ):
            _write_json(workdir / f"{name}.json", payload)
        instrument.record_critic_iteration(
            tmp_path,
            iter_n=0,
            request=request,
            judge_bundle=judge_bundle,
            review=review,
            emit=emit,
            workdir=workdir,
        )

    assembled = assemble_parts(tmp_path)
    critic_iterations = assembled["critic_robustness"]["critic_iterations"]
    assert len(critic_iterations) == 2
    assert len({row["iteration_id"] for row in critic_iterations}) == 2
    assert all(row["phase"] == "FRAMEWORK_AGENT" for row in critic_iterations)
    assert all(row["macro_cycle"] == 0 for row in critic_iterations)

    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=state,
        recorded_operations=assembled.get("operations", []),
        critic_iterations=critic_iterations,
    )
    assert [row["proposal_msg_id"] for row in timeline[0]["ext"]["critic_reviews"]] == [
        "proposal-1",
        "proposal-2",
    ]

    shutil.rmtree(tmp_path / "critic-workdir")
    breakdown = exporter.build(tmp_path)
    reviews = breakdown["timeline"][0]["ext"]["critic_reviews"]
    assert [row["proposal_msg_id"] for row in reviews] == ["proposal-1", "proposal-2"]
    assert all("\\" not in row["review_path"] for row in reviews)
