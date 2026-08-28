"""Focused coverage for the additive SBD V6 bootstrap fields and stages."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.schema import SCHEMA_VERSION_V5
from hyperloom.inference_optimizer.session.sbd_v6 import (
    SCHEMA_VERSION_V6,
    read_timeline_event,
    read_timeline_events,
    write_timeline_event,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    assert breakdown["schema_version"] == SCHEMA_VERSION_V5
    assert breakdown["metadata"]["versions"]["schema_version"] == SCHEMA_VERSION_V6
    assert breakdown["outcome"]["status"] == "failed"
    assert breakdown["outcome"]["stage_reached"] == "model_gate"
    return next(event for event in breakdown["timeline"] if event["type"] == "model_gate")


def test_v6_projection_is_additive_to_v5_breakdown(tmp_path):
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
    write_timeline_event(
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
    write_timeline_event(
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

    assert after["schema_version"] == SCHEMA_VERSION_V5
    v6_keys = {"metadata", "outcome", "timeline", "close"}
    assert {key: value for key, value in after.items() if key not in v6_keys} == {
        key: value for key, value in before.items() if key not in v6_keys
    }
    assert after["metadata"]["versions"]["schema_version"] == SCHEMA_VERSION_V6
    assert after["metadata"]["versions"]["hyperloom"] == "abc1234"
    assert after["metadata"]["task_config"]["launch_env"] == {"TP": "8"}
    assert after["outcome"]["status"] == "completed"
    assert after["outcome"]["stage_reached"] == "close"
    assert "token_usage" not in after["outcome"]
    assert [event["type"] for event in after["timeline"]] == ["install", "model_gate"]
    assert after["close"] == {}
    assert all(event["type"] != "close" for event in after["timeline"])


def test_invalid_v6_event_does_not_change_v5_warnings(tmp_path):
    before = exporter.build(tmp_path)
    path = tmp_path / "reports" / "sbd_v6" / "install.json"
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

    assert not (tmp_path / "reports" / "sbd_v6" / "install.json").exists()
    preflight._persist_install_event(args, tmp_path)

    persisted = read_timeline_event(tmp_path, "install")
    assert persisted is not None
    assert persisted["status"] == "degraded"
    assert [step["step_id"] for step in persisted["ext"]["steps"]] == [
        "load_dotenv",
        "check_shm_disk",
    ]
    assert persisted["ext"]["runtime_snapshot"]["provider_mode"] == "openai"


def test_preflight_hard_failure_creates_session_and_final_sbd(tmp_path, monkeypatch):
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
            raise SystemExit(2)

        preflight._run_install_step(
            event,
            step_id="validate_credentials",
            category="check",
            action=reject_credentials,
        )

    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--model", str(model)])

    with pytest.raises(SystemExit) as exc:
        asyncio.run(optimizer_cli._run_optimize(args))

    assert exc.value.code == 2
    session_dir = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    assert session_dir.is_relative_to(workspace)
    install = read_timeline_event(session_dir, "install")
    assert install is not None
    assert install["status"] == "failed"
    assert install["ext"]["hard_fail_step_id"] == "validate_credentials"
    assert (session_dir / "manifest.json").is_file()
    breakdown = json.loads((session_dir / "session_breakdown.json").read_text(encoding="utf-8"))
    assert [(event["type"], event["status"]) for event in breakdown["timeline"]] == [("install", "failed")]


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


def test_busy_resume_preflight_failure_uses_isolated_session(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.cli as optimizer_cli
    from hyperloom.inference_optimizer.cli import preflight
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    workspace = tmp_path / "sessions"
    resume_dir = workspace / "Qwen-Test" / "active-session"
    original_install = {
        "type": "install",
        "kind": "install",
        "status": "succeeded",
        "start_time": "2026-08-27T01:00:00+00:00",
        "end_time": "2026-08-27T01:01:00+00:00",
        "ext": {"run_kind": "fresh", "steps": []},
    }
    _write_json(resume_dir / "reports" / "sbd_v6" / "install.json", original_install)
    _write_json(resume_dir / "session_breakdown.json", {"sentinel": "active"})
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.setattr(
        optimizer_cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )

    class FakeSessionLock:
        def __init__(self, session_dir):
            self.session_dir = Path(session_dir)

        def acquire(self):
            if self.session_dir == resume_dir:
                raise optimizer_cli.SessionAlreadyRunning(resume_dir, {"pid": 123})
            return self

        def release(self):
            return None

    def fail_preflight(args):
        preflight._begin_install_event(args)
        raise RuntimeError("resume preflight failed")

    monkeypatch.setattr(optimizer_cli, "SessionLock", FakeSessionLock)
    monkeypatch.setattr(optimizer_cli, "_preflight", fail_preflight)
    args = optimizer_cli._build_parser().parse_args(["optimize", "--resume-from", str(resume_dir)])

    with pytest.raises(RuntimeError, match="resume preflight failed"):
        asyncio.run(optimizer_cli._run_optimize(args))

    assert json.loads((resume_dir / "reports" / "sbd_v6" / "install.json").read_text(encoding="utf-8")) == (
        original_install
    )
    assert json.loads((resume_dir / "session_breakdown.json").read_text(encoding="utf-8")) == {"sentinel": "active"}
    failed_session = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    assert failed_session != resume_dir
    install = read_timeline_event(failed_session, "install")
    assert install is not None
    assert install["status"] == "failed"
    assert install["ext"]["run_kind"] == "resume"


def test_timeline_history_retains_fresh_and_resume_events(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import model_gate

    write_timeline_event(
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
    write_timeline_event(
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


def test_timeline_history_bootstraps_legacy_fixed_events(tmp_path):
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

    write_timeline_event(
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
        ("install", "fresh"),
        ("model_gate", "fresh"),
        ("install", "resume"),
    ]


def test_timeline_history_recovers_after_partial_legacy_migration(tmp_path):
    fresh_install = {
        "type": "install",
        "kind": "install",
        "status": "succeeded",
        "start_time": "2026-08-27T01:00:00+00:00",
        "end_time": "2026-08-27T01:01:00+00:00",
        "ext": {"run_kind": "fresh", "steps": []},
    }
    _write_json(tmp_path / "reports" / "sbd_v6" / "install.json", fresh_install)
    _write_json(tmp_path / "reports" / "sbd_v6" / "timeline" / "000001-install.json", fresh_install)
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

    write_timeline_event(
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
        ("install", "fresh"),
        ("model_gate", "fresh"),
        ("install", "resume"),
    ]


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
    assert event["ext"]["failure"]["artifacts"]["breakdown_written"] is True


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
        "write_timeline_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    args = _gate_args(model)
    model_gate._start_model_gate(args, tmp_path)

    assert model_gate._preflight_unsupported_model_arch(args, tmp_path) is False


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


def test_corrupt_model_gate_event_is_safely_normalized(tmp_path):
    from hyperloom.inference_optimizer.cli import model_gate

    path = tmp_path / "reports" / "sbd_v6" / "model_gate.json"
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
