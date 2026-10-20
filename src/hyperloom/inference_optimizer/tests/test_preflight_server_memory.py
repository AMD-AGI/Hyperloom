# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Explicit server limits are checked before model resolution."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom.inference_optimizer.cli import preflight
from hyperloom.inference_optimizer.cli.bootstrap import parse_operator_extra_env
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events


class PreflightReachedCredentials(Exception):
    """Stop successful input checks before any external operation."""


@pytest.fixture
def isolated_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "sessions"))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)
    for name in ("_load_dotenv_fallback", "_load_kernel_agent_env_fallback", "_derive_runtime_paths"):
        monkeypatch.setattr(preflight, name, lambda: None)
    monkeypatch.setattr(preflight, "_normalize_legacy_deepseek_env", dict)
    monkeypatch.setattr(preflight, "_expand_custom_header_env_refs", lambda: None)
    monkeypatch.setattr(preflight, "_validate_credentials", Mock(side_effect=PreflightReachedCredentials))
    monkeypatch.setattr(
        cli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )
    model_resolution = Mock()
    coordinator = Mock()
    monkeypatch.setattr(cli, "_resolve_models_for_run", model_resolution)
    monkeypatch.setattr(cli, "Coordinator", coordinator)
    with patch.dict(os.environ):
        yield model_resolution, coordinator


@pytest.mark.parametrize(
    "server_args",
    [
        "--gpu-memory-utilization=1.5",
        "--gpu-memory-utilization=0",
        "--gpu-memory-utilization=nan",
        "--served-model-name 'a b' --gpu-memory-utilization '1.5'",
        "--gpu-memory-utilization=0.90 --gpu-memory-utilization 1.5",
    ],
)
def test_invalid_memory_stops_before_models_and_records_failure(tmp_path, isolated_preflight, server_args):
    args = cli._build_parser().parse_args(
        ["optimize", "--model", str(tmp_path / "model"), "--framework", "vllm", f"--server-args={server_args}"]
    )
    with pytest.raises(ValueError, match="gpu-memory-utilization.*0 < value <= 1") as exc:
        asyncio.run(cli._run_optimize(args))
    for downstream in isolated_preflight:
        downstream.assert_not_called()
    preflight._validate_credentials.assert_not_called()
    session_dir = Path(os.environ[ENV_CURRENT_SESSION_DIR])
    event = next(row for row in read_timeline_events(session_dir) if row["type"] == "install")
    assert event["status"] == "failed"
    assert event["ext"]["hard_fail_step_id"] == "validate_server_args"
    assert event["ext"]["steps"][-1]["message"] == str(exc.value)
    breakdown = json.loads((session_dir / "session_breakdown.json").read_text())
    assert breakdown["outcome"]["status"] == "failed"


@pytest.mark.parametrize(
    "framework,server_args",
    [
        ("vllm", "--gpu-memory-utilization 0.90"),
        ("vllm", "--gpu-memory-utilization=1"),
        ("vllm", "--served-model-name 'a b' --gpu-memory-utilization '0.90'"),
        ("vllm", "--gpu-memory-utilization 1.5 --gpu-memory-utilization=0.90"),
        ("vllm", '--hf-overrides {"note": "--gpu-memory-utilization 1.5"}'),
        ("vllm", "--max-model-len 8193"),
        ("vllm", ""),
        ("sglang", "--gpu-memory-utilization 1.5"),
    ],
)
def test_accepted_arguments_reach_existing_preflight(isolated_preflight, framework, server_args):
    args = cli._build_parser().parse_args(
        ["optimize", "--model", "model", "--framework", framework, f"--server-args={server_args}"]
    )
    with pytest.raises(PreflightReachedCredentials):
        preflight._preflight(args)
    preflight._validate_credentials.assert_called_once()


def test_resume_uses_pinned_framework_for_explicit_memory(tmp_path, monkeypatch, isolated_preflight):
    session_dir = tmp_path / "existing-session"
    session_dir.mkdir()
    (session_dir / "state.json").write_text(json.dumps({"framework": "vllm"}))
    monkeypatch.delenv("FRAMEWORK", raising=False)
    args = cli._build_parser().parse_args(
        ["optimize", "--resume-from", str(session_dir), "--server-args=--gpu-memory-utilization 1.5"]
    )
    with pytest.raises(ValueError, match="1.5"):
        preflight._preflight(args)
    assert args.framework == "vllm"
    preflight._validate_credentials.assert_not_called()


@pytest.mark.parametrize(
    "argv,env,invalid",
    [
        (["--max-model-len", "8193"], "", True),
        ([], "8193", True),
        (["--server-args=--max-model-len 8193"], "", True),
        (["--server-args=--max-model-len 9K"], "", True),
        (["--max-model-len", "8192"], "8193", False),
        ([], "8192", False),
        (["--server-args=--max-model-len 8193 --max-model-len=8K"], "", False),
        (["--server-args=--max-model-len 8K --max-model-len=8193"], "", True),
        (["--server-args=--max-model-len 8.192k"], "", False),
        (["--server-args=--max-model-len auto"], "", False),
        (["--server-args=--max-model-len -1"], "", False),
    ],
)
def test_explicit_context_limits(tmp_path, monkeypatch, isolated_preflight, argv, env, invalid):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    monkeypatch.setenv("MAX_MODEL_LEN", env)
    args = cli._build_parser().parse_args(["optimize", "--model", str(model), "--framework", "vllm", *argv])
    if invalid:
        with pytest.raises(ValueError, match="max-model-len|MAX_MODEL_LEN") as exc:
            preflight._preflight(args)
        assert "8192" in str(exc.value)
        preflight._validate_credentials.assert_not_called()
    else:
        with pytest.raises(PreflightReachedCredentials):
            preflight._preflight(args)
        preflight._validate_credentials.assert_called_once()


def test_resume_context_limit_uses_recorded_model(tmp_path, isolated_preflight):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 8192}}))
    session_dir = tmp_path / "existing-session"
    session_dir.mkdir()
    (session_dir / "state.json").write_text(json.dumps({"framework": "vllm", "model_path": str(model)}))
    args = cli._build_parser().parse_args(["optimize", "--resume-from", str(session_dir), "--max-model-len", "8193"])
    with pytest.raises(ValueError, match="max-model-len.*8192"):
        preflight._preflight(args)
    preflight._validate_credentials.assert_not_called()


@pytest.mark.parametrize(
    "recorded_args,recorded_pins,argv",
    [
        ("", {}, []),
        ("--max-model-len 8193", {}, []),
        ("--max-model-len 8193", {}, ["--server-args=--max-model-len 8192"]),
        ("", {}, ["--extra-env", "MAX_MODEL_LEN=8193"]),
        ("", {"MAX_MODEL_LEN": "8193"}, []),
        ("", {"MAX_MODEL_LEN": "8193"}, ["--extra-env", "OTHER=1"]),
        ("", {"HL_MODEL_BASE": "recorded"}, []),
        ("", {"HL_MODEL_BASE": "recorded"}, ["--extra-env", "OTHER=1"]),
        ("", {}, ["--extra-env", "HL_MODEL_BASE="]),
    ],
)
def test_resume_limit_inputs_match_restoration(tmp_path, monkeypatch, recorded_args, recorded_pins, argv):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    for base in ("shell", "recorded"):
        (tmp_path / base / "model").mkdir(parents=True)
    recorded_pins = {
        key: str(tmp_path / value) if key == "HL_MODEL_BASE" else value for key, value in recorded_pins.items()
    }
    state = {"model_path": "org/model", "max_model_len": 8192, "operator_server_args": recorded_args}
    state["operator_extra_env"] = recorded_pins
    (session_dir / "state.json").write_text(json.dumps(state))
    args = cli._build_parser().parse_args(["optimize", "--resume-from", str(session_dir), *argv])
    captured = Mock()
    monkeypatch.setattr(preflight, "validate_server_args", captured)
    with patch.dict(os.environ, MAX_MODEL_LEN="8193", HL_MODEL_BASE=str(tmp_path / "shell")):
        original_env = dict(os.environ)
        original_args = vars(args).copy()
        preflight._validate_explicit_server_limits(args)
        assert dict(os.environ) == original_env
        assert vars(args) == original_args
        effective = captured.call_args.args[0]
        pins = parse_operator_extra_env(args) or recorded_pins
        cli._export_operator_launch_shape(server_args=args.server_args or recorded_args, extra_env=pins)
        cli._seed_env_from_resumed_state(SimpleNamespace(**state), pins)
        assert str(effective.max_model_len) == os.environ["MAX_MODEL_LEN"]
        assert effective.server_args == os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "")
        assert captured.call_args.args[1] == resolve_serving_model_path(state["model_path"])


def test_context_limit_uses_serving_model_base(tmp_path, monkeypatch, isolated_preflight):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path))
    args = cli._build_parser().parse_args(
        ["optimize", "--model", "org/model", "--framework", "vllm", "--max-model-len", "8193"]
    )
    with pytest.raises(ValueError, match="max-model-len.*8192"):
        preflight._preflight(args)
    preflight._validate_credentials.assert_not_called()
