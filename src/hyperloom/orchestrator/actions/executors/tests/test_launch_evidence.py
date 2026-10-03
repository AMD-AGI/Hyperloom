# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Observed launch evidence retains its actual-log and process boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shlex

import pytest


def test_nearby_command_artifact_does_not_supply_launch_evidence(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    log = tmp_path / "server.log"
    log.write_text("INFO Engine ready\n")
    flags = ["--speculative-config", '{"nested": {"mode": "real verifier"}}', "--attention-backend", "aiter"]
    command = tmp_path / "vllm_command.txt"
    command.write_text(shlex.join(["vllm", "serve", "/models/accepted", "--port", "8000", *flags]) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text("benchmark:\n  framework: vllm\n  envs:\n    VLLM_USE_AITER: '1'\n")
    result = build_launch_evidence(config_path=config, actual_server_log=str(log), framework="vllm", slot=tmp_path)
    assert result["observed_server_launch_tokens"] == []
    assert result["observed_model_binding"] == {}
    assert result["server_launch_argv_complete"] is False
    assert result["requested_server_env"] == {"VLLM_USE_AITER": "1"}


@pytest.mark.parametrize(
    "line",
    [
        "INFO example only: vllm serve /models/current --attention-backend partial\n",
        "vllm serve /models/current --attention-backend 'unterminated\n",
    ],
)
def test_raw_log_example_and_stale_sibling_cannot_attest_complete_launch(tmp_path: Path, line: str) -> None:
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config, log = tmp_path / "config.yaml", tmp_path / "server.log"
    config.write_text("benchmark:\n  framework: vllm\n")
    log.write_text(line)
    assert not build_launch_evidence(config_path=config, actual_server_log=str(log), framework="vllm", slot=tmp_path)[
        "server_launch_argv_complete"
    ]
    (tmp_path / "server_command.txt").write_text("vllm serve /models/current --attention-backend stale\n")
    log.write_text("vllm serve /models/current --attention-backend current\n")
    assert not build_launch_evidence(config_path=config, actual_server_log=str(log), framework="vllm", slot=tmp_path)[
        "server_launch_argv_complete"
    ]


@pytest.mark.parametrize("sibling_name", ["vllm_command.txt", "server_command.txt"])
def test_stale_sibling_cannot_change_legacy_observed_flags_or_model(tmp_path: Path, sibling_name: str) -> None:
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config, log = tmp_path / "config.yaml", tmp_path / "server.log"
    config.write_text("benchmark:\n  framework: vllm\n  model: /models/declared\n")
    log.write_text("vllm serve /models/current --attention-backend current\n")
    (tmp_path / sibling_name).write_text("vllm serve /models/stale --attention-backend stale\n")
    observed = build_launch_evidence(config_path=config, actual_server_log=str(log), framework="vllm", slot=tmp_path)
    assert shlex.split(observed["observed_server_launch_flags"]) == ["--attention-backend", "current"]
    assert (
        observed["observed_model_binding"]["model_digest"] == "sha256:" + hashlib.sha256(b"/models/current").hexdigest()
    )
    assert observed["observed_model_binding"]["model_digest"] != observed["requested_model_digest"]
    assert observed["server_launch_argv_complete"] is False
