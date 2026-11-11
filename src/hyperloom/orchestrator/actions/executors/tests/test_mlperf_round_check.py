# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An MLPerf point is refused before its round unless the session froze the base its server serves and the harness
can run."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import preflight
from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.actions.executors import benchmark_config as ms
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.tests._session_base import freeze_session_base


def _point(path: Path, script: str = "mlperf_agentic_client.sh") -> Path:
    document = {"benchmark": {"framework": "sglang", "model": "/m", "benchmark_script": script, "envs": {}}}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture
def checked(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(preflight, "check_mlperf_harness", lambda env: calls.append(dict(env)))
    return calls


@pytest.fixture
def session(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(root))
    freeze_session_base(root)
    return root


def test_an_mlperf_point_of_a_frozen_session_preflights_its_harness(session, tmp_path, checked):
    ms.check_mlperf_round(_point(tmp_path / "config.yaml"), {"MLPERF_ENDPOINTS_DIR": "/e"})

    assert checked == [{"MLPERF_ENDPOINTS_DIR": "/e"}]


def test_a_point_on_another_script_is_not_checked(tmp_path, checked):
    ms.check_mlperf_round(_point(tmp_path / "config.yaml", "benchmark_serving.sh"), {})

    assert checked == []


def test_an_mlperf_point_of_a_session_that_froze_no_base_is_refused(tmp_path, monkeypatch, checked):
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(tmp_path))
    SharedState(session_id="old").save(tmp_path)

    with pytest.raises(AgentXPreflightError, match="froze none; start a new session"):
        ms.check_mlperf_round(_point(tmp_path / "config.yaml"), {})
    assert checked == []
