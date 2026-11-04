# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""state.json stays owner-only and unmasked; the retest runner reads a separate, scrubbed, group-readable view."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.state.retest_view import RETEST_VIEW
from hyperloom.orchestrator.state.shared_state import SharedState

_SERVER_LOG = (
    "[2026-09-17 00:00:00] server_args=ServerArgs(model_path='/models/m', tp_size=8, api_key='sekret_key_789', "
    "admin_api_key='adm_key_456', max_total_tokens=120000)\n"
)


def _state_with_real_evidence(tmp_path: Path) -> SharedState:
    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    log = tmp_path / "server.log"
    log.write_text(_SERVER_LOG, encoding="utf-8")
    evidence = build_launch_evidence(
        RequestedLaunch(
            config,
            "sglang",
            server_args="--tp 8 --max-total-tokens 120000 --api-key sekret_key_789",
            server_env={"HF_TOKEN": "hf_live_value_123"},
            model_path="/models/m",
        ),
        actual_server_log=str(log),
        slot=tmp_path / "slot",
    )
    state = SharedState()
    state.framework = "sglang"
    state.claw_session_id = "sid-1"
    state.baseline_tput = 100.0
    state.operator_extra_env = {"HF_TOKEN": "hf_live_value_123"}
    state.operator_server_args = "--tp 8 --api-key sekret_key_789"
    state.current_best_measurement = {"launch_evidence": evidence}
    state.current_best = {
        "measurement": {"launch_evidence": evidence},
        "extra_server_args": "--max-total-tokens 120000",
    }
    state.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "v1",
            "task_id": "t1",
            "workspace": "/w",
            "tput": 120.0,
            "extra_envs": {"HF_TOKEN": "hf_live_value_123"},
            "launch_evidence": evidence,
        }
    ]
    return state


def test_view_carries_no_credential_and_only_the_retest_fields(tmp_path):
    tmp_path.chmod(0o2775)
    state = _state_with_real_evidence(tmp_path)
    state.save(tmp_path)
    text = (tmp_path / RETEST_VIEW).read_text()
    for secret in ("sekret_key_789", "adm_key_456", "hf_live_value_123"):
        assert secret not in text
    view = json.loads(text)
    assert view["schema_version"] == 1
    assert view["framework"] == "sglang" and view["claw_session_id"] == "sid-1"
    assert view["optimization_stack"] == [
        {"action": "explore", "variant_name": "v1", "task_id": "t1", "workspace": "/w", "tput": 120.0}
    ]
    assert stat.S_IMODE((tmp_path / RETEST_VIEW).stat().st_mode) == 0o640


def test_state_json_stays_owner_only_and_unmasked(tmp_path):
    tmp_path.chmod(0o2775)
    state = _state_with_real_evidence(tmp_path)
    state.save(tmp_path)
    assert stat.S_IMODE((tmp_path / "state.json").stat().st_mode) == 0o600
    # The observed launch really holds the keys here, so their absence from the view is the view's doing.
    assert "adm_key_456" in (tmp_path / "state.json").read_text()
    loaded = SharedState.load_or_init(tmp_path)
    assert loaded.operator_extra_env == {"HF_TOKEN": "hf_live_value_123"}
    assert loaded.operator_server_args == "--tp 8 --api-key sekret_key_789"
    assert loaded.current_best["extra_server_args"] == "--max-total-tokens 120000"
