# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator's ``--extra-env`` pins stay out of the group-readable state.json."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.state.operator_env import EXTRA_ENV_SIDECAR, write_state_json
from hyperloom.orchestrator.state.shared_state import SharedState


_PINS = {"HF_TOKEN": "hf_live_value_123", "OPENAI_API_KEY": "sk-live-value-456", "SGLANG_FLAG": "1"}


def _saved(tmp_path: Path) -> SharedState:
    state = SharedState()
    state.operator_extra_env = dict(_PINS)
    state.save(tmp_path)
    return state


def test_state_json_carries_no_extra_env_credential(tmp_path):
    _saved(tmp_path)
    text = (tmp_path / "state.json").read_text()
    for secret in ("hf_live_value_123", "sk-live-value-456"):
        assert secret not in text
    assert json.loads(text)["operator_extra_env"]["SGLANG_FLAG"] == "1"
    sidecar = tmp_path / EXTRA_ENV_SIDECAR
    assert oct(sidecar.stat().st_mode & 0o777) == oct(0o600)


def test_resume_restores_pins_from_sidecar(tmp_path):
    _saved(tmp_path)
    assert "hf_live_value_123" not in (tmp_path / "state.json").read_text()
    assert SharedState.load_or_init(tmp_path).operator_extra_env == _PINS


def test_old_session_with_plaintext_pins_migrates_on_next_save(tmp_path):
    old = SharedState().to_dict()
    old["operator_extra_env"] = dict(_PINS)
    (tmp_path / "state.json").write_text(json.dumps(old))
    state = SharedState.load_or_init(tmp_path)
    assert state.operator_extra_env == _PINS
    state.save(tmp_path)
    assert "hf_live_value_123" not in (tmp_path / "state.json").read_text()
    assert SharedState.load_or_init(tmp_path).operator_extra_env == _PINS


def test_no_pins_removes_a_stale_sidecar(tmp_path):
    _saved(tmp_path)
    state = SharedState.load_or_init(tmp_path)
    state.operator_extra_env = {}
    state.save(tmp_path)
    assert not (tmp_path / EXTRA_ENV_SIDECAR).exists()


def test_write_state_json_masks_and_splits(tmp_path):
    write_state_json(tmp_path, tmp_path / "state.json", {"operator_extra_env": dict(_PINS), "x": 1})
    shown = json.loads((tmp_path / "state.json").read_text())
    assert shown["x"] == 1 and shown["operator_extra_env"]["HF_TOKEN"] != _PINS["HF_TOKEN"]
    assert json.loads((tmp_path / EXTRA_ENV_SIDECAR).read_text()) == {"operator_extra_env": _PINS}


def test_launch_evidence_secrets_never_reach_state_json(tmp_path):
    from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence

    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    evidence = build_launch_evidence(
        RequestedLaunch(
            config,
            "vllm",
            server_args="--tp 8 --api-key sekret_key_789",
            server_env={"HF_TOKEN": "hf_live_value_123", "SGLANG_FLAG": "1"},
            model_path="/models/m",
        ),
        actual_server_log=None,
        slot=tmp_path / "slot",
    )
    state = SharedState()
    state.operator_server_args = "--tp 8 --api-key sekret_key_789"
    state.current_best_measurement = {"launch_evidence": evidence}
    state.current_best = {"measurement": {"launch_evidence": evidence}}
    state.save(tmp_path)
    text = (tmp_path / "state.json").read_text()
    for secret in ("hf_live_value_123", "sekret_key_789"):
        assert secret not in text
    shown = json.loads(text)["current_best_measurement"]["launch_evidence"]
    assert shown["requested_server_env"]["SGLANG_FLAG"] == "1" and "--tp 8" in shown["requested_server_args"]
    # The live state keeps the real values, and --resume gets the operator's --server-args back.
    assert evidence["requested_server_env"]["HF_TOKEN"] == "hf_live_value_123"
    assert SharedState.load_or_init(tmp_path).operator_server_args == "--tp 8 --api-key sekret_key_789"
