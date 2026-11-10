# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The coordinator database is owner-only while the operator's pins hold a credential: those a state records, and those
the run exports (a resume with ``--reset-state`` records none and still launches with the ones it exports)."""

from __future__ import annotations

import json

import pytest

from hyperloom.common.env_safety import OPERATOR_EXTRA_ENV_VAR
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.state_file import database_file_mode

_SERVER_ARGS_ENV = "INFERENCE_OPTIMIZER_SERVER_ARGS"


@pytest.fixture(autouse=True)
def _no_exported_pins(monkeypatch):
    monkeypatch.delenv(_SERVER_ARGS_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_EXTRA_ENV_VAR, raising=False)


def _state(**pins):
    state = SharedState()
    for name, value in pins.items():
        setattr(state, name, value)
    return state


@pytest.mark.parametrize(
    ("recorded", "mode"),
    [
        ({}, 0o664),
        ({"operator_server_args": "--tp 8 --max-total-tokens 120000"}, 0o664),
        ({"operator_extra_env": {"SGLANG_USE_AITER": "1"}}, 0o664),
        ({"operator_server_args": "--tp 8 --api-key not-a-pattern"}, 0o600),
        ({"operator_extra_env": {"HF_TOKEN": "hf_live_value_123"}}, 0o600),
        ({"operator_extra_env": {"HF_ENDPOINT": "https://alice:tok3n@hf-mirror"}}, 0o600),
    ],
    ids=["none", "server-args-knobs", "env-knobs", "recorded-api-key", "recorded-env-token", "recorded-userinfo"],
)
def test_the_pins_a_state_records_decide_the_mode(recorded, mode):
    assert database_file_mode(_state(**recorded)) == mode


@pytest.mark.parametrize(
    ("server_args", "env", "mode"),
    [
        ("--tp 8 --api-key not-a-pattern", {}, 0o600),
        ("", {"HF_TOKEN": "hf_live_value_123"}, 0o600),
        ("--tp 8 --max-total-tokens 120000", {"SGLANG_USE_AITER": "1"}, 0o664),
    ],
    ids=["exported-api-key", "exported-env-token", "exported-knobs"],
)
def test_the_pins_the_run_exports_decide_it_when_the_state_records_none(monkeypatch, server_args, env, mode):
    if server_args:
        monkeypatch.setenv(_SERVER_ARGS_ENV, server_args)
    if env:
        monkeypatch.setenv(OPERATOR_EXTRA_ENV_VAR, json.dumps(env))
    assert database_file_mode(_state()) == mode
