# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from hyperloom.hosted.host_info import Var, read_host_info
from hyperloom.hosted.toolbox_inputs import ToolboxInputsError, grant_limits_totals, hosted_toolbox_inputs

_SHA = "0123456789abcdef0123456789abcdef01234567"
_MEASUREMENT = "sentinel-measurement-key-7731"
_DEV = "sentinel-dev-key-9042"
_TEMPLATES = {"runner": {"1": "ar-runner-g1", "8": "ar-runner-g8"}, "dev": "ar-dev-g1"}
_LIMITS = {"max_sandboxes": 4, "max_gpus": 16, "gpu_hours": 2.5, "max_runtime_sec": 3600}

# Pinned literal: the field names of the multibranch toolbox's input (W137 signature).  Not derived from that package.
_FIELDS = [
    "run_id",
    "session_dir",
    "api_url",
    "sandbox_url",
    "measurement_key",
    "sandbox_templates",
    "source_repo",
    "source_sha",
    "tp",
    "dev_key",
    "grant_limits",
]


@pytest.fixture
def env(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "campaign.json").write_text("{}")
    return {
        Var.INPUT_FILE: str(tmp_path / "campaign.json"),
        Var.RUN_ID: "run-1",
        Var.RUN_CREATED_AT: "2026-10-01T12:30:00Z",
        Var.STATE_DIR: str(tmp_path / "state"),
        Var.MCP_PORT: "8765",
        Var.MCP_TOKEN: "sentinel-mcp-token",
        Var.SANDBOX_TEMPLATES: json.dumps(_TEMPLATES),
        Var.E2B_API_URL: "https://api.sandbox.example.test",
        Var.E2B_SANDBOX_URL: "https://sbx.example.test",
        Var.SANDBOX_KEY_MEASUREMENT: _MEASUREMENT,
        Var.SANDBOX_KEY_DEV: _DEV,
        Var.GRANT_LIMITS: json.dumps(_LIMITS),
        Var.SOURCE_REPO: "AMD-AGI/Hyperloom",
        Var.SOURCE_SHA: _SHA,
    }


def _inputs(env, tmp_path, tp=8):
    return hosted_toolbox_inputs(read_host_info(env), session_dir=tmp_path / "session", tp=tp)


def test_the_key_set_is_exactly_the_toolbox_fields(env, tmp_path):
    assert sorted(_inputs(env, tmp_path)) == sorted(_FIELDS)


def test_every_value_is_mapped_from_its_host_input(env, tmp_path):
    got = _inputs(env, tmp_path)

    assert got == {
        "run_id": "run-1",
        "session_dir": tmp_path / "session",
        "api_url": "https://api.sandbox.example.test",
        "sandbox_url": "https://sbx.example.test",
        "measurement_key": _MEASUREMENT,
        "sandbox_templates": _TEMPLATES,
        "source_repo": "AMD-AGI/Hyperloom",
        "source_sha": _SHA,
        "tp": 8,
        "dev_key": _DEV,
        "grant_limits": _LIMITS,
    }
    assert isinstance(got["session_dir"], Path)
    assert isinstance(got["measurement_key"], str)


def test_dev_key_and_grant_limits_are_none_when_absent(env, tmp_path):
    del env[Var.SANDBOX_KEY_DEV], env[Var.GRANT_LIMITS]

    got = _inputs(env, tmp_path)

    assert got["dev_key"] is None
    assert got["grant_limits"] is None


def test_grant_limits_carry_all_four_keys(env, tmp_path):
    assert set(_inputs(env, tmp_path)["grant_limits"]) == {"max_sandboxes", "max_gpus", "gpu_hours", "max_runtime_sec"}


def test_partial_grant_limits_are_refused_naming_the_missing_totals(env, tmp_path):
    env[Var.GRANT_LIMITS] = json.dumps({"max_sandboxes": 2, "max_gpus": 8})

    with pytest.raises(ToolboxInputsError, match=r"HYPERLOOM_GRANT_LIMITS.*missing gpu_hours, max_runtime_sec"):
        _inputs(env, tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        Var.E2B_API_URL,
        Var.E2B_SANDBOX_URL,
        Var.SANDBOX_KEY_MEASUREMENT,
        Var.SANDBOX_TEMPLATES,
        Var.SOURCE_REPO,
        Var.SOURCE_SHA,
    ],
)
def test_a_missing_required_value_is_refused_by_name(env, tmp_path, name):
    del env[name]

    with pytest.raises(ToolboxInputsError) as raised:
        _inputs(env, tmp_path)

    assert str(raised.value) == f"hosted toolbox needs {name}"


def test_every_missing_value_is_named_in_one_error(env, tmp_path):
    for name in (Var.E2B_API_URL, Var.SOURCE_SHA, Var.SANDBOX_KEY_MEASUREMENT):
        del env[name]

    with pytest.raises(ToolboxInputsError) as raised:
        _inputs(env, tmp_path)

    assert str(raised.value) == (
        f"hosted toolbox needs {Var.E2B_API_URL}, {Var.SANDBOX_KEY_MEASUREMENT}, {Var.SOURCE_SHA}"
    )


@pytest.mark.parametrize("sha", ["0123abc", _SHA.upper(), _SHA + "0", "g" * 40])
def test_a_source_sha_that_is_not_40_lowercase_hex_is_refused_at_read(env, sha):
    env[Var.SOURCE_SHA] = sha

    with pytest.raises(ValueError, match=Var.SOURCE_SHA):
        read_host_info(env)


def test_host_info_never_shows_the_keys(env, caplog):
    info = read_host_info(env)

    with caplog.at_level(logging.DEBUG):
        logging.getLogger("t").info("host %r %s %s", info, info, dataclasses.asdict(info))

    shown = repr(info) + str(info) + repr(dataclasses.asdict(info)) + caplog.text
    assert _MEASUREMENT not in shown
    assert _DEV not in shown
    assert "sentinel-mcp-token" not in shown


def test_grant_limits_totals_is_public_and_converts_all_four_totals():
    from hyperloom.hosted.host_info import GrantLimits

    assert grant_limits_totals(None) is None
    assert (
        grant_limits_totals(GrantLimits(max_sandboxes=4, max_gpus=16, gpu_hours=2.5, max_runtime_sec=3600)) == _LIMITS
    )


def test_grant_limits_totals_refuses_unstated_totals():
    from hyperloom.hosted.host_info import GrantLimits

    with pytest.raises(ToolboxInputsError, match="missing gpu_hours, max_runtime_sec"):
        grant_limits_totals(GrantLimits(max_sandboxes=2, max_gpus=8, gpu_hours=None, max_runtime_sec=None))
