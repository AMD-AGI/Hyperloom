# Copyright Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for bypass_engine.build_server_command framework_python switch (Gap 5)."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors.bypass_engine import build_server_command


def _base(**kw):
    defaults = dict(
        model="/models/m",
        tp=4,
        port=8888,
        max_model_len=None,
        extra_args=[],
        profile_dir=None,
    )
    defaults.update(kw)
    return defaults


# ---------------------------------------------------------------------------
# vLLM: backward-compat — no framework_python → bare vllm serve
# ---------------------------------------------------------------------------

def test_vllm_no_framework_python_uses_console_script():
    cmd = build_server_command(framework="vllm", **_base())
    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"


def test_vllm_no_framework_python_python_exe_ignored():
    cmd = build_server_command(framework="vllm", python_exe="/some/python", **_base())
    assert cmd[0] == "vllm"
    assert "/some/python" not in cmd


# ---------------------------------------------------------------------------
# vLLM: framework_python set → python -m vllm.entrypoints.openai.api_server
# ---------------------------------------------------------------------------

def test_vllm_framework_python_switches_to_module_launch():
    cmd = build_server_command(
        framework="vllm",
        framework_python="/venv/bin/python3.11",
        **_base(),
    )
    assert cmd[0] == "/venv/bin/python3.11"
    assert cmd[1] == "-m"
    assert cmd[2] == "vllm.entrypoints.openai.api_server"


def test_vllm_framework_python_contains_model_and_port():
    cmd = build_server_command(
        framework="vllm",
        framework_python="/venv/bin/python3.11",
        **_base(model="/m/my-model", port=9999),
    )
    assert "/m/my-model" in cmd
    assert "9999" in cmd


def test_vllm_framework_python_includes_max_model_len():
    cmd = build_server_command(
        framework="vllm",
        framework_python="/venv/bin/python3.11",
        **_base(max_model_len=4096),
    )
    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_vllm_framework_python_includes_extra_args():
    cmd = build_server_command(
        framework="vllm",
        framework_python="/venv/bin/python3.11",
        **_base(extra_args=["--enforce-eager"]),
    )
    assert "--enforce-eager" in cmd


def test_vllm_framework_python_includes_profiler():
    cmd = build_server_command(
        framework="vllm",
        framework_python="/venv/bin/python3.11",
        **_base(profile_dir="/tmp/prof"),
    )
    assert "--profiler-config.profiler" in cmd
    assert "/tmp/prof" in cmd


# ---------------------------------------------------------------------------
# sglang: framework_python replaces python_exe
# ---------------------------------------------------------------------------

def test_sglang_framework_python_replaces_python_exe():
    cmd = build_server_command(
        framework="sglang",
        framework_python="/venv/bin/python3.11",
        python_exe="/old/python",
        **_base(),
    )
    assert cmd[0] == "/venv/bin/python3.11"
    assert "/old/python" not in cmd


def test_sglang_no_framework_python_uses_python_exe():
    cmd = build_server_command(
        framework="sglang",
        python_exe="/default/python",
        **_base(),
    )
    assert cmd[0] == "/default/python"


# ---------------------------------------------------------------------------
# atom: framework_python replaces python_exe
# ---------------------------------------------------------------------------

def test_atom_framework_python_replaces_python_exe():
    cmd = build_server_command(
        framework="atom",
        framework_python="/venv/bin/python3.11",
        python_exe="/old/python",
        **_base(),
    )
    assert cmd[0] == "/venv/bin/python3.11"


# ---------------------------------------------------------------------------
# unknown framework raises
# ---------------------------------------------------------------------------

def test_unknown_framework_raises():
    with pytest.raises(ValueError, match="no server launcher"):
        build_server_command(framework="custom_fw", **_base())
