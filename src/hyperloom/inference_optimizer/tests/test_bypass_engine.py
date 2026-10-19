# Copyright Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for bypass_engine.build_server_command framework_python switch."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors.bypass_engine import ServerSpec, build_server_command


def _spec(framework, **kw):
    defaults = dict(
        model="/models/m",
        tp=4,
        port=8888,
        max_model_len=None,
        profile_dir=None,
    )
    defaults.update(kw)
    return ServerSpec(framework=framework, **defaults)


# vLLM: backward-compat — no framework_python → bare vllm serve


def test_vllm_no_framework_python_uses_console_script():
    cmd = build_server_command(_spec("vllm"), [])
    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"


def test_vllm_no_framework_python_python_exe_ignored():
    cmd = build_server_command(_spec("vllm"), [], python_exe="/some/python")
    assert cmd[0] == "vllm"
    assert "/some/python" not in cmd


# vLLM: framework_python set → python -m vllm.entrypoints.openai.api_server


def test_vllm_framework_python_switches_to_module_launch():
    cmd = build_server_command(_spec("vllm"), [], framework_python="/venv/bin/python3.11")
    assert cmd[0] == "/venv/bin/python3.11"
    assert cmd[1] == "-m"
    assert cmd[2] == "vllm.entrypoints.openai.api_server"


def test_vllm_framework_python_contains_model_and_port():
    cmd = build_server_command(
        _spec("vllm", model="/m/my-model", port=9999), [], framework_python="/venv/bin/python3.11"
    )
    assert "/m/my-model" in cmd
    assert "9999" in cmd


def test_vllm_framework_python_includes_max_model_len():
    cmd = build_server_command(_spec("vllm", max_model_len=4096), [], framework_python="/venv/bin/python3.11")
    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_vllm_framework_python_includes_extra_args():
    cmd = build_server_command(_spec("vllm"), ["--enforce-eager"], framework_python="/venv/bin/python3.11")
    assert "--enforce-eager" in cmd


def test_vllm_framework_python_includes_profiler():
    cmd = build_server_command(_spec("vllm", profile_dir="/tmp/prof"), [], framework_python="/venv/bin/python3.11")
    assert "--profiler-config.profiler" in cmd
    assert "/tmp/prof" in cmd


# sglang: framework_python replaces python_exe


def test_sglang_framework_python_replaces_python_exe():
    cmd = build_server_command(_spec("sglang"), [], framework_python="/venv/bin/python3.11", python_exe="/old/python")
    assert cmd[0] == "/venv/bin/python3.11"
    assert "/old/python" not in cmd


def test_sglang_no_framework_python_uses_python_exe():
    cmd = build_server_command(_spec("sglang"), [], python_exe="/default/python")
    assert cmd[0] == "/default/python"


# atom: framework_python replaces python_exe


def test_atom_framework_python_replaces_python_exe():
    cmd = build_server_command(_spec("atom"), [], framework_python="/venv/bin/python3.11", python_exe="/old/python")
    assert cmd[0] == "/venv/bin/python3.11"


# unknown framework raises


def test_unknown_framework_raises():
    with pytest.raises(ValueError, match="no server launcher"):
        build_server_command(_spec("custom_fw"), [])
