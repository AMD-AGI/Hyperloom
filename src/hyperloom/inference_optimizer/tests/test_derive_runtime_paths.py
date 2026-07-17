# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for preflight PATH/LD_LIBRARY_PATH derivation (replaces hyperloom.env.sh)."""

from __future__ import annotations

import os

from hyperloom.inference_optimizer.cli import preflight as cli_preflight


def test_prepend_path_adds_and_dedups(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    cli_preflight._prepend_path("PATH", "/opt/x/bin")
    assert os.environ["PATH"] == f"/opt/x/bin{os.pathsep}/usr/bin{os.pathsep}/bin"
    # Idempotent: already leading -> unchanged.
    cli_preflight._prepend_path("PATH", "/opt/x/bin")
    assert os.environ["PATH"] == f"/opt/x/bin{os.pathsep}/usr/bin{os.pathsep}/bin"
    # Existing-but-not-leading -> moved to front, not duplicated.
    cli_preflight._prepend_path("PATH", "/bin")
    assert os.environ["PATH"] == f"/bin{os.pathsep}/opt/x/bin{os.pathsep}/usr/bin"


def test_prepend_path_empty_entry_noop(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    cli_preflight._prepend_path("PATH", "")
    assert os.environ["PATH"] == "/usr/bin"


def test_derive_runtime_paths_rocm_and_venv(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)

    cli_preflight._derive_runtime_paths()

    path_parts = os.environ["PATH"].split(os.pathsep)
    # ROCm prepended last of the two -> leads; venv follows; system last.
    assert path_parts[0] == os.path.join("/opt/rocm", "bin")
    assert os.path.join("/venv", "bin") in path_parts
    assert os.environ["LD_LIBRARY_PATH"].startswith(os.path.join("/opt/rocm", "lib"))


def test_derive_runtime_paths_isolated_vllm_leads(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/vllm-venv")

    cli_preflight._derive_runtime_paths()

    # vLLM venv prepended last -> must lead PATH.
    assert os.environ["PATH"].split(os.pathsep)[0] == os.path.join("/opt/vllm-venv", "bin")


def test_derive_runtime_paths_noop_without_roots(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    ld_before = os.environ.get("LD_LIBRARY_PATH")

    cli_preflight._derive_runtime_paths()

    assert os.environ["PATH"] == "/usr/bin"
    # No ROCM_PATH -> LD_LIBRARY_PATH must be untouched.
    assert os.environ.get("LD_LIBRARY_PATH") == ld_before
