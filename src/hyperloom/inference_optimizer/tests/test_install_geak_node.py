# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the Node and Codex binaries the installer pins for GEAK."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from ._kernel_tools_harness import sourceable_kernel_tools

_SRC = Path(__file__).resolve().parents[3]
_ARCH = {"x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine()]
_NODE_DIR = f"node-v22.23.3-linux-{_ARCH}"


def _fake_node(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho v{version}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _install(tmp_path: Path, body: str, *, node_on_path: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Source the kernel-agent steps with a fake ``node`` of ``node_on_path`` first on PATH, then run ``body``."""
    fake_bin = tmp_path / "bin"
    _fake_node(fake_bin / "node", node_on_path)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    # Sourcing the steps puts the system directories first on PATH, so the fake node goes in front afterwards.
    script = (
        f'set -euo pipefail\nsource {sourceable_kernel_tools(tmp_path)!s}\nexport PATH="{fake_bin}:$PATH"\n{body}\n'
    )
    controlled = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_", "HYPERLOOM_GEAK_"))}
    controlled.update(
        REPO_ROOT=str(repo),
        HYPERLOOM_CACHE_DIR=str(tmp_path / "cache"),
        GEAK_ROOT=str(tmp_path / "geak"),
        PYTHON=sys.executable,
        PYTHONPATH=str(_SRC),
        **env,
    )
    proc = subprocess.run(["bash", "-c", script], env=controlled, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _node_bin(proc: subprocess.CompletedProcess[str]) -> str:
    return next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("NODE="))


def test_a_new_enough_node_on_path_is_reused(tmp_path):
    proc = _install(tmp_path, 'ensure_geak_node\necho "NODE=${_geak_node_bin}"', node_on_path="22.4.0")

    assert _node_bin(proc) == str(tmp_path / "bin" / "node")


def test_an_old_node_on_path_gives_way_to_the_cached_build(tmp_path):
    cached = _fake_node(tmp_path / "cache" / _NODE_DIR / "bin" / "node", "22.23.3")

    proc = _install(tmp_path, 'ensure_geak_node\necho "NODE=${_geak_node_bin}"', node_on_path="18.19.0")

    assert _node_bin(proc) == str(cached)


def test_an_operator_pinned_node_is_kept(tmp_path):
    proc = _install(
        tmp_path,
        'ensure_geak_node\necho "NODE=${_geak_node_bin}"',
        node_on_path="22.4.0",
        GEAK_NODE_BIN="/opt/node22/bin/node",
    )

    assert _node_bin(proc) == "/opt/node22/bin/node"


def test_the_env_file_and_dotenv_pin_node_and_codex_for_geak(tmp_path):
    env_file = tmp_path / "runtime" / "kernel-agent.env.sh"

    _install(
        tmp_path,
        'ensure_geak_node\n_geak_codex_bin="$(_resolve_geak_codex_bin)"\nwrite_env_file',
        node_on_path="22.4.0",
        KERNEL_AGENT_ENV=str(env_file),
        GEAK_CODEX_BIN="/opt/codex/bin/codex",
    )

    node = tmp_path / "bin" / "node"
    exported = env_file.read_text(encoding="utf-8")
    assert f"export GEAK_NODE_BIN='{node}'" in exported
    assert "export GEAK_CODEX_BIN='/opt/codex/bin/codex'" in exported
    dotenv = (tmp_path / "repo" / ".env").read_text(encoding="utf-8").splitlines()
    assert f"GEAK_NODE_BIN={node}" in dotenv
    assert "GEAK_CODEX_BIN=/opt/codex/bin/codex" in dotenv
