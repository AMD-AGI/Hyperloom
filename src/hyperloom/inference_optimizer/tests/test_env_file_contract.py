# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HYPERLOOM_ENV_FILE must mean the same thing on every surface that still has one.

A checkout is shared across runs while USER_DATA_PATH, HYPERLOOM_RUNTIME_DIR and
the ``*_ROOT`` paths are per-run. Each surface that reads configuration used to
resolve the checkout's ``.env`` on its own, so opting out of one still left the
others importing a previous run's values. These tests pin the contract:

    unset        -> $REPO_ROOT/.env
    <path>       -> that file
    "" | none    -> no file; the caller's environment is the configuration

Both the top installer (now via the ``runtime_env.sh`` asset it sources) and the
Python CLI (``preflight``) read ``HYPERLOOM_ENV_FILE`` directly. ``RUNTIME_ENV_FILE``
is a same-named-looking but unrelated *internal* variable ``runtime_env.sh`` resolves
``HYPERLOOM_ENV_FILE`` into for its own use -- not a second operator-facing contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
TOP_INSTALL = REPO / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"

if not TOP_INSTALL.is_file():  # src-layout checkout
    TOP_INSTALL = REPO / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"


def _checkout_with_env(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".env").write_text(
        "HYPERLOOM_RUN_MODE=docker\nUSER_DATA_PATH=/previous-run/hyperloom\nGEAK_ROOT=/previous-run/GEAK\n"
    )
    return root


def _extract(path: Path, marker: str) -> str:
    """Return ``path`` truncated after the line equal to ``marker``."""
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == marker:
            return "\n".join(lines[: i + 1])
    raise AssertionError(f"marker {marker!r} not found in {path}")


def _run_top_installer(tmp_path: Path, env: dict[str, str], probe: str) -> str:
    """Run the extracted ``load_dotenv_no_clobber`` snippet as a real file.

    The real function now lives in ``runtime_env.sh``, sourced by ``install.sh`` via a path
    derived from ``BASH_SOURCE[0]``. A ``bash -c`` string has no real file for that to resolve
    against, so the source silently fails and -- because the extracted snippet carries
    install.sh's own ``set -euo pipefail`` -- aborts the whole script before the probe ever runs
    (every call then reports the empty string, regardless of what it should have resolved).
    Materializing the snippet next to a copy of ``runtime_env.sh`` gives it a real path to
    resolve the source line against.
    """
    workdir = tmp_path / "installer_snippet"
    workdir.mkdir()
    shutil.copy(TOP_INSTALL.parent / "runtime_env.sh", workdir / "runtime_env.sh")
    script_path = workdir / "install.sh"
    script_path.write_text(f"{_extract(TOP_INSTALL, 'load_dotenv_no_clobber')}\n{probe}\n")
    out = subprocess.run(
        ["bash", str(script_path)],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("opt_out", ["none", "NONE", ""])
def test_top_installer_opt_out_keeps_caller_paths(tmp_path, opt_out):
    root = _checkout_with_env(tmp_path)
    env = {
        "REPO_ROOT": str(root),
        "USER_DATA_PATH": "/this-run/hyperloom",
        "HYPERLOOM_ENV_FILE": opt_out,
    }
    assert _run_top_installer(tmp_path, env, 'echo "$USER_DATA_PATH"') == "/this-run/hyperloom"


def test_top_installer_default_still_reads_the_checkout(tmp_path):
    """The default must not change: only an explicit opt-out disables import."""
    root = _checkout_with_env(tmp_path)
    env = {"REPO_ROOT": str(root), "USER_DATA_PATH": "/this-run/hyperloom"}
    assert _run_top_installer(tmp_path, env, 'echo "$USER_DATA_PATH"') == "/previous-run/hyperloom"


@pytest.mark.parametrize("opt_out", ["none", "NONE", "", "  none  "])
def test_preflight_opt_out(tmp_path, monkeypatch, opt_out):
    from hyperloom.inference_optimizer.cli import preflight

    root = _checkout_with_env(tmp_path)
    monkeypatch.setenv("REPO_ROOT", str(root))
    monkeypatch.setenv("HYPERLOOM_ENV_FILE", opt_out)
    assert preflight._resolve_dotenv_file() is None


def test_preflight_default_and_explicit_path(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import preflight

    root = _checkout_with_env(tmp_path)
    monkeypatch.setenv("REPO_ROOT", str(root))
    monkeypatch.delenv("HYPERLOOM_ENV_FILE", raising=False)
    assert preflight._resolve_dotenv_file() == root / ".env"

    other = tmp_path / "run.env"
    other.write_text("USER_DATA_PATH=/this-run/hyperloom\n")
    monkeypatch.setenv("HYPERLOOM_ENV_FILE", str(other))
    assert preflight._resolve_dotenv_file() == other
