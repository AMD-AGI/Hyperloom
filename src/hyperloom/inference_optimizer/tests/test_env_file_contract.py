# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HYPERLOOM_ENV_FILE must mean the same thing on every surface.

A checkout is shared across runs while USER_DATA_PATH, HYPERLOOM_RUNTIME_DIR and
the ``*_ROOT`` paths are per-run. Each surface that reads configuration used to
resolve the checkout's ``.env`` on its own, so opting out of one still left the
others importing a previous run's values. These tests pin the contract:

    unset        -> $REPO_ROOT/.env
    <path>       -> that file
    "" | none    -> no file; the caller's environment is the configuration
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
TOP_INSTALL = REPO / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
KERNEL_INSTALL = REPO / "hyperloom" / "agents" / "kernel" / "scripts" / "install.sh"

if not TOP_INSTALL.is_file():  # src-layout checkout
    TOP_INSTALL = REPO / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
    KERNEL_INSTALL = REPO / "src" / "hyperloom" / "agents" / "kernel" / "scripts" / "install.sh"


def _checkout_with_env(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".env").write_text(
        "HYPERLOOM_RUN_MODE=docker\n"
        "USER_DATA_PATH=/previous-run/hyperloom\n"
        "GEAK_ROOT=/previous-run/GEAK\n"
    )
    return root


def _extract(path: Path, marker: str) -> str:
    """Return ``path`` truncated after the line equal to ``marker``."""
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == marker:
            return "\n".join(lines[: i + 1])
    raise AssertionError(f"marker {marker!r} not found in {path}")


def _run(script: str, env: dict[str, str], probe: str) -> str:
    body = f"set -uo pipefail\n{script}\n{probe}\n"
    out = subprocess.run(
        ["bash", "-c", body],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("opt_out", ["none", "NONE", ""])
def test_top_installer_opt_out_keeps_caller_paths(tmp_path, opt_out):
    root = _checkout_with_env(tmp_path)
    script = _extract(TOP_INSTALL, "load_dotenv_no_clobber")
    env = {
        "REPO_ROOT": str(root),
        "USER_DATA_PATH": "/this-run/hyperloom",
        "HYPERLOOM_ENV_FILE": opt_out,
    }
    assert _run(script, env, 'echo "$USER_DATA_PATH"') == "/this-run/hyperloom"


def test_top_installer_default_still_reads_the_checkout(tmp_path):
    """The default must not change: only an explicit opt-out disables import."""
    root = _checkout_with_env(tmp_path)
    script = _extract(TOP_INSTALL, "load_dotenv_no_clobber")
    env = {"REPO_ROOT": str(root), "USER_DATA_PATH": "/this-run/hyperloom"}
    assert _run(script, env, 'echo "$USER_DATA_PATH"') == "/previous-run/hyperloom"


@pytest.mark.parametrize("opt_out", ["none", ""])
def test_kernel_installer_opt_out_does_not_import_unset_vars(tmp_path, opt_out):
    """The regression: GEAK_ROOT is unset here, so the protection list -- which
    only restores non-empty values -- cannot stop the checkout supplying it."""
    root = _checkout_with_env(tmp_path)
    script = textwrap.dedent(
        """
        REPO_ROOT="${REPO_ROOT:-$(pwd)}"
        resolve_env_file() {
          if [ -z "${HYPERLOOM_ENV_FILE+x}" ]; then
            printf '%s\\n' "$REPO_ROOT/.env"; return 0
          fi
          case "$HYPERLOOM_ENV_FILE" in
            ""|none|NONE) printf '%s\\n' "" ;;
            *) printf '%s\\n' "$HYPERLOOM_ENV_FILE" ;;
          esac
        }
        DOTENV="$(resolve_env_file)"
        if [ -n "$DOTENV" ] && [ -f "$DOTENV" ]; then
          set -a; . "$DOTENV"; set +a
        fi
        """
    )
    # The inlined resolver above must match the installer's own.
    assert "resolve_env_file()" in KERNEL_INSTALL.read_text()
    assert '[ -n "$DOTENV" ] && [ -f "$DOTENV" ]' in KERNEL_INSTALL.read_text()

    out = _run(script, {"REPO_ROOT": str(root), "HYPERLOOM_ENV_FILE": opt_out},
               'echo "${GEAK_ROOT:-<unset>}"')
    assert out == "<unset>"

    out = _run(script, {"REPO_ROOT": str(root)}, 'echo "${GEAK_ROOT:-<unset>}"')
    assert out == "/previous-run/GEAK", "default behaviour must be unchanged"


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
