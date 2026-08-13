#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for install.sh home-directory resolution.

The Claude Code CLI credentials are written by install.sh but read back through
``Path.home()`` by ``inference_optimizer/cli/credentials.py`` and
``agents/kernel/tools/backends/forge_submit.py``. A hardcoded ``/root/.claude``
on the writing side either aborts the installer (``/root`` unwritable under
``set -euo pipefail``) or strands the credentials where no reader looks.

``_home_dir()`` must therefore resolve exactly what ``Path.home()`` resolves.
The differential tests below run both sides under identical environments,
including the unintuitive cases (an empty ``HOME`` resolves to ``/`` rather
than to the passwd entry, because ``posixpath.expanduser`` only consults
``pwd`` when ``HOME`` is absent from the environment).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "scripts" / "install.sh"
_BASE_PATH = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


def _extract_func(name: str, *, required: bool = True) -> str:
    """Return the top-level ``name() { ... }`` block from install.sh.

    The block is bounded by the next top-level definition and then trimmed to
    its last line-initial ``}``, because a heredoc body can open a line with
    ``}`` too. An optional block returns "" when absent, so a harness can still
    exercise its caller and fail on that caller's own behaviour.
    """
    text = INSTALL_SH.read_text(encoding="utf-8")
    header = re.search(rf"(?m)^{re.escape(name)}\(\) \{{", text)
    if header is None:
        assert not required, f"could not locate {name}() in install.sh"
        return ""
    block = text[header.start() :]
    offset = len(header.group(0))
    following = re.search(r"(?m)^[A-Za-z_][A-Za-z0-9_]*\(\)", block[offset:])
    if following is not None:
        block = block[: offset + following.start()]
    braces = list(re.finditer(r"(?m)^\}", block))
    assert braces, f"could not find the closing brace of {name}() in install.sh"
    return block[: braces[-1].end()]


def _run_bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet under a fully controlled environment."""
    return subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


_HOME_CASES = [
    pytest.param({"HOME": "/tmp/hl-home"}, id="set"),
    pytest.param({"HOME": "/tmp/hl-home//"}, id="trailing-slashes"),
    pytest.param({"HOME": "/"}, id="filesystem-root"),
    pytest.param({"HOME": ""}, id="empty"),
    pytest.param({}, id="unset"),
]


@pytest.mark.parametrize("home_env", _HOME_CASES)
def test_home_dir_matches_python_path_home(home_env: dict[str, str]) -> None:
    """install.sh must resolve the directory the credential readers resolve."""
    env = {"PATH": _BASE_PATH, **home_env}
    shell = _run_bash(
        f"set -euo pipefail\n{_extract_func('_home_dir')}\n_home_dir\n",
        env,
    )
    assert shell.returncode == 0, f"_home_dir failed: {shell.stderr}"
    python = subprocess.run(
        [sys.executable, "-c", "from pathlib import Path; print(Path.home())"],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    shown = home_env.get("HOME", "<unset>")
    assert shell.stdout.strip() == python.stdout.strip(), (
        f"HOME={shown!r}: install.sh resolved {shell.stdout.strip()!r} "
        f"but Path.home() resolved {python.stdout.strip()!r}"
    )


def _run_credential_write(tmp_path: Path, *, home: str) -> subprocess.CompletedProcess[str]:
    """Run ensure_forge_claude_cli() with stubbed log/warn/run and a fake npm."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    npm = fake_bin / "npm"
    npm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    npm.chmod(0o755)
    harness = f"""set -euo pipefail
log()  {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
run()  {{ echo "RUN: $*"; }}
die()  {{ echo "[die] $*" >&2; exit 1; }}
CHECK_ONLY=0
DRY_RUN=0
_ANTHROPIC_KEY_VAL="sk-hl-test-key"
_ANTHROPIC_BASE_URL_VAL="https://gateway.example.com/v1/"
{_extract_func("_home_dir", required=False)}
{_extract_func("ensure_forge_claude_cli")}
ensure_forge_claude_cli
"""
    return _run_bash(harness, {"PATH": f"{fake_bin}:{_BASE_PATH}", "HOME": home})


def test_credentials_land_under_home(tmp_path: Path) -> None:
    """The credential write must target $HOME/.claude, never /root/.claude."""
    home = tmp_path / "home" / "hluser"
    home.mkdir(parents=True)
    proc = _run_credential_write(tmp_path, home=str(home))
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"ensure_forge_claude_cli failed:\n{combined}"
    config = home / ".claude" / "config.json"
    assert config.is_file(), f"credentials not written to {config}\n{combined}"
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["primaryApiKey"] == "sk-hl-test-key"
    assert data["customApiUrl"] == "https://gateway.example.com"
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert "/root" not in combined, combined


def test_installer_has_no_hardcoded_root_home() -> None:
    """No home-relative path may bypass the resolver via /root or a bare $HOME."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "/root/.claude" not in text, "credential paths must derive from _home_dir"
    # _home_dir owns the only HOME reference and guards presence with ${HOME+x};
    # anywhere else a bare ${HOME} is fatal under set -u.
    outside_resolver = text.replace(_extract_func("_home_dir", required=False), "")
    assert "${HOME}" not in outside_resolver, "use _home_dir instead of a bare ${HOME}"


def test_write_env_file_survives_unset_home(tmp_path: Path) -> None:
    """write_env_file() probes a home-relative claude binary; HOME may be unset."""
    sourceable = tmp_path / "install_sourceable.sh"
    # Drop the trailing ``main "$@"`` dispatch so sourcing only defines functions.
    sourceable.write_text(
        re.sub(r'(?m)^main "\$@"\s*$', "", INSTALL_SH.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runtime = tmp_path / "runtime"
    env_file = runtime / "kernel-agent.env.sh"
    script = f"""set -euo pipefail
export ANTHROPIC_API_KEY=sk-hl-test-key
export ANTHROPIC_BASE_URL=https://gateway.example.com
export REPO_ROOT={repo_root}
export USER_DATA_PATH={tmp_path}
export HYPERLOOM_RUNTIME_DIR={runtime}
export KERNEL_AGENT_ENV={env_file}
CHECK_ONLY=0
DRY_RUN=0
source {sourceable}
write_env_file
"""
    proc = _run_bash(script, {"PATH": _BASE_PATH})
    detail = f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert proc.returncode == 0, f"write_env_file crashed with HOME unset:\n{detail}"
    assert env_file.is_file(), f"env file missing:\n{detail}"
