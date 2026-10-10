#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for install.sh home-directory resolution."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


INSTALL_SH = Path(__file__).resolve().parents[1] / "assets" / "install_kernel_tools.sh"
_BASE_PATH = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


def _extract_func(name: str, *, required: bool = True) -> str:
    """Return the top-level ``name() { ... }`` block from install.sh."""
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


_CLAUDE_FN = "ensure_claude_cli"
_FAKE_CLAUDE = '#!/bin/sh\necho "2.1.200 (Claude Code)"\n'
# Only these host tools are visible, so a claude or curl elsewhere on the host cannot satisfy the function under test.
_HOST_TOOLS = ("bash", "mkdir", "chmod", "cat", "id", "getent", "cut")
# Serves a stand-in for Claude Code's native installer: it records its target and installs into ~/.local/bin.
_CURL_STUB = r"""#!/bin/bash
[ "${TEST_CURL_FAIL:-0}" != 1 ] || exit 22
cat <<'EOS'
set -eu
printf '%s' "${1:-}" > "$TEST_STATE/installed-target"
mkdir -p "$HOME/.local/bin"
printf '#!/bin/sh\necho "2.1.200 (Claude Code)"\n' > "$HOME/.local/bin/claude"
chmod +x "$HOME/.local/bin/claude"
EOS
"""


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_claude_cli(
    tmp_path: Path,
    *,
    check_only: bool = False,
    curl: bool = True,
    claude_on_path: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real installer function against a sandboxed PATH, HOME and native installer."""
    sandbox = tmp_path / "sandbox"
    tools = sandbox / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    for name in _HOST_TOOLS:
        found = shutil.which(name)
        if found and not (tools / name).exists():
            (tools / name).symlink_to(found)
    stubs = sandbox / "stubs"
    stubs.mkdir(exist_ok=True)
    if curl:
        _write_executable(stubs / "curl", _CURL_STUB)
    if claude_on_path:
        _write_executable(stubs / "claude", _FAKE_CLAUDE)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": f"{stubs}:{tools}", "HOME": str(home), "TEST_STATE": str(sandbox), **(extra_env or {})}
    script = f"""set -euo pipefail
log() {{ printf '[log] %s\\n' "$*"; }}
warn() {{ printf '[warn] %s\\n' "$*" >&2; }}
die() {{ printf '[die] %s\\n' "$*" >&2; exit 1; }}
CHECK_ONLY={int(check_only)}
DRY_RUN=0
_ANTHROPIC_KEY_VAL=sk-hl-test-key
_ANTHROPIC_BASE_URL_VAL=https://gateway.example.com/v1/
{_extract_func("_home_dir")}
{_extract_func(_CLAUDE_FN)}
{_CLAUDE_FN}
printf 'CLAUDE_ON_PATH=%s\\n' "$(command -v claude || true)"
printf 'CLAUDE_PATH_DIR=%s\\n' "${{_claude_path_dir:-}}"
"""
    return _run_bash(script, env)


def _installed_target(tmp_path: Path) -> str | None:
    marker = tmp_path / "sandbox" / "installed-target"
    return marker.read_text(encoding="utf-8") if marker.exists() else None


def test_installs_claude_into_home_and_puts_it_on_path(tmp_path: Path) -> None:
    proc = _run_claude_cli(tmp_path)
    combined = proc.stdout + proc.stderr
    home_bin = tmp_path / "home" / ".local" / "bin"
    assert proc.returncode == 0, combined
    assert _installed_target(tmp_path) == "latest", combined
    assert f"CLAUDE_ON_PATH={home_bin / 'claude'}" in combined, combined
    assert f"CLAUDE_PATH_DIR={home_bin}" in combined, combined


def test_reuses_the_copy_already_in_home_instead_of_installing_again(tmp_path: Path) -> None:
    """GEAK's earlier ~/.local/bin copy sits in the install location itself, so it goes on PATH as it is."""
    _write_executable(tmp_path / "home" / ".local" / "bin" / "claude", _FAKE_CLAUDE)
    proc = _run_claude_cli(tmp_path)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert _installed_target(tmp_path) is None, combined
    assert f"CLAUDE_ON_PATH={tmp_path / 'home' / '.local' / 'bin' / 'claude'}" in combined, combined


def test_reuses_claude_already_on_path(tmp_path: Path) -> None:
    proc = _run_claude_cli(tmp_path, claude_on_path=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert _installed_target(tmp_path) is None, combined
    assert f"CLAUDE_ON_PATH={tmp_path / 'sandbox' / 'stubs' / 'claude'}" in combined, combined
    config = json.loads((tmp_path / "home" / ".claude" / "config.json").read_text(encoding="utf-8"))
    assert config["primaryApiKey"] == "sk-hl-test-key"
    assert config["customApiUrl"] == "https://gateway.example.com"


def test_version_pin_installs_that_version_ahead_of_the_claude_on_path(tmp_path: Path) -> None:
    """A pin must reach every consumer, so the pinned copy is the one PATH resolves, not the image's."""
    proc = _run_claude_cli(tmp_path, claude_on_path=True, extra_env={"HYPERLOOM_CLAUDE_CODE_VERSION": "2.1.200"})
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert _installed_target(tmp_path) == "2.1.200", combined
    assert f"CLAUDE_ON_PATH={tmp_path / 'home' / '.local' / 'bin' / 'claude'}" in combined, combined


def test_version_pin_wins_when_the_home_bin_dir_is_already_on_path_behind_the_image_claude(tmp_path: Path) -> None:
    """A pin must reach PATH and the env file even when ~/.local/bin was already on PATH, behind another claude."""
    sandbox = tmp_path / "sandbox"
    home_bin = tmp_path / "home" / ".local" / "bin"
    path = f"{sandbox / 'stubs'}:{sandbox / 'tools'}:{home_bin}"
    proc = _run_claude_cli(
        tmp_path, claude_on_path=True, extra_env={"PATH": path, "HYPERLOOM_CLAUDE_CODE_VERSION": "2.1.200"}
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert _installed_target(tmp_path) == "2.1.200", combined
    assert f"CLAUDE_ON_PATH={home_bin / 'claude'}" in combined, combined
    assert f"CLAUDE_PATH_DIR={home_bin}" in combined, combined


@pytest.mark.parametrize(
    ("curl", "extra_env"),
    [(False, {}), (True, {"TEST_CURL_FAIL": "1"})],
    ids=["no-curl", "installer-fails"],
)
def test_installer_fails_when_claude_cannot_be_installed(tmp_path: Path, curl: bool, extra_env: dict[str, str]) -> None:
    """Without a claude CLI every specialist would run without its tools, so the install stops here."""
    proc = _run_claude_cli(tmp_path, curl=curl, extra_env=extra_env)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "[die]" in combined, combined
    assert "CLAUDE_ON_PATH=" not in combined, combined
    assert not (tmp_path / "home" / ".claude" / "config.json").exists()


def test_check_only_reports_a_missing_claude_without_installing(tmp_path: Path) -> None:
    proc = _run_claude_cli(tmp_path, check_only=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "[warn]" in combined, combined
    assert _installed_target(tmp_path) is None, combined
    assert not (tmp_path / "home" / ".claude" / "config.json").exists()


def test_credentials_land_under_home(tmp_path: Path) -> None:
    """The credential write must target $HOME/.claude, never /root/.claude."""
    home = tmp_path / "home" / "hluser"
    home.mkdir(parents=True)
    proc = _run_claude_cli(tmp_path, claude_on_path=True, extra_env={"HOME": str(home)})
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{_CLAUDE_FN} failed:\n{combined}"
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
    # _home_dir owns the only HOME reference and guards presence with ${HOME+x}; anywhere else a bare ${HOME} is fatal
    # under set -u.
    outside_resolver = text.replace(_extract_func("_home_dir", required=False), "")
    # Both spellings: $HOME reads the same to the shell and slips a ${...}-only check.
    bare = re.findall(r"\$\{?HOME\b", outside_resolver)
    assert not bare, f"use _home_dir instead of a bare HOME reference: {bare}"


def _write_env_file(
    tmp_path: Path, *, path: str, prelude: str = "", home: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Source the installer's functions and run write_env_file()."""
    sourceable = tmp_path / "install_sourceable.sh"
    # Drop the trailing ``main "$@"`` dispatch so sourcing only defines functions.
    sourceable.write_text(
        re.sub(r'(?m)^main "\$@"\s*$', "", INSTALL_SH.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runtime = tmp_path / "runtime"
    script = f"""set -euo pipefail
export ANTHROPIC_API_KEY=sk-hl-test-key
export ANTHROPIC_BASE_URL=https://gateway.example.com
export REPO_ROOT={repo_root}
export USER_DATA_PATH={tmp_path}
export HYPERLOOM_RUNTIME_DIR={runtime}
export KERNEL_AGENT_ENV={runtime / "kernel-agent.env.sh"}
CHECK_ONLY=0
DRY_RUN=0
source {sourceable}
{prelude}
write_env_file
"""
    return _run_bash(script, {"PATH": path, **({"HOME": str(home)} if home else {})})


def test_write_env_file_survives_unset_home(tmp_path: Path) -> None:
    """write_env_file() must not depend on HOME being set."""
    proc = _write_env_file(tmp_path, path=_BASE_PATH)
    detail = f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert proc.returncode == 0, f"write_env_file crashed with HOME unset:\n{detail}"
    assert (tmp_path / "runtime" / "kernel-agent.env.sh").is_file(), f"env file missing:\n{detail}"


def test_env_file_points_geak_and_later_shells_at_the_claude_on_path(tmp_path: Path) -> None:
    """GEAK and every sourcing shell resolve the claude on PATH, never a second copy beside it."""
    stubs = tmp_path / "stubs"
    _write_executable(stubs / "claude", _FAKE_CLAUDE)
    home = tmp_path / "home"
    _write_executable(home / ".local" / "bin" / "claude", _FAKE_CLAUDE)
    # After sourcing: the installer prepends the system bin dirs to PATH, where a host claude may live.
    proc = _write_env_file(
        tmp_path, path=_BASE_PATH, prelude=f'PATH="{stubs}:$PATH"\n_claude_path_dir=/opt/claude/bin', home=home
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (tmp_path / "runtime" / "kernel-agent.env.sh").read_text(encoding="utf-8")
    assert f"export GEAK_CLAUDE_BIN='{stubs / 'claude'}'" in text, text
    assert "export PATH='/opt/claude/bin':\"$PATH\"" in text, text
