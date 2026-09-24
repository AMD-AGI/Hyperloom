#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for install.sh home-directory resolution."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "scripts" / "install.sh"
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


def _run_credential_write(
    tmp_path: Path, *, home: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Keep the credential-path tests on a working, offline Claude runtime."""
    return _run_forge_validation(tmp_path, extra_env={"HOME": home, "USERPROFILE": home, **(extra_env or {})})


def _run_forge_validation(
    tmp_path: Path, *, case: str = "ready", check_only: bool = False, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real installer function and Python validation with all external work isolated."""
    bootstrap = tmp_path / "offline_python.py"
    bootstrap.write_text(
        textwrap.dedent(
            """\
            import builtins
            import os
            import subprocess
            import sys
            from dataclasses import replace
            from pathlib import Path
            from unittest.mock import patch

            from kernelforge.agent_backends import claude, registry

            real_import = builtins.__import__
            def offline_import(name, *args, **kwargs):
                if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
                    raise AssertionError("installer must not import the SDK")
                return real_import(name, *args, **kwargs)
            builtins.__import__ = offline_import
            registry._plugins_loaded = True
            registry._providers = {
                name: replace(provider, availability=lambda: True,
                              credentialed=lambda _env, selected=name: selected == os.environ["TEST_PROVIDER"])
                for name, provider in registry._providers.items() if name in {"claude", "codex"}
            }
            case = os.environ["TEST_CASE"]
            def resolve(explicit=""):
                if explicit.strip():
                    return explicit.strip()
                if case in {"missing", "install-fails", "still-missing", "installed-wrong"} and (
                    case == "still-missing" or not Path(os.environ["TEST_INSTALLED"]).exists()
                ):
                    return "claude"
                return sys.executable
            original_validate = claude.ClaudeBackend.validate_runtime
            def validate(runtime):
                print("VALIDATE_RUNTIME " + runtime.provider, file=sys.stderr)
                original_validate(runtime)
            def version(argv, **kwargs):
                assert argv[1:] == ["--version"], argv
                assert kwargs == dict(capture_output=True, timeout=10, check=False), kwargs
                print("CLI_VERSION", file=sys.stderr)
                if case in {"version-repair", "version-repair-fails"}:
                    installed = Path(os.environ["TEST_INSTALLED"])
                    if not installed.exists() or case == "version-repair-fails":
                        return subprocess.CompletedProcess(argv, 7, b"", b"broken Claude CLI")
                if case == "timeout":
                    raise subprocess.TimeoutExpired(argv, 10)
                if case == "native-error":
                    raise OSError("loader failed")
                return subprocess.CompletedProcess(argv, 7 if case == "nonzero" else 0,
                                                   b"other tool" if case in {"wrong", "installed-wrong"} else b"Claude Code test", b"")
            with patch.object(claude, "resolve_claude_cli", resolve), \
                 patch.object(claude.shutil, "which", return_value=None), \
                 patch.object(claude.ClaudeBackend, "validate_runtime", validate), \
                 patch.object(subprocess, "run", version):
                sys.argv = sys.argv[1:]
                exec(compile(sys.stdin.read(), "<installer-python>", "exec"), {"__name__": "__main__"})
            """
        ),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        **{key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ},
        "HOME": home.as_posix(),
        "USERPROFILE": home.as_posix(),
        "TEST_PYTHON": Path(sys.executable).as_posix(),
        "TEST_BOOTSTRAP": bootstrap.as_posix(),
        "TEST_INSTALLED": (tmp_path / "installed").as_posix(),
        "TEST_PROVIDER": "claude",
        "TEST_CASE": case,
        "PYTHONPATH": str(INSTALL_SH.parents[4]),
        "FORGE_AGENT_BACKEND": "auto",
        "KNOWLEDGE_STORE_MODE": "local",
        "KERNEL_OPT_BACKEND_ORDER": "forge",
    }
    env.update(extra_env or {})
    script = f"""set -euo pipefail
log() {{ printf '[log] %s\\n' "$*"; }}
warn() {{ printf '[warn] %s\\n' "$*" >&2; }}
die() {{ printf '[die] %s\\n' "$*" >&2; exit 1; }}
python3() {{ "$TEST_PYTHON" "$TEST_BOOTSTRAP" "$@"; }}
npm() {{ printf 'unexpected direct npm invocation\\n' >&2; exit 99; }}
run() {{
  printf 'RUN: %s\\n' "$*"
  if [[ "$*" == 'npm install '* ]]; then
    [ "$TEST_CASE" != install-fails ] || return 7
    : > "$TEST_INSTALLED"
  fi
}}
CHECK_ONLY={int(check_only)}
DRY_RUN=0
_ANTHROPIC_KEY_VAL=sk-hl-test-key
_ANTHROPIC_BASE_URL_VAL=https://gateway.example.com/v1/
{_extract_func("_home_dir")}
{_extract_func("ensure_forge_claude_cli")}
ensure_forge_claude_cli
printf 'INSTALL_REACHED_END\\n'
"""
    return _run_bash(script, env)


@pytest.mark.parametrize("case", ["missing", "wrong", "timeout", "native-error", "nonzero"])
def test_forge_check_only_warns_without_a_working_claude(tmp_path: Path, case: str) -> None:
    """--check-only reports on the box, it does not fail the installer over it."""
    proc = _run_forge_validation(tmp_path, case=case, check_only=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "INSTALL_REACHED_END" in combined
    assert "VALIDATE_RUNTIME claude" in combined
    assert "[warn]" in combined
    assert "RUN:" not in combined
    assert not (tmp_path / "home" / ".claude" / "config.json").exists()


@pytest.mark.parametrize("case", ["ready", "missing"])
def test_forge_installer_validates_reused_and_installed_cli_once(tmp_path: Path, case: str) -> None:
    proc = _run_forge_validation(tmp_path, case=case)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert combined.count("VALIDATE_RUNTIME claude") == 1, combined
    assert combined.count("CLI_VERSION") == 1, combined
    assert ("npm install -g" in combined) is (case == "missing"), combined
    config = json.loads((tmp_path / "home" / ".claude" / "config.json").read_text(encoding="utf-8"))
    assert config["primaryApiKey"] == "sk-hl-test-key"
    assert config["customApiUrl"] == "https://gateway.example.com"


@pytest.mark.parametrize("case", ["install-fails", "still-missing", "installed-wrong"])
def test_forge_installer_cannot_succeed_after_an_unusable_install(tmp_path: Path, case: str) -> None:
    proc = _run_forge_validation(tmp_path, case=case)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "npm install -g" in combined
    assert "INSTALL_REACHED_END" not in combined
    assert not (tmp_path / "home" / ".claude" / "config.json").exists()


@pytest.mark.parametrize("case", ["wrong", "timeout", "native-error", "nonzero"])
def test_forge_installer_refuses_to_replace_a_broken_default_cli(tmp_path: Path, case: str) -> None:
    proc = _run_forge_validation(tmp_path, case=case)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert combined.count("VALIDATE_RUNTIME claude") == 1, combined
    assert "RUN:" not in combined
    assert "INSTALL_REACHED_END" not in combined


@pytest.mark.parametrize("case", ["version-repair", "version-repair-fails"])
def test_forge_version_pin_reinstalls_broken_default_then_validates(tmp_path: Path, case: str) -> None:
    proc = _run_forge_validation(tmp_path, case=case, extra_env={"HYPERLOOM_CLAUDE_CODE_VERSION": "1.2.3"})
    combined = proc.stdout + proc.stderr
    assert "RUN: npm install -g @anthropic-ai/claude-code@1.2.3" in combined, combined
    assert (tmp_path / "installed").is_file(), combined
    assert combined.count("VALIDATE_RUNTIME claude") == 1, combined
    assert combined.count("CLI_VERSION") == 1, combined
    repaired = case == "version-repair"
    assert (proc.returncode == 0) is repaired, combined
    assert ("INSTALL_REACHED_END" in combined) is repaired, combined
    assert (tmp_path / "home" / ".claude" / "config.json").exists() is repaired


@pytest.mark.parametrize("valid", [False, True], ids=["bad-pin", "valid-pin"])
def test_forge_installer_never_replaces_an_explicit_cli_pin(tmp_path: Path, valid: bool) -> None:
    pin = Path(sys.executable).as_posix() if valid else (tmp_path / "missing-pin").as_posix()
    proc = _run_forge_validation(tmp_path, extra_env={"FORGE_AGENT_CLI": pin, "HYPERLOOM_CLAUDE_CODE_VERSION": "1.2.3"})
    combined = proc.stdout + proc.stderr
    assert (proc.returncode == 0) is valid, combined
    assert combined.count("VALIDATE_RUNTIME claude") == 1, combined
    assert "npm install" not in combined
    assert "npm config" not in combined
    if not valid:
        assert "missing-pin" in combined
        assert "INSTALL_REACHED_END" not in combined


@pytest.mark.parametrize(
    "extra_env",
    [
        {"KERNEL_OPT_BACKEND_ORDER": "geak"},
        {"KERNEL_OPT_BACKEND_ORDER": "forge,geak"},
        {"KERNEL_OPT_BACKEND_ORDER": "", "FRAMEWORK": "vllm"},
    ],
    ids=["explicit-geak", "non-exact-forge", "default-vllm"],
)
def test_forge_installer_ensures_claude_for_every_kernel_backend(tmp_path: Path, extra_env: dict[str, str]) -> None:
    """The claude CLI and its credentials are runtime-wide, not gated on the selected kernel backend."""
    proc = _run_forge_validation(tmp_path, extra_env={"HYPERLOOM_CLAUDE_CODE_VERSION": "1.2.3", **extra_env})
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "RUN: npm install -g @anthropic-ai/claude-code@1.2.3" in combined, combined
    assert combined.count("VALIDATE_RUNTIME claude") == 1, combined
    config = json.loads((tmp_path / "home" / ".claude" / "config.json").read_text(encoding="utf-8"))
    assert config["primaryApiKey"] == "sk-hl-test-key"


def test_installer_still_provisions_claude_under_another_forge_provider(tmp_path: Path) -> None:
    """A non-Claude provider owns FORGE_AGENT_CLI, so only its validation is skipped.

    GEAK drives the default CLI through ``GEAK_CLAUDE_BIN`` whichever provider
    Forge itself runs, so the install and credentials must still happen.
    """
    proc = _run_forge_validation(
        tmp_path, case="missing", extra_env={"TEST_PROVIDER": "codex", "KERNEL_OPT_BACKEND_ORDER": "geak"}
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "INSTALL_REACHED_END" in combined
    assert "npm install -g" in combined, combined
    assert "VALIDATE_RUNTIME" not in combined, "a Codex CLI must not be judged by the Claude --version check"
    config = json.loads((tmp_path / "home" / ".claude" / "config.json").read_text(encoding="utf-8"))
    assert config["primaryApiKey"] == "sk-hl-test-key"


def test_installer_survives_an_unresolvable_forge_provider(tmp_path: Path) -> None:
    """An unusable FORGE_AGENT_BACKEND is the backend's error to raise, not the installer's.

    The default CLI every backend shares is still provisioned.
    """
    proc = _run_forge_validation(
        tmp_path,
        case="missing",
        extra_env={"FORGE_AGENT_BACKEND": "not-a-provider", "KERNEL_OPT_BACKEND_ORDER": "geak"},
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "npm install -g" in combined, combined
    assert "VALIDATE_RUNTIME" not in combined, combined


@pytest.mark.parametrize("backend", ["", "  FORGE  "])
def test_atom_installer_checks_default_and_explicit_forge(tmp_path: Path, backend: str) -> None:
    proc = _run_forge_validation(
        tmp_path, check_only=True, extra_env={"KERNEL_OPT_BACKEND_ORDER": backend, "FRAMEWORK": "atom"}
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert combined.count("VALIDATE_RUNTIME claude") == 1, combined
    assert "RUN:" not in combined


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


def test_npm_prefix_avoids_usr_local_when_it_is_unwritable(tmp_path: Path) -> None:
    """A global npm install needs a writable prefix."""
    home = tmp_path / "home" / "hluser"
    home.mkdir(parents=True)
    # Pinning the version takes the branch that installs unconditionally, so the test does not depend on whether the
    # host already has a claude binary.
    proc = _run_credential_write(tmp_path, home=str(home), extra_env={"HYPERLOOM_CLAUDE_CODE_VERSION": "1.2.3"})
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0, combined
    prefix_lines = [ln for ln in combined.splitlines() if "npm config set prefix" in ln]
    assert prefix_lines, combined
    if os.access("/usr/local/lib", os.W_OK):
        assert all("/usr/local" in ln for ln in prefix_lines), prefix_lines
    else:
        assert all(str(home / ".local") in ln for ln in prefix_lines), prefix_lines


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
