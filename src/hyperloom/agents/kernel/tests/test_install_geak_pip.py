# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ensure_geak()'s pip-install path."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
INSTALL_SH = REPO_ROOT / "src" / "hyperloom" / "agents" / "kernel" / "scripts" / "install.sh"


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=test",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )


def _new_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "geak-remote"
    remote.mkdir()
    _git(remote, "init", "-q", "-b", "main")
    _commit(remote, "v1")
    return remote


def _checkout_state(checkout: Path) -> tuple[str, str, bytes | None, str]:
    source = checkout / "kernel.py"
    return (
        _git(checkout, "status", "--short", "--branch"),
        _git(checkout, "rev-parse", "--abbrev-ref", "HEAD"),
        source.read_bytes() if source.exists() else None,
        _git(checkout, "rev-parse", "--is-shallow-repository"),
    )


def _sourceable_installer(dest_dir: Path) -> Path:
    text = INSTALL_SH.read_text(encoding="utf-8")
    patched = re.sub(r'(?m)^main "\$@"\s*$', "", text)
    copy = dest_dir / "install_sourceable.sh"
    copy.write_text(patched, encoding="utf-8")
    return copy


def _run_installer_for_geak(tmp_path: Path, geak_root: Path, geak_repo: Path) -> subprocess.CompletedProcess[str]:
    sourceable = _sourceable_installer(tmp_path)
    harness = tmp_path / "run-geak.sh"
    harness.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
git() {{
  if [ "${{1:-}}" = ls-remote ] && [ "${{2:-}}" = https://github.com/AMD-AGI/TraceLens.git ]; then
    printf '%s\\t%s\\n' 1111111111111111111111111111111111111111 refs/heads/main
    return 0
  fi
  command git "$@"
}}
export REPO_ROOT={shlex.quote(str(tmp_path / "hyperloom"))}
export KERNEL_AGENT_ROOT={shlex.quote(str(tmp_path / "kernel-agent"))}
export USER_DATA_PATH={shlex.quote(str(tmp_path / "userdata"))}
export HYPERLOOM_CACHE_DIR={shlex.quote(str(tmp_path / "cache"))}
export MAGPIE_PYTHON="$(command -v python3)"
export GEAK_ROOT={shlex.quote(str(geak_root))}
export GEAK_REPO={shlex.quote(str(geak_repo))}
export GEAK_REF=main
export ANTHROPIC_BASE_URL=https://api.example.test
export ANTHROPIC_API_KEY=test-key
export ANTHROPIC_AUTH_TOKEN=test-token
export CLAUDE_CODE_OAUTH_TOKEN=test-oauth-token
. {shlex.quote(str(sourceable))}
run() {{
  if [ "${{1:-}}" = git ]; then
    command "$@"
  else
    printf 'RUN: %s\\n' "$*"
  fi
}}
DRY_RUN=0
CHECK_ONLY=0
ensure_geak
report_status
""",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(harness)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _extract_ensure_geak() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(r"^ensure_geak\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "could not locate ensure_geak() in install.sh"
    return m.group(0)


def _run_ensure_geak(tmp_path: Path, *, package_metadata: bool) -> str:
    """Run the extracted ensure_geak body with stubs; return combined output."""
    geak_root = tmp_path / "os" / "GEAK"
    (geak_root / ".git").mkdir(parents=True)  # take the "already present" path
    (geak_root / "interface").mkdir(parents=True)
    (geak_root / "interface" / "run_e2e.py").write_text("# runner\n", encoding="utf-8")
    if package_metadata:
        (geak_root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log()  {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
run()  {{ echo "RUN: $*"; }}
CHECK_ONLY=0
DRY_RUN=0
GEAK_ROOT="{geak_root}"
# A non-HTTPS override that is a valid `git clone` target but NOT a valid
# `git+...` pip URL — proves we never build such a URL.
GEAK_REPO="git@github.com:acme/GEAK.git"
GEAK_REF="main"
GEAK_E2E_RUNNER="${{GEAK_ROOT}}/interface/run_e2e.py"

{_extract_ensure_geak()}

ensure_geak
"""
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, f"ensure_geak harness failed:\n{proc.stdout}"
    return proc.stdout


def test_operator_geak_checkout_is_untouched(tmp_path: Path) -> None:
    remote = _new_remote(tmp_path)
    (remote / "kernel.py").write_text("upstream-v2\n", encoding="utf-8")
    _commit(remote, "v2")

    checkout = tmp_path / "my-geak"
    _git(tmp_path, "clone", "-q", str(remote), str(checkout))
    _git(checkout, "checkout", "-q", "-b", "my-experiment")
    (checkout / "kernel.py").write_text("my committed experiment\n", encoding="utf-8")
    _commit(checkout, "my experiment")
    with (checkout / "kernel.py").open("a", encoding="utf-8") as stream:
        stream.write("my uncommitted edit\n")

    before = _checkout_state(checkout)
    proc = _run_installer_for_geak(tmp_path, checkout, remote)

    assert proc.returncode == 0, proc.stdout
    assert _checkout_state(checkout) == before, proc.stdout


def test_missing_operator_geak_checkout_is_not_created(tmp_path: Path) -> None:
    remote = _new_remote(tmp_path)
    checkout = tmp_path / "missing-geak"

    proc = _run_installer_for_geak(tmp_path, checkout, remote)

    assert proc.returncode != 0, proc.stdout
    assert not checkout.exists(), proc.stdout


def test_operator_geak_root_must_be_a_valid_checkout(tmp_path: Path) -> None:
    remote = _new_remote(tmp_path)
    checkout = tmp_path / "invalid-geak"
    (checkout / ".git").mkdir(parents=True)
    before = tuple(path.relative_to(checkout) for path in checkout.rglob("*"))

    proc = _run_installer_for_geak(tmp_path, checkout, remote)

    assert proc.returncode != 0, proc.stdout
    assert "operator-supplied GEAK_ROOT is not a git checkout" in proc.stdout
    assert tuple(path.relative_to(checkout) for path in checkout.rglob("*")) == before


def test_operator_geak_worktree_is_accepted_and_untouched(tmp_path: Path) -> None:
    remote = _new_remote(tmp_path)
    primary = tmp_path / "primary-geak"
    _git(tmp_path, "clone", "-q", str(remote), str(primary))
    checkout = tmp_path / "worktree-geak"
    _git(primary, "worktree", "add", "-q", "-b", "my-experiment", str(checkout))
    (checkout / "kernel.py").write_text("my uncommitted edit\n", encoding="utf-8")
    before = _checkout_state(checkout)

    proc = _run_installer_for_geak(tmp_path, checkout, remote)

    assert proc.returncode == 0, proc.stdout
    assert _checkout_state(checkout) == before, proc.stdout
    assert "e2e optimizer geak checkout missing" not in proc.stdout


def test_installer_managed_geak_checkout_still_realigns(tmp_path: Path) -> None:
    remote = _new_remote(tmp_path)

    checkout = tmp_path / "cache" / "GEAK@old"
    checkout.parent.mkdir()
    _git(tmp_path, "clone", "-q", str(remote), str(checkout))
    (remote / "kernel.py").write_text("upstream-v2\n", encoding="utf-8")
    _commit(remote, "v2")
    expected_head = _git(remote, "rev-parse", "HEAD")

    proc = _run_installer_for_geak(tmp_path, checkout, remote)

    assert proc.returncode == 0, proc.stdout
    assert _git(checkout, "rev-parse", "HEAD") == expected_head
    assert (checkout / "kernel.py").read_bytes() == b"upstream-v2\n"


def test_installs_local_checkout_not_git_url(tmp_path: Path) -> None:
    out = _run_ensure_geak(tmp_path, package_metadata=True)
    geak_root = tmp_path / "os" / "GEAK"
    # Installs the local checkout, with GEAK_HOME pointing at it.
    assert f"GEAK_HOME={geak_root} python3 -m pip install" in out, out
    assert f"pip install -q --no-cache-dir --break-system-packages {geak_root}" in out, out
    # Never refetches from the remote via a git+ pip URL.
    assert "git+" not in out, out
    # setup.sh is gone.
    assert "setup.sh" not in out, out
    # SDK still installed; runner present so no "missing" warnings.
    assert "pip install -q --no-cache-dir --break-system-packages claude-agent-sdk anyio" in out, out
    assert "package metadata missing" not in out, out
    assert "e2e runner not found" not in out, out


def test_skips_pip_with_warning_when_no_package_metadata(tmp_path: Path) -> None:
    out = _run_ensure_geak(tmp_path, package_metadata=False)
    geak_root = tmp_path / "os" / "GEAK"
    assert "package metadata missing" in out, out
    # The package install must be skipped (no pip install of the checkout dir)...
    assert f"pip install -q --no-cache-dir --break-system-packages {geak_root}" not in out, out
    # ...but the SDK install still runs.
    assert "claude-agent-sdk anyio" in out, out


def test_static_guards_pip_from_checkout() -> None:
    body = _extract_ensure_geak()
    assert 'python3 -m pip install ${_PIP_FLAGS} "${GEAK_ROOT}"' in body, (
        "ensure_geak must pip-install the local ${GEAK_ROOT} checkout"
    )
    assert 'GEAK_HOME="${GEAK_ROOT}"' in body, "must pass GEAK_HOME to reuse the checkout"
    assert "git+" not in body, "must not build a git+<remote> pip URL"
    assert "setup.sh" not in body, "setup.sh path must be fully removed"
