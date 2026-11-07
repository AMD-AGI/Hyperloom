# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural guards for ``ensure_bubblewrap()`` in install.sh."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SH = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"

APT_LOG = "apt-calls"

# PATH is curated so each test controls which of apt-get and bwrap are discoverable.
_PATH_TOOLS = ("bash", "env", "cat", "echo", "printf", "dirname")


def _ensure_bubblewrap_source() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(r"^ensure_bubblewrap\(\) \{.*?^\}", text, re.S | re.M)
    assert match, "could not locate ensure_bubblewrap() in install.sh"
    return match.group(0)


def _run(
    tmp_path: Path,
    *,
    with_bwrap: bool,
    with_apt: bool,
    apt_rc: int = 0,
    dry_run: int = 0,
    check_only: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in _PATH_TOOLS:
        real = shutil.which(tool)
        if real:
            (bindir / tool).symlink_to(real)
    apt_log = tmp_path / APT_LOG
    if with_apt:
        stub = bindir / "apt-get"
        stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{apt_log}"\nexit {apt_rc}\n', encoding="utf-8")
        stub.chmod(0o755)
    if with_bwrap:
        stub = bindir / "bwrap"
        stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"DRY_RUN={dry_run}",
            f"CHECK_ONLY={check_only}",
            'log() { echo "LOG $*"; }',
            'warn() { echo "WARN $*" >&2; }',
            _ensure_bubblewrap_source(),
            "ensure_bubblewrap",
        ]
    )
    proc = subprocess.run(
        [str(bindir / "bash"), "-c", script],
        env={"PATH": str(bindir)},
        capture_output=True,
        text=True,
        check=False,
    )
    calls = apt_log.read_text(encoding="utf-8").splitlines() if apt_log.exists() else []
    return proc, calls


def test_an_installed_bwrap_is_left_alone(tmp_path):
    proc, calls = _run(tmp_path, with_bwrap=True, with_apt=True)

    assert proc.returncode == 0
    assert calls == []


def test_a_missing_bwrap_is_installed_with_apt(tmp_path):
    proc, calls = _run(tmp_path, with_bwrap=False, with_apt=True)

    assert proc.returncode == 0
    assert any("install" in call and "bubblewrap" in call for call in calls)


@pytest.mark.parametrize("mode", ["dry_run", "check_only"])
def test_dry_run_and_check_only_do_not_install(tmp_path, mode):
    proc, calls = _run(tmp_path, with_bwrap=False, with_apt=True, **{mode: 1})

    assert proc.returncode == 0
    assert calls == []


def test_without_apt_the_install_warns_and_continues(tmp_path):
    proc, _ = _run(tmp_path, with_bwrap=False, with_apt=False)

    assert proc.returncode == 0
    assert "bubblewrap" in proc.stderr


def test_a_failed_apt_install_warns_and_continues(tmp_path):
    proc, calls = _run(tmp_path, with_bwrap=False, with_apt=True, apt_rc=100)

    assert proc.returncode == 0
    assert "bubblewrap" in proc.stderr
    assert any("install" in call for call in calls)
