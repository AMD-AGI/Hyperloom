# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Behavioural guards for ensure_rocprof_compute() in install.sh.

rocprof-compute (apt package rocprofiler-compute) is required by the
kernel-agent before-GEAK roofline step. The installer must detect it first
and only apt-install when missing, fail-soft.

Behaviour contract (asserted below):

* rocprof-compute already on PATH        -> SKIP apt (no apt-get)
* missing + apt available                -> run apt-get update + install
* missing + apt available + --check-only -> NO apt-get
* missing + apt available + --dry-run    -> NO apt-get

The test extracts the real ensure_rocprof_compute body from install.sh (so
it cannot drift from the shipped script) and runs it with a PATH that
injects fake apt-get / rocprof-compute stubs.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IO_INSTALL = REPO_ROOT / "inference_optimizer" / "scripts" / "install.sh"

APT_MARKER = "apt-get-called"


def _extract_ensure_rocprof_compute() -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    block = re.search(
        r"^ROCPROF_COMPUTE_BIN=.*?^ensure_rocprof_compute\(\) \{.*?^\}",
        text,
        re.S | re.M,
    )
    assert block, "could not locate ensure_rocprof_compute block in install.sh"
    return block.group(0)


def _make_exec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path, *, rocprof_present: bool, apt_present: bool,
    check_only: int = 0, dry_run: int = 0,
) -> tuple[str, bool]:
    """Run the extracted body return (stdout, apt_called).

    rocprof presence is controlled via ROCPROF_COMPUTE_BIN/PATH pointing at a
    real stub (present) or a guaranteed-missing target (absent), so the host's
    real /usr/bin/rocprof-compute cannot leak in. apt presence is controlled by
    injecting a fake apt-get into PATH; absence by an empty fake bin.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / APT_MARKER
    rocprof_stub = tmp_path / "rocprof-compute-stub"

    if rocprof_present:
        _make_exec(rocprof_stub, "#!/bin/sh\nexit 0\n")
        rocprof_bin = "definitely-not-a-real-cmd"
        rocprof_path = str(rocprof_stub)
    else:
        rocprof_bin = "definitely-not-a-real-cmd"
        rocprof_path = str(tmp_path / "nonexistent-rocprof")

    if apt_present:
        apt_stub = fake_bin / "fake-apt-get"
        _make_exec(apt_stub, f'#!/bin/sh\ntouch "{marker}"\nexit 0\n')
        apt_bin = str(apt_stub)
    else:
        apt_bin = "definitely-not-a-real-apt"

    # PATH = fake_bin + system bins (so bash/coreutils resolve). The real
    # rocprof-compute / apt-get on PATH are neutralised by overriding the
    # command names the function consults.
    harness = f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="{fake_bin}:/usr/bin:/bin"
export ROCPROF_COMPUTE_BIN="{rocprof_bin}"
export HYPERLOOM_ROCPROF_COMPUTE_PATH="{rocprof_path}"
export ROCPROF_APT_BIN="{apt_bin}"
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
CHECK_ONLY={check_only}
DRY_RUN={dry_run}

{_extract_ensure_rocprof_compute()}

ensure_rocprof_compute
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
    assert proc.returncode == 0, f"harness failed:\n{proc.stdout}"
    return proc.stdout, marker.exists()


def test_skip_when_rocprof_present(tmp_path: Path) -> None:
    out, apt_called = _run(tmp_path, rocprof_present=True, apt_present=True)
    assert not apt_called, f"apt must not run when rocprof-compute exists:\n{out}"
    assert "rocprof-compute present" in out


def test_hyperloom_rocprof_path_override_is_honored(tmp_path: Path) -> None:
    out, apt_called = _run(tmp_path, rocprof_present=True, apt_present=True)
    assert not apt_called, f"HYPERLOOM_ROCPROF_COMPUTE_PATH should skip apt:\n{out}"
    assert "rocprof-compute present" in out


def test_install_when_missing_and_apt_available(tmp_path: Path) -> None:
    out, apt_called = _run(tmp_path, rocprof_present=False, apt_present=True)
    assert apt_called, f"apt-get should have been invoked:\n{out}"
    assert "installing rocprofiler-compute" in out


def test_no_apt_available_degrades(tmp_path: Path) -> None:
    out, apt_called = _run(tmp_path, rocprof_present=False, apt_present=False)
    assert not apt_called
    assert "apt-get unavailable" in out


def test_check_only_does_not_install(tmp_path: Path) -> None:
    out, apt_called = _run(
        tmp_path, rocprof_present=False, apt_present=True, check_only=1,
    )
    assert not apt_called, f"--check-only must not install:\n{out}"


def test_dry_run_does_not_install(tmp_path: Path) -> None:
    out, apt_called = _run(
        tmp_path, rocprof_present=False, apt_present=True, dry_run=1,
    )
    assert not apt_called, f"--dry-run must not install:\n{out}"
    assert "would install" in out
