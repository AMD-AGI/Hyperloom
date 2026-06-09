# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Behavioural + static guards for ensure_magpie()'s idempotent reinstall.

Magpie is editable-installed into the pod-level ``/opt/venv`` that
concurrent sessions on the same Claw pod share. The old ``ensure_magpie``
reran ``pip install -e`` unconditionally on every install, briefly tearing
the egg-link down/up; a sibling session mid-``python -m Magpie`` benchmark
hit that gap and died with "No module named Magpie" (intermittent, could
follow an earlier successful run). The fix skips the editable reinstall
when the checkout exists under ``$MAGPIE_DIR`` AND ``import Magpie``
already resolves into it.

Behaviour contract (asserted below):

* already installed from $MAGPIE_DIR        -> SKIP reinstall (no pip)
* import fails (e.g. torn egg-link, fresh)  -> reinstall
* import resolves elsewhere (stale editable)-> reinstall
* no checkout present                        -> reinstall

The behavioural test extracts the real ``ensure_magpie`` body from
``install.sh`` (so it cannot silently drift from the shipped script) and
runs it with stubbed ``log``/``warn``/``git_fetch_pinned`` and a fake
``$PYTHON`` whose import-probe outcome is scripted per scenario.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
IO_INSTALL = REPO_ROOT / "inference_optimizer" / "scripts" / "install.sh"

PIP_MARKER = "pip-install-called"


def _extract_ensure_magpie() -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(r"^ensure_magpie\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "could not locate ensure_magpie() in install.sh"
    return m.group(0)


def _fake_python(tmp_path: Path, *, probe_path: str | None) -> Path:
    """A stub ``$PYTHON``.

    * ``-c 'import Magpie, os; ...'`` (the probe): prints ``probe_path`` and
      exits 0 when given, else exits 1 (import failure).
    * ``-m pip install ...``: touches PIP_MARKER and exits 0.
    * ``-c "import Magpie"`` (post-reinstall verify): exits 0.
    """
    marker = tmp_path / PIP_MARKER
    probe_line = (
        f'echo "{probe_path}"; exit 0' if probe_path is not None else "exit 1"
    )
    body = f"""#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in
    *"os.path.dirname(Magpie.__file__)"*) {probe_line} ;;
  esac
done
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  touch "{marker}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  # post-reinstall ``import Magpie`` verification
  exit 0
fi
exit 0
"""
    py = tmp_path / "fake_python.sh"
    py.write_text(body, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return py


def _run_ensure_magpie(
    tmp_path: Path, *, checkout_present: bool, probe_path_tmpl: str | None,
) -> tuple[str, bool]:
    """Run the extracted ensure_magpie body; return (stdout, pip_called)."""
    magpie_dir = tmp_path / "runtime" / "Magpie"
    magpie_dir.mkdir(parents=True)
    if checkout_present:
        (magpie_dir / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    # Resolve the probe path template against the concrete magpie_dir.
    probe_path = None
    if probe_path_tmpl is not None:
        probe_path = probe_path_tmpl.format(magpie_dir=str(magpie_dir))

    fake_py = _fake_python(tmp_path, probe_path=probe_path)

    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
git_fetch_pinned() {{ echo "[git_fetch_pinned] $*"; }}
CHECK_ONLY=0
DRY_RUN=0
MAGPIE_DIR="{magpie_dir}"
MAGPIE_REPO="https://example.invalid/Magpie.git"
MAGPIE_REF="deadbeef"
PYTHON="{fake_py}"
PIP_EXTRA=()

{_extract_ensure_magpie()}

ensure_magpie
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
    assert proc.returncode == 0, f"ensure_magpie harness failed:\n{proc.stdout}"
    pip_called = (tmp_path / PIP_MARKER).exists()
    return proc.stdout, pip_called


def test_skip_reinstall_when_already_installed_from_magpie_dir(tmp_path: Path) -> None:
    # checkout present + import resolves inside $MAGPIE_DIR -> skip pip.
    out, pip_called = _run_ensure_magpie(
        tmp_path, checkout_present=True, probe_path_tmpl="{magpie_dir}",
    )
    assert not pip_called, f"reinstall should have been skipped:\n{out}"
    assert "skipping editable reinstall" in out


def test_skip_when_import_resolves_to_subdir_of_magpie_dir(tmp_path: Path) -> None:
    # editable layouts resolve to $MAGPIE_DIR/Magpie -> still "inside" -> skip.
    out, pip_called = _run_ensure_magpie(
        tmp_path, checkout_present=True, probe_path_tmpl="{magpie_dir}/Magpie",
    )
    assert not pip_called, f"reinstall should have been skipped:\n{out}"
    assert "skipping editable reinstall" in out


def test_reinstall_when_import_fails(tmp_path: Path) -> None:
    # checkout present but import probe fails (torn egg-link / fresh) -> reinstall.
    out, pip_called = _run_ensure_magpie(
        tmp_path, checkout_present=True, probe_path_tmpl=None,
    )
    assert pip_called, f"reinstall should have run on import failure:\n{out}"
    assert "Magpie installed OK" in out


def test_reinstall_when_import_resolves_elsewhere(tmp_path: Path) -> None:
    # stale editable points outside $MAGPIE_DIR -> must reinstall, not mask it.
    out, pip_called = _run_ensure_magpie(
        tmp_path,
        checkout_present=True,
        probe_path_tmpl="/opt/venv/lib/python3.10/site-packages/Magpie",
    )
    assert pip_called, f"reinstall should have run for a stale editable:\n{out}"


def test_reinstall_when_no_checkout(tmp_path: Path) -> None:
    # No checkout under $MAGPIE_DIR -> reinstall (clone+install) regardless of import.
    out, pip_called = _run_ensure_magpie(
        tmp_path, checkout_present=False, probe_path_tmpl="{magpie_dir}",
    )
    assert pip_called, f"reinstall should have run with no checkout present:\n{out}"


# ---------------------------------------------------------------------------
# Static guard: the idempotent skip must stay wired in (no regression to the
# old unconditional ``pip install -e`` on every install).
# ---------------------------------------------------------------------------
def test_io_install_magpie_reinstall_is_idempotent_guarded() -> None:
    body = _extract_ensure_magpie()
    assert "skipping editable reinstall" in body, (
        "ensure_magpie must keep the idempotent skip branch"
    )
    # The pip reinstall must live under an else-branch (guarded), not run
    # unconditionally before it.
    assert "Magpie.__file__" in body, "missing import-resolution path probe"
    assert "pip install" in body, "reinstall path must still exist for the miss case"
