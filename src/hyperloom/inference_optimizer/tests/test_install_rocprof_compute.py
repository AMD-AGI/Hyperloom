# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``ensure_rocprof_compute()``.

forge-loop's profiling stage prefers rocprof-compute (roofline / speed-of-light)
and only degrades to the thin rocprofv3 "PMC" path when the tool is missing OR
unusable. Two things break it on a stock ROCm serving image:

  1. rocprofiler-compute (the ``rocprof-compute`` CLI, resolved by KernelForge at
     ``<ROCM_PATH>/libexec/rocprofiler-compute/rocprof_compute_base.py``) is not
     installed — the image ships only rocprofv3.
  2. Even once installed, its 3.4.x CSV converter assumes pandas' legacy
     ``object`` string dtype; pandas>=3.0 (future.infer_string=True) makes its
     Agent_Id merge fail -> "No profiling data found" -> silent PMC fallback.

``ensure_rocprof_compute()`` (gated on the forge backend being selected AND
FORGE_PATH set) apt-installs the tool and pins ``pandas<3`` in the forge
interpreter. Every branch is FAIL-SOFT: a missing tool / failed apt / failed pin
logs and returns 0 (forge still runs on PMC) — it must never abort install.sh.

Regression cover for the 2026-07-30 investigation where every forge run profiled
on PMC (optimization-potential estimable=NO): first because rocprof-compute was
absent, then because pandas 3.0 silently disabled its CSV conversion.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"

APT_MARKER = "apt-install-called"
PIP_MARKER = "pip-install-called"

# coreutils the extracted bash + the stubs need; PATH is curated so we control
# whether apt-get is discoverable (for the no-apt branch).
_PATH_TOOLS = (
    "bash", "sh", "env", "cat", "tr", "touch", "mkdir", "rm", "printf",
    "sed", "grep", "ls", "dirname", "chmod", "ln", "cp", "head", "tail",
)


def _extract_fn(name: str) -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"could not locate {name}() in install.sh"
    return m.group(0)


def _curated_bindir(tmp_path: Path, *, with_apt: bool, apt_stub: Path | None) -> Path:
    """A PATH dir with only the tools we allow — lets us toggle apt-get presence."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for tool in _PATH_TOOLS:
        real = shutil.which(tool)
        if real:
            link = bindir / tool
            if not link.exists():
                link.symlink_to(real)
    if with_apt and apt_stub is not None:
        (bindir / "apt-get").symlink_to(apt_stub)
    return bindir


def _fake_python(tmp_path: Path) -> Path:
    """Stub ``$PYTHON``.

    * ``-m pip install ...``  -> touch PIP_MARKER, exit ``$PIP_RC`` (default 0).
    * ``-``  (version-check heredoc on stdin) -> decide the pandas version:
        - after pip ran AND ``PIP_FIXES=1``  -> print 2.3.3, exit 0 (<3)
        - else per ``PANDAS_STATE``: absent->exit 3, v2->2.3.3/exit0, v3->3.0.3/exit1
    """
    pip_marker = tmp_path / PIP_MARKER
    body = f"""#!/usr/bin/env bash
if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "pip" ]; then
  : > "{pip_marker}"
  exit ${{PIP_RC:-0}}
fi
if [ "${{1:-}}" = "-" ]; then
  cat >/dev/null 2>&1 || true   # consume the heredoc script
  if [ -f "{pip_marker}" ] && [ "${{PIP_FIXES:-0}}" = "1" ]; then
    echo "2.3.3"; exit 0
  fi
  case "${{PANDAS_STATE:-v3}}" in
    absent) exit 3 ;;
    v2) echo "2.3.3"; exit 0 ;;
    *) echo "3.0.3"; exit 1 ;;
  esac
fi
exit 0
"""
    py = tmp_path / "fake_python.sh"
    py.write_text(body, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return py


def _fake_apt(tmp_path: Path, tool_base: Path) -> Path:
    """Stub ``apt-get``: touch APT_MARKER; on ``install`` optionally create the tool."""
    apt_marker = tmp_path / APT_MARKER
    body = f"""#!/usr/bin/env bash
: > "{apt_marker}"
if [ "${{APT_CREATES_TOOL:-0}}" = "1" ] && [ "${{1:-}}" = "install" ]; then
  mkdir -p "{tool_base.parent}"
  : > "{tool_base}"
fi
exit ${{APT_RC:-0}}
"""
    apt = tmp_path / "fake_apt_get.sh"
    apt.write_text(body, encoding="utf-8")
    apt.chmod(apt.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return apt


def _run(
    tmp_path: Path,
    *,
    backend_order: str | None = "forge",
    forge_path: str | None = "/some/KernelForge",
    tool_present: bool = False,
    apt_available: bool = True,
    apt_creates_tool: bool = False,
    apt_rc: int = 0,
    pandas_state: str = "v3",
    pip_rc: int = 0,
    pip_fixes: bool = False,
    check_only: int = 0,
    dry_run: int = 0,
) -> dict:
    """Run the extracted rocprof-compute functions under set -euo pipefail."""
    rocm_root = tmp_path / "rocm"
    tool_base = rocm_root / "libexec" / "rocprofiler-compute" / "rocprof_compute_base.py"
    if tool_present:
        tool_base.parent.mkdir(parents=True, exist_ok=True)
        tool_base.write_text("", encoding="utf-8")

    fake_py = _fake_python(tmp_path)
    apt_stub = _fake_apt(tmp_path, tool_base)
    bindir = _curated_bindir(tmp_path, with_apt=apt_available, apt_stub=apt_stub)

    backend_line = (
        f'export KERNEL_OPT_BACKEND_ORDER="{backend_order}"'
        if backend_order is not None
        else "unset KERNEL_OPT_BACKEND_ORDER || true"
    )
    forge_line = (
        f'export FORGE_PATH="{forge_path}"'
        if forge_path is not None
        else "unset FORGE_PATH || true"
    )

    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
die() {{ echo "[die] $*"; exit 1; }}
CHECK_ONLY={check_only}
DRY_RUN={dry_run}
PYTHON="{fake_py}"
PIP_EXTRA=()
export ROCM_PATH="{rocm_root}"
export PANDAS_STATE="{pandas_state}"
export PIP_RC="{pip_rc}"
export PIP_FIXES="{1 if pip_fixes else 0}"
export APT_CREATES_TOOL="{1 if apt_creates_tool else 0}"
export APT_RC="{apt_rc}"
{backend_line}
{forge_line}

{_extract_fn("_forge_backend_selected")}

{_extract_fn("_ensure_pandas_lt3_for_rocpc")}

{_extract_fn("ensure_rocprof_compute")}

ensure_rocprof_compute
echo "[harness] reached-end rc=$?"
"""
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"PATH": str(bindir)},
        check=False,
    )
    return {
        "out": proc.stdout,
        "rc": proc.returncode,
        "apt_called": (tmp_path / APT_MARKER).exists(),
        "pip_called": (tmp_path / PIP_MARKER).exists(),
        "tool_exists": tool_base.exists(),
        "reached_end": "reached-end" in proc.stdout,
    }


# --- Gate -----------------------------------------------------------------


def test_skip_when_backend_not_forge(tmp_path: Path) -> None:
    r = _run(tmp_path, backend_order="geak")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"] and not r["pip_called"], r["out"]
    assert "forge backend not selected" in r["out"]


def test_forge_matched_as_substring_in_ladder(tmp_path: Path) -> None:
    # "geak,forge" must count as forge selected (ladder ordering).
    r = _run(tmp_path, backend_order="geak,forge", tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert "forge backend not selected" not in r["out"]


def test_skip_when_forge_but_no_forge_path(tmp_path: Path) -> None:
    r = _run(tmp_path, backend_order="forge", forge_path=None)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"] and not r["pip_called"], r["out"]
    assert "FORGE_PATH unset" in r["out"]


# --- Tool install (Step 1) ------------------------------------------------


def test_installs_tool_when_absent(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, apt_creates_tool=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"], r["out"]
    assert r["tool_exists"], r["out"]
    assert "rocprof-compute installed OK" in r["out"]


def test_idempotent_when_tool_present(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"], f"apt should be skipped when tool present:\n{r['out']}"
    assert "already present" in r["out"]


def test_failsoft_when_apt_unavailable(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, apt_available=False)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"], r["out"]
    assert "apt-get unavailable" in r["out"]


def test_failsoft_when_apt_fails_to_produce_tool(tmp_path: Path) -> None:
    # apt runs but does not create the tool (e.g. package unavailable / rc!=0).
    r = _run(tmp_path, tool_present=False, apt_creates_tool=False, apt_rc=100)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"], r["out"]
    assert not r["tool_exists"], r["out"]
    assert "did not produce" in r["out"]
    # No tool -> pandas pin must not run (forge is on PMC anyway).
    assert not r["pip_called"], r["out"]


# --- pandas pin (Step 2) --------------------------------------------------


def test_pins_pandas_when_ge3(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v3", pip_fixes=True)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["pip_called"], f"pandas<3 pin should have run:\n{r['out']}"
    assert "pinning to 'pandas>=2.2.3,<3'" in r["out"]
    assert "pandas pinned OK" in r["out"]


def test_no_pin_when_pandas_lt3(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["pip_called"], f"no pin needed when pandas<3:\n{r['out']}"
    assert "no pin needed" in r["out"]


def test_failsoft_when_pip_pin_fails(tmp_path: Path) -> None:
    # pandas>=3, pip install exits non-zero, version stays >=3.
    r = _run(tmp_path, tool_present=True, pandas_state="v3", pip_rc=5, pip_fixes=False)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["pip_called"], r["out"]
    assert "still incompatible after pin" in r["out"]


def test_failsoft_when_pandas_not_importable(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="absent")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["pip_called"], r["out"]
    assert "pandas not importable" in r["out"]


# --- CHECK_ONLY / DRY_RUN honour ------------------------------------------


def test_dry_run_installs_nothing(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, dry_run=1, pandas_state="v3")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"] and not r["pip_called"], r["out"]
    assert "would run" in r["out"]


def test_check_only_installs_nothing(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, check_only=1, pandas_state="v3")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"] and not r["pip_called"], r["out"]


# --- Static guards: keep the fix wired in ---------------------------------


def test_static_gated_on_forge_and_forge_path() -> None:
    body = _extract_fn("ensure_rocprof_compute")
    assert "_forge_backend_selected" in body
    assert "FORGE_PATH" in body


def test_static_installs_rocprofiler_compute_and_verifies_resolve_path() -> None:
    body = _extract_fn("ensure_rocprof_compute")
    assert "rocprofiler-compute" in body, "apt package name must stay wired"
    # verifies against the exact path KernelForge's resolve_rocpc() checks
    assert "libexec/rocprofiler-compute/rocprof_compute_base.py" in body


def test_static_pins_pandas_lt3() -> None:
    body = _extract_fn("_ensure_pandas_lt3_for_rocpc")
    assert "pandas>=2.2.3,<3" in body


def test_static_fail_soft_no_hard_abort() -> None:
    # Neither function may hard-abort the installer: no die, and no bash-level
    # `exit` STATEMENT (a line whose first token is `exit`). This deliberately
    # ignores `sys.exit(...)` inside the embedded Python version probe (that is
    # the fail-soft mechanism — it returns a code to the surrounding `if`) and
    # any "exit" mentioned in comments.
    for name in ("ensure_rocprof_compute", "_ensure_pandas_lt3_for_rocpc"):
        body = _extract_fn(name)
        assert "die " not in body, f"{name} must be fail-soft (no die)"
        assert not re.search(r"^\s*exit\b", body, re.M), (
            f"{name} must not exit the shell (only `return 0` / fail-soft)"
        )
