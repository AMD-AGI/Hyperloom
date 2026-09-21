# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``ensure_rocprof_compute()``."""

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

# coreutils the extracted bash + the stubs need; PATH is curated so we control whether apt-get is discoverable (for
# the no-apt branch).
_PATH_TOOLS = (
    "bash",
    "sh",
    "env",
    "cat",
    "tr",
    "touch",
    "mkdir",
    "rm",
    "printf",
    "sed",
    "grep",
    "ls",
    "dirname",
    "chmod",
    "ln",
    "cp",
    "head",
    "tail",
)


def _extract_fn(name: str) -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"could not locate {name}() in install.sh"
    return m.group(0)


def _extract_fn_optional(name: str) -> str:
    """Same, but tolerates absence so a missing helper fails the behaviour test
    that needs it rather than every test sharing this harness."""
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    return m.group(0) if m else ""


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
    """Stub ``$PYTHON``."""
    pip_marker = tmp_path / PIP_MARKER
    body = f"""#!/usr/bin/env bash
if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "pip" ]; then
  echo "$*" >> "{pip_marker}"
  exit ${{PIP_RC:-0}}
fi
# `python -m venv <dir>`: lay down a bin/python that is this same stub, so the
# venv's own pip calls land in the same log.
if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "venv" ]; then
  [ "${{VENV_RC:-0}}" = "0" ] || exit "${{VENV_RC}}"
  mkdir -p "${{3}}/bin" && cp "$0" "${{3}}/bin/python" && chmod +x "${{3}}/bin/python"
  exit 0
fi
if [ "${{1:-}}" = "-" ]; then
  _script="$(cat)"                # the heredoc; three callers share this stub
  case "$_script" in
    *_rocm_profiler*)             # wheel-package discovery
      [ -n "${{ROCPC_WHEEL_LIBEXEC:-}}" ] && printf '%s\\n' "$ROCPC_WHEEL_LIBEXEC"
      exit 0 ;;
    *_rocm_sdk_*)                 # ROCm runtime tree holding rocprofiler-sdk
      [ -n "${{ROCM_SDK_ROOT:-}}" ] && printf '%s\\n' "$ROCM_SDK_ROOT"
      exit 0 ;;
    *importlib.metadata*)         # installed `rocm` version; empty on a classic stack
      [ "${{ROCM_WHEEL:-0}}" = "1" ] || exit 1
      printf '%s\\n' "${{ROCM_VERSION:-10.0.0}}"; exit 0 ;;
  esac
  # The exact spec the pin installs -- a bare "pandas" also matches tmp_path
  # names and the requirements.txt path the analyze venv installs from.
  if grep -qF 'pandas>=2.2.3,<3' "{pip_marker}" 2>/dev/null && [ "${{PIP_FIXES:-0}}" = "1" ]; then
    echo "2.3.3"; exit 0
  fi
  case "${{PANDAS_STATE:-v3}}" in
    absent) exit 3 ;;
    v2) echo "2.3.3"; exit 0 ;;
    *) echo "3.0.3"; exit 1 ;;
  esac
fi
# `<libexec>/rocprof-compute --help` probe from _rocpc_effective_python.
case "${{1:-}}" in
  *rocprof-compute) exit ${{PROBE_RC:-0}} ;;
esac
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
# Emit a diagnostic line (like real apt) so the caller's failure tail has content.
echo "apt-get $*: E: simulated apt output" >&2
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
    forge_path: str | None = None,
    repo_root: str | None = None,
    backend_order: str | None = "geak",
    tool_present: bool = False,
    apt_available: bool = True,
    apt_creates_tool: bool = False,
    apt_rc: int = 0,
    pandas_state: str = "v3",
    pip_rc: int = 0,
    pip_fixes: bool = False,
    probe_rc: int = 0,
    tmpdir: str | None = None,
    check_only: int = 0,
    dry_run: int = 0,
    skip_forge_profiling: str | None = None,
    rocm_wheel: bool = False,
    rocm_version: str = "10.0.0",
    wheel_tool_present: bool = False,
    tool_requirements: bool = True,
    venv_rc: int = 0,
    aqlprofile: str = "none",
) -> dict:
    """Run the extracted rocprof-compute functions under set -euo pipefail."""
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers may pass a subdir

    def _lay_down_tool(libexec: Path) -> None:
        libexec.mkdir(parents=True, exist_ok=True)
        (libexec / "rocprof_compute_base.py").write_text("", encoding="utf-8")
        if tool_requirements:
            (libexec / "requirements.txt").write_text("numpy==1.26.4\npandas==2.2.3\n", encoding="utf-8")

    rocm_root = tmp_path / "rocm"
    tool_base = rocm_root / "libexec" / "rocprofiler-compute" / "rocprof_compute_base.py"
    if tool_present:
        _lay_down_tool(tool_base.parent)

    # TheRock's pip ROCm ships the profiler in its own `_rocm_profiler` wheel,
    # NOT under $ROCM_PATH, which points at the separate `_rocm_sdk_devel` package.
    wheel_libexec = tmp_path / "site" / "_rocm_profiler" / "libexec" / "rocprofiler-compute"
    if wheel_tool_present:
        _lay_down_tool(wheel_libexec)

    venv_dir = tmp_path / "rocpc-venv"

    # The ROCm runtime wheel: rocprofiler-sdk plus the aqlprofile library whose
    # unversioned soname some images omit.
    sdk_root = tmp_path / "site" / "_rocm_sdk_core"
    sdk_lib = sdk_root / "lib"
    (sdk_lib / "rocprofiler-sdk").mkdir(parents=True, exist_ok=True)
    if aqlprofile in ("versioned-only", "both"):
        (sdk_lib / "libhsa-amd-aqlprofile64.so.1").write_text("", encoding="utf-8")
    if aqlprofile == "both":
        (sdk_lib / "libhsa-amd-aqlprofile64.so").write_text("", encoding="utf-8")

    # $FORGE_PATH is no longer read by this function at all; the tests set it only to prove that.
    forge_line = f'export FORGE_PATH="{forge_path}"' if forge_path is not None else "unset FORGE_PATH || true"

    fake_py = _fake_python(tmp_path)
    apt_stub = _fake_apt(tmp_path, tool_base)
    bindir = _curated_bindir(tmp_path, with_apt=apt_available, apt_stub=apt_stub)

    backend_line = (
        f'export KERNEL_OPT_BACKEND_ORDER="{backend_order}"'
        if backend_order is not None
        else "unset KERNEL_OPT_BACKEND_ORDER || true"
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
REPO_ROOT="{repo_root if repo_root is not None else REPO_ROOT}"
export ROCM_PATH="{rocm_root}"
export PANDAS_STATE="{pandas_state}"
export PIP_RC="{pip_rc}"
export PIP_FIXES="{1 if pip_fixes else 0}"
export APT_CREATES_TOOL="{1 if apt_creates_tool else 0}"
export APT_RC="{apt_rc}"
export PROBE_RC="{probe_rc}"
export ROCM_WHEEL="{1 if rocm_wheel else 0}"
export ROCM_VERSION="{rocm_version}"
export VENV_RC="{venv_rc}"
export ROCPC_VENV="{venv_dir}"
export ROCM_SDK_ROOT="{sdk_root}"
export ROCPC_WHEEL_LIBEXEC="{wheel_libexec if wheel_tool_present else ""}"
{f'export TMPDIR="{tmpdir}"' if tmpdir is not None else "true"}
{backend_line}
{forge_line}
{f'export SKIP_FORGE_PROFILING="{skip_forge_profiling}"' if skip_forge_profiling is not None else "unset SKIP_FORGE_PROFILING || true"}

{_extract_fn_optional("_rocpc_libexec_dir")}

{_extract_fn_optional("_rocm_wheel_version")}

{_extract_fn_optional("_rocpc_venv_dir")}

{_extract_fn_optional("_ensure_rocpc_analyze_venv")}

{_extract_fn_optional("_rocm_sdk_runtime_root")}

{_extract_fn_optional("_ensure_aqlprofile_soname")}

{_extract_fn("_rocpc_effective_python")}

{_extract_fn("_pandas_major_ge3")}

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
    pip_log = tmp_path / PIP_MARKER
    pip_calls = pip_log.read_text(encoding="utf-8").splitlines() if pip_log.exists() else []
    return {
        "out": proc.stdout,
        "rc": proc.returncode,
        "apt_called": (tmp_path / APT_MARKER).exists(),
        "pip_calls": pip_calls,
        "pip_called": bool(pip_calls),
        # The two distinct pip steps this function performs.
        "extra_installed": any("forge-profiling" in call for call in pip_calls),
        "pandas_pinned": any("pandas>=2.2.3,<3" in call for call in pip_calls),
        "tool_exists": tool_base.exists(),
        "reached_end": "reached-end" in proc.stdout,
        "venv_python_exists": (venv_dir / "bin" / "python").exists(),
        "sdk_lib": sdk_lib,
        # The venv must be filled from the tool's own requirements.txt, never a
        # list this repository maintains in parallel.
        "reqs_installed": any("requirements.txt" in call for call in pip_calls),
    }


# --- Gate: none. Runs unconditionally (the ordering fix) ------------------


def test_installs_under_default_geak(tmp_path: Path) -> None:
    # THE key regression: install.sh runs under geak (forge is set only later at optimize time), so a geak install
    # MUST still set forge's profiling up.
    r = _run(
        tmp_path,
        backend_order="geak",
        tool_present=False,
        apt_creates_tool=True,
        pandas_state="v3",
        pip_fixes=True,
    )
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"], f"tool must install even under geak:\n{r['out']}"
    assert r["pandas_pinned"], f"pandas pin must run even under geak:\n{r['out']}"
    assert "forge backend not selected" not in r["out"]


def test_installs_when_backend_unset(tmp_path: Path) -> None:
    r = _run(tmp_path, backend_order=None, tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert "ensuring roofline profiling deps" in r["out"]


def test_runs_with_forge_path_unset(tmp_path: Path) -> None:
    # Regression for the vendoring: forge ships in this distribution, so an unset FORGE_PATH is the normal case.
    r = _run(tmp_path, forge_path=None, tool_present=False, apt_creates_tool=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"], r["out"]
    assert r["extra_installed"], r["out"]
    assert "FORGE_PATH" not in r["out"], f"FORGE_PATH must no longer take part in the decision:\n{r['out']}"


def test_a_stale_forge_path_changes_nothing(tmp_path: Path) -> None:
    # The mirror image: a leftover pointer at a directory that is not a forge checkout must not resurrect the old gate
    # and skip the install.
    stale = tmp_path / "stale-checkout"
    stale.mkdir()
    r = _run(tmp_path, forge_path=str(stale), tool_present=False, apt_creates_tool=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"] and r["extra_installed"], r["out"]


# --- Step 0: the forge-profiling extra ------------------------------------


def test_installs_the_forge_profiling_extra(tmp_path: Path) -> None:
    # The tool is a Python program; without dash/kaleido/matplotlib/plotille/tqdm it cannot run.
    r = _run(tmp_path, tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["extra_installed"], f"the forge-profiling extra must be installed:\n{r['pip_calls']}"
    assert any("[forge-profiling]" in call for call in r["pip_calls"]), r["pip_calls"]


def test_forge_profiling_extra_is_installed_editable(tmp_path: Path) -> None:
    """Same shape as the main install, or pip replaces it with a copy."""
    r = _run(tmp_path, tool_present=True, pandas_state="v2")
    calls = [c for c in r["pip_calls"] if "[forge-profiling]" in c]
    assert calls, r["pip_calls"]
    for call in calls:
        assert " -e " in f" {call} ", f"the forge-profiling install must be editable: {call}"


def test_skip_forge_profiling_opts_out(tmp_path: Path) -> None:
    """``SKIP_FORGE_PROFILING=1`` is an opt-OUT, and only skips this one step."""
    # pandas 3 so the later pin step has work to do: the opt-out must skip the extra and nothing else.
    r = _run(tmp_path, tool_present=True, pandas_state="v3", skip_forge_profiling="1")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["extra_installed"], f"SKIP_FORGE_PROFILING=1 must skip the extra:\n{r['pip_calls']}"
    assert "SKIP_FORGE_PROFILING=1" in r["out"], "the skip must be logged, not silent"
    assert r["pandas_pinned"], f"the opt-out must skip only the extra:\n{r['pip_calls']}"


def test_forge_profiling_installs_when_the_opt_out_is_not_1(tmp_path: Path) -> None:
    """Only the exact string ``1`` opts out; anything else keeps the default."""
    for value in ("0", "", "true", "yes"):
        r = _run(tmp_path / f"v{value or 'empty'}", tool_present=True, pandas_state="v2", skip_forge_profiling=value)
        assert r["extra_installed"], f"SKIP_FORGE_PROFILING={value!r} must not skip: {r['pip_calls']}"


def test_forge_profiling_extra_install_is_fail_soft(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2", pip_rc=7)
    assert r["rc"] == 0 and r["reached_end"], f"a failed extra install must not abort install.sh:\n{r['out']}"
    assert "installing the forge-profiling extra failed" in r["out"]


def test_forge_profiling_fallback_never_names_the_distribution(tmp_path: Path) -> None:
    """A packaged install has no checkout, and must not re-resolve itself."""
    r = _run(tmp_path, repo_root="", tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not any("hyperloom-inference_optimizer[" in call for call in r["pip_calls"]), (
        f"the fallback must not name the distribution:\n{r['pip_calls']}"
    )


# --- Tool install (Step 1) ------------------------------------------------


def test_installs_tool_when_absent(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, apt_creates_tool=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"] and r["tool_exists"], r["out"]
    assert "rocprof-compute installed OK" in r["out"]


def test_finds_the_tool_in_the_rocm_profiler_wheel(tmp_path: Path) -> None:
    """TheRock's pip ROCm ships the profiler in `_rocm_profiler`, not under $ROCM_PATH.

    $ROCM_PATH points at `_rocm_sdk_devel` there, so a $ROCM_PATH-only lookup
    never sees an installed tool and every run degrades to PMC.
    """
    r = _run(tmp_path, tool_present=False, wheel_tool_present=True, rocm_wheel=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert "already present" in r["out"], f"the wheel-packaged tool must be found:\n{r['out']}"
    assert not r["apt_called"], f"nothing to install; apt must not run:\n{r['out']}"


def test_installs_the_profiler_wheel_on_a_pip_rocm_stack(tmp_path: Path) -> None:
    """apt carries no profiler package on a wheel-ROCm image; pip does."""
    r = _run(tmp_path, tool_present=False, rocm_wheel=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"], f"apt has no such package on this stack:\n{r['out']}"
    calls = [c for c in r["pip_calls"] if "rocm-profiler" in c]
    assert calls, f"the profiler wheel must be installed via pip:\n{r['pip_calls']}"
    for call in calls:
        assert "stable.repo.amd.com/rocm/whl-next" in call, f"must reach AMD's ROCm index: {call}"


def test_profiler_install_is_pinned_to_the_installed_rocm_version(tmp_path: Path) -> None:
    """Asking for the `rocm` metapackage lets pip re-resolve the whole SDK.

    Observed on the py3.12 leg images: `rocm[profiler]` backtracked to 7.14.1
    and reinstalled rocm-sdk-core over the 10.0.0 one torch is built against.
    Pinning the profiler to the version already installed removes the freedom
    pip needs to move anything else.
    """
    r = _run(tmp_path, tool_present=False, rocm_wheel=True, rocm_version="10.0.0", pandas_state="v2")
    calls = [c for c in r["pip_calls"] if "rocm-profiler" in c]
    assert calls, r["pip_calls"]
    for call in calls:
        assert "rocm-profiler==10.0.0" in call, f"the profiler must be pinned to the SDK version: {call}"
        assert "rocm[profiler]" not in call, f"the metapackage lets pip move the SDK: {call}"
        # An --index-url REPLACES the image's own configuration; only whl-next
        # carries 10.0.0, and some images reach it only through their own config.
        assert "--index-url" not in call.replace("--extra-index-url", ""), (
            f"must extend, not replace, the image's index configuration: {call}"
        )


def test_profiler_install_follows_the_reported_rocm_version(tmp_path: Path) -> None:
    """The pin tracks whatever the image actually has, not a baked-in number."""
    r = _run(tmp_path, tool_present=False, rocm_wheel=True, rocm_version="7.14.1", pandas_state="v2")
    calls = [c for c in r["pip_calls"] if "rocm-profiler" in c]
    assert calls, r["pip_calls"]
    assert all("rocm-profiler==7.14.1" in c for c in calls), r["pip_calls"]


def test_keeps_apt_on_a_classic_rocm_image(tmp_path: Path) -> None:
    """A /opt/rocm image has no `rocm` distribution; the apt path stays its route."""
    r = _run(tmp_path, tool_present=False, rocm_wheel=False, apt_creates_tool=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["apt_called"], f"the classic image must still use apt:\n{r['out']}"
    assert not any("rocm-profiler" in c for c in r["pip_calls"]), (
        f"pip must not be used for the profiler on a non-wheel stack:\n{r['pip_calls']}"
    )


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
    assert r["apt_called"] and not r["tool_exists"], r["out"]
    assert "did not produce" in r["out"]
    assert "apt| " in r["out"], f"apt output tail should be surfaced:\n{r['out']}"
    # No tool -> pandas pin must not run (forge is on PMC anyway).
    assert not r["pandas_pinned"], r["pip_calls"]


def test_failsoft_when_apt_log_never_created(tmp_path: Path) -> None:
    # Regression: an unwritable TMPDIR makes the `>"$apt_log"` redirect fail, so the log is never created.
    missing_tmp = tmp_path / "no_such_tmpdir"  # deliberately never created
    r = _run(
        tmp_path,
        tool_present=False,
        apt_available=True,
        apt_creates_tool=False,
        tmpdir=str(missing_tmp),
    )
    assert r["rc"] == 0 and r["reached_end"], f"missing apt_log must stay fail-soft (no abort):\n{r['out']}"
    assert "did not produce" in r["out"]


# --- analyze venv ---------------------------------------------------------


def test_builds_the_analyze_venv_from_the_tool_requirements(tmp_path: Path) -> None:
    """rocprof-compute's analyze mode gates on the exact pins in its own
    requirements.txt. Meeting them in the serving image would pull numpy and
    pandas out from under torch, so they go in a private venv instead."""
    r = _run(tmp_path, tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["venv_python_exists"], f"the analyze venv must be created:\n{r['out']}"
    assert r["reqs_installed"], f"it must be filled from the tool's requirements.txt:\n{r['pip_calls']}"


def test_analyze_venv_is_skipped_when_the_tool_is_absent(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, apt_available=False)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["venv_python_exists"], f"nothing to profile with; no venv:\n{r['out']}"


def test_analyze_venv_is_fail_soft_when_it_cannot_be_created(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2", venv_rc=1)
    assert r["rc"] == 0 and r["reached_end"], f"a venv failure must not abort install.sh:\n{r['out']}"
    assert not r["venv_python_exists"], r["out"]
    assert "analyze venv" in r["out"], "the degradation must be logged, not silent"


def test_analyze_venv_is_fail_soft_when_its_deps_fail(tmp_path: Path) -> None:
    """numpy==1.26.4 has no wheel on every interpreter these images ship."""
    r = _run(tmp_path, tool_present=True, pandas_state="v2", pip_rc=1)
    assert r["rc"] == 0 and r["reached_end"], f"a dep failure must not abort install.sh:\n{r['out']}"
    assert "analyze venv" in r["out"], r["out"]


def test_analyze_venv_tolerates_a_tool_without_requirements(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2", tool_requirements=False)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["reqs_installed"], r["pip_calls"]


def test_dry_run_builds_no_analyze_venv(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, dry_run=1, pandas_state="v3")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["venv_python_exists"], r["out"]


def test_check_only_builds_no_analyze_venv(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, check_only=1, pandas_state="v3")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["venv_python_exists"], r["out"]


def test_static_analyze_venv_never_hardcodes_a_dependency_list() -> None:
    body = _extract_fn("_ensure_rocpc_analyze_venv")
    assert "requirements.txt" in body, "the tool's own file is the source of truth"
    for pinned in ("numpy", "pandas", "dash", "textual"):
        assert pinned not in body, f"{pinned} must not be pinned here; that list goes stale per ROCm release"


def test_static_analyze_venv_path_matches_the_profiling_script() -> None:
    # rocpc_profile.py finds this venv by path alone, with no handshake.
    body = _extract_fn("_rocpc_venv_dir")
    assert "/opt/rocprof-compute-venv" in body
    assert "ROCPC_VENV" in body, "the default must stay overridable"


# --- aqlprofile soname ----------------------------------------------------


def test_restores_the_missing_aqlprofile_soname(tmp_path: Path) -> None:
    """rocprofiler-sdk dlopens the unversioned name, and the rocm-sdk-core wheel
    ships only libhsa-amd-aqlprofile64.so.1. Without the link, every profile run
    aborts in the sdk with SIGABRT and hangs instead of reporting anything."""
    r = _run(tmp_path, tool_present=True, pandas_state="v2", aqlprofile="versioned-only")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    link = r["sdk_lib"] / "libhsa-amd-aqlprofile64.so"
    assert link.is_symlink(), f"the soname link must be created:\n{r['out']}"
    assert link.resolve().name == "libhsa-amd-aqlprofile64.so.1", link.resolve()


def test_leaves_an_existing_aqlprofile_soname_alone(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2", aqlprofile="both")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    link = r["sdk_lib"] / "libhsa-amd-aqlprofile64.so"
    assert link.is_file() and not link.is_symlink(), "an image that ships the soname must be untouched"


def test_aqlprofile_is_fail_soft_when_the_library_is_absent(tmp_path: Path) -> None:
    """A classic /opt/rocm image has no wheel to patch."""
    r = _run(tmp_path, tool_present=True, pandas_state="v2", aqlprofile="none")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not (r["sdk_lib"] / "libhsa-amd-aqlprofile64.so").exists(), r["out"]


def test_dry_run_creates_no_aqlprofile_link(tmp_path: Path) -> None:
    for mode, flags in (("dry", {"dry_run": 1}), ("check", {"check_only": 1})):
        r = _run(tmp_path / mode, tool_present=True, pandas_state="v2", aqlprofile="versioned-only", **flags)
        assert r["rc"] == 0 and r["reached_end"], r["out"]
        assert not (r["sdk_lib"] / "libhsa-amd-aqlprofile64.so").exists(), f"{mode} must touch nothing:\n{r['out']}"


def test_static_aqlprofile_link_is_relative() -> None:
    # An absolute link breaks the moment the venv or image layer moves.
    body = _extract_fn("_ensure_aqlprofile_soname")
    assert "libhsa-amd-aqlprofile64.so.1" in body
    assert "ln -s" in body


# --- pandas pin (Step 2) --------------------------------------------------


def test_pins_pandas_when_ge3(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v3", pip_fixes=True)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["pandas_pinned"], f"pandas<3 pin should have run:\n{r['out']}"
    assert "installing 'pandas>=2.2.3,<3'" in r["out"]
    assert "forge profiling can use rocprof-compute (roofline)" in r["out"]


def test_no_pin_when_pandas_lt3(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, pandas_state="v2")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["pandas_pinned"], f"no pin needed when pandas<3:\n{r['out']}"
    assert "no pin needed" in r["out"]


def test_installs_pandas_when_absent_precludes_later_3x(tmp_path: Path) -> None:
    # Regression for the ordering window: if pandas is not yet installed, we must proactively install pandas<3 (not
    # skip) so a later unconstrained `pip install pandas` (datasets/evaluate) cannot drag pandas>=3 back in.
    r = _run(tmp_path, tool_present=True, pandas_state="absent", pip_fixes=True)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["pandas_pinned"], f"pandas<3 should be installed when absent:\n{r['out']}"
    assert "pandas not yet installed" in r["out"]
    assert "forge profiling can use rocprof-compute (roofline)" in r["out"]


def test_probe_fallback_warns_but_still_pins(tmp_path: Path) -> None:
    # If no candidate interpreter can run `rocprof-compute --help`, fall back to $PYTHON (best effort) with a warning
    # — but still pin, never abort.
    r = _run(tmp_path, tool_present=True, pandas_state="v3", pip_fixes=True, probe_rc=1)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert "could not confirm which interpreter" in r["out"]
    assert r["pandas_pinned"], f"pin must still run on the $PYTHON fallback:\n{r['out']}"


def test_failsoft_when_pip_pin_fails(tmp_path: Path) -> None:
    # pandas>=3, pip install exits non-zero, version stays >=3.
    r = _run(tmp_path, tool_present=True, pandas_state="v3", pip_rc=5, pip_fixes=False)
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert r["pandas_pinned"], r["pip_calls"]
    assert "pandas still incompatible in" in r["out"]


# --- CHECK_ONLY / DRY_RUN honour ------------------------------------------


def test_dry_run_names_the_route_it_would_take(tmp_path: Path) -> None:
    """A dry run that names apt on a wheel stack describes a route that cannot
    work there, and sends whoever reads it after the wrong problem."""
    wheel = _run(tmp_path / "wheel", tool_present=False, rocm_wheel=True, dry_run=1)
    assert wheel["rc"] == 0 and wheel["reached_end"], wheel["out"]
    assert "rocm-profiler==10.0.0" in wheel["out"], f"must name the pip route:\n{wheel['out']}"
    assert "apt-get install" not in wheel["out"], f"apt cannot supply it here:\n{wheel['out']}"

    classic = _run(tmp_path / "classic", tool_present=False, rocm_wheel=False, dry_run=1)
    assert classic["rc"] == 0 and classic["reached_end"], classic["out"]
    assert "apt-get install" in classic["out"], f"the classic route is still apt:\n{classic['out']}"


def test_dry_run_installs_nothing(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=False, dry_run=1, pandas_state="v3")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"] and not r["pip_called"], r["out"]
    assert "would run" in r["out"]


def test_check_only_installs_nothing_and_warns_pandas(tmp_path: Path) -> None:
    r = _run(tmp_path, tool_present=True, check_only=1, pandas_state="v3")
    assert r["rc"] == 0 and r["reached_end"], r["out"]
    assert not r["apt_called"] and not r["pip_called"], r["out"]
    assert "would pin 'pandas>=2.2.3,<3'" in r["out"]


# --- Static guards: keep the fix wired in ---------------------------------


def test_static_ungated() -> None:
    body = _extract_fn("ensure_rocprof_compute")
    # Must NOT gate on the backend order (that was the ordering bug)...
    assert "_forge_backend_selected" not in body
    assert "backend not selected" not in body
    # ...nor on a KernelForge checkout, which no longer exists: forge ships in this distribution, so such a gate would
    # be a permanent skip.
    assert "_kernel_forge_root" not in body
    assert "FORGE_PATH" not in body


def test_static_call_ordered_after_all_pip_steps() -> None:
    # The bare top-level call must run after EVERY pip-installing step, so no later step can re-pull pandas>=3 and the
    # pin's re-check is the truthful final state.
    text = IO_INSTALL.read_text(encoding="utf-8")
    call = text.rindex("\nensure_rocprof_compute\n")

    # Top-level invocations: a line that is exactly a function name (col 0), i.e. ensure_*/chain_* CALLS (definitions
    # are `name() {`, which this won't match).
    invocations = [(m.start(), m.group(1)) for m in re.finditer(r"^(ensure_[a-z_]+|chain_[a-z_]+)$", text, re.M)]
    pip_steps_before = []
    for pos, name in invocations:
        if name == "ensure_rocprof_compute":
            continue
        # Only consider names that are DEFINED in this file, and whose body runs a pip install.
        if not re.search(rf"^{name}\(\) \{{", text, re.M):
            continue
        body = _extract_fn(name)
        if "pip install" in body:
            pip_steps_before.append((pos, name))

    # chain_kernel_agent installs INDIRECTLY (it shells out to another install.sh), so it has no literal "pip install"
    # in its body — include it explicitly as a pip step that must precede the pin.
    chain = text.rindex("\nchain_kernel_agent\n") + 1
    pip_steps_before.append((chain, "chain_kernel_agent"))

    assert pip_steps_before, "expected to find pip-installing steps to order against"
    latest_pos, latest_name = max(pip_steps_before)
    assert call > latest_pos, (
        f"ensure_rocprof_compute (pos {call}) must be called AFTER every "
        f"pip-installing step; last one is {latest_name} (pos {latest_pos})"
    )


def test_static_effective_python_probe_mirrors_resolve_rocpc() -> None:
    body = _extract_fn("_rocpc_effective_python")
    # Same candidate order KernelForge's resolve_rocpc() uses.
    assert '"$PYTHON"' in body
    assert "/usr/bin/python3" in body
    assert "command -v python3" in body
    assert "rocprof-compute" in body and "--help" in body


def test_static_installs_rocprofiler_compute_and_verifies_resolve_path() -> None:
    body = _extract_fn("ensure_rocprof_compute")
    assert "rocprofiler-compute" in body, "apt package name must stay wired"


def test_static_layout_helpers_exist() -> None:
    # The behaviour tests run these through a tolerant extractor; assert here that they are really in install.sh.
    for name in ("_rocpc_libexec_dir", "_rocm_wheel_version"):
        assert _extract_fn(name).strip(), f"{name}() must stay defined in install.sh"


def test_static_discovery_covers_both_rocm_layouts() -> None:
    body = _extract_fn("_rocpc_libexec_dir")
    assert "libexec/rocprofiler-compute" in body, "classic /opt/rocm tree must stay covered"
    assert "_rocm_profiler" in body, "TheRock's pip wheel package must be covered"
    assert "rocprof_compute_base.py" in body, "presence must be proven by the tool's entry module"


def test_static_pins_pandas_lt3() -> None:
    body = _extract_fn("_ensure_pandas_lt3_for_rocpc")
    assert "pandas>=2.2.3,<3" in body


def test_static_fail_soft_no_hard_abort() -> None:
    # Neither function may hard-abort the installer: no die, and no bash-level `exit` STATEMENT (a line whose first
    # token is `exit`).
    for name in ("ensure_rocprof_compute", "_ensure_pandas_lt3_for_rocpc"):
        body = _extract_fn(name)
        assert "die " not in body, f"{name} must be fail-soft (no die)"
        assert not re.search(r"^\s*exit\b", body, re.M), f"{name} must not exit the shell (only `return 0` / fail-soft)"
