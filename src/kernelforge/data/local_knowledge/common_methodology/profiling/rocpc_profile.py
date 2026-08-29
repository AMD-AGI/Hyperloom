#!/usr/bin/env python3
"""Self-contained on-demand rocprof-compute profiling for a GPU kernel.

Runs ROCm Compute Profiler (`rocprof-compute`) on a driver command and prints its
Top-Stats + System Speed-of-Light (and, with --roofline, the empirical roofline)
tables, plus where the raw counters landed. You then classify the bottleneck by
reading `measure_triage.md` + `measure_roofline.md` in this folder.

Design notes (why it looks like this):
  * SELF-CONTAINED: stdlib only. It does NOT import any project package, so it
    keeps working regardless of changes elsewhere in the repo. It only shells out
    to the supported `rocprof-compute` CLI.
  * ZERO-CONFIG availability: rocprof-compute needs its own Python deps — installed
    by the `[forge-profiling]` extra (`pip install -e ".[forge-profiling]"`), or by
    rocprof-compute's requirements.txt. This script AUTO-DETECTS an interpreter
    that can run the CLI — the current interpreter, the system /usr/bin/python3,
    then `python3` on PATH — and runs the profiler under the first that works. If
    none can, it prints an "unavailable" notice and SKIPS (exit 3); there is no env
    var or hand-built venv to configure.
  * DIRECT CLI: it invokes `rocprof-compute` as a subprocess (which isolates its
    sys.exit/global state) and never patches the shared /opt/rocm install. The
    profiled command runs under the CURRENT python (your kernel/torch env).
  * GENERIC: no kernel name, output layout, or bottleneck rule is baked in. It
    prints the profiler's own tables for you to interpret.

Usage:
    python3 rocpc_profile.py --driver <driver.py> [--roofline]
                             [--kernel <index>] [--out DIR]

  --driver   driver/harness that runs the kernel (e.g. forge_driver.py). REQUIRED.
  --roofline also build the empirical roofline (AI + distance-to-roof); one-time ~70s microbench.
  --kernel   isolate ONE kernel by its index from the "Top Stats" table (default: show all + aggregate).
  --out      dir to keep the raw workload/counters in (default: ./forge_profile).

Env: ROCM_PATH (optional) — used only to locate rocprofiler-compute if not at /opt/rocm.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import shutil
import signal
import subprocess
import sys


def _resolve_libexec() -> str | None:
    """Locate the rocprofiler-compute install dir (holds rocprof_compute_base.py)."""
    for root in (os.environ.get("ROCM_PATH", "").strip(), "/opt/rocm"):
        if not root:
            continue
        d = os.path.join(root, "libexec", "rocprofiler-compute")
        if os.path.isfile(os.path.join(d, "rocprof_compute_base.py")):
            return d
    return None


def _python_can_run_rocpc(python: str, libexec: str) -> bool:
    """True iff `python` can run the rocprof-compute CLI.

    A `rocprof-compute --help` under this interpreter runs the launcher's
    verify_deps preflight first, so exit 0 confirms its deps are present.
    """
    try:
        p = subprocess.run(
            [python, os.path.join(libexec, "rocprof-compute"), "--help"],
            capture_output=True, timeout=60,
        )
        return p.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _detect_rocpc_python(libexec: str) -> str | None:
    """First interpreter that can run the rocprof-compute CLI, or None."""
    seen: set[str] = set()
    for py in (sys.executable, "/usr/bin/python3", shutil.which("python3") or ""):
        py = (py or "").strip()
        if not py or py in seen:
            continue
        seen.add(py)
        if _python_can_run_rocpc(py, libexec):
            return py
    return None


# The in-flight rocprof-compute child, so an external SIGTERM (e.g. the agent's
# Bash `timeout`) can reap its whole subtree instead of orphaning rocprofv3.
_CURRENT_PROC = None


def _descendant_pids(root_pid: int) -> list[int]:
    """All descendant PIDs of ``root_pid`` via /proc PPID links (best-effort).

    PPID links survive setsid, so this reaches the rocprofv3 + driver subtree that
    rocprof-compute detaches into its own session. Returns [] on any error.
    """
    children: dict = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)
    out, stack, seen = [], list(children.get(root_pid, [])), set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        stack.extend(children.get(p, []))
    return out


def _kill_tree(pid: int) -> None:
    """SIGKILL a process, its whole descendant tree, and its process group.

    rocprof-compute drives rocprofv3 (one per counter pass) which runs the driver;
    those are detached into their own sessions, so a plain group kill misses them.
    Kill the descendant tree (via /proc) + the group so nothing is orphaned.
    """
    for p in _descendant_pids(pid):
        with contextlib.suppress(OSError):
            os.kill(p, signal.SIGKILL)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _on_terminate(signum, _frame):
    """Reap the in-flight rocprof-compute subtree on external SIGTERM/SIGINT.

    The agent runs this script under a Bash `timeout`; without this, a timeout
    SIGTERM kills only this python and orphans the stuck rocprofv3 + driver, which
    keep holding the GPU and block the caller's pipe read. Kill the whole subtree.
    """
    if _CURRENT_PROC is not None:
        _kill_tree(_CURRENT_PROC.pid)
    os._exit(128 + signum)


def _run(rocpc_python: str, libexec: str, native: list[str], cwd=None, timeout=1200):
    """Run the rocprof-compute CLI in a subprocess and capture output.

    The child runs in its OWN session (setsid). On timeout (or an external
    SIGTERM via :func:`_on_terminate`) the WHOLE descendant tree is SIGKILLed —
    rocprof-compute detaches its rocprofv3 + driver into separate sessions, so a
    plain group kill would orphan them and leave a process pinned to the GPU.
    """
    global _CURRENT_PROC
    cmd = [rocpc_python, os.path.join(libexec, "rocprof-compute"), *native]
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    _CURRENT_PROC = proc
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        try:
            out, err = proc.communicate(timeout=30)
        except Exception:
            out, err = "", ""
        tail = f"\n{out}{err}".rstrip()
        return 124, f"TIMEOUT after {timeout}s{tail}"
    finally:
        _CURRENT_PROC = None
    return proc.returncode, (out + err)


def main() -> int:
    ap = argparse.ArgumentParser(description="Self-contained rocprof-compute profiling.")
    ap.add_argument("--driver", required=True, help="driver/harness that runs the kernel")
    ap.add_argument("--roofline", action="store_true", help="also build the empirical roofline (AI)")
    ap.add_argument("--kernel", default="", help="isolate one kernel by its Top-Stats index")
    ap.add_argument("--out", default="", help="dir to keep the raw workload (default ./forge_profile)")
    a = ap.parse_args()

    # Reap the rocprof-compute subtree if we're killed externally (the agent runs
    # this under a Bash `timeout`), so a stuck rocprofv3 never orphans + poisons GPU.
    signal.signal(signal.SIGTERM, _on_terminate)
    signal.signal(signal.SIGINT, _on_terminate)

    driver_python = sys.executable  # the profiled command runs under THIS python (has torch/etc.)
    libexec = _resolve_libexec()
    if not libexec:
        print("rocprof-compute not found under $ROCM_PATH or /opt/rocm — skipping profiling.")
        return 3
    rocpc_python = _detect_rocpc_python(libexec)
    if not rocpc_python:
        print("rocprof-compute is installed, but its Python deps are not available in any detected "
              "interpreter (current / /usr/bin/python3 / python3 on PATH) — skipping profiling.")
        print("To enable it, install the forge-profiling extra (pip install -e \".[forge-profiling]\") — or "
              f"rocprof-compute's requirements.txt ({libexec}/requirements.txt) — into one of them.")
        return 3

    out = a.out or os.path.join(os.getcwd(), "forge_profile")
    os.makedirs(out, exist_ok=True)

    # 1) profile: replay the driver to collect counters (+ roofline microbench if asked).
    #    Without --roofline, restrict to the System-Speed-of-Light block (-b 2, the
    #    block analyzed below): roughly halves the counter-replay passes and skips
    #    instruction-level groups (e.g. SQ_INST_LEVEL_SMEM) that have intermittently
    #    hung rocprofv3. NOTE -b also SKIPS the roofline microbench, so it is applied
    #    only on the roofline-less path; --roofline keeps the full profile.
    prof = ["profile", "-n", "run"]
    if not a.roofline:
        prof += ["--no-roof", "-b", "2"]
    prof += ["--", driver_python, a.driver, "--profile-run"]
    rc, log = _run(rocpc_python, libexec, prof, cwd=out, timeout=1800)
    if rc != 0:
        # Deps were already verified by _detect_rocpc_python, so a failure here is
        # almost always the driver: it crashed or launched no GPU kernel.
        print("PROFILE FAILED (rocprof-compute exited non-zero). The driver most likely crashed or "
              "launched no GPU kernel — see its traceback in the tail below.")
        print("--- rocprof-compute output tail ---")
        print(log[-1800:])
        return 1

    workloads = glob.glob(os.path.join(out, "workloads", "run", "*"))
    workload = next((w for w in workloads if os.path.isdir(w)), None)
    if not workload:
        print("PROFILE produced no workload dir.")
        print(log[-1000:])
        return 1

    # 2) analyze: Top Stats (0) + System Speed-of-Light (2) [+ Roofline (4)].
    blocks = ["0", "2"] + (["4"] if a.roofline else [])
    an = ["analyze", "-p", workload, "-b", *blocks, "--max-stat-num", "6"]
    if a.kernel:
        an += ["-k", a.kernel]
    rc, report = _run(rocpc_python, libexec, an, timeout=300)
    if rc != 0:
        print("ANALYZE FAILED.")
        print(report[-1500:])
        return 1

    print(report)
    print(f"\nRaw counters + workload kept under: {workload}")
    print("How to interpret: read measure_triage.md and measure_roofline.md "
          "(same folder) — the '% of Peak' column is distance-to-ceiling; with --roofline, "
          "the Roofline section gives arithmetic intensity + distance-to-roof.")
    if not a.kernel:
        print("Tip: to isolate YOUR kernel, find its index in the 'Top Stats' table above and "
              "re-run with --kernel <index>.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
