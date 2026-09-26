# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate the fusion harness into the driver contract the forge-loop reads."""

from __future__ import annotations

from pathlib import Path

_SHIM_TEMPLATE = '''\
"""Generated driver: runs the fusion harness and reports the loop's contract."""

import json
import os
import subprocess
import sys
import tempfile

HARNESS = {harness!r}
ENV_FLAGS = {env_flags!r}
CASE_ID = {case_id!r}
REPORT_LOG = {report_log!r}
FUSED_MODULE = {fused_module!r}
WORKSPACE = {workspace!r}
GIT_ENV = {git_env!r}


def _fused_kernel_authored():
    """Whether a fused kernel exists yet, read off the tracked module.

    The campaign commits that module EMPTY, so the loop's pristine bench runs
    with nothing to compile and a harness that reports ``compiled: false``
    there is describing the baseline, not a failure. Deciding on the file
    rather than on the report keeps a real compile failure loud: an author who
    mislabels one cannot turn it into the other.
    """
    if not FUSED_MODULE:
        return True
    try:
        return os.path.getsize(FUSED_MODULE) > 0
    except OSError:
        return False


def _git(env, *args):
    return subprocess.run(
        ["git", *args],
        cwd=WORKSPACE, capture_output=True, text=True, env=env, timeout=300,
    )


def _measured_tree():
    """Name the tracked tree this run measured, the way git names it.

    The loop stages EVERY tracked modification into its keep commit, so the
    whole tracked tree -- not the recipe's declared source files -- is the
    candidate a report describes. Asking git for the tree id over a scratch
    index also puts both sides of the match on blob bytes, so a .gitattributes
    filter that rewrites a file on its way into the index cannot make the
    worktree and the commit disagree.
    """
    if not WORKSPACE:
        return ""
    with tempfile.TemporaryDirectory() as scratch:
        env = dict(os.environ)
        env.update(GIT_ENV)
        env["GIT_INDEX_FILE"] = os.path.join(scratch, "index")
        for args in (("read-tree", "HEAD"), ("add", "-u")):
            if _git(env, *args).returncode != 0:
                return ""
        written = _git(env, "write-tree")
        return written.stdout.strip() if written.returncode == 0 else ""


def _record(report):
    """Append one harness report so the campaign can recover what it measured.

    The loop's result carries a speedup and nothing else, so the parity and
    per-arm timings would otherwise be lost by the time the manifest is written.
    One short line per append keeps concurrent lanes from interleaving.
    """
    identified = dict(report)
    identified["tracked_tree"] = _measured_tree()
    with open(REPORT_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(identified, sort_keys=True) + "\\n")


def _harness_json(env):
    proc = subprocess.run(
        [sys.executable, HARNESS],
        capture_output=True, text=True, env=env, timeout={timeout},
    )
    sys.stderr.write(proc.stderr)
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{{") and line.endswith("}}"):
            return json.loads(line)
    raise SystemExit(
        "harness printed no JSON object as its last stdout line:\\n" + proc.stdout[-2000:]
    )


def main():
    env = dict(os.environ)
    for flag in ENV_FLAGS:
        env[flag] = "1"
    report = _harness_json(env)
    _record(report)

    if not report.get("compiled", False) and _fused_kernel_authored():
        print("SNR: -99.00 dB")
        print("COMPILE FAILED: " + str(report.get("error") or "unknown"))
        return 1

    parity = report.get("parity") or []
    snrs = [p.get("snr_db") for p in parity if p.get("snr_db") is not None]
    errs = [p.get("max_abs_err") for p in parity if p.get("max_abs_err") is not None]
    if snrs:
        print("SNR: %.2f dB" % min(snrs))
    if errs:
        print("max_diff: %.6e" % max(errs))
    if not snrs and not errs:
        print("SNR: -99.00 dB")
        print("PARITY MISSING: harness reported no comparable shape")
        return 1

    eager_us = report.get("eager_us")
    fused_us = report.get("fused_us")
    if report.get("skipped"):
        # A microbench the harness declined to run (the Mamba/SSM backend cannot init on
        # ROCm) is not a failure: parity still decided correctness, so report the eager
        # time for both arms and let the loop see no speedup rather than an error.
        print("SKIPPED: " + str(report.get("skip_reason") or "microbench unavailable"))
        if eager_us:
            print("case_ms: %s %.6f" % (CASE_ID, float(eager_us) / 1000.0))
            print("wall_ms: %.6f" % (float(eager_us) / 1000.0))
        return 0

    if not fused_us and _fused_kernel_authored():
        print("BENCH MISSING: harness ran the microbench but reported no fused_us")
        return 1

    # With no kernel authored yet there is no fused arm to time, so the eager time
    # anchors the case: that pristine run IS the baseline, not a failed measurement.
    case_us = fused_us or eager_us
    if not case_us:
        print("BENCH MISSING: harness reported no timing for either arm")
        return 1

    print("case_ms: %s %.6f" % (CASE_ID, float(case_us) / 1000.0))
    print("wall_ms: %.6f" % (float(case_us) / 1000.0))
    if eager_us:
        print("eager_ms: %.6f" % (float(eager_us) / 1000.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_driver(
    harness_path: str,
    env_flags: tuple[str, ...] | list[str],
    *,
    report_log: str,
    case_id: str = "decode",
    timeout_sec: int = 1800,
    fused_module: str = "",
    workspace: str = "",
    git_env: dict[str, str] | None = None,
) -> str:
    """Render the driver source for one recipe's harness."""
    return _SHIM_TEMPLATE.format(
        harness=str(harness_path),
        env_flags=tuple(env_flags),
        case_id=case_id,
        timeout=int(timeout_sec),
        report_log=str(report_log),
        fused_module=str(fused_module),
        workspace=str(workspace),
        git_env=dict(git_env or {}),
    )


def write_driver(
    destination: str | Path,
    harness_path: str,
    env_flags: tuple[str, ...] | list[str],
    *,
    report_log: str,
    case_id: str = "decode",
    timeout_sec: int = 1800,
    fused_module: str = "",
    workspace: str = "",
    git_env: dict[str, str] | None = None,
) -> str:
    """Write the driver next to the campaign artifacts and return its path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_driver(
            harness_path,
            env_flags,
            report_log=report_log,
            case_id=case_id,
            timeout_sec=timeout_sec,
            fused_module=fused_module,
            workspace=workspace,
            git_env=git_env,
        ),
        encoding="utf-8",
    )
    return str(path)
