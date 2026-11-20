# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KernelForge tasks for a Primus training step's kernels, and their measured speedups back into the projection.

``primus_targets analyze`` ranks a training step's kernels from Primus' projection, calibrated by a profile.
This turns one of them into a task KernelForge's ``forge-loop`` can run with no trace: a starting kernel (what
the stack runs today), a driver that checks a candidate against an fp32 reference and times it against that
starting kernel at the step's shapes, and a brief with the kernel's share of the iteration. After the loop, the
driver times the kept kernel against the baseline phase by phase, and :func:`speedups_from_phases` turns that
into a ``primus.kernel_speedups`` file for ``primus_targets project``.

Only the LM head's cross-entropy has a task today: Megatron's native fused path, the largest kernel of the
step a rewrite can change on its own.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

ASSETS = Path(__file__).resolve().parent / "primus_forge_assets"
SPEEDUPS_SCHEMA = "primus.kernel_speedups"
TASK_SCHEMA = "primus.forge_task"
_UTF8 = "utf-8"
_DRIVER = "driver.py"
_MS_DIGITS = 3
#: The cross-entropy's HBM passes over the logits at the speed of light: read forward, read and write backward.
_LOGITS_PASSES = 3
_DEFAULT_GPU = "mi355x"
# the scripts run in the stack's image, whose interpreter is not this one
_PYTHON = "python"

# Per op: its starting kernel, driver and brief, and the scope its report rows carry.
TASKS = {
    "cross_entropy": {
        "kernel": "cross_entropy_kernel.py",
        "driver": "cross_entropy_driver.py",
        "program": "cross_entropy_program.md",
        "scope": "output",
        "backend": "triton",
    },
}
# forge-loop's workspace guard rejects a round that leaves files git doesn't ignore
_GITIGNORE = "__pycache__/\n*.pyc\n*.log\nbuild/\nforge_experiments/\n.stop\n"
_GPUS = {"mi355x": {"arch": "gfx950", "hbm_tbps": 8.0}, "mi300x": {"arch": "gfx942", "hbm_tbps": 5.3}}


def _vocab(report: dict[str, Any]) -> int:
    """The padded vocabulary: the LM head forward GEMM's n."""
    for row in report.get("kernels", []):
        dims = row.get("dims") or {}
        if row.get("op") == "lm_head" and dims.get("layout") == "NT" and dims.get("n"):
            return int(dims["n"])
    raise ValueError("the report has no LM head forward GEMM to read the vocabulary from")


def _tokens(report: dict[str, Any]) -> int:
    dep = report.get("deployment") or {}
    seq, mbs = dep.get("seq_length"), dep.get("micro_batch_size")
    if not seq or not mbs:
        raise ValueError("the report's deployment names no seq_length and micro_batch_size")
    return int(seq) * int(mbs) // max(1, int(dep.get("cp") or 1))


def task_spec(
    report: dict[str, Any],
    op: str,
    *,
    tokens: list[int] | None = None,
    check_tokens: list[int] | None = None,
    config: str = "",
) -> dict[str, Any]:
    """The task's facts from a kernel report: its cases, its share of the iteration, phase by phase.

    The GPU is the one the report was projected for.
    """
    if op not in TASKS:
        raise ValueError(f"no forge task for {op}; known: {', '.join(sorted(TASKS))}")
    vocab = _vocab(report)
    cases = sorted(set(tokens or [_tokens(report)]), reverse=True)
    iteration_ms = float((report.get("iteration") or {}).get("iteration_time_ms") or 0.0)
    phases: dict[str, float] = {}
    for row in report.get("kernels", []):
        if row.get("op") == op:
            phase = str(row.get("phase") or "fwd")
            phases[phase] = phases.get(phase, 0.0) + float(row.get("est_total_ms") or 0.0)
    est_ms = sum(phases.values())
    return {
        "schema": TASK_SCHEMA,
        "version": 1,
        "op": op,
        "scope": TASKS[op]["scope"],
        "config": config,
        "gpu": str((report.get("provenance") or {}).get("gpu_arch") or _DEFAULT_GPU),
        "deployment": report.get("deployment") or {},
        "iteration_ms": iteration_ms,
        "phases_ms": {k: round(v, _MS_DIGITS) for k, v in sorted(phases.items())},
        "est_ms": round(est_ms, _MS_DIGITS),
        "share_pct": round(100.0 * est_ms / iteration_ms, _MS_DIGITS) if iteration_ms else 0.0,
        "cases": [{"tokens": t, "vocab": vocab} for t in cases],
        "check_cases": [{"tokens": t, "vocab": vocab} for t in sorted(set(check_tokens or [4096, cases[0]]))],
    }


def _program(spec: dict[str, Any]) -> str:
    gpu = _GPUS.get(spec["gpu"], _GPUS[_DEFAULT_GPU])
    tokens = spec["cases"][0]["tokens"]
    vocab = spec["cases"][0]["vocab"]
    logits_gb = tokens * vocab * 2 / 1e9
    dep = spec["deployment"]
    deployment = ", ".join(
        f"{k}={dep[k]}" for k in ("tp", "pp", "ep", "cp", "micro_batch_size", "seq_length") if k in dep
    )
    template = (ASSETS / TASKS[spec["op"]]["program"]).read_text(encoding=_UTF8)
    return template.format(
        gpu=spec["gpu"].upper(),
        arch=gpu["arch"],
        hbm_tbps=gpu["hbm_tbps"],
        config=spec["config"] or "the training config",
        deployment=deployment,
        share_pct=spec["share_pct"],
        est_ms=spec["est_ms"],
        iteration_ms=spec["iteration_ms"],
        phase_lines=", ".join(f"{k} {v:.1f} ms" for k, v in spec["phases_ms"].items()),
        vocab=vocab,
        tokens_max=tokens,
        logits_gb=logits_gb,
        sol_ms=_LOGITS_PASSES * logits_gb / gpu["hbm_tbps"],
        cases=", ".join(f"{c['tokens']} tokens" for c in spec["cases"]),
        check_cases=", ".join(f"{c['tokens']} tokens" for c in spec["check_cases"]),
    )


def write_task(spec: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """A KernelForge workspace under ``run_dir/workspace``, one first commit, and the loop's start and resume scripts.

    ``run_dir/baseline.json`` (the anchor) is written on a GPU by ``prepare.sh``, which the start script runs first.
    """
    run_dir = run_dir.resolve()
    ws = run_dir / "workspace"
    if ws.exists() and any(ws.iterdir()):
        raise ValueError(f"{ws} is not empty")
    ws.mkdir(parents=True, exist_ok=True)
    assets = TASKS[spec["op"]]
    (ws / "kernel.py").write_text((ASSETS / assets["kernel"]).read_text(encoding=_UTF8), encoding=_UTF8)
    (ws / _DRIVER).write_text((ASSETS / assets["driver"]).read_text(encoding=_UTF8), encoding=_UTF8)
    (ws / "program.md").write_text(_program(spec), encoding=_UTF8)
    (ws / "task.json").write_text(json.dumps(spec, indent=2) + "\n", encoding=_UTF8)
    (ws / ".gitignore").write_text(_GITIGNORE, encoding=_UTF8)
    git = ["git", "-C", str(ws)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    identity = ["-c", "user.name=primus_forge", "-c", "user.email=primus_forge@local"]
    subprocess.run([*git, *identity, "commit", "-q", "-m", "primus_forge: the task"], check=True)

    gpu = _GPUS.get(spec["gpu"], _GPUS[_DEFAULT_GPU])
    baseline = run_dir / "baseline.json"
    loop = [_PYTHON, "-m", "kernelforge.cli", "forge-loop"]
    options = ["--workspace", str(ws), "--gpu-target", gpu["arch"], "--paired-keep", "--early-stop"]
    # the ceiling the campaign judges its attainment by; forge-loop reuses it on --resume only when asked
    options += ["--baseline-json", str(baseline), "--roofline-ceiling", "on"]
    start = [
        *loop,
        "--kernel",
        str(ws / "kernel.py"),
        "--driver",
        str(ws / _DRIVER),
        "--program-md-file",
        str(ws / "program.md"),
        "--kernel-backend",
        assets["backend"],
        "--git-branch",
        "forge-optimize",
        *options,
    ]
    prepare = [_PYTHON, str(ws / _DRIVER)]
    scripts = {
        "prepare.sh": (
            f"#!/bin/sh\nset -e\n{shlex.join(prepare)}\n"
            f"[ -f {shlex.quote(str(baseline))} ] || {shlex.join([*prepare, '--baseline-json', str(baseline)])}\n"
        ),
        "forge_loop.sh": f'#!/bin/sh\nset -e\n{shlex.quote(str(run_dir / "prepare.sh"))}\nexec {shlex.join(start)} "$@"\n',
        "forge_loop_resume.sh": f'#!/bin/sh\nexec {shlex.join([*loop, "--resume", *options])} "$@"\n',
        "measure.sh": (
            f"#!/bin/sh\n# the kept kernel against the baseline, phase by phase\n"
            f'{shlex.join([*prepare, "--phases-json", str(run_dir / "phases.json")])} "$@"\n'
        ),
    }
    out = {}
    for name, text in scripts.items():
        path = run_dir / name
        path.write_text(text, encoding=_UTF8)
        path.chmod(0o755)
        out[name] = str(path)
    out["workspace"] = str(ws)
    return out


def _case_tokens(case_id: str) -> int | None:
    name, _, value = str(case_id).partition("=")
    return int(value) if name == "T" and value.isdigit() else None


def speedups_from_phases(
    phases: dict[str, Any], *, tokens: int | None = None, source: str = "", against: str = "baseline"
) -> list[dict[str, Any]]:
    """One entry per phase: the baseline's time over the kept kernel's, at ``tokens`` (default: the most tokens).

    ``against`` picks the baseline: ``baseline`` (the task's, Megatron's native path for the cross-entropy) or
    ``te`` (Transformer Engine's), for a run whose recipe and calibration use that one.
    """
    cases = {_case_tokens(k): v for k, v in (phases.get("cases") or {}).items() if _case_tokens(k)}
    if not cases:
        return []
    pick = tokens if tokens in cases else max(cases)
    entries = []
    for phase, rec in sorted(cases[pick].items()):
        base, kern = float(rec.get(f"{against}_ms") or 0.0), float(rec.get("kernel_ms") or 0.0)
        if base > 0 and kern > 0:
            entry = {"op": phases["op"], "phase": phase, "speedup": round(base / kern, 4)}
            if phases.get("scope"):
                entry["scope"] = phases["scope"]
            entry["source"] = source or f"forge:{phases['op']}"
            entries.append(entry)
    return entries


def speedups_from_run_state(run_dir: Path, op: str, scope: str | None = None) -> list[dict[str, Any]]:
    """Without a phases file: forward plus backward from KernelForge's run state, the same speedup for both phases."""
    state = json.loads((run_dir / "workspace" / "forge_experiments" / "run_state.json").read_text(encoding=_UTF8))
    base = state.get("baseline_case_times") or {}
    best = state.get("best_case_times") or {}
    common = [k for k in best if k in base and best[k] and base[k]]
    if not common:
        return []
    case = max(common, key=lambda k: _case_tokens(k) or 0)
    x = round(float(base[case]) / float(best[case]), 4)
    out = []
    for phase in ("fwd", "bwd"):
        entry: dict[str, Any] = {"op": op, "phase": phase, "speedup": x}
        if scope:
            entry["scope"] = scope
        entry["source"] = f"forge:{op}:run_state"
        out.append(entry)
    return out
