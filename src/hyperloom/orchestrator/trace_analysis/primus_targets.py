# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel targets for training from Primus' projection, in place of a profiled run.

TraceLens finds the kernels worth optimizing in a profile of the training job,
so every parallelism layout, batch size or recipe it is asked about has to be
launched on GPUs and profiled first. Primus' simulate-mode projection prices
the same iteration on a CPU in seconds, and with ``--kernel-report`` it writes
what it priced: every GEMM, attention call and analytical term of a training
iteration, with its shape, phase (``fwd``/``dgrad``/``wgrad``/``bwd``), calls
per iteration on each pipeline stage, and time (``primus.kernel_report``).

This module closes the training loop for Hyperloom:

* ``analyze`` projects a training config and writes the run directory a
  TraceLens analysis writes (``kernel_candidates.json``, ``kernel_roofline.json``,
  ``tracelens/summary.json``, ``analysis.md``) plus forge ``shapes.json``
  (dense GEMMs) and ``grouped_shapes.json`` (MoE grouped GEMMs). Shapes carry
  ``shape_provenance: primus_projection``.
* ``calibrate`` builds a ``primus.kernel_calibration`` from torch traces of one
  real run of the stack (:mod:`primus_calibration`); every later projection of
  the stack, at any layout, takes it.
* ``feedback`` turns what the lanes measured into ``primus.kernel_speedups``;
  ``project`` re-projects the iteration with them: the end-to-end gain.
* ``tune`` runs the Primus tuning agent's systematic seed search (no LLM) with
  the calibration, then analyzes its top configurations, merged
  (:mod:`primus_search`).
* ``validate`` compares projections with measured runs: iteration time, and how
  far the projected kernel-class mix is from each trace's.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from hyperloom_common.timeutil import now_iso

from ._targets_common import Run as _Run
from ._targets_common import Subject, speedups_from_gemm_tune
from ._targets_common import write_json as _write_json
from ._targets_common import write_sidecars
from . import primus_forge, primus_search

log = logging.getLogger(__name__)

TOOL = "primus_targets"
NAME_PREFIX = "primus"
SHAPE_PROVENANCE = "primus_projection"
TRACE_INPUT_TYPE = "primus"
REPORT_SCHEMA = "primus.kernel_report"
SPEEDUPS_SCHEMA = "primus.kernel_speedups"

_UTF8 = "utf-8"
_OK = "ok"
_APPEND = "append"
_ROOFLINE_FILE = "kernel_roofline.json"
_RUNS_DIR = "kernel-agent"
_GEMM_KINDS = ("gemm", "grouped_gemm")
_COMM_KINDS = ("comm",)
_ITERATION_RE = re.compile(r"Iteration Time:\s*([0-9.,]+)\s*ms")
_BYTES = {"fp8": 1, "mx8": 1, "fp4": 0.5, "mxfp4": 0.5, "bf16": 2, "fp16": 2, "fp32": 4}
_MNK = "mnk"
_PCT = 100.0
#: Decimal places: milliseconds and shares in reports, totals in task rows.
_MS_DIGITS, _TOTAL_DIGITS = 3, 4
_CONFIG, _KERNEL_REPORT, _OUT = "--config", "--kernel-report", "--out"
_YAML_HELP = "the Primus training YAML to project"
_REPORT_HELP = "an existing kernel report (no projection run)"


# ── running the projection ──────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Primus:
    """Where Primus is and how its projection is run."""

    root: str = os.environ.get("PRIMUS_ROOT", "")
    python: str = os.environ.get("PRIMUS_PYTHON", sys.executable)
    gpu_arch: str = os.environ.get("PRIMUS_GPU_ARCH", "mi355x")
    nodes: int = 1
    gpus_per_node: int = 8
    timeout_s: int = 900

    def env(self, **extra: str) -> dict[str, str]:
        """The projection's environment; ``PRIMUS_PYTHONPATH``, when set, replaces this process' ``PYTHONPATH``."""
        env = dict(os.environ)
        path = env.get("PRIMUS_PYTHONPATH", env.get("PYTHONPATH", ""))
        env["PYTHONPATH"] = os.pathsep.join(p for p in (self.root, path) if p)
        env |= {"NNODES": str(self.nodes), "GPUS_PER_NODE": str(self.gpus_per_node), "PRIMUS_GPU_ARCH": self.gpu_arch}
        return env | {k: v for k, v in extra.items() if v}


def project(
    primus: Primus,
    config: str,
    report_path: Path,
    *,
    calibration: str = "",
    speedups: str = "",
) -> dict[str, Any]:
    """Project ``config`` in simulate mode; its kernel report (``iteration`` holds the projected time)."""
    report_path = report_path.resolve()
    config, calibration, speedups = _abs(config), _abs(calibration), _abs(speedups)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        primus.python, "-m", "primus.cli.main", "projection", "performance",
        _CONFIG, config,
        "--profiling-mode", "simulate",
        "--gpu-arch", primus.gpu_arch,
        "--target-nodes", str(primus.nodes),
        _KERNEL_REPORT, str(report_path),
    ]  # fmt: skip
    if calibration:
        cmd += ["--kernel-calibration", calibration]
    if speedups:
        cmd += ["--kernel-speedups", speedups]
    log_path = report_path.with_suffix(".log")
    with open(log_path, "w", encoding=_UTF8) as log_fh:
        proc = subprocess.run(
            cmd, cwd=primus.root or None, env=primus.env(), stdout=log_fh, stderr=subprocess.STDOUT,
            timeout=primus.timeout_s, check=False,
        )  # fmt: skip
    if proc.returncode != 0 or not report_path.is_file():
        tail = _read_text(log_path)[-2000:]
        raise RuntimeError(f"primus projection failed (rc={proc.returncode}): {shlex.join(cmd)}\n{tail}")
    with open(report_path, encoding=_UTF8) as fh:
        report = json.load(fh)
    report.setdefault("iteration", {})
    if not report["iteration"].get("iteration_time_ms"):
        m = _ITERATION_RE.search(_read_text(log_path))
        if m:
            report["iteration"]["iteration_time_ms"] = float(m.group(1).replace(",", ""))
    report["log"] = str(log_path)
    return report


def _read_text(path: Path) -> str:
    return path.read_text(encoding=_UTF8, errors="replace")


def _abs(path: str) -> str:
    """``path`` made absolute: the projection runs from the Primus checkout."""
    return str(Path(path).resolve()) if path else ""


def load_report(path: str) -> dict[str, Any]:
    with open(path, encoding=_UTF8) as fh:
        report = json.load(fh)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{path} is not a {REPORT_SCHEMA} document")
    return report


# ── candidates from the kernel report ───────────────────────────────────────


def candidate_name(row: dict[str, Any]) -> str:
    """``primus::<scope>::<kind>::<op>.<phase>``: unique per priced term, and parseable back."""
    return "::".join((NAME_PREFIX, str(row["scope"]), str(row["kind"]), f"{row['op']}.{row['phase']}"))


def parse_candidate_name(name: str) -> dict[str, str] | None:
    try:
        prefix, scope, kind, term = str(name or "").split("::")
    except ValueError:
        return None
    if prefix != NAME_PREFIX or "." not in term:
        return None
    op, _, phase = term.rpartition(".")
    return {"scope": scope, "kind": kind, "op": op, "phase": phase}


def _is_gemm(kind: str, dims: dict[str, Any]) -> bool:
    """A GEMM term with all of its ``m``, ``n`` and ``k``."""
    return kind in _GEMM_KINDS and all(isinstance(dims.get(x), int) for x in _MNK)


def _tensors(kind: str, dims: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A term's operands and result as stored, for a GEMM (by layout) or attention; none for analytic terms."""
    dtype = dims.get("dtype") or "bf16"
    if _is_gemm(kind, dims):
        m, n, k, layout = dims["m"], dims["n"], dims["k"], dims.get("layout", "NN")
        a = [k, m] if layout.startswith("T") else [m, k]
        b = [n, k] if layout.endswith("T") else [k, n]
        lead = [dims["batch"]] if dims.get("batch", 1) > 1 else []
        return (
            [{"name": "a", "shape": lead + a, "dtype": dtype}, {"name": "b", "shape": lead + b, "dtype": dtype}],
            [{"name": "c", "shape": lead + [m, n], "dtype": "bf16"}],
        )
    if kind == "attention" and isinstance(dims.get("seq"), int):
        b, s, skv = dims["batch"], dims["seq"], dims.get("seq_kv", dims["seq"])
        h, hkv, d, dv = (
            dims["heads"],
            dims.get("kv_heads", dims["heads"]),
            dims["head_dim"],
            dims.get("head_dim_v", dims["head_dim"]),
        )
        return (
            [
                {"name": "q", "shape": [b, h, s, d], "dtype": dtype},
                {"name": "k", "shape": [b, hkv, skv, d], "dtype": dtype},
                {"name": "v", "shape": [b, hkv, skv, dv], "dtype": dtype},
            ],
            [{"name": "o", "shape": [b, h, s, dv], "dtype": dtype}],
        )
    return [], []


def _flops_bytes(kind: str, dims: dict[str, Any]) -> tuple[float, float]:
    if _is_gemm(kind, dims):
        m, n, k, g = dims["m"], dims["n"], dims["k"], dims.get("batch", 1)
        bpe = _BYTES.get(str(dims.get("dtype")), 2)
        return 2.0 * m * n * k * g, g * ((m * k + k * n) * bpe + m * n * 2)
    return 0.0, 0.0


def _shape_text(t: dict[str, Any]) -> str:
    return f"({','.join(str(d) for d in t['shape'])}) {t['dtype']}"


def _invocation(row: dict[str, Any]) -> dict[str, Any]:
    ins, outs = _tensors(row["kind"], row["dims"])
    return {
        "operation": f"{row['op']}.{row['phase']}",
        "input_shapes": [{"shape": _shape_text(t), "call_num": row["calls"]} for t in ins],
        "input_dtypes": [t["dtype"] for t in ins],
        "output_shapes": [_shape_text(t) for t in outs],
        "output_dtypes": [t["dtype"] for t in outs],
        "raw_arg_spec": {"dims": row["dims"], "tensors": [t["name"] for t in ins]},
        "call_count": row["calls"],
    }


def _terms(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One task per priced term (scope, op, phase, kind), on the stage it costs the most, with its shapes as cases."""
    stage_total: dict[int, float] = defaultdict(float)
    by_term: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in report.get("kernels") or []:
        stage_total[row["stage"]] += row["est_total_ms"]
        by_term[(row["scope"], row["op"], row["phase"], row["kind"])][row["stage"]].append(row)
    tasks = []
    for (scope, op, phase, kind), stages in by_term.items():
        stage, rows = max(stages.items(), key=lambda sr: sum(r["est_total_ms"] for r in sr[1]))
        total = sum(r["est_total_ms"] for r in rows)
        tasks.append(
            {
                "scope": scope, "op": op, "phase": phase, "kind": kind, "stage": stage,
                "est_total_ms": total, "calls": sum(r["calls"] for r in rows),
                "share_pct": round(_PCT * total / stage_total[stage], _MS_DIGITS) if stage_total[stage] else 0.0,
                "cases": sorted(rows, key=lambda r: -r["est_total_ms"]),
            }
        )  # fmt: skip
    return sorted(tasks, key=lambda t: -t["est_total_ms"])


def candidate_from_task(task: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """One hot-kernel candidate row, in TraceLens' schema, for one priced term."""
    top = task["cases"][0]
    inv = _invocation(top)
    flops = nbytes = 0.0
    for c in task["cases"]:
        f, b = _flops_bytes(c["kind"], c["dims"])
        flops, nbytes = flops + f * c["calls"], nbytes + b * c["calls"]
    ins, _ = _tensors(top["kind"], top["dims"])
    return {
        "name": candidate_name(task),
        "operation": inv["operation"],
        "candidate_source": TOOL,
        "shape_provenance": SHAPE_PROVENANCE,
        "duration_us": round(1000.0 * task["est_total_ms"], _MS_DIGITS),
        "call_count": int(task["calls"]),
        "gpu_pct": task["share_pct"],
        "shapes": [_shape_text(t) for t in ins],
        "input_shapes": inv["input_shapes"],
        "input_dtypes": inv["input_dtypes"],
        "output_shapes": inv["output_shapes"],
        "output_dtypes": inv["output_dtypes"],
        "raw_arg_spec": inv["raw_arg_spec"],
        "invocation_cases": [_invocation(c) for c in task["cases"]],
        "bound_type": "unknown",
        "bottleneck": "unknown",
        "flops_per_byte": round(flops / nbytes, _MS_DIGITS) if nbytes else None,
        "efficiency_percent": None,
        "is_multigpu": task["kind"] in _COMM_KINDS,
        "num_gpus_recommended": None,
        "source_file": "",
        "primus_task": {
            "scope": task["scope"],
            "op": task["op"],
            "phase": task["phase"],
            "kind": task["kind"],
            "stage": task["stage"],
            "est_total_ms": round(task["est_total_ms"], _TOTAL_DIGITS),
            "calls": task["calls"],
            "estimate_basis": top.get("src"),
            "cases": [{"dims": c["dims"], "calls": c["calls"], "est_ms": c["est_ms"]} for c in task["cases"]],
        }
        | {"deployment": report.get("deployment"), "gpu_arch": (report.get("provenance") or {}).get("gpu_arch")},
    }


def candidates_from_report(report: dict[str, Any], *, min_gpu_pct: float = 0.0) -> list[dict[str, Any]]:
    """Finalized hot-kernel candidates for a ``primus.kernel_report``, by TraceLens' own rules."""
    from ._candidate_assembly import _finalize_candidates
    from ._kernel_roofline import default_roofline_fields

    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"not a Primus kernel report: schema {report.get('schema')!r}")
    rows = [candidate_from_task(t, report) for t in _terms(report) if t["share_pct"] >= min_gpu_pct]
    rows.sort(key=lambda r: -r["duration_us"])
    _finalize_candidates(rows)
    default_roofline_fields(rows)
    return rows


# ── shapes for the GEMM tuners ──────────────────────────────────────────────


def gemm_shapes(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Forge's dense ``{M, N, K}`` and grouped ``{G, M, N, K}`` shapes, most time first, with layout and ops.

    Weighted by the busiest stage's calls per iteration: one rank's view.
    """
    dense: dict[tuple, dict[str, Any]] = {}
    grouped: dict[tuple, dict[str, Any]] = {}
    for task in _terms(report):
        for row in task["cases"]:
            d = row["dims"]
            if not _is_gemm(row["kind"], d):
                continue
            g = d.get("batch", 1)
            key = (g, d["m"], d["n"], d["k"], d.get("dtype"), d.get("layout"))
            into = grouped if g > 1 else dense
            entry = into.setdefault(
                key,
                {"M": d["m"], "N": d["n"], "K": d["k"], "q_dtype_w": d.get("dtype"), "layout": d.get("layout"),
                 "calls": 0, "est_total_ms": 0.0, "ops": []} | ({"G": g} if g > 1 else {}),
            )  # fmt: skip
            entry["calls"] += row["calls"]
            entry["est_total_ms"] = round(entry["est_total_ms"] + row["est_total_ms"], _TOTAL_DIGITS)
            label = f"{row['op']}.{row['phase']}"
            if label not in entry["ops"]:
                entry["ops"].append(label)
    rank = lambda xs: sorted(xs.values(), key=lambda e: -e["est_total_ms"])  # noqa: E731
    return rank(dense), rank(grouped)


# ── the run directory ───────────────────────────────────────────────────────


def _new_run_dir(subject: Subject) -> tuple[str, str, Path]:
    session_id = subject.session_id or uuid.uuid4().hex[:12]
    run_id = f"pr-{uuid.uuid4().hex[:8]}"
    stamp = now_iso(timespec="seconds").split("+")[0].replace("-", "").replace(":", "") + "Z"
    return session_id, run_id, Path(subject.workspace_path) / _RUNS_DIR / "runs" / session_id / f"{stamp}_{run_id}"


def _markdown(report: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    it, dep = report.get("iteration") or {}, report.get("deployment") or {}
    lines = [
        "# Primus kernel targets",
        "",
        f"Projected iteration: {it.get('iteration_time_ms', 0):.1f} ms, "
        f"{it.get('tokens_per_sec_per_gpu', 0):,.0f} tokens/s/GPU "
        f"(TP{dep.get('tp')} PP{dep.get('pp')} EP{dep.get('ep')} CP{dep.get('cp')} DP{dep.get('dp')}, "
        f"{dep.get('microbatches')} microbatches of {dep.get('micro_batch_size')}).",
        "",
    ]
    merged = (report.get("provenance") or {}).get("merged_configs") or []
    if merged:
        lines += [f"Targets merged over {len(merged)} configurations, each weighted equally: {'; '.join(merged)}.", ""]
    lines += [
        "| term | stage | calls | est ms | share % |",
        "|---|---|---|---|---|",
    ]
    for c in candidates:
        t = c["primus_task"]
        lines.append(f"| `{c['name']}` | {t['stage']} | {t['calls']} | {t['est_total_ms']:.2f} | {c['gpu_pct']} |")
    return "\n".join(lines) + "\n"


def write_run(
    report: dict[str, Any],
    subject: Subject,
    *,
    report_path: str,
    min_gpu_pct: float = 0.0,
    extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Lay out a TraceLens-shaped run directory for ``report`` and return the tool's result dict."""
    session_id, run_id, run_dir = _new_run_dir(subject)
    candidates = candidates_from_report(report, min_gpu_pct=min_gpu_pct)
    md_path = run_dir / "analysis.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(report, candidates), encoding=_UTF8)
    doc = {
        "deployment": report.get("deployment"),
        "provenance": report.get("provenance"),
        "e2e": report.get("iteration"),
    }
    run = _Run(subject, doc, run_dir, report_path, candidates, {"kernel_report": report_path})
    task_groups = write_sidecars(run, md_path, [], tool=TOOL, trace_input_type=TRACE_INPUT_TYPE)
    artifacts = run.artifacts
    dense, grouped = gemm_shapes(report)
    if dense:
        artifacts["shapes_json"] = _write_json(run_dir / "shapes.json", dense)
    if grouped:
        artifacts["grouped_shapes_json"] = _write_json(run_dir / "grouped_shapes.json", grouped)
    artifacts["trace_input_manifest"] = _write_json(
        run_dir / "trace_input_manifest.json",
        {"trace_input": report_path, "trace_input_type": TRACE_INPUT_TYPE, "trace_files": [], "created_at": now_iso()},
    )
    artifacts["trace_report_path"] = str(md_path)
    artifacts.update(extra or {})
    route_ext = {key: doc[key] for key in ("deployment", "provenance", "e2e")}
    route_ext |= {
        "terms": len(candidates),
        "dense_gemm_shapes": len(dense),
        "grouped_gemm_shapes": len(grouped),
        "task_groups": len(task_groups),
    }
    return {
        "tool": TOOL,
        "status": _OK,
        "session_id": session_id,
        "run_id": run_id,
        "trace_input_type": TRACE_INPUT_TYPE,
        "hot_kernels": candidates,
        "trace_report_path": str(md_path),
        "analysis_report_path": str(md_path),
        "candidates_path": artifacts["kernel_candidates"],
        "kernel_roofline_path": artifacts["kernel_roofline"],
        "artifact_paths": artifacts,
        "orchestrator_mode": TOOL,
        "trace_health_warnings": [],
        "run_meta": {
            "preflight": {"trace_input": report_path, "trace_input_type": TRACE_INPUT_TYPE},
            "route_ext": route_ext,
        },
    }


# ── speedups fed back ───────────────────────────────────────────────────────


def speedups_from_summary(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Entries for every adopted kernel the session's summary names as a Primus target, and those it can't place."""
    out, unplaced = [], []
    for row in summary.get("by_kernel") or []:
        speedup = row.get("speedup")
        if not row.get("claimed") or not speedup or float(speedup) <= 1.0:
            continue
        parsed = parse_candidate_name(row.get("kernel_name") or "")
        if parsed is None:
            unplaced.append(str(row.get("kernel_name") or row.get("trace_kernel_id") or ""))
            continue
        out.append(
            {
                "op": parsed["op"],
                "phase": parsed["phase"],
                "scope": parsed["scope"],
                "speedup": float(speedup),
                "source": f"kernel-opt:{row.get('lane') or 'kernel'}",
            }
        )
    return out, unplaced


def speedups_document(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": SPEEDUPS_SCHEMA, "version": 1, "speedups": entries}


def _manual_speedup(raw: str) -> dict[str, Any]:
    """``OP[.PHASE]=X``."""
    target, _, x = raw.partition("=")
    op, _, phase = target.strip().partition(".")
    entry: dict[str, Any] = {"op": op, "speedup": float(x), "source": "manual"}
    if phase:
        entry["phase"] = phase
    return entry


# ── the tuning agent ────────────────────────────────────────────────────────


def _target_cluster(primus: Primus, out: Path, hbm_gb: float) -> str:
    path = out / "target_cluster.yaml"
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "target_cluster:\n"
        f"  name: {primus.gpu_arch}_{primus.nodes}nodes\n"
        f"  num_nodes: {primus.nodes}\n"
        f"  gpus_per_node: {primus.gpus_per_node}\n"
        f"  gpu_arch: {primus.gpu_arch}\n"
        "available_for_benchmark:\n"
        "  has_gpu: false\n"
        "optimization:\n"
        "  objective: tokens_per_s_per_gpu\n"
        f"  hbm_capacity_gb: {hbm_gb}\n",
        encoding=_UTF8,
    )
    return str(path)


@dataclasses.dataclass(frozen=True)
class TuneSpec:
    """How the tuning agent searches: its cluster (default: from the Primus settings), inputs and budget."""

    target_cluster: str = ""
    calibration: str = ""
    speedups: str = ""
    seed_budget: int = 12
    hbm_gb: float = 288.0


def tune(primus: Primus, workload: str, out: Path, spec: TuneSpec = TuneSpec()) -> dict[str, Any]:
    """The tuning agent's seed search with the calibration; its best trial and that trial's config."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed_budget = spec.seed_budget
    cluster = _abs(spec.target_cluster) or _target_cluster(primus, out, spec.hbm_gb)
    workload = _abs(workload)
    cmd = [
        primus.python, "-m", "primus.agents.tuning_agent",
        "--workload", workload, "--target-cluster", cluster, "--out-dir", str(out / "tuning"),
        "--seed-only", "--seed-budget", str(seed_budget), "--profiling-mode", "simulate",
    ]  # fmt: skip
    env = primus.env(PRIMUS_KERNEL_CALIBRATION=_abs(spec.calibration), PRIMUS_KERNEL_SPEEDUPS=_abs(spec.speedups))
    log_path = out / "tuning.log"
    with open(log_path, "w", encoding=_UTF8) as fh:
        proc = subprocess.run(
            cmd,
            cwd=primus.root or None,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=primus.timeout_s * max(1, seed_budget),
            check=False,
        )
    summary_path = out / "tuning" / "summary.json"
    if not summary_path.is_file():
        tail = _read_text(log_path)[-2000:]
        raise RuntimeError(f"tuning agent wrote no summary (rc={proc.returncode})\n{tail}")
    best = json.loads(summary_path.read_text(encoding=_UTF8)).get("best") or {}
    cmd_run = (best.get("result") or {}).get("cmd") or []
    winner = cmd_run[cmd_run.index(_CONFIG) + 1] if _CONFIG in cmd_run else ""
    return {
        "best": best,
        "winner_config": winner,
        "tuning_log": str(log_path),
        "summary": str(summary_path),
        "trials": str(summary_path.with_name("trials.jsonl")),
    }


# ── validation ──────────────────────────────────────────────────────────────


def _traces(spec: str) -> list[str]:
    """Trace files from a comma-separated list of paths, globs or directories."""
    out: list[str] = []
    for part in filter(None, (s.strip() for s in spec.split(","))):
        if os.path.isdir(part):
            out += sorted(glob.glob(os.path.join(part, "**", "*.pt.trace.json*"), recursive=True))
        else:
            out += sorted(glob.glob(part)) or [part]
    return out


def validate(report: dict[str, Any], traces: list[str], *, measured_ms: float | None = None) -> dict[str, Any]:
    """A projection against a measured run: iteration time, and the class mix on each traced stage."""
    from .primus_calibration import mix_distance, stage_of_rank, _rank_of

    projected = (report.get("iteration") or {}).get("iteration_time_ms")
    mixes, seen = [], set()
    for path in traces:
        stage = stage_of_rank(report, _rank_of(path))
        if stage in seen:
            continue
        seen.add(stage)
        mixes.append(mix_distance(report, path, stage=stage) | {"trace": path})
    out: dict[str, Any] = {"projected_iteration_ms": projected, "mix": mixes}
    if measured_ms and projected:
        out |= {
            "measured_iteration_ms": measured_ms,
            "error_pct": round(_PCT * (projected - measured_ms) / measured_ms, 2),
        }
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def _primus(args: argparse.Namespace) -> Primus:
    return Primus(
        root=args.primus_root or Primus.root,
        python=args.primus_python or Primus.python,
        gpu_arch=args.gpu_arch,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
    )


def _add_primus_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--primus-root", default="", help="Primus checkout (default: PRIMUS_ROOT)")
    p.add_argument("--primus-python", default="", help="Python with Primus' projection deps (default: PRIMUS_PYTHON)")
    p.add_argument("--gpu-arch", default=Primus.gpu_arch)
    p.add_argument("--nodes", type=int, default=1, help="target nodes")
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--kernel-calibration", default="", help="a primus.kernel_calibration JSON")
    p.add_argument("--kernel-speedups", default="", help="a primus.kernel_speedups JSON")


def _add_analyze(sub: Any) -> None:
    a = sub.add_parser("analyze", help="kernel targets in TraceLens' run-directory contract")
    a.add_argument("--workspace-path", required=True)
    a.add_argument("--session-id", default="")
    a.add_argument("--model-name", default="")
    a.add_argument("--framework", default="megatron")
    a.add_argument("--target-platform", default="")
    a.add_argument("--roofline-output-name", default=_ROOFLINE_FILE)
    a.add_argument(_CONFIG, default="", help=_YAML_HELP)
    a.add_argument(_KERNEL_REPORT, default="", help=_REPORT_HELP)
    a.add_argument("--min-gpu-pct", type=float, default=0.0)
    _add_primus_args(a)


def _add_calibrate(sub: Any) -> None:
    c = sub.add_parser("calibrate", help="a kernel calibration from torch traces of one run of the stack")
    c.add_argument("--traces", required=True, help="trace files, globs or directories (comma-separated)")
    c.add_argument(_CONFIG, default="", help="the profiled run's training YAML (projected uncalibrated)")
    c.add_argument(_KERNEL_REPORT, default="", help="an uncalibrated kernel report of the profiled run")
    c.add_argument(_OUT, required=True)
    c.add_argument(
        "--steady-ms", type=float, default=None, help="the run's iteration time unprofiled (scales out profiler cost)"
    )
    _add_primus_args(c)


def _add_feedback(sub: Any) -> None:
    f = sub.add_parser("feedback", help="measured results as a Primus kernel speedups file")
    f.add_argument("--gemm-tune", action=_APPEND, default=[], help="a gemm-tune result.json")
    f.add_argument("--summary", action=_APPEND, default=[], help="a kernel_optimization_summary.json")
    f.add_argument("--speedup", action=_APPEND, default=[], metavar="OP[.PHASE]=X", help="a speedup by hand")
    f.add_argument("--forge-phases", action=_APPEND, default=[], help="a forge task's phases.json (measure.sh)")
    f.add_argument("--forge-run", action=_APPEND, default=[], help="a forge task's run directory (no phases.json)")
    f.add_argument("--tokens", type=int, default=None, help="the projected step's tokens per micro-batch")
    f.add_argument(
        "--forge-against",
        choices=("baseline", "te"),
        default="baseline",
        help="--forge-phases baseline: the task's (Megatron native) or Transformer Engine's, as the run's recipe uses",
    )
    f.add_argument(_OUT, required=True)


def _add_forge_task(sub: Any) -> None:
    ft = sub.add_parser("forge-task", help="a KernelForge task for one of the step's kernels, with no trace")
    ft.add_argument("--op", default="cross_entropy")
    ft.add_argument(_CONFIG, default="", help=_YAML_HELP)
    ft.add_argument(_KERNEL_REPORT, default="", help=_REPORT_HELP)
    ft.add_argument("--tokens", type=int, action=_APPEND, default=[], help="a scored case's tokens (repeatable)")
    ft.add_argument("--check-tokens", type=int, action=_APPEND, default=[], help="a checked case's tokens")
    ft.add_argument("--run-dir", required=True)
    _add_primus_args(ft)


def _add_project_tune_validate(sub: Any) -> None:
    pr = sub.add_parser("project", help="the iteration-time gain of a speedups file")
    pr.add_argument(_CONFIG, required=True)
    pr.add_argument("--out-dir", required=True)
    _add_primus_args(pr)
    t = sub.add_parser("tune", help="the tuning agent's seed search, calibrated; then analyze its best config")
    t.add_argument("--workload", required=True, help="the Primus training YAML to tune")
    t.add_argument("--target-cluster", default="", help="tuning-agent target cluster YAML (default: from --nodes)")
    t.add_argument("--seed-budget", type=int, default=12)
    t.add_argument("--hbm-gb", type=float, default=288.0)
    t.add_argument(
        "--top-k",
        type=int,
        default=primus_search.top_k_setting(),
        help=f"kernel targets from this many of the best configurations (default 5, or {primus_search.TOP_K_ENV})",
    )
    t.add_argument("--workspace-path", required=True)
    t.add_argument("--session-id", default="")
    _add_primus_args(t)
    v = sub.add_parser("validate", help="a projection against a measured run")
    v.add_argument(_CONFIG, default="")
    v.add_argument(_KERNEL_REPORT, default="")
    v.add_argument("--traces", required=True)
    v.add_argument("--measured-ms", type=float, default=None, help="the run's measured iteration time")
    v.add_argument(_OUT, default="")
    _add_primus_args(v)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {__name__}", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (_add_analyze, _add_calibrate, _add_feedback, _add_forge_task, _add_project_tune_validate):
        add(sub)
    return parser


def _report(args: argparse.Namespace, out: Path, *, calibration: str | None = None) -> tuple[dict[str, Any], str]:
    if args.kernel_report:
        return load_report(args.kernel_report), args.kernel_report
    if not args.config:
        raise SystemExit(f"{args.command} needs --config or --kernel-report")
    path = out / "kernel_report.json"
    if calibration is None:
        cal, fast = args.kernel_calibration, getattr(args, "kernel_speedups", "")
    else:
        cal, fast = calibration, ""
    return project(_primus(args), args.config, path, calibration=cal, speedups=fast), str(path)


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    subject = Subject(
        workspace_path=args.workspace_path,
        session_id=args.session_id,
        model_name=args.model_name,
        framework=args.framework,
        target_platform=args.target_platform,
        roofline_output_name=args.roofline_output_name,
    )
    out = Path(args.workspace_path) / _RUNS_DIR / TRACE_INPUT_TYPE / uuid.uuid4().hex[:8]
    report, path = _report(args, out)
    return write_run(report, subject, report_path=path, min_gpu_pct=args.min_gpu_pct)


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    from .primus_calibration import build_calibration

    out = Path(args.out)
    report, path = _report(args, out.parent / f"{out.stem}_projection", calibration="")
    traces = _traces(args.traces)
    if not traces:
        raise ValueError(f"no traces in {args.traces}")
    doc = build_calibration(report, traces, source=",".join(traces), steady_ms=getattr(args, "steady_ms", None))
    doc["uncalibrated_report"] = path
    _write_json(out, doc)
    return {
        "tool": TOOL,
        "status": _OK,
        "kernel_calibration": str(out),
        "factors": len(doc["factors"]),
        "extras": len(doc["extras"]),
        "measured": doc["measured"],
    }


def _feedback(args: argparse.Namespace) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    unplaced: list[str] = []
    for path in args.gemm_tune:
        with open(path, encoding=_UTF8) as fh:
            entries += speedups_from_gemm_tune(json.load(fh))
    for path in args.summary:
        with open(path, encoding=_UTF8) as fh:
            got, missed = speedups_from_summary(json.load(fh))
        entries += got
        unplaced += missed
    entries += [_manual_speedup(raw) for raw in args.speedup]
    for path in args.forge_phases:
        with open(path, encoding=_UTF8) as fh:
            entries += primus_forge.speedups_from_phases(json.load(fh), tokens=args.tokens, against=args.forge_against)
    for run in args.forge_run:
        spec = json.loads((Path(run) / "workspace" / "task.json").read_text(encoding=_UTF8))
        entries += primus_forge.speedups_from_run_state(Path(run), spec["op"], spec.get("scope"))
    _write_json(Path(args.out), speedups_document(entries))
    return {"tool": TOOL, "status": _OK, "kernel_speedups": args.out, "entries": len(entries), "unplaced": unplaced}


def _forge_task(args: argparse.Namespace) -> dict[str, Any]:
    run = Path(args.run_dir)
    report, path = _report(args, run / "projection")
    spec = primus_forge.task_spec(
        report,
        args.op,
        tokens=args.tokens or None,
        check_tokens=args.check_tokens or None,
        config=args.config or path,
    )
    files = primus_forge.write_task(spec, run)
    return {"tool": TOOL, "status": _OK, "share_pct": spec["share_pct"], "cases": spec["cases"], **files}


def _project(args: argparse.Namespace) -> dict[str, Any]:
    if not args.kernel_speedups:
        raise SystemExit("project needs --kernel-speedups")
    out, primus = Path(args.out_dir), _primus(args)
    base = project(primus, args.config, out / "baseline.json", calibration=args.kernel_calibration)
    fast = project(
        primus,
        args.config,
        out / "with_speedups.json",
        calibration=args.kernel_calibration,
        speedups=args.kernel_speedups,
    )
    t0, t1 = base["iteration"].get("iteration_time_ms"), fast["iteration"].get("iteration_time_ms")
    gain = round(_PCT * (t0 - t1) / t0, _MS_DIGITS) if t0 and t1 else None
    return {
        "tool": TOOL,
        "status": _OK,
        "baseline_iteration_ms": t0,
        "projected_iteration_ms": t1,
        "gain_pct": gain,
        "baseline_report": str(out / "baseline.json"),
        "report": str(out / "with_speedups.json"),
    }


def _tune(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.workspace_path) / _RUNS_DIR / "primus_tune" / uuid.uuid4().hex[:8]
    primus = _primus(args)
    found = tune(
        primus,
        args.workload,
        out,
        TuneSpec(
            target_cluster=args.target_cluster,
            calibration=args.kernel_calibration,
            speedups=args.kernel_speedups,
            seed_budget=args.seed_budget,
            hbm_gb=args.hbm_gb,
        ),
    )
    if not found["winner_config"]:
        return {"tool": TOOL, "status": "failed", "error": "the tuning agent found no legal configuration", **found}
    trials_path = Path(found["trials"])
    ranked = primus_search.ranked_trials(trials_path) if trials_path.is_file() else []
    picked = primus_search.chosen(ranked, found["winner_config"], args.top_k)

    def one(trial: primus_search.Trial, i: int) -> dict[str, Any]:
        name = "winner_report.json" if i == 0 else f"top{i + 1}_report.json"
        return project(
            primus, trial.config, out / name, calibration=args.kernel_calibration, speedups=args.kernel_speedups
        )

    projected = primus_search.project_top(picked, one)
    report_path = out / "winner_report.json"
    report = projected.reports[0]
    if len(projected.reports) > 1:
        report = primus_search.merge_reports(projected.reports, projected.labels)
        report_path = Path(_write_json(out / "targets_report.json", report))
    search_path = _write_json(out / "config_search.json", {"top_k": args.top_k, "configs": projected.configs})
    subject = Subject(workspace_path=args.workspace_path, session_id=args.session_id, framework="megatron")
    extra = {"winner_config": found["winner_config"], "tuning_summary": found["summary"], "config_search": search_path}
    result = write_run(report, subject, report_path=str(report_path), extra=extra)
    result["tuning"] = {"best": found["best"], "winner_config": found["winner_config"], "configs": projected.configs}
    return result


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out).parent if args.out else Path.cwd()
    report, path = _report(args, out / "validation_projection")
    result = {"tool": TOOL, "status": _OK, "kernel_report": path} | validate(
        report, _traces(args.traces), measured_ms=args.measured_ms
    )
    if args.out:
        _write_json(Path(args.out), result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands = {
        "analyze": _analyze,
        "calibrate": _calibrate,
        "feedback": _feedback,
        "project": _project,
        "forge-task": _forge_task,
        "tune": _tune,
        "validate": _validate,
    }
    try:
        result = commands[args.command](args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        log.warning("primus targets failed: %s", exc, exc_info=True)
        result = {"tool": TOOL, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == _OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
