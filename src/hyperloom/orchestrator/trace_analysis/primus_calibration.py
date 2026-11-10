# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Calibrate Primus' simulate-mode projection by a torch profile of the training run.

Primus prices a training iteration from its kernels (``primus.kernel_report``):
every GEMM, attention call and analytical term of one layer of each type, with
its calls per iteration on each pipeline stage. A profile of the same stack
measures what those kernels really take, and the engine's own kernels the
model has no term for (FP8 casts, fused losses, elementwise ops, parameter
all-gathers).

:func:`classify_trace` attributes every kernel of a torch trace to the Primus op
that would price it. A forward kernel's op follows from the ``nn.Module`` path
it was launched under; a backward kernel's from the forward op its autograd node
was recorded for (the profiler's sequence numbers); optimizer, loss and
gradient-hook kernels from the source files on their Python stack.
:func:`build_calibration` pools the measured time per op class against the
report's modelled time into a ``primus.kernel_calibration``: a factor per op
class and direction, and the unmodelled work as extras, each a share of its
scope's modelled time so it scales with the deployment.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CALIBRATION_SCHEMA = "primus.kernel_calibration"
REPORT_SCHEMA = "primus.kernel_report"

ITERATION = "iteration"
OUTPUT = "output"
EMBEDDING = "embedding"
FWD, BWD, STEP = "fwd", "bwd", "step"
#: Pipeline send/recv: mostly waiting on the other stage, which the projection's
#: pipeline schedule prices itself, so it is neither calibrated nor in the mix.
PP_P2P, WAIT = "pp_p2p", "wait"
#: Iteration collectives over every rank (grad norm, loss reduce): a stage that finishes
#: early waits in them for the others, which is pipeline imbalance the schedule already
#: prices, so they cost what the last stage to arrive (the cheapest) spends in them.
_GLOBAL_COLLECTIVES = ((ITERATION, "comm_other", STEP),)

#: Report ops priced as one kernel class (Megatron fuses Q/K/V, and gate/up).
_OP_CLASS = (
    (re.compile(r"^attn_[qkv]$"), "attn_qkv"),
    (re.compile(r"^moe_expert_(gate|up|down)$"), "moe_expert"),
    (re.compile(r"^moe_shared_"), "moe_shared"),
    (re.compile(r"^mlp_"), "mlp"),
    # A permute's backward is an unpermute and vice versa: one class.
    (re.compile(r"^moe_(un)?permute$"), "moe_permute"),
)
_GEMM_NAMES = ("cijk_", "gemm", "kernel_dense", "kernel_grouped", "matmul", "wvsplitk", "hipblaslt")
_FP8_NAMES = ("quantize", "amax", "transpose_2d", "cast_transpose", "fp8", "scaling")
_AUTOGRAD = "autograd::engine::evaluate_function"
_MODULE = "nn.Module: "
#: Mixes are compared on classes carrying at least this share of either side.
_MIX_FLOOR = 0.002


def op_class(op: str) -> str:
    """The kernel class a report op is priced as."""
    for pattern, name in _OP_CLASS:
        if pattern.search(op):
            return name
    return op


def direction(phase: str) -> str:
    return phase if phase in (FWD, STEP) else BWD


# ── reading a trace ─────────────────────────────────────────────────────────


@dataclass
class Launch:
    """One GPU kernel with the host context it was launched in."""

    name: str
    dur_us: float
    tid: Any
    modules: list[str] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    cpu_ops: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    autograd_seq: int | None = None


def _load_events(path: str | Path) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc["traceEvents"] if isinstance(doc, dict) else doc


def _host_context(events: list[dict[str, Any]]) -> tuple[dict[Any, list[dict]], dict[int, list[str]]]:
    """Per launch correlation: the enclosing host frames; per sequence number: the forward op's modules."""
    frames_by_tid: dict[Any, list[dict]] = defaultdict(list)
    launches_by_tid: dict[Any, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat in ("python_function", "cpu_op", "user_annotation"):
            frames_by_tid[e["tid"]].append(e)
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in (e.get("args") or {}):
            launches_by_tid[e["tid"]].append(e)
    stacks: dict[int, list[dict]] = {}
    seq_modules: dict[int, list[str]] = {}
    for tid, launches in launches_by_tid.items():
        timeline = sorted(frames_by_tid.get(tid, []), key=lambda e: (e["ts"], -e.get("dur", 0)))
        launches.sort(key=lambda e: e["ts"])
        open_frames: list[dict] = []
        i = 0
        for launch in launches:
            while i < len(timeline) and timeline[i]["ts"] <= launch["ts"]:
                frame = timeline[i]
                while open_frames and open_frames[-1]["ts"] + open_frames[-1].get("dur", 0) < frame["ts"]:
                    open_frames.pop()
                open_frames.append(frame)
                seq = (frame.get("args") or {}).get("Sequence number")
                if seq is not None and frame.get("cat") == "cpu_op" and not frame["name"].startswith(_AUTOGRAD):
                    seq_modules.setdefault(
                        int(seq), [f["name"][len(_MODULE) :] for f in open_frames if f["name"].startswith(_MODULE)]
                    )
                i += 1
            end = launch["ts"] + launch.get("dur", 0)
            stacks[launch["args"]["correlation"]] = [f for f in open_frames if f["ts"] + f.get("dur", 0) >= end]
    return stacks, seq_modules


def read_launches(path: str | Path) -> tuple[list[Launch], dict[str, Any]]:
    """Every kernel of a torch trace with its host context, and the trace's step span."""
    events = _load_events(path)
    stacks, seq_modules = _host_context(events)
    launches = []
    kernel_us = 0.0
    for e in events:
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        kernel_us += e.get("dur", 0)
        stack = stacks.get((e.get("args") or {}).get("correlation"), [])
        launch = Launch(name=e["name"], dur_us=float(e.get("dur", 0)), tid=None)
        seq = None
        for f in stack:
            name = f["name"]
            launch.tid = f["tid"]
            if f.get("cat") == "python_function":
                if name.startswith(_MODULE):
                    launch.modules.append(name[len(_MODULE) :])
                else:
                    launch.frames.append(name)
            elif f.get("cat") == "cpu_op":
                launch.cpu_ops.append(name)
                if name.startswith(_AUTOGRAD):
                    seq = (f.get("args") or {}).get("Sequence number")
            else:
                launch.annotations.append(name)
        if seq is not None:
            launch.autograd_seq = int(seq)
            if not launch.modules:
                launch.modules = list(seq_modules.get(int(seq), []))
        launches.append(launch)
    steps = [
        e for e in events if str(e.get("name", "")).startswith("ProfilerStep") and e.get("cat") == "gpu_user_annotation"
    ]
    span_ms = max((e.get("dur", 0) for e in steps), default=0.0) / 1e3
    return launches, {"kernel_ms": kernel_us / 1e3, "step_span_ms": span_ms, "steps": len({e["name"] for e in steps})}


# ── attributing kernels to Primus ops ───────────────────────────────────────


@dataclass(frozen=True)
class Layers:
    """How a trace's layer modules map to the report's layer scopes on one stage."""

    scopes: tuple[str, ...]
    default: str

    def scope(self, modules: list[str]) -> str | None:
        for m in reversed(modules):
            if m.startswith(("TransformerLayer_", "MoELayer_")):
                idx = m.rsplit("_", 1)[-1]
                if m.startswith("MoELayer_") and "moe" in self.scopes:
                    return "moe"
                if idx.isdigit() and int(idx) < len(self.scopes):
                    return self.scopes[int(idx)]
                return self.default
        return None


def _has(texts: Iterable[str], *needles: str) -> bool:
    return any(n in t for t in texts for n in needles)


def _direction(launch: Launch, main_tid: Any) -> str:
    if launch.autograd_seq is not None or _has(launch.cpu_ops, _AUTOGRAD):
        return BWD
    if _has(launch.frames, "distributed_data_parallel.py") and _has(launch.frames, "hook"):
        return BWD
    return FWD if launch.tid == main_tid else BWD


def _gemm_op(launch: Launch, scope: str | None) -> tuple[str, str]:
    """``(scope, op class)`` of a GEMM kernel, from the module that ran it."""
    path = " > ".join(launch.modules)
    last = launch.modules[-1] if launch.modules else ""
    if "grouped" in launch.name.lower() or _has(launch.cpu_ops, "grouped_gemm"):
        return scope or "moe", "moe_expert"
    if "SharedExpert" in path:
        return scope or "moe", "moe_shared"
    if "Router" in path:
        return scope or "moe", "moe_router"
    if "Attention" in path:
        return scope or "", "attn_o" if "RowParallel" in last or "linear_proj" in last else "attn_qkv"
    if "MLP" in path:
        return scope or "dense", "mlp"
    if "GPTModel" in path and "TransformerLayer" not in path and "MoELayer" not in path:
        return OUTPUT, "lm_head"
    return scope or "", "gemm_other"


def attribute(launch: Launch, layers: Layers, main_tid: Any) -> tuple[str, str, str]:
    """``(scope, op class, direction)`` of one kernel: the Primus term that prices it, or the engine work it is."""
    n = launch.name.lower()
    ops = [o.lower() for o in launch.cpu_ops]
    frames = launch.frames
    scope = layers.scope(launch.modules)
    if _has(launch.annotations, "Optimizer.step") or _has(frames, "/optimizer/", "optimizer.py", "clip_grads.py"):
        return ITERATION, "optimizer", STEP
    if "nccl" in n and _has(frames, "p2p_communication"):
        return ITERATION, PP_P2P, WAIT
    if "nccl" in n:
        anns = " ".join(launch.annotations)
        if "reduce_scatter" in anns:
            return ITERATION, "dp_grad_reduce", STEP
        if "all_gather" in anns:
            return ITERATION, "dp_param_gather", STEP
        return ITERATION, "comm_other", STEP
    d = _direction(launch, main_tid)
    if "deep_ep" in n or _has(frames, "moe_dispatch_combine"):
        return scope or "moe", "moe_a2a_exposed", d
    if _has(frames, "fused_cross_entropy", "cross_entropy") or _has(ops, "cross_entropy"):
        return OUTPUT, "cross_entropy", d
    if _has(launch.modules, "LanguageModelEmbedding", "VocabParallelEmbedding"):
        return EMBEDDING, "embedding", d
    if "flash_attn" in n or "fmha" in n or _has(ops, "flashattn", "flash_attn", "attention"):
        return scope or layers.default, "attn_sdpa", d
    if any(g in n for g in _GEMM_NAMES):
        s, op = _gemm_op(launch, scope)
        return s or layers.default, op, d
    if "permute" in n or "sort_chunks" in n or _has(ops, "permute"):
        return scope or "moe", "moe_permute", d
    if "topk" in n or (_has(launch.modules, "Router") and "gemm" not in n):
        return scope or "moe", "moe_topk", d
    if any(f in n for f in _FP8_NAMES) or _has(ops, "quantize"):
        if _has(launch.modules, "GroupedMLP", "Experts") or _has(frames, "grouped_gemm"):
            return scope or "moe", "moe_expert_overhead", d
        return scope or layers.default, "fp8_cast", d
    if "norm" in n or _has(ops, "norm") or _has(frames, "rmsnorm", "layernorm"):
        return scope or layers.default, "norm_residual", d
    if _has(frames, "swiglu", "act_with_probs", "activation") or "silu" in n or "glu" in n:
        s = scope or layers.default
        return s, "moe_activation" if s.startswith("moe") else "activation", d
    if _has(frames, "distributed_data_parallel.py"):
        return layers.default, "grad_accumulate", BWD
    if "GPTModel" in " ".join(launch.modules) and scope is None and launch.modules:
        return OUTPUT, "elementwise", d
    return scope or layers.default, "elementwise", d


def _main_tid(launches: list[Launch]) -> Any:
    """The thread forward kernels are launched from: the one launching under the most modules."""
    counts: dict[Any, int] = defaultdict(int)
    for launch in launches:
        if launch.modules and launch.autograd_seq is None:
            counts[launch.tid] += 1
    return max(counts, key=counts.get) if counts else None


@dataclass
class Measured:
    """One trace's kernel time per ``(scope, op class, direction)``, in ms, and its step span."""

    ms: dict[tuple[str, str, str], float]
    calls: dict[tuple[str, str, str], int]
    kernel_ms: float
    step_span_ms: float


def classify_trace(path: str | Path, layers: Layers) -> Measured:
    launches, meta = read_launches(path)
    main = _main_tid(launches)
    ms: dict[tuple[str, str, str], float] = defaultdict(float)
    calls: dict[tuple[str, str, str], int] = defaultdict(int)
    for launch in launches:
        key = attribute(launch, layers, main)
        ms[key] += launch.dur_us / 1e3
        calls[key] += 1
    return Measured(dict(ms), dict(calls), meta["kernel_ms"], meta["step_span_ms"])


# ── the report side ─────────────────────────────────────────────────────────


def load_report(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{path} is not a {REPORT_SCHEMA} document")
    return report


def layers_for_stage(report: dict[str, Any], stage: int) -> Layers:
    dep = report.get("deployment") or {}
    scopes_all = list(dep.get("layer_scopes") or [])
    stage_layers = (dep.get("stage_layers") or [[]])[stage]
    scopes = tuple(scopes_all[i] for i in stage_layers if i < len(scopes_all)) or ("moe",)
    default = max(set(scopes), key=scopes.count)
    return Layers(scopes, default)


def stage_of_rank(report: dict[str, Any], rank: int) -> int:
    """The pipeline stage a global rank is on (Megatron puts PP outermost)."""
    dep = report.get("deployment") or {}
    pp = int(dep.get("pp") or 1)
    gpus = int(dep.get("gpus") or pp)
    return min(pp - 1, rank // max(1, gpus // pp))


def modelled(report: dict[str, Any], stage: int, *, base: bool = True) -> dict[tuple[str, str, str], float]:
    """The report's time per ``(scope, op class, direction)`` on ``stage``: modelled (``base``) or as adjusted."""
    out: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in report.get("kernels") or []:
        if row.get("stage") != stage:
            continue
        if base and str(row.get("src", "")).startswith("calibration"):
            continue
        per_call = row["base_ms"] if base else row["est_ms"]
        out[(row["scope"], op_class(row["op"]), direction(row["phase"]))] += per_call * row["calls"]
    return dict(out)


def _rank_of(path: str) -> int:
    m = re.search(r"rank\[?(\d+)\]?", Path(path).name)
    return int(m.group(1)) if m else 0


# ── the calibration ─────────────────────────────────────────────────────────


def _pool(report: dict[str, Any], traces: list[str]) -> tuple[dict, dict, dict[int, Measured]]:
    """Measured and modelled time per key, summed over stages (each stage averaged over its ranks)."""
    by_stage: dict[int, list[Measured]] = defaultdict(list)
    for path in traces:
        stage = stage_of_rank(report, _rank_of(path))
        by_stage[stage].append(classify_trace(path, layers_for_stage(report, stage)))
    measured: dict[tuple, float] = defaultdict(float)
    model: dict[tuple, float] = defaultdict(float)
    per_stage: dict[int, Measured] = {}
    for stage, runs in by_stage.items():
        keys = {k for r in runs for k in r.ms}
        avg = {k: sum(r.ms.get(k, 0.0) for r in runs) / len(runs) for k in keys}
        per_stage[stage] = Measured(
            avg,
            {k: round(sum(r.calls.get(k, 0) for r in runs) / len(runs)) for k in keys},
            sum(r.kernel_ms for r in runs) / len(runs),
            max(r.step_span_ms for r in runs),
        )
        for k, v in avg.items():
            measured[k] += v
        for k, v in modelled(report, stage).items():
            model[k] += v
    if len(per_stage) > 1:
        for k in _GLOBAL_COLLECTIVES:
            if k in measured:
                measured[k] = len(per_stage) * min(m.ms.get(k, 0.0) for m in per_stage.values())
    return dict(measured), dict(model), per_stage


def build_calibration(
    report: dict[str, Any], traces: list[str], *, source: str = "", steady_ms: float | None = None
) -> dict[str, Any]:
    """A ``primus.kernel_calibration`` from torch traces of the run ``report`` (uncalibrated) projects.

    Each op class the model prices gets ``measured / modelled`` per direction,
    applied to every phase of it the report has. Work the model has no term for
    becomes an extra in its scope and direction, sized as a share of that
    scope's modelled time. With ``steady_ms``, the run's iteration time without the
    profiler, kernel times are scaled by how much the profiler stretched the step.
    """
    measured, model, per_stage = _pool(report, traces)
    span = max((m.step_span_ms for m in per_stage.values()), default=0.0)
    scale = steady_ms / span if steady_ms and span > 0 else 1.0
    measured = {k: v * scale for k, v in measured.items() if k[2] != WAIT}
    phases: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in report.get("kernels") or []:
        phases[(op_class(row["op"]), direction(row["phase"]))].add((row["op"], row["phase"]))

    by_class: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (scope, cls, d), v in measured.items():
        by_class[(cls, d)][0] += v
    for (scope, cls, d), v in model.items():
        by_class[(cls, d)][1] += v

    factors, unmatched = [], []
    for (cls, d), (got, want) in sorted(by_class.items()):
        if want <= 0 or got <= 0:
            if want > 0:
                unmatched.append({"op_class": cls, "direction": d, "modelled_ms": round(want, 3)})
            continue
        for op, phase in sorted(phases.get((cls, d), ())):
            factors.append(
                {
                    "op": op,
                    "phase": phase,
                    "factor": round(got / want, 4),
                    "op_class": cls,
                    "measured_ms": round(got, 3),
                    "modelled_ms": round(want, 3),
                }
            )

    scope_total: dict[tuple[str, str], float] = defaultdict(float)
    for (scope, _cls, d), v in model.items():
        scope_total[(scope, d)] += v
    default_scope = max(
        (k for k in scope_total if k[0] not in (ITERATION, OUTPUT, EMBEDDING)),
        key=scope_total.get,
        default=("moe", BWD),
    )[0]
    extras = []
    for (scope, cls, d), got in sorted(measured.items()):
        if by_class.get((cls, d), [0, 0])[1] > 0 or got <= 0:
            continue
        home = (scope, d) if scope_total.get((scope, d), 0.0) > 0 else (default_scope, d if d != STEP else BWD)
        extras.append(
            {
                "scope": home[0],
                "phase": home[1],
                "op": cls,
                "share": round(got / scope_total[home], 5) if scope_total.get(home) else 0.0,
                "measured_ms": round(got, 3),
            }
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "version": 1,
        "source": source,
        "deployment": report.get("deployment"),
        "factors": factors,
        "extras": [e for e in extras if e["share"] > 0],
        "profiler_scale": round(scale, 4),
        "unmatched_modelled": unmatched,
        "measured": {
            str(stage): {
                "kernel_ms": round(m.kernel_ms, 3),
                "step_span_ms": round(m.step_span_ms, 3),
                "idle_ms": round(max(0.0, m.step_span_ms - m.kernel_ms), 3),
                "pp_p2p_ms": round(sum(v for k, v in m.ms.items() if k[2] == WAIT), 3),
            }
            for stage, m in sorted(per_stage.items())
        },
    }


# ── comparing a report with a trace ─────────────────────────────────────────


def _classes(ms: dict[tuple[str, str, str], float]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for (_scope, cls, d), v in ms.items():
        if d == WAIT:
            continue
        out[f"{cls}.{d}"] += v
    return dict(out)


def mix_distance(report: dict[str, Any], trace: str, *, stage: int | None = None) -> dict[str, Any]:
    """How far the report's kernel-class mix on a rank's stage is from the trace's (total variation, 0-1)."""
    stage = stage_of_rank(report, _rank_of(trace)) if stage is None else stage
    traced = classify_trace(trace, layers_for_stage(report, stage)).ms
    got = _classes(traced)
    want = _classes(modelled(report, stage, base=False))
    tg, tw = sum(got.values()) or 1.0, sum(want.values()) or 1.0
    keys = {k for k in set(got) | set(want) if got.get(k, 0) / tg >= _MIX_FLOOR or want.get(k, 0) / tw >= _MIX_FLOOR}
    rows = sorted(
        (
            {
                "class": k,
                "measured_pct": round(100 * got.get(k, 0) / tg, 2),
                "projected_pct": round(100 * want.get(k, 0) / tw, 2),
            }
            for k in keys
        ),
        key=lambda r: -max(r["measured_pct"], r["projected_pct"]),
    )
    tv = 0.5 * sum(abs(got.get(k, 0) / tg - want.get(k, 0) / tw) for k in set(got) | set(want))
    wait = sum(v for k, v in traced.items() if k[2] == WAIT)
    return {
        "stage": stage,
        "total_variation": round(tv, 4),
        "measured_ms": round(tg, 3),
        "projected_ms": round(tw, 3),
        "pp_p2p_ms": round(wait, 3),
        "classes": rows,
    }
