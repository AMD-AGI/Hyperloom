# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide which EXPLORE variants are worth a real benchmark, from InferaSim.

Every variant in an EXPLORE round boots a full serving engine and runs a full
client benchmark. For the variants that only move what the projection models --
parallelism, concurrency and the running-batch cap, sequence lengths, dtypes --
the projection can say in seconds, without a GPU, which of them are decisively
behind the current stack. Those are not benchmarked.

Everything else still is: a variant whose levers the projection cannot see
(scheduler flags, kernel switches, env levers, source and kernel patches), a
variant mixing such a lever with a visible one, and any variant the projection
cannot answer for. The router only removes variants from the round; it never
adds one, never reorders the survivors, and never produces a number that
reaches KEEP/REVERT. A KEEP still rests on the variant's own measured decision
round, and the dropped variants are recorded as ``SKIPPED_DEDUP`` rows, which
are neither journaled nor written to the knowledge base.

The comparison is always projection against projection, with the stack and the
variant materialized exactly as the round would launch them and priced for the
session's GPU, so the projection's own bias is common to both sides.

The round is projected in one mode, resolved on the stack
(``HYPERLOOM_INFERASIM_MODE``, default ``auto``):

* ``benchmark`` -- calibrated against an in-regime anchor. Under ``auto`` this
  is chosen whenever the anchor store already holds one for the stack, which
  includes the single-point anchors earlier decision rounds recorded; it is
  never harvested for. Kernel-regime levers (attention backend, speculative
  decoding) become visible because each regime has its own anchor, and the
  margin tightens to ``CALIBRATED_MARGIN_PCT`` -- but only for a comparison in
  which both sides are calibrated at regime distance 0. Otherwise the simulate
  margin applies. A variant whose regime has no anchor is benchmarked.
* ``simulate`` -- no anchor. A variant is dropped when it projects more than
  ``SIMULATE_MARGIN_PCT`` behind the stack, a margin above the 9.7% median
  output-throughput error simulate mode scored on held-out AgentX
  configurations.

Of the visible variants that survive, at most ``HYPERLOOM_EXPLORE_PROJECTION_TOP_K``
(by projected gain) are forwarded; ``0`` forwards all of them. The default of
5 is where the replay's shortlist held the measured best on 94% of held-out
sets, against 47% for its single pick.

Off unless ``HYPERLOOM_EXPLORE_PROJECTION=1``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .._grid_base import GridVariant

log = logging.getLogger(__name__)

ENV_ENABLED = "HYPERLOOM_EXPLORE_PROJECTION"
ENV_SIMULATE_MARGIN = "HYPERLOOM_EXPLORE_PROJECTION_MARGIN_PCT"
ENV_CALIBRATED_MARGIN = "HYPERLOOM_EXPLORE_PROJECTION_CALIBRATED_MARGIN_PCT"
ENV_TOP_K = "HYPERLOOM_EXPLORE_PROJECTION_TOP_K"

SIMULATE_MARGIN_PCT = 15.0
# Placeholder until the projected-gap error between two calibrated configs
# sharing an anchor has been measured; tune with the env above.
CALIBRATED_MARGIN_PCT = 5.0
DEFAULT_TOP_K = 5

ROUTE_MEASURE = "measure"
ROUTE_DROP = "drop"
REASON_SLOWER = "projection_decisively_slower"
REASON_OUTRANKED = "projection_outranked"
REASON_UNREADABLE = "projection_unreadable"

STACK = "__projection_stack__"

# Server flags whose effect reaches the projection. A variant that sets any
# other flag carries a lever the projection cannot price, so it is measured.
_VISIBLE_FLAGS = frozenset(
    {
        "--tensor-parallel-size",
        "-tp",
        "--tp",
        "--tp-size",
        "--pipeline-parallel-size",
        "-pp",
        "--pp-size",
        "--ep-size",
        "--expert-parallel-size",
        "--moe-ep-size",
        "--max-num-seqs",
        "--max-running-requests",
        "--kv-cache-dtype",
        "--kv_cache_dtype",
    }
)
# Visible only when each regime is anchored: simulate mode does not price them.
_REGIME_FLAGS = frozenset(
    {
        "--attention-backend",
        "--speculative-config",
        "--speculative-algorithm",
        "--speculative-num-steps",
        "--num-speculative-tokens",
        "--speculative-num-draft-tokens",
        "--speculative-tokens",
    }
)
_VISIBLE_ENVS = frozenset({"TP", "EP", "PP", "CONC", "ISL", "OSL"})


def projection_enabled() -> bool:
    """Off unless explicitly turned on."""
    return str(os.environ.get(ENV_ENABLED, "0")).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ProjectionRequest:
    """What the router needs from the round.

    ``materialize`` writes the benchmark YAML the round would run for a variant
    on the stack (the stack itself for an empty one) under the given directory
    and returns its path. ``gpu`` is the session's GPU, which the projection is
    priced for; ``runtime`` is the deployment's image digest / engine version,
    so measured anchors recorded on another are not used.
    """

    materialize: Callable[[GridVariant, Path], Path]
    output_root: Path
    gpu: str = ""
    runtime: dict[str, str] = field(default_factory=dict)
    grade_on_intvty: bool = False


@dataclass
class Routed:
    """The variants to benchmark, the dropped ones, and an audit summary (None when the router did nothing)."""

    survivors: list[GridVariant]
    dropped: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, Any] | None = None


@dataclass
class VariantProjection:
    """One variant's projection and what the router did with it."""

    name: str
    route: str = ROUTE_MEASURE
    reason: str = ""
    detail: str = ""
    calibrated: bool = False
    regime_distance: int | None = None
    output_throughput: float | None = None
    interactivity: float | None = None
    tput_gap_pct: float | None = None
    intvty_gap_pct: float | None = None
    margin_pct: float | None = None
    extrapolation: list[str] = field(default_factory=list)


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _top_k() -> int:
    try:
        return max(0, int(os.environ.get(ENV_TOP_K, DEFAULT_TOP_K)))
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


def _flags(server_args: str) -> list[str]:
    return [tok.split("=", 1)[0] for tok in (server_args or "").split() if tok.startswith("-")]


def only_visible_levers(variant: GridVariant, mode: str) -> bool:
    """Whether every lever this variant changes is one the projection prices."""
    from .inferasim_bridge import MODE_BENCHMARK

    visible = _VISIBLE_FLAGS | (_REGIME_FLAGS if mode == MODE_BENCHMARK else frozenset())
    if getattr(variant, "remove_args", None) or getattr(variant, "unset_envs", None):
        return False
    if str(getattr(variant, "args_mode", "append") or "append").lower() == "replace":
        return False
    if any(str(k).upper() not in _VISIBLE_ENVS for k in (variant.extra_envs or {})):
        return False
    if getattr(variant, "overlay_pythonpath", "") or getattr(variant, "runtime_override", None):
        return False
    return all(flag in visible for flag in _flags(variant.extra_server_args))


def _projection_key(spec: Any, mode: str) -> tuple:
    from .inferasim_bridge import MODE_BENCHMARK, recipe_from_spec

    key: tuple = (spec.tp, spec.ep, spec.pp, spec.conc, spec.isl, spec.osl, spec.weight_dtype, spec.kv_cache_dtype)
    if mode == MODE_BENCHMARK:
        recipe = recipe_from_spec(spec)
        key += (recipe.get("attention_backend"), recipe.get("speculative"))
    return key


def _gap_pct(candidate: float | None, reference: float | None) -> float | None:
    if not candidate or not reference or reference <= 0:
        return None
    return (candidate / reference - 1.0) * 100.0


def _interactivity(metrics: Any) -> float | None:
    tpot = float(getattr(metrics, "tpot_ms", 0.0) or 0.0)
    return 1000.0 / tpot if tpot > 0 else None


def _extras(metrics: Any) -> dict[str, Any]:
    return getattr(metrics, "extras", None) or {}


def _anchor_distance(metrics: Any) -> int | None:
    value = _extras(metrics).get("anchor_regime_distance")
    return int(value) if value is not None else None


def _estimator(metrics: Any) -> str | None:
    return _extras(metrics).get("estimator")


def _extrapolation(metrics: Any) -> list[str]:
    return list(_extras(metrics).get("extrapolation") or [])


@dataclass
class _Round:
    """The projected stack every variant is compared with, and how the comparison is judged."""

    request: ProjectionRequest
    project: Callable[..., Any]
    mode: str
    stack: Any
    stack_key: tuple
    stack_confident: bool
    simulate_margin: float
    calibrated_margin: float

    def spec_for(self, variant: GridVariant, label: str) -> Any:
        return _spec_for(self.request, variant, label)

    def judge(self, rec: VariantProjection, metrics: Any) -> float | None:
        """Fill ``rec`` from ``metrics``; return the variant's ranking score, or None once it is routed."""
        rec.calibrated = bool(metrics.calibrated)
        rec.regime_distance = _anchor_distance(metrics)
        rec.output_throughput = float(metrics.output_throughput or 0.0)
        rec.interactivity = _interactivity(metrics)
        rec.extrapolation = _extrapolation(metrics)
        rec.tput_gap_pct = _gap_pct(rec.output_throughput, self.stack.output_throughput)
        rec.intvty_gap_pct = _gap_pct(rec.interactivity, _interactivity(self.stack))
        confident = self.stack_confident and rec.calibrated and rec.regime_distance in (0, None)
        rec.margin_pct = self.calibrated_margin if confident else self.simulate_margin
        on_intvty = self.request.grade_on_intvty
        if rec.tput_gap_pct is None or (on_intvty and rec.intvty_gap_pct is None):
            rec.reason = REASON_UNREADABLE
            return None
        behind_tput = rec.tput_gap_pct < -rec.margin_pct
        behind_intvty = rec.intvty_gap_pct is not None and rec.intvty_gap_pct < -rec.margin_pct
        if behind_tput and (behind_intvty or not on_intvty):
            rec.route, rec.reason = ROUTE_DROP, REASON_SLOWER
            rec.detail = (
                f"projected output {rec.tput_gap_pct:+.1f}% vs stack"
                + (f", interactivity {rec.intvty_gap_pct:+.1f}%" if on_intvty else "")
                + f" ({'calibrated' if confident else 'simulate'} margin {rec.margin_pct:.0f}%)"
            )
            return None
        rec.reason = "projected_competitive"
        return float((rec.intvty_gap_pct if on_intvty else rec.tput_gap_pct) or 0.0)


def _spec_for(request: ProjectionRequest, variant: GridVariant, label: str) -> Any:
    from . import inferasim_bridge as bridge

    cfg_path = request.materialize(variant, Path(request.output_root) / label)
    with Path(cfg_path).open(encoding="utf-8") as fh:
        bench = (yaml.safe_load(fh) or {}).get("benchmark") or {}
    spec = bridge.spec_from_benchmark(bench, ambient=False)
    spec.runtime = dict(request.runtime)
    spec.gpu = request.gpu
    return spec


def _open_round(request: ProjectionRequest, project: Callable[..., Any] | None) -> _Round | None:
    """Project the stack; None when it cannot be projected and the round must run untouched."""
    from . import inferasim_anchors as anchors
    from . import inferasim_bridge as bridge

    project = project or anchors.project
    try:
        bridge.projection_mode()
        stack_spec = _spec_for(request, GridVariant(name=STACK), "stack")
        mode = anchors.resolve_mode(stack_spec)
        stack = project(stack_spec, mode=mode)
    except Exception as exc:  # noqa: BLE001 - never fail the round over a projection
        log.warning("explore projection: stack projection failed (%s); benchmarking the full round", exc)
        return None
    return _Round(
        request=request,
        project=project,
        mode=mode,
        stack=stack,
        stack_key=_projection_key(stack_spec, mode),
        stack_confident=(
            mode == bridge.MODE_BENCHMARK and bool(stack.calibrated) and _anchor_distance(stack) in (0, None)
        ),
        simulate_margin=_float_env(ENV_SIMULATE_MARGIN, SIMULATE_MARGIN_PCT),
        calibrated_margin=_float_env(ENV_CALIBRATED_MARGIN, CALIBRATED_MARGIN_PCT),
    )


def _route_one(rnd: _Round, variant: GridVariant, idx: int) -> tuple[VariantProjection, float | None]:
    """Project one variant against the stack; the score is set only for a competitive one."""
    rec = VariantProjection(name=variant.name)
    if not only_visible_levers(variant, rnd.mode):
        rec.reason = "lever_not_projected"
        return rec, None
    try:
        spec = rnd.spec_for(variant, f"v{idx:02d}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        rec.reason, rec.detail = "materialize_failed", str(exc)[-300:]
        return rec, None
    if _projection_key(spec, rnd.mode) == rnd.stack_key:
        rec.reason = "projects_as_stack"
        return rec, None
    try:
        metrics = rnd.project(spec, mode=rnd.mode)
    except Exception as exc:  # noqa: BLE001 - a variant that fails to project is benchmarked
        log.warning("explore projection: could not project %r (%s); benchmarking it", variant.name, exc)
        rec.reason, rec.detail = "projection_failed", str(exc)[-300:]
        return rec, None
    if _estimator(metrics) != _estimator(rnd.stack):
        rec.reason = REASON_UNREADABLE
        rec.detail = f"estimator {_estimator(metrics)} against the stack's {_estimator(rnd.stack)}"
        return rec, None
    return rec, rnd.judge(rec, metrics)


def _keep_top_k(records: list[VariantProjection], candidates: list[tuple[float, int]], top_k: int) -> None:
    if not top_k or len(candidates) <= top_k:
        return
    candidates.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, idx) in enumerate(candidates[top_k:], start=top_k + 1):
        rec = records[idx]
        rec.route, rec.reason = ROUTE_DROP, REASON_OUTRANKED
        rec.detail = f"projected gain {score:+.1f}% ranks {rank} of {len(candidates)}; top {top_k} forwarded"


def _summary(rnd: _Round, records: list[VariantProjection], top_k: int, forwarded: int) -> dict[str, Any]:
    return {
        "mode": rnd.mode,
        "estimator": _estimator(rnd.stack),
        "gpu": rnd.request.gpu,
        "stack": {
            "output_throughput": rnd.stack.output_throughput,
            "interactivity": _interactivity(rnd.stack),
            "calibrated": bool(rnd.stack.calibrated),
            "regime_distance": _anchor_distance(rnd.stack),
            "extrapolation": _extrapolation(rnd.stack),
        },
        "grade_on_intvty": rnd.request.grade_on_intvty,
        "top_k": top_k,
        "forwarded": forwarded,
        "dropped": len(records) - forwarded,
        "variants": [asdict(rec) for rec in records],
    }


def route_variants(
    variants: list[GridVariant], request: ProjectionRequest, *, project: Callable[..., Any] | None = None
) -> Routed:
    """Return the variants to benchmark, the dropped ones, and an audit summary.

    ``project`` is the projection callable (``inferasim_anchors.project``,
    called as ``project(spec, mode=...)``); injectable for tests. Any failure
    that leaves the stack without a projection returns the round untouched:
    the failure mode is the round we already run.
    """
    if not projection_enabled() or not variants:
        return Routed(list(variants))
    rnd = _open_round(request, project)
    if rnd is None:
        return Routed(list(variants))

    records: list[VariantProjection] = []
    candidates: list[tuple[float, int]] = []
    for idx, variant in enumerate(variants):
        rec, score = _route_one(rnd, variant, idx)
        records.append(rec)
        if score is not None:
            candidates.append((score, idx))
    top_k = _top_k()
    _keep_top_k(records, candidates, top_k)

    survivors = [v for v, rec in zip(variants, records) if rec.route == ROUTE_MEASURE]
    dropped = [
        {"name": rec.name, "reason": rec.reason, "detail": rec.detail} for rec in records if rec.route == ROUTE_DROP
    ]
    summary = _summary(rnd, records, top_k, len(survivors))
    root = Path(request.output_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        log.debug("explore projection: could not write summary under %s", root)
    log.info(
        "explore projection (%s, %s): %d/%d variants forwarded to benchmark; dropped %s",
        rnd.mode,
        request.gpu or "default GPU",
        len(survivors),
        len(variants),
        ", ".join(f"{d['name']} ({d['reason']})" for d in dropped) or "nothing",
    )
    return Routed(survivors, dropped, summary)
