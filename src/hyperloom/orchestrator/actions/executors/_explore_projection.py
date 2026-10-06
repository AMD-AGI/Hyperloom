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
variant materialized exactly as the round would launch them, so the
projection's own bias is common to both sides.

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

Variants the caller names in ``exempt`` (an exact past measurement answers
them) are passed through untouched.

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

from ._grid_base import GridVariant

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

REASON_SLOWER = "projection_decisively_slower"
REASON_OUTRANKED = "projection_outranked"

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
class VariantProjection:
    """One variant's projection and what the router did with it."""

    name: str
    route: str  # measure | drop
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


def _anchor_distance(metrics: Any) -> int | None:
    value = (getattr(metrics, "extras", None) or {}).get("anchor_regime_distance")
    return int(value) if value is not None else None


def _estimator(metrics: Any) -> str | None:
    return (getattr(metrics, "extras", None) or {}).get("estimator")


def route_variants(
    variants: list[GridVariant],
    *,
    materialize: Callable[[GridVariant, Path], Path],
    output_root: Path,
    grade_on_intvty: bool = False,
    project: Callable[..., Any] | None = None,
    runtime: dict[str, str] | None = None,
    exempt: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[GridVariant], list[dict[str, str]], dict[str, Any] | None]:
    """Return the variants to benchmark, the dropped ones, and an audit summary.

    Args:
        variants: The round's runnable variants, in run order.
        materialize: Writes the per-variant benchmark YAML the round would run
            for a variant (the stack itself for an empty one) and returns its
            path.
        output_root: Where the materialized configs and ``summary.json`` go.
        grade_on_intvty: The session grades on interactivity, so a variant is
            only decisively behind when it is behind on both axes.
        project: Projection callable (``inferasim_bridge.project``, called as
            ``project(spec, mode=...)``); injectable for tests.
        runtime: Image digest / engine version of the deployment; measured
            anchors recorded on another are not used.
        exempt: Names of variants to pass through without projecting.

    Any failure that leaves the stack without a projection returns the round
    untouched: the failure mode is the round we already run.
    """
    if not projection_enabled() or not variants:
        return list(variants), [], None

    from . import inferasim_bridge as bridge

    project = project or bridge.project
    try:
        bridge.projection_mode()
    except bridge.InferasimBridgeError as exc:
        log.warning("explore projection: %s; benchmarking the full round", exc)
        return list(variants), [], None

    root = Path(output_root)

    def spec_for(variant: GridVariant, label: str) -> Any:
        cfg_path = materialize(variant, root / label)
        with Path(cfg_path).open(encoding="utf-8") as fh:
            bench = (yaml.safe_load(fh) or {}).get("benchmark") or {}
        spec = bridge.spec_from_benchmark(bench, ambient=False)
        spec.runtime = dict(runtime or {})
        return spec

    try:
        stack_spec = spec_for(GridVariant(name=STACK), "stack")
        mode = bridge.resolve_mode(stack_spec)
        stack = project(stack_spec, mode=mode)
    except Exception as exc:  # noqa: BLE001 - never fail the round over a projection
        log.warning("explore projection: stack projection failed (%s); benchmarking the full round", exc)
        return list(variants), [], None

    stack_confident = mode == bridge.MODE_BENCHMARK and bool(stack.calibrated) and _anchor_distance(stack) in (0, None)
    stack_key = _projection_key(stack_spec, mode)
    stack_intvty = _interactivity(stack)
    simulate_margin = _float_env(ENV_SIMULATE_MARGIN, SIMULATE_MARGIN_PCT)
    calibrated_margin = _float_env(ENV_CALIBRATED_MARGIN, CALIBRATED_MARGIN_PCT)

    records: list[VariantProjection] = []
    candidates: list[tuple[float, int]] = []
    for idx, variant in enumerate(variants):
        rec = VariantProjection(name=variant.name, route="measure")
        records.append(rec)
        if variant.name in exempt:
            rec.reason = "measured_before"
            continue
        if not only_visible_levers(variant, mode):
            rec.reason = "lever_not_projected"
            continue
        try:
            spec = spec_for(variant, f"v{idx:02d}")
        except Exception as exc:  # noqa: BLE001
            rec.reason, rec.detail = "materialize_failed", str(exc)[-300:]
            continue
        if _projection_key(spec, mode) == stack_key:
            rec.reason = "projects_as_stack"
            continue
        try:
            metrics = project(spec, mode=mode)
        except Exception as exc:  # noqa: BLE001
            rec.reason, rec.detail = "projection_failed", str(exc)[-300:]
            continue
        if _estimator(metrics) != _estimator(stack):
            rec.reason = "projection_unreadable"
            rec.detail = f"estimator {_estimator(metrics)} against the stack's {_estimator(stack)}"
            continue

        rec.calibrated = bool(metrics.calibrated)
        rec.regime_distance = _anchor_distance(metrics)
        rec.output_throughput = float(metrics.output_throughput or 0.0)
        rec.interactivity = _interactivity(metrics)
        rec.extrapolation = list((metrics.extras or {}).get("extrapolation") or [])
        rec.tput_gap_pct = _gap_pct(rec.output_throughput, stack.output_throughput)
        rec.intvty_gap_pct = _gap_pct(rec.interactivity, stack_intvty)
        confident = stack_confident and rec.calibrated and rec.regime_distance in (0, None)
        rec.margin_pct = calibrated_margin if confident else simulate_margin
        if rec.tput_gap_pct is None or (grade_on_intvty and rec.intvty_gap_pct is None):
            rec.reason = "projection_unreadable"
            continue

        behind_tput = rec.tput_gap_pct < -rec.margin_pct
        behind_intvty = rec.intvty_gap_pct is not None and rec.intvty_gap_pct < -rec.margin_pct
        if behind_tput and (behind_intvty or not grade_on_intvty):
            rec.route, rec.reason = "drop", REASON_SLOWER
            rec.detail = (
                f"projected output {rec.tput_gap_pct:+.1f}% vs stack"
                + (f", interactivity {rec.intvty_gap_pct:+.1f}%" if grade_on_intvty else "")
                + f" ({'calibrated' if confident else 'simulate'} margin {rec.margin_pct:.0f}%)"
            )
            continue
        rec.reason = "projected_competitive"
        score = rec.intvty_gap_pct if grade_on_intvty else rec.tput_gap_pct
        candidates.append((float(score or 0.0), idx))

    top_k = _top_k()
    if top_k and len(candidates) > top_k:
        candidates.sort(key=lambda item: item[0], reverse=True)
        for rank, (score, idx) in enumerate(candidates[top_k:], start=top_k + 1):
            rec = records[idx]
            rec.route, rec.reason = "drop", REASON_OUTRANKED
            rec.detail = f"projected gain {score:+.1f}% ranks {rank} of {len(candidates)}; top {top_k} forwarded"

    survivors = [v for v, rec in zip(variants, records) if rec.route == "measure"]
    dropped = [{"name": rec.name, "reason": rec.reason, "detail": rec.detail} for rec in records if rec.route == "drop"]
    summary = {
        "mode": mode,
        "estimator": _estimator(stack),
        "stack": {
            "output_throughput": stack.output_throughput,
            "interactivity": stack_intvty,
            "calibrated": bool(stack.calibrated),
            "regime_distance": _anchor_distance(stack),
            "extrapolation": list((stack.extras or {}).get("extrapolation") or []),
        },
        "grade_on_intvty": grade_on_intvty,
        "top_k": top_k,
        "forwarded": len(survivors),
        "dropped": len(dropped),
        "variants": [asdict(rec) for rec in records],
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        log.debug("explore projection: could not write summary under %s", root)
    log.info(
        "explore projection (%s): %d/%d variants forwarded to benchmark; dropped %s",
        mode,
        len(survivors),
        len(variants),
        ", ".join(f"{d['name']} ({d['reason']})" for d in dropped) or "nothing",
    )
    return survivors, dropped, summary
