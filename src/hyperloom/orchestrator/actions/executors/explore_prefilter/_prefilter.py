# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Narrow an EXPLORE round before it benchmarks anything, and record what it measured.

Three opt-in filters run once, before the per-variant loop, against the stack
the round opens on, in this order:

1. ``HYPERLOOM_MEASUREMENT_STORE`` -- a variant whose exact launch on this
   stack already reverted on the performance gate (:mod:`_reuse`).
2. ``HYPERLOOM_EXPLORE_SCREEN`` -- a reduced-scale probe puts it decisively
   behind the stack (:mod:`_screen`).
3. ``HYPERLOOM_EXPLORE_PROJECTION`` -- InferaSim projects it decisively behind
   the stack, or outside the top 5 (:mod:`_projection`).

Each only removes variants, as ``SKIPPED_DEDUP`` rows that are neither
journaled nor written to the KB. None produces a number that reaches
KEEP/REVERT: a KEEP always rests on the survivor's own decision round, measured
now. Variants the caller protects -- a stack revalidation, a config replay --
are never filtered.

After a decision round, :func:`record_decision_round` stores a performance
REVERT for filter 1 and, with ``HYPERLOOM_INFERASIM_ANCHOR_STORE`` set, indexes
the measurement as an InferaSim anchor.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.coerce import to_float, to_str_list
from hyperloom.inference_optimizer.canonical_fingerprint import workload_signature
from hyperloom.inference_optimizer.grading import resolved_grading

from . import _reuse as reuse
from ._projection import ProjectionRequest, projection_enabled, route_variants
from ._remeasure import _is_noop_filter_exempt
from ._screen import screen_enabled, screen_variants
from .._grid_base import GridVariant
from .._grid_runner import _build_variant_yaml

log = logging.getLogger(__name__)

REASON_REVERTED_BEFORE = "measured_revert_before"
ARGS_APPEND = "append"
# Reverts the accuracy gate decided; a rerun could measure a different score.
_ACCURACY_REASONS = frozenset({"accuracy_drop", "accuracy_unavailable"})


@dataclass(frozen=True)
class Stack:
    """The stack a variant launches on top of."""

    extra_args: str = ""
    extra_envs: dict[str, str] = field(default_factory=dict)
    remove_args: list[str] = field(default_factory=list)
    unset_envs: list[str] = field(default_factory=list)
    args_mode: str = ARGS_APPEND

    @classmethod
    def from_measured(cls, measured_against: dict[str, Any]) -> Stack:
        """From the ``measured_against`` block a ledger row records."""
        return cls(
            extra_args=str(measured_against.get("extra_server_args") or ""),
            extra_envs=dict(measured_against.get("extra_envs") or {}),
            remove_args=list(measured_against.get("remove_args") or []),
            unset_envs=list(measured_against.get("unset_envs") or []),
            args_mode=str(measured_against.get("args_mode") or ARGS_APPEND),
        )


@dataclass(frozen=True)
class Launch:
    """The removals, unsets and envs a variant launches with on a stack."""

    remove_args: list[str]
    unset_envs: list[str]
    extra_envs: dict[str, str]


def launch_of(gv: GridVariant, stack: Stack) -> Launch:
    if stack.args_mode == "replace":
        remove = to_str_list(getattr(gv, "remove_args", []))
    else:
        remove = list(dict.fromkeys(list(stack.remove_args) + to_str_list(getattr(gv, "remove_args", []))))
    unset = list(dict.fromkeys(list(stack.unset_envs) + to_str_list(getattr(gv, "unset_envs", []))))
    envs = dict(stack.extra_envs)
    for key in gv.unset_envs:
        envs.pop(key, None)
    envs.update(gv.extra_envs)
    return Launch(remove_args=remove, unset_envs=unset, extra_envs=envs)


def _without_pythonpath(envs: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in sorted(envs.items()) if k != "PYTHONPATH"}


@dataclass
class RoundContext:
    """What every launch in the round shares: how to materialize one and what identifies it.

    ``build`` holds the ``model_path`` / ``gpu_type`` / ``benchmark_script``
    the round materializes its configs with. ``decision_protocol`` is ``warm``
    or ``cold``.
    """

    config_path: Path
    output_root: Path
    state: Any
    build: dict[str, Any]
    inherited_args: str
    keep_threshold_pct: float
    decision_protocol: str
    session_dir: Path | None = None

    @classmethod
    def of_round(cls, explore_round: Any) -> RoundContext:
        """The context of an ``explore._ExploreRound`` once its schedule is open."""
        return cls(
            config_path=explore_round.config_path,
            output_root=explore_round.output_root,
            state=explore_round.ss,
            build=explore_round.build,
            inherited_args=explore_round.inherited_args,
            keep_threshold_pct=explore_round.keep_threshold_pct,
            decision_protocol="warm" if explore_round.use_warm_decision else "cold",
            session_dir=explore_round.session_dir,
        )

    @functools.cached_property
    def base_bench(self) -> dict[str, Any]:
        """The ``benchmark`` block of the round's config."""
        try:
            with Path(self.config_path).open(encoding="utf-8") as fh:
                return dict((yaml.safe_load(fh) or {}).get("benchmark") or {})
        except (OSError, yaml.YAMLError):
            return {}

    @property
    def framework(self) -> str:
        return str(self.base_bench.get("framework") or "").lower()

    @property
    def grade_on_intvty(self) -> bool:
        return bool(resolved_grading(self.state)[0])

    @property
    def gpu(self) -> str:
        return str(self.build.get("gpu_type") or getattr(self.state, "gpu_type", "") or "")

    @property
    def runtime(self) -> dict[str, str]:
        return reuse.runtime_identity(getattr(self.state, "stack_fingerprint_meta", None), self.framework)

    def materialize(self, gv: GridVariant, stack: Stack, subdir: Path) -> Path:
        return _build_variant_yaml(
            self.config_path,
            stack.extra_args,
            gv,
            output_subdir=subdir,
            base_args_mode=stack.args_mode,
            base_extra_envs=dict(stack.extra_envs),
            base_remove_args=list(stack.remove_args),
            base_unset_envs=list(stack.unset_envs),
            **self.build,
        )

    def _session_identity(self) -> dict[str, Any]:
        return {
            "model": str(self.build.get("model_path") or ""),
            "framework": self.framework,
            "gpu": self.gpu.lower(),
            "workload_signature": workload_signature(),
            "runtime": dict(sorted(self.runtime.items())),
            "benchmark": reuse.benchmark_identity(
                self.base_bench, volatile_roots=[str(self.output_root), str(self.session_dir or "")]
            ),
            "inherited_args": self.inherited_args,
            "decision_protocol": self.decision_protocol,
            "grade_on_intvty": self.grade_on_intvty,
            "keep_threshold_pct": self.keep_threshold_pct,
        }

    def identity(self, gv: GridVariant, stack: Stack) -> dict[str, Any] | None:
        """What a decision round of ``gv`` on ``stack`` depends on, or None when it cannot be pinned."""
        if getattr(gv, "runtime_override", None):
            return None
        launch = launch_of(gv, stack)
        pythonpath = [str(getattr(gv, "overlay_pythonpath", "") or "")]
        for envs in (launch.extra_envs, stack.extra_envs, self.base_bench.get("envs") or {}):
            pythonpath += str(envs.get("PYTHONPATH") or "").split(":")
        stack_id = {
            "extra_server_args": stack.extra_args,
            "extra_envs": _without_pythonpath(stack.extra_envs),
            "remove_args": list(stack.remove_args),
            "unset_envs": list(stack.unset_envs),
            "args_mode": stack.args_mode,
        }
        variant_id = {
            "fingerprint": getattr(gv, "canonical_fp", ""),
            "extra_server_args": gv.extra_server_args,
            "args_mode": str(getattr(gv, "args_mode", ARGS_APPEND) or ARGS_APPEND),
            "remove_args": launch.remove_args,
            "unset_envs": launch.unset_envs,
            "extra_envs": _without_pythonpath(launch.extra_envs),
            "accepted_kernels": sorted(getattr(gv, "accepted_kernels", []) or []),
        }
        return reuse.measurement_identity(self._session_identity(), stack_id, variant_id, pythonpath)


@dataclass
class Prefiltered:
    """The variants left to benchmark, the dropped ones as skip rows, and the projection's audit summary."""

    runnable: list[GridVariant]
    dropped: list[dict[str, str]] = field(default_factory=list)
    projection: dict[str, Any] | None = None


def _drop_known_reverts(
    variants: list[GridVariant], ctx: RoundContext, stack: Stack
) -> tuple[list[GridVariant], list[dict[str, str]]]:
    if not reuse.reuse_enabled():
        return list(variants), []
    survivors, dropped = [], []
    for gv in variants:
        identity = ctx.identity(gv, stack)
        record = reuse.lookup_revert(reuse.measurement_key(identity)) if identity else None
        if record is None:
            survivors.append(gv)
            continue
        dropped.append(
            {
                "name": gv.name,
                "reason": REASON_REVERTED_BEFORE,
                "detail": (
                    f"this launch reverted on {record.get('ts')} ({record.get('reason') or 'below the bar'}, "
                    f"round {record.get('round_id')} of {record.get('session_dir') or 'an earlier session'})"
                ),
            }
        )
    return survivors, dropped


async def prefilter_variants(
    variants: list[GridVariant], ctx: RoundContext, stack: Stack, revalidation: bool = False
) -> Prefiltered:
    """Run the enabled filters over ``variants`` on ``stack``.

    A revalidation round, a config replay and a resumed revalidation re-measure a config on purpose, so none of
    them is ever filtered.
    """
    protected = {gv.name for gv in variants if revalidation or _is_noop_filter_exempt(gv)}
    candidates = [gv for gv in variants if gv.name not in protected]
    if not candidates or not (reuse.reuse_enabled() or screen_enabled() or projection_enabled()):
        return Prefiltered(list(variants))
    candidates, dropped = _drop_known_reverts(candidates, ctx, stack)
    candidates, screened_out = await asyncio.to_thread(
        screen_variants, candidates, ctx.config_path, session_dir=ctx.session_dir
    )
    request = ProjectionRequest(
        materialize=lambda gv, subdir: ctx.materialize(gv, stack, subdir),
        output_root=Path(ctx.output_root) / "_projection",
        gpu=ctx.gpu,
        runtime=ctx.runtime,
        grade_on_intvty=ctx.grade_on_intvty,
    )
    routed = await asyncio.to_thread(route_variants, candidates, request)
    kept = {id(gv) for gv in routed.survivors}
    runnable = [gv for gv in variants if gv.name in protected or id(gv) in kept]
    return Prefiltered(runnable, [*dropped, *screened_out, *routed.dropped], routed.summary)


@dataclass
class Verdict:
    """How the decision round of a variant came out, and on which stack and anchor."""

    outcome: str
    reason: str
    anchor_tput: float
    round_id: str
    stack: Stack

    @classmethod
    def of_row(cls, row: dict[str, Any]) -> Verdict:
        """From the tested-ledger row of a decision round, which records the stack and anchor it was judged on."""
        return cls(
            outcome=str(row.get("outcome") or ""),
            reason=str(row.get("reason") or ""),
            anchor_tput=to_float(row.get("base_tput"), 0.0) or 0.0,
            round_id=str(row.get("round_id") or ""),
            stack=Stack.from_measured(row.get("measured_against") or {}),
        )


def _record_revert(ctx: RoundContext, gv: GridVariant, result: Any, verdict: Verdict) -> None:
    if verdict.outcome != reuse.OUTCOME_REVERT or verdict.reason in _ACCURACY_REASONS or verdict.anchor_tput <= 0:
        return
    identity = ctx.identity(gv, verdict.stack)
    if identity is None:
        return
    reuse.record_revert(
        reuse.measurement_key(identity),
        identity,
        {
            "reason": verdict.reason,
            "base_tput": verdict.anchor_tput,
            "output_throughput": getattr(result, "output_throughput", None),
            "variant_name": gv.name,
            "round_id": verdict.round_id,
            "session_dir": str(ctx.session_dir or ""),
            "workspace": getattr(result, "workspace", None),
        },
    )


def _record_anchor(ctx: RoundContext, gv: GridVariant, result: Any, verdict: Verdict) -> None:
    from . import inferasim_anchors as anchors
    from . import inferasim_bridge as bridge

    if not os.environ.get(bridge.ENV_ANCHOR_STORE) or not getattr(result, "tpot_mean_ms", None):
        return
    try:
        cfg = ctx.materialize(gv, verdict.stack, Path(ctx.output_root) / "_anchor_spec" / (gv.name or "variant"))
        with Path(cfg).open(encoding="utf-8") as fh:
            spec = bridge.spec_from_benchmark((yaml.safe_load(fh) or {}).get("benchmark") or {}, ambient=False)
        spec.runtime, spec.gpu = ctx.runtime, ctx.gpu
        source = {"round_id": verdict.round_id, "variant": gv.name, "workspace": getattr(result, "workspace", None)}
        anchors.record_measured_anchor(spec, tpot_mean_ms=result.tpot_mean_ms, source=source)
    except Exception as exc:  # noqa: BLE001 - an anchor is a by-product; never fail the round over it
        log.warning("explore: could not record %s as an InferaSim anchor (%s)", gv.name, exc)


async def record_decision_round(ctx: RoundContext, gv: GridVariant, result: Any, row: dict[str, Any]) -> None:
    """Keep what a measured decision round can tell later rounds; a no-op unless a store is configured."""
    if getattr(result, "status", "") != "succeeded":
        return
    verdict = Verdict.of_row(row)
    await asyncio.to_thread(_record_revert, ctx, gv, result, verdict)
    await asyncio.to_thread(_record_anchor, ctx, gv, result, verdict)
