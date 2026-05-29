"""ExploreExecutor — v0.8 M3.

Merges the legacy ``backends`` / ``params`` / ``validate_stack`` actions
into one unified ``explore`` action:

* one yaml meta (``actions/_meta/explore.yaml``),
* one ledger (``SharedState.explore_search``, see ``shared_state.py``),
* one executor (this module).

Per-variant flow:

1. canonical_fingerprint dedup against ``explore_search.tested``
   (rename-resistant; LLM/specialist/default_grid all collapse to the
   same row when content matches).
2. Render the variant's Magpie YAML, run E2E bench.
3. Immediate KEEP/REVERT decision (0.2% gain threshold + accuracy gate
   when the variant trips ``_accuracy_gate.is_high_accuracy_risk``).
4. KEEP triggers an inlined stack rebench: re-bench the cumulative
   stack including the just-KEEP'd variant. If the rebench tput is
   below the configurable threshold (default: baseline_tput * 1.005),
   the variant is evicted (``KEEP_UNSTABLE``) and treated as REVERT.

This deliberately differs from v0.6 backends/params which were "run
the whole batch, then pick best" — v0.8 follows the TBO "one change at
a time" rule (KB_design §3.4 §"Inv-3 serving GPU single tenant" +
KB_design.MD §3 "iron rules").

NOTE on M3 layering: this milestone does NOT yet route specialist
provenance (M5/M6). variants come in with ``provenance ∈
{'default_grid', 'llm_direct'}``; the executor passes provenance through
unchanged to the ledger so the M5 specialist path can fill the third
case (``'specialist:<domain>'``) without touching this file again.

Result schema (returned to the bus):

    status:                   "succeeded" | "failed"
    output_throughput:        float | None  (best variant or last stack rebench)
    best_variant:             dict | None
    winners:                  list[dict]   (KEEP'd in this batch, post-rebench)
    losers:                   list[dict]   (REVERT'd)
    keep_unstable_in_stack:   list[dict]   (KEEP'd then evicted by stack rebench)
    explore_search_update:    dict         (ledger increment)
    discovered_flags_update:  dict | None  (specialist-emitted; M5)
    workspace:                str          (output root)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ...session_paths import runs_dir
from ._accuracy_gate import (
    accuracy_passed,
    is_high_accuracy_risk,
    parse_eval_results,
)
from ._canonical_fingerprint import canonical_fingerprint, workload_signature
from ._explore_roofline_filter import filter_variants_by_roofline
from ._grid_runner import (
    GridVariant,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


# Per-variant KEEP threshold ("0.2% 阈值 + accuracy
# gate"). Looser than v0.6 backends (1.0%) and ``ParamsExecutor`` (0.5%)
# because the inlined stack rebench acts as the second gate — even a
# marginal +0.3% won't survive into ``optimization_stack`` unless the
# cumulative stack still wins after this variant is layered onto it.
DEFAULT_KEEP_THRESHOLD_PCT = 0.2

# Stack rebench stability threshold. After a KEEP, the stack-applied
# rebench tput must beat ``base_tput * (1 + DEFAULT_STACK_STABLE_PCT/100)``;
# otherwise the variant is evicted (KEEP_UNSTABLE → REVERT).
#
# Default lowered from 0.5% → 0.2% so the rebench gate matches
# ``DEFAULT_KEEP_THRESHOLD_PCT`` (per-variant KEEP threshold). The 0.5%
# default was originally KB_design §3.4 §4.4's "保守值"; in practice it
# sat squarely inside the ±1% inter-run noise band observed on MI300X
# (e.g. Qwen3-32B FP8: single-run +0.4% rebench dipping to −0.4% on the
# 2nd run), which silently downgraded otherwise-real wins to
# KEEP_UNSTABLE and pushed `cumulative_gain_validated` to 0%.
# Matching the single-variant KEEP threshold keeps both decisions
# consistent under the same noise floor.
DEFAULT_STACK_STABLE_PCT = 0.2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_explore_search_state() -> dict[str, Any]:
    """Empty :attr:`SharedState.explore_search` ledger (M3 schema v1)."""
    return {
        "schema_version": 1,
        "tested": {},
        "accepted": [],
        "rejected": [],
        "winners_history": [],
        "discovered_flags": [],
        "synergy_attempted": [],
        "domains_round_summary": [],
        "name_index": {},
        "cursor": 0,
        "last_round": {},
    }


def _grid_variants_from_payload(payload: list[Any]) -> list[GridVariant]:
    """Convert the LLM/specialist grid payload into GridVariant objects.

    Variant dict accepted shape (M3, KB_design §3.4 §5.1):

        {
          "name": str (required, unique-in-round),
          "extra_args" | "extra_sglang_args": str,
          "extra_envs": dict[str,str],
          "note": str,
          "provenance": str,            # default_grid / llm_direct / specialist:<domain>
          "kb_evidence": list,          # passthrough (M5/M6 uses)
          "pr_evidence": list,          # passthrough
          "source_evidence": list,      # passthrough
        }

    Unknown keys are ignored. ``provenance`` defaults to ``'default_grid'``
    when the LLM/specialist forgot to stamp it.

    PR-A9 (Arbor-into-Hyperloom): the legacy ``'llm_direct'`` default
    was retired. PolicyGate's ``explore_requires_specialist_provenance``
    rule denies grids whose variants are all ``llm_direct``, so we
    fall back to ``'default_grid'`` for unstamped variants — that
    keeps the executor running on cold-start grids while still
    letting the policy gate flag deliberately llm_direct-stamped
    grids upstream.
    """
    out: list[GridVariant] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        args = str(raw.get("extra_args") or raw.get("extra_sglang_args") or "").strip()
        envs_raw = raw.get("extra_envs") or {}
        envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}
        gv = GridVariant(
            name=str(raw["name"]),
            extra_sglang_args=args,
            extra_envs=envs,
            note=str(raw.get("note") or raw.get("provenance") or ""),
        )
        # Stash the extra M3 metadata on the GridVariant instance so the
        # ledger writer below can pull provenance / evidence without
        # round-tripping through a parallel list. GridVariant is a plain
        # dataclass, so adding attrs is safe (they just aren't part of
        # equality).
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.kb_evidence = list(raw.get("kb_evidence") or [])         # type: ignore[attr-defined]
        gv.pr_evidence = list(raw.get("pr_evidence") or [])         # type: ignore[attr-defined]
        gv.source_evidence = list(raw.get("source_evidence") or []) # type: ignore[attr-defined]
        out.append(gv)
    return out


def _gain_pct(tput: float | None, base_tput: float) -> float | None:
    if (
        not isinstance(tput, (int, float))
        or tput <= 0
        or base_tput <= 0
    ):
        return None
    return (float(tput) - base_tput) / base_tput * 100.0


# ---------------------------------------------------------------------------
# Auto-derived per-variant hard timeout
# ---------------------------------------------------------------------------
# The legacy class default (``variant_timeout_sec=2400``) is a smoke-workload
# floor: it works for fast benches (small models, high TP, short OSL) where
# the baseline run lands in well under 40 min, but on Qwen3-32B TP=1 BF16
# CONC=64 ISL/OSL=1024 NUM_PROMPTS=320 the baseline itself takes ~70 min
# and every variant times out before producing a measurement. Rather than
# pick a new universal constant (which would just push the failure mode to
# the next slow-workload combination), we auto-derive the cap from the
# *measured* baseline runtime that the Coordinator already injects, with a
# safety margin above the soft kill ratio so the layered design (soft kill
# → hard cap) is preserved.
#
# Operator can still override per-task via ``params['variant_timeout_sec']``
# or globally via ``--explore-variant-timeout-sec`` (mirrored into
# SharedState by cli.py and re-injected into every explore task by the
# Coordinator). Floor + ceiling guard against pathological inputs.
DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC = 2400      # 40 min — legacy smoke-workload default
DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC = 14400   # 4 h — matches roofline composite budget
DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN = 0.5   # hard cap ≥ baseline × (kill_ratio + 0.5)


def _compute_explore_variant_timeout(
    baseline_runtime_sec: float,
    kill_ratio: float,
    *,
    floor_sec: int = DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC,
    ceiling_sec: int = DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
    safety_margin: float = DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN,
) -> int:
    """Derive the per-variant hard timeout from the measured baseline.

    Returns ``floor_sec`` when ``baseline_runtime_sec`` is unknown / non-positive
    (cold start, baseline failed, fresh resume before baseline replays). Once
    baseline lands, scales with the actual workload runtime so slow models
    get appropriate budget without operator tuning.

    The hard cap is intentionally **above** the soft kill ratio
    (``baseline_runtime_sec × kill_ratio``); the soft kill is the designed
    upper bound for "this variant is slower than baseline" and the hard cap
    is the catastrophic backstop for hung subprocesses. Inverting them — as
    the legacy 2400 s constant does on slow workloads — defeats the layered
    design (the hard cap fires before the soft kill ever gets a chance).

    Args:
        baseline_runtime_sec: Measured wall-clock of the baseline action.
            Coordinator-injected via ``task.params['baseline_runtime_sec']``.
            Pass ``0`` (or any non-positive) to force the ``floor_sec``
            fallback.
        kill_ratio: ``--explore-overtime-kill-ratio``. Treated as ``1.0`` if
            below 1.0 so the derived timeout never underflows the soft kill.
        floor_sec: Lower bound. Default preserves legacy smoke-workload
            behaviour (40 min) for any path that calls with no baseline.
        ceiling_sec: Upper bound. Default 4 h matches the roofline composite
            timeout; bumping further would risk a single hung variant
            burning a meaningful slice of the wall-clock budget.
        safety_margin: Additive margin on top of ``kill_ratio`` so the hard
            cap stays above the soft kill. ``0.5`` ≈ 50 % of the baseline
            runtime as headroom for one-off variant cold starts (e.g.
            ``--enable-torch-compile`` AOTI compile, fresh aiter shapes).
    """
    if baseline_runtime_sec <= 0:
        return int(floor_sec)
    effective_kill_ratio = max(1.0, float(kill_ratio))
    derived = float(baseline_runtime_sec) * (effective_kill_ratio + float(safety_margin))
    return int(max(floor_sec, min(ceiling_sec, derived)))


def _join_args(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action.

    Subsumes the retired v0.6 ``backends`` / ``params`` /
    ``validate_stack`` executors.
    Per-variant KEEP/REVERT gating plus an inlined per-KEEP stack
    rebench replace the standalone ``validate_stack`` step.
    """

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        variant_timeout_sec: int = 2400,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
        stack_stable_threshold_pct: float = DEFAULT_STACK_STABLE_PCT,
        enable_stack_rebench: bool = True,
    ):
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)
        self.stack_stable_threshold_pct = float(stack_stable_threshold_pct)
        # Stack-rebench can be disabled for unit tests / fast smoke runs.
        # KB_design §3.4 §4.4 says the inlined rebench is the legacy
        # default; flipping this off recovers behaviour.
        self.enable_stack_rebench = bool(enable_stack_rebench)

    async def __call__(self, ctx) -> dict[str, Any]:
        params = dict(ctx.task.params or {})
        # ----- Config / output workspace -----------------------------------
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or default_baseline_config()
        )
        if not config_path.exists():
            return {
                "status": "failed",
                "error_class": "missing_config",
                "error": f"config not found: {config_path}",
            }
        extra = getattr(ctx, "extra", None) or {}
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "explore", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # ----- Workload-contract materialization ---------------------------
        # Same pattern as backends/params: re-run materialize_config_with_envs
        # so the variant YAMLs honour the operator's actual workload
        # (CONC / ISL / OSL / TP / MAX_MODEL_LEN / PRECISION).
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
            }
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            out_name="explore_base.with_envs.yaml",
        )

        # ----- Inputs ------------------------------------------------------
        base_extra_args = str(params.get("base_extra_args") or "").strip()
        base_extra_envs = dict(params.get("base_extra_envs") or {})
        base_tput = float(params.get("base_tput") or 0.0)
        baseline_accuracy = (
            float(params.get("accuracy_baseline") or 0.0)
            or float(params.get("baseline_accuracy") or 0.0)
        )
        keep_threshold_pct = float(params.get(
            "keep_threshold_pct", self.keep_threshold_pct,
        ))
        stack_stable_threshold_pct = float(params.get(
            "stack_stable_threshold_pct", self.stack_stable_threshold_pct,
        ))
        enable_stack_rebench = bool(params.get(
            "enable_stack_rebench", self.enable_stack_rebench,
        ))

        # Fix E (per-variant overtime kill — Q1: anchored on baseline
        # wall-clock; Q4: single-variant runs only, NOT stack_rebench).
        # The Coordinator injects ``baseline_runtime_sec`` and
        # ``explore_overtime_kill_ratio`` into task.params from
        # SharedState. When either is missing / non-positive the
        # deadline stays None and the legacy ``variant_timeout_sec``
        # hard cap is the only gate.
        baseline_runtime_sec_raw = params.get("baseline_runtime_sec")
        try:
            baseline_runtime_sec = (
                float(baseline_runtime_sec_raw)
                if baseline_runtime_sec_raw is not None else 0.0
            )
        except (TypeError, ValueError):
            baseline_runtime_sec = 0.0
        overtime_kill_ratio_raw = params.get("explore_overtime_kill_ratio")
        try:
            overtime_kill_ratio = (
                float(overtime_kill_ratio_raw)
                if overtime_kill_ratio_raw is not None else 0.0
            )
        except (TypeError, ValueError):
            overtime_kill_ratio = 0.0
        if baseline_runtime_sec > 0 and overtime_kill_ratio > 0:
            overtime_deadline_sec: float | None = (
                baseline_runtime_sec * overtime_kill_ratio
            )
        else:
            overtime_deadline_sec = None

        # Resolve the per-variant hard cap. Precedence:
        #   1. ``params['variant_timeout_sec']`` — explicit per-task override
        #      (LLM proposal, operator-injected via Coordinator from the
        #      ``--explore-variant-timeout-sec`` CLI flag, or
        #      ``INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC`` env).
        #   2. Auto-derive from the measured baseline runtime + soft kill
        #      ratio (both Coordinator-injected). Scales with the actual
        #      workload so slow models don't time out before producing a
        #      measurement; see ``_compute_explore_variant_timeout``.
        #   3. ``self.variant_timeout_sec`` — class default, retained as a
        #      conservative floor when the auto-derive has no baseline yet
        #      (cold start / first round).
        explicit_timeout = params.get("variant_timeout_sec")
        if explicit_timeout is not None:
            timeout_sec = int(explicit_timeout)
        else:
            # Operator-tunable headroom. When unset (or negative), fall back
            # to the helper's default. Negative values clamp to 0 (no
            # headroom — hard cap collapses onto the soft kill ratio).
            safety_margin_raw = params.get("variant_timeout_safety_margin")
            try:
                safety_margin = (
                    max(0.0, float(safety_margin_raw))
                    if safety_margin_raw is not None
                    else DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN
                )
            except (TypeError, ValueError):
                safety_margin = DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN
            timeout_sec = _compute_explore_variant_timeout(
                baseline_runtime_sec=baseline_runtime_sec,
                kill_ratio=overtime_kill_ratio,
                floor_sec=int(self.variant_timeout_sec),
                safety_margin=safety_margin,
            )

        # Resolve framework from materialized YAML (informational only —
        # both sglang and vllm route through the same EXTRA_*_ARGS env;
        # _grid_runner picks the correct env name on render).
        try:
            with config_path.open(encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f) or {}
            framework = str(
                (_cfg.get("benchmark") or {}).get("framework") or ""
            ).lower()
        except Exception:  # noqa: BLE001
            framework = ""

        # ----- Variant grid ------------------------------------------------
        grid_payload = params.get("grid") or []
        if not isinstance(grid_payload, list) or not grid_payload:
            return {
                "status": "failed",
                "error_class": "empty_grid",
                "error": (
                    "explore: params.grid must be a non-empty list of variant "
                    "dicts. The Orchestration prompt "
                    "should fill this from specialist proposals / "
                    "SharedState.discovered_flags / default_grid."
                ),
                "workspace": output_root.as_posix(),
            }
        grid = _grid_variants_from_payload(grid_payload)
        if not grid:
            return {
                "status": "failed",
                "error_class": "empty_grid",
                "error": "explore: params.grid contains no valid variants",
                "workspace": output_root.as_posix(),
            }

        # ----- explore_search dedup ----------------------------------------
        search = dict(
            params.get("explore_search") or _initial_explore_search_state()
        )
        # Defensive default fill (resume / first-run guards).
        for key, default in (
            ("schema_version", 1),
            ("tested", {}),
            ("rejected", []),
            ("name_index", {}),
            ("cursor", 0),
            ("winners_history", []),
            ("synergy_attempted", []),
            ("discovered_flags", []),
            ("domains_round_summary", []),
        ):
            search.setdefault(key, default)

        def _entry_fp(entry: Any) -> str:
            if not isinstance(entry, dict):
                return ""
            fp = entry.get("fingerprint")
            if fp:
                return str(fp)
            return canonical_fingerprint(
                str(entry.get("extra_sglang_args") or ""),
                dict(entry.get("extra_envs") or {}),
            )

        tested_dict = search.get("tested") or {}
        seen_fps: set[str] = set(tested_dict.keys())
        for v in tested_dict.values():
            seen_fps.add(_entry_fp(v))
        for v in search.get("accepted") or []:
            seen_fps.add(_entry_fp(v))
        for v in search.get("rejected") or []:
            seen_fps.add(_entry_fp(v))
        seen_fps.discard("")
        name_index = dict(search.get("name_index") or {})

        # Per-variant fingerprint. We attach
        # the fingerprint as an attribute so the result loop doesn't
        # have to recompute. GridVariant.fingerprint already uses the
        # content-only hash; canonical_fingerprint is the same hash so
        # the two are interchangeable here.
        ws_sig = workload_signature()

        unique_in_round: dict[str, GridVariant] = {}
        skipped_dup: list[dict[str, Any]] = []
        for gv in grid:
            fp = canonical_fingerprint(gv.extra_sglang_args, gv.extra_envs)
            gv.canonical_fp = fp  # type: ignore[attr-defined]
            if fp in seen_fps:
                skipped_dup.append({
                    "name": gv.name,
                    "fingerprint": fp,
                    "reason": "ledger_dup",
                })
                continue
            if fp in unique_in_round:
                # In-round duplicate — keep the first occurrence.
                skipped_dup.append({
                    "name": gv.name,
                    "fingerprint": fp,
                    "reason": "round_dup",
                })
                continue
            unique_in_round[fp] = gv

        runnable: list[GridVariant] = list(unique_in_round.values())
        log.info(
            "explore dedup: payload=%d → runnable=%d (ledger_dup+round_dup=%d)",
            len(grid), len(runnable), len(skipped_dup),
        )

        # Opt-in roofline-categorized filter (PR-B).
        # Coordinator-injected ``roofline_hard_gate=True`` together with a
        # non-empty ``roofline_saturation_snapshot`` activates the gate;
        # the executor drops variants whose flags target only directions
        # the latest roofline run reports above the saturation threshold.
        # Dropped variants land in ``skipped_dup`` with
        # ``reason='roofline_saturated'`` so the per-variant outcomes
        # collector and ``state.json`` audit trail surface them next to
        # the dedup skips. Default is off — when ``roofline_hard_gate``
        # is missing / falsy, the soft advisory remains the only signal
        # (legacy behaviour).
        if bool(params.get("roofline_hard_gate", False)) and runnable:
            saturation_snapshot = params.get("roofline_saturation_snapshot")
            if isinstance(saturation_snapshot, dict) and saturation_snapshot:
                kept_runnable, dropped_by_roofline = filter_variants_by_roofline(
                    runnable, saturation_snapshot,
                )
                if dropped_by_roofline:
                    for entry in dropped_by_roofline:
                        skipped_dup.append({
                            "name": entry.get("name", ""),
                            "extra_sglang_args": entry.get("extra_sglang_args", ""),
                            "reason": "roofline_saturated",
                            "categories": entry.get("categories", []),
                            "saturated_directions": entry.get(
                                "saturated_directions", [],
                            ),
                        })
                    log.info(
                        "explore roofline gate: %d/%d variants dropped "
                        "(saturated=%s)",
                        len(dropped_by_roofline),
                        len(runnable),
                        ",".join(sorted(
                            d for d, p in saturation_snapshot.items()
                            if isinstance(p, (int, float)) and float(p) >= 80.0
                        )),
                    )
                runnable = kept_runnable

        round_id_seed = int(search.get("cursor") or 0) + 1
        round_id = f"explore-{round_id_seed:03d}"

        # ----- Per-variant serial run loop ---------------------------------
        winners: list[dict[str, Any]] = []
        losers: list[dict[str, Any]] = []
        keep_unstable: list[dict[str, Any]] = []
        tested_update: dict[str, dict[str, Any]] = dict(tested_dict)
        rejected_update: list[dict[str, Any]] = list(search.get("rejected") or [])
        winners_history_update: list[dict[str, Any]] = list(
            search.get("winners_history") or []
        )

        # ``stack_extra_args`` / ``stack_extra_envs`` carry the running
        # accumulation; after a KEEP they get extended with the KEEP'd
        # variant so the *next* variant in the same batch is benched on
        # top of the freshest stack ("重新
        # 计算 base_extra_args 给后续 variant").
        stack_extra_args = base_extra_args
        stack_extra_envs = dict(base_extra_envs)
        running_base_tput = base_tput
        # Track the in-batch stack of KEEP'd entries so the per-KEEP
        # stack rebench can recompose the full args from scratch (vs
        # incrementally) if the operator passes ``stack_rebench='full'``
        # — defaults to incremental which matches the runtime semantics.
        in_batch_keeps: list[dict[str, Any]] = []

        last_run_tput: float | None = None  # rebench/single-variant tput

        if runnable:
            for idx, gv in enumerate(runnable):
                fp = getattr(gv, "canonical_fp", "")
                provenance = getattr(gv, "provenance", "llm_direct")
                slot = output_root / f"v{idx:02d}_{_safe(gv.name)}"
                slot.mkdir(parents=True, exist_ok=True)
                # 1. Run the single variant on top of the running stack.
                #    ``soft_deadline_sec`` is the Fix-E overtime kill;
                #    stack_rebench below intentionally omits it (Q4).
                results = await run_grid(
                    base_yaml_path=config_path,
                    base_extra_args=stack_extra_args,
                    grid=[gv],
                    output_root=slot,
                    variant_timeout_sec=timeout_sec,
                    model_path=resolved_model,
                    gpu_type=resolved_gpu,
                    benchmark_script=override_script,
                    result_dir=override_result_dir,
                    soft_deadline_sec=overtime_deadline_sec,
                )
                if not results:
                    # Defensive — run_grid returns a list of the same
                    # length as the input grid; we only fall here if a
                    # serialization bug ate the result.
                    log.warning("explore: variant %s produced no result", gv.name)
                    continue
                r = results[0]

                # Fix E (Q3c): the per-variant overtime gate fired.
                # Record a dedicated ``KILLED_OVERTIME`` row carrying
                # ``runtime_sec`` + ``wall_clock_ratio_vs_baseline``
                # (no tput, no gain — Q3 explicitly said "don't fake a
                # number"). Skip every downstream gate (KEEP / REVERT /
                # accuracy / stack_rebench) and dedup so a re-proposal
                # of the same fingerprint hits the ``tested`` ledger
                # immediately. The running stack is NOT advanced.
                if getattr(r, "killed_overtime", False):
                    variant_runtime = float(r.runtime_sec or 0.0)
                    wall_clock_ratio = (
                        round(variant_runtime / baseline_runtime_sec, 3)
                        if baseline_runtime_sec > 0 else None
                    )
                    tested_update[fp] = {
                        "fingerprint": fp,
                        "name": gv.name,
                        "extra_sglang_args": gv.extra_sglang_args,
                        "extra_envs": dict(gv.extra_envs),
                        "note": gv.note,
                        "outcome": "KILLED_OVERTIME",
                        "status": r.status,
                        "tput": None,
                        "gain_pct": None,
                        "base_tput": running_base_tput,
                        "round_id": round_id,
                        "ts": _now_iso(),
                        "provenance": provenance,
                        "workload_signature": ws_sig,
                        "framework": framework,
                        "workspace": r.workspace,
                        "runtime_sec": round(variant_runtime, 2),
                        "wall_clock_ratio_vs_baseline": wall_clock_ratio,
                        "baseline_runtime_sec": round(
                            baseline_runtime_sec, 2,
                        ),
                        "overtime_kill_ratio": overtime_kill_ratio,
                    }
                    if gv.name:
                        name_index[gv.name] = fp
                    rejected_update.append({
                        "fingerprint": fp,
                        "name": gv.name,
                        "extra_sglang_args": gv.extra_sglang_args,
                        "extra_envs": dict(gv.extra_envs),
                        "note": gv.note,
                        "reason": "killed_overtime",
                        "gain_pct": None,
                        "tput": None,
                        "runtime_sec": round(variant_runtime, 2),
                        "wall_clock_ratio_vs_baseline": wall_clock_ratio,
                        "round_id": round_id,
                        "ts": _now_iso(),
                        "provenance": provenance,
                    })
                    losers.append({
                        "fingerprint": fp,
                        "name": gv.name,
                        "extra_sglang_args": gv.extra_sglang_args,
                        "extra_envs": dict(gv.extra_envs),
                        "provenance": provenance,
                        "gain_pct": None,
                        "tput": None,
                        "reason": "killed_overtime",
                        "workspace": r.workspace,
                        "runtime_sec": round(variant_runtime, 2),
                        "wall_clock_ratio_vs_baseline": wall_clock_ratio,
                    })
                    log.warning(
                        "explore: variant %s KILLED_OVERTIME "
                        "(runtime=%.1fs vs baseline=%.1fs, ratio=%.2fx, "
                        "kill_ratio=%.2fx); skipping KEEP/REVERT ladder.",
                        gv.name, variant_runtime, baseline_runtime_sec,
                        wall_clock_ratio if wall_clock_ratio is not None else -1.0,
                        overtime_kill_ratio,
                    )
                    continue

                # Carry the variant's content-only fingerprint forward
                # (run_grid recomputes from args/envs which match exactly).
                gain = _gain_pct(r.output_throughput, running_base_tput)
                outcome = "FAILED"
                reason: str = ""
                if r.status != "succeeded" or gain is None:
                    outcome = "FAILED"
                    reason = (r.error or "")[-256:] or "no_measurement"
                elif gain < keep_threshold_pct:
                    outcome = "REVERT"
                    reason = "gain_below_threshold"
                else:
                    # 2. Accuracy gate (only for high-risk variants).
                    accuracy_ok = True
                    accuracy_value: float | None = None
                    if (
                        baseline_accuracy > 0
                        and is_high_accuracy_risk(
                            extra_args=gv.extra_sglang_args,
                            extra_envs=gv.extra_envs,
                        )
                    ):
                        eval_out = parse_eval_results(slot)
                        accuracy_value = eval_out.get("accuracy")
                        if isinstance(accuracy_value, (int, float)):
                            accuracy_ok = accuracy_passed(
                                baseline_accuracy, float(accuracy_value),
                            )
                        else:
                            # No eval result emitted; KB_design §3.4 §7
                            # is silent on this exact case but we follow
                            # the legacy BackendsExecutor convention of
                            # "no accuracy data => skip the gate" so
                            # high-risk flags without an eval don't get
                            # auto-rejected on a benign measurement gap.
                            accuracy_ok = True
                    if not accuracy_ok:
                        outcome = "REVERT"
                        reason = "accuracy_drop"
                    else:
                        outcome = "KEEP"

                tested_update[fp] = {
                    "fingerprint": fp,
                    "name": gv.name,
                    "extra_sglang_args": gv.extra_sglang_args,
                    "extra_envs": dict(gv.extra_envs),
                    "note": gv.note,
                    "outcome": outcome,
                    "status": r.status,
                    "tput": r.output_throughput,
                    "gain_pct": gain,
                    "base_tput": running_base_tput,
                    "round_id": round_id,
                    "ts": _now_iso(),
                    "provenance": provenance,
                    "workload_signature": ws_sig,
                    "framework": framework,
                    "workspace": r.workspace,
                }
                if gv.name:
                    name_index[gv.name] = fp

                # ---- KEEP path (with inlined stack rebench) ---------------
                if outcome == "KEEP":
                    keep_entry = {
                        "fingerprint": fp,
                        "name": gv.name,
                        "extra_sglang_args": gv.extra_sglang_args,
                        "extra_envs": dict(gv.extra_envs),
                        "note": gv.note,
                        "provenance": provenance,
                        "gain_pct": gain,
                        "tput": r.output_throughput,
                        "single_workspace": r.workspace,
                        "round_id": round_id,
                        "accepted_at_round": round_id,
                        "ts": _now_iso(),
                    }
                    # Layer onto the running stack BEFORE rebench.
                    next_args = _join_args(stack_extra_args, gv.extra_sglang_args)
                    next_envs = dict(stack_extra_envs)
                    next_envs.update(gv.extra_envs)
                    in_batch_keeps.append(keep_entry)

                    stack_rebench_tput: float | None = None
                    stack_rebench_workspace: str | None = None
                    stack_rebench_warnings: list[str] = []

                    if enable_stack_rebench and base_tput > 0:
                        rebench_slot = slot / "stack_rebench"
                        rebench_slot.mkdir(parents=True, exist_ok=True)
                        rebench_variant = GridVariant(
                            name=f"{gv.name}__stack_rebench",
                            extra_sglang_args="",  # all args are in next_args
                            extra_envs={},          # all envs are in next_envs
                            note="stack_rebench",
                        )
                        rebench_results = await run_grid(
                            base_yaml_path=config_path,
                            base_extra_args=next_args,
                            grid=[rebench_variant],
                            output_root=rebench_slot,
                            variant_timeout_sec=timeout_sec,
                            model_path=resolved_model,
                            gpu_type=resolved_gpu,
                            benchmark_script=override_script,
                            result_dir=override_result_dir,
                        )
                        rb = rebench_results[0] if rebench_results else None
                        # ALWAYS apply the in-batch env stack to the
                        # rebench step (run_grid doesn't see envs via
                        # base_extra_envs — we have to materialise via
                        # the variant.extra_envs path).  Patch the
                        # rebench variant with the merged envs and
                        # re-run if needed.
                        if rb is not None and next_envs:
                            # If the merged envs aren't already on the
                            # rebench result, re-run with them. This is
                            # cheap-ish (an extra dict update on the YAML)
                            # but only when envs are non-empty.
                            if rb.extra_envs != next_envs:
                                rebench_variant_envs = GridVariant(
                                    name=f"{gv.name}__stack_rebench_envs",
                                    extra_sglang_args="",
                                    extra_envs=next_envs,
                                    note="stack_rebench_envs",
                                )
                                rebench_envs_slot = slot / "stack_rebench_envs"
                                rebench_envs_slot.mkdir(parents=True, exist_ok=True)
                                rb2 = await run_grid(
                                    base_yaml_path=config_path,
                                    base_extra_args=next_args,
                                    grid=[rebench_variant_envs],
                                    output_root=rebench_envs_slot,
                                    variant_timeout_sec=timeout_sec,
                                    model_path=resolved_model,
                                    gpu_type=resolved_gpu,
                                    benchmark_script=override_script,
                                    result_dir=override_result_dir,
                                )
                                if rb2:
                                    rb = rb2[0]

                        if rb is not None and rb.status == "succeeded":
                            stack_rebench_tput = rb.output_throughput
                            stack_rebench_workspace = rb.workspace
                            stack_rebench_warnings = list(rb.nonfatal_warnings)
                        elif rb is not None:
                            stack_rebench_warnings.append(
                                f"stack_rebench_failed:{(rb.error or '')[-120:]}"
                            )
                        else:
                            stack_rebench_warnings.append("stack_rebench_no_result")

                        stable_floor = base_tput * (
                            1.0 + stack_stable_threshold_pct / 100.0
                        )
                        # KEEP_UNSTABLE: rebench didn't beat the cumulative
                        # stability floor — evict the KEEP and treat as REVERT.
                        if (
                            stack_rebench_tput is None
                            or stack_rebench_tput < stable_floor
                        ):
                            log.warning(
                                "explore: variant %s KEEP -> KEEP_UNSTABLE "
                                "(stack_rebench_tput=%s vs stable_floor=%.2f "
                                "with base_tput=%.2f * (1+%.2f%%))",
                                gv.name, stack_rebench_tput, stable_floor,
                                base_tput, stack_stable_threshold_pct,
                            )
                            tested_update[fp]["outcome"] = "KEEP_UNSTABLE"
                            tested_update[fp]["stack_rebench_tput"] = stack_rebench_tput
                            tested_update[fp]["stack_rebench_workspace"] = stack_rebench_workspace
                            tested_update[fp]["stack_rebench_warnings"] = stack_rebench_warnings
                            keep_unstable.append({
                                **keep_entry,
                                "stack_rebench_tput": stack_rebench_tput,
                                "stack_rebench_workspace": stack_rebench_workspace,
                                "stack_rebench_warnings": stack_rebench_warnings,
                            })
                            rejected_update.append({
                                "fingerprint": fp,
                                "name": gv.name,
                                "extra_sglang_args": gv.extra_sglang_args,
                                "extra_envs": dict(gv.extra_envs),
                                "note": gv.note,
                                "reason": "stack_unstable",
                                "gain_pct": gain,
                                "tput": r.output_throughput,
                                "stack_rebench_tput": stack_rebench_tput,
                                "round_id": round_id,
                                "ts": _now_iso(),
                                "provenance": provenance,
                            })
                            # Pop the just-added entry — the stack
                            # rolls back to the prior accumulation.
                            in_batch_keeps.pop()
                            continue
                        else:
                            # Stable — fold the variant onto the
                            # cumulative stack so the NEXT variant is
                            # benched against the new baseline.
                            stack_extra_args = next_args
                            stack_extra_envs = next_envs
                            running_base_tput = stack_rebench_tput
                            last_run_tput = stack_rebench_tput
                            keep_entry["stack_rebench_tput"] = stack_rebench_tput
                            keep_entry["stack_rebench_workspace"] = stack_rebench_workspace
                            keep_entry["stack_rebench_warnings"] = stack_rebench_warnings
                            tested_update[fp]["stack_rebench_tput"] = stack_rebench_tput
                            tested_update[fp]["stack_rebench_workspace"] = stack_rebench_workspace
                            tested_update[fp]["stack_rebench_warnings"] = stack_rebench_warnings
                    else:
                        # Stack rebench disabled — KEEP based on the
                        # single-variant measurement and update the
                        # running baseline naively.
                        stack_extra_args = next_args
                        stack_extra_envs = next_envs
                        running_base_tput = r.output_throughput or running_base_tput
                        last_run_tput = r.output_throughput

                    winners.append(keep_entry)
                    winners_history_update.append({
                        "round_id": round_id,
                        "variant_name": gv.name,
                        "fingerprint": fp,
                        "gain_pct": gain,
                        "extra_args": gv.extra_sglang_args,
                        "extra_envs": dict(gv.extra_envs),
                        "provenance": provenance,
                        "ts": _now_iso(),
                    })
                    continue

                # ---- REVERT / FAILED ----
                rejected_update.append({
                    "fingerprint": fp,
                    "name": gv.name,
                    "extra_sglang_args": gv.extra_sglang_args,
                    "extra_envs": dict(gv.extra_envs),
                    "note": gv.note,
                    "reason": reason or "not_keep",
                    "gain_pct": gain,
                    "tput": r.output_throughput,
                    "round_id": round_id,
                    "ts": _now_iso(),
                    "provenance": provenance,
                })
                losers.append({
                    "fingerprint": fp,
                    "name": gv.name,
                    "extra_sglang_args": gv.extra_sglang_args,
                    "extra_envs": dict(gv.extra_envs),
                    "provenance": provenance,
                    "gain_pct": gain,
                    "tput": r.output_throughput,
                    "reason": reason or "not_keep",
                    "workspace": r.workspace,
                })
                if r.output_throughput:
                    last_run_tput = r.output_throughput

        # ----- Ledger compaction (dedup + accepted preservation) -----------
        accepted_fps_now = {_entry_fp(v) for v in (search.get("accepted") or [])}
        accepted_fps_now.discard("")
        rejected_dedup: dict[str, dict[str, Any]] = {}
        for entry in rejected_update:
            fp = str(entry.get("fingerprint") or "")
            if not fp or fp in accepted_fps_now:
                continue
            rejected_dedup[fp] = entry

        # KB_gaps/Gap-08 / flat
        # per-variant outcomes for the Coordinator's per-variant T3
        # hook. Built from ``tested_update`` (this round only) plus
        # ``skipped_dup`` so KEEP / REVERT / FAILED / KEEP_UNSTABLE /
        # SKIPPED_DEDUP are all exposed. The Coordinator iterates
        # this list and calls ``cortex_kb.verify`` once per variant.
        # Stays JSON-friendly so v0.6 readers stay happy.
        reasons_by_fp: dict[str, str] = {
            str(r.get("fingerprint") or ""): str(r.get("reason") or "")
            for r in rejected_update
            if r.get("round_id") == round_id
        }
        per_variant_outcomes: list[dict[str, Any]] = []
        for fp_key, te in tested_update.items():
            if te.get("round_id") != round_id:
                continue
            outcome = str(te.get("outcome") or "")
            if outcome not in (
                "KEEP", "REVERT", "FAILED", "KEEP_UNSTABLE",
                "KILLED_OVERTIME",
            ):
                continue
            metrics: dict[str, Any] = {}
            if te.get("tput") is not None:
                metrics["tput"] = te.get("tput")
            if te.get("gain_pct") is not None:
                metrics["gain_pct"] = te.get("gain_pct")
            if te.get("stack_rebench_tput") is not None:
                metrics["stack_rebench_tput"] = te.get("stack_rebench_tput")
            # Fix E: surface wall-clock + kill ratio so the orchestration
            # LLM (and downstream KB writers) see "ran too slow → early
            # kill" instead of an opaque FAILED row with no signal.
            if te.get("runtime_sec") is not None:
                metrics["runtime_sec"] = te.get("runtime_sec")
            if te.get("wall_clock_ratio_vs_baseline") is not None:
                metrics["wall_clock_ratio_vs_baseline"] = te.get(
                    "wall_clock_ratio_vs_baseline",
                )
            per_variant_outcomes.append({
                "variant_name": str(te.get("name") or ""),
                "outcome":      outcome,
                "fingerprint":  fp_key,
                "provenance":   str(te.get("provenance") or ""),
                "metrics":      metrics,
                "reason":       reasons_by_fp.get(fp_key, ""),
            })
        for sd in skipped_dup:
            per_variant_outcomes.append({
                "variant_name": str(sd.get("name") or ""),
                "outcome":      "SKIPPED_DEDUP",
                "fingerprint":  str(sd.get("fingerprint") or ""),
                "provenance":   "",
                "metrics":      {},
                "reason":       str(sd.get("reason") or ""),
            })

        # ``last_round`` summary for the prompt / breakdown.
        killed_overtime_fps = [
            str(te.get("fingerprint") or "")
            for te in tested_update.values()
            if te.get("round_id") == round_id
            and te.get("outcome") == "KILLED_OVERTIME"
        ]
        last_round_summary = {
            "round_id": round_id,
            "base_tput": base_tput,
            "base_extra_args": base_extra_args,
            "tested": [w["fingerprint"] for w in winners] + [
                lr["fingerprint"] for lr in losers
            ],
            "round_winners": [w["fingerprint"] for w in winners],
            "keep_unstable": [k["fingerprint"] for k in keep_unstable],
            "killed_overtime": killed_overtime_fps,
            "skipped_dup": skipped_dup,
            "ts": _now_iso(),
        }

        search_update = {
            "schema_version": 1,
            "tested": tested_update,
            "rejected": list(rejected_dedup.values()),
            "name_index": name_index,
            "cursor": len(tested_update),
            "winners_history": winners_history_update,
            "synergy_attempted": list(search.get("synergy_attempted") or []),
            "discovered_flags": list(search.get("discovered_flags") or []),
            "domains_round_summary": list(search.get("domains_round_summary") or []),
            "last_round": last_round_summary,
        }

        # ----- Best variant + status ---------------------------------------
        best_winner = max(
            winners,
            key=lambda w: float(w.get("gain_pct") or 0.0),
            default=None,
        )
        best_gain_pct = float(best_winner.get("gain_pct") or 0.0) if best_winner else 0.0

        if winners:
            output_throughput = (
                float(running_base_tput)
                if last_run_tput is not None
                else None
            )
        else:
            output_throughput = None

        # Successful in the M3 sense = at least one bench actually
        # produced a measurement (KEEP, REVERT-with-gain, KEEP_UNSTABLE)
        # OR was deliberately reaped by the Fix-E overtime gate
        # (KILLED_OVERTIME) — the latter is a real, useful signal the
        # LLM needs to see in ``per_variant_outcomes`` to avoid
        # re-proposing the same heavy variant on the next round.
        produced_measurement = any(
            t.get("outcome") in (
                "KEEP", "REVERT", "KEEP_UNSTABLE", "KILLED_OVERTIME",
            )
            for t in tested_update.values()
            if t.get("round_id") == round_id
        )
        status = "succeeded" if produced_measurement or winners else "failed"

        return {
            "status": status,
            "base_tput": base_tput,
            "running_base_tput": running_base_tput,
            "output_throughput": output_throughput,
            "best_variant": best_winner,
            "best_gain_pct": best_gain_pct,
            "winners": winners,
            "losers": losers,
            "keep_unstable_in_stack": keep_unstable,
            "skipped_dup": skipped_dup,
            # flat per-variant outcomes for T3.
            "per_variant_outcomes": per_variant_outcomes,
            "explore_search_update": search_update,
            "discovered_flags_update": None,
            "round_id": round_id,
            "workspace": output_root.as_posix(),
            "framework": framework,
            # gain_pct for the audit trail (best gain of the batch).
            "gain_pct": best_gain_pct,
            "explore_grid_exhausted": not runnable,
        }


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


explore_executor = ExploreExecutor()


__all__ = [
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_STACK_STABLE_PCT",
    "ExploreExecutor",
    "explore_executor",
]
