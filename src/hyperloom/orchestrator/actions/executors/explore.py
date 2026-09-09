# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env import is_truthy
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import (
    passes_intvty_gate,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    total_tput_of,
    total_tput_serving_grading_enabled,
)
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...state.failure_evidence import (
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    make_failure_id,
    tail_excerpt,
)
from ...state.shared_state import (
    first_positive_tput,
    framework_is_scriptable,
    resolve_anchor_with_drift,
    stack_base_params,
)
from ..stop_attribution import (
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
    stopped_by_the_run_class,
)
from ._accuracy_gate import (
    accuracy_passed,
    parse_eval_results,
)
from . import _framework_switch_manifest as _switch_manifest
from ._canonical_fingerprint import workload_signature
from ._proposal_identity import effective_fingerprint, normalize_proposal
from ._grid_runner import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    _MN_BACKENDS_PRIORITY,
    _MN_PARAMS_PRIORITY,
    GridVariant,
    _kill_stale_servers,
    _num_gpus_for_config,
    _resolve_session_dir,
    apply_aiter_moe_pin_filter,
    apply_compatibility_filter,
    apply_multi_node_invalid_variants,
    apply_user_skip_list,
    reorder_grid_for_multi_node,
    resolve_skip_spec,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from ._grid_server_args import compose_server_args, server_args_env_name
from ._ray_serving import maybe_serving_lease

from ._server_lifecycle import (
    resolve_lifecycle_params,
    teardown_lifecycle_server,
)
from ._workload_envs import (
    FrameworkScriptMismatchError,
    agentx_enabled,
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


_now_iso = functools.partial(now_iso, "auto")


def _initial_explore_search_state() -> dict[str, Any]:
    """Empty :attr:`SharedState.explore_search` ledger."""
    return {
        "schema_version": 1,
        "tested": {},
        "accepted": [],
        "rejected": [],
        "winners_history": [],
        "domains_round_summary": [],
        "name_index": {},
        "cursor": 0,
        "last_round": {},
    }


# Audit/provenance metadata stashed on a GridVariant that must survive being rebuilt into a derived variant.
_CARRIED_VARIANT_ATTRS: tuple[str, ...] = (
    "provenance",
    "scope",
    "overlay_pythonpath",
    "accepted_kernels",
    "kb_evidence",
    "pr_evidence",
    "source_evidence",
)


def _explore_eval_disabled(shared_state: Any, params: dict[str, Any]) -> bool:
    """Whether Magpie lm_eval is opted out for this explore run.

    ``--no-eval`` persists on ``SharedState.eval_disabled``. Task param
    ``disable_run_eval`` is the same opt-out for internally queued explores.
    """
    if is_truthy(params.get("disable_run_eval")):
        return True
    return bool(getattr(shared_state, "eval_disabled", False))


def _carry_variant_metadata(src: Any, dst: Any) -> Any:
    """Copy the carried audit metadata from ``src`` onto ``dst``."""
    for attr in _CARRIED_VARIANT_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst


def _variant_control_fields(variant: Any) -> dict[str, Any]:
    """Return non-default remove/unset/replace controls for identity and ledger rows."""
    remove_args = to_str_list(getattr(variant, "remove_args", []))
    unset_envs = to_str_list(getattr(variant, "unset_envs", []))
    args_mode = str(getattr(variant, "args_mode", "append") or "append").strip().lower()
    out: dict[str, Any] = {}
    if remove_args:
        out["remove_args"] = remove_args
    if unset_envs:
        out["unset_envs"] = unset_envs
    if args_mode == "replace":
        out["args_mode"] = "replace"
    return out


def _grid_variants_from_payload(payload: list[Any]) -> list[GridVariant]:
    """Convert the LLM/specialist grid payload into GridVariant objects."""
    out: list[GridVariant] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        fields = normalize_proposal(raw)
        gv = GridVariant(
            name=fields["name"],
            extra_server_args=fields["extra_args"],
            extra_envs=fields["extra_envs"],
            note=str(raw.get("note") or raw.get("provenance") or ""),
            remove_args=fields["remove_args"],
            unset_envs=fields["unset_envs"],
            args_mode=fields["args_mode"],
        )
        # Stash extra metadata on the GridVariant so the ledger writer can pull provenance/evidence.
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.scope = str(raw.get("scope") or "")  # type: ignore[attr-defined]
        # Authored-kernel overlay dir (PYTHONPATH prefix); "" for env/flag variants.
        gv.overlay_pythonpath = str(raw.get("overlay_pythonpath") or "")  # type: ignore[attr-defined]
        # Authored kernels this variant's overlay installs.
        gv.accepted_kernels = [  # type: ignore[attr-defined]
            str(k).strip() for k in (raw.get("accepted_kernels") or []) if str(k).strip()
        ]
        gv.kb_evidence = list(raw.get("kb_evidence") or [])  # type: ignore[attr-defined]
        gv.pr_evidence = list(raw.get("pr_evidence") or [])  # type: ignore[attr-defined]
        gv.source_evidence = list(raw.get("source_evidence") or [])  # type: ignore[attr-defined]
        # Framework-rewrite lever this variant attributes to, and how.
        gv.framework_lever = str(raw.get("framework_lever") or "")  # type: ignore[attr-defined]
        gv.framework_lever_source = str(raw.get("framework_lever_source") or "")  # type: ignore[attr-defined]
        out.append(gv)
    return out


def framework_lever_grid(shared_state: Any) -> list[dict[str, Any]]:
    """Build explore variants that attribute each registered rewrite lever."""
    if shared_state is None:
        return []
    rows = list(getattr(shared_state, "authored_framework_levers", None) or [])
    if not rows:
        return []
    pending = [row for row in rows if isinstance(row, dict) and row.get("attributed_gain_pct") is None]
    if not pending:
        return []
    dormant = [row for row in pending if not row.get("default_on")]
    active = [row for row in pending if row.get("default_on")]
    payload: list[dict[str, Any]] = []
    if dormant:
        for variant in _switch_manifest.additive_variants(dormant):
            payload.append({**variant, "framework_lever_source": "additive"})
    if active:
        for variant in _switch_manifest.leave_one_out_variants(active):
            payload.append({**variant, "framework_lever_source": "leave_one_out"})
    return payload


def _framework_lever_attributions(
    per_variant_outcomes: list[dict[str, Any]],
    lever_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive each rewrite lever's own contribution from this round's outcomes."""
    if not lever_payload:
        return []
    by_name = {
        str(v.get("name") or ""): v
        for v in lever_payload
        if isinstance(v, dict) and str(v.get("framework_lever") or "")
    }
    if not by_name:
        return []
    out: list[dict[str, Any]] = []
    for row in per_variant_outcomes:
        seed = by_name.get(str(row.get("variant_name") or ""))
        if seed is None:
            continue
        measured = (row.get("metrics") or {}).get("gain_pct")
        if not isinstance(measured, (int, float)):
            continue
        source = str(seed.get("framework_lever_source") or "")
        gain = -float(measured) if source == "leave_one_out" else float(measured)
        out.append(
            {
                "switch": str(seed["framework_lever"]),
                "gain_pct": round(gain, 4),
                "source": source,
                "variant_name": str(row.get("variant_name") or ""),
                "outcome": str(row.get("outcome") or ""),
            }
        )
    return out


# Curated MTP-capable model class set (needs multi-token-prediction heads).
_ATOM_MTP_CAPABLE_MODEL_CLASSES: frozenset[str] = frozenset(
    {
        "moe_mla",
        "moe_mla_nsa",
    }
)


def _atom_default_grid(
    *,
    model_class: str,
    conc: int,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """Atom default explore grid, seeded from atom's known perf knobs."""
    mc_l = (model_class or "").strip().lower()
    is_moe = "moe" in mc_l
    is_mla = "mla" in mc_l
    is_fp8 = "fp8" in mc_l or mc_l.endswith("_fp8")
    is_mtp_capable = mc_l in _ATOM_MTP_CAPABLE_MODEL_CLASSES

    variants: list[GridVariant] = []

    def _add(name: str, args: str) -> None:
        """Append a ``default_grid``-provenance variant to the grid."""
        gv = GridVariant(
            name=name,
            extra_server_args=args,
            extra_envs={},
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # ``atom_level_3`` is atom's default, so use ``atom_level_2`` as the off-default contrast.
    _add("atom_level_2", "--level 2")
    _add("atom_prefix_cache", "--enable_prefix_caching")

    if is_fp8:
        _add("atom_kv_fp8", "--kv_cache_dtype fp8")

    if is_moe:
        _add("atom_ep", "--enable-expert-parallel")

    if is_mla:
        _add("atom_dp_attn", "--enable-dp-attention")

    if is_mtp_capable:
        _add(
            "atom_mtp_3",
            "--method mtp --num-speculative-tokens 3",
        )
        _add(
            "atom_mtp_1",
            "--method mtp --num-speculative-tokens 1",
        )

    if conc and conc > 0:
        # Bracket the live concurrency so cudagraph captures the actual decode batch sizes the workload spends most of
        # its time at.
        cg_sizes = sorted({1, 2, 4, 8, 16, int(conc)})
        cg_str = "[" + ",".join(str(s) for s in cg_sizes) + "]"
        _add(
            "atom_cudagraph_bracket",
            f"--cudagraph-capture-sizes {cg_str}",
        )

    return variants


def _xdit_default_grid(
    *,
    model_class: str,
    conc: int = 0,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """xDiT (diffusion) default explore grid, seeded from the empirical KB."""
    variants: list[GridVariant] = []

    def _add(name: str, *, envs: dict[str, str]) -> None:
        """Append a ``default_grid``-provenance env-only variant."""
        gv = GridVariant(
            name=name,
            extra_server_args="",
            extra_envs=dict(envs),
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # AMD buffer load/store instructions — directionally correct, BF16-safe.
    _add("xdit_buffer_ops", envs={"AMDGCN_USE_BUFFER_OPS": "1"})
    # torch.compile reduce-overhead: expose an explicit on/off contrast.
    _add("xdit_compile_reduce_overhead", envs={"XDIT_USE_TORCH_COMPILE": "1"})
    _add("xdit_no_compile", envs={"XDIT_USE_TORCH_COMPILE": "0"})
    # Confirm the safe attention backend (aiter).
    _add("xdit_attn_aiter", envs={"XDIT_ATTENTION_BACKEND": "aiter"})
    return variants


_CONFIG_REPLAY_PROVENANCE = frozenset({"geak_revalidate"})


def _is_config_replay_variant(variant: Any) -> bool:
    """Whether a variant replays an already-validated config verbatim."""
    return str(getattr(variant, "provenance", "") or "").strip() in _CONFIG_REPLAY_PROVENANCE


def filter_operator_pinned_envs(
    grid: list[GridVariant],
    baseline_envs: dict[str, Any] | None,
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that overwrite an env the operator pinned in the baseline."""
    pinned = {str(k).strip().upper() for k in (baseline_envs or {}) if str(k).strip()}
    if not pinned:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    for gv in grid:
        clash = sorted(
            key for key in (str(k).strip().upper() for k in (getattr(gv, "extra_envs", None) or {})) if key in pinned
        )
        # A replay reproduces a config another component already measured, so it carries the pinned values verbatim by
        # construction.
        if clash and not _is_config_replay_variant(gv):
            dropped.append(
                (
                    str(getattr(gv, "name", "?")),
                    f"overwrites baseline-pinned env {', '.join(clash)}, which would make "
                    "its number incomparable to the baseline",
                )
            )
            continue
        kept.append(gv)
    return kept, dropped


def _default_grid_for_framework(
    framework: str,
    *,
    model_class: str,
    conc: int = 0,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """Framework-keyed default grid dispatch."""
    fw = (framework or "").strip().lower()
    if fw == "atom":
        return _atom_default_grid(
            model_class=model_class,
            conc=conc,
            isl=isl,
            osl=osl,
        )
    if fw == "xdit":
        return _xdit_default_grid(
            model_class=model_class,
            conc=conc,
            isl=isl,
            osl=osl,
        )
    return []


# Auto-derived per-variant hard timeout: derive the cap from the Coordinator-injected measured baseline runtime plus a
# safety margin above the soft-kill ratio (preserves soft-kill → hard-cap layering).
DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC = 2400  # 40 min
DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC = 14400  # 4 h — roofline composite budget
DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN = 0.5  # hard cap ≥ baseline × (kill_ratio + 0.5)
# AgentX ceiling.
AGENTX_EXPLORE_TIMEOUT_CEILING_SEC = 28800  # 8 h


def _compute_explore_variant_timeout(
    baseline_runtime_sec: float,
    kill_ratio: float,
    *,
    floor_sec: int = DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC,
    ceiling_sec: int = DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
    safety_margin: float = DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN,
) -> int:
    """Derive the per-variant hard timeout from the measured baseline."""
    if baseline_runtime_sec <= 0:
        return int(floor_sec)
    effective_kill_ratio = max(1.0, float(kill_ratio))
    derived = float(baseline_runtime_sec) * (effective_kill_ratio + float(safety_margin))
    return int(max(floor_sec, min(ceiling_sec, derived)))


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action."""

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        variant_timeout_sec: int = 2400,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the explore executor and its gating thresholds."""
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the merged ``explore`` action for one task."""
        try:
            return await self._run_explore(ctx)
        finally:
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                try:
                    await asyncio.to_thread(_kill_stale_servers)
                except Exception:  # noqa: BLE001 - best-effort safety net
                    log.warning(
                        "explore: post-run _kill_stale_servers failed",
                        exc_info=True,
                    )

    async def _run_explore(self, ctx) -> dict[str, Any]:
        """Run body for :meth:`__call__`; see its docstring for the wrapper."""
        params = dict(ctx.task.params or {})
        # ----- Config / output workspace -----------------------------------
        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not config_path.exists():
            return {
                "status": "failed",
                "error_class": "missing_config",
                "error": f"config not found: {config_path}",
            }
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or extra.get("state")
        eval_disabled = _explore_eval_disabled(shared_state, params)
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "explore", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # ----- Workload-contract materialization --------------------------- Re-materialize so variant YAMLs honour
        # the operator's actual workload (CONC / ISL / OSL / TP / MAX_MODEL_LEN / PRECISION).
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=str(getattr(shared_state, "model_path", "") or "") if shared_state else "",
            for_serving=True,
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
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
        try:
            config_path = materialize_config_with_envs(
                config_path,
                output_root,
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=override_script,
                extra_envs={"RUN_EVAL": "false"} if eval_disabled else None,
                out_name="explore_base.with_envs.yaml",
            )
        except FrameworkScriptMismatchError as exc:
            return {
                "status": "failed",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
            }

        # ----- Inputs ------------------------------------------------------ Params snapshot the anchor and the stack
        # it was measured on together; a KEEP landing while this task queued invalidates both, so refresh them as a
        # pair.
        ss = extra.get("shared_state") or extra.get("state")
        snapshot_tput = float(params.get("base_tput") or 0.0)
        # Revalidation reproduces the saved stack, so it never re-anchors.
        anchor, anchor_drifted = (
            (snapshot_tput, False)
            if params.get("source") == "resume_stack_revalidate"
            else resolve_anchor_with_drift(snapshot_tput, ss)
        )
        if anchor > snapshot_tput:
            if anchor_drifted:
                log.warning("explore: anchor drift %.1f -> %.1f; re-reading base args", snapshot_tput, anchor)
            params["base_tput"] = anchor
            cb = getattr(ss, "current_best", None)
            # Only current_best carries args; a baseline_tput anchor leaves the params stack (seeded from the baseline
            # record) authoritative.
            if first_positive_tput(cb) > 0:
                params.update(stack_base_params(cb))
        base_extra_args = str(params.get("base_extra_args") or "").strip()
        base_extra_envs = dict(params.get("base_extra_envs") or {})
        base_remove_args = to_str_list(params.get("base_remove_args"))
        base_unset_envs = to_str_list(params.get("base_unset_envs"))
        base_args_mode = str(params.get("base_args_mode") or "append").strip().lower()
        base_tput = float(params.get("base_tput") or 0.0)
        # The measured baseline outranks a proposed one.
        baseline_accuracy = float(getattr(ss, "baseline_accuracy", 0.0) or 0.0) if ss is not None else 0.0
        if baseline_accuracy <= 0:
            baseline_accuracy = float(params.get("accuracy_baseline") or 0.0) or float(
                params.get("baseline_accuracy") or 0.0
            )
        keep_threshold_pct = float(
            params.get(
                "keep_threshold_pct",
                self.keep_threshold_pct,
            )
        )

        # per-variant overtime kill — anchored on baseline wall-clock.
        baseline_runtime_sec_raw = params.get("baseline_runtime_sec")
        try:
            baseline_runtime_sec = float(baseline_runtime_sec_raw) if baseline_runtime_sec_raw is not None else 0.0
        except (TypeError, ValueError):
            baseline_runtime_sec = 0.0
        overtime_kill_ratio_raw = params.get("explore_overtime_kill_ratio")
        try:
            overtime_kill_ratio = float(overtime_kill_ratio_raw) if overtime_kill_ratio_raw is not None else 0.0
        except (TypeError, ValueError):
            overtime_kill_ratio = 0.0
        # WARM measure-round anchor (client-only).
        baseline_warm_runtime_sec_raw = params.get("baseline_warm_runtime_sec")
        try:
            baseline_warm_runtime_sec = (
                float(baseline_warm_runtime_sec_raw) if baseline_warm_runtime_sec_raw is not None else 0.0
            )
        except (TypeError, ValueError):
            baseline_warm_runtime_sec = 0.0
        # Per-variant hard cap precedence: explicit ``params['variant_timeout_sec']`` → auto-derive from baseline
        # runtime + kill ratio (see ``_compute_explore_variant_timeout``) → ``self.variant_timeout_sec`` floor (no
        # baseline yet).
        explicit_timeout = params.get("variant_timeout_sec")
        if explicit_timeout is not None:
            timeout_sec = int(explicit_timeout)
        else:
            # Operator-tunable headroom; negative clamps to 0.
            safety_margin_raw = params.get("variant_timeout_safety_margin")
            try:
                safety_margin = (
                    max(0.0, float(safety_margin_raw))
                    if safety_margin_raw is not None
                    else DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN
                )
            except (TypeError, ValueError):
                safety_margin = DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN
            # The stock 4h ceiling assumes a synthetic round measured in minutes.
            _ceiling = AGENTX_EXPLORE_TIMEOUT_CEILING_SEC if agentx_enabled() else DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC
            timeout_sec = _compute_explore_variant_timeout(
                baseline_runtime_sec=baseline_runtime_sec,
                kill_ratio=overtime_kill_ratio,
                floor_sec=int(self.variant_timeout_sec),
                ceiling_sec=_ceiling,
                safety_margin=safety_margin,
            )

        # Resolve framework from materialized YAML (for the ledger + the atom seed-grid fallback below).
        try:
            with config_path.open(encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f) or {}
            framework = str((_cfg.get("benchmark") or {}).get("framework") or "").lower()
            # Pull CONC so the seed grid's cudagraph-bracket variant brackets it.
            _yaml_envs = (_cfg.get("benchmark") or {}).get("envs") or {}
            _base_inherited_args = str(_yaml_envs.get(server_args_env_name(framework)) or "").strip()
        except (OSError, yaml.YAMLError) as exc:
            log.warning("explore: could not resolve framework from %s: %s", config_path, exc)
            framework = ""
            _yaml_envs = {}
            _base_inherited_args = ""
        _effective_inherited_args = "" if base_args_mode == "replace" else _base_inherited_args

        # ----- Variant grid ------------------------------------------------
        grid_payload = params.get("grid") or []
        if not isinstance(grid_payload, list):
            grid_payload = []
        # Framework-rewrite levers first.
        lever_payload = framework_lever_grid(extra.get("shared_state") or extra.get("state"))
        if lever_payload:
            existing_names = {str(v.get("name") or "") for v in grid_payload if isinstance(v, dict)}
            fresh = [v for v in lever_payload if str(v.get("name") or "") not in existing_names]
            if fresh:
                log.info(
                    "explore: seeding %d framework-rewrite lever variant(s) for attribution",
                    len(fresh),
                )
                grid_payload = fresh + list(grid_payload)
        if not grid_payload:
            # No LLM variants: fall through to the framework's programmatic seed grid instead of failing the task.
            seed_model_class = str(params.get("model_class") or "").strip() or os.environ.get("MODEL_CLASS", "").strip()
            seed_conc = 0
            try:
                _conc_raw = params.get("conc") or _yaml_envs.get("CONC") or os.environ.get("CONC") or 0
                seed_conc = int(_conc_raw)
            except (TypeError, ValueError):
                seed_conc = 0
            seed_isl = 0
            seed_osl = 0
            try:
                seed_isl = int(params.get("isl") or _yaml_envs.get("ISL") or os.environ.get("ISL") or 0)
                seed_osl = int(params.get("osl") or _yaml_envs.get("OSL") or os.environ.get("OSL") or 0)
            except (TypeError, ValueError):
                # Non-integer seed hint; fall back to the default grid below.
                pass
            seed = _default_grid_for_framework(
                framework,
                model_class=seed_model_class,
                conc=seed_conc,
                isl=seed_isl,
                osl=seed_osl,
            )
            if seed:
                log.info(
                    "explore: empty grid for framework=%s; falling through "
                    "to %d default_grid seed variants "
                    "(model_class=%r conc=%d)",
                    framework or "?",
                    len(seed),
                    seed_model_class or "?",
                    seed_conc,
                )
                grid_payload = [
                    {
                        "name": v.name,
                        "extra_server_args": v.extra_server_args,
                        "extra_envs": dict(v.extra_envs or {}),
                        "note": v.note or "default_grid",
                        "provenance": getattr(v, "provenance", "default_grid"),
                    }
                    for v in seed
                ]
        if not isinstance(grid_payload, list) or not grid_payload:
            return {
                "status": "failed",
                "error_class": "empty_grid",
                "error": (
                    "explore: params.grid must be a non-empty list of variant "
                    "dicts. The Orchestration prompt "
                    "should fill this from specialist proposals / default_grid."
                ),
                "workspace": output_root.as_posix(),
            }
        grid = _grid_variants_from_payload(grid_payload)

        if framework == "custom":
            grid, _mc_dropped = filter_operator_pinned_envs(grid, _yaml_envs)
            for _nm, _reason in _mc_dropped:
                log.warning("explore: dropping variant %s (%s)", _nm, _reason)

        if not grid:
            return {
                "status": "failed",
                "error_class": "empty_grid",
                "error": "explore: params.grid contains no valid variants",
                "workspace": output_root.as_posix(),
            }

        # ----- explore_search ledger (history seed) -------------------------
        search = dict(params.get("explore_search") or _initial_explore_search_state())
        # Defensive default fill (resume / first-run guards).
        for key, default in (
            ("schema_version", 1),
            ("tested", {}),
            ("rejected", []),
            ("name_index", {}),
            ("cursor", 0),
            ("winners_history", []),
            ("domains_round_summary", []),
        ):
            search.setdefault(key, default)

        tested_dict = search.get("tested") or {}
        inherited_name_index: dict[str, Any] = dict(search.get("name_index") or {})

        # Attach the per-variant fingerprint as an attribute so the result loop needn't recompute.
        ws_sig = workload_signature()

        unique_in_round: dict[str, GridVariant] = {}
        skipped_dup: list[dict[str, Any]] = []
        for gv in grid:
            fp = effective_fingerprint(
                gv.extra_server_args,
                gv.extra_envs,
                controls=_variant_control_fields(gv),
                base_remove_args=base_remove_args,
                base_unset_envs=base_unset_envs,
                base_args_mode=base_args_mode,
            )
            gv.canonical_fp = fp  # type: ignore[attr-defined]
            if fp in unique_in_round:
                # In-round duplicate — keep the first occurrence.
                skipped_dup.append(
                    {
                        "name": gv.name,
                        "fingerprint": fp,
                        "reason": "round_dup",
                    }
                )
                continue
            unique_in_round[fp] = gv

        runnable: list[GridVariant] = list(unique_in_round.values())

        # Re-proposals are still benchmarked; the tested ledger already carries
        # each prior outcome and is rendered in full, so this only counts them.
        re_proposed = sum(1 for fp in unique_in_round if isinstance(tested_dict.get(fp), dict))

        log.info(
            "explore dedup: payload=%d → runnable=%d (round_dup=%d re_proposed=%d)",
            len(grid),
            len(runnable),
            len(skipped_dup),
            re_proposed,
        )

        # Multi-node grid shaping.
        if runnable:
            runnable, _mn_dropped = apply_multi_node_invalid_variants(runnable)
            # Honour an operator-pinned SGLANG_USE_AITER=0: drop variants that would re-enable the (hang-prone) aiter
            # MoE runner.
            runnable, _aiter_dropped = apply_aiter_moe_pin_filter(runnable)
            # xDiT do-not-set list, plus flags the model class or the installed server does not support.
            runnable, _compat_dropped = apply_compatibility_filter(
                runnable,
                framework=framework,
                model_path=resolved_model,
            )
            # Operator-supplied --skip-variants patterns.
            runnable, _skip_dropped = apply_user_skip_list(
                runnable,
                skip_spec=resolve_skip_spec(params),
            )
            for _d in (*_mn_dropped, *_aiter_dropped, *_compat_dropped, *_skip_dropped):
                skipped_dup.append(
                    {
                        "name": _d.get("name", ""),
                        "reason": _d.get("source", "grid_invalid"),
                        "detail": _d.get("reason", ""),
                    }
                )
            runnable = reorder_grid_for_multi_node(
                runnable,
                priority_tags=_MN_PARAMS_PRIORITY + _MN_BACKENDS_PRIORITY,
            )

        round_id_seed = int(search.get("cursor") or 0) + 1
        round_id = f"explore-{round_id_seed:03d}"

        # ----- Per-variant serial run loop ---------------------------------
        winners: list[dict[str, Any]] = []
        losers: list[dict[str, Any]] = []
        # This round's own ledger writes, kept apart from the ledger it inherited and merged over it once the loop is
        # done.
        round_tested: dict[str, dict[str, Any]] = {}
        round_name_index: dict[str, Any] = {}
        rejected_update: list[dict[str, Any]] = list(search.get("rejected") or [])
        winners_history_update: list[dict[str, Any]] = list(search.get("winners_history") or [])

        # ``stack_extra_args`` / ``stack_extra_envs`` carry the running accumulation; after a KEEP they extend with
        # the KEEP'd variant.
        stack_extra_args = base_extra_args
        stack_extra_envs = dict(base_extra_envs)
        stack_remove_args = list(dict.fromkeys(base_remove_args))
        stack_unset_envs = list(dict.fromkeys(base_unset_envs))
        stack_base_args_mode = base_args_mode
        running_base_tput = base_tput

        def _measured_against() -> dict[str, Any]:
            """The stack this variant launched on top of, as it stands now.

            Read at each write rather than captured once: every KEEP advances
            the stack, so a variant later in the round was measured against a
            different one than the round opened with. Carried verbatim for the
            same reason ``base_tput`` is -- a consumer that reconstructs the
            stack from the session's current config gets whichever one is
            current, not the one this variant was judged on.
            """
            return {
                "throughput": running_base_tput,
                "accuracy": baseline_accuracy or None,
                "extra_server_args": stack_extra_args,
                "extra_envs": dict(stack_extra_envs),
                "remove_args": list(stack_remove_args),
                "unset_envs": list(stack_unset_envs),
                "args_mode": stack_base_args_mode,
            }

        grade_on_total = total_tput_serving_grading_enabled(
            scriptable=framework_is_scriptable(framework),
            benchmark_mode=str(getattr(ss, "benchmark_mode", "") or ""),
        )
        _anchor_perf, _anchor_reason = resolve_grading_anchor_perf(ss) if grade_on_total else (None, "")
        if grade_on_total and _anchor_reason:
            log.info(
                "explore: total-throughput grading unavailable (%s); grading this round on output throughput",
                _anchor_reason,
            )
        running_base_perf = _anchor_perf

        # Single-node server_lifecycle eligibility (multi-node / non-builtin script / profiler-on falls back to a cold
        # decision round instead of one that re-attaches to the warmup's server).
        lifecycle = resolve_lifecycle_params(config_path)
        lifecycle_eligible = bool(lifecycle.get("eligible"))
        lifecycle_framework = str(lifecycle.get("framework") or "")
        lifecycle_port = int(lifecycle.get("port") or 0)

        # Warm-decision mode.
        use_warm_decision = lifecycle_eligible and bool(getattr(ss, "baseline_double_run", True))
        # Decision-round overtime anchor: the WARM measure time when warm-decision is active and available, else the
        # cold baseline wall-clock (legacy).
        decision_anchor_sec = (
            baseline_warm_runtime_sec if (use_warm_decision and baseline_warm_runtime_sec > 0) else baseline_runtime_sec
        )
        # The soft deadline is anchored on the warm client-only measure time and enforced from the server-ready
        # marker, so both the measured runtime and this anchor exclude cold boot / warmup.
        if decision_anchor_sec > 0 and overtime_kill_ratio > 0:
            decision_deadline_sec: float | None = decision_anchor_sec * overtime_kill_ratio
        else:
            decision_deadline_sec = None

        # One Ray serving lease (actor) spans the WHOLE round; every variant reuses it.
        round_serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path)) if runnable else None
        # Stop testing further variants once the session wall-clock budget runs out; untested variants stay out of the
        # ledger so a resume can retry them.
        session_deadline_sec, session_expected_sec = session_grid_bounds(
            extra.get("shared_state") or extra.get("state")
        )
        warmup_expected_sec = (baseline_runtime_sec if baseline_runtime_sec > 0 else None) or session_expected_sec
        decision_expected_sec = (decision_anchor_sec if decision_anchor_sec > 0 else None) or session_expected_sec
        # Set when the loop stops because the run stopped it -- the budget ran out, or the orchestrator cancelled the
        # action -- so the round can say so instead of reporting a bare, unattributed failure: a variant that never
        # ran is not a variant that failed.
        run_stop: StoppedByTheRun | None = None
        run_stop_detail = ""
        session_budget_untested = 0

        def _stopped_by_the_run(result: Any, *, variant: GridVariant, idx: int, round_label: str) -> bool:
            """Whether the run stopped this round, and record it if it did."""
            nonlocal run_stop, run_stop_detail, session_budget_untested
            stopped = stopped_by_the_run_class(getattr(result, "error_class", "") if result is not None else "")
            if stopped is None:
                return False
            run_stop = stopped
            run_stop_detail = stopped.interrupted
            session_budget_untested = len(runnable) - idx
            log.warning(
                "explore: the %s round of variant %s was stopped by the run (%s); it and the "
                "%d variant(s) after it stay out of the ledger so a resume can retry them",
                round_label,
                variant.name,
                stopped.error_class,
                session_budget_untested - 1,
            )
            return True

        try:
            for idx, gv in enumerate(runnable):
                # A warm-decision variant pays for both rounds, so admitting it on the decision round alone would let
                # it in and then strand it mid-variant with a discarded warmup and no measurement.
                if decision_expected_sec is not None:
                    fit_required_sec = float(decision_expected_sec) + (
                        float(warmup_expected_sec or 0.0) if use_warm_decision else 0.0
                    )
                else:
                    fit_required_sec = float(timeout_sec)
                if session_deadline_sec is not None and (session_deadline_sec - time.monotonic()) < fit_required_sec:
                    run_stop = STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
                    run_stop_detail = run_stop.never_started
                    session_budget_untested = len(runnable) - idx
                    log.warning(
                        "explore: session budget cannot fit another variant "
                        "(needs %.0fs); stopping after %d/%d variant(s)",
                        fit_required_sec,
                        idx,
                        len(runnable),
                    )
                    break
                fp = getattr(gv, "canonical_fp", "")
                provenance = getattr(gv, "provenance", "llm_direct")
                scope = str(getattr(gv, "scope", "") or "")
                control_fields = _variant_control_fields(gv)
                if stack_base_args_mode == "replace":
                    run_remove_args = to_str_list(getattr(gv, "remove_args", []))
                else:
                    run_remove_args = list(
                        dict.fromkeys(stack_remove_args + to_str_list(getattr(gv, "remove_args", [])))
                    )
                run_unset_envs = list(dict.fromkeys(stack_unset_envs + to_str_list(getattr(gv, "unset_envs", []))))
                run_extra_envs = dict(stack_extra_envs)
                run_extra_envs.update(gv.extra_envs)
                if eval_disabled:
                    run_extra_envs["RUN_EVAL"] = "false"
                run_gv = GridVariant(
                    name=gv.name,
                    extra_server_args=gv.extra_server_args,
                    extra_envs=run_extra_envs,
                    note=gv.note,
                    remove_args=run_remove_args,
                    unset_envs=run_unset_envs,
                    args_mode=str(getattr(gv, "args_mode", "append") or "append"),
                )
                _carry_variant_metadata(gv, run_gv)
                # The decision round is timed against a throughput-only anchor, so it measures throughput only: the
                # warmup round already evaluated accuracy and ``parse_eval_results`` falls back to that score.
                decision_gv = run_gv
                if use_warm_decision:
                    decision_envs = dict(run_extra_envs)
                    decision_envs["RUN_EVAL"] = "false"
                    decision_gv = _carry_variant_metadata(
                        run_gv,
                        GridVariant(
                            name=gv.name,
                            extra_server_args=gv.extra_server_args,
                            extra_envs=decision_envs,
                            note=gv.note,
                            remove_args=run_remove_args,
                            unset_envs=run_unset_envs,
                            args_mode=str(getattr(gv, "args_mode", "append") or "append"),
                        ),
                    )
                slot = output_root / f"v{idx:02d}_{_safe(gv.name)}"
                slot.mkdir(parents=True, exist_ok=True)
                # The warmup and decision rounds share this slot as the lifecycle pid_dir so the decision round
                # re-attaches to the server the warmup left hot.
                variant_lifecycle = (
                    {"cleanup": False, "pid_dir": str(slot), "port": lifecycle_port} if lifecycle_eligible else None
                )
                # Ray-managed GPU execution (§12 T1): reuse the round-level Ray lease (actor) for this variant's
                # warmup and decision rounds; they reuse one persistent server, so no GPU process outlives the lease.
                variant_lease = round_serving_lease
                try:
                    # Warm-decision warmup round.
                    if use_warm_decision:
                        warmup_slot = slot / "warmup_round"
                        warmup_slot.mkdir(parents=True, exist_ok=True)
                        warmup_results = await run_grid(
                            base_yaml_path=config_path,
                            base_extra_args=stack_extra_args,
                            grid=[run_gv],
                            output_root=warmup_slot,
                            variant_timeout_sec=timeout_sec,
                            model_path=resolved_model,
                            gpu_type=resolved_gpu,
                            benchmark_script=override_script,
                            result_dir=override_result_dir,
                            soft_deadline_sec=None,
                            server_lifecycle=variant_lifecycle,
                            base_args_mode=stack_base_args_mode,
                            serving_lease=variant_lease,
                            session_deadline_sec=session_deadline_sec,
                            variant_expected_sec=warmup_expected_sec,
                        )
                        w = warmup_results[0] if warmup_results else None
                        if _stopped_by_the_run(w, variant=gv, idx=idx, round_label="warmup"):
                            break
                        if w is None or getattr(w, "status", "") != "succeeded":
                            werr = (getattr(w, "error", "") or "")[-200:] if w is not None else "no_result"
                            log.warning(
                                "explore: variant %s warmup round failed (%s); skipping decision round.",
                                gv.name,
                                werr,
                            )
                            round_tested[fp] = {
                                "fingerprint": fp,
                                "name": gv.name,
                                "extra_server_args": gv.extra_server_args,
                                "extra_envs": dict(gv.extra_envs),
                                **control_fields,
                                "note": gv.note,
                                "outcome": "FAILED",
                                "status": getattr(w, "status", "failed") if w is not None else "failed",
                                "tput": None,
                                "gain_pct": None,
                                "base_tput": running_base_tput,
                                "round_id": round_id,
                                "ts": _now_iso(),
                                "provenance": provenance,
                                "workload_signature": ws_sig,
                                "framework": framework,
                                "reason": "warmup_failed",
                                "error_class": w.error_class if w is not None else "",
                                "server_log_path": w.server_log_path if w is not None else None,
                                "launch_evidence": dict(w.launch_evidence or {}) if w is not None else {},
                                "launch_evidence_path": w.launch_evidence_path if w is not None else None,
                                "stage": FAILURE_STAGE_WARMUP,
                                "error_excerpt": tail_excerpt(w.error) if w is not None else None,
                                "workspace": w.workspace if w is not None else None,
                                "raw_result_path": w.raw_result_path if w is not None else None,
                                # It launched on a stack even though it never
                                # reached a gate, so no gates are recorded and
                                # nothing validated it.
                                "measured_against": _measured_against(),
                            }
                            if gv.name:
                                round_name_index[gv.name] = fp
                            rejected_update.append(
                                {
                                    "fingerprint": fp,
                                    "name": gv.name,
                                    "extra_server_args": gv.extra_server_args,
                                    "extra_envs": dict(gv.extra_envs),
                                    **control_fields,
                                    "note": gv.note,
                                    "reason": "warmup_failed",
                                    "gain_pct": None,
                                    "tput": None,
                                    "round_id": round_id,
                                    "ts": _now_iso(),
                                    "provenance": provenance,
                                    "stage": FAILURE_STAGE_WARMUP,
                                    "error_excerpt": tail_excerpt(w.error) if w is not None else None,
                                    "workspace": w.workspace if w is not None else None,
                                }
                            )
                            losers.append(
                                {
                                    "fingerprint": fp,
                                    "name": gv.name,
                                    "extra_server_args": gv.extra_server_args,
                                    "extra_envs": dict(gv.extra_envs),
                                    **control_fields,
                                    "provenance": provenance,
                                    "gain_pct": None,
                                    "tput": None,
                                    "reason": "warmup_failed",
                                    "workspace": getattr(w, "workspace", None) if w is not None else None,
                                }
                            )
                            continue
                    # Decision round: warm (re-attaches to the warmup's hot server, client-only) when
                    # ``use_warm_decision``, otherwise a fresh cold boot.
                    results = await run_grid(
                        base_yaml_path=config_path,
                        base_extra_args=stack_extra_args,
                        grid=[decision_gv],
                        output_root=slot,
                        variant_timeout_sec=timeout_sec,
                        model_path=resolved_model,
                        gpu_type=resolved_gpu,
                        benchmark_script=override_script,
                        result_dir=override_result_dir,
                        soft_deadline_sec=decision_deadline_sec,
                        server_lifecycle=variant_lifecycle,
                        base_args_mode=stack_base_args_mode,
                        preclean_before_run=not use_warm_decision,
                        server_already_ready=use_warm_decision,
                        serving_lease=variant_lease,
                        session_deadline_sec=session_deadline_sec,
                        variant_expected_sec=decision_expected_sec,
                    )
                    if not results:
                        # run_grid returns one result per grid entry.
                        log.warning(
                            "explore: variant %s produced no result",
                            gv.name,
                        )
                        continue
                    r = results[0]
                    if _stopped_by_the_run(r, variant=gv, idx=idx, round_label="decision"):
                        break

                    # Overtime gate fired: record a ``KILLED_OVERTIME`` row (no faked tput/gain), skip downstream
                    # gates, leave the stack unadvanced.
                    if getattr(r, "killed_overtime", False):
                        variant_runtime = float(r.runtime_sec or 0.0)
                        wall_clock_ratio = (
                            round(variant_runtime / decision_anchor_sec, 3) if decision_anchor_sec > 0 else None
                        )
                        # Rough output tok/s salvaged from partial server.log.
                        est_tput = getattr(r, "estimated_output_throughput", None)
                        round_tested[fp] = {
                            "fingerprint": fp,
                            "name": gv.name,
                            "extra_server_args": gv.extra_server_args,
                            "extra_envs": dict(gv.extra_envs),
                            **control_fields,
                            "note": gv.note,
                            "outcome": "KILLED_OVERTIME",
                            "status": r.status,
                            "tput": None,
                            "gain_pct": None,
                            "estimated_output_throughput": est_tput,
                            "base_tput": running_base_tput,
                            # Killed before any gate ruled: the stack it ran on
                            # is known, its verdicts are not.
                            "measured_against": _measured_against(),
                            "round_id": round_id,
                            "ts": _now_iso(),
                            "provenance": provenance,
                            "workload_signature": ws_sig,
                            "framework": framework,
                            "workspace": r.workspace,
                            "runtime_sec": round(variant_runtime, 2),
                            "wall_clock_ratio_vs_baseline": wall_clock_ratio,
                            "baseline_runtime_sec": round(
                                baseline_runtime_sec,
                                2,
                            ),
                            "overtime_anchor_sec": round(decision_anchor_sec, 2),
                            "overtime_anchor_kind": (
                                "warm"
                                if decision_anchor_sec == baseline_warm_runtime_sec and baseline_warm_runtime_sec > 0
                                else "cold"
                            ),
                            "overtime_kill_ratio": overtime_kill_ratio,
                            "stage": FAILURE_STAGE_DECISION,
                            "error_class": "killed_overtime",
                        }
                        if gv.name:
                            round_name_index[gv.name] = fp
                        rejected_update.append(
                            {
                                "fingerprint": fp,
                                "name": gv.name,
                                "extra_server_args": gv.extra_server_args,
                                "extra_envs": dict(gv.extra_envs),
                                **control_fields,
                                "note": gv.note,
                                "reason": "killed_overtime",
                                "gain_pct": None,
                                "tput": None,
                                "estimated_output_throughput": est_tput,
                                "runtime_sec": round(variant_runtime, 2),
                                "wall_clock_ratio_vs_baseline": wall_clock_ratio,
                                "round_id": round_id,
                                "ts": _now_iso(),
                                "provenance": provenance,
                            }
                        )
                        losers.append(
                            {
                                "fingerprint": fp,
                                "name": gv.name,
                                "extra_server_args": gv.extra_server_args,
                                "extra_envs": dict(gv.extra_envs),
                                **control_fields,
                                "provenance": provenance,
                                "gain_pct": None,
                                "tput": None,
                                "estimated_output_throughput": est_tput,
                                "reason": "killed_overtime",
                                "workspace": r.workspace,
                                "runtime_sec": round(variant_runtime, 2),
                                "wall_clock_ratio_vs_baseline": wall_clock_ratio,
                            }
                        )
                        log.warning(
                            "explore: variant %s KILLED_OVERTIME "
                            "(runtime=%.1fs vs %s anchor=%.1fs, ratio=%.2fx, "
                            "kill_ratio=%.2fx, est_output_tput=%s tok/s); "
                            "skipping KEEP/REVERT ladder.",
                            gv.name,
                            variant_runtime,
                            "warm"
                            if (decision_anchor_sec == baseline_warm_runtime_sec and baseline_warm_runtime_sec > 0)
                            else "cold",
                            decision_anchor_sec,
                            wall_clock_ratio if wall_clock_ratio is not None else -1.0,
                            overtime_kill_ratio,
                            f"{est_tput:.1f}" if est_tput is not None else "n/a",
                        )
                        continue

                    # Decision-round gain is the gate: a variant KEEPs when it clears keep_threshold and the accuracy
                    # gate.
                    cand_snap = perf_snapshot_from_mapping(
                        {
                            "output_throughput": r.output_throughput,
                            "input_throughput": r.input_throughput,
                            "total_throughput": r.total_token_throughput,
                            "intvty_p90": r.intvty_p90,
                            "tpot_p90_ms": r.tpot_p90_ms,
                        }
                    )
                    gain: float | None
                    outcome = "FAILED"
                    reason: str = ""
                    _graded_on_total = False
                    # Each gate's verdict as it rules, in the order it ruled.
                    # Recorded here because this is where it is known: read off
                    # the outcome afterwards, "REVERT" cannot say which gate
                    # ended the arc, and a gate that never ran is indistinguishable
                    # from one that ruled against the variant.
                    decision_gates: list[dict[str, Any]] = []
                    if grade_on_total and running_base_perf and cand_snap:
                        _graded_on_total = True
                        intvty_ok = passes_intvty_gate(cand_snap, running_base_perf)
                        decision_gates.append(
                            {
                                "gate": "intvty_p90",
                                "passed": intvty_ok,
                                "observed": cand_snap.get("intvty_p90"),
                                # The anchor is the reference; the noise band it
                                # is allowed to regress within belongs to the
                                # gate, as the tolerance does for accuracy.
                                "threshold": running_base_perf.get("intvty_p90"),
                                "reason": "" if intvty_ok else "intvty_regression",
                            }
                        )
                        if intvty_ok:
                            gain = gain_pct(total_tput_of(cand_snap), total_tput_of(running_base_perf))
                        else:
                            gain = None
                            outcome = "REVERT"
                            reason = "intvty_regression"
                    else:
                        if grade_on_total and running_base_perf:
                            log.info(
                                "explore: variant %r missing graded axes (intvty_p90=%s total=%s); "
                                "grading on output throughput",
                                gv.name,
                                r.intvty_p90,
                                r.total_token_throughput,
                            )
                        gain = gain_pct(r.output_throughput, running_base_tput)
                    if not reason:
                        if r.status != "succeeded" or gain is None:
                            reason = (r.error or "")[-1200:] or "no_measurement"
                        else:
                            # A measurement exists, so this gate ruled. It does
                            # not rule at all when there is none, which is why
                            # no row is appended above.
                            decision_gates.append(
                                {
                                    "gate": "keep_threshold",
                                    "passed": gain >= keep_threshold_pct,
                                    "observed": gain,
                                    "threshold": keep_threshold_pct,
                                    "reason": "" if gain >= keep_threshold_pct else "gain_below_threshold",
                                }
                            )
                            if gain < keep_threshold_pct:
                                outcome = "REVERT"
                                reason = "gain_below_threshold"
                    accuracy_value: float | None = None
                    accuracy_reference: float | None = None
                    accuracy_gated = False
                    if outcome == "FAILED" and not reason:
                        # Accuracy gate.
                        from hyperloom.inference_optimizer import framework_registry

                        scriptable = framework_registry.is_scriptable(framework)
                        accuracy_ok = True
                        accuracy_value: float | None = None
                        # Serving still needs a measured baseline to compare against; scriptable compares against a
                        # fixed 1.0.
                        if scriptable or baseline_accuracy > 0:
                            accuracy_gated = True
                            eval_out = parse_eval_results(slot, framework=framework)
                            accuracy_value = eval_out.get("accuracy")
                            if isinstance(accuracy_value, (int, float)):
                                # Scriptable maps gate pass→1.0 / fail→0.0, so compare against a perfect reference
                                # (1.0); serving compares vs the measured baseline.
                                reference = 1.0 if scriptable else baseline_accuracy
                                accuracy_reference = reference
                                accuracy_ok = accuracy_passed(
                                    reference,
                                    float(accuracy_value),
                                )
                            else:
                                # No eval result.
                                accuracy_ok = False
                        if accuracy_gated:
                            decision_gates.append(
                                {
                                    "gate": "accuracy",
                                    "passed": accuracy_ok,
                                    "observed": accuracy_value if isinstance(accuracy_value, (int, float)) else None,
                                    "threshold": accuracy_reference,
                                    "reason": ""
                                    if accuracy_ok
                                    else ("accuracy_unavailable" if accuracy_value is None else "accuracy_drop"),
                                }
                            )
                        if not accuracy_ok:
                            outcome = "REVERT"
                            reason = "accuracy_unavailable" if accuracy_value is None else "accuracy_drop"
                        else:
                            outcome = "KEEP"

                    decision_tput = r.output_throughput
                    round_tested[fp] = {
                        "fingerprint": fp,
                        "name": gv.name,
                        "extra_server_args": gv.extra_server_args,
                        "extra_envs": dict(gv.extra_envs),
                        **control_fields,
                        "note": gv.note,
                        "outcome": outcome,
                        "status": r.status,
                        "tput": decision_tput,
                        "decision_tput": decision_tput,
                        "input_throughput": r.input_throughput,
                        "total_throughput": r.total_token_throughput,
                        "intvty_p90": r.intvty_p90,
                        "tpot_p90_ms": r.tpot_p90_ms,
                        "gain_pct": gain,
                        "graded_objective": "total_throughput" if _graded_on_total else "output_throughput",
                        "base_tput": running_base_tput,
                        "round_id": round_id,
                        "ts": _now_iso(),
                        "provenance": provenance,
                        "scope": scope,
                        "workload_signature": ws_sig,
                        "framework": framework,
                        "workspace": r.workspace,
                        "error_class": r.error_class or "",
                        "server_log_path": r.server_log_path,
                        "launch_evidence": dict(r.launch_evidence or {}),
                        "launch_evidence_path": r.launch_evidence_path,
                        "stage": FAILURE_STAGE_DECISION,
                        "measured_against": _measured_against(),
                        "gates": decision_gates,
                        # What stood behind an adoption. A session with no
                        # baseline accuracy gates nothing, so its KEEPs rest on
                        # throughput alone -- a weaker claim than one an
                        # accuracy gate ruled on, and the two must not read
                        # alike. Only a KEEP carries it, matching the field's
                        # adoption-scoped meaning elsewhere: on a reverted row
                        # "accuracy_pass" would name the gate that refused it.
                        # What ruled against those is in ``gates``.
                        "validation_basis": (
                            ("accuracy_pass" if accuracy_gated else "keep_verdict_unscored")
                            if outcome == "KEEP"
                            else ""
                        ),
                    }
                    if gv.name:
                        round_name_index[gv.name] = fp

                    # ---- KEEP path ----
                    if outcome == "KEEP":
                        # Layer onto the running stack.
                        next_effective_args = compose_server_args(
                            inherited_args=_effective_inherited_args,
                            base_extra_args=stack_extra_args,
                            variant_extra_args=gv.extra_server_args,
                            remove_args=run_remove_args,
                            args_mode="replace"
                            if stack_base_args_mode == "replace"
                            else getattr(gv, "args_mode", "append"),
                        )
                        next_stack_args = compose_server_args(
                            inherited_args="",
                            base_extra_args=stack_extra_args,
                            variant_extra_args=gv.extra_server_args,
                            remove_args=to_str_list(getattr(gv, "remove_args", [])),
                            args_mode=getattr(gv, "args_mode", "append"),
                        )
                        next_envs = dict(stack_extra_envs)
                        for k in run_unset_envs:
                            next_envs.pop(str(k), None)
                        next_envs.update(gv.extra_envs)
                        effective_control_fields = dict(control_fields)
                        if run_remove_args:
                            effective_control_fields["remove_args"] = list(run_remove_args)
                        if run_unset_envs:
                            effective_control_fields["unset_envs"] = list(run_unset_envs)
                        persist_effective_args = bool(
                            run_remove_args
                            or str(getattr(gv, "args_mode", "append") or "append").strip().lower() == "replace"
                            or stack_base_args_mode == "replace"
                        )
                        if persist_effective_args:
                            effective_control_fields["args_mode"] = "replace"
                        keep_entry = {
                            "fingerprint": fp,
                            "name": gv.name,
                            "candidate_extra_server_args": gv.extra_server_args,
                            "candidate_extra_envs": dict(gv.extra_envs or {}),
                            "recipe_delta": {
                                "extra_server_args": gv.extra_server_args,
                                "extra_envs": dict(gv.extra_envs or {}),
                                **control_fields,
                            },
                            "extra_server_args": next_effective_args if persist_effective_args else next_stack_args,
                            "effective_extra_server_args": next_effective_args,
                            "extra_envs": dict(next_envs),
                            **effective_control_fields,
                            "note": gv.note,
                            "provenance": provenance,
                            # Names of the authored kernels this config carried, when an overlay was loaded.
                            "accepted_kernels": list(getattr(gv, "accepted_kernels", []) or []),
                            "gain_pct": gain,
                            "graded_objective": "total_throughput" if _graded_on_total else "output_throughput",
                            # The verdict this KEEP rests on.
                            "accuracy": accuracy_value,
                            "tput": decision_tput,
                            "decision_tput": decision_tput,
                            # The axes this KEEP was graded on travel with it: current_best becomes the next round's
                            # anchor, and an anchor without them degrades the session.
                            "input_throughput": r.input_throughput,
                            "total_throughput": r.total_token_throughput,
                            "intvty_p90": r.intvty_p90,
                            "tpot_p90_ms": r.tpot_p90_ms,
                            "single_workspace": r.workspace,
                            "launch_evidence": dict(r.launch_evidence or {}),
                            "launch_evidence_path": r.launch_evidence_path,
                            "round_id": round_id,
                            "accepted_at_round": round_id,
                            "ts": _now_iso(),
                        }
                        # The variant KEEPs on the round that graded it.
                        stack_extra_args = next_effective_args if persist_effective_args else next_stack_args
                        stack_extra_envs = next_envs
                        stack_remove_args = list(run_remove_args)
                        stack_unset_envs = list(run_unset_envs)
                        stack_base_args_mode = "replace" if persist_effective_args else "append"
                        if decision_tput and decision_tput > 0:
                            running_base_tput = decision_tput
                        if grade_on_total and cand_snap and _graded_on_total:
                            running_base_perf = cand_snap
                        elif grade_on_total and not _graded_on_total:
                            log.info(
                                "explore: KEEP %r graded on output throughput; clearing the total anchor "
                                "so the rest of this round grades on output too",
                                gv.name,
                            )
                            running_base_perf = None

                        winners.append(keep_entry)
                        winners_history_update.append(
                            {
                                "round_id": round_id,
                                "variant_name": gv.name,
                                "fingerprint": fp,
                                "gain_pct": gain,
                                "extra_args": gv.extra_server_args,
                                "extra_envs": dict(gv.extra_envs),
                                **control_fields,
                                "provenance": provenance,
                                "scope": scope,
                                "ts": _now_iso(),
                            }
                        )
                        continue

                    # ---- REVERT / FAILED ----
                    rejected_update.append(
                        {
                            "fingerprint": fp,
                            "name": gv.name,
                            "extra_server_args": gv.extra_server_args,
                            "extra_envs": dict(gv.extra_envs),
                            **control_fields,
                            "note": gv.note,
                            "reason": reason or "not_keep",
                            "gain_pct": gain,
                            "tput": decision_tput,
                            "round_id": round_id,
                            "ts": _now_iso(),
                            "provenance": provenance,
                            "error_class": r.error_class or "",
                            "server_log_path": r.server_log_path,
                        }
                    )
                    losers.append(
                        {
                            "fingerprint": fp,
                            "name": gv.name,
                            "extra_server_args": gv.extra_server_args,
                            "extra_envs": dict(gv.extra_envs),
                            **control_fields,
                            "provenance": provenance,
                            "gain_pct": gain,
                            "tput": decision_tput,
                            "reason": reason or "not_keep",
                            "workspace": r.workspace,
                        }
                    )
                finally:
                    # Reap THIS variant's persistent server on every exit path (idempotent + no-op when reuse was
                    # ineligible).
                    if lifecycle_eligible:
                        teardown_lifecycle_server(
                            pid_dir=slot,
                            framework=lifecycle_framework,
                            port=lifecycle_port,
                        )
        finally:
            # Release the round's Ray serving lease/actor exactly once (this was a per-variant ``ray.kill`` before —
            # the raylet worker churn that destabilised the single-node cluster).
            if round_serving_lease is not None:
                round_serving_lease.close()

        # ----- Ledger compaction (per-fingerprint last-wins) ---------------- This round's writes over the ledger it
        # inherited: a re-run fingerprint replaces its earlier row, which is what a fresh measurement means, and a
        # variant this round rolled back leaves the earlier row standing.
        tested_update: dict[str, dict[str, Any]] = {**tested_dict, **round_tested}
        name_index: dict[str, Any] = {**inherited_name_index, **round_name_index}
        rejected_dedup: dict[str, dict[str, Any]] = {}
        for entry in rejected_update:
            fp = str(entry.get("fingerprint") or "")
            if not fp:
                continue
            rejected_dedup[fp] = entry

        # Flat per-variant outcomes for the Coordinator's per-variant fact-write hook (this round's outcomes).
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
                "KEEP",
                "REVERT",
                "FAILED",
                "KILLED_OVERTIME",
            ):
                continue
            metrics: dict[str, Any] = {}
            if te.get("tput") is not None:
                metrics["tput"] = te.get("tput")
            if te.get("gain_pct") is not None:
                metrics["gain_pct"] = te.get("gain_pct")
            # The anchor this variant's gain was measured against. Carried
            # verbatim rather than left to be back-solved from the gain: each
            # KEEP advances ``running_base_tput``, so a consumer dividing the
            # gain out of the final throughput reconstructs whichever anchor
            # happens to be current, not the one this variant was judged on --
            # and for a FAILED or killed variant there is no gain to divide.
            if te.get("base_tput") is not None:
                metrics["base_tput"] = te.get("base_tput")
            # Rough decode tput salvaged from a killed-overtime variant's
            # partial server.log. Informational only (no ``tput``/gain).
            if te.get("estimated_output_throughput") is not None:
                metrics["estimated_output_throughput"] = te.get(
                    "estimated_output_throughput",
                )
            # Surface wall-clock + kill ratio so the LLM/KB sees "ran too slow → early kill" instead of an opaque
            # FAILED row.
            if te.get("runtime_sec") is not None:
                metrics["runtime_sec"] = te.get("runtime_sec")
            if te.get("wall_clock_ratio_vs_baseline") is not None:
                metrics["wall_clock_ratio_vs_baseline"] = te.get(
                    "wall_clock_ratio_vs_baseline",
                )
            per_variant_outcomes.append(
                {
                    "variant_name": str(te.get("name") or ""),
                    "outcome": outcome,
                    "fingerprint": fp_key,
                    "failure_id": make_failure_id(
                        task_id=str(ctx.task.task_id),
                        fingerprint=fp_key,
                        variant_name=str(te.get("name") or ""),
                    ),
                    "stage": str(te.get("stage") or FAILURE_STAGE_DECISION),
                    "error_excerpt": te.get("error_excerpt"),
                    "provenance": str(te.get("provenance") or ""),
                    "scope": str(te.get("scope") or ""),
                    "metrics": metrics,
                    "reason": reasons_by_fp.get(fp_key, ""),
                    "error_class": str(te.get("error_class") or ""),
                    "server_log_path": te.get("server_log_path"),
                    "workspace": te.get("workspace"),
                    "raw_result_path": te.get("raw_result_path"),
                    # Carry the variant knobs so the journal's ``classify_change_kind`` can classify the change kind.
                    "variant": {
                        "name": str(te.get("name") or ""),
                        "extra_server_args": str(te.get("extra_server_args") or ""),
                        "extra_envs": dict(te.get("extra_envs") or {}),
                        "note": str(te.get("note") or ""),
                    },
                    # The verdicts and the stack behind them, as the round
                    # ruled. Absent keys mean the variant never got that far:
                    # no gate ruled on a warmup failure, and nothing validated
                    # one that was killed before it was graded.
                    "measured_against": te.get("measured_against") or {},
                    "gates": [gate for gate in (te.get("gates") or []) if isinstance(gate, dict)],
                    "validation_basis": str(te.get("validation_basis") or ""),
                }
            )
        for sd in skipped_dup:
            per_variant_outcomes.append(
                {
                    "variant_name": str(sd.get("name") or ""),
                    "outcome": "SKIPPED_DEDUP",
                    "fingerprint": str(sd.get("fingerprint") or ""),
                    "provenance": "",
                    "metrics": {},
                    "reason": str(sd.get("reason") or ""),
                }
            )

        lever_attributions = _framework_lever_attributions(
            per_variant_outcomes,
            lever_payload,
        )
        if lever_attributions:
            log.info(
                "explore: attributed %d framework rewrite lever(s): %s",
                len(lever_attributions),
                ", ".join(f"{a['switch']}={a['gain_pct']:+.2f}%" for a in lever_attributions),
            )

        # ``last_round`` summary for the prompt / breakdown.
        killed_overtime_fps = [
            str(te.get("fingerprint") or "")
            for te in tested_update.values()
            if te.get("round_id") == round_id and te.get("outcome") == "KILLED_OVERTIME"
        ]
        last_round_summary = {
            "round_id": round_id,
            "base_tput": base_tput,
            "base_extra_args": base_extra_args,
            "tested": [w["fingerprint"] for w in winners] + [lr["fingerprint"] for lr in losers],
            "round_winners": [w["fingerprint"] for w in winners],
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

        # Each KEEP advances ``running_base_tput``, so this is the final stack.
        output_throughput = float(running_base_tput) if winners else None

        # Successful = at least one bench produced a measurement or was reaped by the overtime gate (KILLED_OVERTIME
        # is a real signal).
        produced_measurement = any(
            t.get("outcome")
            in (
                "KEEP",
                "REVERT",
                "KILLED_OVERTIME",
            )
            for t in tested_update.values()
            if t.get("round_id") == round_id
        )
        status = "succeeded" if produced_measurement or winners else "failed"
        # A round that measured nothing because the run stopped it is not the same as one whose variants failed, and
        # it used to be reported as a bare ``failed`` with no error_class at all -- nothing downstream could tell the
        # two apart, so the KB could learn that these variants are bad.
        budget_error: dict[str, Any] = {}
        if status == "failed" and run_stop is not None:
            budget_error = {
                "error_class": run_stop.error_class,
                "error": (
                    f"{run_stop_detail}; {session_budget_untested} variant(s) went unmeasured "
                    "and stay out of the ledger so a resume can retry them"
                ),
            }

        return {
            "status": status,
            **budget_error,
            "session_budget_untested": session_budget_untested,
            "base_tput": base_tput,
            "running_base_tput": running_base_tput,
            "output_throughput": output_throughput,
            "best_variant": best_winner,
            "best_gain_pct": best_gain_pct,
            "winners": winners,
            "losers": losers,
            "skipped_dup": skipped_dup,
            # flat per-variant outcomes.
            "per_variant_outcomes": per_variant_outcomes,
            "framework_lever_attributions": lever_attributions,
            "explore_search_update": search_update,
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
    "ExploreExecutor",
    "explore_executor",
]
