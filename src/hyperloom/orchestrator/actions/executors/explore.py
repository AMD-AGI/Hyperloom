# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config
from hyperloom.common.benchmark_config import is_magpie_agentx
import functools
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.coerce import to_float, to_int, to_str_list
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.launch_log_evidence import settings_the_engine_rewrote
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    VERDICT_REVERT,
    GradingAnchor,
    agentx_active,
    constraint_fields,
    graded_axes_of,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    stamp_output_per_gpu,
)
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...state.failure_evidence import (
    UNMEASURED_OUTCOMES,
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    classify_failure_attribution,
    make_failure_id,
    tail_excerpt,
)
from ...state.shared_state import (
    first_positive_tput,
    resolve_anchor_with_drift,
    resolve_graded_comparison,
    stack_base_params,
)
from ..stop_attribution import (
    StoppedByTheRun,
    stopped_by_the_run_class,
    variant_budget_stop,
)
from ._accuracy_gate import (
    accuracy_passed,
    parse_eval_results,
)
from . import _framework_switch_manifest as _switch_manifest
from hyperloom.inference_optimizer.canonical_fingerprint import workload_signature
from ._proposal_identity import BaseControls, controls_of, effective_fingerprint, normalize_proposal
from .measured_variants import GAIN_BELOW_THRESHOLD, MeasuredOn, measured_verdict, stack_of_measured
from ._grid_base import DEFAULT_KEEP_THRESHOLD_PCT, TS_FAILED, TS_SKIPPED_DEDUP
from .benchmark_result import double_run_requested
from ._grid_runner import (
    GridLaunch,
    GridRecipe,
    GridStop,
    GridVariant,
    SessionDirField,
    _build_variant_yaml,
    _num_gpus_for_config,
    _safe,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from ._grid_variant_filter import (
    ServingStack,
    apply_aiter_moe_pin_filter,
    apply_compatibility_filter,
    apply_multi_node_invalid_variants,
    apply_user_skip_list,
    reorder_grid_for_multi_node,
    resolve_skip_spec,
)
from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
)
from ._ray_serving import maybe_serving_lease
from ._spec_decode_context_copy import CONTEXT_COPY_REFUSAL, context_copy_refusal

from ._launch_restatement import Launch, config_envs, launch_change, observed_launch_after, read_launch
from ._server_argv import ConfigUnreadable, config_server_argv
from ._server_lifecycle import (
    resolve_lifecycle_params,
    teardown_lifecycle_server,
)
from ._workload_envs import (
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
)


log = logging.getLogger(__name__)


_now_iso = functools.partial(now_iso, "auto")

STACK_REVALIDATE_SOURCE: str = "stack_revalidate"

# ``skipped_dup`` reason for a variant whose launch the session already measured on the same stack.
ALREADY_MEASURED: str = "already_measured"
# ... for one that starts the stack's own server, or a server this round already launched.
BASELINE_NOOP: str = "baseline_noop"
ROUND_DUP: str = "round_dup"

#: The provenance of a grid row that names none: the model proposed it directly.
_DIRECT_PROVENANCE = "llm_direct"


def is_stack_revalidation(params: dict | None) -> bool:
    return str((params or {}).get("source") or "") == STACK_REVALIDATE_SOURCE


# Audit/provenance metadata stashed on a GridVariant that must survive being rebuilt into a derived variant.
_CARRIED_VARIANT_ATTRS: tuple[str, ...] = (
    "provenance",
    "reasoning_origin",
    "scope",
    "overlay_pythonpath",
    "accepted_kernels",
    "experience_citations",
)


def _decision_fields(gv: Any) -> dict[str, Any]:
    """Why this variant was proposed, as authored at action time, for every row that records it."""
    return {
        "note": gv.note,
        "reasoning_origin": str(getattr(gv, "reasoning_origin", "") or ""),
        "experience_citations": list(getattr(gv, "experience_citations", []) or []),
    }


def _variant_with_eval_off(gv: Any) -> Any:
    """Return a copy of ``gv`` with ``RUN_EVAL`` forced off.

    The variant layer is the last one :func:`run_grid` applies over the
    accumulated stack, so forcing the key here holds whatever the stack carries.
    """
    envs = dict(gv.extra_envs)
    envs["RUN_EVAL"] = "false"
    return _carry_variant_metadata(
        gv,
        GridVariant(
            name=gv.name,
            extra_server_args=gv.extra_server_args,
            extra_envs=envs,
            note=gv.note,
            remove_args=to_str_list(getattr(gv, "remove_args", [])),
            unset_envs=to_str_list(getattr(gv, "unset_envs", [])),
            args_mode=str(getattr(gv, "args_mode", "append") or "append"),
        ),
    )


def _carry_variant_metadata(src: Any, dst: Any) -> Any:
    """Copy the carried audit metadata from ``src`` onto ``dst``."""
    for attr in _CARRIED_VARIANT_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst


def _variant_fingerprint(variant: Any) -> str:
    """The canonical fingerprint stamped on a variant before the round launched it."""
    return str(getattr(variant, "canonical_fp", "") or "")


def _keep_uses_effective_args(
    benchmark: dict[str, Any], base_mode: str, variant_mode: str, remove_args: list[str], unset_envs: list[str]
) -> bool:
    """Native KEEP edits remain relative to the verified recipe unless explicitly replaced."""
    return (
        base_mode == "replace"
        or variant_mode == "replace"
        or (not is_magpie_agentx(benchmark) and bool(remove_args or unset_envs))
    )


def _action_reasoning(raw: dict[str, Any]) -> tuple[str, str]:
    """Return the action-authored rationale and the payload field that supplied it."""
    for field in ("reasoning", "rationale", "reason", "note"):
        value = str(raw.get(field) or "").strip()
        if value:
            return value, f"action_payload.{field}"
    return "", ""


def _grid_variants_from_payload(payload: list[Any]) -> list[GridVariant]:
    """Convert the LLM/specialist grid payload into GridVariant objects."""
    out: list[GridVariant] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        fields = normalize_proposal(raw)
        reasoning, reasoning_origin = _action_reasoning(raw)
        gv = GridVariant(
            name=fields["name"],
            extra_server_args=fields["extra_args"],
            extra_envs=fields["extra_envs"],
            note=reasoning,
            remove_args=fields["remove_args"],
            unset_envs=fields["unset_envs"],
            args_mode=fields["args_mode"],
        )
        # Stash extra metadata on the GridVariant so the ledger writer can pull provenance.
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.reasoning_origin = reasoning_origin  # type: ignore[attr-defined]
        gv.scope = str(raw.get("scope") or "")  # type: ignore[attr-defined]
        # Authored-kernel overlay dir (PYTHONPATH prefix); "" for env/flag variants.
        gv.overlay_pythonpath = str(raw.get("overlay_pythonpath") or "")  # type: ignore[attr-defined]
        # Authored kernels this variant's overlay installs.
        gv.accepted_kernels = [  # type: ignore[attr-defined]
            str(k).strip() for k in (raw.get("accepted_kernels") or []) if str(k).strip()
        ]
        # Validated against what the proposing agent was shown before the grid reached this executor.
        gv.experience_citations = [  # type: ignore[attr-defined]
            dict(item) for item in (raw.get("experience_citations") or []) if isinstance(item, dict)
        ]
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


def _xdit_default_grid() -> list[GridVariant]:
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


def observed_launch_from_state(state: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """The running server's resolved config and env, empty unless ``current_best`` observed them.

    A setting the engine rewrote after parsing it is left out of the config: its
    resolved value is not the value a flag would pass, so it reads as unknown
    and a variant touching it runs.
    """
    measurement = getattr(state, "current_best_measurement", None)
    if not isinstance(measurement, dict) or not measurement:
        return {}, {}
    evidence = measurement.get("launch_evidence")
    if not isinstance(evidence, Mapping):
        return {}, {}

    raw_config = evidence.get("observed_server_config")
    config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
    raw_adjusted = evidence.get("engine_adjusted_settings")
    for name in settings_the_engine_rewrote(config, raw_adjusted if isinstance(raw_adjusted, Mapping) else None):
        config.pop(name, None)
    raw_env = evidence.get("requested_server_env")
    # Same shape as a variant's extra_envs, so the two compare directly.
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, Mapping) else {}
    return config, env


#: Names the throwaway variant that materializes the unchanged baseline to diff against.
_NOOP_PROBE_BASE_NAME = "__baseline_noop_probe_base__"


def _materialized_launch(recipe: GridRecipe, gv: GridVariant, output_subdir: Path) -> tuple[str, dict[str, str]]:
    """The server argv and benchmark envs ``run_grid`` would launch *gv* with."""
    out_path = _build_variant_yaml(recipe, gv, output_subdir=output_subdir)
    return config_server_argv(out_path).text, config_envs(out_path)


#: What reading a probe launch can raise; a launch that cannot be read is never judged unchanged.
_UNREADABLE_LAUNCH = (OSError, ValueError, yaml.YAMLError, ConfigUnreadable)


def _probe_launch(recipe: GridRecipe, gv: GridVariant, output_subdir: Path) -> Launch:
    """The whole launch ``run_grid`` would start *gv* with on ``recipe``'s base."""
    out_path = _build_variant_yaml(recipe, gv, output_subdir=output_subdir)
    return read_launch(out_path, [*to_str_list(recipe.base.unset_envs), *to_str_list(gv.unset_envs)])


def filter_context_copy_drafts(
    grid: list[GridVariant], *, recipe: GridRecipe
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that introduce or change a context-copying speculative-decoding draft (synthetic sessions only).

    Each variant and its base are materialized through ``_build_variant_yaml``, the way ``run_grid`` builds the round,
    so a draft reached through args, replace mode, envs or a server-args env counts. A launch that cannot be
    materialized is judged on the variant's own args and envs against an empty base.

    Returns the variants still to run, plus ``(name, reason)`` for each drop.
    """
    if not grid:
        return [], []
    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="explore_spec_probe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            base = _materialized_launch(recipe, GridVariant(_NOOP_PROBE_BASE_NAME), tmp_root / "base")
        except (OSError, ValueError, yaml.YAMLError, ConfigUnreadable) as exc:
            log.warning("explore: could not materialize the base to read its speculative decoding (%s)", exc)
            base = ("", {})
        for idx, gv in enumerate(grid):
            against = base
            try:
                launch = _materialized_launch(recipe, gv, tmp_root / f"v{idx}")
            except (OSError, ValueError, yaml.YAMLError, ConfigUnreadable) as exc:
                log.warning("explore: could not materialize %s to read its speculative decoding (%s)", gv.name, exc)
                launch = (str(gv.extra_server_args or ""), {str(k): str(v) for k, v in (gv.extra_envs or {}).items()})
                against = ("", {})
            reason = context_copy_refusal(against[0], against[1], launch[0], launch[1])
            if reason:
                dropped.append((gv.name, reason))
            else:
                kept.append(gv)
    return kept, dropped


def filter_baseline_noop_variants(
    grid: list[GridVariant],
    *,
    observed: tuple[Mapping[str, Any], Mapping[str, str]],
    recipe: GridRecipe,
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that would launch the server the stack is already running.

    The baseline and each variant are materialized through ``_build_variant_yaml`` and read back whole, so a
    removal, a replaced argv, an overlay or an AgentX launch override shows as the change it is; ``launch_change``
    then discounts what the observed server already resolved. A variant whose launch cannot be read always runs;
    a stack revalidation round never reaches this filter.

    Returns the variants still to run, plus ``(name, reason)`` for each drop.
    """
    if not grid:
        return [], []
    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="explore_noop_probe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            base = _probe_launch(recipe, GridVariant(_NOOP_PROBE_BASE_NAME), tmp_root / "base")
        except _UNREADABLE_LAUNCH as exc:
            log.warning(
                "explore: baseline-noop probe could not read the baseline launch (%s); keeping every variant", exc
            )
            return list(grid), []
        for idx, gv in enumerate(grid):
            try:
                launch = _probe_launch(recipe, gv, tmp_root / f"v{idx}")
            except _UNREADABLE_LAUNCH as exc:
                log.warning("explore: baseline-noop probe could not read variant %s (%s); keeping it", gv.name, exc)
                kept.append(gv)
                continue
            if launch_change(base, launch, observed):
                kept.append(gv)
            else:
                dropped.append((gv.name, "launches the server the stack already runs (baseline noop)"))
    return kept, dropped


def restates_the_baseline_launch(state: Any, args: str, envs: Mapping[str, Any]) -> bool:
    """Whether ``args`` and ``envs`` laid onto the baseline's materialized config start the baseline's own server.

    Judged as a variant on that config is. The observed launch discounts settings only when ``current_best`` was
    observed running that very config.
    """
    config_path = str(getattr(state, "baseline_config_path", "") or "")
    if not config_path:
        return False
    measurement = getattr(state, "current_best_measurement", None)
    evidence = measurement.get("launch_evidence") if isinstance(measurement, Mapping) else None
    observed_here = isinstance(evidence, Mapping) and evidence.get("materialized_config_path") == config_path
    _, dropped = filter_baseline_noop_variants(
        [GridVariant("warm_replay", str(args or ""), {str(k): str(v) for k, v in envs.items()})],
        observed=observed_launch_from_state(state) if observed_here else ({}, {}),
        recipe=GridRecipe(base_yaml_path=Path(config_path)),
    )
    return bool(dropped)


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
) -> list[GridVariant]:
    """Framework-keyed default grid dispatch."""
    fw = (framework or "").strip().lower()
    if fw == "atom":
        return _atom_default_grid(model_class=model_class, conc=conc)
    if fw == "xdit":
        return _xdit_default_grid()
    return []


class _RoundRefused(Exception):
    """A round that cannot start, carrying the action result that says why."""

    def __init__(self, error_class: str, error: str, **extra: Any) -> None:
        super().__init__(error)
        self.result: dict[str, Any] = {"status": "failed", "error_class": error_class, "error": error, **extra}


def _materialized_round_config(
    params: dict[str, Any], output_root: Path, target: BenchmarkTarget, grading: Any
) -> Path:
    # Re-materialize the baseline config so variant YAMLs honour the operator's actual workload.
    #
    # CONC / ISL / OSL / TP / MAX_MODEL_LEN / PRECISION all reach the variants through it.
    config_path = Path(params["config_path"])
    if not config_path.exists():
        raise _RoundRefused("missing_config", f"config not found: {config_path}")
    try:
        return prepare_grid_benchmark_config(
            config_path,
            output_root,
            target=BenchmarkTarget(
                model_path=target.model_path or None,
                gpu_type=target.gpu_type or None,
                benchmark_script=target.benchmark_script or None,
            ),
            options=MaterializeOptions(out_name="explore_base.with_envs.yaml", grading=grading),
        )
    except FrameworkScriptMismatchError as exc:
        raise _RoundRefused("framework_script_mismatch", str(exc)) from exc


def _framework_of(config_path: Path) -> tuple[str, dict[str, Any]]:
    try:
        with config_path.open(encoding="utf-8") as handle:
            bench = (yaml.safe_load(handle) or {}).get("benchmark") or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("explore: could not resolve framework from %s: %s", config_path, exc)
        return "", {}
    return str(bench.get("framework") or "").lower(), bench


def _anchor_the_params(params: dict[str, Any], ss: Any) -> None:
    # Refresh the anchor and the stack it was measured on, which a KEEP landing while this task queued invalidates.
    #
    # Params snapshot the two together, so they are refreshed as a pair. A revalidation
    # reproduces the saved stack, so it never re-anchors.
    snapshot_tput = to_float(params.get("base_tput"), 0.0) or 0.0
    anchor, drifted = (
        (snapshot_tput, False)
        if params.get("source") == STACK_REVALIDATE_SOURCE
        else resolve_anchor_with_drift(snapshot_tput, ss)
    )
    if anchor <= snapshot_tput:
        return
    if drifted:
        log.warning("explore: anchor drift %.1f -> %.1f; re-reading base args", snapshot_tput, anchor)
    params["base_tput"] = anchor
    cb = getattr(ss, "current_best", None)
    # Only current_best carries args; a baseline_tput anchor leaves the params stack (seeded from the baseline
    # record) authoritative.
    if first_positive_tput(cb) > 0:
        params.update(stack_base_params(cb))


def _lever_seeded_payload(payload: list[Any], levers: list[dict[str, Any]]) -> list[Any]:
    named = {str(v.get("name") or "") for v in payload if isinstance(v, dict)}
    fresh = [v for v in levers if str(v.get("name") or "") not in named]
    if not fresh:
        return payload
    log.info("explore: seeding %d framework-rewrite lever variant(s) for attribution", len(fresh))
    return fresh + list(payload)


def _seed_grid_payload(params: dict[str, Any], framework: str, yaml_envs: dict[str, Any]) -> list[Any]:
    model_class = str(params.get("model_class") or "").strip() or os.environ.get("MODEL_CLASS", "").strip()
    conc = to_int(params.get("conc") or yaml_envs.get("CONC") or os.environ.get("CONC") or 0, 0) or 0
    seed = _default_grid_for_framework(framework, model_class=model_class, conc=conc)
    if not seed:
        return []
    log.info(
        "explore: empty grid for framework=%s; falling through to %d default_grid seed variants "
        "(model_class=%r conc=%d)",
        framework or "?",
        len(seed),
        model_class or "?",
        conc,
    )
    return [
        {
            "name": v.name,
            "extra_server_args": v.extra_server_args,
            "extra_envs": dict(v.extra_envs or {}),
            "note": v.note or "default_grid",
            "provenance": getattr(v, "provenance", "default_grid"),
        }
        for v in seed
    ]


# The numbers a measured variant produced. ``base_tput`` is carried verbatim rather than left to be back-solved
# from the gain: each KEEP advances the round anchor, so a consumer dividing the gain out of the final throughput
# reconstructs whichever anchor happens to be current, and a FAILED variant has no gain to divide.
_VARIANT_METRIC_KEYS: tuple[str, ...] = ("tput", "gain_pct", "base_tput")

# The ledgers' views of the row the round records for a variant.
_VARIANT_FIELDS: tuple[str, ...] = (
    "fingerprint",
    "name",
    "extra_server_args",
    "extra_envs",
    "remove_args",
    "unset_envs",
    "args_mode",
)
_LOSER_FIELDS: tuple[str, ...] = (*_VARIANT_FIELDS, "provenance", "gain_pct", "tput", "reason", "workspace")
_REJECTED_FIELDS: tuple[str, ...] = (
    *_LOSER_FIELDS,
    "round_id",
    "ts",
    "stage",
    "error_class",
    "error_excerpt",
    "server_log_path",
)


def _ledger_view(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: row[key] for key in fields if key in row}


def _variant_knobs(te: dict[str, Any]) -> dict[str, Any]:
    # Carry the variant knobs so the journal's ``classify_change_kind`` can classify the change kind.
    return {
        "name": str(te.get("name") or ""),
        "extra_server_args": str(te.get("extra_server_args") or ""),
        "extra_envs": dict(te.get("extra_envs") or {}),
        "remove_args": list(te.get("remove_args") or []),
        "unset_envs": list(te.get("unset_envs") or []),
        "args_mode": str(te.get("args_mode") or "append"),
        "note": str(te.get("note") or ""),
        "reasoning_origin": str(te.get("reasoning_origin") or ""),
        "experience_citations": list(te.get("experience_citations") or []),
    }


def _outcome_row(te: dict[str, Any], fingerprint: str, reason: str, task_id: str) -> dict[str, Any]:
    name = str(te.get("name") or "")
    outcome = str(te.get("outcome") or "")
    return {
        "variant_name": name,
        "outcome": outcome,
        "fingerprint": fingerprint,
        "failure_id": make_failure_id(task_id=task_id, fingerprint=fingerprint, variant_name=name),
        "stage": str(te.get("stage") or FAILURE_STAGE_DECISION),
        "error_excerpt": te.get("error_excerpt"),
        "provenance": str(te.get("provenance") or ""),
        "scope": str(te.get("scope") or ""),
        "metrics": {key: te[key] for key in _VARIANT_METRIC_KEYS if te.get(key) is not None},
        "reason": reason,
        "error_class": str(te.get("error_class") or ""),
        "failure_attribution": classify_failure_attribution(
            error_class=te.get("error_class"),
            error_excerpt=te.get("error_excerpt"),
            reason=reason,
        )
        if outcome in UNMEASURED_OUTCOMES
        else "",
        "server_log_path": te.get("server_log_path"),
        "workspace": te.get("workspace"),
        "raw_result_path": te.get("raw_result_path"),
        "variant": _variant_knobs(te),
        # The verdicts and the stack behind them, as the round ruled. Absent keys mean the variant never got that
        # far: no gate ruled on a warmup failure, and nothing validated one that was killed before it was graded.
        "measured_against": te.get("measured_against") or {},
        "gates": [gate for gate in (te.get("gates") or []) if isinstance(gate, dict)],
        "validation_basis": str(te.get("validation_basis") or ""),
    }


def _sanitized_overrides(params: dict[str, Any]) -> tuple[str | None, str | None]:
    try:
        return sanitize_script_name(params.get("benchmark_script")), sanitize_result_dir(params.get("result_dir"))
    except ValueError as exc:
        raise _RoundRefused("bad_param", str(exc)) from exc


class _ExploreRound:
    """One explore round: the stack it accumulates, the variants it measures, and the ledgers it writes."""

    def __init__(self, ctx: Any, params: dict[str, Any], output_root: Path) -> None:
        extra = getattr(ctx, "extra", None) or {}
        self.ctx = ctx
        self.params = params
        self.output_root = output_root
        self.ss = extra.get("shared_state")
        self.phase_deadline = extra.get("phase_deadline")
        script, self.result_dir = _sanitized_overrides(params)
        self.target = BenchmarkTarget(
            model_path=resolve_session_model_path(
                params=params,
                state_model_path=str(getattr(self.ss, "model_path", "") or ""),
            ),
            gpu_type=str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower(),
            benchmark_script=script,
        )
        grading = getattr(self.ss, "grading", None)
        self.config_path = _materialized_round_config(params, output_root, self.target, grading)
        _anchor_the_params(params, self.ss)
        self.framework, bench = _framework_of(self.config_path)
        self.bench = bench
        self.yaml_envs = bench.get("envs") or {}
        self.base_args = str(params.get("base_extra_args") or "").strip()
        self.base_envs = dict(params.get("base_extra_envs") or {})
        self.base_remove_args = to_str_list(params.get("base_remove_args"))
        self.base_unset_envs = to_str_list(params.get("base_unset_envs"))
        self.args_mode = str(params.get("base_args_mode") or "append").strip().lower()
        inherited = str(self.yaml_envs.get(server_args_env_name(self.framework)) or "").strip()
        self.inherited_args = "" if self.args_mode == "replace" else inherited
        # ``args`` / ``envs`` carry the running accumulation; after a KEEP they extend with the KEEP'd variant.
        self.args = self.base_args
        self.envs = dict(self.base_envs)
        self.remove_args = list(dict.fromkeys(self.base_remove_args))
        self.unset_envs = list(dict.fromkeys(self.base_unset_envs))
        self.base_tput = to_float(params.get("base_tput"), 0.0) or 0.0
        self.running_tput = self.base_tput
        self.running_perf: Any = None
        self.keep_threshold_pct = float(params["keep_threshold_pct"])
        # The measured baseline outranks a proposed one.
        self.baseline_accuracy = to_float(getattr(self.ss, "baseline_accuracy", 0.0), 0.0) or 0.0
        if self.baseline_accuracy <= 0:
            self.baseline_accuracy = (
                to_float(params.get("accuracy_baseline"), 0.0) or to_float(params.get("baseline_accuracy"), 0.0) or 0.0
            )
        # A context-copying draft is refused on the synthetic benchmark, whose repetition loops it would copy.
        self.refuse_context_copy = not agentx_active(
            benchmark_mode=getattr(self.ss, "benchmark_mode", "")
        ) and not is_stack_revalidation(params)
        self.ws_sig = workload_signature()
        # Seeded by the Coordinator from the durable cursor: this executor holds no history of its own, so it
        # cannot count the rounds before this one.
        self.round_id = f"explore-{(to_int(params.get('explore_search_cursor'), 0) or 0) + 1:03d}"
        self.observed = observed_launch_from_state(self.ss)
        self.opening_stack = self.launched_stack()
        self.lever_payload: list[dict[str, Any]] = []
        self.grid: list[GridVariant] = []
        self.runnable: list[GridVariant] = []
        self.skipped: list[dict[str, Any]] = []
        self.winners: list[dict[str, Any]] = []
        self.losers: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.winners_history: list[dict[str, Any]] = []
        self.tested: dict[str, dict[str, Any]] = {}
        # Servers this round launched, and the variants that reached the measured check and were answered by the
        # stack or the ledger instead (a restatement of the stack, or a verdict the session already holds).
        self.launched: dict[str, str] = {}
        self.judged_without_measuring = 0
        # Set when the loop stops because the run stopped it -- the budget ran out, or the orchestrator cancelled
        # the action -- so the round can say so instead of reporting a bare, unattributed failure: a variant that
        # never ran is not a variant that failed.
        self.run_stop: StoppedByTheRun | None = None
        self.run_stop_detail = ""
        self.untested = 0
        self.lease: Any = None

    async def run(self) -> dict[str, Any]:
        self._resolve_grid()
        self._shape_runnable()
        self._open_schedule()
        # One Ray serving lease (actor) spans the WHOLE round; every variant reuses it.
        self.lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(self.config_path)) if self.runnable else None
        try:
            for idx, gv in enumerate(self.runnable):
                if self._refused(gv, idx):
                    if self.run_stop is not None:
                        break
                    continue
                slot = self.output_root / f"v{idx:02d}_{_safe(gv.name)}"
                slot.mkdir(parents=True, exist_ok=True)
                run_gv = self._staged(gv)
                try:
                    measurement = await self._measure(gv, run_gv, idx, slot)
                    if self.run_stop is not None:
                        break
                    if measurement is not None:
                        self._settle(gv, run_gv, measurement, slot)
                finally:
                    # Reap THIS variant's persistent server on every exit path (idempotent + no-op when reuse was
                    # ineligible).
                    if self.lifecycle_eligible:
                        teardown_lifecycle_server(
                            pid_dir=slot, framework=self.lifecycle_framework, port=self.lifecycle_port
                        )
        finally:
            # Release the round's Ray serving lease/actor exactly once (this was a per-variant ``ray.kill`` before
            # -- the raylet worker churn that destabilised the single-node cluster).
            if self.lease is not None:
                self.lease.close()
        return self._result()

    # ----- the stack the round accumulates ---------------------------------

    def measured_against(self) -> dict[str, Any]:
        """The stack this variant launched on top of, as it stands now.

        Read at each write rather than captured once: every KEEP advances the
        stack, so a variant later in the round was measured against a different
        one than the round opened with. Carried verbatim for the same reason
        ``base_tput`` is -- a consumer that reconstructs the stack from the
        session's current config gets whichever one is current, not the one
        this variant was judged on.
        """
        return {
            "throughput": self.running_tput,
            "accuracy": self.baseline_accuracy or None,
            "extra_server_args": self.args,
            "extra_envs": dict(self.envs),
            "remove_args": list(self.remove_args),
            "unset_envs": list(self.unset_envs),
            "args_mode": self.args_mode,
        }

    def launched_stack(self) -> dict[str, Any]:
        """The stack as it launches: its args with the recipe's inherited ones and the removals applied.

        A KEEP can switch the stack to ``replace`` mode, which carries those inherited args in its own, so only
        the composed args say which settings a KEEP changed.
        """
        stack = stack_of_measured(self.measured_against())
        stack["extra_server_args"] = compose_server_args(
            inherited_args=self.inherited_args,
            base_extra_args=self.args,
            remove_args=self.remove_args,
            args_mode=self.args_mode,
        )
        return stack

    def _recipe(self) -> GridRecipe:
        return GridRecipe(
            base_yaml_path=self.config_path,
            base=LaunchLayer(
                extra_server_args=self.args,
                extra_envs=dict(self.envs),
                remove_args=list(self.remove_args),
                unset_envs=list(self.unset_envs),
                args_mode=self.args_mode,
            ),
            target=self.target,
        )

    def _run_remove_args(self, gv: GridVariant) -> list[str]:
        own = to_str_list(getattr(gv, "remove_args", []))
        return own if self.args_mode == "replace" else list(dict.fromkeys(self.remove_args + own))

    # ----- the grid --------------------------------------------------------

    def _resolve_grid(self) -> None:
        payload = self.params.get("grid") or []
        if not isinstance(payload, list):
            payload = []
        self.lever_payload = framework_lever_grid(self.ss)
        payload = _lever_seeded_payload(payload, self.lever_payload)
        if not payload:
            payload = _seed_grid_payload(self.params, self.framework, self.yaml_envs)
        if not payload:
            raise _RoundRefused(
                "empty_grid",
                "explore: params.grid must be a non-empty list of variant "
                "dicts. The Orchestration prompt "
                "should fill this from specialist proposals / default_grid.",
                workspace=self.output_root.as_posix(),
            )
        grid = _grid_variants_from_payload(payload)
        if self.framework == "custom":
            grid, pinned = filter_operator_pinned_envs(grid, self.yaml_envs)
            for name, reason in pinned:
                log.warning("explore: dropping variant %s (%s)", name, reason)
        if not grid:
            raise _RoundRefused(
                "empty_grid",
                "explore: params.grid contains no valid variants",
                workspace=self.output_root.as_posix(),
            )
        # Every variant carries its fingerprint, so the result loop needn't recompute it. In-round duplicates are
        # judged at their turn, against what this round has actually launched: a twin that was only skipped (a
        # restatement of the stack) has not claimed the launch.
        for gv in grid:
            gv.canonical_fp = effective_fingerprint(  # type: ignore[attr-defined]
                gv.extra_server_args,
                gv.extra_envs,
                controls=controls_of(vars(gv)),
                base=BaseControls(self.base_remove_args, self.base_unset_envs, self.args_mode),
            )
        self.grid = grid

    def _shape_runnable(self) -> None:
        runnable, invalid = apply_multi_node_invalid_variants(list(self.grid))
        # Honour an operator-pinned SGLANG_USE_AITER=0: drop variants that would re-enable the (hang-prone) aiter
        # MoE runner.
        runnable, aiter_pinned = apply_aiter_moe_pin_filter(runnable)
        # xDiT do-not-set list, plus flags the model class does not support.
        runnable, incompatible = apply_compatibility_filter(
            runnable,
            serving=ServingStack(
                framework=self.framework,
                model_path=self.target.model_path,
                gpu_type=self.target.gpu_type or str(getattr(self.ss, "gpu_type", "") or ""),
                stack_fingerprint=getattr(self.ss, "stack_fingerprint_meta", None),
            ),
            base_server_args=compose_server_args(
                inherited_args=self.inherited_args,
                base_extra_args=self.base_args,
                remove_args=self.base_remove_args,
                args_mode=self.args_mode,
            ),
        )
        # Operator-supplied --skip-variants patterns.
        runnable, skipped = apply_user_skip_list(runnable, skip_spec=resolve_skip_spec(self.params))
        for dropped in (*invalid, *aiter_pinned, *incompatible, *skipped):
            self.skipped.append(
                {
                    "name": dropped.get("name", ""),
                    "reason": dropped.get("source", "grid_invalid"),
                    "detail": dropped.get("reason", ""),
                }
            )
        self.runnable = reorder_grid_for_multi_node(runnable)
        self._drop_leading_noops()

    def _drop_leading_noops(self) -> None:
        # Drop the variants ahead of every other one that only restate the stack's observed launch.
        #
        # They change nothing while the stack stays as observed, and can only launch on it, so they go before a
        # serving lease is taken; any other is judged at its turn, against the stack a KEEP has moved.
        before = len(self.runnable)
        if self.runnable and not is_stack_revalidation(self.params):
            kept, dropped = filter_baseline_noop_variants(
                self.runnable,
                observed=self.observed,
                recipe=GridRecipe(
                    base_yaml_path=self.config_path,
                    base=LaunchLayer(
                        extra_server_args=self.base_args,
                        extra_envs=self.base_envs,
                        remove_args=self.base_remove_args,
                        unset_envs=self.base_unset_envs,
                        args_mode=self.args_mode,
                    ),
                    target=self.target,
                ),
            )
            kept_ids = {id(gv) for gv in kept}
            leading = next((i for i, gv in enumerate(self.runnable) if id(gv) in kept_ids), before)
            for gv, (_, reason) in zip(self.runnable[:leading], dropped[:leading]):
                self._decided_without_measuring(gv, BASELINE_NOOP, detail=reason)
            self.runnable = self.runnable[leading:]
        log.info(
            "explore grid: payload=%d → runnable=%d (leading baseline_noop=%d)",
            len(self.grid),
            len(self.runnable),
            before - len(self.runnable),
        )

    def _open_schedule(self) -> None:
        # Single-node server_lifecycle eligibility (multi-node / non-builtin script / profiler-on falls back to a
        # cold decision round instead of one that re-attaches to the warmup's server).
        lifecycle = resolve_lifecycle_params(self.config_path)
        self.lifecycle_eligible = bool(lifecycle.get("eligible"))
        self.lifecycle_framework = str(lifecycle.get("framework") or "")
        self.lifecycle_port = to_int(lifecycle.get("port"), 0) or 0
        self.use_warm_decision = self.lifecycle_eligible and double_run_requested(self.params)
        # Stop testing further variants once the session wall-clock budget runs out; untested variants stay out of
        # the ledger so a resume can retry them.
        self.session_deadline_sec, session_expected_sec = session_grid_bounds(self.ss)
        # per-variant overtime kill -- anchored on baseline wall-clock, or on the measured WARM duration
        # (client-only) when this round reuses a server.
        warm_sec = to_float(self.params.get("baseline_warm_runtime_sec"), 0.0) or 0.0
        cold_sec = to_float(self.params.get("baseline_runtime_sec"), 0.0) or 0.0
        decision_sec = warm_sec if (self.use_warm_decision and warm_sec > 0) else cold_sec
        self.warmup_expected_sec = (cold_sec if cold_sec > 0 else None) or session_expected_sec
        self.decision_expected_sec = (decision_sec if decision_sec > 0 else None) or session_expected_sec
        # Round-local grading anchor. Variants stack within a round, so a KEEP advances the figure the next
        # variant is graded against; the session anchor would grade every variant against the round's opening.
        self.grade_on_intvty, _ = resolved_grading(self.ss)
        self.running_perf, anchor_reason = resolve_grading_anchor_perf(self.ss) if self.grade_on_intvty else (None, "")
        if self.grade_on_intvty and self.running_perf is None:
            # AgentX grading needs both axes on the round anchor. Without them every variant fails closed below;
            # output figures are diagnostic only and must not become throughput KEEPs.
            log.info("explore: AgentX grading unavailable for this round (%s)", anchor_reason)

    # ----- one variant -----------------------------------------------------

    def _refused(self, gv: GridVariant, idx: int) -> bool:
        if twin := self.launched.get(self._launch_identity(gv)):
            self._decided_without_measuring(gv, ROUND_DUP, duplicate_of=twin)
            return True
        if restated := self._restates_stack(gv):
            self._decided_without_measuring(gv, BASELINE_NOOP, detail=restated)
            self.judged_without_measuring += 1
            return True
        if self._already_measured(gv):
            self.judged_without_measuring += 1
            return True
        return self._out_of_budget(idx) or self._refuses_context_copy(gv)

    def _decided_without_measuring(
        self, gv: GridVariant, reason: str, *, detail: str = "", duplicate_of: str = ""
    ) -> None:
        """Record ``gv`` as answered by the stack or by a twin this round launched, not measured.

        Its ledger row marks the proposal decided, so the untested-proposal queue moves past it; a change the ledger
        already holds keeps its row, since a measurement outranks a skip.
        """
        fingerprint = _variant_fingerprint(gv)
        log.info("explore: skipping %s variant %s (%s)", reason, gv.name, detail or duplicate_of)
        self.skipped.append(
            {"name": gv.name, "fingerprint": fingerprint, "reason": reason, "detail": detail or duplicate_of}
        )
        known = (getattr(self.ss, "explore_search", None) or {}).get("tested") or {}
        if not fingerprint or fingerprint in self.tested or fingerprint in known:
            return
        self.tested[fingerprint] = {
            **self._variant_facts(gv),
            "outcome": TS_SKIPPED_DEDUP,
            "reason": reason,
            "duplicate_of": duplicate_of,
            "round_id": self.round_id,
            "ts": _now_iso(),
            "provenance": getattr(gv, "provenance", _DIRECT_PROVENANCE),
            "workload_signature": self.ws_sig,
        }

    def _launch_identity(self, gv: GridVariant) -> str:
        """The server ``gv`` would launch on the stack as it stands now; its fingerprint when that cannot be read."""
        with tempfile.TemporaryDirectory(prefix="explore_launch_") as tmp_dir:
            try:
                return _probe_launch(self._recipe(), gv, Path(tmp_dir)).identity
            except _UNREADABLE_LAUNCH as exc:
                log.warning("explore: could not read the launch of %s (%s); deduping on its fingerprint", gv.name, exc)
                return _variant_fingerprint(gv)

    def _restates_stack(self, gv: GridVariant) -> str:
        """Why ``gv`` only restates the launch the stack runs as it stands now, or ``""`` when it changes it.

        The observed launch is what the server ran when the round opened; a KEEP since then has made the observed
        value of whatever it changed stale, and ``observed_launch_after`` leaves that out. A stack revalidation
        re-measures the stack on purpose, so it never restates it.
        """
        if is_stack_revalidation(self.params):
            return ""
        _, restated = filter_baseline_noop_variants(
            [gv],
            observed=observed_launch_after(*self.observed, self.opening_stack, self.launched_stack(), self.framework),
            recipe=self._recipe(),
        )
        return restated[0][1] if restated else ""

    def _already_measured(self, gv: GridVariant) -> bool:
        """Whether the session ledger already holds a verdict for ``gv`` on the stack it would launch on.

        A KEEP earlier in the round advances that stack, so the check runs per variant rather than once for the
        round. A stack revalidation exists to re-measure, and a kernel overlay changes the launch without changing
        the fingerprint, so neither is ever skipped.
        """
        if self.ss is None or is_stack_revalidation(self.params) or getattr(gv, "overlay_pythonpath", ""):
            return False
        fingerprint = _variant_fingerprint(gv)
        prior = measured_verdict(
            getattr(self.ss, "explore_search", None),
            fingerprint,
            variant={"extra_args": gv.extra_server_args, "extra_envs": gv.extra_envs, **controls_of(vars(gv))},
            now=MeasuredOn(stack_of_measured(self.measured_against()), self.ws_sig, self.keep_threshold_pct),
        )
        if prior is None:
            return False
        detail = (
            f"measured as {prior.get('name') or fingerprint} in {prior.get('round_id') or 'an earlier round'} on "
            f"this stack: {prior.get('outcome')}"
        )
        log.info("explore: skipping already-measured variant %s (%s)", gv.name, detail)
        self.skipped.append({"name": gv.name, "fingerprint": fingerprint, "reason": ALREADY_MEASURED, "detail": detail})
        return True

    def _out_of_budget(self, idx: int) -> bool:
        # Whether the run cannot fit another variant, recording the stop when it cannot.
        #
        # A warm-decision variant pays for both rounds, so admitting it on the decision round alone would let it in
        # and then strand it mid-variant with a discarded warmup and no measurement.
        if self.decision_expected_sec is not None:
            needed = float(self.decision_expected_sec) + (
                float(self.warmup_expected_sec or 0.0) if self.use_warm_decision else 0.0
            )
        else:
            from ._subprocess_kill import resolve_benchmark_timeouts

            needed = resolve_benchmark_timeouts()[1]
        self.run_stop = variant_budget_stop(self.session_deadline_sec, self.phase_deadline, needed)
        if self.run_stop is None:
            return False
        self.run_stop_detail = self.run_stop.never_started
        self.untested = len(self.runnable) - idx
        log.warning(
            "explore: cannot fit another variant (%s, needs %.0fs); stopping after %d/%d variant(s)",
            self.run_stop.error_class,
            needed,
            idx,
            len(self.runnable),
        )
        return True

    def _refuses_context_copy(self, gv: GridVariant) -> bool:
        # Whether the synthetic benchmark refuses this variant's speculative-decoding draft, recording it.
        #
        # Judged against the stack as it stands now: a KEEP earlier in this round may have changed it.
        if not self.refuse_context_copy:
            return False
        _, refused = filter_context_copy_drafts([gv], recipe=self._recipe())
        if not refused:
            return False
        log.warning("explore: refusing variant %s (%s)", gv.name, refused[0][1])
        self.skipped.append({"name": gv.name, "reason": CONTEXT_COPY_REFUSAL, "detail": refused[0][1]})
        return True

    def _staged(self, gv: GridVariant) -> GridVariant:
        envs = dict(self.envs)
        for key in gv.unset_envs:
            envs.pop(key, None)
        envs.update(gv.extra_envs)
        run_gv = GridVariant(
            name=gv.name,
            extra_server_args=gv.extra_server_args,
            extra_envs=envs,
            note=gv.note,
            remove_args=list(gv.remove_args),
            unset_envs=list(dict.fromkeys(self.unset_envs + to_str_list(getattr(gv, "unset_envs", [])))),
            args_mode=str(getattr(gv, "args_mode", "append") or "append"),
        )
        return _carry_variant_metadata(gv, run_gv)

    def _stopped_by_the_run(self, result: Any, gv: GridVariant, idx: int, round_label: str) -> bool:
        """Whether the run stopped this round, and record it if it did."""
        stopped = stopped_by_the_run_class(getattr(result, "error_class", "") if result is not None else "")
        if stopped is None:
            return False
        self.run_stop = stopped
        self.run_stop_detail = stopped.interrupted
        self.untested = len(self.runnable) - idx
        log.warning(
            "explore: the %s round of variant %s was stopped by the run (%s); it and the "
            "%d variant(s) after it stay out of the ledger so a resume can retry them",
            round_label,
            gv.name,
            stopped.error_class,
            self.untested - 1,
        )
        return True

    async def _measure(self, gv: GridVariant, run_gv: GridVariant, idx: int, slot: Path) -> Any:
        self.launched[self._launch_identity(gv)] = _variant_fingerprint(gv)
        # The warmup and decision rounds share this slot as the lifecycle pid_dir so the decision round
        # re-attaches to the server the warmup left hot. Ray-managed GPU execution (12 T1): both reuse the
        # round-level lease, and one persistent server, so no GPU process outlives it.
        lifecycle = (
            {"cleanup": False, "pid_dir": str(slot), "port": self.lifecycle_port} if self.lifecycle_eligible else None
        )
        warmup = await self._warmup(gv, run_gv, idx, slot, lifecycle) if self.use_warm_decision else None
        if self.use_warm_decision and warmup is None:
            return None
        # The decision round is warm (it re-attaches to the warmup's hot server, client-only) when a warmup
        # preceded it, and skips eval then: it is timed against a throughput-only anchor, and
        # ``parse_eval_results`` falls back to the score the warmup took. Without a warmup there is nothing to
        # fall back to, so a cold decision round keeps its own eval.
        results = await run_grid(
            grid=[_variant_with_eval_off(run_gv) if self.use_warm_decision else run_gv],
            output_root=slot,
            recipe=self._recipe(),
            launch=GridLaunch(
                result_dir=self.result_dir,
                serving_lease=self.lease,
                server_lifecycle=lifecycle,
                server_already_ready=self.use_warm_decision,
                ready_server_log=warmup.server_log_path if warmup is not None else None,
            ),
            stop=GridStop(
                session_deadline_sec=self.session_deadline_sec, variant_expected_sec=self.decision_expected_sec
            ),
        )
        return None if self._stopped_by_the_run(results[0], gv, idx, "decision") else results[0]

    async def _warmup(self, gv: GridVariant, run_gv: GridVariant, idx: int, slot: Path, lifecycle: Any) -> Any:
        warmup_slot = slot / "warmup_round"
        warmup_slot.mkdir(parents=True, exist_ok=True)
        results = await run_grid(
            grid=[run_gv],
            output_root=warmup_slot,
            recipe=self._recipe(),
            launch=GridLaunch(
                result_dir=self.result_dir,
                serving_lease=self.lease,
                server_lifecycle=lifecycle,
                warmup_round=True,
            ),
            stop=GridStop(
                session_deadline_sec=self.session_deadline_sec, variant_expected_sec=self.warmup_expected_sec
            ),
        )
        warmup = results[0] if results else None
        if self._stopped_by_the_run(warmup, gv, idx, "warmup"):
            return None
        if getattr(warmup, "status", "") != "succeeded":
            self._warmup_failed(gv, warmup)
            return None
        return warmup

    def _warmup_failed(self, gv: GridVariant, warmup: Any) -> None:
        log.warning(
            "explore: variant %s warmup round failed (%s); skipping decision round.",
            gv.name,
            (getattr(warmup, "error", "") or "")[-200:] if warmup is not None else "no_result",
        )
        self._record(
            gv,
            {
                **self._variant_facts(gv),
                "outcome": TS_FAILED,
                "status": getattr(warmup, "status", "failed") if warmup is not None else "failed",
                "tput": None,
                "gain_pct": None,
                "base_tput": self.running_tput,
                "round_id": self.round_id,
                "ts": _now_iso(),
                "provenance": getattr(gv, "provenance", _DIRECT_PROVENANCE),
                "workload_signature": self.ws_sig,
                "framework": self.framework,
                "reason": "warmup_failed",
                "error_class": warmup.error_class if warmup is not None else "",
                "server_log_path": warmup.server_log_path if warmup is not None else None,
                "launch_evidence": dict(warmup.launch_evidence or {}) if warmup is not None else {},
                "launch_evidence_path": warmup.launch_evidence_path if warmup is not None else None,
                "stage": FAILURE_STAGE_WARMUP,
                "error_excerpt": tail_excerpt(warmup.error) if warmup is not None else None,
                "workspace": warmup.workspace if warmup is not None else None,
                "raw_result_path": warmup.raw_result_path if warmup is not None else None,
                # It launched on a stack even though it never reached a gate, so no gates are recorded and
                # nothing validated it.
                "measured_against": self.measured_against(),
            },
        )

    # ----- the verdict -----------------------------------------------------

    def _settle(self, gv: GridVariant, run_gv: GridVariant, r: Any, slot: Path) -> None:
        # Rule on this variant's measurement and fold a KEEP onto the round's stack.
        #
        # A variant KEEPs when it clears the graded verdict and the accuracy gate. The axes and the threshold floor
        # belong to resolve_graded_comparison, which every lane shares.
        variant_meas = {
            GRADED_OUTPUT: r.output_throughput,
            "input_throughput": r.input_throughput,
            "total_throughput": r.total_token_throughput,
            GRADED_INTVTY: r.intvty_p90,
            "tpot_p90_ms": r.tpot_p90_ms,
            GRADED_INTVTY_P50: r.intvty_p50,
            GRADED_DURATION: r.duration_seconds,
            GRADED_ERROR_RATE: r.request_error_rate,
            **constraint_fields(r),
        }
        stamp_output_per_gpu(variant_meas, getattr(self.ss, "tp", None))
        row = {
            **self._variant_facts(gv),
            **self._graded_verdict(gv, r, variant_meas),
            "status": r.status,
            "tput": r.output_throughput,
            **graded_axes_of(variant_meas),
            "base_tput": self.running_tput,
            "round_id": self.round_id,
            "ts": _now_iso(),
            "provenance": getattr(gv, "provenance", _DIRECT_PROVENANCE),
            "scope": str(getattr(gv, "scope", "") or ""),
            "workload_signature": self.ws_sig,
            "framework": self.framework,
            "workspace": r.workspace,
            "error_class": r.error_class or "",
            "server_log_path": r.server_log_path,
            "launch_evidence": dict(r.launch_evidence or {}),
            "launch_evidence_path": r.launch_evidence_path,
            "stage": FAILURE_STAGE_DECISION,
            "measured_against": self.measured_against(),
            "validation_basis": "",
        }
        if row["outcome"] == TS_FAILED and not row["reason"]:
            self._apply_accuracy_gate(row, slot)
        self._record(gv, row)
        if row["outcome"] == "KEEP":
            self.winners.append(self._keep(gv, run_gv, r, row, variant_meas))

    def _variant_facts(self, gv: GridVariant) -> dict[str, Any]:
        return {
            "fingerprint": _variant_fingerprint(gv),
            "name": gv.name,
            "extra_server_args": gv.extra_server_args,
            "extra_envs": dict(gv.extra_envs),
            **controls_of(vars(gv)),
            **_decision_fields(gv),
        }

    def _graded_verdict(self, gv: GridVariant, r: Any, variant_meas: dict[str, Any]) -> dict[str, Any]:
        # How the graded comparison ruled on this measurement, and which gate ended the arc.
        #
        # The gate is recorded where it is known: read off the outcome afterwards, "REVERT" cannot say which gate
        # ruled, and a gate that never ran is indistinguishable from one that ruled against the variant.
        graded = resolve_graded_comparison(
            self.ss,
            variant_meas,
            keep_threshold_pct=self.keep_threshold_pct,
            anchor=GradingAnchor(
                perf=self.running_perf,
                tput=self.running_tput,
            ),
        )
        on_intvty = graded.graded_on_intvty
        gain: float | None = None
        # The graded movement of a variant the verdict refused. ``gain`` stays None on a REVERT so no consumer
        # counts a refused variant's gain as a win, but the ledger row still has to say the axis was measured:
        # rendered with neither, a guard-refused variant reads as unmeasured.
        graded_gain: float | None = None
        outcome = TS_FAILED
        reason = ""
        if r.status != "succeeded":
            reason = (r.error or "")[-1200:] or "no_measurement"
        elif graded.degrade_reason:
            # Same fail-closed rule as ``lift_to_current_best``: an AgentX session that could not grade on
            # interactivity does not KEEP on output throughput instead.
            outcome = "FAILED"
            reason = graded.degrade_reason
            log.info("explore: variant %r not comparable (%s)", gv.name, graded.degrade_reason)
        elif graded.verdict == VERDICT_REVERT:
            outcome = "REVERT"
            graded_gain = gain_pct(graded.candidate, graded.reference) if graded.reference > 0 else None
            axes = (
                f"intvty {graded.reference:.1f}->{graded.candidate:.1f} "
                f"tput {graded.tput_reference:.1f}->{graded.tput_candidate:.1f}"
            )
            # A vetoed variant is refused a round earlier than the promotion gate would, so it is never folded
            # onto the stack and never becomes the anchor the rest of the batch is graded against.
            reason = graded.veto_reason or (f"{graded.refused_by} ({axes})" if on_intvty else GAIN_BELOW_THRESHOLD)
        else:
            gain = gain_pct(graded.candidate, graded.reference)
        gates: list[dict[str, Any]] = []
        if r.status == "succeeded":
            gates.append(
                {
                    "gate": graded.vetoed_by
                    if graded.veto_reason
                    else ("graded_axes" if (on_intvty or graded.degrade_reason) else "keep_threshold"),
                    "passed": False if graded.degrade_reason else graded.verdict != VERDICT_REVERT,
                    # The anchor is the reference; the floor the candidate has to clear belongs to the gate, as
                    # the tolerance does for accuracy.
                    "observed": graded.candidate if on_intvty else gain_pct(graded.candidate, graded.reference),
                    "threshold": graded.reference if on_intvty else self.keep_threshold_pct,
                    "reason": reason,
                }
            )
        return {
            "outcome": outcome,
            # Which gate refused the variant. The prompt ledger renders this row, and without the reason a REVERT
            # on the tail guard is indistinguishable from one the benchmark never measured.
            "reason": reason,
            "gain_pct": gain,
            "graded_gain_pct": graded_gain,
            "graded_objective": graded.objective,
            "gates": gates,
        }

    def _apply_accuracy_gate(self, row: dict[str, Any], slot: Path) -> None:
        # Rule on the round's accuracy evidence for a variant no earlier gate refused.
        #
        # A session with no baseline accuracy gates nothing, so its KEEPs rest on throughput alone -- a weaker
        # claim than one an accuracy gate ruled on, and ``validation_basis`` must not read alike for the two.
        from hyperloom.inference_optimizer import framework_registry

        scriptable = framework_registry.is_scriptable(self.framework)
        observed: Any = None
        reference: float | None = None
        passed = True
        # Serving still needs a measured baseline to compare against; scriptable compares against a fixed 1.0.
        gated = scriptable or self.baseline_accuracy > 0
        if gated:
            observed = parse_eval_results(
                slot, framework=self.framework, benchmark_mode=str(getattr(self.ss, "benchmark_mode", "") or "")
            ).get("accuracy")
            if isinstance(observed, (int, float)):
                reference = 1.0 if scriptable else self.baseline_accuracy
                passed = accuracy_passed(reference, float(observed))
            else:
                passed = False
            row["gates"].append(
                {
                    "gate": "accuracy",
                    "passed": passed,
                    "observed": observed if isinstance(observed, (int, float)) else None,
                    "threshold": reference,
                    "reason": "" if passed else ("accuracy_unavailable" if observed is None else "accuracy_drop"),
                }
            )
        row["outcome"] = "KEEP" if passed else "REVERT"
        row["validation_basis"] = ("accuracy_pass" if gated else "keep_verdict_unscored") if passed else ""
        if not passed:
            row["reason"] = "accuracy_unavailable" if observed is None else "accuracy_drop"

    # ----- the ledgers -----------------------------------------------------

    def _record(self, gv: GridVariant, row: dict[str, Any]) -> None:
        self.tested[row["fingerprint"]] = row
        if row["outcome"] != "KEEP":
            self.rejected.append({**_ledger_view(row, _REJECTED_FIELDS), **_decision_fields(gv)})
            self.losers.append(_ledger_view(row, _LOSER_FIELDS))
            return
        self.winners_history.append(
            {
                "round_id": row["round_id"],
                "variant_name": row["name"],
                "fingerprint": row["fingerprint"],
                "gain_pct": row["gain_pct"],
                "extra_args": row["extra_server_args"],
                "extra_envs": row["extra_envs"],
                **controls_of(vars(gv)),
                "provenance": row["provenance"],
                "scope": row["scope"],
                "ts": row["ts"],
            }
        )

    def _keep(self, gv: GridVariant, run_gv: GridVariant, r: Any, row: dict[str, Any], meas: dict[str, Any]) -> dict:
        control_fields = controls_of(vars(gv))
        run_remove_args = self._run_remove_args(gv)
        next_effective_args = compose_server_args(
            inherited_args=self.inherited_args,
            base_extra_args=self.args,
            variant_extra_args=gv.extra_server_args,
            remove_args=run_remove_args,
            args_mode="replace" if self.args_mode == "replace" else gv.args_mode,
        )
        next_stack_args = compose_server_args(
            base_extra_args=self.args,
            variant_extra_args=gv.extra_server_args,
            remove_args=gv.remove_args,
            args_mode=gv.args_mode,
        )
        effective = dict(control_fields)
        if run_remove_args:
            effective["remove_args"] = list(run_remove_args)
        if run_gv.unset_envs:
            effective["unset_envs"] = list(run_gv.unset_envs)
        persist = _keep_uses_effective_args(
            self.bench, self.args_mode, gv.args_mode, run_remove_args, run_gv.unset_envs
        )
        if persist:
            effective["args_mode"] = "replace"
        # The variant KEEPs on the round that graded it.
        self.args = next_effective_args if persist else next_stack_args
        self.envs = dict(run_gv.extra_envs)
        self.remove_args = list(run_remove_args)
        self.unset_envs = list(run_gv.unset_envs)
        self.args_mode = "replace" if persist else "append"
        if row["tput"] and row["tput"] > 0:
            self.running_tput = row["tput"]
        if self.grade_on_intvty:
            # Only a comparable intvty KEEP advances the round anchor. Degraded measurements never reach here.
            self.running_perf = perf_snapshot_from_mapping(meas)
        return {
            "fingerprint": row["fingerprint"],
            "name": gv.name,
            "candidate_extra_server_args": gv.extra_server_args,
            "candidate_extra_envs": dict(gv.extra_envs or {}),
            "recipe_delta": {
                "extra_server_args": gv.extra_server_args,
                "extra_envs": dict(gv.extra_envs or {}),
                **control_fields,
            },
            "extra_server_args": self.args,
            "effective_extra_server_args": next_effective_args,
            "extra_envs": dict(run_gv.extra_envs),
            **effective,
            **_decision_fields(gv),
            "provenance": row["provenance"],
            # Names of the authored kernels this config carried, when an overlay was loaded.
            "accepted_kernels": list(getattr(gv, "accepted_kernels", []) or []),
            "gain_pct": row["gain_pct"],
            "graded_objective": row["graded_objective"],
            # The verdict this KEEP rests on. ``None`` means the variant was not gated (not high-risk, or no
            # baseline) rather than that it scored nothing.
            "accuracy": next((g["observed"] for g in row["gates"] if g["gate"] == "accuracy"), None),
            "tput": row["tput"],
            # The axes this KEEP was graded on travel with it: current_best becomes the next round's anchor, and
            # an anchor without them degrades the session.
            **graded_axes_of(meas),
            # Promotion re-checks the budgets against this row, not the round's measurement.
            **constraint_fields(r),
            "single_workspace": r.workspace,
            "launch_evidence": dict(r.launch_evidence or {}),
            "launch_evidence_path": r.launch_evidence_path,
            "round_id": self.round_id,
            "accepted_at_round": self.round_id,
            "ts": _now_iso(),
        }

    def _per_variant_outcomes(self, reasons: dict[str, str]) -> list[dict[str, Any]]:
        task_id = str(self.ctx.task.task_id)
        outcomes = [
            _outcome_row(te, fingerprint, reasons.get(fingerprint, ""), task_id)
            for fingerprint, te in self.tested.items()
            if te.get("outcome") in ("KEEP", "REVERT", TS_FAILED)
        ]
        outcomes.extend(
            {
                "variant_name": str(sd.get("name") or ""),
                "outcome": TS_SKIPPED_DEDUP,
                "fingerprint": str(sd.get("fingerprint") or ""),
                "provenance": "",
                "metrics": {},
                "reason": str(sd.get("reason") or ""),
            }
            for sd in self.skipped
        )
        return outcomes

    def _result(self) -> dict[str, Any]:
        rejected = {row["fingerprint"]: row for row in self.rejected if row["fingerprint"]}
        outcomes = self._per_variant_outcomes({fp: str(row.get("reason") or "") for fp, row in rejected.items()})
        levers = _framework_lever_attributions(outcomes, self.lever_payload)
        if levers:
            log.info(
                "explore: attributed %d framework rewrite lever(s): %s",
                len(levers),
                ", ".join(f"{a['switch']}={a['gain_pct']:+.2f}%" for a in levers),
            )
        best = max(self.winners, key=lambda w: to_float(w.get("gain_pct"), 0.0) or 0.0, default=None)
        best_gain_pct = (to_float(best.get("gain_pct"), 0.0) or 0.0) if best else 0.0
        # A round whose every variant already had its verdict, or only restated the stack, is answered, as if
        # measured again.
        answered = bool(self.runnable) and self.judged_without_measuring == len(self.runnable)
        measured = any(t.get("outcome") in ("KEEP", "REVERT") for t in self.tested.values())
        status = "succeeded" if measured or self.winners or answered else "failed"
        return {
            "status": status,
            # A round that measured nothing because the run stopped it is not the same as one whose variants
            # failed. As a bare ``failed`` with no error_class nothing downstream could tell the two apart, and
            # the KB would learn that these variants are bad.
            **self._budget_error(status),
            "base_tput": self.base_tput,
            # Each KEEP advances the running anchor, so this is the final stack.
            "output_throughput": float(self.running_tput) if self.winners else None,
            "best_variant": best,
            "best_gain_pct": best_gain_pct,
            "winners": self.winners,
            "losers": self.losers,
            "skipped_dup": self.skipped,
            "per_variant_outcomes": outcomes,
            "framework_lever_attributions": levers,
            "explore_search_update": {
                "tested": self.tested,
                "rejected": list(rejected.values()),
                "winners_history": self.winners_history,
                "last_round": self._last_round_summary(),
            },
            "round_id": self.round_id,
            "workspace": self.output_root.as_posix(),
            "framework": self.framework,
            # gain_pct for the audit trail (best gain of the batch).
            "gain_pct": best_gain_pct,
            "explore_grid_exhausted": not self.runnable,
        }

    def _budget_error(self, status: str) -> dict[str, Any]:
        if status != "failed" or self.run_stop is None:
            return {}
        return {
            "error_class": self.run_stop.error_class,
            "error": (
                f"{self.run_stop_detail}; {self.untested} variant(s) went unmeasured "
                "and stay out of the ledger so a resume can retry them"
            ),
        }

    def _last_round_summary(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "base_tput": self.base_tput,
            "base_extra_args": self.base_args,
            "tested": [w["fingerprint"] for w in self.winners] + [lr["fingerprint"] for lr in self.losers],
            "round_winners": [w["fingerprint"] for w in self.winners],
            "skipped_dup": self.skipped,
            "ts": _now_iso(),
        }


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action."""

    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the explore executor and its gating thresholds."""
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = session_dir
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the merged ``explore`` action for one task."""
        params = dict(ctx.task.params or {})
        params["config_path"] = params.get("config_path") or self.default_config_path or default_baseline_config()
        params.setdefault("keep_threshold_pct", self.keep_threshold_pct)
        extra = getattr(ctx, "extra", None) or {}
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "explore", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            return await _ExploreRound(ctx, params, output_root).run()
        except _RoundRefused as refusal:
            return refusal.result


explore_executor = ExploreExecutor()


__all__ = [
    "ExploreExecutor",
    "explore_executor",
]
