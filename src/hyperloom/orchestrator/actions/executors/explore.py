# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor.

The unified ``explore`` action (one yaml meta, one
``SharedState.explore_search`` ledger, one executor).

Per-variant flow:

1. canonical_fingerprint dedup within the submitted grid only; historical
   ``explore_search`` results are evidence, never an eligibility gate.
2. Render the variant's Magpie YAML, run E2E bench.
3. Immediate KEEP/REVERT decision (``DEFAULT_KEEP_THRESHOLD_PCT`` gain
   threshold + accuracy gate on every variant that has a reference).

Follows the "one change at a time" rule (single-tenant serving GPU).
``provenance`` passes through to the ledger unchanged so the specialist
path can fill ``'specialist:<domain>'``.

The result payload is consumed by writeback's ``explore`` promote branch; see
the ``return`` at the end of ``__call__`` for the authoritative field set.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...state.failure_evidence import (
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    make_failure_id,
    tail_excerpt,
)
from ...state.shared_state import first_positive_tput, resolve_anchor_with_drift, stack_base_params
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
    """Empty :attr:`SharedState.explore_search` ledger.

    Returns:
        dict[str, Any]: A fresh explore-search ledger with all sections
        initialized to their empty defaults.
    """
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


# Audit/provenance metadata stashed on a GridVariant that must survive being
# rebuilt into a derived variant.
_CARRIED_VARIANT_ATTRS: tuple[str, ...] = (
    "provenance",
    "scope",
    "overlay_pythonpath",
    "accepted_kernels",
    "kb_evidence",
    "pr_evidence",
    "source_evidence",
)


def _carry_variant_metadata(src: Any, dst: Any) -> Any:
    """Copy the carried audit metadata from ``src`` onto ``dst``.

    Args:
        src: The variant to read metadata from.
        dst: The derived variant to stamp.

    Returns:
        ``dst``, for chaining.
    """
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
    """Convert the LLM/specialist grid payload into GridVariant objects.

    Variant dict shape:

        {
          "name": str (required, unique-in-round),
          "extra_args" | "extra_server_args": str,
          "extra_envs": dict[str,str],
          "remove_args": list[str],      # inherited/base flags to remove
          "unset_envs": list[str],       # inherited env keys to remove
          "args_mode": "append"|"replace",
          "note": str,
          "provenance": str,            # llm_direct / default_grid / specialist:<tag>
          "scope": str,                 # specialist dial: domain / domains / freeform (advisory)
          "kb_evidence": list,          # passthrough
          "pr_evidence": list,          # passthrough
          "source_evidence": list,      # passthrough
        }

    Unknown keys ignored; unstamped ``provenance`` defaults to
    ``'default_grid'`` (keeps seed grids distinct from ``'llm_direct'``).

    Args:
        payload: List of variant dicts from the LLM/specialist grid.

    Returns:
        The parsed ``GridVariant`` objects (entries without a name skipped).
    """
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
        # Stash extra metadata on the GridVariant so the ledger writer can
        # pull provenance/evidence.
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.scope = str(raw.get("scope") or "")  # type: ignore[attr-defined]
        # Authored-kernel overlay dir (PYTHONPATH prefix); "" for env/flag
        # variants. Consumed by _grid_runner._build_variant_yaml.
        gv.overlay_pythonpath = str(raw.get("overlay_pythonpath") or "")  # type: ignore[attr-defined]
        # Authored kernels this variant's overlay installs. Carried so the
        # decision row names the kernel instead of inheriting the flag string as
        # its whole identity; empty for every flag/env-only variant.
        gv.accepted_kernels = [  # type: ignore[attr-defined]
            str(k).strip() for k in (raw.get("accepted_kernels") or []) if str(k).strip()
        ]
        gv.kb_evidence = list(raw.get("kb_evidence") or [])  # type: ignore[attr-defined]
        gv.pr_evidence = list(raw.get("pr_evidence") or [])  # type: ignore[attr-defined]
        gv.source_evidence = list(raw.get("source_evidence") or [])  # type: ignore[attr-defined]
        # Framework-rewrite lever this variant attributes to, and how. Carried so
        # the round can report each rewrite's own contribution instead of only
        # whether the variant won.
        gv.framework_lever = str(raw.get("framework_lever") or "")  # type: ignore[attr-defined]
        gv.framework_lever_source = str(raw.get("framework_lever_source") or "")  # type: ignore[attr-defined]
        out.append(gv)
    return out


def framework_lever_grid(shared_state: Any) -> list[dict[str, Any]]:
    """Build explore variants that attribute each registered rewrite lever.

    Registered levers are the switches behind framework-level source rewrites
    that were accepted by ``integrate_patch``. They arrive in one of two states,
    and each needs the opposite experiment:

    * **dormant** (the authored bundle passed correctness but not the throughput
      threshold, so the code is applied with every switch off) — switch one lever
      plus its dependency closure ON and see what it adds;
    * **on** (the bundle cleared the gate and its switches are part of the running
      configuration) — switch one lever plus everything that depends on it OFF and
      see what the stack loses.

    The closure is what makes either experiment meaningful. An enabler measured
    alone shows nothing, and a dependent measured without its enabler shows the
    cost of a broken configuration rather than the lever's contribution.

    Levers that already carry an attribution are skipped, so a later round spends
    its legs on something new.

    Args:
        shared_state: The live SharedState (duck-typed; ``None`` yields ``[]``).

    Returns:
        Variant payload dicts ready for :func:`_grid_variants_from_payload`.
    """
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
    """Derive each rewrite lever's own contribution from this round's outcomes.

    The sign convention differs by experiment, and getting it wrong would invert
    every verdict:

    * an **additive** variant switched the lever on, so the measured gain *is* its
      contribution;
    * a **leave-one-out** variant switched it off, so its contribution is the
      negation of the measured gain — a stack that drops 8% without a lever means
      that lever was worth about 8%.

    Only variants naming a single primary lever are attributed. A whole-stack
    combination variant carries no primary lever and is a combination test, not an
    attribution of any one rewrite.

    Args:
        per_variant_outcomes: This round's per-variant outcome rows.
        lever_payload: The lever variants seeded into this round.

    Returns:
        ``{switch, gain_pct, source, variant_name, outcome}`` rows.
    """
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
# Cross-reference atom's ``atom/model_engine/`` before adding entries.
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
    """Atom default explore grid, seeded from atom's known perf knobs.

    Covers the atom CLI surface (compile/cudagraph bracket, prefix cache,
    KV fp8, MoE EP, MLA DP-attention, MTP), each gated on model_class.
    ``apply_compatibility_filter`` drops any that the installed ``atom`` does
    not list in its ``--help``. Variant names are ``atom_``-prefixed for
    cross-session disambiguation.

    Args:
        model_class: Model-class label that gates which variants are emitted.
        conc: Live concurrency used to bracket cudagraph capture sizes.
        isl: Input sequence length (reserved for future gating).
        osl: Output sequence length (reserved for future gating).

    Returns:
        The curated list of atom ``GridVariant`` seeds.
    """
    mc_l = (model_class or "").strip().lower()
    is_moe = "moe" in mc_l
    is_mla = "mla" in mc_l
    is_fp8 = "fp8" in mc_l or mc_l.endswith("_fp8")
    is_mtp_capable = mc_l in _ATOM_MTP_CAPABLE_MODEL_CLASSES

    variants: list[GridVariant] = []

    def _add(name: str, args: str) -> None:
        """Append a ``default_grid``-provenance variant to the grid.

        Args:
            name (str): Unique variant name (``atom_`` prefixed).
            args (str): The extra server args for the variant.

        Returns:
            None: Appends to the enclosing ``variants`` list.
        """
        gv = GridVariant(
            name=name,
            extra_server_args=args,
            extra_envs={},
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # ``atom_level_3`` is atom's default, so use ``atom_level_2`` as the
    # off-default contrast.
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
        # Bracket the live concurrency so cudagraph captures the actual
        # decode batch sizes the workload spends most of its time at.
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
    """xDiT (diffusion) default explore grid, seeded from the empirical KB.

    Only BF16-safe knobs are emitted (precision is locked). Known-regression /
    crash knobs are omitted here, and ``xdit_blacklist_reason`` drops them from
    a proposal that reintroduces one. Variant names are ``xdit_``-prefixed for
    cross-session disambiguation.

    Args:
        model_class: Model-class label (reserved for future DiT gating).
        conc: Live concurrency (unused for diffusion; kept for signature parity).
        isl: Input sequence length (unused; signature parity).
        osl: Output sequence length (unused; signature parity).

    Returns:
        The curated list of xDiT ``GridVariant`` seeds.
    """
    variants: list[GridVariant] = []

    def _add(name: str, *, envs: dict[str, str]) -> None:
        """Append a ``default_grid``-provenance env-only variant.

        Args:
            name (str): Unique variant name (``xdit_`` prefixed).
            envs (dict[str, str]): The per-variant env overrides.

        Returns:
            None: Appends to the enclosing ``variants`` list.
        """
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
    """Whether a variant replays an already-validated config verbatim.

    Such a variant carries the workload spec (resolution / frames / steps) as
    part of the config being reproduced, so the off-spec filter — which reads
    those keys as an attempt to move the spec — must not judge it.
    """
    return str(getattr(variant, "provenance", "") or "").strip() in _CONFIG_REPLAY_PROVENANCE


def filter_operator_pinned_envs(
    grid: list[GridVariant],
    baseline_envs: dict[str, Any] | None,
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that overwrite an env the operator pinned in the baseline.

    A pinned env is part of what the headline number means. Overwriting one does
    not produce a faster configuration of the same workload, it produces a
    different measurement wearing the baseline's name — and the quality gate
    cannot catch it, because changing a repeat count or a warmup depth leaves
    the output identical. Adding a key the baseline never set is exactly what
    exploration is for, so only overwrites are refused.

    This is for ``custom`` alone. A shipped framework's pinned value does not
    mean "locked" — several pin a knob at its off value precisely so explore can
    flip it — and those frameworks declare what is genuinely immutable in their
    own blacklist. An operator has no blacklist to write in, so their pins carry
    the stricter reading; it is the one place they can say "this must not move".

    Args:
        grid: Candidate variants for this round.
        baseline_envs: ``benchmark.envs`` from the materialized baseline config.

    Returns:
        A ``(kept, dropped)`` tuple, where ``dropped`` holds
        ``(variant_name, reason)`` pairs for logging.
    """
    pinned = {str(k).strip().upper() for k in (baseline_envs or {}) if str(k).strip()}
    if not pinned:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    for gv in grid:
        clash = sorted(
            key for key in (str(k).strip().upper() for k in (getattr(gv, "extra_envs", None) or {})) if key in pinned
        )
        # A replay reproduces a config another component already measured, so it
        # carries the pinned values verbatim by construction.
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
    """Framework-keyed default grid dispatch.

    Atom and xDiT return curated seed grids; sglang / vllm / unknown
    return ``[]`` ("no programmatic seed") and rely on LLM-emitted
    ``default_grid`` variants.

    Args:
        framework: Inference framework name to dispatch on.
        model_class: Model-class label forwarded to the seed grid builder.
        conc: Live concurrency forwarded to the seed grid builder.
        isl: Input sequence length forwarded to the seed grid builder.
        osl: Output sequence length forwarded to the seed grid builder.

    Returns:
        The framework's default ``GridVariant`` seeds, or ``[]`` when none.
    """
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


# Auto-derived per-variant hard timeout: derive the cap from the
# Coordinator-injected measured baseline runtime plus a safety margin above the
# soft-kill ratio (preserves soft-kill → hard-cap layering). Override per-task
# via ``params['variant_timeout_sec']``; floor/ceiling guard pathological inputs.
DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC = 2400  # 40 min
DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC = 14400  # 4 h — roofline composite budget
DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN = 0.5  # hard cap ≥ baseline × (kill_ratio + 0.5)
# AgentX ceiling. A measured round (35B / conc 64 / 3600s window) is ~111 min,
# so the stock 4h ceiling would clamp the hard cap under the soft kill and
# invert the layering. 8h keeps the ordering intact for baselines up to ~2.3h at
# the default kill ratio, which covers the models this mode targets.
AGENTX_EXPLORE_TIMEOUT_CEILING_SEC = 28800  # 8 h


def _compute_explore_variant_timeout(
    baseline_runtime_sec: float,
    kill_ratio: float,
    *,
    floor_sec: int = DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC,
    ceiling_sec: int = DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
    safety_margin: float = DEFAULT_EXPLORE_TIMEOUT_SAFETY_MARGIN,
) -> int:
    """Derive the per-variant hard timeout from the measured baseline.

    Returns ``floor_sec`` when ``baseline_runtime_sec`` is non-positive;
    otherwise scales with the workload runtime. The hard cap stays **above**
    the soft kill ratio (the catastrophic backstop sits above the designed
    "slower than baseline" bound); inverting them defeats the layering.

    Args:
        baseline_runtime_sec: Measured baseline wall-clock (Coordinator-
            injected). Non-positive forces the ``floor_sec`` fallback.
        kill_ratio: ``--explore-overtime-kill-ratio``; clamped to ≥1.0 so the
            derived timeout never underflows the soft kill.
        floor_sec: Lower bound (default 40 min smoke-workload behaviour).
        ceiling_sec: Upper bound (default 4 h, roofline composite timeout).
        safety_margin: Additive margin on ``kill_ratio`` keeping the hard cap
            above the soft kill (~50% headroom for one-off variant cold starts).
    """
    if baseline_runtime_sec <= 0:
        return int(floor_sec)
    effective_kill_ratio = max(1.0, float(kill_ratio))
    derived = float(baseline_runtime_sec) * (effective_kill_ratio + float(safety_margin))
    return int(max(floor_sec, min(ceiling_sec, derived)))


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action.

    Per-variant KEEP/REVERT gating over a running optimization stack.
    """

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        variant_timeout_sec: int = 2400,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the explore executor and its gating thresholds.

        Args:
            default_config_path (Path | str | None): Fallback benchmark
                config path; resolved from defaults when ``None``.
            session_dir (Path | str | None): Session output directory;
                auto-resolved when ``None``.
            variant_timeout_sec (int): Legacy per-variant hard timeout
                floor. Defaults to ``2400``.
            keep_threshold_pct (float): Minimum gain to KEEP a variant.
                Defaults to :data:`DEFAULT_KEEP_THRESHOLD_PCT`.
        """
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the merged ``explore`` action for one task.

        Resolves the benchmark config and output workspace, builds the
        candidate grid (programmatic seed and/or LLM/specialist variants),
        and benchmarks each variant with per-variant KEEP/REVERT gating.

        Args:
            ctx: The action runner context carrying the task and params.

        Returns:
            dict[str, Any]: The explore result payload (status plus the
            accepted/rejected variants and ledger updates), or a failure
            dict on error.
        """
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
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "explore", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # ----- Workload-contract materialization ---------------------------
        # Re-materialize so variant YAMLs honour the operator's actual
        # workload (CONC / ISL / OSL / TP / MAX_MODEL_LEN / PRECISION).
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
                out_name="explore_base.with_envs.yaml",
            )
        except FrameworkScriptMismatchError as exc:
            return {
                "status": "failed",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
            }

        # ----- Inputs ------------------------------------------------------
        # Params snapshot the anchor and the stack it was measured on together;
        # a KEEP landing while this task queued invalidates both, so refresh them
        # as a pair. Revalidation reproduces the saved stack, so it never re-reads.
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
            # Only current_best carries args; a baseline_tput anchor leaves the
            # params stack (seeded from the baseline record) authoritative.
            if first_positive_tput(cb) > 0:
                params.update(stack_base_params(cb))
        base_extra_args = str(params.get("base_extra_args") or "").strip()
        base_extra_envs = dict(params.get("base_extra_envs") or {})
        base_remove_args = to_str_list(params.get("base_remove_args"))
        base_unset_envs = to_str_list(params.get("base_unset_envs"))
        base_args_mode = str(params.get("base_args_mode") or "append").strip().lower()
        base_tput = float(params.get("base_tput") or 0.0)
        baseline_accuracy = float(params.get("accuracy_baseline") or 0.0) or float(
            params.get("baseline_accuracy") or 0.0
        )
        if baseline_accuracy <= 0 and ss is not None:
            baseline_accuracy = float(getattr(ss, "baseline_accuracy", 0.0) or 0.0)
        keep_threshold_pct = float(
            params.get(
                "keep_threshold_pct",
                self.keep_threshold_pct,
            )
        )

        # per-variant overtime kill — anchored on baseline wall-clock.
        # Coordinator injects ``baseline_runtime_sec`` +
        # ``explore_overtime_kill_ratio``; if either is missing the deadline
        # stays None and only the ``variant_timeout_sec`` hard cap gates.
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
        # WARM measure-round anchor (client-only). When warm-decision is active
        # the overtime kill anchors on this; falls back to the cold baseline.
        baseline_warm_runtime_sec_raw = params.get("baseline_warm_runtime_sec")
        try:
            baseline_warm_runtime_sec = (
                float(baseline_warm_runtime_sec_raw) if baseline_warm_runtime_sec_raw is not None else 0.0
            )
        except (TypeError, ValueError):
            baseline_warm_runtime_sec = 0.0
        # Per-variant hard cap precedence: explicit
        # ``params['variant_timeout_sec']`` → auto-derive from baseline
        # runtime + kill ratio (see ``_compute_explore_variant_timeout``) →
        # ``self.variant_timeout_sec`` floor (no baseline yet).
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
            # The stock 4h ceiling assumes a synthetic round measured in
            # minutes. An AgentX round is a fixed measurement window plus corpus
            # load, per-lane warmup and drain -- measured at ~111 min for a 35B
            # model at conc 64 -- so the ceiling clamps the hard cap BELOW the
            # soft kill and inverts the layering this function documents: the
            # generic timeout fires first and the round is recorded as a plain
            # timeout instead of KILLED_OVERTIME with its diagnostic ratio.
            # Raise the ceiling (not the kill ratio) so the ordering holds for
            # the long baselines AgentX produces.
            _ceiling = AGENTX_EXPLORE_TIMEOUT_CEILING_SEC if agentx_enabled() else DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC
            timeout_sec = _compute_explore_variant_timeout(
                baseline_runtime_sec=baseline_runtime_sec,
                kill_ratio=overtime_kill_ratio,
                floor_sec=int(self.variant_timeout_sec),
                ceiling_sec=_ceiling,
                safety_margin=safety_margin,
            )

        # Resolve framework from materialized YAML (for the ledger + the
        # atom seed-grid fallback below).
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
        # Framework-rewrite levers first. Each one is a measured, already-applied
        # source rewrite awaiting its individual number, so it is both cheaper to
        # judge and better evidenced than a proposed config knob. Prepending also
        # means an LLM-supplied grid does not crowd the attribution out of the
        # round's budget.
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
            # No LLM variants: fall through to the framework's programmatic
            # seed grid instead of failing the task.
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
                    "should fill this from specialist proposals / "
                    "SharedState.discovered_flags / default_grid."
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
            ("synergy_attempted", []),
            ("discovered_flags", []),
            ("domains_round_summary", []),
        ):
            search.setdefault(key, default)

        tested_dict = search.get("tested") or {}
        inherited_name_index: dict[str, Any] = dict(search.get("name_index") or {})

        # Attach the per-variant fingerprint as an attribute so the result
        # loop needn't recompute.
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
        log.info(
            "explore dedup: payload=%d → runnable=%d (round_dup=%d)",
            len(grid),
            len(runnable),
            len(skipped_dup),
        )

        # Multi-node grid shaping. Both helpers short-circuit in single-node
        # mode, leaving ``runnable`` bit-for-bit identical. In multi-node mode:
        # drop known-regression variants (cuda-graph-max-bs < CONC), then
        # surface likely-winners first so a max-hours cut still benches the
        # strong candidates.
        if runnable:
            runnable, _mn_dropped = apply_multi_node_invalid_variants(runnable)
            # Honour an operator-pinned SGLANG_USE_AITER=0: drop variants that
            # would re-enable the (hang-prone) aiter MoE runner. No-op unless
            # the pin is set. Self-gates, so safe to run in any mode.
            runnable, _aiter_dropped = apply_aiter_moe_pin_filter(runnable)
            # xDiT do-not-set list, plus flags the model class or the installed
            # server does not support. Proposals reach here unfiltered.
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
        # This round's own ledger writes, kept apart from the ledger it inherited
        # and merged over it once the loop is done. A fingerprint may be re-run
        # across rounds, so writing straight into the merged dict would lose the
        # earlier round's measured row under this round's write.
        round_tested: dict[str, dict[str, Any]] = {}
        round_name_index: dict[str, Any] = {}
        rejected_update: list[dict[str, Any]] = list(search.get("rejected") or [])
        winners_history_update: list[dict[str, Any]] = list(search.get("winners_history") or [])

        # ``stack_extra_args`` / ``stack_extra_envs`` carry the running
        # accumulation; after a KEEP they extend with the KEEP'd variant.
        stack_extra_args = base_extra_args
        stack_extra_envs = dict(base_extra_envs)
        stack_remove_args = list(dict.fromkeys(base_remove_args))
        stack_unset_envs = list(dict.fromkeys(base_unset_envs))
        stack_base_args_mode = base_args_mode
        running_base_tput = base_tput

        # Single-node server_lifecycle eligibility (multi-node / non-builtin
        # script / profiler-on falls back to a cold decision round instead of
        # one that re-attaches to the warmup's server).
        lifecycle = resolve_lifecycle_params(config_path)
        lifecycle_eligible = bool(lifecycle.get("eligible"))
        lifecycle_framework = str(lifecycle.get("framework") or "")
        lifecycle_port = int(lifecycle.get("port") or 0)

        # Warm-decision mode. Run a discarded cold warmup round first so the
        # decision round reuses the hot server (client-only) and is measured
        # warm — apples-to-apples with ``baseline_tput``. Mirrors both conjuncts
        # the baseline gates its cold+hot double-run on, so the two sides measure
        # hot together or cold together: a session that opted out of the double
        # run has a COLD ``baseline_tput`` and must be graded cold.
        use_warm_decision = lifecycle_eligible and bool(getattr(ss, "baseline_double_run", True))
        # Decision-round overtime anchor: the WARM measure time when warm-decision
        # is active and available, else the cold baseline wall-clock (legacy).
        decision_anchor_sec = (
            baseline_warm_runtime_sec if (use_warm_decision and baseline_warm_runtime_sec > 0) else baseline_runtime_sec
        )
        # The soft deadline is anchored on the warm client-only measure time and
        # enforced from the server-ready marker, so both the measured runtime and
        # this anchor exclude cold boot / warmup.
        if decision_anchor_sec > 0 and overtime_kill_ratio > 0:
            decision_deadline_sec: float | None = decision_anchor_sec * overtime_kill_ratio
        else:
            decision_deadline_sec = None

        # One Ray serving lease (actor) spans the WHOLE round; every variant
        # reuses it. The per-variant server is still (re)booted via run_grid and
        # reaped by teardown_lifecycle_server (a driver-side pgid kill, which is
        # raylet-free) between variants, so switching server args never churns a
        # Ray worker — only the actor's child server restarts inside the same
        # long-lived worker. The lease/actor is closed exactly once at round end
        # (the ``finally`` after the loop) instead of per variant: the old
        # per-variant ``ray.kill`` made raylet reap a heavyweight GPU worker on
        # every variant, which destabilised the single-node raylet and took the
        # whole session down with it.
        round_serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path)) if runnable else None
        # Stop testing further variants once the session wall-clock budget runs
        # out; untested variants stay out of the ledger so a resume can retry them.
        #
        # What a normally-behaving round needs, as opposed to ``timeout_sec``,
        # which is the catastrophic backstop (``baseline x (kill_ratio + margin)``
        # ~= baseline x 2). Gating on the backstop abandons the tail of the budget
        # to variants that would have finished comfortably: with a 20-min baseline
        # it refuses to start with 30 min left, for a round that needs ~20. The
        # params values win over the session's because an operator may override
        # them per task; ``None`` when neither is known, which leaves the stricter
        # backstop check in place rather than guessing.
        #
        # The two rounds are estimated separately because they cost different
        # amounts: the warmup pass pays a cold server boot and is discarded, while
        # the decision round is client-only against the hot server -- which is
        # exactly the split ``decision_anchor_sec`` already draws for the overtime
        # kill.
        session_deadline_sec, session_expected_sec = session_grid_bounds(
            extra.get("shared_state") or extra.get("state")
        )
        warmup_expected_sec = (baseline_runtime_sec if baseline_runtime_sec > 0 else None) or session_expected_sec
        decision_expected_sec = (decision_anchor_sec if decision_anchor_sec > 0 else None) or session_expected_sec
        # Set when the loop stops because the run stopped it -- the budget ran
        # out, or the orchestrator cancelled the action -- so the round can say
        # so instead of reporting a bare, unattributed failure: a variant that
        # never ran is not a variant that failed. ``run_stop_detail`` is the
        # lead clause, which differs by whether a round was already under way.
        run_stop: StoppedByTheRun | None = None
        run_stop_detail = ""
        session_budget_untested = 0

        def _stopped_by_the_run(result: Any, *, variant: GridVariant, idx: int, round_label: str) -> bool:
            """Whether the run stopped this round, and record it if it did.

            A reaped round measured nothing, so the variant is left out of every
            ledger exactly as an unadmitted one is: writing it as ``FAILED``
            would make a resume skip a variant nothing ever measured, and would
            teach the KB that these knobs are bad because a clock ran out.

            Args:
                result: The round's :class:`VariantResult`, or ``None``.
                variant: The variant the round was measuring, for the log line.
                idx: Its index in ``runnable``, for the untested count.
                round_label: Which round was stopped, for the log line.

            Returns:
                bool: ``True`` when the caller must stop testing variants.
            """
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
                # A warm-decision variant pays for both rounds, so admitting it on
                # the decision round alone would let it in and then strand it
                # mid-variant with a discarded warmup and no measurement.
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
                # The decision round is timed against a throughput-only anchor, so
                # it measures throughput only: the warmup round already evaluated
                # accuracy and ``parse_eval_results`` falls back to that score.
                # Without a warmup there is nothing to fall back to, so the
                # decision round keeps its own eval.
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
                # The warmup and decision rounds share this slot as the
                # lifecycle pid_dir so the decision round re-attaches to the
                # server the warmup left hot.
                variant_lifecycle = (
                    {"cleanup": False, "pid_dir": str(slot), "port": lifecycle_port} if lifecycle_eligible else None
                )
                # Ray-managed GPU execution (§12 T1): reuse the round-level Ray
                # lease (actor) for this variant's warmup and decision
                # rounds; they reuse one persistent server, so no GPU
                # process outlives the lease. ``None`` on the local path keeps the
                # legacy behaviour. The actor is NOT closed per variant — only its
                # child server is reaped in the ``finally`` below (raylet-free);
                # the lease/actor is released once at round end.
                variant_lease = round_serving_lease
                try:
                    # Warm-decision warmup round. Boot the variant's server once
                    # and DISCARD the cold measurement so the decision round runs
                    # warm / client-only. cleanup=false keeps the server hot; no
                    # soft_deadline (only the hard variant_timeout cap gates the
                    # warmup, so a one-time cold boot doesn't trip the kill).
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
                                "stage": FAILURE_STAGE_WARMUP,
                                "error_excerpt": tail_excerpt(w.error) if w is not None else None,
                                "workspace": w.workspace if w is not None else None,
                                "raw_result_path": w.raw_result_path if w is not None else None,
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
                    # Decision round: warm (re-attaches to the warmup's hot
                    # server, client-only) when ``use_warm_decision``, otherwise a
                    # fresh cold boot. It is the round the variant is graded on.
                    # ``soft_deadline_sec`` is the overtime kill.
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

                    # Overtime gate fired: record a ``KILLED_OVERTIME`` row (no
                    # faked tput/gain), skip downstream gates, leave the stack
                    # unadvanced.
                    if getattr(r, "killed_overtime", False):
                        variant_runtime = float(r.runtime_sec or 0.0)
                        wall_clock_ratio = (
                            round(variant_runtime / decision_anchor_sec, 3) if decision_anchor_sec > 0 else None
                        )
                        # Rough output tok/s salvaged from partial server.log.
                        # Informational only: ``tput`` stays None so this never
                        # enters winner selection or gain math.
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

                    # Decision-round gain is the gate: a variant KEEPs when it
                    # clears keep_threshold and the accuracy gate.
                    gain = gain_pct(r.output_throughput, running_base_tput)
                    outcome = "FAILED"
                    reason: str = ""
                    if r.status != "succeeded" or gain is None:
                        reason = (r.error or "")[-1200:] or "no_measurement"
                    elif gain < keep_threshold_pct:
                        outcome = "REVERT"
                        reason = "gain_below_threshold"
                    else:
                        # Accuracy gate. Every variant is gated: the eval runs on
                        # every warmup round regardless, so the score is already
                        # on disk and the flag catalogue that used to decide
                        # whether to read it only discarded numbers already paid
                        # for -- and missed atom's precision knobs entirely. For
                        # scriptable frameworks the image-quality gate is the
                        # sole correctness signal, so a missing gate fails closed.
                        from hyperloom.inference_optimizer import framework_registry

                        scriptable = framework_registry.is_scriptable(framework)
                        accuracy_ok = True
                        accuracy_value: float | None = None
                        # Serving still needs a measured baseline to compare
                        # against; scriptable compares against a fixed 1.0.
                        if scriptable or baseline_accuracy > 0:
                            eval_out = parse_eval_results(slot, framework=framework)
                            accuracy_value = eval_out.get("accuracy")
                            if isinstance(accuracy_value, (int, float)):
                                # Scriptable maps gate pass→1.0 / fail→0.0, so
                                # compare against a perfect reference (1.0);
                                # serving compares vs the measured baseline.
                                reference = 1.0 if scriptable else baseline_accuracy
                                accuracy_ok = accuracy_passed(
                                    reference,
                                    float(accuracy_value),
                                )
                            else:
                                # No eval result. Both scriptable and serving
                                # fail closed: a gated variant (scriptable, or a
                                # serving variant with a baseline) that yields no
                                # accuracy verdict likely broke the eval
                                # path, so the change is reverted. The former
                                # serving throughput-only skip is removed. Baseline
                                # is where a missing accuracy result halts the run;
                                # post-baseline it is a per-variant REVERT.
                                accuracy_ok = False
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
                        "gain_pct": gain,
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
                        "stage": FAILURE_STAGE_DECISION,
                    }
                    if gv.name:
                        round_name_index[gv.name] = fp

                    # ---- KEEP path ----
                    if outcome == "KEEP":
                        # Layer onto the running stack. For
                        # removal variants, next_args/next_envs are the
                        # effective launch config that must persist if the KEEP
                        # survives; gv.extra_* remain only the candidate delta.
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
                            # Names of the authored kernels this config carried,
                            # when an overlay was loaded. Empty for a flags-only
                            # variant, so a downstream reader can tell a config
                            # gain from a gain that also had a kernel running.
                            "accepted_kernels": list(getattr(gv, "accepted_kernels", []) or []),
                            "gain_pct": gain,
                            # The verdict this KEEP rests on. ``None`` means the
                            # variant was not gated (not high-risk, or no
                            # baseline to compare against) rather than that it
                            # scored nothing — without it the ledger cannot say
                            # afterwards whether a kept config was ever checked.
                            "accuracy": accuracy_value,
                            "tput": decision_tput,
                            "decision_tput": decision_tput,
                            "single_workspace": r.workspace,
                            "round_id": round_id,
                            "accepted_at_round": round_id,
                            "ts": _now_iso(),
                        }
                        # The variant KEEPs on the round that graded it. Folding
                        # it onto the stack advances the anchor the next in-batch
                        # variant is graded against.
                        stack_extra_args = next_effective_args if persist_effective_args else next_stack_args
                        stack_extra_envs = next_envs
                        stack_remove_args = list(run_remove_args)
                        stack_unset_envs = list(run_unset_envs)
                        stack_base_args_mode = "replace" if persist_effective_args else "append"
                        if decision_tput and decision_tput > 0:
                            running_base_tput = decision_tput

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
                    # Reap THIS variant's persistent server on every exit path
                    # (idempotent + no-op when reuse was ineligible). This is a
                    # driver-side pgid kill (raylet-free), so the next variant
                    # boots a fresh server inside the SAME long-lived actor. The
                    # Ray lease/actor itself is released once at round end (§4.2:
                    # the server is always reaped before the lease is dropped).
                    if lifecycle_eligible:
                        teardown_lifecycle_server(
                            pid_dir=slot,
                            framework=lifecycle_framework,
                            port=lifecycle_port,
                        )
        finally:
            # Release the round's Ray serving lease/actor exactly once (this was
            # a per-variant ``ray.kill`` before — the raylet worker churn that
            # destabilised the single-node cluster).
            if round_serving_lease is not None:
                round_serving_lease.close()

        # ----- Ledger compaction (per-fingerprint last-wins) ----------------
        # This round's writes over the ledger it inherited: a re-run fingerprint
        # replaces its earlier row, which is what a fresh measurement means, and a
        # variant this round rolled back leaves the earlier row standing.
        tested_update: dict[str, dict[str, Any]] = {**tested_dict, **round_tested}
        name_index: dict[str, Any] = {**inherited_name_index, **round_name_index}
        rejected_dedup: dict[str, dict[str, Any]] = {}
        for entry in rejected_update:
            fp = str(entry.get("fingerprint") or "")
            if not fp:
                continue
            rejected_dedup[fp] = entry

        # Flat per-variant outcomes for the Coordinator's per-variant
        # fact-write hook (this round's outcomes).
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
            # Rough decode tput salvaged from a killed-overtime variant's
            # partial server.log. Informational only (no ``tput``/gain).
            if te.get("estimated_output_throughput") is not None:
                metrics["estimated_output_throughput"] = te.get(
                    "estimated_output_throughput",
                )
            # Surface wall-clock + kill ratio so the LLM/KB sees "ran too slow
            # → early kill" instead of an opaque FAILED row.
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
                    # Carry the variant knobs so the journal's
                    # ``classify_change_kind`` can classify the change kind.
                    "variant": {
                        "name": str(te.get("name") or ""),
                        "extra_server_args": str(te.get("extra_server_args") or ""),
                        "extra_envs": dict(te.get("extra_envs") or {}),
                        "note": str(te.get("note") or ""),
                    },
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

        # Each KEEP advances ``running_base_tput``, so this is the final stack.
        output_throughput = float(running_base_tput) if winners else None

        # Successful = at least one bench produced a measurement or was reaped
        # by the overtime gate (KILLED_OVERTIME is a real signal).
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
        # A round that measured nothing because the run stopped it is not the
        # same as one whose variants failed, and it used to be reported as a bare
        # ``failed`` with no error_class at all -- nothing downstream could tell
        # the two apart, so the KB could learn that these variants are bad.
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
            "discovered_flags_update": None,
            "round_id": round_id,
            "workspace": output_root.as_posix(),
            "framework": framework,
            # gain_pct for the audit trail (best gain of the batch).
            "gain_pct": best_gain_pct,
            "explore_grid_exhausted": not runnable,
        }


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names.

    Args:
        name (str): The raw variant name.

    Returns:
        str: A slug with non-alphanumeric characters replaced by ``_``,
        truncated to 60 characters.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


explore_executor = ExploreExecutor()


__all__ = [
    "ExploreExecutor",
    "explore_executor",
]
