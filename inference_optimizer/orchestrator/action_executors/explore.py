# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ExploreExecutor — v0.8 M3.

Merges the legacy ``backends`` / ``params`` / ``validate_stack`` actions
into one unified ``explore`` action (one yaml meta, one
``SharedState.explore_search`` ledger, one executor).

Per-variant flow:

1. canonical_fingerprint dedup against ``explore_search.tested``
   (rename-resistant).
2. Render the variant's Magpie YAML, run E2E bench.
3. Immediate KEEP/REVERT decision (``DEFAULT_KEEP_THRESHOLD_PCT`` gain
   threshold + accuracy gate when ``is_high_accuracy_risk``).
4. KEEP triggers an inlined stack rebench; if the rebench tput is below
   the threshold (default baseline_tput * 1.005) the variant is evicted
   (``KEEP_UNSTABLE`` → REVERT).

Follows the TBO "one change at a time" rule (KB_design §3.4 "Inv-3
serving GPU single tenant" + §3 "iron rules"), unlike v0.6's
run-batch-then-pick-best. ``provenance`` passes through to the ledger
unchanged so the M5 specialist path can fill ``'specialist:<domain>'``.

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
from ._explore_roofline_filter import compute_saturation_advisory
from ._grid_runner import (
    _MN_BACKENDS_PRIORITY,
    _MN_PARAMS_PRIORITY,
    GridVariant,
    _resolve_session_dir,
    apply_multi_node_invalid_variants,
    reorder_grid_for_multi_node,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


# Per-variant KEEP threshold (gain-pct + accuracy gate); the inlined stack
# rebench is the second gate. Override per-task via ``params['keep_threshold_pct']``.
DEFAULT_KEEP_THRESHOLD_PCT = 1.0

# Stack rebench stability threshold: after a KEEP, rebench tput must beat
# ``base_tput * (1 + DEFAULT_STACK_STABLE_PCT/100)`` else evict
# (KEEP_UNSTABLE → REVERT). Set below the KEEP threshold so a genuine win
# losing a little headroom when layered isn't immediately evicted. Override
# via ``params['stack_stable_threshold_pct']``.
DEFAULT_STACK_STABLE_PCT = 0.5

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        str: The current UTC timestamp in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def _initial_explore_search_state() -> dict[str, Any]:
    """Empty :attr:`SharedState.explore_search` ledger (M3 schema v1).

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


def _coerce_args_str(value: Any) -> str:
    """Coerce a payload ``extra_args`` / ``extra_server_args`` value into a
    shell-arg string.

    The Orchestration/specialist LLM sometimes emits the server flags as a
    JSON list (``["--max-num-batched-tokens", "32768"]``) instead of a single
    string. A naive ``str(list)`` yields the Python repr
    (``"['--max-num-batched-tokens', '32768']"``) which Magpie's
    ``vllm serve ... $EXTRA_VLLM_ARGS`` splices verbatim, so the server rejects
    it as ``unrecognized arguments`` and every server-arg variant aborts.
    Lists/tuples are space-joined into individual tokens.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


def _grid_variants_from_payload(payload: list[Any]) -> list[GridVariant]:
    """Convert the LLM/specialist grid payload into GridVariant objects.

    Variant dict shape (M3, KB_design §3.4 §5.1):

        {
          "name": str (required, unique-in-round),
          "extra_args" | "extra_server_args": str,
          "extra_envs": dict[str,str],
          "note": str,
          "provenance": str,            # llm_direct / default_grid / specialist:<tag>
          "scope": str,                 # specialist dial: domain / domains / freeform (advisory)
          "kb_evidence": list,          # passthrough
          "pr_evidence": list,          # passthrough
          "source_evidence": list,      # passthrough
        }

    Unknown keys ignored; unstamped ``provenance`` defaults to
    ``'default_grid'`` (keeps seed grids distinct from ``'llm_direct'``).
    """
    out: list[GridVariant] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        args = _coerce_args_str(
            raw.get("extra_args") or raw.get("extra_server_args") or ""
        ).strip()
        envs_raw = raw.get("extra_envs") or {}
        envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}
        gv = GridVariant(
            name=str(raw["name"]),
            extra_server_args=args,
            extra_envs=envs,
            note=str(raw.get("note") or raw.get("provenance") or ""),
        )
        # Stash extra metadata on the GridVariant so the ledger writer can
        # pull provenance/evidence; safe on the plain dataclass.
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.scope = str(raw.get("scope") or "")                       # type: ignore[attr-defined]
        gv.kb_evidence = list(raw.get("kb_evidence") or [])         # type: ignore[attr-defined]
        gv.pr_evidence = list(raw.get("pr_evidence") or [])         # type: ignore[attr-defined]
        gv.source_evidence = list(raw.get("source_evidence") or []) # type: ignore[attr-defined]
        out.append(gv)
    return out


# Atom default grid seed: ``_atom_default_grid`` returns a curated grid from
# atom's CLI flag space; only atom has a programmatic seed today (sglang/vllm
# callers get ``[]`` and rely on LLM-emitted ``default_grid`` variants).
# Variants are gated up-front on ``model_class``; ``apply_compatibility_filter``
# (``_grid_runner.py``) is the second-line drop for flags missing from
# ``atom --help``.

# Curated MTP-capable model class set (needs multi-token-prediction heads,
# DeepSeek family today). Cross-reference atom's ``atom/model_engine/``
# before adding entries rather than guessing from the model name.
_ATOM_MTP_CAPABLE_MODEL_CLASSES: frozenset[str] = frozenset({
    "moe_mla",
    "moe_mla_nsa",
})


def _atom_default_grid(
    *,
    model_class: str,
    conc: int,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """Atom EXPLORE default grid, seeded from atom's known perf knobs.

    Covers the atom CLI surface (compile/cudagraph bracket, prefix cache,
    KV fp8, MoE EP, MLA DP-attention, MTP), each gated on model_class.
    ``apply_compatibility_filter`` is the second-line drop for flags not in
    ``atom --help``. Variant names are ``atom_``-prefixed for cross-session
    disambiguation.
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

    # ``atom_level_3`` is atom's default (atom_mi*x.sh injects ``--level 3``),
    # so a level-3 variant would A/B against itself; use ``atom_level_2`` as
    # the off-default contrast. (Level 2's ``compile_sizes is None`` crash in
    # cuda_piecewise_backend.py:54 only fires with ``--torch-profiler-dir``,
    # which EXPLORE variants don't pass.)
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


def _default_grid_for_framework(
    framework: str,
    *,
    model_class: str,
    conc: int = 0,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """Framework-keyed default grid dispatch.

    Atom returns a curated seed grid; sglang / vllm / unknown return ``[]``
    ("no programmatic seed") and rely on LLM-emitted ``default_grid`` variants.
    """
    fw = (framework or "").strip().lower()
    if fw == "atom":
        return _atom_default_grid(
            model_class=model_class, conc=conc, isl=isl, osl=osl,
        )
    return []


def _gain_pct(tput: float | None, base_tput: float) -> float | None:
    """Compute the percentage throughput gain over a baseline.

    Args:
        tput (float | None): The variant's throughput.
        base_tput (float): The baseline throughput to compare against.

    Returns:
        float | None: The gain as a percentage, or ``None`` when either
        input is non-positive or ``tput`` is not numeric.
    """
    if (
        not isinstance(tput, (int, float))
        or tput <= 0
        or base_tput <= 0
    ):
        return None
    return (float(tput) - base_tput) / base_tput * 100.0


# Auto-derived per-variant hard timeout: rather than a universal constant
# (the legacy 2400s smoke floor times out slow workloads like Qwen3-32B TP=1
# ~70 min baselines), derive the cap from the Coordinator-injected measured
# baseline runtime plus a safety margin above the soft-kill ratio (preserves
# soft-kill → hard-cap layering). Override per-task via
# ``params['variant_timeout_sec']`` or ``--explore-variant-timeout-sec``;
# floor/ceiling guard pathological inputs.
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


def _join_args(*parts: str) -> str:
    """Join non-empty, stripped argument fragments with single spaces.

    Args:
        *parts (str): Argument fragments; empty/whitespace ones are
            dropped.

    Returns:
        str: The space-joined argument string.
    """
    return " ".join(p.strip() for p in parts if p and p.strip())


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action.

    Subsumes the retired v0.6 ``backends`` / ``params`` / ``validate_stack``
    executors via per-variant KEEP/REVERT gating + inlined per-KEEP stack rebench.
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
            stack_stable_threshold_pct (float): Stability band for the
                stack rebench. Defaults to :data:`DEFAULT_STACK_STABLE_PCT`.
            enable_stack_rebench (bool): Whether to run the inlined
                per-KEEP stack rebench. Defaults to ``True``.
        """
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)
        self.stack_stable_threshold_pct = float(stack_stable_threshold_pct)
        # Stack-rebench (default on) can be disabled for tests / fast smoke runs.
        self.enable_stack_rebench = bool(enable_stack_rebench)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the merged ``explore`` action for one task.

        Resolves the benchmark config and output workspace, builds the
        candidate grid (programmatic seed and/or LLM/specialist variants),
        benchmarks each variant with per-variant KEEP/REVERT gating, and
        optionally performs an inlined per-KEEP stack rebench.

        Args:
            ctx: The action runner context carrying the task and params.

        Returns:
            dict[str, Any]: The explore result payload (status plus the
            accepted/rejected variants and ledger updates), or a failure
            dict on error.
        """
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
        # Re-materialize so variant YAMLs honour the operator's actual
        # workload (CONC / ISL / OSL / TP / MAX_MODEL_LEN / PRECISION).
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
        base_extra_args = str(params.get("base_extra_args") or "").strip()
        base_extra_envs = dict(params.get("base_extra_envs") or {})
        base_tput = float(params.get("base_tput") or 0.0)
        # Backstop: when params carries no positive ``base_tput``, recover the
        # comparison anchor from live SharedState (else every ``_gain_pct``
        # returns None and real wins get marked FAILED). Prefer running best,
        # fall back to the original baseline.
        if base_tput <= 0:
            ss = extra.get("shared_state") or extra.get("state")
            if ss is not None:
                cb = getattr(ss, "current_best", None) or {}
                cb_tput = cb.get("tput") if isinstance(cb, dict) else None
                if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                    base_tput = float(cb_tput)
                else:
                    base_tput = float(
                        getattr(ss, "baseline_tput", 0.0) or 0.0
                    )
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

        # per-variant overtime kill — anchored on baseline wall-clock,
        # single-variant runs only (not stack_rebench). Coordinator injects
        # ``baseline_runtime_sec`` + ``explore_overtime_kill_ratio``; if
        # either is missing the deadline stays None and only the
        # ``variant_timeout_sec`` hard cap gates.
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

        # Per-variant hard cap precedence: explicit
        # ``params['variant_timeout_sec']`` → auto-derive from baseline
        # runtime + kill ratio (see ``_compute_explore_variant_timeout``) →
        # ``self.variant_timeout_sec`` floor (no baseline yet).
        explicit_timeout = params.get("variant_timeout_sec")
        if explicit_timeout is not None:
            timeout_sec = int(explicit_timeout)
        else:
            # Operator-tunable headroom; unset uses the helper default,
            # negative clamps to 0 (hard cap collapses onto the soft kill).
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

        # Resolve framework from materialized YAML (for the ledger + the
        # atom seed-grid fallback below).
        try:
            with config_path.open(encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f) or {}
            framework = str(
                (_cfg.get("benchmark") or {}).get("framework") or ""
            ).lower()
            # Pull CONC so the seed grid's cudagraph-bracket variant brackets
            # the live decode concurrency.
            _yaml_envs = (_cfg.get("benchmark") or {}).get("envs") or {}
        except Exception:  # noqa: BLE001
            framework = ""
            _yaml_envs = {}

        # ----- Variant grid ------------------------------------------------
        grid_payload = params.get("grid") or []
        if not isinstance(grid_payload, list) or not grid_payload:
            # No LLM variants: fall through to the framework's programmatic
            # seed grid (stamped ``provenance='default_grid'``) instead of
            # failing the task.
            seed_model_class = (
                str(params.get("model_class") or "").strip()
                or os.environ.get("MODEL_CLASS", "").strip()
            )
            seed_conc = 0
            try:
                _conc_raw = (
                    params.get("conc")
                    or _yaml_envs.get("CONC")
                    or os.environ.get("CONC")
                    or 0
                )
                seed_conc = int(_conc_raw)
            except (TypeError, ValueError):
                seed_conc = 0
            seed_isl = 0
            seed_osl = 0
            try:
                seed_isl = int(
                    params.get("isl")
                    or _yaml_envs.get("ISL")
                    or os.environ.get("ISL")
                    or 0
                )
                seed_osl = int(
                    params.get("osl")
                    or _yaml_envs.get("OSL")
                    or os.environ.get("OSL")
                    or 0
                )
            except (TypeError, ValueError):
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
                    framework or "?", len(seed),
                    seed_model_class or "?", seed_conc,
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
            """Resolve a ledger entry's variant fingerprint.

            Args:
                entry (Any): A ledger entry; expected to be a dict with
                    a ``fingerprint`` or server-args/envs to derive one.

            Returns:
                str: The stored or canonically-derived fingerprint, or
                ``""`` when ``entry`` is not a dict.
            """
            if not isinstance(entry, dict):
                return ""
            fp = entry.get("fingerprint")
            if fp:
                return str(fp)
            return canonical_fingerprint(
                str(entry.get("extra_server_args") or ""),
                dict(entry.get("extra_envs") or {}),
            )

        # Conditional dedup. KEEP'd variants are permanently blocked (never
        # re-proposed). A variant that ran but did not promote (REVERT /
        # KEEP_UNSTABLE / no_promote) only stays blocked while its prior measured
        # gain is below the current KEEP bar; once the (decaying) bar drops to or
        # below that gain the variant unblocks so a later cycle can re-test it.
        # Infra failures (KILLED_OVERTIME / FAILED) stay blocked regardless of
        # gain. Unblocking only lifts the hard skip — the variant still re-runs
        # the full KEEP + stack-rebench gate, so a stale measurement can't
        # promote on its own.
        gain_unlockable = {"REVERT", "KEEP_UNSTABLE", "no_promote"}

        def _is_blocked(entry: Any) -> bool:
            if not isinstance(entry, dict):
                return True
            if str(entry.get("outcome") or "") in gain_unlockable:
                try:
                    prior_gain = float(entry.get("gain_pct"))
                except (TypeError, ValueError):
                    return True
                return prior_gain < keep_threshold_pct
            return True

        tested_dict = search.get("tested") or {}
        seen_fps: set[str] = set()
        unlocked_reference: list[dict[str, Any]] = []
        for fp_key, v in tested_dict.items():
            if _is_blocked(v):
                seen_fps.add(str(fp_key))
                seen_fps.add(_entry_fp(v))
            elif isinstance(v, dict):
                unlocked_reference.append(v)
        # accepted == KEEP'd: always blocked.
        for v in search.get("accepted") or []:
            seen_fps.add(_entry_fp(v))
        for v in search.get("rejected") or []:
            if _is_blocked(v):
                seen_fps.add(_entry_fp(v))
            elif isinstance(v, dict):
                unlocked_reference.append(v)
        seen_fps.discard("")
        if unlocked_reference:
            log.info(
                "explore: %d prior sub-threshold variant(s) unblocked at "
                "keep_threshold=%.3f%% for re-test",
                len(unlocked_reference), keep_threshold_pct,
            )
        name_index = dict(search.get("name_index") or {})

        # Attach the per-variant fingerprint as an attribute so the result
        # loop needn't recompute (content-only hash, same as canonical).
        ws_sig = workload_signature()

        unique_in_round: dict[str, GridVariant] = {}
        skipped_dup: list[dict[str, Any]] = []
        for gv in grid:
            fp = canonical_fingerprint(gv.extra_server_args, gv.extra_envs)
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

        # Roofline saturation advisory (annotates, never drops): flags
        # variants targeting only already-saturated directions so the
        # Orchestration prompt can reprioritise without code dropping work.
        roofline_advisory: list[dict[str, Any]] = []
        saturation_snapshot = params.get("roofline_saturation_snapshot")
        if isinstance(saturation_snapshot, dict) and saturation_snapshot and runnable:
            roofline_advisory = compute_saturation_advisory(
                runnable, saturation_snapshot,
            )
            if roofline_advisory:
                log.info(
                    "explore roofline advisory: %d/%d variants flagged "
                    "as likely_saturated (saturated=%s)",
                    len(roofline_advisory),
                    len(runnable),
                    ",".join(sorted(
                        d for d, p in saturation_snapshot.items()
                        if isinstance(p, (int, float)) and float(p) >= 80.0
                    )),
                )

        # Multi-node grid shaping (companion to the roofline gate above).
        # Both helpers short-circuit on ``is_multi_node() is False``: the
        # invalid filter returns ``(list(grid), [])`` and reorder preserves the
        # original order, so the single-node ``runnable`` is bit-for-bit
        # identical and ``skipped_dup`` gains nothing — single-node behaviour is
        # never altered (hard requirement). In multi-node mode: drop
        # known-regression variants (cuda-graph-max-bs < CONC), then surface
        # likely-winners first so a max-hours cut still benches the strong
        # candidates. (apply_single_node_invalid_variants is intentionally NOT
        # called here: it would mutate the single-node grid.)
        if runnable:
            runnable, _mn_dropped = apply_multi_node_invalid_variants(runnable)
            for _d in _mn_dropped:
                skipped_dup.append({
                    "name": _d.get("name", ""),
                    "reason": _d.get("source", "grid_invalid"),
                    "detail": _d.get("reason", ""),
                })
            runnable = reorder_grid_for_multi_node(
                runnable,
                priority_tags=_MN_PARAMS_PRIORITY + _MN_BACKENDS_PRIORITY,
            )

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
        # accumulation; after a KEEP they extend with the KEEP'd variant so
        # the next variant is benched on the freshest stack.
        stack_extra_args = base_extra_args
        stack_extra_envs = dict(base_extra_envs)
        running_base_tput = base_tput
        # In-batch KEEP'd entries (for full vs incremental stack recompose).
        in_batch_keeps: list[dict[str, Any]] = []

        last_run_tput: float | None = None  # rebench/single-variant tput

        if runnable:
            for idx, gv in enumerate(runnable):
                fp = getattr(gv, "canonical_fp", "")
                provenance = getattr(gv, "provenance", "llm_direct")
                scope = str(getattr(gv, "scope", "") or "")
                slot = output_root / f"v{idx:02d}_{_safe(gv.name)}"
                slot.mkdir(parents=True, exist_ok=True)
                # 1. Run the single variant on the running stack.
                #    ``soft_deadline_sec`` is the overtime kill; stack_rebench
                #    below intentionally omits it (Q4).
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
                    # Defensive — run_grid returns one result per grid entry.
                    log.warning("explore: variant %s produced no result", gv.name)
                    continue
                r = results[0]

                # Overtime gate fired: record a ``KILLED_OVERTIME`` row with
                # runtime_sec + wall_clock_ratio (no faked tput/gain, per Q3),
                # skip all downstream gates + dedup, leave the stack unadvanced.
                if getattr(r, "killed_overtime", False):
                    variant_runtime = float(r.runtime_sec or 0.0)
                    wall_clock_ratio = (
                        round(variant_runtime / baseline_runtime_sec, 3)
                        if baseline_runtime_sec > 0 else None
                    )
                    tested_update[fp] = {
                        "fingerprint": fp,
                        "name": gv.name,
                        "extra_server_args": gv.extra_server_args,
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
                        "extra_server_args": gv.extra_server_args,
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
                        "extra_server_args": gv.extra_server_args,
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

                gain = _gain_pct(r.output_throughput, running_base_tput)
                outcome = "FAILED"
                reason: str = ""
                if r.status != "succeeded" or gain is None:
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
                            extra_args=gv.extra_server_args,
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
                            # No eval result: follow the legacy "no accuracy
                            # data => skip the gate" convention so high-risk
                            # flags aren't auto-rejected on a benign gap.
                            accuracy_ok = True
                    if not accuracy_ok:
                        outcome = "REVERT"
                        reason = "accuracy_drop"
                    else:
                        outcome = "KEEP"

                tested_update[fp] = {
                    "fingerprint": fp,
                    "name": gv.name,
                    "extra_server_args": gv.extra_server_args,
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
                        "extra_server_args": gv.extra_server_args,
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
                    next_args = _join_args(stack_extra_args, gv.extra_server_args)
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
                            extra_server_args="",  # all args in next_args
                            extra_envs={},          # all envs in next_envs
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
                        # Apply the in-batch env stack to the rebench step
                        # (run_grid only sees envs via variant.extra_envs),
                        # re-running when the merged envs aren't present.
                        if rb is not None and next_envs:
                            if rb.extra_envs != next_envs:
                                rebench_variant_envs = GridVariant(
                                    name=f"{gv.name}__stack_rebench_envs",
                                    extra_server_args="",
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
                        # KEEP_UNSTABLE: rebench missed the stability floor —
                        # evict the KEEP and treat as REVERT.
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
                                "extra_server_args": gv.extra_server_args,
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
                            # Roll the stack back to the prior accumulation.
                            in_batch_keeps.pop()
                            continue
                        else:
                            # Stable — fold the variant onto the stack so the
                            # next variant benches against the new baseline.
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
                        # Stack rebench disabled — KEEP on the single-variant
                        # measurement, advance the running baseline naively.
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
                        "extra_args": gv.extra_server_args,
                        "extra_envs": dict(gv.extra_envs),
                        "provenance": provenance,
                        "scope": scope,
                        "ts": _now_iso(),
                    })
                    continue

                # ---- REVERT / FAILED ----
                rejected_update.append({
                    "fingerprint": fp,
                    "name": gv.name,
                    "extra_server_args": gv.extra_server_args,
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
                    "extra_server_args": gv.extra_server_args,
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

        # Flat per-variant outcomes for the Coordinator's per-variant
        # fact-write hook (KEEP / REVERT / FAILED / KEEP_UNSTABLE /
        # SKIPPED_DEDUP from this round). JSON-friendly for older readers.
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
            # surface wall-clock + kill ratio so the LLM/KB sees "ran too
            # slow → early kill" instead of an opaque FAILED row.
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

        # Successful = at least one bench produced a measurement (KEEP /
        # REVERT / KEEP_UNSTABLE) or was reaped by the overtime gate
        # (KILLED_OVERTIME is a real signal the LLM needs to see).
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
            "roofline_advisory": roofline_advisory,
            # flat per-variant outcomes.
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
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_STACK_STABLE_PCT",
    "ExploreExecutor",
    "explore_executor",
]
