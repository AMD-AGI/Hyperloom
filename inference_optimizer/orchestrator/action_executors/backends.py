"""Real ``backends`` ActionRunner — DESIGN v0.6 §16 backends action.

Mirrors marathon/skills/actions/backends.md DFS protocol:

  1. Discover backend / scheduling flags from sglang's ``server_args.py``
     (AST parse — keeps the grid in sync with whatever sglang version
     is installed; no hand-curated allowlist drifting out of date).
  2. Test each variant independently on top of the current best
     ``EXTRA_SGLANG_ARGS`` (or empty if no current best).
  3. Pick the winners (output_throughput > base_tput × 1.01).
  4. Optionally run the combined-winners run (deferred to integrate /
     P2-4 — keeps this runner's wall time bounded).

Result schema returned to the bus::

    status:           "succeeded" | "failed" | "no_winners"
    base_tput:        float — what we measured against
    grid_size:        int
    all_results:      [VariantResult.to_dict() for v in grid]
    winners:          [VariantResult.to_dict() for w in winners]   # > +1%
    best_variant:     VariantResult.to_dict() | None
    best_gain_pct:    float
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from ._grid_runner import GridVariant, VariantResult, pick_winners, run_grid, _resolve_session_dir
from ._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default seed grid — mirrors the 5-tier marathon hierarchy. Used when AST
# discovery fails or sglang isn't installed under /sgl-workspace.
DEFAULT_BACKENDS_GRID: list[GridVariant] = [
    # Tier 1: attention/decode backend switches
    GridVariant("attn_aiter",      "--attention-backend aiter",
                 note="tier1_attention"),
    GridVariant("attn_triton",     "--attention-backend triton",
                 note="tier1_attention"),
    GridVariant("decode_aiter",    "--decode-attention-backend aiter",
                 note="tier1_decode_attn"),
    # Tier 2: scheduling
    GridVariant("sched_lpm",       "--schedule-policy lpm",
                 note="tier2_schedule"),
    GridVariant("sched_dfs",       "--schedule-policy dfs-weight",
                 note="tier2_schedule"),
    GridVariant("disable_overlap", "--disable-overlap-schedule",
                 note="tier2_overlap"),
    # Tier 3: fusion
    GridVariant("enable_fused_moe","--enable-fused-moe",
                 note="tier3_fusion"),
    GridVariant("enable_mixed",    "--enable-mixed-chunk",
                 note="tier3_fusion"),
    # Tier 4: MoE/GEMM
    GridVariant("moe_aiter",       "--moe-runner-backend aiter",
                 note="tier4_moe"),
    # Tier 5: comm
    GridVariant("custom_ar",       "--enable-custom-ar",
                 note="tier5_comm"),
]

# vLLM-specific backends grid. The SGLang grid above uses flags vLLM doesn't
# recognize (--attention-backend, --schedule-policy, --enable-fused-moe, etc.)
# which causes every variant to silently fail. This grid uses flags from vLLM
# 0.17-0.20+ CLI (`vllm serve --help`). Sources:
#   - marathon_optimization/marathon_harness/skills/KNOWLEDGE-BASE.md
#   - marathon_optimization/marathon_harness/skills/actions/params.md
#   - .cursor/skills/inference-optimization/KNOWLEDGE-BASE.md (Kimi-K2.5, gpt-oss)
#
# Validated wins from marathon:
#   - gpu-memory-utilization 0.90 + max-num-seqs 256 = +84% (Kimi-K2.5)
#   - FP8 KV + max-num-seqs 512 + max-cudagraph 2048 = +4.3% (gpt-oss-120b)
#   - FULL_AND_PIECEWISE compile is ESSENTIAL for gpt-oss (enforce-eager = -85.8%)
# Tier 1: highest-impact variants, run first. These are the most likely to
# produce wins based on marathon KB validated results. ~10 variants.
_VLLM_TIER1: list[GridVariant] = [
    # Memory + KV (marathon: +84% Kimi-K2.5 from gpu-mem + max-seqs combo)
    GridVariant("vllm_kv_fp8",             "--kv-cache-dtype fp8_e4m3",
                 note="kv_cache"),
    GridVariant("vllm_gpu_mem_0_95",       "--gpu-memory-utilization 0.95",
                 note="memory"),
    GridVariant("vllm_max_seqs_512",       "--max-num-seqs 512",
                 note="scheduling"),
    # Compile (marathon: ESSENTIAL for gpt-oss, +85.8% vs enforce-eager)
    GridVariant("vllm_full_piecewise",
                 "--compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\","
                 "\"custom_ops\":[\"all\"]}'",
                 note="compile"),
    GridVariant("vllm_cudagraph_2048",     "--max-cudagraph-capture-size 2048",
                 note="cuda_graph"),
    # AITER core (most common production toggle)
    GridVariant("vllm_aiter_on",
                 extra_envs={"VLLM_ROCM_USE_AITER": "1"},
                 note="rocm_aiter"),
    # Attention backend (new, high potential)
    GridVariant("vllm_attn_aiter_fa",      "--attention-backend ROCM_AITER_FA",
                 note="attention_backend"),
    # Prefix cache off (some MoE models faster without it)
    GridVariant("vllm_no_prefix_cache",    "--no-enable-prefix-caching",
                 note="cache"),
    # Batched tokens (prefill throughput)
    GridVariant("vllm_batched_tokens_16k", "--max-num-batched-tokens 16384",
                 note="prefill"),
    # Max seqs high (decode throughput at high CONC)
    GridVariant("vllm_max_seqs_1024",      "--max-num-seqs 1024",
                 note="scheduling"),
]

# Tier 2: additional variants, run after Tier 1 if time budget allows.
_VLLM_TIER2: list[GridVariant] = [
    GridVariant("vllm_gpu_mem_0_90",       "--gpu-memory-utilization 0.90",
                 note="memory"),
    GridVariant("vllm_gpu_mem_0_92",       "--gpu-memory-utilization 0.92",
                 note="memory"),
    GridVariant("vllm_block_size_1",       "--block-size 1",
                 note="cache_mla"),
    GridVariant("vllm_max_seqs_256",       "--max-num-seqs 256",
                 note="scheduling"),
    GridVariant("vllm_max_seqs_128",       "--max-num-seqs 128",
                 note="scheduling"),
    GridVariant("vllm_batched_tokens_32k", "--max-num-batched-tokens 32768",
                 note="prefill"),
    GridVariant("vllm_compile_off",        "--enforce-eager",
                 note="compile_off"),
    GridVariant("vllm_cudagraph_512",      "--max-cudagraph-capture-size 512",
                 note="cuda_graph"),
    GridVariant("vllm_aiter_off",
                 extra_envs={"VLLM_ROCM_USE_AITER": "0"},
                 note="rocm_aiter"),
    GridVariant("vllm_aiter_linear",
                 extra_envs={"VLLM_ROCM_USE_AITER": "1",
                             "VLLM_ROCM_USE_AITER_LINEAR": "1"},
                 note="rocm_aiter_linear"),
    GridVariant("vllm_aiter_rmsnorm",
                 extra_envs={"VLLM_ROCM_USE_AITER": "1",
                             "VLLM_ROCM_USE_AITER_RMSNORM": "1"},
                 note="rocm_aiter_rmsnorm"),
    GridVariant("vllm_aiter_fp8bmm",
                 extra_envs={"VLLM_ROCM_USE_AITER": "1",
                             "VLLM_ROCM_USE_AITER_FP8BMM": "1"},
                 note="rocm_aiter_fp8bmm"),
    GridVariant("vllm_aiter_fp4_asm",
                 extra_envs={"VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1"},
                 note="rocm_fp4"),
    GridVariant("vllm_aiter_triton_rope",
                 extra_envs={"VLLM_ROCM_USE_AITER_TRITON_ROPE": "1"},
                 note="rocm_rope"),
    GridVariant("vllm_quick_reduce_int4",
                 extra_envs={"VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4"},
                 note="rocm_collectives"),
    GridVariant("vllm_buffer_ops_off",
                 extra_envs={"AMDGCN_USE_BUFFER_OPS": "0"},
                 note="rocm_buffer"),
    GridVariant("vllm_no_scratch_reclaim",
                 extra_envs={"HSA_NO_SCRATCH_RECLAIM": "1"},
                 note="rocm_scratch"),
    GridVariant("vllm_shuffle_kv_layout",
                 extra_envs={"VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1"},
                 note="rocm_kv_layout"),
]

# Full grid = Tier 1 first, then Tier 2. Tier 1 always runs; Tier 2 runs
# when the grid is not time-constrained. The BackendsExecutor runs the full
# list sequentially (Tier 1 items appear first).
DEFAULT_VLLM_BACKENDS_GRID: list[GridVariant] = _VLLM_TIER1 + _VLLM_TIER2


# ---------------------------------------------------------------------------
# Synergy groups: pairs/triples of variants that often need to be combined
# to see real gains. Phase 2 of BackendsExecutor tests these after phase 1
# identifies individual winners.
#
# Format: each entry is a list of variant names from the grid above. If ALL
# members of a group are present in the grid (even if they didn't individually
# win), we generate a combo variant that stacks them all.
#
# The trigger condition is: at least ONE member of the group won in phase 1.
# This way we don't blindly test all combos, only those where at least one
# constituent showed promise.
SYNERGY_GROUPS: list[list[str]] = [
    # FP8 KV cache unlocks FP8BMM / aiter linear / shuffle KV layout
    ["vllm_kv_fp8", "vllm_aiter_fp8bmm"],
    ["vllm_kv_fp8", "vllm_aiter_linear"],
    ["vllm_kv_fp8", "vllm_shuffle_kv_layout"],
    # Full aiter stack (common in production tomls)
    ["vllm_aiter_on", "vllm_aiter_linear", "vllm_aiter_rmsnorm"],
    ["vllm_aiter_on", "vllm_aiter_linear", "vllm_aiter_rmsnorm", "vllm_aiter_fp8bmm"],
    # aiter + kv fp8 (the most common production stack)
    ["vllm_kv_fp8", "vllm_aiter_on", "vllm_aiter_linear", "vllm_aiter_rmsnorm", "vllm_aiter_fp8bmm"],
    # Memory + scheduling (often co-beneficial)
    ["vllm_gpu_mem_0_95", "vllm_max_seqs_512"],
    ["vllm_kv_fp8", "vllm_max_seqs_512"],
    # Compile + scheduling
    ["vllm_full_piecewise", "vllm_cudagraph_2048"],
]


def _build_synergy_combos(
    grid: list[GridVariant],
    winner_names: set[str],
) -> list[GridVariant]:
    """Generate combo variants from synergy groups where at least one member won."""
    return _build_synergy_combos_from_groups(
        grid, list(SYNERGY_GROUPS), require_one_winner=winner_names,
    )


def _build_synergy_combos_from_groups(
    grid: list[GridVariant],
    groups: list[list[str]],
    *,
    require_one_winner: set[str] | None = None,
) -> list[GridVariant]:
    """Materialize combo variants from a list of name-groups.

    Used both by the static SYNERGY_GROUPS path and the LLM-injected
    ``params.synergy_groups`` path. When ``require_one_winner`` is set,
    a group is only kept if at least one of its members appears in the
    set (the original "at least one member won in phase 1" gate). Pass
    ``None`` (LLM-injected case) to skip that gate entirely — the LLM is
    explicit about what to test.
    """
    grid_by_name = {v.name: v for v in grid}
    combos: list[GridVariant] = []
    seen_combo_names: set[str] = set()

    for group in groups:
        if not group:
            continue
        if not all(name in grid_by_name for name in group):
            continue
        if (
            require_one_winner is not None
            and not any(name in require_one_winner for name in group)
        ):
            continue
        combo_name = "combo_" + "+".join(group)
        if combo_name in seen_combo_names:
            continue
        seen_combo_names.add(combo_name)

        combined_args_parts: list[str] = []
        combined_envs: dict[str, str] = {}
        for name in group:
            v = grid_by_name[name]
            if v.extra_sglang_args.strip():
                combined_args_parts.append(v.extra_sglang_args.strip())
            combined_envs.update(v.extra_envs)

        combos.append(GridVariant(
            name=combo_name,
            extra_sglang_args=" ".join(combined_args_parts),
            extra_envs=combined_envs,
            note="synergy_combo",
        ))

    return combos


def _auto_synergy_combos(
    grid: list[GridVariant],
    winner_names: set[str],
    *,
    max_combo_size: int = 3,
    max_combos: int = 8,
) -> list[GridVariant]:
    """T2 — generate combos as Cartesian products over phase-1 winners.

    Used when ``params.synergy_mode == "auto"``: the LLM doesn't have to
    enumerate group lists, the executor pairs every winner with every
    other winner up to ``max_combo_size`` members, capped at
    ``max_combos`` total to bound runtime.

    Mirrors marathon's "test combinations of all winners" step
    (marathon/skills/actions/backends.md §5).
    """
    import itertools
    if not winner_names:
        return []
    grid_by_name = {v.name: v for v in grid}
    candidates = sorted(n for n in winner_names if n in grid_by_name)
    if len(candidates) < 2:
        return []
    groups: list[list[str]] = []
    for size in range(2, max(2, min(max_combo_size, len(candidates))) + 1):
        for combo in itertools.combinations(candidates, size):
            groups.append(list(combo))
            if len(groups) >= max_combos:
                break
        if len(groups) >= max_combos:
            break
    return _build_synergy_combos_from_groups(grid, groups,
                                              require_one_winner=None)


_BACKEND_KEYWORDS = (
    "backend", "enable_", "disable_", "fused", "mixed", "overlap",
    "schedule", "allreduce", "fusion",
)

# Default search paths for AST-based flag discovery. Override via env so
# operators with non-standard installs (custom forks, pip-editable in a
# different prefix) can still get dynamic discovery without code changes.
DEFAULT_SGLANG_SERVER_ARGS = Path(
    os.environ.get(
        "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS",
        "/sgl-workspace/sglang/python/sglang/srt/server_args.py",
    )
)
DEFAULT_VLLM_ARG_UTILS = Path(
    os.environ.get(
        "INFERENCE_OPTIMIZER_VLLM_ARG_UTILS",
        "/sgl-workspace/vllm/vllm/engine/arg_utils.py",
    )
)


def discover_backend_flags(
    server_args_path: Path = DEFAULT_SGLANG_SERVER_ARGS,
) -> list[str]:
    """AST-parse sglang's server_args.py and return discovered flag names.

    Mirrors the heuristic from marathon backends.md Step 1. Returns
    sorted unique CLI flag names like ``--enable-aiter`` /
    ``--attention-backend``. No values — the caller composes those.
    Returns ``[]`` if the file is missing or unparseable.
    """
    if not server_args_path.exists():
        log.info("discover_backend_flags: %s not found, returning empty",
                  server_args_path)
        return []
    try:
        source = server_args_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        log.warning("discover_backend_flags: parse failed: %s", exc)
        return []
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and any(
            kw in node.attr for kw in _BACKEND_KEYWORDS
        ):
            # e.g. self.enable_overlap_scheduler → --enable-overlap-scheduler
            flags.add("--" + node.attr.replace("_", "-"))
    return sorted(flags)


# vLLM exposes its CLI through `EngineArgs` / `AsyncEngineArgs` add_cli_args
# helpers. AST-walking those for argparse-style add_argument calls would be
# more accurate but heavy; instead we mine attribute names that match the
# same keyword set used for SGLang. The fallback is intentionally permissive
# because vLLM versions reshuffle their classes frequently.
_VLLM_BACKEND_KEYWORDS = (
    "backend", "enable_", "disable_", "fused", "mixed",
    "compilation", "cudagraph", "kv_cache", "attention", "rocm",
    "aiter", "moe", "scheduler",
)


def discover_vllm_backend_flags(
    arg_utils_path: Path = DEFAULT_VLLM_ARG_UTILS,
) -> list[str]:
    """AST-parse vLLM's arg_utils.py and return likely backend flag names.

    Mirrors :func:`discover_backend_flags` for the vLLM stack. Returns a
    sorted list of CLI-style flags (``--kv-cache-dtype``, ``--max-num-seqs``).
    Empty when the file is missing — keeps the executor working on hosts
    that ship only one of {sglang, vllm}.
    """
    if not arg_utils_path.exists():
        log.info("discover_vllm_backend_flags: %s not found, returning empty",
                  arg_utils_path)
        return []
    try:
        source = arg_utils_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        log.warning("discover_vllm_backend_flags: parse failed: %s", exc)
        return []
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and any(
            kw in node.attr for kw in _VLLM_BACKEND_KEYWORDS
        ):
            flags.add("--" + node.attr.replace("_", "-"))
        elif isinstance(node, ast.keyword) and node.arg and any(
            kw in node.arg for kw in _VLLM_BACKEND_KEYWORDS
        ):
            flags.add("--" + node.arg.replace("_", "-"))
    return sorted(flags)


def _augment_grid_with_discovered_flags(
    base_grid: list[GridVariant],
    discovered: list[str],
    *,
    framework: str,
) -> list[GridVariant]:
    """Append GridVariant stubs for AST-discovered flags not in the base grid.

    For every discovered flag whose name doesn't already appear in any
    existing variant's args, we synthesize a single boolean-style probe
    (just the flag, no value) so the grid runner can A/B test it. Flags
    that look like enum/integer parameters (containing keywords like
    ``backend`` / ``size`` / ``num`` / ``ratio``) are skipped here because
    a bare flag is meaningless for those — the LLM is expected to inject
    explicit values via ``params.grid``. This keeps the auto-augmentation
    safe (no malformed CLI) while still expanding the search by ~5-15
    boolean toggles per session.
    """
    if not discovered:
        return list(base_grid)
    fw = (framework or "").strip().lower()
    existing_args = " ".join(v.extra_sglang_args for v in base_grid).lower()
    existing_envs = {k.lower() for v in base_grid for k in v.extra_envs}
    out: list[GridVariant] = list(base_grid)
    # Heuristic: skip flags that obviously want a value, not a bare presence.
    _NEEDS_VALUE = (
        "backend", "size", "num", "ratio", "fraction", "policy",
        "dtype", "tokens", "len", "level", "interval", "config",
        "batch", "graph", "step", "round",
    )
    appended = 0
    for flag in discovered:
        # ``flag`` is always ``--foo-bar``; convert back to attr name.
        attr = flag.lstrip("-").replace("-", "_")
        if any(kw in attr for kw in _NEEDS_VALUE):
            continue
        if flag.lower() in existing_args:
            continue
        # Avoid env-shadowed toggles (rare, but guards against duplicate
        # work when DEFAULT_VLLM_BACKENDS_GRID already exposes the same
        # behavior through extra_envs).
        if attr.upper() in existing_envs:
            continue
        out.append(GridVariant(
            name=f"{fw or 'auto'}_discovered_{attr}",
            extra_sglang_args=flag,
            note="discovered_ast",
        ))
        appended += 1
    if appended:
        log.info(
            "augmented %s grid with %d AST-discovered boolean variants "
            "(base=%d total=%d)",
            fw or "auto", appended, len(base_grid), len(out),
        )
    return out


# ---------------------------------------------------------------------------
class BackendsExecutor:
    """ActionRunner for the ``backends`` action."""

    def __init__(
        self,
        *,
        default_grid: list[GridVariant] | None = None,
        default_vllm_grid: list[GridVariant] | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        variant_timeout_sec: int = 900,
    ):
        self.default_grid = list(default_grid or DEFAULT_BACKENDS_GRID)
        self.default_vllm_grid = list(default_vllm_grid or DEFAULT_VLLM_BACKENDS_GRID)
        # None = resolve at call time from $FRAMEWORK (sglang/vllm).
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.variant_timeout_sec = variant_timeout_sec

    async def __call__(self, ctx) -> dict[str, Any]:
        params = ctx.task.params or {}
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or default_baseline_config()
        )
        if not config_path.exists():
            return {"status": "failed",
                    "error_class": "missing_config",
                    "error": f"config not found: {config_path}"}
        extra = getattr(ctx, "extra", None) or {}
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "backends", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Workload-contract materialization (single source of truth).
        # See params.py for the full rationale; same fix applies here so
        # backend variants run at the operator's actual workload (CONC/ISL/
        # OSL/TP/MAX_MODEL_LEN/PRECISION) rather than the shipped YAML's
        # smoke defaults. Idempotent when input already matches process env.
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            out_name="backends_base.with_envs.yaml",
        )

        base_extra_args = params.get("base_extra_args", "")
        base_tput = float(params.get("base_tput", 0.0))
        grid_override = params.get("grid")
        # Resolve framework once — needed for both grid selection and the
        # AST-discovery augmentation path below.
        import yaml
        with config_path.open(encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        framework = str(
            (_cfg.get("benchmark") or {}).get("framework") or ""
        ).lower()
        is_vllm = "vllm" in framework
        if grid_override:
            grid = [
                GridVariant(name=v["name"],
                            extra_sglang_args=v.get("extra_sglang_args", ""),
                            extra_envs=v.get("extra_envs", {}) or {},
                            note=v.get("note", ""))
                for v in grid_override
            ]
        else:
            # Pick the framework-appropriate grid; SGLang flags (--attention-
            # backend, --schedule-policy) are rejected by vLLM and vice versa.
            grid = (
                list(self.default_vllm_grid)
                if is_vllm
                else list(self.default_grid)
            )
            # T1: AST-discover boolean flags from the live framework's
            # server_args / arg_utils and append them as auto-probes. The
            # heuristic in `_augment_grid_with_discovered_flags` skips
            # value-typed flags (where a bare presence is meaningless),
            # so the appended set is intentionally conservative — the LLM
            # remains the source of truth for value-typed combinations
            # via ``params.grid``.
            disable_discovery = bool(params.get("disable_discovery", False))
            if not disable_discovery:
                # Resolve the discovery path through the module globals so
                # tests / operators can monkeypatch DEFAULT_*_PATH at
                # runtime without re-importing this module. (Function
                # default args bind at def-time; module attrs don't.)
                import sys as _sys
                _self_mod = _sys.modules[__name__]
                if is_vllm:
                    src_path = _self_mod.DEFAULT_VLLM_ARG_UTILS
                    discovered = discover_vllm_backend_flags(
                        arg_utils_path=src_path,
                    )
                else:
                    src_path = _self_mod.DEFAULT_SGLANG_SERVER_ARGS
                    discovered = discover_backend_flags(
                        server_args_path=src_path,
                    )
                src_path = str(src_path)
                if discovered:
                    grid = _augment_grid_with_discovered_flags(
                        grid, discovered, framework=framework or "auto",
                    )
                    self._record_discovered(
                        framework=framework or ("vllm" if is_vllm else "sglang"),
                        backend_flags=discovered,
                        source_path=src_path,
                    )
        timeout_sec = int(params.get("variant_timeout_sec",
                                       self.variant_timeout_sec))

        # `resolved_model` / `resolved_gpu` were resolved above for the
        # materialization step; reuse them here so each variant's YAML
        # overrides the legacy hardcoded model + benchmark_script fields.
        # --- Phase 1: single-variable grid ---
        results = await run_grid(
            base_yaml_path=config_path,
            base_extra_args=base_extra_args,
            grid=grid,
            output_root=output_root,
            variant_timeout_sec=timeout_sec,
            model_path=resolved_model,
            gpu_type=resolved_gpu,
        )
        winners = pick_winners(results, baseline_tput=base_tput)
        winner_names = {w.name for w in winners}

        # --- Phase 2: synergy combos ---
        # Many ROCm env toggles only show gains when combined (e.g.
        # FP8BMM needs KV fp8, aiter_linear needs aiter base). After
        # phase 1, we generate combo variants from known synergy groups
        # and test them. This avoids underestimating stacked configs.
        #
        # Three sources of synergy groups (T2):
        #   1. ``params.synergy_groups`` — LLM-injected list of name lists
        #   2. ``params.synergy_mode == "auto"`` — Cartesian over winners
        #      (capped to small N to bound runtime)
        #   3. ``SYNERGY_GROUPS`` module constant (default fallback)
        # The Coordinator-side SharedState ``synergy_attempted`` set is
        # consulted to skip combos already tested in earlier rounds.
        combo_results: list[VariantResult] = []
        groups_override = params.get("synergy_groups")
        synergy_mode = str(params.get("synergy_mode") or "").strip().lower()
        if isinstance(groups_override, list) and groups_override:
            override_groups = [
                [str(x) for x in g if isinstance(x, str)]
                for g in groups_override
                if isinstance(g, list) and g
            ]
        else:
            override_groups = None
        already_attempted: set[str] = set(
            params.get("synergy_attempted") or []
        )

        def _combo_key(names: list[str]) -> str:
            return "+".join(sorted(str(n) for n in names if n))

        new_attempts: list[list[str]] = []
        if len(winners) >= 1:
            if override_groups is not None:
                combos = _build_synergy_combos_from_groups(
                    grid, override_groups,
                )
            elif synergy_mode == "auto":
                combos = _auto_synergy_combos(
                    grid, winner_names,
                    max_combo_size=int(params.get("max_combo_size", 3)),
                    max_combos=int(params.get("max_combos", 8)),
                )
            else:
                combos = _build_synergy_combos(grid, winner_names)

            if already_attempted:
                combos = [
                    c for c in combos
                    if _combo_key(c.name.removeprefix("combo_").split("+"))
                    not in already_attempted
                ]
            if combos:
                log.info("backends phase 2: testing %d synergy combos "
                         "(mode=%s skipped_attempted=%d)",
                         len(combos),
                         synergy_mode or ("override" if override_groups else "auto-default"),
                         len(already_attempted))
                combo_results = await run_grid(
                    base_yaml_path=config_path,
                    base_extra_args=base_extra_args,
                    grid=combos,
                    output_root=output_root / "combos",
                    variant_timeout_sec=timeout_sec,
                    model_path=resolved_model,
                    gpu_type=resolved_gpu,
                )
                # Track which combos were actually tested so the next
                # round won't replay them. Coordinator persists this
                # back into SharedState via the result-dict field below.
                for c in combos:
                    new_attempts.append(
                        c.name.removeprefix("combo_").split("+")
                    )

        all_results = results + combo_results
        all_winners = pick_winners(all_results, baseline_tput=base_tput)
        best = max(
            (r for r in all_results
             if r.status == "succeeded"
             and isinstance(r.output_throughput, (int, float))),
            default=None,
            key=lambda r: r.output_throughput or 0.0,
        )
        best_gain = (
            ((best.output_throughput - base_tput) / base_tput * 100.0)
            if best and base_tput > 0 else 0.0
        )

        successful_runs = [r for r in all_results if r.status == "succeeded"]

        return {
            "status": "succeeded" if successful_runs else "failed",
            "base_tput": base_tput,
            "grid_size": len(all_results),
            "all_results": [r.to_dict() for r in all_results],
            "winners": [w.to_dict() for w in all_winners],
            "best_variant": best.to_dict() if best else None,
            "best_gain_pct": best_gain,
            "output_throughput": best.output_throughput if best else None,
            "workspace": output_root.as_posix(),
            "phase2_combos_tested": len(combo_results),
            # T1/T2 — Coordinator picks these up to update SharedState.
            "discovered_flags_update": self._pop_discovered_update(),
            "synergy_attempted_new": new_attempts,
        }

    # ------------------------------------------------------------------
    # T1/T2 — discovery-cache + SharedState plumbing
    # ------------------------------------------------------------------
    def _record_discovered(
        self,
        *,
        framework: str,
        backend_flags: list[str],
        source_path: str,
    ) -> None:
        """Stage one framework's discovered flags for the next result dict.

        The executor is stateless across invocations w.r.t. SharedState
        (the Coordinator owns state.json). We stash the latest discovery
        on a per-instance buffer that gets drained into the result dict
        by ``_pop_discovered_update``.
        """
        self._pending_discovered = {
            "framework": framework,
            "backend_flags": list(backend_flags),
            "source_path": source_path,
        }

    def _pop_discovered_update(self) -> dict[str, Any] | None:
        out = getattr(self, "_pending_discovered", None)
        self._pending_discovered = None
        return out


backends_executor = BackendsExecutor()


__all__ = [
    "DEFAULT_BACKENDS_GRID",
    "DEFAULT_SGLANG_SERVER_ARGS",
    "DEFAULT_VLLM_ARG_UTILS",
    "DEFAULT_VLLM_BACKENDS_GRID",
    "SYNERGY_GROUPS",
    "BackendsExecutor",
    "backends_executor",
    "discover_backend_flags",
    "discover_vllm_backend_flags",
]
