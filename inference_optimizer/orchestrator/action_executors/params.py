"""Real ``params`` ActionRunner — DESIGN v0.6 §16 params action.

Mirrors marathon/skills/actions/params.md PARAM_GRID, with additional SGLang
runtime toggles learned from InferenceX's validated launch recipes:

  * cuda-graph-max-bs $CONC
  * num-continuous-decode-steps {8,16,32}
  * mem-fraction-static 0.85 / 0.90
  * schedule-conservativeness 0.5
  * chunked-prefill-size 65536
  * radix/cache, tokenizer, stream interval, and ROCm/TileLang env toggles

Plus an optional NCCL_GRID via ``extra_envs`` (NCCL_MIN_NCHANNELS / NCCL_ALGO).

The runner now follows a round-based incremental search:

* test a bounded batch of single candidates against the same current base
* combine the positive candidates and re-benchmark the combination
* persist accepted/rejected/tested state so resume continues at the next
  untested candidate instead of replaying old work
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from ...session_paths import runs_dir
from ._grid_runner import GridVariant, VariantResult, run_grid, _resolve_session_dir
from ._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)
from .backends import (
    DEFAULT_SGLANG_SERVER_ARGS,
    DEFAULT_VLLM_ARG_UTILS,
)


log = logging.getLogger(__name__)


# T1 — keywords identifying server-parameter-style attrs in framework
# server_args.py / arg_utils.py. These are intentionally narrower than
# backends._BACKEND_KEYWORDS: backends toggles dispatch paths (boolean
# flags), params tune scheduling / memory / batching values (typed args
# the LLM should fill in via params.grid). The set below was distilled
# from marathon/skills/actions/params.md PARAM_GRID + KNOWLEDGE-BASE
# server-parameter reference.
_PARAM_KEYWORDS = (
    "max_num", "max_running", "max_prefill", "max_seq",
    "cuda_graph", "cudagraph",
    "decode_steps", "continuous_decode",
    "mem_fraction", "memory_utilization", "gpu_memory",
    "chunked_prefill", "schedule_conservativeness", "schedule_policy",
    "tokenizer_worker", "stream_interval",
    "block_size", "kv_cache_dtype",
    "max_cudagraph", "compilation_config",
    "radix_cache", "prefix_cach",
)


def discover_param_flags(
    *,
    framework: str = "sglang",
    server_args_path: Path | None = None,
) -> list[str]:
    """AST-discover server-parameter flags from the live framework source.

    Mirrors :func:`backends.discover_backend_flags` for parameter-tuning
    attrs. Returns CLI-style flag names (``--cuda-graph-max-bs``,
    ``--max-num-seqs``, ...) — values are NOT inferred; the caller / LLM
    fills them in via ``params.grid`` injection. Returns ``[]`` when the
    source file is missing.

    ``framework`` accepts ``"sglang"`` (default) or ``"vllm"``; pass
    ``server_args_path`` to override the default search path entirely.
    """
    if server_args_path is None:
        fw = (framework or "sglang").strip().lower()
        server_args_path = (
            DEFAULT_VLLM_ARG_UTILS if "vllm" in fw
            else DEFAULT_SGLANG_SERVER_ARGS
        )
    if not server_args_path.exists():
        log.info("discover_param_flags: %s not found, returning empty",
                  server_args_path)
        return []
    try:
        source = server_args_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        log.warning("discover_param_flags: parse failed: %s", exc)
        return []
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and any(
            kw in node.attr for kw in _PARAM_KEYWORDS
        ):
            flags.add("--" + node.attr.replace("_", "-"))
        elif isinstance(node, ast.keyword) and node.arg and any(
            kw in node.arg for kw in _PARAM_KEYWORDS
        ):
            flags.add("--" + node.arg.replace("_", "-"))
    return sorted(flags)


# ---------------------------------------------------------------------------
DEFAULT_PARAMS_GRID: list[GridVariant] = [
    GridVariant("cuda_graph_max_bs_8",       "--cuda-graph-max-bs 8",
                 note="cuda_graph"),
    # KB/marathon: cuda graph coverage is often the biggest server-param lever.
    # Keep multiple powers-of-two-ish caps because the best value depends on
    # target CONC and available HBM; repeated flags are okay because SGLang
    # argparse uses the last value.
    GridVariant("cuda_graph_max_bs_16",      "--cuda-graph-max-bs 16",
                 note="cuda_graph"),
    GridVariant("cuda_graph_max_bs_32",      "--cuda-graph-max-bs 32",
                 note="cuda_graph"),
    GridVariant("cuda_graph_max_bs_64",      "--cuda-graph-max-bs 64",
                 note="cuda_graph"),
    GridVariant("decode_steps_8",             "--num-continuous-decode-steps 8",
                 note="decode_steps"),
    GridVariant("decode_steps_16",            "--num-continuous-decode-steps 16",
                 note="decode_steps"),
    GridVariant("decode_steps_32",            "--num-continuous-decode-steps 32",
                 note="decode_steps"),
    GridVariant("mem_fraction_0_85",          "--mem-fraction-static 0.85",
                 note="memory"),
    GridVariant("mem_fraction_0_90",          "--mem-fraction-static 0.90",
                 note="memory"),
    GridVariant("mem_fraction_0_80",          "--mem-fraction-static 0.80",
                 note="memory"),
    GridVariant("schedule_conservativeness_0_5",
                 "--schedule-conservativeness 0.5",
                 note="scheduling"),
    GridVariant("schedule_conservativeness_0_3",
                 "--schedule-conservativeness 0.3",
                 note="scheduling"),
    GridVariant("schedule_conservativeness_0_7",
                 "--schedule-conservativeness 0.7",
                 note="scheduling"),
    GridVariant("schedule_conservativeness_1_0",
                 "--schedule-conservativeness 1.0",
                 note="scheduling"),
    GridVariant("chunked_prefill_32k",        "--chunked-prefill-size 32768",
                 note="prefill"),
    GridVariant("chunked_prefill_64k",        "--chunked-prefill-size 65536",
                 note="prefill"),
    GridVariant("chunked_prefill_128k",       "--chunked-prefill-size 131072",
                 note="prefill"),
    GridVariant("max_prefill_tokens_32k",     "--max-prefill-tokens 32768",
                 note="prefill"),
    GridVariant("max_prefill_tokens_64k",     "--max-prefill-tokens 65536",
                 note="prefill"),
    # InferenceX SGLang recipes often disable radix cache for large MoE
    # throughput runs; keep it as a candidate rather than a global default.
    GridVariant("disable_radix_cache",         "--disable-radix-cache",
                 note="cache"),
    GridVariant("tokenizer_workers_8",         "--tokenizer-worker-num 8",
                 note="tokenizer"),
    GridVariant("tokenizer_workers_16",        "--tokenizer-worker-num 16",
                 note="tokenizer"),
    GridVariant("stream_interval_30",          "--stream-interval 30",
                 note="streaming"),
    GridVariant("stream_interval_50",          "--stream-interval 50",
                 note="streaming"),
    GridVariant("max_running_requests_128",    "--max-running-requests 128",
                 note="scheduling"),
    GridVariant("max_running_requests_256",    "--max-running-requests 256",
                 note="scheduling"),
    # Env toggles are injected through Magpie benchmark.envs. Unsupported
    # SGLang versions should ignore them; the A/B run decides whether to keep.
    GridVariant("sglang_multi_stream_overlap",
                 extra_envs={"SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1"},
                 note="overlap"),
    GridVariant("sglang_flashmla_tilelang",
                 extra_envs={"SGLANG_HACK_FLASHMLA_BACKEND": "tilelang"},
                 note="attention"),
    GridVariant("sglang_tilelang_indexer",
                 extra_envs={"SGLANG_OPT_USE_TILELANG_INDEXER": "true"},
                 note="indexer"),
]


DEFAULT_VLLM_PARAMS_GRID: list[GridVariant] = [
    GridVariant("vllm_kv_cache_fp8",           "--kv-cache-dtype fp8",
                 note="kv_cache"),
    GridVariant("vllm_block_size_256",         "--block-size 256",
                 note="cache"),
    GridVariant("vllm_no_prefix_cache",        "--no-enable-prefix-caching",
                 note="cache"),
    GridVariant("vllm_cudagraph_capture_512",
                 "--max-cudagraph-capture-size 512",
                 note="cuda_graph"),
    GridVariant("vllm_cudagraph_capture_2048",
                 "--max-cudagraph-capture-size 2048",
                 note="cuda_graph"),
    GridVariant(
        "vllm_full_piecewise_compile",
        "--compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\","
        "\"custom_ops\":[\"all\"]}'",
        note="compile",
    ),
    GridVariant("vllm_fp4_indexer_cache",
                 "--attention_config.use_fp4_indexer_cache=True",
                 note="indexer"),
    GridVariant("vllm_gpu_mem_0_90",           "--gpu-memory-utilization 0.90",
                 note="memory"),
    GridVariant("vllm_gpu_mem_0_95",           "--gpu-memory-utilization 0.95",
                 note="memory"),
]


# NCCL grid — applied via env vars rather than CLI flags.
DEFAULT_NCCL_GRID: list[GridVariant] = [
    GridVariant("nccl_min_nchannels_32",
                 extra_envs={"NCCL_MIN_NCHANNELS": "32"},
                 note="collectives"),
    GridVariant("nccl_algo_ring",
                 extra_envs={"NCCL_ALGO": "Ring"},
                 note="collectives"),
    GridVariant("nccl_algo_tree",
                 extra_envs={"NCCL_ALGO": "Tree"},
                 note="collectives"),
]


# ---------------------------------------------------------------------------
def _variant_to_dict(v: GridVariant) -> dict[str, Any]:
    return {
        "name": v.name,
        "extra_sglang_args": v.extra_sglang_args,
        "extra_envs": dict(v.extra_envs),
        "note": v.note,
    }


def _dict_to_variant(v: dict[str, Any]) -> GridVariant:
    return GridVariant(
        name=str(v["name"]),
        extra_sglang_args=str(v.get("extra_sglang_args", "") or ""),
        extra_envs=dict(v.get("extra_envs", {}) or {}),
        note=str(v.get("note", "") or ""),
    )


def _join_args(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _merge_envs(variants: list[GridVariant]) -> dict[str, str]:
    envs: dict[str, str] = {}
    for v in variants:
        envs.update({str(k): str(val) for k, val in v.extra_envs.items()})
    return envs


def _config_framework(config_path: Path) -> str:
    try:
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return ""
    return str((cfg.get("benchmark") or {}).get("framework") or "").lower()


def _result_gain(tput: float | None, base_tput: float) -> float | None:
    if not isinstance(tput, (int, float)) or tput <= 0 or base_tput <= 0:
        return None
    return (float(tput) - base_tput) / base_tput * 100.0


def _result_with_effective_args(
    result: VariantResult,
    *,
    base_extra_args: str,
    variant_args: str,
) -> dict[str, Any]:
    d = result.to_dict()
    d["extra_sglang_args"] = _join_args(base_extra_args, variant_args)
    d["candidate_extra_sglang_args"] = variant_args
    return d


def _initial_search_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "accepted": [],
        "rejected": [],
        "tested": {},
        "cursor": 0,
        "last_round": {},
    }


# ---------------------------------------------------------------------------
class ParamsExecutor:
    """ActionRunner for the ``params`` action."""

    def __init__(
        self,
        *,
        default_grid: list[GridVariant] | None = None,
        default_vllm_grid: list[GridVariant] | None = None,
        default_nccl_grid: list[GridVariant] | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        variant_timeout_sec: int = 2400,
        include_nccl: bool = False,
        default_max_candidates_per_round: int = 0,
        keep_threshold_pct: float = 0.5,
    ):
        self.default_grid = list(default_grid or DEFAULT_PARAMS_GRID)
        self.default_vllm_grid = list(default_vllm_grid or DEFAULT_VLLM_PARAMS_GRID)
        self.default_nccl_grid = list(default_nccl_grid or DEFAULT_NCCL_GRID)
        # None = resolve at call time from $FRAMEWORK (sglang/vllm).
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.variant_timeout_sec = variant_timeout_sec
        self.include_nccl = include_nccl
        self.default_max_candidates_per_round = int(default_max_candidates_per_round)
        self.keep_threshold_pct = float(keep_threshold_pct)

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
            or runs_dir(self.session_dir, "params", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Workload-contract materialization (single source of truth).
        # Whether `config_path` points at the shipped YAML, an explicit
        # operator path, or — most commonly — the baseline executor's already
        # materialized YAML plumbed through Coordinator, we re-run
        # materialize_config_with_envs() so the process env (CONC/ISL/OSL/TP/
        # MAX_MODEL_LEN/PRECISION/RUN_EVAL/...) is honored. This is the fix
        # for the "baseline 4367 tok/s vs variants ~360 tok/s" benchmark
        # fairness bug: previously _grid_runner._build_variant_yaml read the
        # raw shipped YAML's smoke defaults (CONC=8/ISL=256/OSL=256/TP=1)
        # while baseline ran at the operator's real workload. Re-materializing
        # is idempotent when the input YAML already matches process env.
        # Per-variant CONC/ISL/OSL overrides (used by sweep) still win
        # because _build_variant_yaml applies variant.extra_envs last.
        resolved_model_for_render = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu_for_render = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model_for_render or None,
            gpu_type=resolved_gpu_for_render or None,
            out_name="params_base.with_envs.yaml",
        )

        base_extra_args = params.get("base_extra_args", "")
        base_tput = float(params.get("base_tput", 0.0))
        timeout_sec = int(params.get("variant_timeout_sec",
                                       self.variant_timeout_sec))
        keep_threshold_pct = float(params.get("keep_threshold_pct",
                                              self.keep_threshold_pct))
        max_candidates = int(params.get(
            "max_candidates_per_round", self.default_max_candidates_per_round,
        ))

        # Compose grid: flags first, then optional NCCL.
        grid_override = params.get("grid")
        framework = _config_framework(config_path)
        if grid_override:
            grid = [
                GridVariant(name=v["name"],
                            extra_sglang_args=v.get("extra_sglang_args", ""),
                            extra_envs=v.get("extra_envs", {}) or {},
                            note=v.get("note", ""))
                for v in grid_override
            ]
        else:
            grid = (
                list(self.default_vllm_grid)
                if "vllm" in framework
                else list(self.default_grid)
            )
            if self.include_nccl or params.get("include_nccl"):
                grid += list(self.default_nccl_grid)

        # T1 — AST-discover parameter flag names from the live framework
        # so the Orchestration prompt sees the full namespace and can
        # synthesize value-filled candidates via params.grid. We do NOT
        # auto-augment the grid here (most param flags need a value, not
        # a bare presence — see backends._augment_grid_with_discovered_flags
        # rationale). Discovered flags are surfaced to SharedState only.
        # We resolve the discovery path through this module's globals so
        # tests / operators can monkeypatch DEFAULT_*_PATH at runtime.
        discovered_param_flags: list[str] = []
        discovered_source = ""
        if not bool(params.get("disable_discovery", False)):
            import sys as _sys
            _self_mod = _sys.modules[__name__]
            src_path = (
                _self_mod.DEFAULT_VLLM_ARG_UTILS if "vllm" in framework
                else _self_mod.DEFAULT_SGLANG_SERVER_ARGS
            )
            discovered_param_flags = discover_param_flags(
                framework=framework, server_args_path=src_path,
            )
            discovered_source = str(src_path)

        search = dict(params.get("params_search") or _initial_search_state())
        search.setdefault("schema_version", 1)
        search.setdefault("accepted", [])
        search.setdefault("rejected", [])
        search.setdefault("tested", {})
        search.setdefault("cursor", 0)
        base_variant_name = str(params.get("base_variant_name") or "").strip()
        if base_variant_name:
            already_accepted = any(
                isinstance(v, dict) and v.get("name") == base_variant_name
                for v in search.get("accepted", [])
            )
            if not already_accepted:
                base_variant = next(
                    (v for v in grid if v.name == base_variant_name),
                    None,
                )
                search["accepted"] = [
                    (
                        _variant_to_dict(base_variant)
                        if base_variant is not None
                        else {
                            "name": base_variant_name,
                            "extra_sglang_args": base_extra_args,
                            "extra_envs": {},
                            "note": "seeded_from_current_best",
                        }
                    ),
                    *list(search.get("accepted") or []),
                ]

        accepted_names = {
            str(v.get("name")) for v in search.get("accepted", [])
            if isinstance(v, dict) and v.get("name")
        }
        rejected_names = {
            str(v.get("name")) for v in search.get("rejected", [])
            if isinstance(v, dict) and v.get("name")
        }
        tested_names = set((search.get("tested") or {}).keys())

        candidates = [
            v for v in grid
            if v.name not in accepted_names
            and v.name not in rejected_names
            and v.name not in tested_names
        ]
        if max_candidates > 0:
            candidates = candidates[:max_candidates]

        if not candidates:
            return {
                "status": "succeeded",
                "base_tput": base_tput,
                "grid_size": 0,
                "all_results": [],
                "winners": [],
                "best_variant": None,
                "best_gain_pct": 0.0,
                "output_throughput": None,
                "workspace": output_root.as_posix(),
                "params_search_update": search,
                "params_search_exhausted": True,
                "discovered_flags_update": (
                    {
                        "framework": framework or "sglang",
                        "param_flags": discovered_param_flags,
                        "source_path": discovered_source,
                    }
                    if discovered_param_flags else None
                ),
            }

        # Reuse the resolved model/gpu from the materialization step above
        # so each Magpie variant benchmarks the user's selected model on
        # the user's selected GPU rather than the YAML's hardcoded fallback.
        resolved_model = resolved_model_for_render
        resolved_gpu = resolved_gpu_for_render
        single_results = await run_grid(
            base_yaml_path=config_path,
            base_extra_args=base_extra_args,
            grid=candidates,
            output_root=output_root,
            variant_timeout_sec=timeout_sec,
            model_path=resolved_model,
            gpu_type=resolved_gpu,
        )

        candidate_by_name = {v.name: v for v in candidates}
        threshold_tput = base_tput * (1.0 + keep_threshold_pct / 100.0)
        round_winners = [
            r for r in single_results
            if r.status == "succeeded"
            and isinstance(r.output_throughput, (int, float))
            and r.output_throughput > threshold_tput
        ]

        combo_results: list[VariantResult] = []
        if round_winners:
            winner_variants = [candidate_by_name[r.name] for r in round_winners]
            # If exactly one new winner is tested on an already accepted base,
            # the single run is already "accepted + D"; no extra duplicate run.
            if len(winner_variants) > 1:
                combo = GridVariant(
                    "combo_" + "_".join(v.name for v in winner_variants),
                    extra_sglang_args=_join_args(
                        *(v.extra_sglang_args for v in winner_variants),
                    ),
                    extra_envs=_merge_envs(winner_variants),
                    note="combo_winners",
                )
                combo_results = await run_grid(
                    base_yaml_path=config_path,
                    base_extra_args=base_extra_args,
                    grid=[combo],
                    output_root=output_root / "combo",
                    variant_timeout_sec=timeout_sec,
                    model_path=resolved_model,
                    gpu_type=resolved_gpu,
                )

        all_results = single_results + combo_results
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

        accepted = [
            _dict_to_variant(v) for v in (search.get("accepted") or [])
            if isinstance(v, dict) and v.get("name")
        ]
        accepted_next = list(accepted)
        selected_new_names: list[str] = []
        best_variant_dict: dict[str, Any] | None = None
        if best:
            if best.name.startswith("combo_"):
                selected_new_names = [r.name for r in round_winners]
                selected_args = str(best.extra_sglang_args or "")
                best_variant_dict = _result_with_effective_args(
                    best, base_extra_args=base_extra_args,
                    variant_args=selected_args,
                )
            elif best.name in candidate_by_name and best_gain >= keep_threshold_pct:
                selected_new_names = [best.name]
                selected_args = candidate_by_name[best.name].extra_sglang_args
                best_variant_dict = _result_with_effective_args(
                    best, base_extra_args=base_extra_args,
                    variant_args=selected_args,
                )

        for name in selected_new_names:
            accepted_next.append(candidate_by_name[name])

        tested = dict(search.get("tested") or {})
        rejected = list(search.get("rejected") or [])
        for r in single_results:
            gain = _result_gain(r.output_throughput, base_tput)
            tested[r.name] = {
                "result": r.to_dict(),
                "gain_pct": gain,
                "base_tput": base_tput,
            }
            if r.name not in set(selected_new_names):
                reason = "not_keep" if (gain is None or gain < keep_threshold_pct) \
                    else "combo_conflict"
                rejected.append({
                    **_variant_to_dict(candidate_by_name[r.name]),
                    "reason": reason,
                    "gain_pct": gain,
                    "tput": r.output_throughput,
                })

        accepted_dicts = [_variant_to_dict(v) for v in accepted_next]
        # Deduplicate rejected by name while preserving latest reason.
        rejected_by_name = {
            str(v.get("name")): v for v in rejected
            if isinstance(v, dict) and v.get("name")
            and str(v.get("name")) not in {a["name"] for a in accepted_dicts}
        }
        search_update = {
            "schema_version": 1,
            "accepted": accepted_dicts,
            "rejected": list(rejected_by_name.values()),
            "tested": tested,
            "cursor": len(tested),
            "last_round": {
                "base_tput": base_tput,
                "base_extra_args": base_extra_args,
                "tested": [r.name for r in single_results],
                "round_winners": [r.name for r in round_winners],
                "selected_new": list(selected_new_names),
                "combo_tested": [r.name for r in combo_results],
            },
        }

        successful_runs = [r for r in all_results if r.status == "succeeded"]

        return {
            "status": "succeeded" if successful_runs else "failed",
            "base_tput": base_tput,
            "grid_size": len(single_results),
            "total_runs": len(all_results),
            "single_results": [r.to_dict() for r in single_results],
            "combo_results": [r.to_dict() for r in combo_results],
            "all_results": [r.to_dict() for r in all_results],
            "winners": [w.to_dict() for w in round_winners],
            "best_variant": best_variant_dict,
            "best_gain_pct": best_gain,
            "output_throughput": best.output_throughput if best else None,
            "workspace": output_root.as_posix(),
            "params_search_update": search_update,
            "params_search_exhausted": len(search_update["tested"]) >= len(grid),
            "discovered_flags_update": (
                {
                    "framework": framework or "sglang",
                    "param_flags": discovered_param_flags,
                    "source_path": discovered_source,
                }
                if discovered_param_flags else None
            ),
        }


params_executor = ParamsExecutor()


__all__ = [
    "DEFAULT_NCCL_GRID",
    "DEFAULT_PARAMS_GRID",
    "DEFAULT_VLLM_PARAMS_GRID",
    "ParamsExecutor",
    "discover_param_flags",
    "params_executor",
]
