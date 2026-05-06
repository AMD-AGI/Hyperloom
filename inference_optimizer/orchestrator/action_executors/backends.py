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

from ...paths import asset_root
from ._grid_runner import GridVariant, VariantResult, pick_winners, run_grid


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


_BACKEND_KEYWORDS = (
    "backend", "enable_", "disable_", "fused", "mixed", "overlap",
    "schedule", "allreduce", "fusion",
)


def discover_backend_flags(
    server_args_path: Path = Path("/sgl-workspace/sglang/python/sglang/srt/server_args.py"),
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


# ---------------------------------------------------------------------------
class BackendsExecutor:
    """ActionRunner for the ``backends`` action."""

    def __init__(
        self,
        *,
        default_grid: list[GridVariant] | None = None,
        default_config_path: Path | str | None = None,
        default_output_root: Path | str = "/workspace/hyperloom",
        variant_timeout_sec: int = 900,
    ):
        self.default_grid = list(default_grid or DEFAULT_BACKENDS_GRID)
        self.default_config_path = (
            Path(default_config_path) if default_config_path
            else asset_root() / "scripts" / "configs" / "baseline_sglang.yaml"
        )
        self.default_output_root = Path(default_output_root)
        self.variant_timeout_sec = variant_timeout_sec

    async def __call__(self, ctx) -> dict[str, Any]:
        params = ctx.task.params or {}
        config_path = Path(params.get("config_path") or self.default_config_path)
        if not config_path.exists():
            return {"status": "failed",
                    "error_class": "missing_config",
                    "error": f"config not found: {config_path}"}
        output_root = Path(
            params.get("output_dir")
            or (self.default_output_root / f"backends-{ctx.task.task_id[:8]}")
        )
        output_root.mkdir(parents=True, exist_ok=True)

        base_extra_args = params.get("base_extra_args", "")
        base_tput = float(params.get("base_tput", 0.0))
        grid_override = params.get("grid")
        if grid_override:
            grid = [
                GridVariant(name=v["name"],
                            extra_sglang_args=v.get("extra_sglang_args", ""),
                            extra_envs=v.get("extra_envs", {}) or {},
                            note=v.get("note", ""))
                for v in grid_override
            ]
        else:
            grid = list(self.default_grid)
        timeout_sec = int(params.get("variant_timeout_sec",
                                       self.variant_timeout_sec))

        # Resolve runtime model_path / gpu_type (task.params > $MODEL_PATH /
        # $GPU_TYPE) and forward so each variant's YAML overrides the legacy
        # hardcoded model + benchmark_script fields.
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
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
        best = max(
            (r for r in results
             if r.status == "succeeded"
             and isinstance(r.output_throughput, (int, float))),
            default=None,
            key=lambda r: r.output_throughput or 0.0,
        )
        best_gain = (
            ((best.output_throughput - base_tput) / base_tput * 100.0)
            if best and base_tput > 0 else 0.0
        )

        return {
            "status": "succeeded" if results else "failed",
            "base_tput": base_tput,
            "grid_size": len(results),
            "all_results": [r.to_dict() for r in results],
            "winners": [w.to_dict() for w in winners],
            "best_variant": best.to_dict() if best else None,
            "best_gain_pct": best_gain,
            "output_throughput": best.output_throughput if best else None,
            "workspace": output_root.as_posix(),
        }


backends_executor = BackendsExecutor()


__all__ = [
    "DEFAULT_BACKENDS_GRID",
    "BackendsExecutor",
    "backends_executor",
    "discover_backend_flags",
]
