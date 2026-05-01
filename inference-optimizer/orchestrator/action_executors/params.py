"""Real ``params`` ActionExecutor — DESIGN v0.6 §16 params action.

Mirrors marathon/skills/actions/params.md PARAM_GRID:

  * cuda-graph-max-bs $CONC
  * num-continuous-decode-steps {8,16,32}
  * mem-fraction-static 0.85 / 0.90
  * schedule-conservativeness 0.5
  * chunked-prefill-size 65536

Plus an optional NCCL_GRID via ``extra_envs`` (NCCL_MIN_NCHANNELS / NCCL_ALGO).

Same flow as backends_executor: each variant is tested independently on
top of the current ``base_extra_args``; winners are those that beat
``base_tput`` by > +1%.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...paths import asset_root
from ._grid_runner import GridVariant, VariantResult, pick_winners, run_grid


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
DEFAULT_PARAMS_GRID: list[GridVariant] = [
    GridVariant("cuda_graph_max_bs_8",       "--cuda-graph-max-bs 8",
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
    GridVariant("schedule_conservativeness_0_5",
                 "--schedule-conservativeness 0.5",
                 note="scheduling"),
    GridVariant("chunked_prefill_64k",        "--chunked-prefill-size 65536",
                 note="prefill"),
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
class ParamsExecutor:
    """ActionExecutor for the ``params`` action."""

    def __init__(
        self,
        *,
        default_grid: list[GridVariant] | None = None,
        default_nccl_grid: list[GridVariant] | None = None,
        default_config_path: Path | str | None = None,
        default_output_root: Path | str = "/workspace/hyperloom",
        variant_timeout_sec: int = 900,
        include_nccl: bool = False,
    ):
        self.default_grid = list(default_grid or DEFAULT_PARAMS_GRID)
        self.default_nccl_grid = list(default_nccl_grid or DEFAULT_NCCL_GRID)
        self.default_config_path = (
            Path(default_config_path) if default_config_path
            else asset_root() / "scripts" / "configs" / "baseline_qwen3_8b_sglang.yaml"
        )
        self.default_output_root = Path(default_output_root)
        self.variant_timeout_sec = variant_timeout_sec
        self.include_nccl = include_nccl

    async def __call__(self, ctx) -> dict[str, Any]:
        params = ctx.task.params or {}
        config_path = Path(params.get("config_path") or self.default_config_path)
        if not config_path.exists():
            return {"status": "failed",
                    "error_class": "missing_config",
                    "error": f"config not found: {config_path}"}
        output_root = Path(
            params.get("output_dir")
            or (self.default_output_root / f"params-{ctx.task.task_id[:8]}")
        )
        output_root.mkdir(parents=True, exist_ok=True)

        base_extra_args = params.get("base_extra_args", "")
        base_tput = float(params.get("base_tput", 0.0))
        timeout_sec = int(params.get("variant_timeout_sec",
                                       self.variant_timeout_sec))

        # Compose grid: flags first, then optional NCCL.
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
            if self.include_nccl or params.get("include_nccl"):
                grid += list(self.default_nccl_grid)

        results = await run_grid(
            base_yaml_path=config_path,
            base_extra_args=base_extra_args,
            grid=grid,
            output_root=output_root,
            variant_timeout_sec=timeout_sec,
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


params_executor = ParamsExecutor()


__all__ = [
    "DEFAULT_NCCL_GRID",
    "DEFAULT_PARAMS_GRID",
    "ParamsExecutor",
    "params_executor",
]
