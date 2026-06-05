"""Real ``sweep`` ActionRunner sweep action.

Full ISL/OSL/CONC sweep with
the optimized server config to map the Pareto frontier. P2-3 keeps the
implementation simple — relaunches sglang once per (CONC, ISL, OSL)
combo via the same Magpie shell. A future single-server mode (one
launch, many bench calls) is a natural follow-up if wall time becomes
the bottleneck.

Inputs (task.params):

* ``config_path``      — base Magpie YAML (defaults to baseline asset)
* ``base_extra_args``  — current best EXTRA_SGLANG_ARGS to layer in
* ``conc_values``      — list of int CONC, default [4, 16, 64]
* ``isl_osl_configs``  — list of "<ISL>:<OSL>" str, default ["1024:1024",
                          "8192:1024", "1024:8192"]
* ``num_prompts_factor`` — multiplier vs CONC (default 5; adaptive
                            default for OSL ≤ 1024)

Result::

    status:        "succeeded" | "failed"
    sweep_grid:    [{conc, isl, osl, output_throughput, ttft_mean_ms,
                    e2el_mean_ms, status, workspace, error}]
    pareto_front:  subset of sweep_grid that's not dominated
    best_for_each_conc: dict[conc → entry with highest tput]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from ._grid_runner import (
    GridVariant,
    VariantResult,
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


DEFAULT_CONC_VALUES = [4, 16, 64]
DEFAULT_ISL_OSL = ["1024:1024", "8192:1024", "1024:8192"]
DEFAULT_NUM_PROMPTS_FACTOR = 5


def _coerce_int(value: Any) -> int:
    """Best-effort int coercion that never raises; non-numeric / None
    collapse to 0 so an unset ``max_model_len`` disables filtering."""
    if value is None or value == "":
        return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _build_grid(
    *,
    conc_values: list[int],
    isl_osl_configs: list[str],
    num_prompts_factor: int,
    base_extra_args: str,
    max_model_len: int = 0,
) -> tuple[list[GridVariant], list[dict[str, Any]]]:
    """Fan out CONC × (ISL, OSL) into per-combo Magpie variants.

    Each variant overrides ``CONC`` / ``ISL`` / ``OSL`` envs in the YAML
    so the same Magpie shell reuses our existing baseline machinery.

    When ``max_model_len`` is positive, any combo whose ``ISL + OSL``
    exceeds it is dropped up front: the server rejects every request in
    that combo with ``VLLMValidationError: maximum context length`` so
    the benchmark always aborts with an invalid measurement. Filtering
    here avoids burning a launch + warmup on a guaranteed failure.

    Returns ``(runnable_variants, skipped_records)`` where each skipped
    record documents why a combo was dropped so the sweep result keeps
    it visible instead of silently pretending it never existed.
    """
    out: list[GridVariant] = []
    skipped: list[dict[str, Any]] = []
    for conc in conc_values:
        num_prompts = max(int(conc) * num_prompts_factor, conc)
        for io_cfg in isl_osl_configs:
            try:
                isl_str, osl_str = io_cfg.split(":", 1)
                isl, osl = int(isl_str), int(osl_str)
            except (ValueError, AttributeError) as exc:
                log.warning("sweep: malformed isl_osl=%s: %s — skipping",
                             io_cfg, exc)
                continue
            name = f"conc{conc}_isl{isl}_osl{osl}"
            if max_model_len > 0 and (isl + osl) > max_model_len:
                reason = (
                    f"isl+osl={isl + osl} exceeds max_model_len="
                    f"{max_model_len}"
                )
                log.warning(
                    "sweep: skipping variant %s: %s "
                    "(server would reject every request)",
                    name, reason,
                )
                skipped.append({
                    "name":        name,
                    "conc":        conc,
                    "isl":         isl,
                    "osl":         osl,
                    "status":      "skipped",
                    "skip_reason": reason,
                })
                continue
            out.append(GridVariant(
                name=name,
                extra_server_args=base_extra_args,
                extra_envs={
                    "CONC":         str(conc),
                    "ISL":          str(isl),
                    "OSL":          str(osl),
                    "NUM_PROMPTS":  str(num_prompts),
                },
                note=f"conc={conc} isl={isl} osl={osl}",
            ))
    return out, skipped


def _result_dict(v: VariantResult) -> dict[str, Any]:
    d = v.to_dict()
    # Pull conc/isl/osl out of extra_envs so consumers don't have to
    # parse them.
    envs = v.extra_envs or {}
    d["conc"] = int(envs.get("CONC", 0))
    d["isl"] = int(envs.get("ISL", 0))
    d["osl"] = int(envs.get("OSL", 0))
    return d


def _pareto_front(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Naive O(N²) Pareto for (max output_throughput, min e2el_mean_ms)."""
    succ = [e for e in entries if e["status"] == "succeeded"
            and isinstance(e.get("output_throughput"), (int, float))
            and isinstance(e.get("e2el_mean_ms"), (int, float))]
    front: list[dict[str, Any]] = []
    for cand in succ:
        dominated = False
        for other in succ:
            if other is cand:
                continue
            if (other["output_throughput"] >= cand["output_throughput"]
                and other["e2el_mean_ms"] <= cand["e2el_mean_ms"]
                and (other["output_throughput"] > cand["output_throughput"]
                     or other["e2el_mean_ms"] < cand["e2el_mean_ms"])):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return front


# ---------------------------------------------------------------------------
class SweepExecutor:
    """ActionRunner for the ``sweep`` action."""

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_conc_values: list[int] | None = None,
        default_isl_osl_configs: list[str] | None = None,
        default_num_prompts_factor: int = DEFAULT_NUM_PROMPTS_FACTOR,
        variant_timeout_sec: int = 2400,
    ):
        # None = resolve at call time from $FRAMEWORK (sglang/vllm). Tests
        # that pass an explicit fixture path keep their override.
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.default_conc_values = list(default_conc_values or DEFAULT_CONC_VALUES)
        self.default_isl_osl_configs = list(
            default_isl_osl_configs or DEFAULT_ISL_OSL
        )
        self.default_num_prompts_factor = int(default_num_prompts_factor)
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
            or runs_dir(self.session_dir, "sweep", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Workload-contract materialization. Sweep deliberately overrides
        # CONC/ISL/OSL/NUM_PROMPTS per variant via _build_grid below, so
        # those four envs are immaterial here, but TP/MAX_MODEL_LEN/
        # PRECISION/RUN_EVAL/ROCR_VISIBLE_DEVICES still flow from process
        # env onto the materialized YAML and become the per-variant base.
        # Without this step `_build_variant_yaml` would silently inherit
        # the shipped YAML's TP=1 default and run sweep variants single-
        # GPU on a TP=8 model. Idempotent when input already matches env.
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
            out_name="sweep_base.with_envs.yaml",
        )

        conc_values = list(params.get("conc_values") or self.default_conc_values)
        isl_osl_configs = list(
            params.get("isl_osl_configs") or self.default_isl_osl_configs
        )
        num_prompts_factor = int(
            params.get("num_prompts_factor", self.default_num_prompts_factor)
        )
        base_extra_args = params.get("base_extra_args", "")
        timeout_sec = int(params.get("variant_timeout_sec",
                                       self.variant_timeout_sec))

        # Drop ISL+OSL combos that can't fit the server's context window
        # before launching anything (see _build_grid). Resolved from the
        # task params first, then the MAX_MODEL_LEN process env that the
        # workload contract materializes onto every variant YAML.
        max_model_len = _coerce_int(
            params.get("max_model_len")
            or os.environ.get("MAX_MODEL_LEN")
        )

        grid, skipped_variants = _build_grid(
            conc_values=conc_values,
            isl_osl_configs=isl_osl_configs,
            num_prompts_factor=num_prompts_factor,
            base_extra_args=base_extra_args,
            max_model_len=max_model_len,
        )

        # `resolved_model` / `resolved_gpu` were resolved above for the
        # materialization step; reuse them here. See baseline.py /
        # _grid_runner.py for the rationale on why both must flow through.
        results = await run_grid(
            base_yaml_path=config_path,
            base_extra_args="",  # sweep variants carry args themselves
            grid=grid,
            output_root=output_root,
            variant_timeout_sec=timeout_sec,
            model_path=resolved_model,
            gpu_type=resolved_gpu,
            benchmark_script=override_script,
            result_dir=override_result_dir,
        )

        entries = [_result_dict(v) for v in results]
        # Surface context-window-skipped combos alongside the run
        # entries so the grid stays complete and consumers can see why a
        # combo never ran. Skipped entries never enter the Pareto / best
        # selections below (those filter on status == "succeeded").
        entries.extend(skipped_variants)
        front = _pareto_front(entries)

        # Best per CONC.
        best_for_each_conc: dict[int, dict[str, Any]] = {}
        for e in entries:
            if e["status"] != "succeeded":
                continue
            cur = best_for_each_conc.get(e["conc"])
            if cur is None or (
                isinstance(e.get("output_throughput"), (int, float))
                and isinstance(cur.get("output_throughput"), (int, float))
                and e["output_throughput"] > cur["output_throughput"]
            ):
                best_for_each_conc[e["conc"]] = e

        successful_entries = [e for e in entries if e.get("status") == "succeeded"]

        return {
            "status": "succeeded" if successful_entries else "failed",
            "grid_size": len(entries),
            "sweep_grid": entries,
            "pareto_front": front,
            "best_for_each_conc": {str(k): v
                                    for k, v in best_for_each_conc.items()},
            "workspace": output_root.as_posix(),
        }


sweep_executor = SweepExecutor()


__all__ = [
    "DEFAULT_CONC_VALUES",
    "DEFAULT_ISL_OSL",
    "DEFAULT_NUM_PROMPTS_FACTOR",
    "SweepExecutor",
    "sweep_executor",
]
