"""Real ``profile`` ActionRunner — Magpie SGLang run with torch profiler on.

DESIGN v0.6 §16 profile action.

Reuses the BaselineExecutor shell-out machinery; the only meaningful
difference is the YAML config — the profile config has
``profiler.torch_profiler.enabled: true`` so Magpie writes a
``torch_trace/`` directory inside the workspace.

Result schema (delivered on the bus as ``delegated_result``)::

    status:        "succeeded" | "failed"
    framework:     "sglang"
    model:         path
    request/output/total_token_throughput, latency stats (same as baseline)
    workspace:     absolute path of the Magpie workspace
    trace_dir:     absolute path of the torch_trace dir (or None)
    report_path:   absolute path of benchmark_report.json

Downstream consumers (Kernel agent → tracelens_analysis.py) only need
``trace_dir``; we surface the rest so the same SharedState promotion
logic the baseline path uses (current_best, output_throughput, etc.)
works unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from ...paths import asset_root
from ..roofline_integration import collect_profile_roofline, server_profiling_env
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


# Legacy constant kept pointing at the sglang profile yaml for fixture use.
# Runtime sglang/vllm selection goes through `_default_profile_config()`.
PROFILE_DEFAULT_CONFIG = (
    asset_root() / "scripts" / "configs" / "profile_sglang.yaml"
)
PROFILE_DEFAULT_TIMEOUT_SEC = 1500     # Magpie + sglang profile is heavier


def _default_profile_config() -> Path:
    """Resolve default profile YAML based on $FRAMEWORK env (sglang/vllm)."""
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    name = "profile_vllm.yaml" if fw == "vllm" else "profile_sglang.yaml"
    return asset_root() / "scripts" / "configs" / name


class ProfileExecutor(BaselineExecutor):
    """Subclass that swaps the default config + extracts trace_dir."""

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        default_output_root: Path | str | None = None,
        default_timeout_sec: int = PROFILE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
    ):
        super().__init__(
            magpie_python=magpie_python,
            default_config_path=default_config_path,
            default_output_root=default_output_root,
            default_timeout_sec=default_timeout_sec,
            cwd=cwd,
        )

    def _resolve_default_config(self) -> Path:
        """Override BaselineExecutor's resolver to pick the profile yaml."""
        return _default_profile_config()

    async def __call__(self, ctx) -> dict[str, Any]:
        preload_env = server_profiling_env()
        if preload_env:
            params = dict(ctx.task.params or {})
            extra_envs = dict(params.get("extra_envs") or {})
            extra_envs.update(preload_env)
            params["extra_envs"] = extra_envs
            ctx.task.params = params

        result = await super().__call__(ctx)
        # Augment with trace_dir if the workspace produced one.
        workspace_str = result.get("workspace")
        if workspace_str:
            trace_dir = Path(workspace_str) / "torch_trace"
            if trace_dir.is_dir():
                # Find the actual trace .json.gz files; pick the one most
                # likely to be the main rank trace (first by name).
                trace_files = sorted(trace_dir.glob("*.trace.json.gz"))
                result["trace_dir"] = str(trace_dir)
                result["trace_files"] = [str(p) for p in trace_files]
                if trace_files:
                    result["main_trace_path"] = str(trace_files[0])
                else:
                    log.warning(
                        "profile_executor: torch_trace dir exists but no "
                        ".trace.json.gz files in %s", trace_dir,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                log.warning(
                    "profile_executor: workspace=%s has no torch_trace dir",
                    workspace_str,
                )

            if os.environ.get("ENABLE_PMC_ROOFLINE", "1").strip().lower() not in {
                "0", "false", "no",
            }:
                pmc = await asyncio.to_thread(
                    collect_profile_roofline,
                    session_dir=Path(workspace_str),
                    duration_ms=int(os.environ.get("PMC_PROFILE_DURATION_MS", "15000")),
                    precision=os.environ.get("PRECISION", "fp16").strip() or "fp16",
                )
                result["pmc_roofline"] = pmc
                if pmc.get("pmc_summary_path"):
                    result["pmc_summary_path"] = pmc["pmc_summary_path"]
                if pmc.get("roofline_path"):
                    result["roofline_path"] = pmc["roofline_path"]
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_CONFIG",
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]
