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

import logging
import os
from pathlib import Path
from typing import Any

from ...paths import asset_root
from ._inferencex_patcher import (
    ensure_benchmark_lib_patched,
    ensure_benchmark_serving_patched,
)
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


# Legacy constant kept pointing at the sglang profile yaml for fixture use.
# Runtime sglang/vllm selection goes through `_default_profile_config()`.
PROFILE_DEFAULT_CONFIG = (
    asset_root() / "scripts" / "configs" / "profile_sglang.yaml"
)
PROFILE_DEFAULT_TIMEOUT_SEC = 2400     # Magpie + sglang profile is heavier, 40 min wall cap


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
        session_dir: Path | str | None = None,
        default_timeout_sec: int = PROFILE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
    ):
        super().__init__(
            magpie_python=magpie_python,
            default_config_path=default_config_path,
            session_dir=session_dir,
            default_timeout_sec=default_timeout_sec,
            cwd=cwd,
        )

    def _resolve_default_config(self) -> Path:
        """Override BaselineExecutor's resolver to pick the profile yaml."""
        return _default_profile_config()

    async def __call__(self, ctx) -> dict[str, Any]:
        # Override action label so per-task output lands under runs/profile/
        # rather than runs/baseline/ when the runner derives the path.
        params = ctx.task.params or {}
        extra = getattr(ctx, "extra", None) or {}
        if not (params.get("output_dir") or extra.get("workspace")):
            output_dir = self._resolve_workspace(ctx, "profile")
            output_dir.mkdir(parents=True, exist_ok=True)
            # Stash so BaselineExecutor.__call__ picks it up via extra.
            if extra is None:
                ctx.extra = {"workspace": str(output_dir)}
            else:
                extra["workspace"] = str(output_dir)
        # Issue #194 §2: ensure InferenceX's benchmark_lib.sh honours
        # $NUM_PROMPTS. _workload_envs computes a NUM_PROMPTS large
        # enough to reach the steady-state window, but unpatched
        # upstream stomps it on every PROFILE=1 run — silently
        # producing empty traces. The patch is backward-compatible
        # (no-op when NUM_PROMPTS is unset) and idempotent, so calling
        # this on every profile launch costs ~1 file read after the
        # first success.
        ensure_benchmark_lib_patched()
        # PR-D §2: ensure InferenceX `benchmark_serving.py` reads our
        # `PROFILE_EXTRA_BODY` env var. Without this patch the
        # `/start_profile` request bakes in upstream's hardcoded
        # `extra_body={"num_steps": 1, ...}` and silently drops
        # shape_discovery / roofline_annotations / the steady-state
        # start_step computed by `_workload_envs.py`. Same idempotent
        # atomic-replace shape as `ensure_benchmark_lib_patched`.
        ensure_benchmark_serving_patched()
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
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_CONFIG",
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]
