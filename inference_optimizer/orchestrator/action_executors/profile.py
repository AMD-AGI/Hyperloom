"""Real ``profile`` ActionRunner — Magpie SGLang run with torch profiler on.

DESIGN v0.6 §16 profile action.

Reuses the BaselineExecutor shell-out machinery; the only meaningful
difference is the YAML config — the profile config has
``profiler.torch_profiler.enabled: true`` so Magpie writes trace files under
``torch_trace/`` or, for TraceLens-patched vLLM graph capture,
``capture_traces/``.

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


def _trace_files_for_dir(trace_dir: Path) -> list[Path]:
    """Return trace files under ``trace_dir`` in a stable order.

    Magpie's classic torch-profiler path writes under
    ``<benchmark_workspace>/torch_trace``. TraceLens-patched vLLM writes graph
    capture traces under ``<profile_task>/capture_traces``. Both use
    ``*.trace.json.gz`` names, and nested layouts are possible as the profiler
    evolves.
    """
    return sorted(trace_dir.rglob("*.trace.json.gz"))


def _candidate_trace_dirs(workspace: Path) -> list[Path]:
    """Trace directories to probe for a Magpie profile workspace."""
    return [
        workspace / "torch_trace",
        workspace / "capture_traces",
        workspace.parent / "capture_traces",
    ]


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

    def _resolve_mn_round_trace_root(self, ctx) -> str:
        """Return per-round wekafs trace dir for multi-node, or '' otherwise.

        Layout (Q2 = multi-level): one dir per RayJob, one subdir per
        profile round, ``torch_trace/`` underneath::

            /wekafs/hyperloom/profile-traces/
              <rayjob_id>/
                <round_id>/
                  torch_trace/
                    *.trace.json.gz

        ``round_id`` is the orchestrator task id (already unique per
        round and stable across the magpie spawn → trace consumption
        boundary). ``rayjob_id`` is taken from
        ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` (which ``cli.py`` exports as
        ``/wekafs/hyperloom/profile-traces/<rayjob>/torch_trace`` on
        provisioning) by stripping the trailing ``torch_trace`` segment.
        Falls back to ``<rayjob>=default`` if the env is missing so the
        executor still produces a usable path rather than crashing.
        """
        from ._multi_node_env import is_multi_node
        if not is_multi_node():
            return ""
        provisioned = os.environ.get(
            "HYPERLOOM_MN_PROFILE_TRACE_DIR", ""
        ).strip()
        if provisioned:
            base = Path(provisioned)
            if base.name == "torch_trace":
                rayjob_root = base.parent
            else:
                rayjob_root = base
        else:
            rayjob_root = Path("/wekafs/hyperloom/profile-traces/default")
        round_id = str(getattr(ctx.task, "task_id", "") or "round").strip() or "round"
        return str(rayjob_root / round_id / "torch_trace")

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
                extra = ctx.extra
            else:
                extra["workspace"] = str(output_dir)

        # Multi-node only: pre-restart the inference server with this
        # round's profiler dir so sglang launches with
        # ``--torch-profiler-dir <round_path>`` and writes traces to a
        # round-scoped wekafs directory both pods and the sandbox can
        # see. Mark ctx.extra so BaselineExecutor.__call__ doesn't run a
        # second restart on top of ours. No-op in single-node mode (the
        # helper short-circuits and ``round_trace_root`` is "").
        round_trace_root = self._resolve_mn_round_trace_root(ctx)
        if round_trace_root:
            from ._multi_node_server_lifecycle import (
                ServerRestartFailed,
                restart_server_for_round,
            )
            try:
                # PD knobs auto-resolved by the helper from $PD_* env
                # (cli.py exported them). See baseline.py for rationale.
                await restart_server_for_round(
                    extra_sglang_args=str(params.get("extra_sglang_args") or ""),
                    torch_profiler_dir=round_trace_root,
                    framework=os.environ.get("FRAMEWORK") or None,
                    model_path=(
                        str(params.get("model_path") or "").strip()
                        or os.environ.get("MODEL_PATH") or None
                    ),
                    tp=int(os.environ.get("TP") or 0) or None,
                    ep=int(os.environ.get("EP") or 0) or None,
                )
            except ServerRestartFailed as exc:
                return {
                    "status": "failed",
                    "error_class": "mn_server_restart_failed",
                    "error": str(exc),
                    "trace_dir": round_trace_root,
                }
            extra["mn_round_restarted"] = True

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
        # Multi-node: trace files live at the round-scoped wekafs dir we
        # just restarted with. We do NOT read $HYPERLOOM_MN_PROFILE_TRACE_DIR
        # here because the helper restored it back to the rayjob-root
        # default after restart so a subsequent non-profile round won't
        # leak this round's path. Single-node falls through to the
        # workspace/torch_trace branch below.
        workspace_str = result.get("workspace")
        if round_trace_root:
            # Multi-node branch: torch traces land at the round-scoped
            # wekafs path the helper restarted sglang with. main's
            # `_candidate_trace_dirs` is workspace-local, so it never
            # matches in multi-node — handle it explicitly here.
            trace_dir = Path(round_trace_root)
            if trace_dir.is_dir():
                trace_files = sorted(trace_dir.glob("*.trace.json.gz"))
                result["trace_dir"] = str(trace_dir)
                result["trace_files"] = [str(p) for p in trace_files]
                if trace_files:
                    result["main_trace_path"] = str(trace_files[0])
                else:
                    log.warning(
                        "profile_executor: multi-node trace dir %s exists "
                        "but no .trace.json.gz files found yet (server "
                        "pods may still be flushing)", trace_dir,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                log.warning(
                    "profile_executor: round trace dir %s does not exist "
                    "after magpie completed; check sglang server logs for "
                    "torch profiler errors",
                    round_trace_root,
                )
        elif workspace_str:
            # Single-node branch: main's multi-candidate trace discovery.
            workspace = Path(workspace_str)
            selected_trace_dir: Path | None = None
            selected_trace_files: list[Path] = []
            existing_empty_dirs: list[Path] = []
            for trace_dir in _candidate_trace_dirs(workspace):
                if not trace_dir.is_dir():
                    continue
                trace_files = _trace_files_for_dir(trace_dir)
                if trace_files:
                    selected_trace_dir = trace_dir
                    selected_trace_files = trace_files
                    break
                existing_empty_dirs.append(trace_dir)

            if selected_trace_dir is not None:
                result["trace_dir"] = str(selected_trace_dir)
                result["trace_files"] = [str(p) for p in selected_trace_files]
                result["main_trace_path"] = str(selected_trace_files[0])
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                result["status"] = "failed"
                result["error_class"] = "no_trace_files"
                probed = ", ".join(
                    str(p) for p in _candidate_trace_dirs(workspace)
                )
                result["error"] = (
                    f"no .trace.json.gz under {workspace_str} (probed: {probed})"
                )
                if existing_empty_dirs:
                    log.warning(
                        "profile_executor: trace dirs exist but no "
                        ".trace.json.gz files in %s",
                        ", ".join(str(p) for p in existing_empty_dirs),
                    )
                else:
                    log.warning(
                        "profile_executor: workspace=%s has no trace dir "
                        "(checked: %s)",
                        workspace_str,
                        ", ".join(str(p) for p in _candidate_trace_dirs(workspace)),
                    )
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_CONFIG",
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]
