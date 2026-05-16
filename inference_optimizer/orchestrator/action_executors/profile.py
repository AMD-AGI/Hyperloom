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

import yaml

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


def _preferred_main_trace_path(trace_dir: Path, trace_files: list[Path]) -> Path:
    """Trace path to pass downstream to TraceLens.

    The old `sorted(...)[0]` behaviour picked `TP-0-DECODE.trace.json.gz`
    before `merged-*.trace.json.gz`, which hands TraceLens a tiny single-rank
    decode slice instead of the large annotated trace its splitter expects.
    Prefer the merged trace when Magpie produced one; otherwise pass the trace
    directory so kernel-agent can apply its own ordering instead of pinning a
    single staged file.
    """
    merged = sorted(p for p in trace_files if p.name.startswith("merged-"))
    return merged[0] if merged else trace_dir


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

    def _after_materialize_config(
        self, config_path: Path, output_dir: Path,
    ) -> dict[str, Any] | None:
        """Patch the exact InferenceX checkout Magpie will execute.

        `$INFERENCEX_PATH` alone is not enough: Magpie resolves an empty
        `benchmark.inferencex_path` to its own sibling checkout. Read the
        materialized YAML and patch that resolved path, so NUM_PROMPTS and
        PROFILE_EXTRA_BODY cannot be applied to one checkout while Magpie runs
        another.
        """
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "error_class": "profile_config_unreadable",
                "error": f"cannot read materialized profile config {config_path}: {exc}",
            }
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
        inferencex_path = ""
        if isinstance(bench, dict):
            inferencex_path = str(bench.get("inferencex_path") or "").strip()
        if not inferencex_path:
            inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
        if not inferencex_path:
            log.warning(
                "profile_executor: no benchmark.inferencex_path / "
                "INFERENCEX_PATH configured; skipping InferenceX profile "
                "patch validation"
            )
            return None

        ix_root = Path(inferencex_path)
        lib_ok = ensure_benchmark_lib_patched(ix_root)
        serving_ok = ensure_benchmark_serving_patched(ix_root)
        lib_path = ix_root / "benchmarks" / "benchmark_lib.sh"
        serving_path = ix_root / "utils" / "bench_serving" / "benchmark_serving.py"

        def _contains(path: Path, needle: str) -> bool:
            try:
                return needle in path.read_text(encoding="utf-8")
            except OSError:
                return False

        lib_valid = _contains(lib_path, '${NUM_PROMPTS:-$max_concurrency}')
        serving_valid = _contains(serving_path, "PROFILE_EXTRA_BODY")
        if not (lib_ok and serving_ok and lib_valid and serving_valid):
            return {
                "status": "failed",
                "error_class": "profile_inferencex_patch_failed",
                "error": (
                    "profile requires InferenceX to honour NUM_PROMPTS and "
                    "PROFILE_EXTRA_BODY, but the checkout Magpie will use is "
                    f"not patched: inferencex_path={ix_root}, "
                    f"benchmark_lib_ok={lib_ok}/{lib_valid}, "
                    f"benchmark_serving_ok={serving_ok}/{serving_valid}"
                ),
                "inferencex_path": str(ix_root),
                "benchmark_lib": str(lib_path),
                "benchmark_serving": str(serving_path),
            }
        return None

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
        result = await super().__call__(ctx)
        # Augment with trace_dir if the workspace produced one.
        workspace_str = result.get("workspace")
        if workspace_str:
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
                main_trace = _preferred_main_trace_path(
                    selected_trace_dir, selected_trace_files,
                )
                result["main_trace_path"] = str(main_trace)
                result["profile_trace_selection_reason"] = (
                    "merged_trace_preferred"
                    if main_trace.name.startswith("merged-")
                    else "trace_dir_preferred"
                )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
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
