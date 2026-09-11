# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``target_analysis`` ActionRunner — external baseline comparison."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.common.env import env_str
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.baseline_comparison.inferencex_client import base_url
from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import (
    _clear_competitor_target,
    _persist,
    analyze,
    to_inferencex_name,
)
from hyperloom.inference_optimizer.baseline_comparison.types import BaselineQuery, BaselineSummary, BenchmarkMode
from ...loop.sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


def _env_int(name: str, default: int = 0) -> int:
    """Read an integer environment variable with a fallback default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning(
            "target_analysis_executor: env %s=%r is not an integer; falling back to %d",
            name,
            raw,
            default,
        )
        return default


class TargetAnalysisExecutor:
    """ActionRunner for the ``target_analysis`` action kind."""

    def __init__(
        self,
        *,
        compare_against_gpu: str,
        session_dir: Path | str | None = None,
    ):
        """Initialize the executor with the pinned comparison reference."""
        self.compare_against_gpu = (compare_against_gpu or "").strip()
        if session_dir is not None:
            self.session_dir: Path | None = Path(session_dir)
        else:
            self.session_dir = None

    def _resolve_session_dir(self, ctx: RunnerContext) -> Path | None:
        """Resolve session_dir: ``ctx.extra["session_dir"]`` > ``task.params["session_dir"]`` > constructor arg > ``paths.session_dir()``; ``None`` when nothing resolves."""
        extra = getattr(ctx, "extra", None) or {}
        cand = extra.get("session_dir")
        if cand:
            return Path(cand)
        params = ctx.task.params or {}
        cand = params.get("session_dir")
        if cand:
            return Path(cand)
        if self.session_dir is not None:
            return self.session_dir
        try:
            from hyperloom.inference_optimizer.session.paths import session_dir as _sd

            sd = _sd()
            return sd if sd.exists() else None
        except Exception:  # noqa: BLE001
            return None

    def _resolve_session_dir_for_cleanup(self, ctx: RunnerContext) -> Path | None:
        """Best-effort session root for stale artefact cleanup when analyze cannot run."""
        resolved = self._resolve_session_dir(ctx)
        if resolved is not None:
            return resolved
        try:
            from hyperloom.inference_optimizer.session.paths import session_dir as _sd

            return _sd()
        except Exception:  # noqa: BLE001
            return None

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the external-baseline comparison and persist report artefacts."""
        params = dict(ctx.task.params or {})

        from ._workload_envs import agentx_active

        state = (getattr(ctx, "extra", None) or {}).get("shared_state")
        benchmark_mode: BenchmarkMode = "agentx" if agentx_active(state) else "synthetic"
        model_path = str(params.get("model_path") or getattr(state, "model_path", "") or env_str("MODEL_PATH"))

        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            cleanup_dir = self._resolve_session_dir_for_cleanup(ctx)
            if cleanup_dir is not None:
                _clear_competitor_target(cleanup_dir)
            log.warning(
                "target_analysis_executor: could not resolve session_dir; skipping (no artefacts will be written)",
            )
            return {
                "status": "succeeded",
                "kind": ctx.task.kind,
                "note": "skipped: no session_dir",
                "baseline_status": "skipped",
                "reason": "no_session_dir",
            }

        compare_against_gpu = str(params.get("compare_against_gpu") or self.compare_against_gpu or "").strip()
        if not compare_against_gpu:
            log.info(
                "target_analysis_executor: no compare_against_gpu set; writing skipped summary and returning",
            )
            try:
                summary = analyze(
                    session_dir=session_dir,
                    model_path=model_path,
                    compare_against_gpu="",
                    benchmark_mode=benchmark_mode,
                )
            except OSError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("target_analysis_executor: analyze() raised: %s", exc)
                return self._analyzer_failure(ctx, session_dir, model_path, "", benchmark_mode, exc)
            return self._format_result(ctx, summary, session_dir)

        framework = str(params.get("framework") or getattr(state, "framework", "") or env_str("FRAMEWORK"))
        precision = str(params.get("precision") or env_str("PRECISION") or getattr(state, "precision", ""))
        isl = int(params.get("isl") or _env_int("ISL", 0))
        osl = int(params.get("osl") or _env_int("OSL", 0))

        try:
            summary = analyze(
                session_dir=session_dir,
                model_path=model_path,
                compare_against_gpu=compare_against_gpu,
                framework=framework,
                precision=precision,
                isl=isl,
                osl=osl,
                benchmark_mode=benchmark_mode,
            )
        except OSError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("target_analysis_executor: analyze() raised: %s", exc)
            return self._analyzer_failure(ctx, session_dir, model_path, compare_against_gpu, benchmark_mode, exc)
        return self._format_result(ctx, summary, session_dir)

    def _analyzer_failure(
        self,
        ctx: RunnerContext,
        session_dir: Path,
        model_path: str,
        gpu: str,
        benchmark_mode: BenchmarkMode,
        exc: Exception,
    ) -> dict[str, Any]:
        _clear_competitor_target(session_dir)
        summary = BaselineSummary(
            query=BaselineQuery(
                model=to_inferencex_name(model_path) or "",
                gpu=gpu,
                benchmark_mode=benchmark_mode,
                isl=None if benchmark_mode == "agentx" else 0,
                osl=None if benchmark_mode == "agentx" else 0,
            ),
            fetched_at=now_iso(timespec="seconds", z_suffix=True),
            row_count=0,
            best=None,
            status="fetch_error",
            reason="analyzer_crash",
            warning=str(exc),
            source=base_url(),
        )
        try:
            _persist(summary, session_dir=session_dir)
        except OSError:
            log.warning("target_analysis_executor: could not persist failed analysis", exc_info=True)
        result = self._format_result(ctx, summary, session_dir)
        result["note"] = f"analyzer crashed: {exc}"
        return result

    def _format_result(
        self,
        ctx: RunnerContext,
        summary: Any,
        session_dir: Path,
    ) -> dict[str, Any]:
        """Build the small bus-friendly result payload (pointer + status; the heavy JSON stays on disk)."""
        from hyperloom.inference_optimizer.session.session_paths import target_analysis_report_md, target_baseline_json

        json_path = target_baseline_json(session_dir)
        md_path = target_analysis_report_md(session_dir)
        out = {
            "status": "succeeded",
            "kind": ctx.task.kind,
            "baseline_status": getattr(summary, "status", "unknown"),
            "reason": getattr(summary, "reason", ""),
            "warning": getattr(summary, "warning", ""),
            "row_count": getattr(summary, "row_count", 0),
            "json_path": str(json_path),
            "md_path": str(md_path),
        }
        best = getattr(summary, "best", None)
        if best is not None:
            out["best_tput_per_gpu"] = best.tput_per_gpu
            out["best_conc"] = best.conc
            out["best_decode_tp"] = best.decode_tp
            if getattr(getattr(summary, "query", None), "benchmark_mode", "synthetic") == "agentx":
                out["best_e2e_norm_intvty_p90"] = best.e2e_norm_intvty_p90
                out["best_benchmark_id"] = best.benchmark_id
        log.info(
            "target_analysis_executor: status=%s reason=%s rows=%d (%s)",
            out["baseline_status"],
            out["reason"] or "-",
            out["row_count"],
            out["warning"] or "ok",
        )
        return out


__all__ = ["TargetAnalysisExecutor"]
