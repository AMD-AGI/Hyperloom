# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``target_analysis`` ActionRunner — external baseline comparison.

Fetches matching reference rows live from the InferenceX benchmarks API into a
``BaselineSummary`` and persists ``target_analysis/target_baseline.json`` +
report MD. On a successful, dimension-aligned match it also writes a measured
``competitor_target.json`` (``source`` = the API URL) that the gap
advisory consumes as *direction, not a gate*. It never writes SharedState, the
Objective, or scoring, and never gates any KEEP/REVERT decision — so any
reference number reaching a prompt is API-measured, never LLM-authored.

Failure policy: never fail the task. Any error (HTTP, mapping miss, zero rows,
malformed env) is recorded in ``BaselineSummary.status`` / ``.warning`` and the
runner returns ``status="succeeded"``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.common.env import env_str
from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import (
    _clear_competitor_target,
    analyze,
)
from ...loop.sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


def _env_int(name: str, default: int = 0) -> int:
    """Read an integer environment variable with a fallback default.

    Args:
        name (str): The environment variable name.
        default (int): Value returned when the var is unset or non-integer.

    Returns:
        int: The parsed integer, or ``default`` when unset / unparseable.
    """
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
    """ActionRunner for the ``target_analysis`` action kind.

    The GPU reference is pinned by the CLI at start-up; model name / precision
    / framework / shape are pulled from env each call. ``task.params`` may
    override any field for tests (compare_against_gpu / model_path / framework
    / precision / isl / osl).
    """

    def __init__(
        self,
        *,
        compare_against_gpu: str,
        session_dir: Path | str | None = None,
    ):
        """Initialize the executor with the pinned comparison reference.

        Args:
            compare_against_gpu (str): The GPU reference identifier the session
                compares against (immutable for the session).
            session_dir (Path | str | None): Fallback session root used when
                the context does not supply one.
        """
        self.compare_against_gpu = (compare_against_gpu or "").strip()
        if session_dir is not None:
            self.session_dir: Path | None = Path(session_dir)
        else:
            self.session_dir = None

    def _resolve_session_dir(self, ctx: RunnerContext) -> Path | None:
        """Resolve session_dir: ``ctx.extra["session_dir"]`` >
        ``task.params["session_dir"]`` > constructor arg >
        ``paths.session_dir()``; ``None`` when nothing resolves.

        Args:
            ctx: The runner context carrying ``task.params`` and ``extra``.

        Returns:
            The resolved session directory, or ``None`` when nothing resolves.
        """
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
        """Best-effort session root for stale artefact cleanup when analyze cannot run.

        Unlike :meth:`_resolve_session_dir`, the fallback from
        ``paths.session_dir()`` is returned even when the directory does not
        yet exist, so a pre-existing ``competitor_target.json`` from an older
        run can still be removed.

        Args:
            ctx: The runner context carrying ``task.params`` and ``extra``.

        Returns:
            The resolved session directory, or ``None`` when nothing resolves.
        """
        resolved = self._resolve_session_dir(ctx)
        if resolved is not None:
            return resolved
        try:
            from hyperloom.inference_optimizer.session.paths import session_dir as _sd

            return _sd()
        except Exception:  # noqa: BLE001
            return None

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the external-baseline comparison and persist report artefacts.

        Resolves the session dir and comparison reference, invokes
        :func:`analyze` (folding matching InferenceX rows into a
        ``BaselineSummary`` and writing JSON / MD artefacts), and returns a
        bus-friendly summary result. Never fails the task: upstream / mapping
        errors are recorded in the summary status and ``status="succeeded"``
        is returned.

        Args:
            ctx (RunnerContext): The runner context carrying ``task.params``
                overrides and ``extra`` (session dir).

        Returns:
            dict[str, Any]: A ``status="succeeded"`` result dict pointing at
                the persisted artefacts plus the comparison status / reason.
        """
        params = dict(ctx.task.params or {})
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
                    model_path=str(params.get("model_path") or env_str("MODEL_PATH")),
                    compare_against_gpu="",
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("target_analysis_executor: analyze() raised: %s", exc)
                return {
                    "status": "succeeded",
                    "kind": ctx.task.kind,
                    "note": f"analyzer crashed: {exc}",
                    "baseline_status": "fetch_error",
                    "reason": "analyzer_crash",
                }
            return self._format_result(ctx, summary, session_dir)

        model_path = str(params.get("model_path") or env_str("MODEL_PATH"))
        framework = str(params.get("framework") or env_str("FRAMEWORK"))
        precision = str(params.get("precision") or env_str("PRECISION"))
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
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("target_analysis_executor: analyze() raised: %s", exc)
            return {
                "status": "succeeded",
                "kind": ctx.task.kind,
                "note": f"analyzer crashed: {exc}",
                "baseline_status": "fetch_error",
                "reason": "analyzer_crash",
            }
        return self._format_result(ctx, summary, session_dir)

    def _format_result(
        self,
        ctx: RunnerContext,
        summary: Any,
        session_dir: Path,
    ) -> dict[str, Any]:
        """Build the small bus-friendly result payload (pointer + status;
        the heavy JSON stays on disk).

        Args:
            ctx: The runner context (supplies ``task.kind``).
            summary: The baseline comparison summary object.
            session_dir: Session directory the report artefacts live under.

        Returns:
            The bus-friendly result dict with status, pointers, and best-point
            metrics when available.
        """
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
        log.info(
            "target_analysis_executor: status=%s reason=%s rows=%d (%s)",
            out["baseline_status"],
            out["reason"] or "-",
            out["row_count"],
            out["warning"] or "ok",
        )
        return out


__all__ = ["TargetAnalysisExecutor"]
