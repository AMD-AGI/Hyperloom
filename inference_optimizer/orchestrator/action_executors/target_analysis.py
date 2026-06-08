# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Real ``target_analysis`` ActionRunner — external baseline comparison.

Replaces the no-op stub for ``target_analysis`` registered by
``cli._register_executors`` when ``--compare-against-gpu`` is supplied.

What this runner does (and what it deliberately does NOT do):

* DOES   — pull the matching reference rows from InferenceX, fold them
  into a :class:`BaselineSummary`, and persist
  ``target_analysis/target_baseline.json`` + report MD under the
  session dir.
* DOES NOT — touch ``SharedState``, return data shaped for any
  ``Objective``, populate ``target_summary``, or surface anything for
  prompt builders. The output is report-only by design (see chat
  decision "S2": only :class:`ReportExecutor` reads the artefacts).

Failure policy:
    *Never* fail the task. Upstream HTTP errors, mapping miss, zero
    rows after filtering, even a malformed env — all are recorded in
    ``BaselineSummary.status`` / ``BaselineSummary.warning`` and the
    runner returns ``status="succeeded"``. The optimizer loop must not
    be sensitive to whether InferenceX is reachable.

Wire-up (in :func:`inference_optimizer.cli._register_executors`)::

    if compare_against_gpu:
        coordinator.sub.register_executor(
            "target_analysis",
            TargetAnalysisExecutor(compare_against_gpu=compare_against_gpu,
                                    session_dir=session_dir),
        )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ...baseline_comparison.target_analyzer import analyze
from ..sub_agent_runner import RunnerContext


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
            "target_analysis_executor: env %s=%r is not an integer; "
            "falling back to %d", name, raw, default,
        )
        return default


def _env_str(name: str) -> str:
    """Read a stripped string environment variable.

    Args:
        name (str): The environment variable name.

    Returns:
        str: The stripped value, or ``""`` when unset.
    """
    return os.environ.get(name, "").strip()


class TargetAnalysisExecutor:
    """ActionRunner for the ``target_analysis`` action kind.

    Parameters are immutable per-session: the comparison reference is
    pinned by the CLI at start-up and the executor pulls model name /
    precision / framework / shape from process env on every call (so
    a resume after env changes picks up the new values, but the
    GPU reference never changes mid-session).

    ``RunnerContext.task.params`` may override any field for tests:

        compare_against_gpu / model_path / framework / precision / isl / osl
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
        """Same resolution order as :class:`ReportExecutor`.

        Order: ``ctx.extra["session_dir"]`` > ``task.params["session_dir"]``
        > constructor arg > ``paths.session_dir()``. Returns ``None`` only
        when nothing resolves to an existing directory.

        Args:
            ctx (RunnerContext): The runner context whose ``extra`` /
                ``task.params`` may carry a ``session_dir``.

        Returns:
            Path | None: The resolved session directory, or ``None`` when none
                resolves to an existing directory.
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
            from ...paths import session_dir as _sd
            sd = _sd()
            return sd if sd.exists() else None
        except Exception:  # noqa: BLE001
            return None

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the external-baseline comparison and persist report artefacts.

        Resolves the session dir and comparison reference, invokes
        :func:`analyze` (folding matching InferenceX rows into a
        ``BaselineSummary`` and writing JSON / MD artefacts), and returns a
        report-only result. Never fails the task: upstream / mapping errors are
        recorded in the summary status and ``status="succeeded"`` is returned.

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
            log.warning(
                "target_analysis_executor: could not resolve session_dir; "
                "skipping (no artefacts will be written)",
            )
            return {
                "status": "succeeded",
                "kind":   ctx.task.kind,
                "note":   "skipped: no session_dir",
                "baseline_status": "skipped",
                "reason": "no_session_dir",
            }

        compare_against_gpu = str(
            params.get("compare_against_gpu") or self.compare_against_gpu or ""
        ).strip()
        if not compare_against_gpu:
            log.info(
                "target_analysis_executor: no compare_against_gpu set; "
                "writing skipped summary and returning",
            )
            try:
                summary = analyze(
                    session_dir=session_dir,
                    model_path=str(params.get("model_path") or _env_str("MODEL_PATH")),
                    compare_against_gpu="",
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("target_analysis_executor: analyze() raised: %s", exc)
                return {
                    "status": "succeeded",
                    "kind":   ctx.task.kind,
                    "note":   f"analyzer crashed: {exc}",
                    "baseline_status": "fetch_error",
                    "reason": "analyzer_crash",
                }
            return self._format_result(ctx, summary, session_dir)

        model_path = str(params.get("model_path") or _env_str("MODEL_PATH"))
        framework = str(params.get("framework") or _env_str("FRAMEWORK"))
        precision = str(params.get("precision") or _env_str("PRECISION"))
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
                "kind":   ctx.task.kind,
                "note":   f"analyzer crashed: {exc}",
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
        """Build the bus-friendly result payload.

        Keeps the dict small so the resulting ``delegated_result`` event
        isn't bloated — the heavy JSON is on disk; this payload is just
        a pointer + status.

        Args:
            ctx (RunnerContext): The runner context (for ``task.kind``).
            summary (Any): The ``BaselineSummary`` produced by :func:`analyze`.
            session_dir (Path): The session root the artefacts were written to.

        Returns:
            dict[str, Any]: A compact ``status="succeeded"`` result dict with
                artefact paths and summary status fields.
        """
        from ...session_paths import target_analysis_report_md, target_baseline_json
        json_path = target_baseline_json(session_dir)
        md_path = target_analysis_report_md(session_dir)
        out = {
            "status":          "succeeded",
            "kind":            ctx.task.kind,
            "baseline_status": getattr(summary, "status", "unknown"),
            "reason":          getattr(summary, "reason", ""),
            "warning":         getattr(summary, "warning", ""),
            "row_count":       getattr(summary, "row_count", 0),
            "json_path":       str(json_path),
            "md_path":         str(md_path),
        }
        best = getattr(summary, "best", None)
        if best is not None:
            out["best_tput_per_gpu"] = best.tput_per_gpu
            out["best_conc"] = best.conc
            out["best_decode_tp"] = best.decode_tp
        log.info(
            "target_analysis_executor: status=%s reason=%s rows=%d (%s)",
            out["baseline_status"], out["reason"] or "-",
            out["row_count"], out["warning"] or "ok",
        )
        return out


__all__ = ["TargetAnalysisExecutor"]
