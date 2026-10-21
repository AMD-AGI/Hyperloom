# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rows the kernel timeline records for the campaigns the KERNEL phase settles.

Every route here reports a campaign that has already finished, so each function
reads a result mapping and writes one lane of the timeline event.
"""

from __future__ import annotations

import logging as _logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.inference_optimizer.breakdown.recorder import tool_versions

log = _logging.getLogger(__name__)


def settled_text(result: Mapping[str, Any], *keys: str, default: str = "") -> str:
    """The first non-empty value under ``keys`` as text, else ``default``.

    The producers on these routes name the same fact differently -- a run's end is
    ``ended_at`` or ``ts``, its failure is ``error`` or ``skip_reason`` -- and a
    result carries only the spelling its own producer wrote.
    """
    for key in keys:
        value = result.get(key)
        if value:
            return str(value)
    return default


class RunnerRun(NamedTuple):
    """What the coordinator itself observed of one delegated GEAK runner process.

    Each field overrides the runner's own account of it; left unset, the result
    the runner wrote is what the timeline reports.
    """

    started_at: str = ""
    duration_sec: float | None = None
    runner_timeout_sec: int | None = None
    kill_timeout_sec: int | None = None
    recovered_from_disk: bool = False


class _CampaignFacts(NamedTuple):
    """What every row of one GEMM tuning campaign reports identically."""

    run_id: str
    status: str
    shapes_total: Any
    tuned_file: str
    gain_pct: Any
    graded_objective: str
    micro_decision: str
    integrate_ref: str
    started_at: str
    ended_at: str
    duration_sec: Any


def record_gemm_campaign(recorder: Any, session_dir: Path, result: Mapping[str, Any], *, macro_cycle: int) -> None:
    """Record a settled GEMM campaign in the Forge lane -- one row per tuner that ran."""
    facts = _campaign_facts(result, macro_cycle=macro_cycle)
    tuners = result.get("tuners_run")
    rows = [row for row in tuners if isinstance(row, dict)] if isinstance(tuners, list) else []
    for row in rows:
        _record_tuner_run(recorder, row, facts)
    if not rows:
        _record_untuned_campaign_run(recorder, result, facts)
    backend = settled_text(result, "backend", "engine").lower()
    if backend:
        tool_versions.record_tool_version(session_dir, tool=backend)


def _campaign_facts(result: Mapping[str, Any], *, macro_cycle: int) -> _CampaignFacts:
    """The campaign-wide half of every row this GEMM campaign reports."""
    capture = result.get("shape_capture")
    capture = capture if isinstance(capture, dict) else {}
    workspace = settled_text(result, "workspace")
    return _CampaignFacts(
        run_id=settled_text(result, "task_id") or (Path(workspace).name if workspace else "") or f"gemm-c{macro_cycle}",
        status=settled_text(result, "status", default="unknown"),
        shapes_total=result.get("shapes_total", capture.get("shape_count")),
        tuned_file=settled_text(result, "tuned_file", "config_path"),
        # _validate_gemm_tuning_e2e writes the validated gain to e2e_gain_pct, not gain_pct -- no
        # producer on this path ever sets a bare "gain_pct" key.
        gain_pct=result.get("e2e_gain_pct"),
        graded_objective=settled_text(result, "graded_objective"),
        micro_decision=settled_text(result, "micro_decision", "decision"),
        integrate_ref=settled_text(result, "integration_id"),
        started_at=settled_text(result, "started_at"),
        ended_at=settled_text(result, "ended_at", "ts"),
        duration_sec=result.get("duration_sec"),
    )


def _record_tuner_run(recorder: Any, row: Mapping[str, Any], facts: _CampaignFacts) -> None:
    """Record what one named tuner of a GEMM campaign did."""
    tuner = settled_text(row, "tuner")
    artifact = settled_text(row, "artifact", "env_value")
    # The campaign's e2e-validated gain names the axis the KEEP was graded on; it
    # belongs on the tuner whose artifact was actually applied, not on every row
    # this cycle produced.
    applied = bool(facts.tuned_file) and artifact == facts.tuned_file
    recorder.record_gemm_tuning_run(
        run_id=f"{facts.run_id}-{tuner}" if tuner else facts.run_id,
        status=settled_text(row, "status", default=facts.status),
        shapes_total=row.get("total_shapes", facts.shapes_total),
        shapes_tuned=row.get("improved_shapes"),
        config_path=artifact,
        gain_pct=facts.gain_pct if applied else None,
        graded_objective=facts.graded_objective if applied else "",
        tuner=tuner,
        micro_decision=facts.micro_decision,
        integrate_ref=facts.integrate_ref,
        started_at=facts.started_at,
        ended_at=facts.ended_at,
        duration_sec=row.get("elapsed_s", facts.duration_sec),
        error_class=settled_text(row, "error_class"),
        failure_reason=settled_text(row, "error", "skip_reason", "error_class"),
    )


def _record_untuned_campaign_run(recorder: Any, result: Mapping[str, Any], facts: _CampaignFacts) -> None:
    """Record the single row a GEMM campaign that named no tuner reports for itself."""
    recorder.record_gemm_tuning_run(
        run_id=facts.run_id,
        status=facts.status,
        shapes_total=facts.shapes_total,
        shapes_tuned=result.get("shapes_tuned"),
        config_path=facts.tuned_file,
        gain_pct=facts.gain_pct,
        graded_objective=facts.graded_objective,
        tuner=settled_text(result, "tuner"),
        micro_decision=facts.micro_decision,
        integrate_ref=facts.integrate_ref,
        started_at=facts.started_at,
        ended_at=facts.ended_at,
        duration_sec=facts.duration_sec,
        error_class=settled_text(result, "error_class"),
        failure_reason=settled_text(result, "error", "skip_reason", "error_class"),
    )


def record_geak_delegation(
    recorder: Any, result: Mapping[str, Any], *, handoff: Mapping[str, Any], run: RunnerRun
) -> None:
    """Record the delegated GEAK runner's terminal state."""
    versions = result.get("versions")
    recorder.record_geak_delegation(
        runner_status=settled_text(result, "status", default="unknown"),
        started_at=run.started_at or settled_text(result, "started_at"),
        ended_at=settled_text(result, "ended_at", default=datetime.now(timezone.utc).isoformat()),
        duration_sec=run.duration_sec if run.duration_sec is not None else result.get("duration_sec"),
        error_class=settled_text(result, "error_class"),
        error=settled_text(result, "error"),
        returncode=result.get("returncode"),
        runner_timeout_sec=(
            run.runner_timeout_sec if run.runner_timeout_sec is not None else result.get("runner_timeout_s")
        ),
        kill_timeout_sec=(run.kill_timeout_sec if run.kill_timeout_sec is not None else result.get("kill_timeout_s")),
        exp_root=settled_text(result, "exp_root") or settled_text(handoff, "exp_root"),
        eval_dir=settled_text(result, "eval_dir") or settled_text(handoff, "eval_dir"),
        report_path=settled_text(result, "report_path"),
        versions=versions if isinstance(versions, dict) else {},
        recovered_from_disk=run.recovered_from_disk,
        stages_reached=result.get("stages_reached"),
    )
