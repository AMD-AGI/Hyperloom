# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rows the kernel timeline records for the campaigns the KERNEL phase settles.

Every route here reports a campaign that has already finished, so each function
reads a result mapping and writes one lane of the timeline event.  The results
reach the phase from four different producers, which is what :class:`SettledRun`
exists for.
"""

from __future__ import annotations

import json
import logging as _logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.inference_optimizer.breakdown.recorder import tool_versions

from ..actions.stop_attribution import stopped_by_the_run_class

log = _logging.getLogger(__name__)


class SettledRun:
    """A settled campaign's result, read by the fact wanted rather than the key it landed under.

    The producers on these routes name the same fact differently -- a run's end is
    ``ended_at`` or ``ts``, its failure is ``error`` or ``skip_reason`` -- and a
    result carries only the spelling its own producer wrote.
    """

    def __init__(self, result: Mapping[str, Any]) -> None:
        self._result = result

    def text(self, *keys: str, default: str = "") -> str:
        """The first non-empty value under ``keys`` as text, else ``default``."""
        for key in keys:
            value = self._result.get(key)
            if value:
                return str(value)
        return default

    def status(self, default: str = "unknown") -> str:
        """The terminal state the run settled on."""
        return self.text("status", default=default)

    def started_at(self) -> str:
        """When the run started, as its own producer stamped it."""
        return self.text("started_at")

    def ended_at(self, default: str = "") -> str:
        """When the run ended, as its own producer stamped it."""
        return self.text("ended_at", "ts", default=default)

    def error_class(self) -> str:
        """The failure taxonomy class the run settled on."""
        return self.text("error_class")

    def error(self) -> str:
        """What the run reported went wrong."""
        return self.text("error")

    def failure_reason(self) -> str:
        """Why the run did not land, whichever way its producer spelled it."""
        return self.text("error", "skip_reason", "error_class")

    def eval_dir(self) -> str:
        """Where the run was to write what it evaluated."""
        return self.text("eval_dir")


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


def record_controller_rewrites(
    recorder: Any, session_dir: Path, result: Mapping[str, Any], *, macro_cycle: int
) -> None:
    """Record settled Controller integrations as Forge kernel rewrites."""
    integration = result.get("integration")
    rows = integration.get("results") if isinstance(integration, dict) else None
    if not isinstance(rows, list):
        return
    from hyperloom.inference_optimizer.breakdown.recorder.instrument import (
        record_backend_versions_and_timeline,
    )

    cycle = int(result.get("macro_cycle") or macro_cycle)
    run_id = str(result.get("run_id") or f"controller-c{cycle}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("operator_id"):
            continue
        record_backend_versions_and_timeline(
            session_dir,
            _rewrite_row(row, run_id=run_id, attempt_id=f"controller-c{cycle}-{index}"),
            recorder=recorder,
        )


def _integration_decision(status: str) -> str:
    """The verdict an integration status settles on."""
    if status == "kept":
        return "KEEP"
    if status.startswith("reverted"):
        return "REVERT"
    if status.startswith("skipped") or stopped_by_the_run_class(status):
        return "SKIPPED"
    return "FAILED"


def _micro_speedup(gain_pct: Any) -> float | None:
    """A percentage gain read back as a speedup ratio, or ``None`` when it is not a number."""
    if isinstance(gain_pct, bool) or not isinstance(gain_pct, (int, float)):
        return None
    return 1.0 + float(gain_pct) / 100.0


def _rewrite_row(row: Mapping[str, Any], *, run_id: str, attempt_id: str) -> dict[str, Any]:
    """The kernel-rewrite payload one settled Controller integration row reports."""
    kernel_id = str(row.get("operator_id") or "")
    status = str(row.get("status") or "unknown").lower()
    decision = _integration_decision(status)
    speedup = _micro_speedup(row.get("gain_pct"))
    qualified = kernel_id.split(":")
    return {
        "kernel_id": kernel_id,
        "kernel_name": qualified[2] if len(qualified) > 2 else "",
        "run_id": run_id,
        "status": status,
        "attempts": [
            {
                "attempt_id": attempt_id,
                "backend": "forge",
                "status": status,
                "decision": decision,
                "micro_speedup": speedup,
                # The integration status *is* this route's failure taxonomy --
                # ``reverted_apply_conflict``, ``skipped_dirty_worktree`` -- and an
                # integration row carries no separate class. Stamping it here is what
                # makes ``error_class`` answerable on this route: the GEAK and fusion
                # lanes both fill it, so a reader asking why a candidate did not land
                # had one field that was empty only for forge.
                "error_class": "" if status == "kept" else status,
                "error": str(row.get("reason") or ""),
            }
        ],
        "verification": {
            "best_attempt_id": attempt_id if decision == "KEEP" else "",
            "micro_speedup": speedup,
        },
        "proposal": {"decision": decision},
    }


def record_gemm_campaign(recorder: Any, session_dir: Path, result: Mapping[str, Any], *, macro_cycle: int) -> None:
    """Record a settled GEMM campaign in the Forge lane -- one row per tuner that ran.

    A campaign can run several tuners (e.g. ``fmoe_ck`` and ``a4w4_blockscale``) in one cycle;
    each has its own status, shape coverage and failure. Collapsing them into one row joined
    `tuner` names with a comma and let one tuner's `failure_reason` land on a row whose
    `gain_pct` came from a different, successful tuner -- a row that read as both "complete"
    and "failed" at once. The campaign's own validated gain (``result["e2e_gain_pct"]``, set by
    ``_validate_gemm_tuning_e2e`` only after e2e validation of the winning candidate) is
    attached to the tuner whose artifact was actually applied (``tuned_file``); a tuner's own
    best micro speedup is never substituted for it, since a per-shape timing ratio is not the
    axis the KEEP verdict was graded on.
    """
    facts = _campaign_facts(result, macro_cycle=macro_cycle)
    tuners = result.get("tuners_run")
    rows = [row for row in tuners if isinstance(row, dict)] if isinstance(tuners, list) else []
    for row in rows:
        _record_tuner_run(recorder, row, facts)
    if not rows:
        _record_untuned_campaign_run(recorder, result, facts)
    backend = SettledRun(result).text("backend", "engine").lower()
    if backend:
        tool_versions.record_tool_version(session_dir, tool=backend)


def _campaign_facts(result: Mapping[str, Any], *, macro_cycle: int) -> _CampaignFacts:
    """The campaign-wide half of every row this GEMM campaign reports."""
    fields = SettledRun(result)
    capture = result.get("shape_capture")
    capture = capture if isinstance(capture, dict) else {}
    workspace = fields.text("workspace")
    return _CampaignFacts(
        run_id=fields.text("task_id") or (Path(workspace).name if workspace else "") or f"gemm-c{macro_cycle}",
        status=fields.status(),
        shapes_total=result.get("shapes_total", capture.get("shape_count")),
        tuned_file=fields.text("tuned_file", "config_path"),
        # _validate_gemm_tuning_e2e writes the validated gain to e2e_gain_pct, not gain_pct -- no
        # producer on this path ever sets a bare "gain_pct" key.
        gain_pct=result.get("e2e_gain_pct"),
        graded_objective=fields.text("graded_objective"),
        micro_decision=fields.text("micro_decision", "decision"),
        integrate_ref=fields.text("integration_id"),
        started_at=fields.started_at(),
        ended_at=fields.ended_at(),
        duration_sec=result.get("duration_sec"),
    )


def _record_tuner_run(recorder: Any, row: Mapping[str, Any], facts: _CampaignFacts) -> None:
    """Record what one named tuner of a GEMM campaign did."""
    fields = SettledRun(row)
    tuner = fields.text("tuner")
    artifact = fields.text("artifact", "env_value")
    # The campaign's e2e-validated gain names the axis the KEEP was graded on; it
    # belongs on the tuner whose artifact was actually applied, not on every row
    # this cycle produced.
    applied = bool(facts.tuned_file) and artifact == facts.tuned_file
    recorder.record_gemm_tuning_run(
        run_id=f"{facts.run_id}-{tuner}" if tuner else facts.run_id,
        status=fields.status(facts.status),
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
        error_class=fields.error_class(),
        failure_reason=fields.failure_reason(),
    )


def _record_untuned_campaign_run(recorder: Any, result: Mapping[str, Any], facts: _CampaignFacts) -> None:
    """Record the single row a GEMM campaign that named no tuner reports for itself."""
    fields = SettledRun(result)
    recorder.record_gemm_tuning_run(
        run_id=facts.run_id,
        status=facts.status,
        shapes_total=facts.shapes_total,
        shapes_tuned=result.get("shapes_tuned"),
        config_path=facts.tuned_file,
        gain_pct=facts.gain_pct,
        graded_objective=facts.graded_objective,
        tuner=fields.text("tuner"),
        micro_decision=facts.micro_decision,
        integrate_ref=facts.integrate_ref,
        started_at=facts.started_at,
        ended_at=facts.ended_at,
        duration_sec=facts.duration_sec,
        error_class=fields.error_class(),
        failure_reason=fields.failure_reason(),
    )


def record_fusion_campaign(recorder: Any, session_dir: Path, result: Mapping[str, Any]) -> None:
    """Record a settled fusion campaign in the Forge lane."""
    fields = SettledRun(result)
    recorder.record_fusion_run(
        run_id=fields.text("fusion_run_id"),
        status=fields.status(),
        pattern=fields.text("pattern", "fusion_pattern"),
        target_module=fields.text("target_module", "kernel_name"),
        applied=bool(result.get("kept")),
        gain_pct=result.get("gain_pct"),
        patch_path=fields.text("patch_path", "source_patch"),
        micro_decision=fields.text("micro_decision", "decision"),
        integrate_ref=fields.text("integration_id"),
        started_at=fields.started_at(),
        ended_at=fields.ended_at(),
        duration_sec=result.get("duration_sec"),
        error_class=fields.error_class(),
        failure_reason=fields.failure_reason(),
    )
    tool_versions.record_tool_version(session_dir, tool="forge")
    agent_backend = fields.text("agent_backend").lower()
    if agent_backend:
        tool_versions.record_tool_version(session_dir, tool=agent_backend)


def note_absent_backends(result: dict[str, Any], *, handoff: Mapping[str, Any]) -> None:
    """Stamp the backends GEAK's environment report lists as absent onto ``result``."""
    absent = geak_absent_backends(SettledRun(result).eval_dir() or SettledRun(handoff).eval_dir())
    if not absent:
        return
    result["absent_backends"] = absent
    log.warning(
        "GEAK could not use these backends (provisioning gaps, not measured no-wins): %s",
        "; ".join(f"{name}: {probe}" for name, probe in absent.items()),
    )


def geak_absent_backends(eval_dir: str) -> dict[str, str]:
    """``{backend: probe}`` for the backends GEAK's environment report lists as absent.

    GEAK records them in ``<eval_dir>/env_report.json`` and deep in its final
    report; they are provisioning gaps, and nothing else carries them to the
    optimizer's log or reports.
    """
    if not eval_dir:
        return {}
    try:
        report = json.loads((Path(eval_dir) / "env_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    absent = report.get("absent_backends") if isinstance(report, dict) else None
    if not isinstance(absent, dict):
        return {}
    return {
        str(name): str(entry.get("probe") or "") if isinstance(entry, dict) else str(entry or "")
        for name, entry in absent.items()
    }


def record_geak_delegation(
    recorder: Any, result: Mapping[str, Any], *, handoff: Mapping[str, Any], run: RunnerRun
) -> None:
    """Record the delegated GEAK runner's terminal state."""
    fields = SettledRun(result)
    handed = SettledRun(handoff)
    versions = result.get("versions")
    recorder.record_geak_delegation(
        runner_status=fields.status(),
        started_at=run.started_at or fields.started_at(),
        ended_at=fields.text("ended_at", default=datetime.now(timezone.utc).isoformat()),
        duration_sec=run.duration_sec if run.duration_sec is not None else result.get("duration_sec"),
        error_class=fields.error_class(),
        error=fields.error(),
        returncode=result.get("returncode"),
        runner_timeout_sec=(
            run.runner_timeout_sec if run.runner_timeout_sec is not None else result.get("runner_timeout_s")
        ),
        kill_timeout_sec=(run.kill_timeout_sec if run.kill_timeout_sec is not None else result.get("kill_timeout_s")),
        exp_root=fields.text("exp_root") or handed.text("exp_root"),
        eval_dir=fields.eval_dir() or handed.eval_dir(),
        report_path=fields.text("report_path"),
        versions=versions if isinstance(versions, dict) else {},
        recovered_from_disk=run.recovered_from_disk,
        stages_reached=result.get("stages_reached"),
    )
