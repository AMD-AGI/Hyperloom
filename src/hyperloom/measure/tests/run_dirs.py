# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An inference_suite run_dir with only the report the measure readers take a round's values from."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

from inference_suite.contracts import ExitCodes, FailedStep, FixedMetrics, RunLayout, RunStatus
from inference_suite.tests.test_report_contract import _eval, _report

FIXED = {"output_throughput": 5000.0, "completed_requests": 64, "requested_requests": 64, "request_error_rate": 0.0}


def run_dir_report(
    fixed: Mapping[str, float | int | None] = FIXED,
    scores: Mapping[str, float] | None = None,
    eval_code: int | None = None,
    task_version: str | None = "3",
    failed_step: FailedStep | None = None,
    failure_detail: str | None = None,
) -> bytes:
    """The ``report.json`` of a round that measured ``fixed`` and, when ``scores`` is given, evaluated them.

    With ``failed_step``, the round failed at that step before it measured anything, for ``failure_detail`` (by default
    the step's name).
    """
    evaluation = None
    if scores is not None:
        evaluation = dataclasses.replace(_eval(), scores=dict(scores), task_version=task_version)
    codes = ExitCodes(client=0, eval=eval_code)
    if failed_step is not None:
        failed = {
            "run_status": RunStatus.FAILED,
            "failed_step": failed_step,
            "failure_detail": failure_detail or failed_step.value,
        }
        report = _report(profile=None, fixed=FixedMetrics.of({}), eval=None, codes=ExitCodes(client=None), **failed)
    else:
        report = _report(profile=None, fixed=FixedMetrics.of(fixed), eval=evaluation, codes=codes)
    return json.dumps(report.to_document()).encode()


def write_run_dir(
    directory: Path,
    fixed: Mapping[str, float | int | None] = FIXED,
    scores: Mapping[str, float] | None = None,
    eval_code: int | None = None,
    failed_step: FailedStep | None = None,
    failure_detail: str | None = None,
) -> Path:
    """A run_dir at ``directory`` holding :func:`run_dir_report` of the rest."""
    directory.mkdir(parents=True, exist_ok=True)
    RunLayout(directory).report.write_bytes(
        run_dir_report(fixed, scores, eval_code, failed_step=failed_step, failure_detail=failure_detail)
    )
    return directory
