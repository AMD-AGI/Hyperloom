# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read one inference_suite run_dir back into Hyperloom: its report and launch, its measurement, its GPU artifact.

A run_dir does not change once its round ends, so it is read once and never re-read while it settles. A report that
breaks its contract is a defect of the runner and raises. ``gpu_metrics.json`` is derived by the host, so it is
written into the Hyperloom round directory that holds the run_dir, never into the run_dir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.round_measurement import round_measurement
from hyperloom_common.jsonio import read_json_strict
from inference_suite.contracts import Report, RunLayout
from inference_suite.contracts.launch import Launch
from inference_suite.contracts.layout import REPORT_FILE

from ._gpu_metrics import write_gpu_metrics_from_report


def is_round_folder(path: Path) -> bool:
    """Whether ``path`` is an inference_suite run_dir, which every round that ran leaves with its report."""
    return RunLayout(path).report.is_file()


@dataclass(frozen=True)
class RoundFolder:
    """One finished run_dir; ``launch`` is ``None`` for a round refused before its launch was composed."""

    layout: RunLayout
    report: Report
    launch: Launch | None

    @classmethod
    def read(cls, run_dir: Path) -> RoundFolder:
        """Raises ``OSError``, ``ValueError`` (``ReportError``, ``LaunchError``) for a report or launch off contract."""
        layout = RunLayout(run_dir)
        report = Report.from_document(read_json_strict(layout.report))
        launch = Launch.from_document(read_json_strict(layout.launch)) if layout.launch.is_file() else None
        return cls(layout, report, launch)

    def measurement(self) -> dict[str, Any]:
        measurement = round_measurement(self.report, self.launch)
        if self.layout.inferencex_result.is_file():
            measurement["raw_result_path"] = str(self.layout.inferencex_result)
        return measurement

    def write_gpu_metrics(self, round_dir: Path) -> str | None:
        """Write ``gpu_metrics.json`` from the report's ``gpu_monitor`` into ``round_dir``; the path, or ``None``."""
        return write_gpu_metrics_from_report(round_dir, self.report.to_document(), source=REPORT_FILE)


__all__ = ["RoundFolder", "is_round_folder"]
