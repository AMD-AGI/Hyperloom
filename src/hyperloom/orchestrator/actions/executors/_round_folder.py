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


#: The file Magpie writes its own measurement to, inside the workspace it creates.
MAGPIE_REPORT_FILE = "benchmark_report.json"


def is_round_folder(path: Path) -> bool:
    """Whether ``path`` is an inference_suite run_dir, which every round that ran leaves with its report."""
    return RunLayout(path).report.is_file()


def round_report_path(workspace: Path) -> Path:
    """The report a round left in ``workspace``: an inference_suite run_dir's, else Magpie's."""
    return RunLayout(workspace).report if is_round_folder(workspace) else workspace / MAGPIE_REPORT_FILE


def read_round_launch(run_dir: Path) -> Launch | None:
    """The launch ``run_dir`` records; ``None`` when it holds no ``launch.json``.

    Raises ``OSError``, ``ValueError`` (``LaunchError``) for a launch off contract.
    """
    path = RunLayout(run_dir).launch
    return Launch.from_document(read_json_strict(path)) if path.is_file() else None


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
        return cls(layout, report, read_round_launch(run_dir))

    def measurement(self) -> dict[str, Any]:
        measurement = round_measurement(self.report, self.launch)
        if self.layout.inferencex_result.is_file():
            measurement["raw_result_path"] = str(self.layout.inferencex_result)
        return measurement

    def write_gpu_metrics(self, round_dir: Path) -> str | None:
        """Write ``gpu_metrics.json`` from the report's ``gpu_monitor`` into ``round_dir``; the path, or ``None``."""
        return write_gpu_metrics_from_report(round_dir, self.report.to_document(), source=REPORT_FILE)


__all__ = ["MAGPIE_REPORT_FILE", "RoundFolder", "is_round_folder", "read_round_launch", "round_report_path"]
