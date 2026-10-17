# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public name of the ``analysis.json`` reader, for callers outside the trace-analysis package."""

from __future__ import annotations

from ._analysis_json import load_report_tasks

__all__ = ["load_report_tasks"]
