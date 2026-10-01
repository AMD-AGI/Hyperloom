# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round's steps in pipeline order. ``kind`` is script, agent (Claude Code) or gpu."""

from __future__ import annotations

from meta_rsi.rsi.pipeline import Step
from meta_rsi.rsi.steps import ab, check, data, levers, report

STEPS = (
    Step("fetch", "script", data.fetch),
    Step("analyze", "script", data.analyze, needs=("fetch",)),
    Step("findings", "agent", levers.findings, needs=("analyze",)),
    Step("implement", "agent", levers.implement, needs=("findings",)),
    Step("replay", "script", data.replay, needs=("implement",)),
    Step("suite", "script", check.suite, needs=("implement",)),
    Step("scenario", "script", data.scenario, needs=("analyze",)),
    Step("ab", "gpu", ab.ab, needs=("suite", "scenario")),
    Step("compare", "script", check.compare, needs=("ab",)),
    Step("diagnose", "agent", report.diagnose, needs=("compare",)),
    Step("report", "agent", report.report, needs=("compare",)),
    Step("publish", "script", report.publish, needs=("implement",)),
)
