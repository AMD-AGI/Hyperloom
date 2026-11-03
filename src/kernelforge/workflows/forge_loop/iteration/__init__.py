# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The jobs one iteration of the optimization loop is made of."""

#: Every module one iteration's work was split across, so a caller that has to
#: reach all of them does not carry its own list.
ITERATION_MODULES = (
    "attempt_verdict",
    "best_publication",
    "campaign_report",
    "campaign_runtime",
    "case_coverage",
    "ceiling_runtime",
    "iteration_phases",
    "iteration_types",
    "keep_checkpoint",
    "lane_runtime",
    "locked_run",
    "outcome_runtime",
    "resume_recovery",
    "round_admission",
    "scoring_runtime",
    "supervisor_review",
    "workspace_git",
)
