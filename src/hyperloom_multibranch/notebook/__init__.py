# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Experiment notebook: test records, claims, same-state dedup, branch logs and the inbox.

The other modules code against the protocols of :mod:`hyperloom_multibranch.contracts.notebook`; :class:`LocalNotebook`
implements them on a session directory and is the fake the end-to-end skeleton runs on. What is built on the notebook
sits under it: ``knowledge`` (Recipe KB write-back and warm start, the session breakdown, the Experience KB), ``rsi``
(search-tree archive and replay, lineage analysis, startup brief, frontier scoring, dispersion prior), ``dashboard``,
and ``tools``, the notebook as tools for a specialist's model.
"""

from hyperloom_multibranch.notebook.local.feeds import STALL_TESTS, BriefTriggers
from hyperloom_multibranch.notebook.local.notebook import LocalNotebook

__all__ = ["STALL_TESTS", "BriefTriggers", "LocalNotebook"]
