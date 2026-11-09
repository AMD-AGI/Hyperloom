# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The experiment blackboard: the one shared store a multibranch session's branches, rule gate, integrator, toolbox and
scheduler coordinate through, by its test records, claims with same-state dedup, branch logs, learnings and inbox.

The other modules code against the protocols of :mod:`hyperloom_multibranch.contracts.blackboard`; :class:`LocalBlackboard`
implements them on a session directory and is the fake the end-to-end skeleton runs on. What is built on the blackboard
sits under it: ``knowledge`` (Recipe KB write-back and warm start, the session breakdown, the Experience KB), ``rsi``
(the cross-session archive, lineage analysis and the startup brief, the specialists' direction menus and lessons, the
stall evidence, and the scheduling playbook with its replay and evaluation ladder), ``dashboard``, and ``tools``, the
blackboard as tools for a specialist's model.
"""

from hyperloom_multibranch.blackboard.local.feeds import STALL_TESTS, BriefTriggers
from hyperloom_multibranch.blackboard.local.blackboard import LocalBlackboard

__all__ = ["STALL_TESTS", "BriefTriggers", "LocalBlackboard"]
