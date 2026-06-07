# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Persistent findings sink.

The reactor hands :class:`~robustness_agent.decision.action_ladder.Finding`
records to a sink each tick. M1 only writes a JSONL file under the
session directory; the M5 milestone replaces / augments this with an
HTTP POST to robustness-server.
"""

from .sink import FindingSink, FindingSinkConfig

__all__ = ["FindingSink", "FindingSinkConfig"]
