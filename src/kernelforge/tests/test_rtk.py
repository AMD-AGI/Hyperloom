# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the rtk token-filter helpers (rtk.py).

``prefix()`` was added so agent prompts advertise the ``rtk`` wrapper ONLY when
the binary is actually on PATH — otherwise the agent would prefix every shell
command with a missing binary. This pins prefix()/wrap_command consistency with
is_available(), independent of whether rtk happens to be installed."""

from __future__ import annotations
