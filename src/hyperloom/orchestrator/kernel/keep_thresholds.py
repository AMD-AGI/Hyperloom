# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much a kernel-stack step has to win by to be kept.

A leaf on purpose: the integrate handler reads it to grade a step and the
integrate result reads it to report one, and neither should have to import the
other to learn a number.
"""

from __future__ import annotations

#: A step in an incremental stack keeps on any honest gain: the stack's value is
#: the sum of several small wins, and a per-step bar set at the threshold a
#: first patch has to clear would reject every one of them.
STACK_INCREMENTAL_KEEP_THRESHOLD_PCT = 0.5
