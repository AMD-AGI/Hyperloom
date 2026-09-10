# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workload knob fallbacks, shared by the CLI and the orchestrator. Stdlib-only.

These live here rather than in ``cli.parser`` because both sides need them and
``cli.parser`` imports from ``hyperloom.orchestrator``: an orchestrator module
reaching back into the parser for them would close an import cycle. ``common`` is
a leaf, so both can read the same numbers without one.

Applied when the operator passes neither the CLI flag nor an inherited value.
Flags default to ``None`` so "omitted" stays distinguishable from "typed the
default"; the resolver in ``cli`` applies these constants only for genuinely
unset knobs (issue #903).
"""

from __future__ import annotations

DEFAULT_ISL = 1024
DEFAULT_OSL = 1024
DEFAULT_CONC = 64
DEFAULT_TP = 1
DEFAULT_EP = 1
DEFAULT_PRECISION = "bf16"
