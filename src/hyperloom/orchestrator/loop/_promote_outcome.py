# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a settled task's promote handler reports back to the writeback that called it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperloom.inference_optimizer.session.optimization_journal import Verdict


@dataclass
class _PromoteOutcome:
    """Mutable carrier threaded through the per-kind promote handlers;
    ``early_return`` skips the shared audit/save tail (conc_sweep).

    ``verdict`` is the settlement's single answer to "was this adopted";
    ``adopted_variants`` names, by fingerprint, the explore winners whose lift
    landed."""

    verdict: Verdict
    adopted_variants: set[str] = field(default_factory=set)
    changed: bool = False
    audit_decision: str | None = None
    audit_extras: dict[str, Any] = field(default_factory=dict)
    early_return: bool = False
