# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KernelForge's decision labels against the ``outcome.decision`` values its Hyperloom-KB declaration accepts."""

from __future__ import annotations

from hyperloom_kb.collect import load_mapping
from kernelforge.workflows.forge_loop.campaign.decision_labels import DECISION_LABELS


def test_the_kernelforge_declaration_accepts_exactly_the_decision_labels() -> None:
    decision = load_mapping("kernelforge-v1").declaration.field("outcome", "decision")

    assert set(decision.values) == DECISION_LABELS
