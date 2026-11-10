# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Credit the predictor for grid variants orchestration copies off the queue.

Orchestration composes its own grids, and a variant it took from a predictor
row arrives labelled however orchestration chose. Matching each variant's
fingerprint against the predictor's recorded proposals recovers the proposer
without relying on that label, and without touching the grid, which stays
orchestration's. Only an exact match is claimed, so an edited copy keeps its
author.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyperloom.orchestrator.actions.executors._proposal_identity import content_fingerprint
from hyperloom.orchestrator.predictor.rows import PROVENANCE, QUEUE_DOMAIN


def _offered(rounds: Sequence[Any]) -> set[str]:
    """Fingerprints of every proposal the predictor queued."""
    return {
        content_fingerprint(proposal)
        for entry in rounds or ()
        if isinstance(entry, Mapping) and entry.get("domain") == QUEUE_DOMAIN
        for proposal in entry.get("proposal_set") or ()
        if isinstance(proposal, Mapping)
    }


def stamp_predictor_provenance(grid: Sequence[Any], rounds: Sequence[Any]) -> int:
    """Stamp exact copies of predictor proposals in ``grid``; returns how many were stamped."""
    offered = _offered(rounds)
    stamped = 0
    for variant in grid or ():
        if isinstance(variant, dict) and content_fingerprint(variant) in offered:
            variant["provenance"] = PROVENANCE
            stamped += 1
    return stamped
