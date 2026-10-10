# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Credit the predictor for grid variants orchestration copies off the queue.

Orchestration composes its own grids, and a variant it took from a predictor
row arrives labelled however orchestration chose. Matching each variant's
fingerprint against the predictor's recorded proposals recovers the proposer
without relying on that label, and without touching the grid, which stays
orchestration's. Only an exact match is claimed, so an edited copy keeps its
author. A variant that contains a proposal's whole delta names it in
``primatune_contains``, for measurement only: nothing schedules or promotes on
it, and specialist variants are left out because a specialist never sees the
queue.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyperloom.orchestrator.actions.executors._proposal_identity import content_fingerprint, normalize_proposal
from hyperloom.orchestrator.predictor.rows import PROVENANCE, QUEUE_DOMAIN


def delta_pairs(extra_args: str, extra_envs: Mapping[str, str]) -> frozenset[tuple[str, str]]:
    """A launch delta as ``(flag, value)`` pairs plus ``env:``-prefixed variables; a bare switch pairs with ``""``.

    Pairs rather than tokens: ``--a 1 --b 2`` and ``--a 2 --b 1`` flatten to the
    same tokens, and a subset test on those reports containment that is not there.
    """
    tokens = str(extra_args or "").split()
    pairs: set[tuple[str, str]] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("-"):
            continue
        name, sep, value = token.partition("=")
        if not sep and index < len(tokens) and not tokens[index].startswith("-"):
            value = tokens[index]
            index += 1
        pairs.add((name, value))
    pairs.update((f"env:{key}", str(value)) for key, value in (extra_envs or {}).items())
    return frozenset(pairs)


def _offered(rounds: Sequence[Any]) -> tuple[set[str], list[tuple[str, frozenset[tuple[str, str]]]]]:
    """Fingerprints of every proposal the predictor queued, and each one's name with its delta pairs."""
    offered: set[str] = set()
    deltas: list[tuple[str, frozenset[tuple[str, str]]]] = []
    proposals = (
        proposal
        for entry in rounds or ()
        if isinstance(entry, Mapping) and entry.get("domain") == QUEUE_DOMAIN
        for proposal in entry.get("proposal_set") or ()
        if isinstance(proposal, Mapping)
    )
    for proposal in proposals:
        offered.add(content_fingerprint(proposal))
        fields = normalize_proposal(proposal)
        pairs = delta_pairs(fields["extra_args"], fields["extra_envs"])
        # An empty delta is contained by every variant.
        if pairs:
            deltas.append((fields["name"], pairs))
    return offered, deltas


def stamp_predictor_provenance(grid: Sequence[Any], rounds: Sequence[Any]) -> tuple[int, int]:
    """Stamp exact copies of predictor proposals in ``grid`` and mark the supersets; returns ``(stamped, containing)``."""
    offered, deltas = _offered(rounds)
    stamped = containing = 0
    if not offered:
        return stamped, containing
    for variant in grid or ():
        if not isinstance(variant, dict):
            continue
        if content_fingerprint(variant) in offered:
            variant["provenance"] = PROVENANCE
            stamped += 1
        elif not variant.get("domain") and not str(variant.get("provenance") or "").startswith("specialist"):
            fields = normalize_proposal(variant)
            pairs = delta_pairs(fields["extra_args"], fields["extra_envs"])
            names = [name for name, delta in deltas if delta <= pairs]
            if names:
                variant["primatune_contains"] = names
                containing += 1
    return stamped, containing
