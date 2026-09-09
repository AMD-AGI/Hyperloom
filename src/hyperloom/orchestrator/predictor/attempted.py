# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Queue identity for predictor rows, and what this session has attempted.

Both the pump and the request builder need to know which of the predictor's own
proposals are still sitting on the queue unmeasured: the pump so it does not
file a second copy, the request builder so the service can withhold them from
its answer. The pump already imports the request builder, so the shared query
lives here -- a leaf that imports neither -- rather than in either of them.

"Attempted" spans two ledgers that answer different questions.
``explore_search["tested"]`` is what actually ran. The untested queue is what
was offered and passed over, and only the second one explains a predictor that
re-proposes a lever with rising confidence: nothing had measured it, so nothing
marked it answered.
"""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.actions.executors._proposal_identity import (
    effective_fingerprint,
    is_executable,
    normalize_proposal,
)

#: Audit label on everything the predictor produced. Not a scheduling gate:
#: ``proposer_for()`` passes an unknown provenance through unchanged, so this
#: reaches the stack, the journal and the breakdown without a closed set.
PROVENANCE = "primatune"

#: ``domain`` on the queue rows, so the renderer and any offline analysis can
#: tell them from a specialist's. Deliberately not a member of the specialist
#: domain vocabulary: nothing dispatches it.
QUEUE_DOMAIN = "primatune"

#: Sort ahead of specialist proposals in the untested queue. A predictor row
#: carries no gap, so without an explicit priority it would rank below every
#: gap-anchored proposal -- see ``_untested_proposal_rows``.
QUEUE_PRIORITY = 1

#: Proposals that form the dispatchable batch. Matches the grid size
#: orchestration is told to target, which is the whole point: one decision
#: point's answer is one grid, so the batch can be dispatched as a unit.
MAX_PROPOSALS = 4

#: Proposals surfaced in total, batch plus overflow. Orchestration's own ceiling
#: is "4 per grid, hard maximum 6", so 6 is the most it could ever dispatch in
#: one round and anything past that is genuinely surplus.
#:
#: The batch size used to be a discard threshold, which threw away fresh
#: proposals for no gain: one round returned six eligible deltas and the fifth
#: and sixth were dropped while the queue still had room for them. Keeping them
#: also gives the batch something to refill from when exclusion thins it, which
#: is cheaper than spending a second generation round on the same question.
MAX_QUEUED = 6


def delta_pairs(extra_args: Any, extra_envs: Any) -> frozenset[tuple[str, str]]:
    """A launch delta as ``(name, value)`` pairs, for containment tests.

    Pairs rather than loose tokens because a token set cannot tell
    ``--a 1 --b 2`` from ``--a 2 --b 1`` -- both flatten to the same four
    tokens, and a subset test on them reports containment that never happened.
    A bare switch pairs with the empty string so it still has to be present.

    Args:
        extra_args (Any): The delta's launch flags, as a shell string.
        extra_envs (Any): The delta's environment variables.

    Returns:
        frozenset[tuple[str, str]]: Flag pairs plus ``env:``-prefixed variables.
    """
    tokens = str(extra_args or "").split()
    pairs: set[tuple[str, str]] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("-"):
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            pairs.add((name, value))
            continue
        value = ""
        if index < len(tokens) and not tokens[index].startswith("-"):
            value = tokens[index]
            index += 1
        pairs.add((token, value))
    if isinstance(extra_envs, dict):
        pairs.update((f"env:{key}", str(val)) for key, val in extra_envs.items())
    return frozenset(pairs)


def tested_delta_keys(state: Any | None) -> set[str]:
    """Delta fingerprints of every variant this session has benched.

    ``explore_search["tested"]`` is keyed on exactly this fingerprint, so the
    keys are the answer.

    Args:
        state (Any | None): The ``SharedState``.

    Returns:
        set[str]: The fingerprints, empty when nothing has run.
    """
    if state is None:
        return set()
    search = getattr(state, "explore_search", None)
    tested = search.get("tested") if isinstance(search, dict) else None
    return {str(key) for key in tested} if isinstance(tested, dict) else set()


def queued_unbenched(state: Any | None) -> dict[str, dict[str, Any]]:
    """The predictor's own queue rows that no explore round has benched yet.

    Restricted to :data:`QUEUE_DOMAIN` rounds on purpose. A specialist's
    proposal sitting on the same queue is already dispatchable, but suppressing
    a predictor row because a specialist happened to name the same flag would
    hand that flag's provenance to the specialist and make the predictor's
    contribution read smaller than it was -- and the arm-versus-arm comparison
    this feeds is the reason the label exists.

    Args:
        state (Any | None): The ``SharedState``.

    Returns:
        dict[str, dict[str, Any]]: Delta fingerprint to the normalized row,
            with ``name`` / ``votes`` / ``samples`` / ``round_id`` carried
            across for the request builder. Newest occurrence wins.
    """
    if state is None:
        return {}
    benched = tested_delta_keys(state)
    out: dict[str, dict[str, Any]] = {}
    for entry in getattr(state, "specialist_rounds", None) or []:
        if not isinstance(entry, dict) or str(entry.get("domain") or "") != QUEUE_DOMAIN:
            continue
        round_id = str(entry.get("round_id") or entry.get("task_id") or "")
        for proposal in entry.get("proposal_set") or []:
            if not isinstance(proposal, dict):
                continue
            row = normalize_proposal(proposal)
            if not is_executable(row):
                continue
            fingerprint = effective_fingerprint(row["extra_args"], row["extra_envs"])
            if fingerprint in benched:
                continue
            row["round_id"] = round_id
            for field in ("votes", "samples", "batch"):
                if field in proposal:
                    row[field] = proposal[field]
            out[fingerprint] = row
    return out
