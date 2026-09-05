# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The progress budget: what it counts, where it is kept, and that it ends.

The incident this exists for is a session that never stopped. Progress between
rounds was decided by comparing two failure identities built from log text, one
side of which was a raw tail carrying per-run paths, so almost every round
compared unequal and read as forward progress -- and every forward round reset
the counter that was supposed to stop the run. The corpus contains no session
that ever reached the cap.

What replaces it is arithmetic over a ledger, and the two properties worth
testing are exactly the two the old scheme lacked: the budget cannot be handed
back, and the total is bounded no matter what the boots do.
"""

from __future__ import annotations

import re
import time
from itertools import count
from pathlib import Path

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.bringup.budget import (
    ADVANCE,
    DIGEST_CREDITS,
    EVIDENCE_STALL_BUDGET,
    LADDER_ADVANCES,
    NEW_DIGEST,
    SESSION_OBSERVATION_CEILING,
    STALL,
    charge_of,
    digest_of,
    round_advanced,
    session_budget,
    stage_of,
)
from hyperloom.orchestrator.bringup.observe import observe_bringup
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.rehearsal import boot_log_for
from hyperloom.orchestrator.state.round_store import BOOTED, FAILED, RoundStore

#: The package root, for the structural scan.
_PACKAGE = Path(__file__).resolve().parents[3]

#: An attribute assignment to either derived counter. Neither may have one: a
#: field is written after the thing it counts happened, and a crash in that gap
#: gives the credit back. The ledger's own SQL is not matched -- there the name
#: is a column, never an attribute of a state object.
_STATE_WRITE = re.compile(r"\.(stage_high_water|digests_spent|stall_spent|digest_credits|stall_streak)\s*=[^=]")


@pytest.fixture
def store(tmp_path):
    """A real round store over a scratch session database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield RoundStore(db)
    db.close()


def _wall(letter: str) -> str:
    """Render a failure whose identity is distinct from every other letter's.

    The traceback frame is what makes an unrecognised wall placeable and
    dedupable; the letter is what makes this wall a different one. It is a
    letter rather than a number because the digest masks digit runs, so two
    walls that differ only in an operand are deliberately the same wall.

    Args:
        letter: The token that distinguishes this wall.

    Returns:
        str: The failure text a scenario attempt would print.
    """
    return "\n".join(
        (
            "Traceback (most recent call last):",
            '  File "engine/core.py", line 214, in _advance',
            f"    raise RuntimeError(gap_{letter})",
            f"RuntimeError: capability gap_{letter} is not implemented",
        )
    )


async def _observe(store: RoundStore, index: int, *, server_log: str) -> str:
    """Classify one boot and charge its observation to the ledger.

    Args:
        store: The round store.
        index: The round's index, used to make request ids unique.
        server_log: The server child's log for this attempt.

    Returns:
        str: The failure digest that was charged.
    """
    observation = observe_bringup(server_log=server_log).observation
    digest = digest_of(observation)
    await store.observe(
        f"round-{index}",
        actor_task_id=f"task-{index}",
        stage=stage_of(observation),
        failure_digest=digest,
        now_unix=time.time(),
        request_id=f"observe-{index}",
    )
    return digest


# --- the bound -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_round_that_peels_one_blocker_forever_still_terminates(store):
    """The adversary the old detector could not stop, stopped by counting.

    Every round clears the previous wall and reveals a brand-new one, so every
    round is genuine forward progress by any predicate that asks "is this a
    different failure than last time". The ladder runs out, the digest credits
    run out, and the session stops -- inside the ceiling, with no predicate
    anywhere deciding it had had enough.
    """
    digests: list[str] = []
    budget = None
    for index in count():
        assert index < SESSION_OBSERVATION_CEILING, "the session funded more rounds than the budget allows"
        # Climb while there is ladder left, then keep peeling blockers at the
        # top of it -- which is the shape of a session that cannot boot.
        stages = list(LadderStage)
        stage = stages[min(index, len(stages) - 1)]
        digests.append(
            await _observe(store, index, server_log=boot_log_for(stage, message=_wall(chr(ord("a") + index))))
        )
        budget = await session_budget(store)
        if budget.exhausted:
            break

    assert budget is not None and budget.exhausted
    assert budget.reason
    assert budget.observations <= SESSION_OBSERVATION_CEILING
    # And the run really was the pathological one: no two rounds hit the same
    # wall, and every round either climbed or produced a failure the session
    # had never seen -- so nothing here would have been refused by a test that
    # asks only whether the failure changed.
    assert len(set(digests)) == len(digests)
    assert budget.advances + budget.digests_spent == budget.observations
    assert budget.stall_spent == 0


@pytest.mark.asyncio
async def test_rounds_that_show_nothing_new_spend_the_stall_budget(store):
    """The same wall, round after round, ends the run on the stall budget."""
    for index in range(EVIDENCE_STALL_BUDGET + 1):
        await _observe(store, index, server_log=boot_log_for(LadderStage.ENGINE_INIT, message=_wall("a")))
        budget = await session_budget(store)
        if index < EVIDENCE_STALL_BUDGET:
            assert not budget.exhausted
    budget = await session_budget(store)
    assert budget.stall_spent == EVIDENCE_STALL_BUDGET
    assert budget.exhausted


def test_the_ceiling_is_the_sum_of_three_bounded_quantities():
    """Every observation lands in exactly one bucket, so the sum is the bound."""
    assert LADDER_ADVANCES == len(LadderStage)
    assert SESSION_OBSERVATION_CEILING == LADDER_ADVANCES + DIGEST_CREDITS + EVIDENCE_STALL_BUDGET
    assert charge_of(stage=50, digest="d", high_water=40, seen=frozenset()) == ADVANCE
    assert charge_of(stage=40, digest="d", high_water=40, seen=frozenset()) == NEW_DIGEST
    assert charge_of(stage=40, digest="d", high_water=40, seen=frozenset({"d"})) == STALL
    assert charge_of(stage=0, digest="", high_water=0, seen=frozenset()) == STALL


# --- where the counters live ----------------------------------------------


def test_neither_counter_has_a_write_site_in_session_state():
    """The budget is derived by query; nothing assigns it, so nothing rolls it back.

    A credit stored as a field is written after the round it paid for ran. A
    crash between those two moments resurrects it, and an allowance that can be
    resurrected bounds nothing -- which is the whole reason the count lives in
    the append-only ledger instead.
    """
    offenders = [
        f"{path.relative_to(_PACKAGE)}:{index}: {line.strip()}"
        for path in _PACKAGE.rglob("*.py")
        if path != Path(__file__)
        for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if _STATE_WRITE.search(line)
    ]
    assert offenders == []


@pytest.mark.asyncio
async def test_only_an_observation_raises_the_high_water_mark(store):
    """Renew and settle carry a round through its life; neither claims progress."""
    now = time.time()
    opened = await store.open("r1", holder_task_id="holder", lease_sec=600.0, now_unix=now, request_id="open-1")
    assert opened.ok

    await store.renew("r1", holder_task_id="holder", fence=1, lease_sec=600.0, now_unix=now, request_id="renew-1")
    assert (await store.get("r1")).stage_high_water == 0

    await store.observe(
        "r1",
        actor_task_id="holder",
        stage=int(LadderStage.ENGINE_INIT),
        failure_digest="d",
        now_unix=now,
        request_id="observe-1",
    )
    assert (await store.get("r1")).stage_high_water == int(LadderStage.ENGINE_INIT)

    # A shallower boot cannot unwatch what a deeper one was seen to reach.
    await store.observe(
        "r1",
        actor_task_id="holder",
        stage=int(LadderStage.IMPORT),
        failure_digest="e",
        now_unix=now,
        request_id="observe-2",
    )
    assert (await store.get("r1")).stage_high_water == int(LadderStage.ENGINE_INIT)

    await store.settle("r1", holder_task_id="holder", fence=1, outcome=FAILED, now_unix=now, request_id="settle-1")
    assert (await store.get("r1")).stage_high_water == int(LadderStage.ENGINE_INIT)
    assert await store.stage_high_water() == int(LadderStage.ENGINE_INIT)


@pytest.mark.asyncio
async def test_an_observation_is_recorded_even_when_no_round_holds_it(store):
    """A boot watched outside a round still spends budget.

    The alternative is an escape hatch: a caller that lost the round, or never
    had one, would hand back the allowance its boot had already used.
    """
    await store.observe(
        "",
        actor_task_id="",
        stage=0,
        failure_digest="",
        now_unix=time.time(),
        request_id="observe-unheld",
    )
    budget = await session_budget(store)
    assert (budget.observations, budget.stall_spent) == (1, 1)


@pytest.mark.asyncio
async def test_a_settled_round_does_not_refuse_a_late_observation(store):
    """Evidence is not ownership, so it is never rejected on a stale round."""
    now = time.time()
    await store.open("r1", holder_task_id="holder", lease_sec=600.0, now_unix=now, request_id="open-1")
    await store.settle("r1", holder_task_id="holder", fence=1, outcome=BOOTED, now_unix=now, request_id="settle-1")
    result = await store.observe(
        "r1",
        actor_task_id="someone-else",
        stage=int(LadderStage.HTTP_READY),
        failure_digest="",
        now_unix=now,
        request_id="observe-late",
    )
    assert result.ok
    assert (await session_budget(store)).observations == 1


# --- what a round is kept for ---------------------------------------------


def test_a_round_advances_on_a_deeper_wall_or_a_different_one():
    """Both halves come from the same producer, so the comparison means something."""
    shallow = observe_bringup(server_log=boot_log_for(LadderStage.CONFIG_VALIDATE, message=_wall("a"))).observation
    deeper = observe_bringup(server_log=boot_log_for(LadderStage.ENGINE_INIT, message=_wall("a"))).observation
    sideways = observe_bringup(server_log=boot_log_for(LadderStage.ENGINE_INIT, message=_wall("b"))).observation

    assert round_advanced(shallow, deeper) is True
    assert round_advanced(deeper, sideways) is True
    assert round_advanced(deeper, deeper) is False
    assert round_advanced(deeper, shallow) is False
    assert round_advanced(None, shallow) is True
    assert round_advanced(shallow, None) is False
