# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The baseline rule answers from the round store as it stands, and the acquire agrees; the gate reads no store."""

from __future__ import annotations

import dis
from types import FunctionType

import pytest

from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.policy import gate as gate_module
from hyperloom.orchestrator.policy import validators as validators_module
from hyperloom.orchestrator.policy.bringup_round import ENABLEMENT_ROUND_IN_FLIGHT, bringup_round_denial
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.rehearsal import VirtualClock
from hyperloom.orchestrator.state.round_store import (
    BOOTED,
    EXCLUDED,
    EXPIRED_REAPED,
    RoundStore,
)

_LEASE = 600.0


@pytest.fixture
def store(tmp_path):
    """A :class:`RoundStore` over a real temp session database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    yield RoundStore(db)
    db.close()


@pytest.fixture
def clock():
    """A clock nobody waits for."""
    return VirtualClock()


async def _denied_rule(store: RoundStore, *, task_id: str = "") -> str | None:
    """The rule a baseline is refused under right now, or ``None`` when it is admitted."""
    denied = await bringup_round_denial("baseline", store, task_id=task_id)
    return None if denied is None else denied.rule


@pytest.mark.asyncio
async def test_a_baseline_is_denied_while_the_round_is_open_and_not_after(store, clock):
    """A live round denies; the round that booted and settled denies nothing."""
    assert await _denied_rule(store) is None
    opened = await store.open(
        "round-1",
        now_unix=clock.wall(),
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
    )
    assert opened.ok

    denied = await bringup_round_denial("baseline", store)
    assert denied is not None and denied.rule == ENABLEMENT_ROUND_IN_FLIGHT
    assert "round-1" in str(denied) and "baseline-1" in str(denied)

    clock.advance(120.0)
    settled = await store.settle(await store.get("round-1"), now_unix=clock.wall(), outcome=BOOTED)
    assert settled.ok

    assert await _denied_rule(store) is None


@pytest.mark.asyncio
async def test_the_rule_reads_the_store_at_each_call(store, clock):
    """Nothing is carried between calls: a round opened or settled in between changes the very next answer."""
    answers = [await _denied_rule(store)]
    await store.open("round-1", now_unix=clock.wall(), holder_task_id="baseline-1", lease_sec=_LEASE)
    answers.append(await _denied_rule(store))
    await store.settle(await store.get("round-1"), now_unix=clock.wall(), outcome=BOOTED)
    answers.append(await _denied_rule(store))
    await store.open("round-2", now_unix=clock.wall(), holder_task_id="baseline-2", lease_sec=_LEASE)
    answers.append(await _denied_rule(store))

    assert answers == [None, ENABLEMENT_ROUND_IN_FLIGHT, None, ENABLEMENT_ROUND_IN_FLIGHT]


@pytest.mark.asyncio
async def test_only_a_baseline_is_judged(store, clock):
    """The round excludes a second bring-up, not the work the round itself dispatches."""
    await store.open("round-1", now_unix=clock.wall(), holder_task_id="spec-1", lease_sec=_LEASE)

    for action_name in ("specialist", "integrate_patch", "explore", "targeted_build", ""):
        assert await bringup_round_denial(action_name, store) is None


@pytest.mark.asyncio
async def test_an_expired_round_denies_nothing_once_settled(store, clock):
    """Settling an expired round releases the machine."""
    await store.open(
        "round-1",
        now_unix=clock.wall(),
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
    )
    clock.advance(_LEASE + 1.0)
    await store.settle(await store.get("round-1"), now_unix=clock.wall(), outcome=EXPIRED_REAPED)

    assert await _denied_rule(store) is None

    # And the acquire agrees, which is the answer that counts.
    opened = await store.open(
        "round-next",
        now_unix=clock.wall(),
        holder_task_id="baseline-2",
        lease_sec=_LEASE,
    )
    assert opened.ok


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [_LEASE + 1.0, _LEASE * 100])
async def test_an_open_round_keeps_denying_after_lease_expiry_until_settled(store, clock, elapsed):
    """Elapsed time cannot release ownership; only the holder's settlement can."""
    opened = await store.open(
        "round-1",
        now_unix=clock.wall(),
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
    )
    assert opened.ok
    assert await _denied_rule(store) == ENABLEMENT_ROUND_IN_FLIGHT

    clock.advance(elapsed)
    assert await _denied_rule(store) == ENABLEMENT_ROUND_IN_FLIGHT

    blocked = await store.open(
        "round-next",
        now_unix=clock.wall(),
        holder_task_id="baseline-2",
        lease_sec=_LEASE,
    )
    assert not blocked.ok
    assert blocked.reason == EXCLUDED

    settled = await store.settle(await store.get("round-1"), now_unix=clock.wall(), outcome=BOOTED)
    assert settled.ok

    assert await _denied_rule(store) is None
    acquired = await store.open(
        "round-next",
        now_unix=clock.wall(),
        holder_task_id="baseline-2",
        lease_sec=_LEASE,
    )
    assert acquired.ok


@pytest.mark.asyncio
async def test_across_ticks_nothing_reaches_open_while_the_round_is_held(store, clock):
    """Ten ticks of a role that will not stop asking, against one live round."""
    held = await store.open(
        "round-held",
        now_unix=clock.wall(),
        holder_task_id="baseline-holder",
        lease_sec=_LEASE * 100,
    )
    assert held.ok

    for tick in range(10):
        clock.advance(30.0)
        assert await _denied_rule(store) == ENABLEMENT_ROUND_IN_FLIGHT
        # And the acquire agrees, for the attempt the rule stopped short of.
        attempt = await store.open(
            f"round-{tick}",
            now_unix=clock.wall(),
            holder_task_id=f"baseline-{tick}",
            lease_sec=_LEASE,
        )
        assert not attempt.ok
        assert attempt.reason == EXCLUDED

    # The original holder still holds it: nothing the loop did took the machine.
    still = await store.get("round-held")
    assert still is not None and still.holder_task_id == "baseline-holder"


@pytest.mark.asyncio
async def test_the_round_holders_own_bring_up_is_admitted_tick_after_tick(store, clock):
    """A revalidation baseline is admitted because it holds the round it opened."""
    opened = await store.open(
        "revalidation-1",
        now_unix=clock.wall(),
        holder_task_id="reval-1",
        lease_sec=_LEASE,
    )
    assert opened.ok

    for _tick in range(6):
        clock.advance(30.0)
        assert await _denied_rule(store, task_id="reval-1") is None

    # And every other baseline is still refused while that round holds.
    assert await _denied_rule(store, task_id="someone-else") == ENABLEMENT_ROUND_IN_FLIGHT
    assert await _denied_rule(store) == ENABLEMENT_ROUND_IN_FLIGHT


@pytest.mark.asyncio
async def test_a_round_handed_to_a_successor_admits_the_successor_and_refuses_the_old_holder(store, clock):
    """The holder is read with the round, so a hand-off moves the exemption with it."""
    await store.open("round-1", now_unix=clock.wall(), holder_task_id="spec-1", lease_sec=_LEASE)
    moved = await store.handoff(
        await store.get("round-1"), new_holder_task_id="integrate-1", now_unix=clock.wall(), lease_sec=_LEASE
    )
    assert moved.ok

    assert await _denied_rule(store, task_id="integrate-1") is None
    assert await _denied_rule(store, task_id="spec-1") == ENABLEMENT_ROUND_IN_FLIGHT


def _reachable_code(*entries: object) -> list:
    """Return every code object reachable from ``entries`` inside the gate's own modules.

    Args:
        entries: Functions to start from.

    Returns:
        list: Code objects of the entries, their nested comprehensions and
        closures, and of every gate or validator function they call by name.
    """
    modules = (gate_module, validators_module)
    seen: dict[int, object] = {}
    queue = [e.__code__ for e in entries if isinstance(e, FunctionType)]
    while queue:
        code = queue.pop()
        if id(code) in seen:
            continue
        seen[id(code)] = code
        queue.extend(c for c in code.co_consts if hasattr(c, "co_code"))
        for name in code.co_names:
            for module in modules:
                target = getattr(module, name, None)
                if isinstance(target, FunctionType) and target.__module__ in {m.__name__ for m in modules}:
                    queue.append(target.__code__)
    return list(seen.values())


def test_no_validator_on_the_intent_path_reaches_a_database_call():
    """The chokepoint runs on every intent; a lock taken here stalls the loop."""
    gate = PolicyGate()
    validators = [
        getattr(type(gate), name)
        for name in dir(type(gate))
        if name.startswith(("_validate", "validate")) and isinstance(getattr(type(gate), name, None), FunctionType)
    ]
    assert len(validators) >= 6, "the validator sweep found almost nothing; the naming changed"

    banned = {
        "execute",
        "executemany",
        "executescript",
        "fetchone",
        "fetchall",
        "commit",
        "cursor",
        "transaction",
        "connect",
        "sqlite3",
        "RoundStore",
        "SqliteConnection",
        "open_rounds",
        "held",
        "SpecialistGpuPool",
    }
    for code in _reachable_code(*validators):
        loaded = {instr.argval for instr in dis.get_instructions(code) if isinstance(instr.argval, str)}
        assert not (loaded & banned), f"{code.co_name} reaches {sorted(loaded & banned)}"

    # And the gate cannot name a store even to pass one on.
    assert not {"sqlite3", "SqliteConnection", "RoundStore", "SpecialistGpuPool"} & (
        set(vars(gate_module)) | set(vars(validators_module))
    )
