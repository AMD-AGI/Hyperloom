# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gate refuses from facts read before it ran, and the acquire agrees.

The property under test is that a resource rule does not release itself. The
gate reads facts the repair pass left; what lifts a refusal is a re-read
showing the resource free, an explicit bypass, or the round's own holder
asking. Firing twice is not one of them, because a second bring-up against a
held machine fights the first for the same cards whether or not the gate
already said so once.

The rounds here are opened against a real database on the virtual clock the
rehearsal seam supplies, so a lease that runs out an hour after the round
opened is a line rather than an hour.
"""

from __future__ import annotations

import dis
from types import FunctionType

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.policy import gate as gate_module
from hyperloom.orchestrator.policy import projection as projection_module
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.policy.projection import (
    RULE_GPU_POOL_DISABLED,
    RULE_ROUND_IN_FLIGHT,
    ResourceFacts,
)
from hyperloom.orchestrator.rehearsal import VirtualClock
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state.round_store import (
    BOOTED,
    EXCLUDED,
    EXPIRED_REAPED,
    EXPIRED_UNREAPED,
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


def _gate(facts: ResourceFacts | None = None) -> PolicyGate:
    """A gate with no session facts of its own beyond the resource facts."""
    return PolicyGate(role_registry=default_role_registry(), resources=facts or ResourceFacts())


def _baseline() -> Intent:
    """A baseline delegate, the intent every rule here judges."""
    return Intent(type=IntentType.DELEGATE, payload={"action_name": "baseline", "params": {}})


async def _reread(store: RoundStore, facts: ResourceFacts, now: float) -> None:
    """Re-read the facts, as the repair pass does at the top of a tick."""
    facts.update(None, rounds=await store.excluding(now))


@pytest.mark.asyncio
async def test_the_gate_denies_while_the_exclusion_holds_and_not_after(store, clock):
    """A live round denies; the round that booted and settled denies nothing."""
    facts = ResourceFacts()
    gate = _gate(facts)
    opened = await store.open(
        "round-1",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    assert opened.ok

    await _reread(store, facts, clock.wall())
    with pytest.raises(PolicyDenied) as denied:
        gate.validate_intent("orchestration", _baseline())
    assert denied.value.rule == RULE_ROUND_IN_FLIGHT
    assert "baseline-1" in str(denied.value)

    clock.advance(120.0)
    settled = await store.settle(
        "round-1",
        holder_task_id="baseline-1",
        fence=opened.fence,
        outcome=BOOTED,
        now_unix=clock.wall(),
        request_id="req-settle",
    )
    assert settled.ok

    await _reread(store, facts, clock.wall())
    gate.validate_intent("orchestration", _baseline())


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [EXPIRED_UNREAPED, EXPIRED_REAPED])
async def test_a_settled_round_denies_nothing_however_it_ended(store, clock, outcome):
    """Settling releases, whether or not anything confirmed the holder dead.

    A process-group reap cannot prove a tree gone, so "unconfirmed" is the
    ordinary answer. Holding the machine on the strength of what nobody could
    observe is the shape that trapped a session before.
    """
    facts = ResourceFacts()
    gate = _gate(facts)
    opened = await store.open(
        "round-1",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    clock.advance(_LEASE + 1.0)
    await store.settle(
        "round-1",
        holder_task_id="baseline-1",
        fence=opened.fence,
        outcome=outcome,
        now_unix=clock.wall(),
        request_id="req-settle",
    )

    await _reread(store, facts, clock.wall())
    gate.validate_intent("orchestration", _baseline())

    # And the acquire agrees, which is the answer that counts.
    assert (
        await store.open(
            "round-next",
            holder_task_id="baseline-2",
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id="req-open-2",
        )
    ).ok


@pytest.mark.asyncio
async def test_an_open_round_nobody_settled_stops_denying_when_its_lease_runs_out(store, clock):
    """The exclusion is time-bounded, so no round can hold the machine for good."""
    facts = ResourceFacts()
    gate = _gate(facts)
    assert (
        await store.open(
            "round-1",
            holder_task_id="baseline-1",
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id="req-open",
        )
    ).ok

    await _reread(store, facts, clock.wall())
    with pytest.raises(PolicyDenied) as denied:
        gate.validate_intent("orchestration", _baseline())
    assert denied.value.rule == RULE_ROUND_IN_FLIGHT

    clock.advance(_LEASE + 1.0)
    await _reread(store, facts, clock.wall())
    gate.validate_intent("orchestration", _baseline())


@pytest.mark.asyncio
async def test_a_rule_that_denied_last_attempt_denies_this_one_too(store, clock):
    """The facts have not changed, so neither has the answer.

    A rule that let the second consecutive attempt through would hand the
    machine to a bring-up while the first one still holds it -- the case the
    rule exists for.
    """
    facts = ResourceFacts()
    gate = _gate(facts)
    await store.open(
        "round-1",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    await _reread(store, facts, clock.wall())

    for _attempt in range(2):
        with pytest.raises(PolicyDenied) as denied:
            gate.validate_intent("orchestration", _baseline())
        assert denied.value.rule == RULE_ROUND_IN_FLIGHT


@pytest.mark.asyncio
async def test_across_ticks_nothing_reaches_open_while_the_round_is_held(store, clock):
    """Ten ticks of a role that will not stop asking, against one live round.

    Every attempt is refused, and the sweep for a second holder confirms the
    gate never let one through: the round the loop started with is the round
    still standing at the end.
    """
    facts = ResourceFacts()
    gate = _gate(facts)
    held = await store.open(
        "round-held",
        holder_task_id="baseline-holder",
        lease_sec=_LEASE * 100,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    assert held.ok

    for tick in range(10):
        clock.advance(30.0)
        await _reread(store, facts, clock.wall())
        with pytest.raises(PolicyDenied) as denied:
            gate.validate_intent("orchestration", _baseline())
        assert denied.value.rule == RULE_ROUND_IN_FLIGHT
        # And the acquire agrees, for the attempt the gate stopped short of.
        attempt = await store.open(
            f"round-{tick}",
            holder_task_id=f"baseline-{tick}",
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id=f"req-{tick}",
        )
        assert not attempt.ok
        assert attempt.reason == EXCLUDED

    # The original holder still holds it: nothing the loop did took the machine.
    still = await store.get("round-held")
    assert still is not None and still.holder_task_id == "baseline-holder"


@pytest.mark.asyncio
async def test_the_round_holders_own_bring_up_is_admitted_tick_after_tick(store, clock):
    """A revalidation baseline is admitted because it holds the round it opened.

    The rule keeps refusing while the round stands, and a denial at dispatch
    does not defer the row, it cancels it -- so anything that refused the
    holder would cancel the very bring-up the round was opened for, tick after
    tick. The holder arm reads the acquire the row already won.
    """
    facts = ResourceFacts()
    gate = _gate(facts)
    opened = await store.open(
        "revalidation-1",
        holder_task_id="reval-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    assert opened.ok

    for _tick in range(6):
        clock.advance(30.0)
        await _reread(store, facts, clock.wall())
        gate.validate_dispatched_task("baseline", {"reason": "enablement_revalidation"}, task_id="reval-1")

    # And every other baseline is still refused while that round holds.
    with pytest.raises(PolicyDenied) as denied:
        gate.validate_dispatched_task("baseline", {}, task_id="someone-else")
    assert denied.value.rule == RULE_ROUND_IN_FLIGHT


def _reachable_code(*entries: object) -> list:
    """Return every code object reachable from ``entries`` inside the policy layer.

    Args:
        entries: Functions to start from.

    Returns:
        list: Code objects of the entries, their nested comprehensions and
        closures, and of every policy-layer function they call by name.
    """
    modules = (gate_module, projection_module)
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
    """The chokepoint runs on every intent; a lock taken here stalls the loop.

    Asserted on what the code can reach and on what the module can name, not on
    the import closure: ``policy.projection`` imports the round store, which
    imports ``sqlite3``, so the closure contains it and always will. What must
    not exist is a call.
    """
    gate = PolicyGate(role_registry=default_role_registry())
    validators = [
        getattr(type(gate), name)
        for name in dir(type(gate))
        if name.startswith(("_validate", "validate")) and isinstance(getattr(type(gate), name, None), FunctionType)
    ]
    assert len(validators) > 10, "the validator sweep found almost nothing; the naming changed"

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
        "excluding",
    }
    for code in _reachable_code(*validators):
        loaded = {instr.argval for instr in dis.get_instructions(code) if isinstance(instr.argval, str)}
        assert not (loaded & banned), f"{code.co_name} reaches {sorted(loaded & banned)}"

    # And the gate cannot name a connection even to pass one on.
    assert not {"sqlite3", "SqliteConnection", "RoundStore"} & set(vars(gate_module))


def test_the_resource_rules_refuse_nothing_until_the_facts_are_read():
    """A gate with no repair pass behind it sends every attempt to its acquire.

    Unread facts are zeros, and a zero pool read as a configured zero refuses
    every GPU dispatch on a session that never installed a pass.
    """
    request = {"needs_gpu": True, "gpu_count": 8}
    _gate().validate_intent("orchestration", _baseline())
    _gate()._validate_specialist_gpu_request(request)

    # Once read, the same request is judged against what was found.
    facts = ResourceFacts()
    facts.update(None)
    with pytest.raises(PolicyDenied) as denied:
        _gate(facts)._validate_specialist_gpu_request(request)
    assert denied.value.rule == RULE_GPU_POOL_DISABLED
