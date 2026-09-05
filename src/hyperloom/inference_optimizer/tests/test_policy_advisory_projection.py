# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gate refuses from a snapshot, and the acquire agrees -- without a GPU.

The property under test is that a resource rule does not release itself. The
gate reads a snapshot, so its refusal is labelled advisory and dated; what
lifts it is a newer snapshot showing the resource free, an explicit bypass, or
the round's own holder asking. Firing twice is not one of them, because a
second bring-up against a held machine fights the first for the same cards
whether or not the gate already said so once.

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
    RULE_ROUND_IN_FLIGHT,
    AdvisoryLedger,
    ResourceProjection,
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
_GRACE = 90.0


@pytest.fixture
def store(tmp_path):
    """A :class:`RoundStore` over a real temp session database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    yield RoundStore(db, reap_grace_sec=_GRACE)
    db.close()


@pytest.fixture
def clock():
    """A clock nobody waits for."""
    return VirtualClock()


def _gate(ledger: AdvisoryLedger | None = None) -> PolicyGate:
    """A gate with no session facts of its own beyond the projection."""
    return PolicyGate(role_registry=default_role_registry(), advisory=ledger)


def _baseline() -> Intent:
    """A baseline delegate, the intent every rule here judges."""
    return Intent(type=IntentType.DELEGATE, payload={"action_name": "baseline", "params": {}})


async def _snapshot(store: RoundStore, ledger: AdvisoryLedger, now: float) -> None:
    """Re-take the snapshot, as a coordinator tick does."""
    ledger.refresh(ResourceProjection.of(None, now_unix=now, rounds=await store.excluding(now)))


@pytest.mark.asyncio
async def test_the_gate_denies_while_the_exclusion_holds_and_not_after(store, clock):
    """A live round denies; the round that booted and settled denies nothing."""
    ledger = AdvisoryLedger()
    gate = _gate(ledger)
    opened = await store.open(
        "round-1",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    assert opened.ok

    await _snapshot(store, ledger, clock.wall())
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

    await _snapshot(store, ledger, clock.wall())
    gate.validate_intent("orchestration", _baseline())


@pytest.mark.asyncio
async def test_an_expired_round_never_confirmed_dead_still_denies(store, clock):
    """Nothing said the holder died, so nothing says the cards are free.

    The contrast is the reaped round beside it: a confirmed kill plus its grace
    is evidence, and evidence expires. An unreaped one carries none, so no
    amount of elapsed time turns it into a release.
    """
    ledger = AdvisoryLedger()
    gate = _gate(ledger)
    opened = await store.open(
        "round-unreaped",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    clock.advance(_LEASE + 1.0)
    await store.settle(
        "round-unreaped",
        holder_task_id="baseline-1",
        fence=opened.fence,
        outcome=EXPIRED_UNREAPED,
        now_unix=clock.wall(),
        request_id="req-settle",
    )

    clock.advance(86_400.0)
    await _snapshot(store, ledger, clock.wall())
    with pytest.raises(PolicyDenied) as denied:
        gate.validate_intent("orchestration", _baseline())
    assert denied.value.rule == RULE_ROUND_IN_FLIGHT
    assert "nothing ever confirmed the holder dead" in str(denied.value)

    # And the acquire agrees, which is the answer that counts.
    blocked = await store.open(
        "round-next",
        holder_task_id="baseline-2",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open-2",
    )
    assert not blocked.ok
    assert blocked.reason == EXCLUDED


@pytest.mark.asyncio
async def test_a_reaped_round_releases_once_its_grace_is_spent(store, clock):
    """The counterpart: a confirmed kill excludes only until the cards settle."""
    ledger = AdvisoryLedger()
    gate = _gate(ledger)
    opened = await store.open(
        "round-reaped",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    clock.advance(_LEASE + 1.0)
    killed = clock.wall()
    await store.settle(
        "round-reaped",
        holder_task_id="baseline-1",
        fence=opened.fence,
        outcome=EXPIRED_REAPED,
        now_unix=killed,
        request_id="req-settle",
        kill_confirmed_unix=killed,
    )

    await _snapshot(store, ledger, clock.wall())
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", _baseline())

    clock.advance(_GRACE + 1.0)
    await _snapshot(store, ledger, clock.wall())
    gate.validate_intent("orchestration", _baseline())


@pytest.mark.asyncio
async def test_a_rule_that_denied_last_attempt_denies_this_one_too(store, clock):
    """The snapshot has not changed, so neither has the answer.

    A rule that let the second consecutive attempt through would hand the
    machine to a bring-up while the first one still holds it -- the case the
    rule exists for.
    """
    ledger = AdvisoryLedger()
    gate = _gate(ledger)
    await store.open(
        "round-1",
        holder_task_id="baseline-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="req-open",
    )
    await _snapshot(store, ledger, clock.wall())

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
    ledger = AdvisoryLedger()
    gate = _gate(ledger)
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
        await _snapshot(store, ledger, clock.wall())
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
    ledger = AdvisoryLedger()
    gate = _gate(ledger)
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
        now = clock.wall()
        ledger.refresh(ResourceProjection.of(None, now_unix=now, rounds=await store.excluding(now)))
        gate.validate_dispatched_task("baseline", {"reason": "enablement_revalidation"}, task_id="reval-1")

    # And every other baseline is still refused while that round holds.
    with pytest.raises(PolicyDenied) as denied:
        gate.validate_dispatched_task("baseline", {}, task_id="someone-else")
    assert denied.value.rule == RULE_ROUND_IN_FLIGHT


def test_an_advisory_denial_says_so_and_dates_its_evidence():
    """A reader must be able to tell advice from a verdict, and how old it is."""
    ledger = AdvisoryLedger(
        ResourceProjection(
            taken_unix=1_800_000_123.5,
            excluding_round_id="round-1",
            excluding_round_holder="baseline-1",
        )
    )
    with pytest.raises(PolicyDenied) as denied:
        _gate(ledger).validate_intent("orchestration", _baseline())
    message = str(denied.value)
    assert "advisory" in message
    assert "1800000123.500" in message
    assert "at the acquire" in message


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
    advisory_entries = [
        projection_module.baseline_advisory,
        projection_module.specialist_gpu_advisory,
        projection_module.extend_lease_advisory,
        projection_module.AdvisoryLedger.deny,
    ]

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
    for code in _reachable_code(*validators, *advisory_entries):
        loaded = {instr.argval for instr in dis.get_instructions(code) if isinstance(instr.argval, str)}
        assert not (loaded & banned), f"{code.co_name} reaches {sorted(loaded & banned)}"

    # And the gate cannot name a connection even to pass one on.
    assert not {"sqlite3", "SqliteConnection", "RoundStore"} & set(vars(gate_module))
