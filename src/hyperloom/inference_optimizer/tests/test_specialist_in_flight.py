# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A catalogue domain holds at most one queued or running single-domain specialist, whatever gap each names."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.policy.specialist_in_flight import SPECIALIST_DOMAIN_IN_FLIGHT
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.specialists.in_flight import held_domain, in_flight_domains

_SYSTEM = "system_specialist"
_IDLE_GAP = "gap.system.gpu_idle_85pct"


@pytest.fixture
async def coord(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    silent = ScriptedPlan(turns=[])
    c = Coordinator(
        make_session_dir(),
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    c.shared_state.phase = "FRAMEWORK_AGENT"
    yield c
    await c.stop()


async def _delegate(c: Coordinator, **params: Any) -> None:
    payload = {"action_name": "specialist", "params": {"mode": "research", **params}}
    await c.router.handle_delegate("orchestration", Intent(type=IntentType.DELEGATE, payload=payload))


def _ledger_gap(c: Coordinator, *, domain_hint: str) -> None:
    c.shared_state.upsert_gap({"canonical_id": _IDLE_GAP, "symptom": "GPU 85% idle", "domain_hint": domain_hint})


async def _queued_domains(c: Coordinator) -> list[str]:
    return [str(t.params.get("domain") or t.params.get("tags")) for t in await c.tasks.by_state("queued")]


async def _in_flight_denials(c: Coordinator) -> int:
    observed = await c.bus.tail(topic="observation")
    return sum(
        m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == SPECIALIST_DOMAIN_IN_FLIGHT
        for m in observed
    )


async def test_a_second_delegate_for_a_queued_domain_is_refused_whatever_gap_it_names(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.system.gpu_idle_85pct_host_pacing")
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.launch_overhead.idle_85pct_conc4_decode")

    assert await _queued_domains(coord) == [_SYSTEM]
    assert await _in_flight_denials(coord) == 1


async def test_a_running_specialist_holds_its_domain(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.a")
    (first,) = await coord.tasks.by_state("queued")
    await coord.tasks.transition(first.task_id, "running", evidence={})

    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.b")

    assert await coord.tasks.by_state("queued") == []
    assert await _in_flight_denials(coord) == 1


async def test_a_finished_holder_frees_its_domain_and_other_domains_stay_open(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.a")
    await _delegate(coord, domain="compiler_specialist", gap_canonical_id="gap.c")
    holder = next(t for t in await coord.tasks.by_state("queued") if t.params["domain"] == _SYSTEM)
    await coord.tasks.transition(holder.task_id, "running", evidence={})
    await coord.tasks.transition(holder.task_id, "succeeded", evidence={})

    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.b")

    assert sorted(await _queued_domains(coord)) == ["compiler_specialist", _SYSTEM]
    assert await _in_flight_denials(coord) == 0


async def test_a_tag_only_dispatch_holds_the_domain_its_tag_resolves_to(coord):
    await _delegate(coord, tags=["systems"], gap_canonical_id="gap.a")
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.b")

    assert len(await coord.tasks.by_state("queued")) == 1
    assert await _in_flight_denials(coord) == 1


@pytest.mark.parametrize("shape", [{"task_description": "trace the decode host gap"}, {"scope": "domain"}])
async def test_a_dispatch_citing_only_a_gap_holds_the_domain_the_gap_hints(coord, shape):
    _ledger_gap(coord, domain_hint=_SYSTEM)
    await _delegate(coord, gap_canonical_id=_IDLE_GAP, sub_kind="host_pacing", **shape)
    await _delegate(coord, gap_canonical_id=_IDLE_GAP, sub_kind="launch_overhead", **shape)
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.b")

    assert await _queued_domains(coord) == [_SYSTEM]
    assert await _in_flight_denials(coord) == 2


async def test_a_one_tag_dispatch_holds_the_domain_its_cited_gap_backfills(coord):
    _ledger_gap(coord, domain_hint="compiler_specialist")
    await _delegate(coord, tags=["systems"], gap_canonical_id=_IDLE_GAP, sub_kind="host_pacing")
    await _delegate(coord, tags=["systems"], gap_canonical_id=_IDLE_GAP, sub_kind="launch_overhead")
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.s")

    assert sorted(await _queued_domains(coord)) == ["compiler_specialist", _SYSTEM]
    assert await _in_flight_denials(coord) == 1


async def test_a_single_tag_domains_dispatch_is_refused_like_a_single_domain_one(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.a")
    await _delegate(coord, scope="domains", tags=["systems"], gap_canonical_id="gap.b")

    assert len(await coord.tasks.by_state("queued")) == 1
    assert await _in_flight_denials(coord) == 1


async def test_a_queued_cross_domain_specialist_holds_none_of_its_domains(coord):
    await _delegate(coord, tags=["systems", "compiler"], gap_canonical_id="gap.a")
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.b")
    await _delegate(coord, domain="compiler_specialist", gap_canonical_id="gap.c")

    assert len(await coord.tasks.by_state("queued")) == 3
    assert await _in_flight_denials(coord) == 0


async def test_a_cross_domain_dispatch_is_admitted_while_one_of_its_domains_is_held(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.a")
    await _delegate(coord, tags=["systems", "compiler"], gap_canonical_id="gap.b")

    assert len(await coord.tasks.by_state("queued")) == 2
    assert await _in_flight_denials(coord) == 0


async def test_freeform_dispatches_hold_no_domain(coord):
    await _delegate(coord, scope="freeform", task_description="trace the decode host gap")
    await _delegate(coord, scope="freeform", task_description="audit the attention backend choice")

    assert len(await coord.tasks.by_state("queued")) == 2
    assert await _in_flight_denials(coord) == 0


async def test_orchestration_sees_which_domains_queued_specialists_hold(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.a")
    (holder,) = await coord.tasks.by_state("queued")

    in_flight = coord.conversation._context_running_tasks_reader()

    assert f"queued specialists (each holds its domain): {_SYSTEM} task_id={holder.task_id}" in in_flight


def test_the_first_specialist_of_each_domain_holds_it():
    first = SimpleNamespace(kind="specialist", params={"domain": _SYSTEM})
    tasks = [
        SimpleNamespace(kind="explore", params={"domain": "compiler_specialist"}),
        first,
        SimpleNamespace(kind="specialist", params={"tags": ["systems"]}),
        SimpleNamespace(kind="specialist", params={"scope": "freeform", "task_description": "x"}),
    ]

    assert in_flight_domains(tasks) == {_SYSTEM: first}


@pytest.mark.parametrize(
    ("params", "held"),
    [
        ({"domain": _SYSTEM}, _SYSTEM),
        ({"tags": ["systems"]}, _SYSTEM),
        ({"scope": "domains", "tags": ["systems"]}, _SYSTEM),
        ({"tags": ["systems", "compiler"]}, ""),
        ({"scope": "freeform", "domain": _SYSTEM, "task_description": "x"}, ""),
        ({"domain": "custom_specialist", "scope": "domain"}, ""),
    ],
)
def test_only_a_single_domain_dispatch_holds_a_domain(params, held):
    assert held_domain(params) == held
