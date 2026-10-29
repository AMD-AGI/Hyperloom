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
from hyperloom.orchestrator.loop.proposals import PendingProposal
from hyperloom.orchestrator.policy.specialist_in_flight import (
    SPECIALIST_DOMAIN_IN_FLIGHT,
    SPECIALIST_MODULE_IN_FLIGHT,
)
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.specialists.in_flight import held_domain, held_module_group, in_flight_domains
from hyperloom.orchestrator.specialists.profile import SCOPE_DOMAIN, resolve_specialist_profile

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


def _approved_specialist(msg_id: str, **params: Any) -> PendingProposal:
    payload = {"action_name": "specialist", "params": {"mode": "research", **params}}
    return PendingProposal(msg_id, "orchestration", "specialist", predicted_gain_pct=0.0, payload=payload)


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


async def test_a_scope_domain_dispatch_over_two_tags_holds_the_domain_it_is_anchored_on(coord):
    await _delegate(coord, domain=_SYSTEM, gap_canonical_id="gap.a")
    await _delegate(coord, scope="domain", tags=["systems", "compiler"], gap_canonical_id="gap.b")

    assert len(await coord.tasks.by_state("queued")) == 1
    assert await _in_flight_denials(coord) == 1


async def test_an_approved_specialist_proposal_is_judged_like_a_delegate_when_it_materializes(coord):
    _ledger_gap(coord, domain_hint=_SYSTEM)
    await coord.proposals.materialize_approved_proposal(_approved_specialist("p1", gap_canonical_id=_IDLE_GAP))
    await coord.proposals.materialize_approved_proposal(
        _approved_specialist("p2", domain=_SYSTEM, gap_canonical_id="b")
    )

    assert await _queued_domains(coord) == [_SYSTEM]
    observed = await coord.bus.tail(topic="observation")
    skipped = [m.payload["reason"] for m in observed if m.payload.get("kind") == "proposal_materialize_skipped"]
    assert skipped == [SPECIALIST_DOMAIN_IN_FLIGHT]


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
    "params",
    [
        {"domain": _SYSTEM},
        {"tags": ["systems"]},
        {"scope": "domains", "tags": ["systems"]},
        {"scope": "domain", "tags": ["systems", "compiler"]},
        {"tags": ["systems", "compiler"]},
        {"domain": _SYSTEM, "tags": ["systems", "compiler"]},
        {"scope": "freeform", "domain": _SYSTEM, "task_description": "x"},
    ],
)
def test_a_dispatch_holds_its_domain_exactly_when_its_profile_scope_is_domain(params):
    single_domain = resolve_specialist_profile(params).scope == SCOPE_DOMAIN

    assert held_domain(params) == (_SYSTEM if single_domain else "")


def test_the_coordinator_owned_custom_specialist_holds_no_domain():
    assert held_domain({"domain": "custom_specialist", "scope": "domain"}) == ""


async def _queued_mandates(c: Coordinator) -> list[str]:
    return [str(t.params.get("task_description")) for t in await c.tasks.by_state("queued")]


async def _module_denials(c: Coordinator) -> int:
    observed = await c.bus.tail(topic="observation")
    return sum(
        m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == SPECIALIST_MODULE_IN_FLIGHT
        for m in observed
    )


async def _freeform(c: Coordinator, mandate: str, **params: Any) -> None:
    await _delegate(c, scope="freeform", task_description=mandate, **params)


async def test_a_module_group_holds_at_most_one_queued_or_running_specialist(coord, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")
    await _freeform(coord, "fuse the MoE sort", module_group="moe")
    (holder,) = await coord.tasks.by_state("queued")
    await coord.tasks.transition(holder.task_id, "running", evidence={})

    await _freeform(coord, "retune the MoE GEMMs", module_group="MoE")
    await _freeform(coord, "shrink the allreduce", module_group="allreduce/comm")
    await _freeform(coord, "trace the decode host gap")

    assert await _queued_mandates(coord) == ["shrink the allreduce", "trace the decode host gap"]
    assert await _module_denials(coord) == 1


@pytest.mark.parametrize(("switch", "group"), [("0", "moe"), ("1", "draft")])
async def test_a_dispatch_without_a_known_group_or_with_the_bench_off_is_never_refused_for_one(
    coord, monkeypatch, switch, group
):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", switch)
    await _freeform(coord, "first", module_group=group)
    await _freeform(coord, "second", module_group=group)

    assert await _queued_mandates(coord) == ["first", "second"]
    assert await _module_denials(coord) == 0


async def test_each_task_of_a_freeform_wave_holds_the_module_group_it_names(coord, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")
    tasks = [
        {"task_description": "fuse the MoE sort", "module_group": "moe"},
        {"task_description": "retune the MoE GEMMs", "module_group": "moe"},
        {"task_description": "shrink the allreduce", "module_group": "allreduce/comm"},
    ]
    await _delegate(coord, scope="freeform", tasks=tasks)

    assert await _queued_mandates(coord) == ["fuse the MoE sort", "shrink the allreduce"]
    assert await _module_denials(coord) == 1


_MOE_KERNEL = "mfma_moe1_silu_mul_afp4_wfp4"
_GEMM_KERNEL = "ck_gemm_a8w8_blockscale_b_preshuffle"


def _decode_capture(c: Coordinator, tmp_path: Any, *, moe_us: float, gemm_us: float) -> None:
    from kernelforge.tests.trace_fixtures import kernel, step, write_trace

    events = [
        step("step[VERIFY bs=2]", 0.0, 100.0),
        kernel(_MOE_KERNEL, 1.0, moe_us),
        kernel(_GEMM_KERNEL, 2.0 + moe_us, gemm_us),
    ]
    c.shared_state.last_profile_trace = str(write_trace(tmp_path / "r-TP-0-EP-0.trace.json.gz", events))


async def test_a_dispatch_naming_no_group_owns_the_group_of_the_kernels_its_gap_names(coord, monkeypatch, tmp_path):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")
    _decode_capture(coord, tmp_path, moe_us=60.0, gemm_us=40.0)
    for cid, symptom in (
        ("gap.gemm", f"{_GEMM_KERNEL} runs at 40% of its roofline"),
        ("gap.moe", "mfma_moe1_silu_mul pads every expert to the full block"),
        ("gap.idle", "GPU 85% idle between decode steps"),
    ):
        coord.shared_state.upsert_gap({"canonical_id": cid, "symptom": symptom})
    await _freeform(coord, "fix the GEMM gap", gap_canonical_id="gap.gemm")
    await _freeform(coord, "fix the MoE gap", gap_canonical_id="gap.moe")
    await _freeform(coord, "fix the idle gap", gap_canonical_id="gap.idle")
    await _freeform(coord, "author the PR", gap_canonical_id="gap.moe", task_kind="framework_authoring")
    await _freeform(coord, "fix the MoE gap again", gap_canonical_id="gap.moe")
    await _freeform(coord, "retune the MoE GEMMs", module_group="moe")

    queued = await coord.tasks.by_state("queued")
    assert [(t.params["task_description"], t.params.get("module_group")) for t in queued] == [
        ("fix the GEMM gap", "gemm"),
        ("fix the MoE gap", "moe"),
        ("fix the idle gap", None),
        ("author the PR", None),
    ]
    assert queued[1].params["roofline_evidence"]["owned_module"] == {
        "group": "moe",
        "rows": [
            "| module | VERIFY bs=2 | top kernel |",
            "|---|---:|---|",
            f"| moe | 0.060 ms (60.0%) | `{_MOE_KERNEL}` |",
        ],
    }
    assert await _module_denials(coord) == 2


async def test_snake_case_words_naming_no_capture_kernel_own_no_group(coord, monkeypatch, tmp_path):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")
    _decode_capture(coord, tmp_path, moe_us=60.0, gemm_us=40.0)
    coord.shared_state.upsert_gap({"canonical_id": "gap.sched", "symptom": "chunked_prefill stalls at max_num_seqs=64"})
    await _freeform(coord, "fix the scheduler gap", gap_canonical_id="gap.sched")

    (queued,) = await coord.tasks.by_state("queued")
    assert queued.params.get("module_group") is None


@pytest.mark.parametrize(("moe_us", "gemm_us", "group"), [(60.0, 40.0, "moe"), (40.0, 60.0, "gemm")])
async def test_a_dispatch_naming_kernels_of_several_groups_owns_the_one_with_the_most_decode_time(
    coord, monkeypatch, tmp_path, moe_us, gemm_us, group
):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")
    _decode_capture(coord, tmp_path, moe_us=moe_us, gemm_us=gemm_us)
    await _freeform(coord, f"fuse {_GEMM_KERNEL} into {_MOE_KERNEL}")

    (queued,) = await coord.tasks.by_state("queued")
    assert queued.params["module_group"] == group


@pytest.mark.parametrize(("switch", "shown"), [("1", True), ("0", False)])
async def test_orchestration_sees_the_module_groups_and_which_ones_live_specialists_hold(
    coord, monkeypatch, switch, shown
):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", switch)
    await _freeform(coord, "fuse the MoE sort", module_group="moe")
    (holder,) = await coord.tasks.by_state("queued")

    in_flight = coord.conversation._context_running_tasks_reader()

    expected = (
        "specialist module groups (set params.module_group to own one; one queued or running specialist each): "
        "allreduce/comm, moe, indexer/topk, attention/mla, norm, gemm, quant, sampling/spec, elementwise/copy; "
        f"held: moe task_id={holder.task_id}"
    )
    lines = [line for line in in_flight.splitlines() if line.startswith("specialist module groups")]
    assert lines == ([expected] if shown else [])


@pytest.mark.parametrize(
    ("params", "group"),
    [
        ({"module_group": " MoE "}, "moe"),
        ({"module_group": "indexer/topk", "domain": _SYSTEM}, "indexer/topk"),
        ({"module_group": "draft"}, ""),
        ({}, ""),
        ({"module_group": "moe", "domain": "custom_specialist"}, ""),
    ],
)
def test_a_dispatch_holds_the_module_group_it_names(monkeypatch, params, group):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")

    assert held_module_group(params) == group
