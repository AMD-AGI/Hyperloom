# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch-dial matrix + freeform sanity-gate tests for the unified specialist."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.specialists.domains import KNOWLEDGE_DOMAIN_TAG_SET, SPECIALIST_DOMAINS
from hyperloom.orchestrator.specialists.profile import (
    DEFAULT_BENCH,
    DEFAULT_MODE,
    DEFAULT_SCOPE,
    MODE_PATCH,
    MODE_RESEARCH,
    SCOPE_DOMAIN,
    SCOPE_FREEFORM,
    SpecialistProfile,
    requires_gpu,
    resolve_specialist_profile,
    specialist_lanes,
    uses_whole_machine_gpu_lane,
    wall_budget_base_min,
)


# --------------------------------------------------------------------------- # resolve_specialist_profile — the dial
# matrix --------------------------------------------------------------------------- #
def test_bare_dispatch_defaults_to_freeform_research():
    """A bare dispatch (no scope, no domain/tag anchor) resolves to the cheap, read-only freeform/research mode."""
    prof = resolve_specialist_profile(None)
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH
    assert prof.bench is False
    assert prof.reserves_benchmark_lane is False


def test_anchored_dispatch_keeps_patch_default():
    """A dispatch that carries a domain anchor but no explicit dials keeps the historical single-domain, patch-authoring behaviour."""
    prof = resolve_specialist_profile({"domain": "serving_specialist"})
    assert prof == SpecialistProfile(
        scope=DEFAULT_SCOPE,
        mode=DEFAULT_MODE,
        bench=DEFAULT_BENCH,
    )
    assert prof.scope == SCOPE_DOMAIN
    assert prof.mode == MODE_PATCH
    assert prof.bench is False


def test_unknown_values_fall_back_without_raising():
    prof = resolve_specialist_profile(
        {"scope": "galaxy", "mode": "telepathy", "domain": "serving_specialist"},
    )
    assert prof.scope == DEFAULT_SCOPE
    assert prof.mode == DEFAULT_MODE


def test_unknown_scope_without_anchor_falls_back_to_freeform():
    prof = resolve_specialist_profile(
        {"scope": "galaxy", "mode": "telepathy"},
    )
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH


def test_freeform_defaults_to_research():
    """Freeform recon is read-only research mode unless told otherwise."""
    prof = resolve_specialist_profile({"scope": "freeform"})
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH
    assert prof.is_freeform is True
    assert prof.reserves_benchmark_lane is False


@pytest.mark.parametrize("truthy", [True, "true", "1", "yes", "on", 1])
def test_bench_requires_patch_mode_truthy(truthy):
    prof = resolve_specialist_profile({"mode": "patch", "bench": truthy})
    assert prof.bench is True
    assert prof.reserves_benchmark_lane is True


@pytest.mark.parametrize("falsy", [False, "false", "0", "no", "off", 0, None])
def test_bench_falsy_values(falsy):
    prof = resolve_specialist_profile({"mode": "patch", "bench": falsy})
    assert prof.bench is False
    assert prof.reserves_benchmark_lane is False


def test_reserves_benchmark_lane_only_for_bench_capable():
    """Only bench-capable patch specialists reserve the benchmark lane."""
    assert resolve_specialist_profile({"mode": "patch", "bench": True}).reserves_benchmark_lane is True
    authoring = {"framework_agent_authoring": True, "domain": "serving_specialist"}
    assert resolve_specialist_profile(authoring).reserves_benchmark_lane is False
    assert resolve_specialist_profile({"mode": "research"}).reserves_benchmark_lane is False
    assert resolve_specialist_profile(None).reserves_benchmark_lane is False


def test_bench_is_meaningless_for_research_mode():
    """Even an explicit ``bench=true`` is dropped when the worker can't patch."""
    prof = resolve_specialist_profile({"mode": "research", "bench": True})
    assert prof.bench is False
    assert prof.reserves_benchmark_lane is False


@pytest.fixture
def whole_machine(monkeypatch):
    """Pin the single-node / visible-GPU probe that gates enablement GPU leases."""

    def _set(available: bool) -> None:
        monkeypatch.setattr("hyperloom.orchestrator.specialists.profile._whole_machine_available", lambda: available)

    return _set


def test_requires_gpu_variants(whole_machine):
    """GPU for needs_gpu / bench always; for enablement only when the whole machine is available."""
    whole_machine(True)
    assert requires_gpu({"needs_gpu": True}) is True
    assert requires_gpu({"mode": "patch", "bench": True}) is True
    assert requires_gpu({"enablement": True}) is True
    assert requires_gpu({"framework_agent_authoring": True, "mode": "patch"}) is False
    assert requires_gpu({"mode": "research"}) is False
    assert requires_gpu(None) is False
    whole_machine(False)
    assert requires_gpu({"enablement": True}) is False
    assert requires_gpu({"needs_gpu": True}) is True


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"mode": "patch"}, ["research_lane"]),
        ({"framework_agent_authoring": True, "mode": "patch"}, ["research_lane"]),
        ({"needs_gpu": True}, ["gpu_research_lane"]),
        ({"enablement": True}, ["gpu_research_lane"]),
        ({"mode": "patch", "bench": True}, ["gpu_research_lane", "benchmark_lane"]),
    ],
)
def test_specialist_lanes(whole_machine, params, expected):
    """CPU specialists hold research_lane; GPU specialists hold gpu_research_lane instead; bench adds benchmark_lane."""
    whole_machine(True)
    assert specialist_lanes(params, ["research_lane"]) == expected


def test_whole_machine_lane_is_enablement_or_bench():
    """FRAMEWORK authoring is CPU; enablement and bench specialists lease the whole machine."""
    assert uses_whole_machine_gpu_lane({"enablement": True}) is True
    assert uses_whole_machine_gpu_lane({"mode": "patch", "bench": True}) is True
    assert uses_whole_machine_gpu_lane({"framework_agent_authoring": True}) is False


def test_wall_budget_base_min_by_mode():
    """Budget base is 60 min for patch, 10 min for research."""
    assert wall_budget_base_min({"mode": "patch"}) == 60.0
    assert wall_budget_base_min({"mode": "research"}) == 10.0
    assert wall_budget_base_min({"scope": "freeform"}) == 10.0
    assert wall_budget_base_min(None) == 10.0
    assert wall_budget_base_min({"domain": "serving_specialist"}) == 60.0


# --------------------------------------------------------------------------- # Freeform sanity gate (PolicyGate)
# --------------------------------------------------------------------------- #
@pytest.fixture
def gate():
    return PolicyGate()


def _dispatch(params: dict) -> dict:
    return {"action_name": "specialist", "params": params}


def test_freeform_single_task_ok(gate):
    gate._validate_specialist_dispatch(
        _dispatch({"scope": "freeform", "task_description": "Profile the decode path and report top stalls."}),
    )


def test_freeform_skips_tag_and_gap_requirements(gate):
    """Freeform carries no domain/tag/gap anchor — the tag/gap checks that a single-domain dispatch would trip must NOT fire here."""
    gate._validate_specialist_dispatch(
        _dispatch({"scope": "freeform", "task_description": "A short mandate."}),
    )


@pytest.mark.parametrize(
    "params",
    [{"scope": "freeform", "task_description": "   "}, {"scope": "freeform"}],
    ids=["blank", "absent"],
)
def test_freeform_empty_description_rejected(gate, params):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(_dispatch(params))
    assert exc.value.rule == "specialist_freeform_empty_description"


def test_a_freeform_dispatch_is_one_task_and_a_tasks_list_stands_in_for_no_description(gate):
    """``params.tasks`` is not a fan-out: a delegate is one task, so a list of them leaves the description missing."""
    tasks = [{"task_description": "Investigate prefill batching."}, {"task_description": "Audit KV-cache allocation."}]
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(_dispatch({"scope": "freeform", "tasks": tasks}))
    assert exc.value.rule == "specialist_freeform_empty_description"


def test_a_freeform_description_of_any_length_is_admitted(gate):
    """The description is the whole mandate; the gate asks only that there is one."""
    gate._validate_specialist_dispatch(_dispatch({"scope": "freeform", "task_description": "x" * 50_000}))


@pytest.mark.parametrize("framework", ["tensorrt", "", None])
def test_unregistered_framework_param_rejected(gate, framework):
    """A specialist resolves its source tree from params.framework, so the name must be one the registry knows."""
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            _dispatch({"scope": "freeform", "task_description": "A short mandate.", "framework": framework}),
        )
    assert exc.value.rule == "specialist_framework_unregistered"


def test_registered_framework_param_admitted(gate):
    gate._validate_specialist_dispatch(
        _dispatch({"scope": "freeform", "task_description": "A short mandate.", "framework": "vllm"}),
    )


@pytest.mark.parametrize(
    "desc",
    [
        "clean up with rm -rf / now",
        "run mkfs.ext4 on the scratch disk",
        "please shutdown the host afterwards",
    ],
)
def test_freeform_destructive_text_allowed(gate, desc):
    gate._validate_specialist_dispatch(
        _dispatch({"scope": "freeform", "task_description": desc}),
    )


# --------------------------------------------------------------------------- # Non-freeform scope gating still
# applies --------------------------------------------------------------------------- #
_REAL_TAGS = sorted(KNOWLEDGE_DOMAIN_TAG_SET)


@pytest.mark.skipif(len(_REAL_TAGS) < 1, reason="no knowledge-domain tags")
def test_domains_scope_with_one_tag_allowed(gate):
    gate._validate_specialist_dispatch(
        _dispatch(
            {
                "scope": "domains",
                "tags": [_REAL_TAGS[0]],
                "gap_canonical_id": "gap.x.session-1",
            }
        ),
    )


@pytest.mark.skipif(len(_REAL_TAGS) < 2, reason="need >=2 knowledge-domain tags")
def test_single_domain_scope_with_multiple_tags_allowed(gate):
    gate._validate_specialist_dispatch(
        _dispatch(
            {
                "scope": "domain",
                "tags": _REAL_TAGS[:2],
                "gap_canonical_id": "gap.x.session-1",
            }
        ),
    )


_MAX_TURNS_DISPATCHES = [
    pytest.param({"scope": "freeform", "task_description": "probe"}, id="freeform"),
    pytest.param({"domain": "serving_specialist", "gap_canonical_id": "gap.x"}, id="anchored"),
]


@pytest.mark.parametrize("params", _MAX_TURNS_DISPATCHES)
@pytest.mark.parametrize("max_turns", [None, 0, 1, "7", 1001, 50_000])
def test_a_max_turns_that_is_a_count_is_admitted_whatever_its_size(gate, params, max_turns):
    """The wall-clock budget bounds a specialist, so the dial has no ceiling."""
    gate._validate_specialist_dispatch(_dispatch({**params, "max_turns": max_turns}))


@pytest.mark.parametrize("params", _MAX_TURNS_DISPATCHES)
@pytest.mark.parametrize("max_turns", [-1, "many", "", [3]])
def test_a_max_turns_that_is_not_a_count_is_denied(gate, params, max_turns):
    """Refused at the intent, so the value never reaches the executor to fail a specialist that ran nothing."""
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(_dispatch({**params, "max_turns": max_turns}))

    assert exc.value.rule == "specialist_dispatch_source"
    assert "params.max_turns" in str(exc.value)


def test_resolve_specialist_max_turns_zero_uses_default():
    from hyperloom.orchestrator.specialists.runner import resolve_specialist_max_turns

    assert resolve_specialist_max_turns(0, default=1000) == 1000
    assert resolve_specialist_max_turns(None, default=42) == 42
    assert resolve_specialist_max_turns("", default=42) == 42
    assert resolve_specialist_max_turns(5, default=1000) == 5
    assert resolve_specialist_max_turns(50_000, default=1000) == 50_000


# --------------------------------------------------------------------------- # Tool surface: TaskKill and
# SlashCommand are denied. --------------------------------------------------------------------------- #
def test_specialists_are_denied_only_the_process_cleanup_tools():
    """A specialist gets every tool except these two, so patch authoring and KB writes stay open."""
    from hyperloom.orchestrator.specialists.runner import SPECIALIST_TOOL_DENYLIST

    assert SPECIALIST_TOOL_DENYLIST == frozenset({"KillShell", "SlashCommand"})


# --------------------------------------------------------------------------- # domain KEY -> kb_anchor translation
# --------------------------------------------------------------------------- #
def test_normalize_dispatch_tags_translates_key_to_anchor():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # A domain KEY in params.tags is translated to its kb_anchor.
    assert normalize_dispatch_tags({"tags": ["serving_specialist"]}) == ["framework"]
    assert normalize_dispatch_tags({"tags": ["kernel_switch_specialist"]}) == ["kernel_agent"]


def test_normalize_dispatch_tags_keeps_valid_anchor_and_dedups():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # An already-valid anchor is preserved unchanged.
    assert normalize_dispatch_tags({"tags": ["framework"]}) == ["framework"]
    # serving_specialist -> framework collapses with an explicit framework tag.
    assert normalize_dispatch_tags({"tags": ["serving_specialist", "framework"]}) == ["framework"]


def test_normalize_dispatch_tags_passes_garbage_through():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # Genuinely unknown tags are NOT invented into an anchor — they pass through verbatim so the runner can synthesize
    # an empty result.
    assert normalize_dispatch_tags({"tags": ["totally_bogus"]}) == ["totally_bogus"]


def test_normalize_dispatch_tags_domain_alias_translated():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # The legacy single-tag params.domain alias is translated the same way.
    assert normalize_dispatch_tags({"domain": "system_specialist"}) == ["systems"]


def test_dispatch_with_domain_key_tag_no_longer_rejected(gate):
    """tags=['serving_specialist'] (a KEY) must validate cleanly (key->framework)."""
    gate._validate_specialist_dispatch(
        _dispatch(
            {
                "scope": "domain",
                "tags": ["serving_specialist"],
                "gap_canonical_id": "gap.framework.scheduler.session-1",
            }
        ),
    )


def test_dispatch_with_garbage_tag_allowed(gate):
    """An out-of-vocabulary tag is observed, not denied; the runner synthesizes an empty result."""
    gate._validate_specialist_dispatch(
        _dispatch(
            {
                "scope": "domain",
                "tags": ["totally_bogus"],
                "gap_canonical_id": "gap.x.session-1",
            }
        ),
    )


def test_specialist_emit_hint_lists_every_llm_selectable_domain():
    """The hint must name every domain Orchestration is allowed to pick."""
    from types import SimpleNamespace

    from hyperloom.orchestrator.prompts.prompt_builder import (
        _format_emit_hint,
    )

    # _format_emit_hint only reads meta.name.
    hint = _format_emit_hint(SimpleNamespace(name="specialist"))
    selectable = [d.key for d in SPECIALIST_DOMAINS if d.llm_selectable]
    assert selectable
    for key in selectable:
        assert key in hint, key
    for key in (d.key for d in SPECIALIST_DOMAINS if not d.llm_selectable):
        assert key not in hint, key
    assert "max_turns?=<int turn cap; omit or 0 for the default>" in hint


def test_gate_denies_orchestration_naming_a_coordinator_dispatched_domain(gate):
    """A domain only the Coordinator dispatches is refused to Orchestration, with the authoring domain to use instead."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    intent = Intent(
        type=IntentType.DELEGATE,
        payload=_dispatch({"domain": "enablement_specialist", "gap_canonical_id": "gap.x", "framework": "sglang"}),
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "specialist_domain_coordinator_dispatched"
    assert "serving_specialist" in exc.value.hint


# --------------------------------------------------------------------------- # domain.default_mode
# --------------------------------------------------------------------------- #
def test_research_mode_param_forces_research_mode():
    """params['mode']='research' must override the global default."""
    profile = resolve_specialist_profile({"mode": "research", "domain": "serving_specialist"})
    assert profile.mode == MODE_RESEARCH
    assert profile.bench is False


def test_research_default_mode_domain_resolves_to_research():
    """Domains with default_mode='research' yield mode=research when no explicit mode is given."""
    from hyperloom.orchestrator.specialists.domains import get_domain

    research_keys = [d.key for d in SPECIALIST_DOMAINS if d.default_mode == "research"]
    assert research_keys, "the registry declares no research-mode domain"
    for key in research_keys:
        domain = get_domain(key)
        assert domain is not None
        profile = resolve_specialist_profile({"domain": key}, domain=domain)
        assert profile.mode == MODE_RESEARCH, f"{key} should resolve to research mode"


def test_patch_capable_domains_default_to_patch():
    """Patch-capable domains have default_mode='patch'."""
    from hyperloom.orchestrator.specialists.domains import get_domain

    for key in ("serving_specialist", "kernel_switch_specialist", "comm_specialist"):
        domain = get_domain(key)
        assert domain is not None
        assert domain.default_mode == "patch"
        profile = resolve_specialist_profile({"domain": key}, domain=domain)
        assert profile.mode == MODE_PATCH
