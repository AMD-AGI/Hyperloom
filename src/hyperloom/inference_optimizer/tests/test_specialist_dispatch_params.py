# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch-dial matrix + freeform sanity-gate tests for the unified specialist.

Covers the four orthogonal dials (``scope`` / ``mode`` / ``bench`` / ``lane``)
resolved by ``resolve_specialist_profile`` and the mechanical PolicyGate that
guards ``scope='freeform'`` dispatches.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.policy.gate import (
    KNOWLEDGE_DOMAIN_TAG_SET,
    PolicyDenied,
    PolicyGate,
    SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS,
    SPECIALIST_FREEFORM_WAVE_MAX,
)
from hyperloom.orchestrator.specialists.domains import SPECIALIST_DOMAINS
from hyperloom.orchestrator.specialists.profile import (
    DEFAULT_BENCH,
    DEFAULT_LANE,
    DEFAULT_MODE,
    DEFAULT_SCOPE,
    LANE_CPU,
    LANE_GPU,
    MODE_PATCH,
    MODE_RESEARCH,
    SCOPE_DOMAIN,
    SCOPE_FREEFORM,
    SpecialistProfile,
    holds_serving_slot,
    resolve_specialist_profile,
    uses_whole_machine_gpu_lane,
)


# --------------------------------------------------------------------------- #
# resolve_specialist_profile — the dial matrix
# --------------------------------------------------------------------------- #
def test_bare_dispatch_defaults_to_freeform_research_cpu():
    """A truly bare dispatch (no scope, no domain/tag anchor) resolves to the
    cheap, read-only freeform/research/CPU lane — safe & cheap first."""
    prof = resolve_specialist_profile(None)
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH
    assert prof.bench is False
    assert prof.lane == LANE_CPU
    assert prof.reserves_benchmark_lane is False


def test_anchored_dispatch_keeps_legacy_patch_gpu_default():
    """A dispatch that carries a domain anchor but no explicit dials keeps the
    historical single-domain, patch-authoring, GPU-leased behaviour."""
    prof = resolve_specialist_profile({"domain": "serving_specialist"})
    assert prof == SpecialistProfile(
        scope=DEFAULT_SCOPE,
        mode=DEFAULT_MODE,
        bench=DEFAULT_BENCH,
        lane=DEFAULT_LANE,
    )
    assert prof.scope == SCOPE_DOMAIN
    assert prof.mode == MODE_PATCH
    assert prof.bench is False
    assert prof.lane == LANE_GPU


def test_unknown_values_fall_back_without_raising():
    prof = resolve_specialist_profile(
        {"scope": "galaxy", "mode": "telepathy", "lane": "quantum", "domain": "serving_specialist"},
    )
    assert prof.scope == DEFAULT_SCOPE
    assert prof.mode == DEFAULT_MODE
    assert prof.lane == LANE_GPU


def test_unknown_scope_without_anchor_falls_back_to_freeform():
    prof = resolve_specialist_profile(
        {"scope": "galaxy", "mode": "telepathy", "lane": "quantum"},
    )
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH
    assert prof.lane == LANE_CPU


def test_freeform_defaults_to_research_on_cpu():
    """Freeform recon is read-only research on the CPU lane unless told otherwise."""
    prof = resolve_specialist_profile({"scope": "freeform"})
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH
    assert prof.lane == LANE_CPU
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


def test_holds_serving_slot_only_for_bench_capable():
    """phase-3 §4 / invariant §6.3: only bench-capable patch specialists hold
    the whole-machine serving_slot; authoring-only (incl. framework authoring)
    holds num_gpus only so it can share the GPU queue."""
    # Bench-capable patch specialist -> holds the slot.
    assert holds_serving_slot({"mode": "patch", "bench": True}) is True
    # Framework authoring is NOT bench-capable by default -> no slot, but it
    # still draws from the whole-machine pool (uses_whole_machine_gpu_lane).
    fw = {"framework_agent_authoring": True, "domain": "serving_specialist"}
    assert holds_serving_slot(fw) is False
    assert uses_whole_machine_gpu_lane(fw) is True
    # A bench-capable framework specialist DOES hold the slot for that window.
    assert holds_serving_slot({"framework_agent_authoring": True, "mode": "patch", "bench": True}) is True
    # Plain research / non-bench GPU probe -> no slot.
    assert holds_serving_slot({"mode": "research"}) is False
    assert holds_serving_slot(None) is False


def test_bench_is_meaningless_for_research_mode():
    """Even an explicit ``bench=true`` is dropped when the worker can't patch."""
    prof = resolve_specialist_profile({"mode": "research", "bench": True})
    assert prof.bench is False
    assert prof.reserves_benchmark_lane is False


def test_explicit_lane_overrides_default():
    assert resolve_specialist_profile({"mode": "research", "lane": "gpu"}).lane == LANE_GPU
    assert resolve_specialist_profile({"mode": "patch", "lane": "cpu"}).lane == LANE_CPU


def test_research_mode_defaults_to_cpu_lane():
    assert resolve_specialist_profile({"scope": "domain", "mode": "research"}).lane == LANE_CPU


# --------------------------------------------------------------------------- #
# Freeform sanity gate (PolicyGate)
# --------------------------------------------------------------------------- #
@pytest.fixture
def orchestration_role():
    return default_role_registry()["orchestration"]


@pytest.fixture
def gate():
    return PolicyGate(role_registry=default_role_registry())


def _dispatch(params: dict) -> dict:
    return {"action_name": "specialist", "params": params}


def test_freeform_single_task_ok(gate, orchestration_role):
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch({"scope": "freeform", "task_description": "Profile the decode path and report top stalls."}),
    )


def test_freeform_wave_ok(gate, orchestration_role):
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "freeform",
                "tasks": [
                    {"task_description": "Investigate prefill batching."},
                    {"task_description": "Audit KV-cache allocation."},
                ],
            }
        ),
    )


def test_freeform_skips_tag_and_gap_requirements(gate, orchestration_role):
    """Freeform carries no domain/tag/gap anchor — the tag/gap checks that a
    single-domain dispatch would trip must NOT fire here."""
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch({"scope": "freeform", "task_description": "A short mandate."}),
    )


def test_freeform_empty_description_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "task_description": "   "}),
        )
    assert exc.value.rule == "specialist_freeform_empty_description"


def test_freeform_missing_description_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform"}),
        )
    assert exc.value.rule == "specialist_freeform_empty_description"


def test_freeform_description_too_long_rejected(gate, orchestration_role):
    huge = "x" * (SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS + 1)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "task_description": huge}),
        )
    assert exc.value.rule == "specialist_freeform_description_too_long"


@pytest.mark.parametrize(
    "desc",
    [
        "clean up with rm -rf / now",
        "run mkfs.ext4 on the scratch disk",
        "please shutdown the host afterwards",
    ],
)
def test_freeform_destructive_text_allowed(gate, orchestration_role, desc):
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch({"scope": "freeform", "task_description": desc}),
    )


def test_freeform_empty_wave_falls_through_to_single_task(gate, orchestration_role):
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch({"scope": "freeform", "tasks": [], "task_description": "probe"}),
    )


def test_freeform_wave_too_large_rejected(gate, orchestration_role):
    tasks = [{"task_description": f"task {i}"} for i in range(SPECIALIST_FREEFORM_WAVE_MAX + 1)]
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "tasks": tasks}),
        )
    assert exc.value.rule == "specialist_freeform_wave_too_large"


def test_freeform_wave_non_dict_task_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "tasks": ["not a dict"]}),
        )
    assert exc.value.rule == "specialist_freeform_wave_invalid_task"
    assert "tasks[0]" in str(exc.value)


def test_freeform_wave_empty_task_description_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "tasks": [{"task_description": ""}]}),
        )
    assert exc.value.rule == "specialist_freeform_empty_description"
    assert "tasks[0]" in str(exc.value)


def test_freeform_wave_all_invalid_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "tasks": ["bad", {"task_description": ""}]}),
        )
    assert exc.value.rule == "specialist_freeform_wave_invalid_task"


# --------------------------------------------------------------------------- #
# Non-freeform scope gating still applies
# --------------------------------------------------------------------------- #
_REAL_TAGS = sorted(KNOWLEDGE_DOMAIN_TAG_SET)


@pytest.mark.skipif(len(_REAL_TAGS) < 1, reason="no knowledge-domain tags")
def test_domains_scope_with_one_tag_allowed(gate, orchestration_role):
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "domains",
                "tags": [_REAL_TAGS[0]],
                "gap_canonical_id": "gap.x.session-1",
            }
        ),
    )


# --------------------------------------------------------------------------- #
# Gap auto-fill from the gaps[] ledger (friction symmetry, point 4)
# --------------------------------------------------------------------------- #
def _gate_with_gaps(gaps: list[dict]) -> PolicyGate:
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    for g in gaps:
        state.upsert_gap(g)
    return PolicyGate(role_registry=default_role_registry(), shared_state=state)


def test_domain_dispatch_backfills_gap_from_ledger(orchestration_role):
    gate = _gate_with_gaps(
        [
            {
                "canonical_id": "gap.framework.scheduler.session-1",
                "domain_hint": "serving_specialist",
                "severity": "high",
            },
        ]
    )
    params = {"domain": "serving_specialist"}  # no gap_canonical_id
    gate._validate_specialist_dispatch(orchestration_role, _dispatch(params))
    # Mutated in place so the downstream dispatch carries the canonical id.
    assert params["gap_canonical_id"] == "gap.framework.scheduler.session-1"


def test_gap_backfill_prefers_high_severity_then_least_attempted(orchestration_role):
    gate = _gate_with_gaps(
        [
            {
                "canonical_id": "gap.low",
                "domain_hint": "serving_specialist",
                "severity": "medium",
                "attempts": [{"x": 1}],
            },
            {"canonical_id": "gap.win", "domain_hint": "framework", "severity": "high"},
        ]
    )
    params = {"domain": "serving_specialist"}
    gate._validate_specialist_dispatch(orchestration_role, _dispatch(params))
    # framework is serving_specialist's kb_anchor, so both match; high wins.
    assert params["gap_canonical_id"] == "gap.win"


def test_gap_backfill_noop_when_no_anchor_match_still_rejects(orchestration_role):
    gate = _gate_with_gaps(
        [
            {"canonical_id": "gap.kernel", "domain_hint": "kernel_switch_specialist", "severity": "high"},
        ]
    )
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"domain": "comm_specialist"}),
        )
    assert exc.value.rule == "specialist_dispatch_source"
    assert "gap" in str(exc.value)


@pytest.mark.skipif(len(_REAL_TAGS) < 2, reason="need >=2 knowledge-domain tags")
def test_single_domain_scope_with_multiple_tags_allowed(gate, orchestration_role):
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "domain",
                "tags": _REAL_TAGS[:2],
                "gap_canonical_id": "gap.x.session-1",
            }
        ),
    )


# --------------------------------------------------------------------------- #
# GPU request is governed by the same ceiling for *every* scope — freeform is
# not a hole around the GPU-pool accounting.
# --------------------------------------------------------------------------- #
def _gate_with_gpu_capacity(capacity: int, *, tp: int = 0) -> PolicyGate:
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.gpu_specialist_capacity = capacity
    state.tp = tp
    return PolicyGate(role_registry=default_role_registry(), shared_state=state)


def test_freeform_gpu_request_clears_ceiling(orchestration_role, monkeypatch):
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP"):
        monkeypatch.delenv(name, raising=False)
    gate = _gate_with_gpu_capacity(2)
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "freeform",
                "task_description": "micro-bench the decode attention kernel",
                "needs_gpu": True,
                "gpu_count": 2,
            }
        ),
    )


def test_freeform_gpu_request_rejected_when_pool_disabled(orchestration_role):
    gate = _gate_with_gpu_capacity(0)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "task_description": "needs a GPU but pool is off",
                    "needs_gpu": True,
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_pool_disabled"


def test_freeform_gpu_request_exceeds_capacity_rejected(orchestration_role):
    gate = _gate_with_gpu_capacity(1)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "task_description": "asks for more GPUs than the pool has",
                    "needs_gpu": True,
                    "gpu_count": 4,
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_request_exceeds_capacity"


def test_freeform_gpu_request_nonpositive_count_rejected(orchestration_role):
    gate = _gate_with_gpu_capacity(2)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "task_description": "bad gpu count",
                    "needs_gpu": True,
                    "gpu_count": 0,
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_request_invalid"


def test_domain_gpu_request_still_governed_after_refactor(orchestration_role):
    """Regression: the GPU check extracted into _validate_specialist_gpu_request
    must still fire on the domain-anchored path."""
    gate = _gate_with_gpu_capacity(0)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "domain": "serving_specialist",
                    "gap_canonical_id": "gap.framework.x.session-1",
                    "needs_gpu": True,
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_pool_disabled"


def test_bench_specialist_without_explicit_needs_gpu_is_gated(orchestration_role, monkeypatch):
    """A bench-enabled (mode=patch & bench=true) specialist auto-defaults
    needs_gpu=True at dispatch; the gate must mirror that so it is rejected
    when the pool is disabled.

    Both pools have to be empty for the denial to apply: a bench specialist
    leases from the whole-machine pool, which the gate exempts from a zero
    specialist ceiling whenever that pool has cards. The visible-device env is
    therefore pinned empty -- otherwise the pool is resolved from the host's real
    GPUs and the test only passes on a GPU-less machine.
    """
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP", "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    gate = _gate_with_gpu_capacity(0)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "task_description": "patch + bench the decode attention kernel",
                    "mode": "patch",
                    "bench": True,
                    # no explicit needs_gpu — the bench profile implies it.
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_pool_disabled"


def test_bench_specialist_whole_machine_lane_allows_full_node(orchestration_role, monkeypatch):
    """A bench specialist takes the whole-machine, time-shared GPU lane, so
    serving occupying the whole node (TP == #GPUs) does not deny it — the
    serving-disjoint carve does not apply to the whole-machine pool."""
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP", "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    gate = _gate_with_gpu_capacity(4, tp=4)
    # gpu_count floored to serving TP=4; whole-machine pool has all 4 cards.
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "freeform",
                "task_description": "start a TP-sharded server and rebench a patch",
                "mode": "patch",
                "bench": True,
                "gpu_count": 1,
            }
        ),
    )


def test_bench_specialist_denied_when_whole_machine_too_small(orchestration_role, monkeypatch):
    """A bench specialist is still denied when the whole node physically has
    fewer cards than the serving TP it must shard a server across."""
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP", "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    gate = _gate_with_gpu_capacity(4, tp=4)
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "task_description": "start a TP-sharded server and rebench a patch",
                    "mode": "patch",
                    "bench": True,
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_request_exceeds_capacity"
    assert "effective gpu_count=4" in str(exc.value)
    assert "whole-machine GPU pool size=2" in str(exc.value)


def test_bench_specialist_omitted_gpu_count_allows_whole_machine(orchestration_role, monkeypatch):
    """Omitting gpu_count defaults a bench specialist to serving TP and is valid
    when the whole-machine pool has at least that many cards."""
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP", "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    gate = _gate_with_gpu_capacity(8, tp=4)
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "freeform",
                "task_description": "start a TP-sharded server and rebench a patch",
                "mode": "patch",
                "bench": True,
            }
        ),
    )


def test_research_specialist_without_needs_gpu_is_not_gated(orchestration_role):
    """A non-bench (research) specialist needs no GPU, so the pool-disabled
    gate must NOT fire for it even when capacity is 0."""
    gate = _gate_with_gpu_capacity(0)
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "freeform",
                "task_description": "read-only profile the decode path",
                "mode": "research",
            }
        ),
    )


def test_freeform_wave_mixed_valid_and_invalid_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "tasks": [
                        {"task_description": "valid task"},
                        {"task_description": ""},
                    ],
                }
            ),
        )
    assert exc.value.rule == "specialist_freeform_empty_description"
    assert "tasks[1]" in str(exc.value)


def test_freeform_negative_max_turns_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "task_description": "probe", "max_turns": -1}),
        )
    assert "max_turns" in str(exc.value)


def test_freeform_wave_task_negative_max_turns_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "tasks": [{"task_description": "probe", "max_turns": -1}],
                }
            ),
        )
    assert "tasks[0].max_turns" in str(exc.value)


def test_freeform_max_turns_above_cap_rejected(gate, orchestration_role):
    from hyperloom.orchestrator.specialists.domains import SPECIALIST_MAX_TURNS_HARD_CAP

    with pytest.raises(PolicyDenied):
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch(
                {
                    "scope": "freeform",
                    "task_description": "probe",
                    "max_turns": SPECIALIST_MAX_TURNS_HARD_CAP + 1,
                }
            ),
        )


def test_prepare_scoring_proposals_preserves_output_name_whitespace():
    from hyperloom.orchestrator.scoring.proposal_scorer import (
        _normalise_model_scores,
        _prepare_scoring_proposals,
    )

    entries = _prepare_scoring_proposals([{"name": " x "}])
    parsed = {"scores": {"proposal_0": {"score": 5, "reason": "ok"}}}
    out = _normalise_model_scores(parsed, scoring_entries=entries)
    assert " x " in out
    assert "x" not in out


def test_resolve_specialist_max_turns_zero_uses_default():
    from hyperloom.orchestrator.specialists.runner import resolve_specialist_max_turns

    assert resolve_specialist_max_turns(0, default=1000) == 1000
    assert resolve_specialist_max_turns(None, default=42) == 42
    assert resolve_specialist_max_turns("", default=42) == 42
    assert resolve_specialist_max_turns(5, default=1000) == 5


# --------------------------------------------------------------------------- #
# Tool surface: TaskKill and SlashCommand are denied.
# --------------------------------------------------------------------------- #
def test_denylist_blocks_process_kill_tools():
    from hyperloom.orchestrator.specialists.runner import SPECIALIST_TOOL_DENYLIST

    assert "KillShell" in SPECIALIST_TOOL_DENYLIST
    assert "SlashCommand" in SPECIALIST_TOOL_DENYLIST
    assert "Task" not in SPECIALIST_TOOL_DENYLIST
    assert "TodoWrite" not in SPECIALIST_TOOL_DENYLIST


# --------------------------------------------------------------------------- #
# domain KEY -> kb_anchor translation
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

    # Genuinely unknown tags are NOT invented into an anchor — they pass
    # through verbatim so the runner can synthesize an empty result.
    assert normalize_dispatch_tags({"tags": ["totally_bogus"]}) == ["totally_bogus"]


def test_normalize_dispatch_tags_domain_alias_translated():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # The legacy single-tag params.domain alias is translated the same way.
    assert normalize_dispatch_tags({"domain": "system_specialist"}) == ["systems"]


def test_dispatch_with_domain_key_tag_no_longer_rejected(gate, orchestration_role):
    """tags=['serving_specialist'] (a KEY) must validate cleanly (key->framework)."""
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "domain",
                "tags": ["serving_specialist"],
                "gap_canonical_id": "gap.framework.scheduler.session-1",
            }
        ),
    )


def test_dispatch_with_garbage_tag_allowed(gate, orchestration_role):
    """An out-of-vocabulary tag is observed, not denied; the runner synthesizes an empty result."""
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "domain",
                "tags": ["totally_bogus"],
                "gap_canonical_id": "gap.x.session-1",
            }
        ),
    )


def test_specialist_emit_hint_lists_every_llm_selectable_domain():
    """The hint must name every domain Orchestration is allowed to pick.

    Derived from the registry: a domain added without appearing in the hint is
    one the LLM can never choose, and one listed but not selectable is an
    invitation PolicyGate will refuse.
    """
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


# --------------------------------------------------------------------------- #
# gate gpu_count default aligned with dispatcher at serving_tp=0
# --------------------------------------------------------------------------- #
def test_bench_specialist_no_serving_tp_defaults_to_whole_machine(orchestration_role, monkeypatch):
    """When serving_tp=0, a bench specialist with no explicit gpu_count must
    default to the whole-machine pool size in the gate, matching the
    dispatcher's fallback to ``gpu_pool.capacity``."""
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP", "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    # tp=0: no serving, gpu_specialist_capacity=4 (4-card node).
    gate = _gate_with_gpu_capacity(4, tp=0)
    # gate's default gpu_count must be 4 (whole-machine), agreeing with the
    # dispatcher that leases all 4 cards.
    gate._validate_specialist_dispatch(
        orchestration_role,
        _dispatch(
            {
                "scope": "freeform",
                "task_description": "bench patch before any serving baseline exists",
                "mode": "patch",
                "bench": True,
                "needs_gpu": True,
                # gpu_count omitted → gate defaults to whole-machine size.
            }
        ),
    )


# --------------------------------------------------------------------------- #
# domain.default_mode → mode=research, lane=cpu
# --------------------------------------------------------------------------- #
def test_research_mode_param_forces_research_mode():
    """params['mode']='research' must override the global default."""
    profile = resolve_specialist_profile({"mode": "research", "domain": "serving_specialist"})
    assert profile.mode == MODE_RESEARCH
    assert profile.lane == LANE_CPU
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
        assert profile.lane == LANE_CPU, f"{key} should resolve to cpu lane"


def test_patch_capable_domains_default_to_patch():
    """Patch-capable domains have default_mode='patch'."""
    from hyperloom.orchestrator.specialists.domains import get_domain

    for key in ("serving_specialist", "kernel_switch_specialist", "comm_specialist"):
        domain = get_domain(key)
        assert domain is not None
        assert domain.default_mode == "patch"
        profile = resolve_specialist_profile({"domain": key}, domain=domain)
        assert profile.mode == MODE_PATCH


# --------------------------------------------------------------------------- #
# Deferring the paid proposers to the free one
# --------------------------------------------------------------------------- #
def _gate_with_predictor_state(**fields) -> PolicyGate:
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.phase = "FRAMEWORK_AGENT"
    state.framework = "vllm"
    state.macro_cycle = 0
    state.predictor_chain_cycle = 0
    for name, value in fields.items():
        setattr(state, name, value)
    return PolicyGate(role_registry=default_role_registry(), shared_state=state)


@pytest.fixture
def predictor_active(monkeypatch):
    """A configured, enqueueing predictor."""
    from hyperloom.orchestrator.predictor import config as predictor_config

    monkeypatch.setenv(predictor_config.ENV_ENDPOINT, "http://predictor:8973")
    monkeypatch.setenv(predictor_config.ENV_MODE, predictor_config.MODE_ACTIVE)
    monkeypatch.delenv(predictor_config.ENV_MAX_CHAIN, raising=False)


def test_specialist_deferred_while_the_predictor_leads(predictor_active, orchestration_role):
    """The predictor costs no API spend; a specialist was 97% of a session's.

    Denied rather than queued: the point is that the specialist subprocess never
    starts, which is where essentially all of the cost sits.
    """
    gate = _gate_with_predictor_state(predictor_chain_steps=0)
    with pytest.raises(PolicyDenied) as excinfo:
        gate._validate_specialist_dispatch(
            orchestration_role, _dispatch({"scope": "freeform", "task_description": "tune it"})
        )
    assert excinfo.value.rule == "specialist_deferred_to_predictor"


def test_specialist_admitted_once_the_predictor_stops_landing(
    monkeypatch, predictor_active, orchestration_role
):
    from hyperloom.orchestrator.predictor import config as predictor_config

    monkeypatch.setenv(predictor_config.ENV_MAX_CHAIN, "2")
    gate = _gate_with_predictor_state(predictor_chain_steps=2)
    gate._validate_specialist_dispatch(
        orchestration_role, _dispatch({"scope": "freeform", "task_description": "tune it"})
    )


def test_anchored_specialist_is_deferred_too(predictor_active, orchestration_role):
    """Every LLM specialist is paid, not just the freeform ones."""
    gate = _gate_with_predictor_state(predictor_chain_steps=0)
    with pytest.raises(PolicyDenied) as excinfo:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"domain": "serving_specialist", "gap_canonical_id": "gap.x"}),
        )
    assert excinfo.value.rule == "specialist_deferred_to_predictor"


def test_no_deferral_outside_the_framework_phase(predictor_active, orchestration_role):
    """PRELUDE's research scout and static recon must not be held back."""
    gate = _gate_with_predictor_state(phase="PRELUDE", predictor_chain_steps=0)
    gate._validate_specialist_dispatch(
        orchestration_role, _dispatch({"scope": "freeform", "task_description": "survey"})
    )


def test_no_deferral_without_a_predictor(monkeypatch, orchestration_role):
    """Suppressing every proposer would leave the phase nothing to benchmark."""
    from hyperloom.orchestrator.predictor import config as predictor_config

    monkeypatch.delenv(predictor_config.ENV_ENDPOINT, raising=False)
    gate = _gate_with_predictor_state(predictor_chain_steps=0)
    gate._validate_specialist_dispatch(
        orchestration_role, _dispatch({"scope": "freeform", "task_description": "tune it"})
    )


def test_no_deferral_for_a_framework_the_predictor_cannot_answer_for(
    predictor_active, orchestration_role
):
    gate = _gate_with_predictor_state(framework="atom", predictor_chain_steps=0)
    gate._validate_specialist_dispatch(
        orchestration_role, _dispatch({"scope": "freeform", "task_description": "tune it"})
    )
