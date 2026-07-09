# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch-dial matrix + freeform sanity-gate tests for the unified specialist.

Covers the four orthogonal dials (``scope`` / ``mode`` / ``bench`` / ``lane``)
resolved by ``resolve_specialist_profile`` and the lightweight mechanical
PolicyGate that guards ``scope='freeform'`` dispatches (the channel that
absorbed the retired ``dynamic_specialist`` wave worker).
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
    resolve_specialist_profile,
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
    # Unknown scope + a domain anchor -> inferred single-domain, patch/gpu.
    prof = resolve_specialist_profile(
        {"scope": "galaxy", "mode": "telepathy", "lane": "quantum", "domain": "serving_specialist"},
    )
    assert prof.scope == DEFAULT_SCOPE
    assert prof.mode == DEFAULT_MODE
    assert prof.lane == LANE_GPU  # mode resolved to patch -> gpu lane


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


def test_domains_scope_is_cross_domain():
    prof = resolve_specialist_profile({"scope": "domains"})
    assert prof.is_cross_domain is True
    assert prof.is_freeform is False


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
    "redline",
    [
        "clean up with rm -rf / now",
        "run mkfs.ext4 on the scratch disk",
        "please shutdown the host afterwards",
    ],
)
def test_freeform_redline_rejected(gate, orchestration_role, redline):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "task_description": redline}),
        )
    assert exc.value.rule == "specialist_freeform_redline"


def test_freeform_empty_wave_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "tasks": []}),
        )
    assert exc.value.rule == "specialist_freeform_wave_invalid"


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
    assert exc.value.rule == "specialist_freeform_task_invalid"


def test_freeform_wave_empty_task_description_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({"scope": "freeform", "tasks": [{"task_description": ""}]}),
        )
    assert exc.value.rule == "specialist_freeform_empty_description"


# --------------------------------------------------------------------------- #
# Non-freeform scope gating still applies
# --------------------------------------------------------------------------- #
_REAL_TAGS = sorted(KNOWLEDGE_DOMAIN_TAG_SET)


@pytest.mark.skipif(len(_REAL_TAGS) < 1, reason="no knowledge-domain tags")
def test_domains_scope_requires_multiple_tags(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
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
    assert exc.value.rule == "specialist_scope_too_narrow"


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
def test_single_domain_scope_rejects_multiple_tags(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
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
    assert exc.value.rule == "specialist_scope_mismatch"


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


def test_bench_specialist_without_explicit_needs_gpu_is_gated(orchestration_role):
    """A bench-enabled (mode=patch & bench=true) specialist auto-defaults
    needs_gpu=True at dispatch (_warm_specialist_params); the gate must mirror
    that so it is rejected when the pool is disabled instead of slipping past
    the no-needs_gpu early return and stalling as an unschedulable GPU task."""
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
                    # NOTE: no explicit needs_gpu — the bench profile implies it.
                }
            ),
        )
    assert exc.value.rule == "specialist_gpu_pool_disabled"


def test_bench_specialist_whole_machine_lane_allows_full_node(orchestration_role, monkeypatch):
    """A bench specialist takes the whole-machine, time-shared GPU lane
    (serialized with serving via ``gpu_research_lane``; server torn down between
    rounds), so serving occupying the whole node (TP == #GPUs) NO LONGER denies
    it — the serving-disjoint carve does not apply to the whole-machine pool.

    Regression for the EXPLORE-phase GPU-specialist fix: previously this was
    rejected with ``specialist_gpu_request_exceeds_capacity`` (disjoint pool
    size 0)."""
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP", "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    gate = _gate_with_gpu_capacity(4, tp=4)
    # gpu_count is floored to serving TP=4; the whole-machine pool has all 4
    # cards (no serving carve), so this is now schedulable and must not raise.
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
    # Must not raise: research/CPU dispatch never contends for the GPU pool.
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


# --------------------------------------------------------------------------- #
# Tool surface: TodoWrite and Task are granted (single-layer leaf fan-out;
# leaves inherit the parent's VISIBLE_DEVICES so they cannot oversubscribe).
# --------------------------------------------------------------------------- #
def test_task_tool_granted_and_todowrite_granted():
    from hyperloom.orchestrator.specialists.runner import (
        DEFAULT_SPECIALIST_TOOLS,
        SPECIALIST_TOOL_DENYLIST,
        SpecialistRunner,
    )

    assert "TodoWrite" in DEFAULT_SPECIALIST_TOOLS
    assert "Task" in DEFAULT_SPECIALIST_TOOLS
    assert "Task" not in SPECIALIST_TOOL_DENYLIST

    runner = SpecialistRunner(backend_factory=lambda *a, **k: None)

    # Default tool surface includes Task and TodoWrite.
    default_resolved = runner._resolve_tools(None)
    assert "Task" in default_resolved
    assert "TodoWrite" in default_resolved

    resolved = runner._resolve_tools(["Read", "Task", "TodoWrite", "Bash"])
    assert {"Read", "Task", "TodoWrite", "Bash"} <= set(resolved)


# --------------------------------------------------------------------------- #
# Problem 2 — domain KEY -> kb_anchor translation (fixes specialist_unknown_domain
# on legitimate keys leaking through params.tags untranslated).
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


def test_normalize_dispatch_tags_passes_garbage_through_for_rejection():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # Genuinely unknown tags are NOT invented into an anchor — they pass
    # through verbatim so PolicyGate's specialist_unknown_domain still fires.
    assert normalize_dispatch_tags({"tags": ["totally_bogus"]}) == ["totally_bogus"]


def test_normalize_dispatch_tags_domain_alias_translated():
    from hyperloom.orchestrator.specialists.domains import normalize_dispatch_tags

    # The legacy single-tag params.domain alias is translated the same way.
    assert normalize_dispatch_tags({"domain": "system_specialist"}) == ["systems"]


def test_dispatch_with_domain_key_tag_no_longer_rejected(gate, orchestration_role):
    """The exact bug: tags=['serving_specialist'] (a KEY) used to be mis-rejected
    as specialist_unknown_domain. It must now validate cleanly (key->framework)."""
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


def test_dispatch_with_genuine_garbage_tag_still_rejected(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
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
    assert exc.value.rule == "specialist_unknown_domain"


def test_specialist_emit_hint_lists_all_eight_llm_domains():
    """Fix 2: the specialist emit hint must enumerate all 8 LLM-selectable
    domains (the two read-only scouts were previously omitted)."""
    from types import SimpleNamespace

    from hyperloom.orchestrator.prompts.prompt_builder import (
        _format_emit_hint,
    )

    # _format_emit_hint only reads meta.name.
    hint = _format_emit_hint(SimpleNamespace(name="specialist"))
    for key in (
        "serving_specialist",
        "kernel_switch_specialist",
        "comm_specialist",
        "compiler_specialist",
        "system_specialist",
        "pr_intel_specialist",
        "research_scout_specialist",
        "static_recon_specialist",
    ):
        assert key in hint, key
