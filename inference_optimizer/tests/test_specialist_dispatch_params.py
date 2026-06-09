# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch-dial matrix + freeform sanity-gate tests for the unified specialist.

Covers the four orthogonal dials (``scope`` / ``mode`` / ``bench`` / ``lane``)
resolved by ``resolve_specialist_profile`` and the lightweight mechanical
PolicyGate that guards ``scope='freeform'`` dispatches (the channel that
absorbed the retired ``dynamic_specialist`` wave worker).
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import (
    KNOWLEDGE_DOMAIN_TAG_SET,
    PolicyDenied,
    PolicyGate,
    SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS,
    SPECIALIST_FREEFORM_WAVE_MAX,
)
from inference_optimizer.orchestrator.specialist_profile import (
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
def test_bare_dispatch_keeps_legacy_defaults():
    """An empty / legacy ``specialist`` dispatch resolves to the historical
    single-domain, patch-authoring, GPU-leased behaviour."""
    prof = resolve_specialist_profile(None)
    assert prof == SpecialistProfile(
        scope=DEFAULT_SCOPE, mode=DEFAULT_MODE, bench=DEFAULT_BENCH, lane=DEFAULT_LANE,
    )
    assert prof.scope == SCOPE_DOMAIN
    assert prof.mode == MODE_PATCH
    assert prof.bench is False
    assert prof.lane == LANE_GPU


def test_unknown_values_fall_back_without_raising():
    prof = resolve_specialist_profile(
        {"scope": "galaxy", "mode": "telepathy", "lane": "quantum"},
    )
    assert prof.scope == DEFAULT_SCOPE
    assert prof.mode == DEFAULT_MODE
    assert prof.lane == LANE_GPU  # mode resolved to patch -> gpu lane


def test_freeform_defaults_to_research_on_cpu():
    """Freeform recon is read-only research on the CPU lane unless told otherwise."""
    prof = resolve_specialist_profile({"scope": "freeform"})
    assert prof.scope == SCOPE_FREEFORM
    assert prof.mode == MODE_RESEARCH
    assert prof.lane == LANE_CPU
    assert prof.is_freeform is True
    assert prof.grants_bench_tool is False


def test_domains_scope_is_cross_domain():
    prof = resolve_specialist_profile({"scope": "domains"})
    assert prof.is_cross_domain is True
    assert prof.is_freeform is False


@pytest.mark.parametrize("truthy", [True, "true", "1", "yes", "on", 1])
def test_bench_requires_patch_mode_truthy(truthy):
    prof = resolve_specialist_profile({"mode": "patch", "bench": truthy})
    assert prof.bench is True
    assert prof.grants_bench_tool is True


@pytest.mark.parametrize("falsy", [False, "false", "0", "no", "off", 0, None])
def test_bench_falsy_values(falsy):
    prof = resolve_specialist_profile({"mode": "patch", "bench": falsy})
    assert prof.bench is False
    assert prof.grants_bench_tool is False


def test_bench_is_meaningless_for_research_mode():
    """Even an explicit ``bench=true`` is dropped when the worker can't patch."""
    prof = resolve_specialist_profile({"mode": "research", "bench": True})
    assert prof.bench is False
    assert prof.grants_bench_tool is False


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
        _dispatch({
            "scope": "freeform",
            "tasks": [
                {"task_description": "Investigate prefill batching."},
                {"task_description": "Audit KV-cache allocation."},
            ],
        }),
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
            orchestration_role, _dispatch({"scope": "freeform"}),
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


@pytest.mark.parametrize("redline", [
    "clean up with rm -rf / now",
    "run mkfs.ext4 on the scratch disk",
    "please shutdown the host afterwards",
])
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
    tasks = [
        {"task_description": f"task {i}"}
        for i in range(SPECIALIST_FREEFORM_WAVE_MAX + 1)
    ]
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
            _dispatch({
                "scope": "domains",
                "tags": [_REAL_TAGS[0]],
                "gap_canonical_id": "gap.x.session-1",
            }),
        )
    assert exc.value.rule == "specialist_scope_too_narrow"


@pytest.mark.skipif(len(_REAL_TAGS) < 2, reason="need >=2 knowledge-domain tags")
def test_single_domain_scope_rejects_multiple_tags(gate, orchestration_role):
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_specialist_dispatch(
            orchestration_role,
            _dispatch({
                "scope": "domain",
                "tags": _REAL_TAGS[:2],
                "gap_canonical_id": "gap.x.session-1",
            }),
        )
    assert exc.value.rule == "specialist_scope_mismatch"
