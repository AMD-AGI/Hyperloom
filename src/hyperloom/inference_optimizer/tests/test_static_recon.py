# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the static-recon specialist."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge import static_recon_checklist as src
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.phases.internal import InternalTasksPhase
from hyperloom.orchestrator.specialists.domains import (
    SPECIALIST_DOMAIN_KEYS,
    get_domain,
)


# --- checklist lookup -------------------------------------------------------
def test_checklist_fp8_rocm_matches_cutlass_guard():
    ids = [e.id for e in src.entries_for(gpu_type="MI300X", precision="fp8")]
    assert "rocm.fp8.cutlass_only_guard" in ids
    assert "rocm.mxfp8.smallm_dispatch_gap" not in ids


def test_checklist_precision_is_token_exact_not_substring():
    """``fp8`` must not match ``mxfp8`` and vice versa."""
    fp8 = {e.id for e in src.entries_for(gpu_type="MI300X", precision="fp8")}
    mxfp8 = {e.id for e in src.entries_for(gpu_type="MI355X", precision="mxfp8")}
    assert "rocm.mxfp8.smallm_dispatch_gap" in mxfp8
    assert "rocm.mxfp8.smallm_dispatch_gap" not in fp8
    assert "rocm.fp8.cutlass_only_guard" in fp8
    assert "rocm.fp8.cutlass_only_guard" not in mxfp8


def test_checklist_precision_tokenizes_compound_value():
    """A compound precision like ``fp8_e4m3`` still matches the ``fp8`` entry."""
    ids = {e.id for e in src.entries_for(gpu_type="MI300X", precision="fp8_e4m3")}
    assert "rocm.fp8.cutlass_only_guard" in ids


def test_checklist_non_rocm_gpu_yields_nothing():
    assert src.entries_for(gpu_type="H100", precision="fp8") == []


def test_checklist_any_precision_entry_applies_to_rocm_bf16():
    ids = [e.id for e in src.entries_for(gpu_type="MI300X", precision="bf16")]
    assert ids == ["rocm.moe.aiter_backend_activation_gap"]


def test_source_hint_directories_dedup_and_ordered():
    dirs = src.source_hint_directories_for(gpu_type="MI300X", precision="fp8")
    assert dirs
    assert len(dirs) == len(set(dirs))
    assert "vllm/model_executor/layers/quantization/" in dirs


def test_render_checklist_for_prompt_empty():
    assert src.render_checklist_for_prompt([]) == ""


def test_render_checklist_for_prompt_includes_id_and_bridge():
    entries = src.entries_for(gpu_type="MI300X", precision="fp8")
    text = src.render_checklist_for_prompt(entries)
    assert "rocm.fp8.cutlass_only_guard" in text
    assert "bridge:" in text


# --- shared-expert fusion entry ---------------------------------------------


def test_shared_expert_fusion_entry_in_mxfp8_rocm():
    """rocm.moe.shared_expert_fusion must appear for ROCm + mxfp8 runs."""
    ids = [e.id for e in src.entries_for(gpu_type="MI355X", precision="mxfp8")]
    assert "rocm.moe.shared_expert_fusion" in ids


def test_shared_expert_fusion_entry_not_in_fp8():
    """rocm.moe.shared_expert_fusion must NOT appear for plain fp8 (mxfp8-only entry)."""
    ids = [e.id for e in src.entries_for(gpu_type="MI300X", precision="fp8")]
    assert "rocm.moe.shared_expert_fusion" not in ids


def test_shared_expert_fusion_source_dirs_include_vllm_and_sglang():
    entries = src.entries_for(gpu_type="MI355X", precision="mxfp8")
    fusion = next(e for e in entries if e.id == "rocm.moe.shared_expert_fusion")
    dirs = " ".join(fusion.source_dirs)
    assert "vllm/model_executor/layers/fused_moe/" in dirs
    assert "python/sglang/srt/layers/moe/" in dirs


def test_filter_entries_for_model_keeps_entry_when_has_shared_expert():
    """filter_entries_for_model passes the shared-expert entry through when flag is set."""
    entries = src.entries_for(gpu_type="MI355X", precision="mxfp8")
    filtered = src.filter_entries_for_model(entries, {"has_shared_expert": True})
    ids = [e.id for e in filtered]
    assert "rocm.moe.shared_expert_fusion" in ids


def test_filter_entries_for_model_removes_entry_when_no_shared_expert():
    """Core anti-noise assertion: entry is dropped for plain MoE / non-MoE models."""
    entries = src.entries_for(gpu_type="MI355X", precision="mxfp8")
    for model_info in ({}, {"has_shared_expert": False}, {"is_moe": True}):
        filtered = src.filter_entries_for_model(entries, model_info)
        ids = [e.id for e in filtered]
        assert "rocm.moe.shared_expert_fusion" not in ids, f"entry leaked through for model_info={model_info}"


def test_filter_entries_for_model_preserves_other_entries():
    """Other checklist entries must not be affected by the filter."""
    entries = src.entries_for(gpu_type="MI355X", precision="mxfp8")
    filtered = src.filter_entries_for_model(entries, {})
    other_ids = {e.id for e in filtered}
    all_ids = {e.id for e in entries}
    assert all_ids - {"rocm.moe.shared_expert_fusion"} == other_ids


def test_render_checklist_for_prompt_includes_shared_expert_fusion():
    """render_checklist_for_prompt emits the entry id and bridge text."""
    entries = src.entries_for(gpu_type="MI355X", precision="mxfp8")
    shared = [e for e in entries if e.id == "rocm.moe.shared_expert_fusion"]
    text = src.render_checklist_for_prompt(shared)
    assert "rocm.moe.shared_expert_fusion" in text
    assert "bridge:" in text


# --- domain registration ----------------------------------------------------
def test_static_recon_domain_registered():
    assert "static_recon_specialist" in SPECIALIST_DOMAIN_KEYS
    d = get_domain("static_recon_specialist")
    assert d is not None
    assert d.kb_anchor == "static_recon"


# --- _consume_static_recon gap seeding --------------------------------------
def _stub_coord(session_dir):
    """A minimal stand-in carrying the attributes _consume_static_recon touches."""
    state = SharedState(session_id="t", model_name="m")
    return SimpleNamespace(shared_state=state, session_dir=session_dir)


def test_consume_static_recon_seeds_gaps(tmp_path):
    stub = _stub_coord(tmp_path)
    payload = {
        "domain": "static_recon_specialist",
        "recon": {
            "bridge_candidates": [
                {
                    "id": "rocm.fp8.cutlass_only_guard",
                    "predicate_file": "vllm/.../fp8.py",
                    "predicate_name": "cutlass_fp8_supported",
                    "why_disabled_here": "CUDA-only; False on ROCm",
                    "consequence": "dense Linear falls back to bf16",
                    "bridge_sketch": "use per-token act + per-channel weight",
                    "domain_hint": "freeform",
                },
            ],
        },
    }
    InternalTasksPhase.consume_static_recon(stub, payload)
    gap = stub.shared_state.find_gap("gap.static_recon.rocm.fp8.cutlass_only_guard")
    assert gap is not None
    assert gap["layer"] == "static_recon"
    assert gap["domain_hint"] == "freeform"
    assert "cutlass_fp8_supported" in gap["symptom"]
    assert "Bridge:" in gap["symptom"]


def test_consume_static_recon_drops_incomplete_candidates(tmp_path):
    stub = _stub_coord(tmp_path)
    payload = {
        "recon": {
            "bridge_candidates": [
                {"id": "no_file", "why_disabled_here": "x"},
                {"id": "no_why", "predicate_file": "a.py"},
                "not-a-dict",
            ],
        },
    }
    InternalTasksPhase.consume_static_recon(stub, payload)
    assert stub.shared_state.gaps == []


def test_consume_static_recon_handles_missing_recon_block(tmp_path):
    stub = _stub_coord(tmp_path)
    InternalTasksPhase.consume_static_recon(stub, {"domain": "static_recon_specialist"})
    assert stub.shared_state.gaps == []


def test_consume_static_recon_sanitizes_id_into_canonical(tmp_path):
    stub = _stub_coord(tmp_path)
    payload = {
        "recon": {
            "bridge_candidates": [
                {
                    "id": "weird id/with spaces",
                    "predicate_file": "a.py",
                    "why_disabled_here": "because",
                },
            ],
        },
    }
    InternalTasksPhase.consume_static_recon(stub, payload)
    cids = [g["canonical_id"] for g in stub.shared_state.gaps]
    assert len(cids) == 1
    assert cids[0].startswith("gap.static_recon.")
    assert " " not in cids[0] and "/" not in cids[0]


_SEED_PAYLOAD = {"recon": {"bridge_candidates": [{"id": "g1", "predicate_file": "a.py", "why_disabled_here": "w"}]}}


def _dispatch_stub(session_dir):
    """A stand-in carrying what the static-recon enqueue touches, with a task registry that mints a fresh row."""

    async def _warm(params):
        return None

    async def _create(spec):
        return SimpleNamespace(task_id="t-recon"), False

    state = SharedState(session_id="t", model_name="m")
    state.static_recon_enabled = True
    return SimpleNamespace(
        shared_state=state,
        session_dir=session_dir,
        tasks=SimpleNamespace(create_or_return_existing=_create),
        _coord=SimpleNamespace(
            specialist_dispatch=SimpleNamespace(warm_specialist_params=_warm),
            gpu_lanes=SimpleNamespace(framework_authoring_lanes_ttl=lambda params, base_ttl_sec: ([], base_ttl_sec)),
        ),
    )


def _consume_site(session_dir):
    stub = _stub_coord(session_dir)
    return stub.shared_state, lambda: InternalTasksPhase.consume_static_recon(stub, _SEED_PAYLOAD)


def _enqueue_site(session_dir):
    stub = _dispatch_stub(session_dir)
    return stub.shared_state, lambda: asyncio.run(
        InternalTasksPhase._enqueue_internal_static_recon_task(stub, reason="r", slug="s")
    )


_SAVE_SITES = {
    "consume": (
        _consume_site,
        lambda state: [g["canonical_id"] for g in state.gaps] == ["gap.static_recon.g1"],
    ),
    "enqueue": (_enqueue_site, lambda state: state.static_recon_runs == 1),
}


@pytest.mark.parametrize("site", _SAVE_SITES)
def test_static_recon_save_os_error_is_logged_and_the_bookkeeping_stays(tmp_path, monkeypatch, caplog, site):
    make, kept = _SAVE_SITES[site]
    state, run = make(tmp_path)
    err = OSError("disk full")
    monkeypatch.setattr(state, "save", _raiser(err))

    with caplog.at_level(logging.ERROR, logger="hyperloom.orchestrator.phases.internal"):
        run()

    assert kept(state)
    assert [r.exc_info[1] for r in caplog.records if r.exc_info] == [err]


@pytest.mark.parametrize("site", _SAVE_SITES)
def test_static_recon_save_serialization_defect_propagates(tmp_path, monkeypatch, site):
    state, run = _SAVE_SITES[site][0](tmp_path)
    monkeypatch.setattr(state, "save", _raiser(TypeError("not JSON serializable")))

    with pytest.raises(TypeError, match="not JSON serializable"):
        run()


def _raiser(exc: BaseException):
    def _save(session_dir=None) -> None:
        raise exc

    return _save


# -- workload precision -----------------------------------------------------
class _PrecisionState:
    """Minimal SharedState stand-in carrying the two precision sources."""

    def __init__(self, precision: str, model_info: dict) -> None:
        self.precision = precision
        self.model_info = model_info


def test_workload_precision_prefers_the_checkpoint_over_the_env_label():
    state = _PrecisionState("fp8", {"quantization": "mxfp8"})
    assert src.workload_precision(state) == "mxfp8"


def test_workload_precision_falls_back_to_shared_state():
    assert src.workload_precision(_PrecisionState("fp8", {})) == "fp8"
    assert src.workload_precision(_PrecisionState("", {})) == ""


def test_mxfp8_checkpoint_gets_the_mxfp8_focus_directories():
    """An MXFP8 model labelled fp8 by $PRECISION must not get the fp8 entries."""
    precision = src.workload_precision(_PrecisionState("fp8", {"quantization": "mxfp8"}))
    dirs = src.source_hint_directories_for(gpu_type="MI355X", precision=precision)
    assert "vllm/model_executor/kernels/linear/mxfp8/" in dirs
    assert "vllm/model_executor/layers/fused_moe/experts/" in dirs
    assert "vllm/model_executor/layers/quantization/" not in dirs


# --- config-arm plateau re-dispatch -----------------------------------------
def _recon_recorder(*, phase: str, plateaued: bool, monkeypatch):
    """Stub whose ``_enqueue_internal_static_recon_task`` records its kwargs."""
    calls: list[dict] = []

    async def _enqueue(**kwargs):
        calls.append(kwargs)

    state = SharedState(session_id="t", model_name="m")
    state.phase = phase
    state.macro_cycle = 2
    monkeypatch.setattr(
        machine_state,
        "per_lever_dryness",
        lambda _state: (False, {"config_arm_plateaued": plateaued}),
    )
    stub = SimpleNamespace(shared_state=state, _enqueue_internal_static_recon_task=_enqueue)
    return stub, calls


def test_config_plateau_recon_dispatches_under_its_own_idempotency_key(monkeypatch):
    """The plateau pass must not be deduplicated against the PRELUDE one."""
    stub, calls = _recon_recorder(phase="FRAMEWORK_AGENT", plateaued=True, monkeypatch=monkeypatch)
    asyncio.run(InternalTasksPhase.maybe_enqueue_config_plateau_static_recon(stub))
    assert calls == [{"reason": "config_arm_plateau", "slug": "cycle2"}]


def test_config_plateau_recon_silent_while_the_config_arm_still_pays(monkeypatch):
    stub, calls = _recon_recorder(phase="FRAMEWORK_AGENT", plateaued=False, monkeypatch=monkeypatch)
    asyncio.run(InternalTasksPhase.maybe_enqueue_config_plateau_static_recon(stub))
    assert calls == []


def test_config_plateau_recon_is_scoped_to_the_optimisation_phase(monkeypatch):
    stub, calls = _recon_recorder(phase="SWEEP", plateaued=True, monkeypatch=monkeypatch)
    asyncio.run(InternalTasksPhase.maybe_enqueue_config_plateau_static_recon(stub))
    assert calls == []


def test_config_plateau_recon_respects_the_disable_switch(monkeypatch):
    stub, calls = _recon_recorder(phase="FRAMEWORK_AGENT", plateaued=True, monkeypatch=monkeypatch)
    stub.shared_state.static_recon_enabled = False
    asyncio.run(InternalTasksPhase.maybe_enqueue_config_plateau_static_recon(stub))
    assert calls == []
