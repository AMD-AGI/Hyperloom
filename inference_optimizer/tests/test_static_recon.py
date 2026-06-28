# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the static-recon specialist (explore-opt-5 capability A).

Covers the seed checklist lookup/rendering, domain registration, and the
Coordinator-side ``_consume_static_recon`` gap-seeding (bridge candidates ->
gaps[]), without spinning up a full Coordinator.
"""

from __future__ import annotations

from types import SimpleNamespace

from inference_optimizer.orchestrator import static_recon_checklist as src
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.specialist_domains import (
    SPECIALIST_DOMAIN_KEYS,
    get_domain,
)


# --- checklist lookup -------------------------------------------------------
def test_checklist_fp8_rocm_matches_cutlass_guard():
    ids = [e.id for e in src.entries_for(gpu_type="MI300X", precision="fp8")]
    assert "rocm.fp8.cutlass_only_guard" in ids
    # mxfp8-only entry must NOT leak into an fp8 run.
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
    assert dirs  # non-empty
    assert len(dirs) == len(set(dirs))  # de-duplicated
    assert "vllm/model_executor/layers/quantization/" in dirs


def test_render_checklist_for_prompt_empty():
    assert src.render_checklist_for_prompt([]) == ""


def test_render_checklist_for_prompt_includes_id_and_bridge():
    entries = src.entries_for(gpu_type="MI300X", precision="fp8")
    text = src.render_checklist_for_prompt(entries)
    assert "rocm.fp8.cutlass_only_guard" in text
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
    Coordinator._consume_static_recon(stub, payload)
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
                {"id": "no_file", "why_disabled_here": "x"},  # missing predicate_file
                {"id": "no_why", "predicate_file": "a.py"},  # missing why
                "not-a-dict",
            ],
        },
    }
    Coordinator._consume_static_recon(stub, payload)
    assert stub.shared_state.gaps == []


def test_consume_static_recon_handles_missing_recon_block(tmp_path):
    stub = _stub_coord(tmp_path)
    Coordinator._consume_static_recon(stub, {"domain": "static_recon_specialist"})
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
    Coordinator._consume_static_recon(stub, payload)
    cids = [g["canonical_id"] for g in stub.shared_state.gaps]
    assert len(cids) == 1
    assert cids[0].startswith("gap.static_recon.")
    assert " " not in cids[0] and "/" not in cids[0]
