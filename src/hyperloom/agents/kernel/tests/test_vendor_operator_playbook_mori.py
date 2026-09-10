###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Vendor-operator-playbook routing for mori's EP dispatch/combine."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
_BACKENDS_DIR = _TOOLS_DIR / "backends"
if str(_BACKENDS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKENDS_DIR))

import tracelens_analysis as tla  # noqa: E402
from _vendor_operator_playbooks import (  # noqa: E402
    load_vendor_operator_playbooks,
    match_vendor_operator_playbook,
    resolve_kernel_anchor_path,
    _reset_vendor_operator_playbooks_cache,
    _role_haystack,
)

_MORI_SITE_PACKAGES_FILE = "/opt/venv/lib/python3.12/site-packages/mori/ops/dispatch_combine.py"


@pytest.fixture(autouse=True)
def _fresh_registry_cache():
    _reset_vendor_operator_playbooks_cache()
    yield
    _reset_vendor_operator_playbooks_cache()


def _mori_dispatch_candidate(**overrides) -> dict:
    candidate = {
        "kernel_id": "k010",
        "name": "mori::EpDispatchCombineOp::dispatch",
        "operation": "dispatch",
        "duration_us": 700.0,
        "call_count": 10,
        "source_file": _MORI_SITE_PACKAGES_FILE,
        "source_type": "unknown",
        "library": "mori",
        "shapes": [],
    }
    candidate.update(overrides)
    return candidate


def _mori_combine_candidate(**overrides) -> dict:
    candidate = {
        "kernel_id": "k011",
        "name": "mori::EpDispatchCombineOp::combine",
        "operation": "combine",
        "duration_us": 300.0,
        "call_count": 10,
        "source_file": _MORI_SITE_PACKAGES_FILE,
        "source_type": "unknown",
        "library": "mori",
        "shapes": [],
    }
    candidate.update(overrides)
    return candidate


# --- 1. registry matcher -----------------------------------------------------


def test_match_vendor_operator_playbook_matches_mori_dispatch_and_combine():
    dispatch_match = match_vendor_operator_playbook(_mori_dispatch_candidate())
    combine_match = match_vendor_operator_playbook(_mori_combine_candidate())

    assert dispatch_match is not None
    assert dispatch_match["id"] == "mori_ep_dispatch_combine"
    assert dispatch_match["role"] == "dispatch"
    assert combine_match is not None
    assert combine_match["id"] == "mori_ep_dispatch_combine"
    assert combine_match["role"] == "combine"


def test_role_haystack_takes_trailing_segment_of_fully_qualified_operation():
    """A fully-qualified ``operation`` must not reintroduce the dispatch/ combine ambiguity _last_symbol_segment() exists to resolve."""
    combine_candidate = {
        "name": "mori::EpDispatchCombineOp::combine",
        "operation": "mori::EpDispatchCombineOp::combine",
        "library": "mori",
    }
    assert _role_haystack(combine_candidate) == "combine"

    dispatch_candidate = {
        "name": "mori::EpDispatchCombineOp::dispatch",
        "operation": "mori::EpDispatchCombineOp::dispatch",
        "library": "mori",
    }
    assert _role_haystack(dispatch_candidate) == "dispatch"

    combine_match = match_vendor_operator_playbook(combine_candidate)
    assert combine_match is not None
    assert combine_match["role"] == "combine"


def test_match_vendor_operator_playbook_ignores_unrelated_kernels():
    gemm_candidate = {
        "name": "aiter::gemm_a8w8",
        "operation": "gemm",
        "source_file": "/sgl-workspace/aiter/aiter/gemm_a8w8.py",
        "library": "aiter",
    }
    assert match_vendor_operator_playbook(gemm_candidate) is None
    # "mori" alone (no dispatch/combine marker) must not match either -- the registry requires both an identity marker
    # and a role marker.
    mori_other = {"name": "mori::shmem::init", "library": "mori"}
    assert match_vendor_operator_playbook(mori_other) is None


def test_match_vendor_operator_playbook_matches_via_trace_launcher_file_when_graph_captured():
    """A CUDA/HIP-graph-captured launch is reconstructed by TraceLens as a "Synthetic Op" (e.g."""
    dispatch_candidate = {
        "name": "vllm::moe_forward_shared->EpDispatchIntraNodeKernel_bf16 (Synthetic Op)",
        "device_kernel_name": "EpDispatchIntraNodeKernel_bf16",
        "operation": "",
        "library": "",
        "source_file": "",
        "kernel_repo": "",
        "trace_launcher_file": "/usr/local/lib/python3.12/dist-packages/mori/jit/hip_driver.py",
    }
    combine_candidate = {
        "name": "hipGraphLaunch->EpCombineIntraNodeKernel_bf16_nop2p (Synthetic Op)",
        "device_kernel_name": "EpCombineIntraNodeKernel_bf16_nop2p",
        "operation": "",
        "library": "",
        "source_file": "",
        "kernel_repo": "",
        "trace_launcher_file": "/usr/local/lib/python3.12/dist-packages/mori/jit/hip_driver.py",
    }

    dispatch_match = match_vendor_operator_playbook(dispatch_candidate)
    combine_match = match_vendor_operator_playbook(combine_candidate)

    assert dispatch_match is not None
    assert dispatch_match["id"] == "mori_ep_dispatch_combine"
    assert dispatch_match["role"] == "dispatch"
    assert combine_match is not None
    assert combine_match["id"] == "mori_ep_dispatch_combine"
    assert combine_match["role"] == "combine"


# --- 2. classify_patchability + _finalize_candidates -------------------------


def test_classify_patchability_routes_mori_dispatch_and_combine():
    dispatch_ok, dispatch_reason = tla.classify_patchability(_mori_dispatch_candidate())
    combine_ok, combine_reason = tla.classify_patchability(_mori_combine_candidate())

    assert (dispatch_ok, dispatch_reason) == (True, "")
    assert (combine_ok, combine_reason) == (True, "")


def test_classify_patchability_routes_graph_captured_mori_synthetic_ops():
    """Same as ``test_classify_patchability_routes_mori_dispatch_and_combine`` but for the real, graph-captured candidate shape (empty library/ source_file/kernel_repo, mori identity only in ``trace_launcher_file``) -- without the ``_candidate_haystack`` fix this fell through to ``"source file not resolved"`` instead of routing to the playbook."""
    dispatch_candidate = {
        "name": "vllm::moe_forward_shared->EpDispatchIntraNodeKernel_bf16 (Synthetic Op)",
        "library": "",
        "source_file": "",
        "kernel_repo": "",
        "trace_launcher_file": "/usr/local/lib/python3.12/dist-packages/mori/jit/hip_driver.py",
    }
    combine_candidate = {
        "name": "hipGraphLaunch->EpCombineIntraNodeKernel_bf16_nop2p (Synthetic Op)",
        "library": "",
        "source_file": "",
        "kernel_repo": "",
        "trace_launcher_file": "/usr/local/lib/python3.12/dist-packages/mori/jit/hip_driver.py",
    }

    dispatch_ok, dispatch_reason = tla.classify_patchability(dispatch_candidate)
    combine_ok, combine_reason = tla.classify_patchability(combine_candidate)

    assert (dispatch_ok, dispatch_reason) == (True, "")
    assert (combine_ok, combine_reason) == (True, "")


def test_finalize_candidates_stamps_vendor_playbook_and_sums_gpu_pct():
    candidates = [
        _mori_dispatch_candidate(),
        _mori_combine_candidate(),
        {
            "name": "rmsnorm_kernel",
            "duration_us": 100.0,
            "call_count": 10,
            "source_file": "/path/to/rmsnorm.cu",
            "source_type": "hip_cpp",
            "shapes": [[16, 1024]],
        },
    ]
    # total_dur = 700 + 300 + 100 = 1100 -> dispatch=63.636%, combine=27.273%.
    out = tla._finalize_candidates(candidates, total_dur=1100.0)
    by_name = {item["name"]: item for item in out}

    dispatch = by_name["mori::EpDispatchCombineOp::dispatch"]
    combine = by_name["mori::EpDispatchCombineOp::combine"]
    other = by_name["rmsnorm_kernel"]

    for item in (dispatch, combine):
        assert item["reusable_native_kernel"] is True
        assert item["patch_strategy"] == "vendor_playbook"
        assert item["vendor_operator_playbook"]["id"] == "mori_ep_dispatch_combine"
        assert item["vendor_playbook_group_id"] == "mori_ep_dispatch_combine"

    assert dispatch["vendor_playbook_role"] == "dispatch"
    assert combine["vendor_playbook_role"] == "combine"
    assert dispatch["vendor_playbook_aggregate_gpu_pct"] == pytest.approx(combine["vendor_playbook_aggregate_gpu_pct"])
    assert dispatch["vendor_playbook_aggregate_gpu_pct"] == pytest.approx(dispatch["gpu_pct"] + combine["gpu_pct"])
    assert dispatch["vendor_playbook_aggregate_gpu_pct"] == pytest.approx(90.909, abs=0.01)
    assert sorted(dispatch["vendor_playbook_group_kernel_ids"]) == sorted([dispatch["kernel_id"], combine["kernel_id"]])

    # An unrelated candidate must be untouched by the vendor-playbook pass.
    assert "patch_strategy" not in other
    assert "vendor_operator_playbook" not in other
    assert "vendor_playbook_aggregate_gpu_pct" not in other


def test_finalize_candidates_fills_source_file_for_real_vendor_binary_shape():
    """mori's dispatch/combine are compiled bindings with no on-disk .py/.cu source -- TraceLens realistically hands classify_patchability a candidate with an *empty* source_file (unlike the fixtures above, which set one for unrelated reasons)."""
    candidates = [
        _mori_dispatch_candidate(source_file=""),
        _mori_combine_candidate(source_file=""),
    ]
    out = tla._finalize_candidates(candidates, total_dur=1000.0)
    by_name = {item["name"]: item for item in out}
    dispatch = by_name["mori::EpDispatchCombineOp::dispatch"]
    combine = by_name["mori::EpDispatchCombineOp::combine"]

    for item in (dispatch, combine):
        assert item["reusable_native_kernel"] is True
        source_file = str(item.get("source_file") or "")
        assert source_file, "source_file must not be empty (would be skipped as missing_native_source)"
        assert source_file.endswith("mori_ep_config.py")
        # The stand-in must look like a real path so it survives looks_like_source_path()/the non-empty CLI gate
        # either way.
        assert tla.looks_like_source_path(source_file)


def test_playbook_anchor_overrides_a_same_word_grep_collision():
    """A registry match outranks whatever the grep tier guessed."""
    collision = "/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/x_dispatch.h"
    candidates = [
        _mori_dispatch_candidate(source_file=collision),
        _mori_combine_candidate(source_file=collision),
    ]
    out = tla._finalize_candidates(candidates, total_dur=1000.0)

    for item in out:
        assert item["patch_strategy"] == "vendor_playbook"
        assert item["source_file"] != collision
        assert str(item["source_file"]).endswith("mori_ep_config.py")


def test_playbook_anchor_also_overrides_a_correct_grep_hit():
    """The override does not depend on the guess being wrong."""
    candidates = [
        _mori_dispatch_candidate(source_file=_MORI_SITE_PACKAGES_FILE),
        _mori_combine_candidate(source_file=_MORI_SITE_PACKAGES_FILE),
    ]
    out = tla._finalize_candidates(candidates, total_dur=1000.0)

    for item in out:
        assert item["patch_strategy"] == "vendor_playbook"
        assert str(item["source_file"]).endswith("mori_ep_config.py")
        assert item["source_file_superseded_by_playbook"] == _MORI_SITE_PACKAGES_FILE


def test_an_anchor_that_replaces_nothing_leaves_no_breadcrumb(monkeypatch):
    """Nothing displaced, nothing recorded."""
    monkeypatch.setattr(tla, "kernel_search_roots", lambda: ())
    candidates = [_mori_dispatch_candidate(source_file="")]
    out = tla._finalize_candidates(candidates, total_dur=1000.0)

    assert str(out[0]["source_file"]).endswith("mori_ep_config.py")
    assert "source_file_superseded_by_playbook" not in out[0]


def test_the_registry_refuses_an_entry_with_no_kernel_anchor(monkeypatch, caplog, tmp_path):
    """The anchorless entry is kept out of the pipeline, not guarded against."""
    registry = tmp_path / "vendor_operator_playbooks.json"
    registry.write_text(
        json.dumps(
            {
                "playbooks": [
                    {"id": "with-anchor", "kernel_anchor": "mori_ep_config.py"},
                    {"id": "no-anchor", "role": "dispatch"},
                    {"id": "blank-anchor", "kernel_anchor": "   "},
                ]
            }
        ),
        encoding="utf-8",
    )
    # Redirect the registry rather than patching ``Path.read_text``, which is ``pathlib.Path``'s and would answer for
    # every read in the process.
    monkeypatch.setattr("_vendor_operator_playbooks._REGISTRY_PATH", registry)
    _reset_vendor_operator_playbooks_cache()
    with caplog.at_level(logging.WARNING, logger="_vendor_operator_playbooks"):
        loaded = load_vendor_operator_playbooks()
    _reset_vendor_operator_playbooks_cache()

    assert [entry["id"] for entry in loaded] == ["with-anchor"]
    assert "no-anchor" in caplog.text
    assert "blank-anchor" in caplog.text


def _write_fake_mori_bundle(project_root: Path) -> Path:
    """Plant a substitute bundle where ``resource_path`` looks before the package."""
    bundle = project_root / "examples" / "mori_ep_dispatch_combine"
    bundle.mkdir(parents=True)
    (bundle / "mori_ep_config.py").write_text("def get_ep_launch_config():\n    return {}\n", encoding="utf-8")
    (bundle / "driver.py").write_text("# real, hand-written mori driver\n", encoding="utf-8")
    (bundle / "program.md").write_text("# mori dispatch/combine task\n", encoding="utf-8")
    return bundle


def test_resolve_kernel_anchor_path_is_always_absolute(monkeypatch, tmp_path):
    """A relative ``source_file`` stand-in is later reinterpreted by ``Path(...).resolve()`` against whatever the apply-stage process's CWD happens to be, not against the KernelForge bundle it was meant to name -- resolve_kernel_anchor_path() must never return a bare relative string, whether it resolves against the packaged tree or against an operator's $KERNELFORGE_PROJECT_ROOT substitution (PR #1191 review finding #8)."""
    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    assert playbook is not None

    packaged_anchor = resolve_kernel_anchor_path(playbook)
    assert packaged_anchor
    assert Path(packaged_anchor).is_absolute()
    # With the bundle packaged, the stand-in names a file that actually exists rather than a synthetic
    # /nonexistent-forge-path placeholder.
    assert Path(packaged_anchor).is_file()

    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    overridden_anchor = resolve_kernel_anchor_path(playbook)
    assert overridden_anchor
    assert Path(overridden_anchor).is_absolute()
    assert overridden_anchor.startswith(str(project_root))


def test_registry_json_is_valid_and_ships_in_package_data():
    """The JSON registry parses and pyproject.toml declares it as package-data (mirrors KernelForge's own wheel-packaging regression for framework/mori/)."""
    registry_path = _TOOLS_DIR / "vendor_operator_playbooks.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    playbook_ids = {p["id"] for p in data["playbooks"]}
    assert "mori_ep_dispatch_combine" in playbook_ids

    pyproject = registry_path.parents[5] / "pyproject.toml"
    assert pyproject.is_file()
    text = pyproject.read_text(encoding="utf-8")
    assert "vendor_operator_playbooks.json" in text
