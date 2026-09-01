###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Vendor-operator-playbook routing for mori's EP dispatch/combine.

Closes the gap KernelForge PR #88 left explicit: Hyperloom's own
TraceLens-driven pipeline had no path to mori (a pip-installed compiled
library, classified ``vendor_binary`` with no rewritable source) even though
a validated KernelForge forge-loop task bundle exists for it. These tests
pin, end to end within Hyperloom's own boundary:

1. the registry matcher recognizes mori dispatch/combine by name (
   ``_vendor_operator_playbooks``);
2. ``classify_patchability`` + ``_finalize_candidates`` route both candidates
   to ``reusable_native_kernel=True`` / ``patch_strategy="vendor_playbook"``
   and sum their GPU share (dispatch+combine are one logical round trip);
3. ``forge_submit.submit()`` recognizes ``patch_strategy="vendor_playbook"``,
   copies the KernelForge ``examples/mori_ep_dispatch_combine/`` bundle into
   the forge worktree, and invokes forge-loop with the bundle's own
   driver/config/program.md, the mori KB env knob, and the six documented
   tunable params as ``--target-functions``;
4. dispatch and combine are invoked as **one** Forge session, not two: the
   second candidate to reach ``submit()`` reuses the first's result instead
   of launching a second forge-loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
_BACKENDS_DIR = _TOOLS_DIR / "backends"
if str(_BACKENDS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKENDS_DIR))

import tracelens_analysis as tla  # noqa: E402
import forge_submit  # noqa: E402
import kernel_optimization as ko  # noqa: E402
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
    """A fully-qualified ``operation`` must not reintroduce the dispatch/
    combine ambiguity _last_symbol_segment() exists to resolve.

    This repo's own convention (_task_group_contract.logical_operator_name,
    _bypass_report.py's task-group builder) is to set ``operation`` to a
    fully-qualified ``Class::method`` symbol. ``EpDispatchCombineOp`` itself
    contains the substring "dispatch", so taking ``operation`` verbatim
    would make a *combine* candidate whose operation is
    ``mori::EpDispatchCombineOp::combine`` match "dispatch" first (registry
    order), mislabeling it (PR #1191 review finding #6).
    """
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
    # "mori" alone (no dispatch/combine marker) must not match either --
    # the registry requires both an identity marker and a role marker.
    mori_other = {"name": "mori::shmem::init", "library": "mori"}
    assert match_vendor_operator_playbook(mori_other) is None


def test_match_vendor_operator_playbook_matches_via_trace_launcher_file_when_graph_captured():
    """A CUDA/HIP-graph-captured launch is reconstructed by TraceLens as a
    "Synthetic Op" (e.g. ``vllm::moe_forward_shared->EpDispatchIntraNodeKernel_bf16
    (Synthetic Op)`` or ``hipGraphLaunch->EpCombineIntraNodeKernel_bf16_nop2p
    (Synthetic Op)``) with no surviving module chain, so ``library``,
    ``source_file``, and ``kernel_repo`` all resolve empty -- this is the
    actual shape produced end to end for a real DeepSeek-V2 EP+DP vLLM
    serving trace, not a hypothetical. The only field that still carries the
    mori identity marker is ``trace_launcher_file``, the Python frame that
    first launched the op (``.../site-packages/mori/jit/hip_driver.py``).
    """
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
    """Same as ``test_classify_patchability_routes_mori_dispatch_and_combine``
    but for the real, graph-captured candidate shape (empty library/
    source_file/kernel_repo, mori identity only in ``trace_launcher_file``)
    -- without the ``_candidate_haystack`` fix this fell through to
    ``"source file not resolved"`` instead of routing to the playbook.
    """
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
    """mori's dispatch/combine are compiled bindings with no on-disk .py/.cu
    source -- TraceLens realistically hands classify_patchability a candidate
    with an *empty* source_file (unlike the fixtures above, which set one for
    unrelated reasons). Confirm _finalize_candidates still fills in a
    path-shaped stand-in so kernel_optimization.py's CLI gate (which skips
    any candidate with a falsy source_file as "missing_native_source" before
    it ever reaches forge_submit.submit()) does not reject the candidate.
    """
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
        # The stand-in must look like a real path so it survives
        # looks_like_source_path()/the non-empty CLI gate either way.
        assert tla.looks_like_source_path(source_file)


def test_playbook_anchor_overrides_a_same_word_grep_collision():
    """A registry match outranks whatever the grep tier guessed.

    These operators reduce to the keywords "dispatch" and "combine", which
    collide with unrelated vendor files (``mxfp4_moe_aux_dispatch.h``,
    ``fmha_fwd_d64_bf16_combine.cu``) once the search roots actually resolve.
    The registry is a curated statement that the operator is tuned through a
    task bundle, so handing a backend the colliding path would rewrite the
    wrong file.
    """
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
    """The override does not depend on the guess being wrong.

    ``dispatch_combine.py`` under site-packages really is where these operators
    live, so this is the case where the grep tier was right. The anchor still
    wins: the registry says the operator is tuned through a task bundle, and a
    backend handed the device source has nothing to rewrite there. The
    displaced path stays on the row so the override is auditable rather than
    silent.
    """
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
    """Nothing displaced, nothing recorded.

    The search roots are emptied so the grep tier cannot resolve anything: on a
    host with the frameworks installed it reaches the same-word collision this
    file's other cases describe, which is a displacement rather than the
    graph-captured no-source shape under test here.
    """
    monkeypatch.setattr(tla, "kernel_search_roots", lambda: ())
    candidates = [_mori_dispatch_candidate(source_file="")]
    out = tla._finalize_candidates(candidates, total_dur=1000.0)

    assert str(out[0]["source_file"]).endswith("mori_ep_config.py")
    assert "source_file_superseded_by_playbook" not in out[0]


def test_the_registry_refuses_an_entry_with_no_kernel_anchor(monkeypatch, caplog, tmp_path):
    """The anchorless entry is kept out of the pipeline, not guarded against.

    Every consumer of a match overrides ``source_file`` with the anchor, so an
    entry without one substitutes nothing for whatever tier resolved the path
    and lands the candidate as ``missing_native_source``. Guarding each consumer
    was tried and went wrong in a way the data never justified: the guard's
    marker gated on ``source_file``, so a row with neither anchor nor path read
    as anchor-backed, went into ``protected_ids``, and the row most in need of
    the review was refused it. Refusing the entry at load makes the shape
    unreachable instead.
    """
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
    # Redirect the registry rather than patching ``Path.read_text``, which is
    # ``pathlib.Path``'s and would answer for every read in the process.
    monkeypatch.setattr("_vendor_operator_playbooks._REGISTRY_PATH", registry)
    _reset_vendor_operator_playbooks_cache()
    with caplog.at_level(logging.WARNING, logger="_vendor_operator_playbooks"):
        loaded = load_vendor_operator_playbooks()
    _reset_vendor_operator_playbooks_cache()

    assert [entry["id"] for entry in loaded] == ["with-anchor"]
    assert "no-anchor" in caplog.text
    assert "blank-anchor" in caplog.text


def test_a_playbook_candidate_is_not_open_to_review_rewriting():
    """Protection cannot key on the tier that resolved the replaced path.

    ``source_file`` holds the task bundle's anchor whatever tier ran before, so
    keying protection on that tier alone leaves the row open: a review proposal
    could point the field back at framework source, routing the candidate to a
    backend with nothing to rewrite and losing the playbook route.
    """
    resolved_by_grep = {
        "kernel_id": "k010",
        "source_resolution_method": "name_grep",
        "op_to_source_status": "",
    }
    assert tla._is_curated_resolution(resolved_by_grep) is False

    assert tla._is_curated_resolution({**resolved_by_grep, "patch_strategy": "vendor_playbook"}) is True


# --- 3 & 4. forge_submit.submit() vendor-playbook route + one-session dedup --


#: Captured before any test monkeypatches it, so a test that injects a resolver
#: failure can hand the real one back partway through.
_real_resolve_vendor_task_bundle = forge_submit._resolve_vendor_task_bundle


def _write_fake_mori_bundle(project_root: Path) -> Path:
    """Plant a substitute bundle where ``resource_path`` looks before the package.

    ``$KERNELFORGE_PROJECT_ROOT`` is the surviving override now that $FORGE_PATH
    is gone: the layout under it mirrors the packaged data tree, so the same
    relative path resolves against either.
    """
    bundle = project_root / "examples" / "mori_ep_dispatch_combine"
    bundle.mkdir(parents=True)
    (bundle / "mori_ep_config.py").write_text("def get_ep_launch_config():\n    return {}\n", encoding="utf-8")
    (bundle / "driver.py").write_text("# real, hand-written mori driver\n", encoding="utf-8")
    (bundle / "program.md").write_text("# mori dispatch/combine task\n", encoding="utf-8")
    return bundle


def _stub_run_loop(monkeypatch, captured_calls: list[dict]):
    def fake_run_loop(**kwargs):
        captured_calls.append(kwargs)
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=1.9,
            best_ms=1.47,
            improved=True,
            output="forge-loop completed",
            error=None,
            timed_out=False,
            checkpoint=None,
            total_improved=True,
            incremental_improved=True,
            mean_case_speedup=1.29,
        )

    monkeypatch.setattr(forge_submit, "_run_vendor_playbook_loop_via_cli", fake_run_loop)


def test_submit_vendor_playbook_copies_bundle_and_invokes_forge_loop(monkeypatch, tmp_path):
    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(forge_submit, "_resolve_gpu_target", lambda _candidate: "gfx942")

    captured: list[dict] = []
    _stub_run_loop(monkeypatch, captured)

    candidate = match_vendor_operator_playbook(_mori_dispatch_candidate())
    dispatch_candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=candidate,
        vendor_playbook_role="dispatch",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")
    output_dir = tmp_path / "forge" / "session1" / "attempt_dispatch"

    result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=output_dir,
        source_type="unknown",
        candidate=dispatch_candidate,
        timeout_s=3600,
    )

    assert result["returncode"] == 0
    assert result["skipped"] is False
    assert result["vendor_playbook_id"] == "mori_ep_dispatch_combine"
    assert result["vendor_playbook_role"] == "dispatch"
    assert result["vendor_playbook_reused"] is False
    assert result["improved"] is True
    assert result["mean_case_speedup"] == pytest.approx(1.29)
    # kernel_optimization.py's build_verification() only credits a forge
    # attempt's speedup when BOTH of these are present on the result dict
    # (see run_attempt's field copy) -- missing either one silently
    # downgrades a real KEEP-worthy improvement to PARTIAL. A live e2e run
    # against the real mori_ep_dispatch_combine bundle on 8 MI300X GPUs
    # caught this gap: forge-loop measured and committed a validated
    # 1.21x speedup, but the CLI's own proposal still read PARTIAL /
    # "no measurable speedup found" because these fields were missing.
    workspace = output_dir / "worktree"
    assert result["total_improved"] is True
    assert result["incremental_improved"] is True
    assert result["forge_workspace"] == str(workspace)

    assert (workspace / "mori_ep_config.py").is_file()
    assert (workspace / "driver.py").read_text() == "# real, hand-written mori driver\n"
    assert (workspace / "program.md").is_file()
    # forge-loop's IterationLoop runs `git status`/`git checkout` against the
    # workspace to snapshot/restore each attempt; a bare copy with no `.git`
    # fails its very first git call ("not a git repository").
    assert (workspace / ".git").is_dir()
    import subprocess as _subprocess  # noqa: PLC0415

    status = _subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.strip() == "", "copied bundle must be committed (clean tree)"

    assert len(captured) == 1
    call = captured[0]
    assert call["kernel_anchor"] == str(workspace / "mori_ep_config.py")
    assert call["driver"] == str(workspace / "driver.py")
    assert call["kernel_backend"] == "aiter"
    assert call["target_functions"] == ["get_ep_launch_config", "dispatch", "combine"]
    assert call["extra_env"]["KERNELFORGE_INCLUDE_MORI_KB"] == "1"
    assert call["program_md_file"] == str(workspace / "program.md")

    # --- regression: a real measured improvement must actually reach KEEP ---
    # This is the pipeline stage the live e2e run caught failing: forge_submit's
    # result dict feeds run_attempt() -> build_verification() -> make_proposal(),
    # and a gap anywhere in that field handoff silently downgrades a validated,
    # correctness-passed, 1.2x-speedup KEEP into a PARTIAL "no measurable
    # speedup" verdict even though forge-loop itself never disagreed.
    attempt = {
        "attempt_id": "forge-e2e",
        "backend": "forge",
        "status": "completed",
        "backend_paths": {
            "output_dir": str(output_dir),
            "cli_workspace": result["cli_workspace"],
            "forge_workspace": result["forge_workspace"],
        },
        "optimized_path": str(output_dir / "forge-e2e_stdout.log"),
        "mean_case_speedup": result["mean_case_speedup"],
        "total_improved": result["total_improved"],
        "incremental_improved": result["incremental_improved"],
        "improved": result["improved"],
        "pristine_baseline_ms": None,
        "best_ms": result["best_ms"],
    }
    # main() converts the CLI's "true"/"false"/"unknown" strings to bool/None
    # before calling build_verification (see kernel_optimization.py's
    # correctness = None if ... == "unknown" else ... == "true"); mirror that
    # here rather than passing the raw CLI string.
    args = argparse.Namespace(
        source_file=str(workspace / "mori_ep_config.py"),
        kernel_repo="",
        correctness_passed=True,
        accuracy_passed=None,
        micro_speedup=None,
        e2e_gain_pct=None,
        dry_run=False,
    )
    verification = ko.build_verification(args, [attempt], benchmark_available=False)
    assert verification["micro_speedup_source"] != "default_unmeasured"
    assert verification["micro_speedup"] == pytest.approx(1.29)
    assert verification["artifact_valid"] is True, verification["artifact_error"]
    assert verification["artifact_source"] == "source_file"
    assert verification["correctness_passed"] is True

    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "KEEP", proposal["reasons"]

    # Without an orchestrator-supplied correctness signal (the standalone-CLI
    # case this e2e test actually exercised), the measured speedup must still
    # be visible -- NEEDS_REVIEW ("evidence needs confirmation"), never the
    # misleading PARTIAL "no measurable speedup found" a missing-field
    # regression would silently produce.
    args_no_correctness = argparse.Namespace(**{**vars(args), "correctness_passed": None})
    verification_nc = ko.build_verification(args_no_correctness, [attempt], benchmark_available=False)
    assert verification_nc["micro_speedup_source"] != "default_unmeasured"
    proposal_nc = ko.make_proposal(verification_nc)
    assert proposal_nc["decision"] == "NEEDS_REVIEW", proposal_nc["reasons"]


def test_submit_vendor_playbook_dedupes_dispatch_and_combine_into_one_session(monkeypatch, tmp_path):
    """dispatch+combine invoke KernelForge as ONE Forge session, not two."""
    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(forge_submit, "_resolve_gpu_target", lambda _candidate: "gfx942")

    captured: list[dict] = []
    _stub_run_loop(monkeypatch, captured)

    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    dispatch_candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=dict(playbook, role="dispatch"),
        vendor_playbook_role="dispatch",
    )
    combine_candidate = _mori_combine_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=dict(playbook, role="combine"),
        vendor_playbook_role="combine",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")

    # Same session ("session1"), two different per-kernel attempt dirs -- this
    # is exactly the shape the orchestrator produces when it dispatches
    # dispatch and combine as two separate --kernel-id subprocess attempts.
    dispatch_result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=tmp_path / "forge" / "session1" / "attempt_dispatch",
        candidate=dispatch_candidate,
        timeout_s=3600,
    )
    combine_result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=tmp_path / "forge" / "session1" / "attempt_combine",
        candidate=combine_candidate,
        timeout_s=3600,
    )

    # Only one real forge-loop invocation for the whole group.
    assert len(captured) == 1
    assert dispatch_result["vendor_playbook_reused"] is False
    assert combine_result["vendor_playbook_reused"] is True
    # The reused result carries the same measured outcome, but its own role.
    assert combine_result["mean_case_speedup"] == dispatch_result["mean_case_speedup"]
    assert combine_result["best_ms"] == dispatch_result["best_ms"]
    assert combine_result["vendor_playbook_role"] == "combine"
    assert dispatch_result["vendor_playbook_role"] == "dispatch"
    # --- regression: one measurement must not be counted twice downstream ---
    # dispatch and combine are two separate kernel_ids that each produce a
    # full result dict carrying the identical mean_case_speedup/best_ms from
    # the one shared forge-loop session; only the role that actually ran it
    # may be marked as independently counted, or a benefit collector summing
    # across kernel_ids would double-count one real 1.25x measurement as two
    # (PR #1191 review finding #4).
    assert dispatch_result["vendor_playbook_independently_counted"] is True
    assert combine_result["vendor_playbook_independently_counted"] is False
    # A second forge worktree/workspace copy is never made for the reused role.
    assert not (tmp_path / "forge" / "session1" / "attempt_combine" / "worktree").exists()

    # --- regression: the reused (combine) attempt must not lose the artifact ---
    # A real run caught this: kernel_optimization.py's invoke_backend()
    # unconditionally overwrites result["output_dir"] with THIS attempt's own
    # (empty, never-populated-by-submit) directory right after submit()
    # returns; _candidate_artifact_paths() only looks under
    # cli_workspace/optimized_versions and output_dir/optimized_versions, so
    # combine's empty tree made artifact_valid=False and make_proposal()
    # return PARTIAL even though dispatch's speedup fields were present.
    combine_output_dir = tmp_path / "forge" / "session1" / "attempt_combine"
    assert (combine_output_dir / "optimized_versions").is_dir()
    assert list((combine_output_dir / "optimized_versions").iterdir()), (
        "the reused attempt's own output_dir must carry a physical copy of "
        "the artifact, since kernel_optimization.py's invoke_backend() "
        "clobbers backend_paths['output_dir'] to point here regardless of "
        "what submit() returned"
    )
    combine_attempt = {
        "attempt_id": "forge-e2e-combine",
        "backend": "forge",
        "status": "completed",
        "backend_paths": {
            # Mirrors invoke_backend()'s real behavior: output_dir is always
            # reset to *this* attempt's own directory, while cli_workspace
            # is copied through verbatim from whatever submit() returned
            # (the winner's, for a reused result).
            "output_dir": str(combine_output_dir),
            "cli_workspace": combine_result["cli_workspace"],
        },
        "optimized_path": str(combine_output_dir / "forge-e2e-combine_stdout.log"),
        "mean_case_speedup": combine_result["mean_case_speedup"],
        "total_improved": combine_result["total_improved"],
        "incremental_improved": combine_result["incremental_improved"],
        "improved": combine_result["improved"],
        "pristine_baseline_ms": None,
        "best_ms": combine_result["best_ms"],
    }
    combine_args = argparse.Namespace(
        source_file=str(tmp_path / "forge" / "session1" / "attempt_dispatch" / "worktree" / "mori_ep_config.py"),
        kernel_repo="",
        correctness_passed=True,
        accuracy_passed=None,
        micro_speedup=None,
        e2e_gain_pct=None,
        dry_run=False,
    )
    combine_verification = ko.build_verification(combine_args, [combine_attempt], benchmark_available=False)
    assert combine_verification["artifact_valid"] is True, combine_verification["artifact_error"]
    combine_proposal = ko.make_proposal(combine_verification)
    assert combine_proposal["decision"] == "KEEP", combine_proposal["reasons"]


def test_submit_vendor_playbook_runs_from_the_packaged_bundle_without_forge_path(monkeypatch, tmp_path):
    """No $FORGE_PATH is the normal case now, and it must reach forge-loop.

    This used to be ``test_submit_vendor_playbook_reports_missing_forge_path``
    and asserted the opposite: an unset env var hard-failed the submission with
    ``skipped=True``. KernelForge ships inside this distribution, so the task
    bundle is packaged and there is nothing left to configure -- if this ever
    goes back to skipping, every mori vendor-playbook attempt silently does
    nothing on a stock install.
    """
    calls: list[dict] = []
    _stub_run_loop(monkeypatch, calls)
    # This is the only test here that drives submit() far enough to resolve a
    # gfx target, and that resolver ends in rocminfo. Left to the host, the test
    # passes on a GPU box and fails on a CI runner -- and what it is about is the
    # bundle, not the hardware. Name the target so the answer is the same either way.
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=playbook,
        vendor_playbook_role="dispatch",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")
    output_dir = tmp_path / "forge" / "session2" / "attempt_dispatch"

    result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=output_dir,
        candidate=candidate,
        timeout_s=3600,
    )

    assert result.get("skipped") is not True, result.get("stderr_tail")
    assert calls, "forge-loop was never invoked"
    # The bundle really was copied, from the packaged tree rather than a checkout.
    workspace = output_dir / "worktree"
    assert (workspace / "mori_ep_config.py").is_file()
    assert (workspace / "driver.py").is_file()


def test_submit_vendor_playbook_skips_when_the_bundle_cannot_be_resolved(monkeypatch, tmp_path):
    """An unresolvable bundle is still a fail-soft skip, not an exception.

    The packaged tree makes this unreachable in practice; the branch stays
    because a $KERNELFORGE_PROJECT_ROOT override or a truncated install can
    still produce it, and the caller contract is "write a result, never raise
    past the claim".
    """
    monkeypatch.setattr(forge_submit, "_resolve_vendor_task_bundle", lambda relative: tmp_path / "absent" / relative)

    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=playbook,
        vendor_playbook_role="dispatch",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")

    result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=tmp_path / "forge" / "session2" / "attempt_dispatch",
        candidate=candidate,
        timeout_s=3600,
    )

    assert result["skipped"] is True
    assert result["returncode"] == 2
    assert "task bundle not found" in result["stderr_tail"]


def test_submit_vendor_playbook_writes_result_when_bundle_copy_raises(monkeypatch, tmp_path):
    """A raised exception after claiming must not orphan claimed.lock.

    Regression: the claimed section used to have no blanket exception
    handling -- only a couple of specific call sites (e.g.
    _copy_vendor_task_bundle's OSError) were caught. Anything else raised
    after the claim (a bad env lookup, an unexpected failure resolving the
    branch/gpu target, etc.) propagated straight out of submit() with
    claimed.lock left on disk and no result.json ever written -- every
    subsequent submission for that group (siblings this session, or a
    retry) would then wait the full poll deadline and still find nothing,
    forever. This raises from _new_forge_branch, a call site with no
    dedicated try/except of its own, specifically to exercise the
    catch-all wrapper in _submit_vendor_playbook rather than one of the
    pre-existing specific except clauses.
    """
    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(forge_submit, "_resolve_gpu_target", lambda _candidate: "gfx942")

    def _boom(_output_dir, _source_file):
        raise RuntimeError("simulated unexpected failure resolving the forge branch")

    monkeypatch.setattr(forge_submit, "_new_forge_branch", _boom)

    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=playbook,
        vendor_playbook_role="dispatch",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")
    output_dir = tmp_path / "forge" / "session3" / "attempt_dispatch"

    result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=output_dir,
        candidate=candidate,
        timeout_s=3600,
    )

    # submit() must return a graceful failure, never raise.
    assert result["skipped"] is True
    assert result["returncode"] == 2
    assert "simulated unexpected failure" in result["stderr_tail"]

    lock_dir = forge_submit._vendor_playbook_lock_dir(output_dir, "mori_ep_dispatch_combine")
    assert (lock_dir / "claimed.lock").is_file()
    # The critical assertion: a result.json MUST exist, or the lock is
    # orphaned and no sibling/retry for this group can ever proceed again.
    assert (lock_dir / "result.json").is_file()
    cached = forge_submit._read_vendor_playbook_cached_result(lock_dir)
    assert cached is not None
    assert cached["skipped"] is True

    # A sibling (e.g. combine) submitted right after must get the cached
    # failure back immediately, not hang until the wait deadline.
    combine_candidate = _mori_combine_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=dict(playbook, role="combine"),
        vendor_playbook_role="combine",
    )
    combine_result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=tmp_path / "forge" / "session3" / "attempt_combine",
        candidate=combine_candidate,
        timeout_s=3600,
    )
    assert combine_result["vendor_playbook_reused"] is True
    assert combine_result["skipped"] is True


def test_resolve_kernel_anchor_path_is_always_absolute(monkeypatch, tmp_path):
    """A relative ``source_file`` stand-in is later reinterpreted by
    ``Path(...).resolve()`` against whatever the apply-stage process's CWD
    happens to be, not against the KernelForge bundle it was meant to name
    -- resolve_kernel_anchor_path() must never return a bare relative string,
    whether it resolves against the packaged tree or against an operator's
    $KERNELFORGE_PROJECT_ROOT substitution (PR #1191 review finding #8).
    """
    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    assert playbook is not None

    packaged_anchor = resolve_kernel_anchor_path(playbook)
    assert packaged_anchor
    assert Path(packaged_anchor).is_absolute()
    # With the bundle packaged, the stand-in names a file that actually exists
    # rather than a synthetic /nonexistent-forge-path placeholder.
    assert Path(packaged_anchor).is_file()

    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    overridden_anchor = resolve_kernel_anchor_path(playbook)
    assert overridden_anchor
    assert Path(overridden_anchor).is_absolute()
    assert overridden_anchor.startswith(str(project_root))


def test_submit_vendor_playbook_writes_optimization_report_with_correctness_pass(monkeypatch, tmp_path):
    """The vendor-playbook path reuses one forge-loop run but used to never
    write ``optimization_report.md``, so ``kernel_optimization.py``'s
    correctness extraction (which scans ``cli_workspace``/
    ``optimization_report.md`` for a "[correctness] pass" marker) never
    found a signal and ``make_proposal()`` could never return KEEP even when
    SNR validation had already passed inside forge-loop (PR #1191 review
    finding #5).
    """
    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(forge_submit, "_resolve_gpu_target", lambda _candidate: "gfx942")
    _stub_run_loop(monkeypatch, [])

    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=playbook,
        vendor_playbook_role="dispatch",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")
    output_dir = tmp_path / "forge" / "session_report" / "attempt_dispatch"

    result = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=output_dir,
        candidate=candidate,
        timeout_s=3600,
    )

    assert result["vendor_playbook_independently_counted"] is True
    assert result["cli_workspace"] == str(output_dir)
    report_path = output_dir / "optimization_report.md"
    assert report_path.is_file(), "vendor-playbook path must write optimization_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert "[correctness] pass" in report_text


def test_submit_vendor_playbook_stale_failure_cache_allows_retry(monkeypatch, tmp_path):
    """A cached FAILURE only de-dupes submissions within
    ``_VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S``; once it ages out, a fresh
    submission must actually retry instead of one transient failure
    (an unresolvable task bundle, here) permanently wedging the whole
    playbook group for the rest of the session (PR #1191 review finding #2).

    The transient failure used to be "FORGE_PATH unset", which no longer fails
    at all now that the bundle is packaged; it is injected directly instead.
    """
    monkeypatch.setattr(forge_submit, "_resolve_vendor_task_bundle", lambda relative: tmp_path / "absent" / relative)
    playbook = match_vendor_operator_playbook(_mori_dispatch_candidate())
    candidate = _mori_dispatch_candidate(
        patch_strategy="vendor_playbook",
        vendor_operator_playbook=playbook,
        vendor_playbook_role="dispatch",
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("# fallback prompt\n", encoding="utf-8")
    output_dir = tmp_path / "forge" / "session_ttl" / "attempt_dispatch"

    first = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=output_dir,
        candidate=candidate,
        timeout_s=3600,
    )
    assert first["skipped"] is True  # unresolvable bundle -> a real failure

    lock_dir = forge_submit._vendor_playbook_lock_dir(output_dir, "mori_ep_dispatch_combine")
    result_path = lock_dir / "result.json"
    assert result_path.is_file()

    # Immediately retrying (still within the TTL window) must reuse the
    # cached failure rather than re-running -- this is the intended
    # short-window dedup for a genuinely concurrent dispatch+combine pair.
    second = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=tmp_path / "forge" / "session_ttl" / "attempt_dispatch_retry_fresh",
        candidate=candidate,
        timeout_s=3600,
    )
    assert second["vendor_playbook_reused"] is True

    # Age the cached failure past the TTL by rewinding result.json's mtime
    # (simulates enough wall-clock time passing within the same session).
    stale_mtime = time.time() - forge_submit._VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S - 1.0
    os.utime(result_path, (stale_mtime, stale_mtime))

    project_root = tmp_path / "kernelforge-project"
    _write_fake_mori_bundle(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))
    # Lift the injected failure so the retry can actually resolve a bundle.
    monkeypatch.setattr(forge_submit, "_resolve_vendor_task_bundle", _real_resolve_vendor_task_bundle)
    monkeypatch.setattr(forge_submit, "_resolve_gpu_target", lambda _candidate: "gfx942")
    captured: list[dict] = []
    _stub_run_loop(monkeypatch, captured)

    third = forge_submit.submit(
        source_file=_MORI_SITE_PACKAGES_FILE,
        prompt_file=prompt_file,
        output_dir=tmp_path / "forge" / "session_ttl" / "attempt_dispatch_retry_stale",
        candidate=candidate,
        timeout_s=3600,
    )
    assert third["vendor_playbook_reused"] is False, "a stale failure must not be reused"
    assert third["skipped"] is False
    assert len(captured) == 1, "the retry must actually launch forge-loop"


def test_claim_vendor_playbook_run_steals_a_claim_orphaned_by_a_dead_process(tmp_path):
    """A holder killed by SIGKILL/OOM/node-restart never writes
    ``result.json`` and never releases ``claimed.lock``; every later
    submission used to poll ``_wait_for_vendor_playbook_result`` all the way
    to its deadline (``timeout_s`` + 300s) and still find nothing -- for a
    60-minute-budget attempt, an hour burned per submission. A claim older
    than its own attempt budget (plus grace) must instead be treated as
    abandoned and stolen (PR #1191 review finding #3).
    """
    lock_dir = tmp_path / "vendor_playbook_locks" / "mori_ep_dispatch_combine"
    lock_dir.mkdir(parents=True)
    claim_path = lock_dir / "claimed.lock"
    timeout_s = 3600
    dead_pid_claimed_at = time.time() - timeout_s - forge_submit._VENDOR_PLAYBOOK_CLAIM_STALE_GRACE_S - 1.0
    claim_path.write_text(
        json.dumps({"pid": 999999, "claimed_at": dead_pid_claimed_at, "nonce": "dead"}),
        encoding="utf-8",
    )

    assert forge_submit._claim_is_stale(claim_path, timeout_s) is True
    assert forge_submit._claim_vendor_playbook_run(lock_dir, timeout_s) is True

    stolen = json.loads(claim_path.read_text(encoding="utf-8"))
    assert stolen["pid"] == os.getpid()
    assert stolen["nonce"] != "dead"


def test_claim_vendor_playbook_run_does_not_steal_a_live_claim(tmp_path):
    """A claim well within its holder's own budget must not be stolen out
    from under a genuinely still-running attempt."""
    lock_dir = tmp_path / "vendor_playbook_locks" / "mori_ep_dispatch_combine"
    lock_dir.mkdir(parents=True)
    claim_path = lock_dir / "claimed.lock"
    timeout_s = 3600
    claim_path.write_text(
        json.dumps({"pid": 123, "claimed_at": time.time() - 5.0, "nonce": "alive"}),
        encoding="utf-8",
    )

    assert forge_submit._claim_is_stale(claim_path, timeout_s) is False
    assert forge_submit._claim_vendor_playbook_run(lock_dir, timeout_s) is False
    # The live claim must be left untouched.
    assert json.loads(claim_path.read_text(encoding="utf-8"))["nonce"] == "alive"


def test_registry_json_is_valid_and_ships_in_package_data():
    """The JSON registry parses and pyproject.toml declares it as package-data
    (mirrors KernelForge's own wheel-packaging regression for framework/mori/)."""
    registry_path = _TOOLS_DIR / "vendor_operator_playbooks.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    playbook_ids = {p["id"] for p in data["playbooks"]}
    assert "mori_ep_dispatch_combine" in playbook_ids

    pyproject = registry_path.parents[5] / "pyproject.toml"
    assert pyproject.is_file()
    text = pyproject.read_text(encoding="utf-8")
    assert "vendor_operator_playbooks.json" in text
