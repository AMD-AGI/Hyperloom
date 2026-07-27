"""Tests for operation-centric KernelForge knowledge-base identities."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))
import _bypass_report as bypass_report  # noqa: E402
import _invocation_spec as invocation_spec  # noqa: E402
from _kernel_kb_identity import build_kernel_identities  # noqa: E402


def _aiter_ck_candidate(
    *,
    operation: str = "aiter::gemm_a8w8_blockscale",
) -> dict:
    return {
        "name": "gemm_a8w8_blockscale",
        "kernel_kind": "aiter_ck",
        "source_type": "hip_cpp",
        "source_file": "/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm.cu",
        "kernel_repo": "/sgl-workspace/aiter",
        "input_shapes": [{"shape": "(64,7168) fp8"}],
        "task_group": {
            "task_group_id": "tg001",
            "operator_identity": {
                "version": 2,
                "source_kind": "native",
                "source_path": "/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm.cu",
                "operation": operation,
                "function": "ck::kernel_gemm_xdl_cshuffle_v3",
            },
        },
    }


def test_builds_aiter_ck_a8w8_identity():
    identities = build_kernel_identities(_aiter_ck_candidate())

    assert identities == [
        {
            "identity_version": 1,
            "kernel_project": "aiter",
            "implementation": "ck",
            "source_kind": "native",
            "operation": "gemm_a8w8_blockscale",
            "identity_digest": "5129be65a99f",
            "kernel_page_slug": (
                "kernelforge-exp/kernels/aiter/ck/"
                "gemm_a8w8_blockscale--5129be65a99f"
            ),
            "confidence": "high",
        }
    ]


def test_same_source_with_different_operations_has_different_page_slug():
    first = build_kernel_identities(
        _aiter_ck_candidate(operation="aiter::gemm_a8w8_blockscale")
    )[0]
    second = build_kernel_identities(
        _aiter_ck_candidate(operation="aiter::gemm_a8w8_bpreshuffle")
    )[0]

    assert first["kernel_page_slug"] != second["kernel_page_slug"]
    assert first["identity_digest"] != second["identity_digest"]


def test_shape_changes_do_not_change_page_slug():
    first_candidate = _aiter_ck_candidate()
    second_candidate = copy.deepcopy(first_candidate)
    second_candidate["input_shapes"] = [{"shape": "(4096,7168) fp8"}]

    first = build_kernel_identities(first_candidate)[0]
    second = build_kernel_identities(second_candidate)[0]

    assert first["kernel_page_slug"] == second["kernel_page_slug"]


def test_absolute_path_changes_do_not_change_page_slug():
    first_candidate = _aiter_ck_candidate()
    second_candidate = copy.deepcopy(first_candidate)
    second_candidate["source_file"] = "/opt/relocated/package/kernel.cu"
    second_candidate["kernel_repo"] = "/opt/relocated/package"
    second_candidate["task_group"]["operator_identity"]["source_path"] = (
        "/opt/relocated/package/kernel.cu"
    )

    first = build_kernel_identities(first_candidate)[0]
    second = build_kernel_identities(second_candidate)[0]

    assert first["kernel_page_slug"] == second["kernel_page_slug"]


def test_repo_metadata_and_source_type_taxonomy_are_deterministic_secondary_evidence():
    candidate = _aiter_ck_candidate()
    candidate.update(
        kernel_kind="triton",
        source_type="triton",
        repo_url="https://github.com/ROCm/aiter.git",
    )
    candidate["task_group"]["operator_identity"].update(
        source_kind="py",
        source_path="/tmp/relocated/gemm.py",
    )

    identity = build_kernel_identities(candidate)[0]

    assert identity["kernel_project"] == "aiter"
    assert identity["implementation"] == "triton"
    assert identity["source_kind"] == "py"


def test_project_alias_matches_kernel_kind_and_repo_metadata():
    kernel_kind_candidate = _aiter_ck_candidate()
    repo_metadata_candidate = _aiter_ck_candidate()
    repo_metadata_candidate.update(
        kernel_kind="ck",
        kernel_repo="/usr/local/lib/python3.12/dist-packages/aiter_meta",
    )

    kind_identity = build_kernel_identities(kernel_kind_candidate)[0]
    repo_metadata_identity = build_kernel_identities(repo_metadata_candidate)[0]

    assert kind_identity["kernel_project"] == "aiter"
    assert repo_metadata_identity["kernel_project"] == "aiter"
    assert (
        kind_identity["kernel_page_slug"]
        == repo_metadata_identity["kernel_page_slug"]
    )


def test_matching_provider_namespace_is_removed_without_losing_suffixes():
    namespaced = build_kernel_identities(
        _aiter_ck_candidate(operation="aiter::gemm_a8w8_blockscale_ck_stage2")
    )[0]
    dotted = build_kernel_identities(
        _aiter_ck_candidate(operation="aiter.gemm_a8w8_blockscale_ck_stage2")
    )[0]
    unqualified = build_kernel_identities(
        _aiter_ck_candidate(operation="gemm_a8w8_blockscale_ck_stage2")
    )[0]
    other_provider = build_kernel_identities(
        _aiter_ck_candidate(operation="vllm::gemm_a8w8_blockscale_ck_stage2")
    )[0]

    assert namespaced["operation"] == "gemm_a8w8_blockscale_ck_stage2"
    assert namespaced["kernel_page_slug"] == unqualified["kernel_page_slug"]
    assert dotted["kernel_page_slug"] == unqualified["kernel_page_slug"]
    assert other_provider["operation"] == "vllm::gemm_a8w8_blockscale_ck_stage2"
    assert other_provider["kernel_page_slug"] != namespaced["kernel_page_slug"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda candidate: candidate["task_group"]["operator_identity"].update(
            operation="unknown"
        ),
        lambda candidate: candidate["task_group"]["operator_identity"].update(
            operation="(unlinked)"
        ),
        lambda candidate: candidate["task_group"]["operator_identity"].update(
            source_kind="unknown"
        ),
        lambda candidate: candidate.update(kernel_kind="unknown", source_type="unknown"),
        lambda candidate: (
            candidate.update(
                kernel_kind="ck",
                kernel_repo="/tmp/repo",
                source_file="/tmp/kernel.cu",
            ),
            candidate["task_group"]["operator_identity"].update(
                source_path="/tmp/kernel.cu"
            ),
        ),
        lambda candidate: candidate.update(
            kernel_kind="",
            source_type="hip_cpp",
        ),
    ],
)
def test_unknown_or_unlinked_fields_do_not_create_shared_identity(mutate):
    candidate = _aiter_ck_candidate()
    mutate(candidate)

    assert build_kernel_identities(candidate) == []


def test_digest_uses_only_canonical_identity_fields():
    candidate = _aiter_ck_candidate()
    identity = build_kernel_identities(candidate)[0]
    canonical_fields = {
        key: identity[key]
        for key in (
            "identity_version",
            "kernel_project",
            "implementation",
            "source_kind",
            "operation",
        )
    }
    canonical = json.dumps(
        canonical_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    assert identity["identity_digest"] == hashlib.sha256(
        canonical.encode("ascii")
    ).hexdigest()[:12]

    candidate["source_file"] = "/different/source.cu"
    candidate["task_group"]["operator_identity"]["function"] = "different_function"
    candidate["input_shapes"] = [{"shape": "(1,1) fp8"}]
    assert (
        build_kernel_identities(candidate)[0]["identity_digest"]
        == identity["identity_digest"]
    )


def test_invocation_spec_keeps_operator_identity_and_empty_kb_contract():
    candidate = _aiter_ck_candidate()
    spec = invocation_spec.build_invocation_spec(candidate)

    assert (
        spec["workload"]["task_group"]["operator_identity"]
        == candidate["task_group"]["operator_identity"]
    )
    assert spec["kb"]["kernel_identities"] == build_kernel_identities(candidate)

    candidate["task_group"]["operator_identity"]["source_kind"] = "unlinked"
    unknown_spec = invocation_spec.build_invocation_spec(candidate)
    assert unknown_spec["kb"] == {"kernel_identities": []}


def test_skill_and_bypass_routes_build_the_same_kernel_identity(monkeypatch):
    operation = "gemm_a8w8_blockscale_ck"
    skill_candidate = _aiter_ck_candidate(operation=operation)
    monkeypatch.setattr(
        bypass_report,
        "resolve_source_metadata",
        lambda *_args, **_kwargs: {
            "source_file": "/usr/local/lib/python3.12/dist-packages/"
            "aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm.cu",
            "method": "op_to_source",
            "kernel_kind": "aiter_ck",
        },
    )
    bypass_output = bypass_report.build_candidates(
        {
            "kernels": [
                {
                    "name": "aiter_ck_kernel_gemm_xdl_cshuffle",
                    "op_name": f"aiter::{operation}",
                    "gpu_time_us": 100.0,
                    "gpu_pct": 100.0,
                    "count": 1,
                }
            ]
        },
        framework="vllm",
        target_platform="MI300X",
    )
    bypass_candidate = bypass_output["hot_kernels"][0]

    assert bypass_candidate["kernel_kind"] == "aiter_ck"
    assert build_kernel_identities(bypass_candidate) == build_kernel_identities(
        skill_candidate
    )


def test_trace_aiter_triton_identity_without_benchmark_discovery(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "aiter"
    source = repo / "aiter" / "ops" / "triton" / "silu.py"
    source.parent.mkdir(parents=True)
    source.write_text("def silu(x):\n    return x\n", encoding="utf-8")
    (repo / "op_tests").mkdir()
    monkeypatch.setattr(
        bypass_report,
        "editable_trace_source",
        lambda path, _kind: path,
    )

    output = bypass_report.build_candidates(
        {
            "kernels": [
                {
                    "name": "triton_silu_kernel",
                    "op_name": "aiter::silu",
                    "op_kernel_file": str(source),
                    "op_kernel_backend": "triton",
                    "gpu_time_us": 100.0,
                    "gpu_pct": 100.0,
                    "count": 1,
                }
            ]
        },
        framework="vllm",
        target_platform="MI300X",
        discover_benchmarks=False,
    )
    candidate = output["hot_kernels"][0]
    identities = build_kernel_identities(candidate)

    assert candidate["kernel_kind"] == "triton"
    assert candidate["kernel_repo"] == str(repo.resolve())
    assert candidate["benchmark_files"] == []
    assert identities[0]["kernel_project"] == "aiter"
    assert identities[0]["implementation"] == "triton"
    assert identities[0]["confidence"] == "high"
