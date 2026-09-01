"""Tests for PR Monitor query-context resolution."""

from __future__ import annotations

import os

import pytest

from kernelforge.knowledge.pr_query_context import (
    KERNEL_BACKEND_REPO_MAP,
    PR_REPOS_EXPECTED,
    PR_REPOS_WISHLIST,
    REASON_REPO_UNRESOLVED,
    REASON_REPO_UNTRACKED,
    build_context,
    check_whitelist,
    extract_keywords,
    normalize_file_path,
    normalize_kernel_backend,
    parse_git_remote,
    resolve_repo,
)

TRACKED = PR_REPOS_EXPECTED


def test_normalize_kernel_backend_reduces_a_label_to_its_key():
    assert normalize_kernel_backend("flydsl") == "flydsl"
    assert normalize_kernel_backend("  AITER ") == "aiter"
    assert normalize_kernel_backend("") == ""


@pytest.mark.parametrize(
    "url,expected",
    [
        ("git@github.com:ROCm/aiter.git", "ROCm/aiter"),
        ("https://github.com/ROCm/aiter.git", "ROCm/aiter"),
        ("https://github.com/ROCm/aiter", "ROCm/aiter"),
        ("ssh://git@github.com/sgl-project/sglang.git", "sgl-project/sglang"),
        ("", ""),
        ("not-a-remote", ""),
        ("https://github.com/single", ""),
        ("https://github.com/", ""),
        ("/shared_nfs/local/KernelForge", ""),
    ],
)
def test_parse_git_remote(url, expected):
    assert parse_git_remote(url) == expected


@pytest.mark.parametrize("kernel_backend,repo", sorted(KERNEL_BACKEND_REPO_MAP.items()))
def test_mapped_kernel_backends_resolve_to_a_tracked_repo(kernel_backend, repo):
    assert resolve_repo(kernel_backend=kernel_backend, tracked=TRACKED) == (repo, "")
    assert repo in TRACKED


@pytest.mark.parametrize("kernel_backend", ["ck", "hipblaslt"])
def test_unmapped_kernel_backends_fall_to_repo_unresolved(kernel_backend):
    """Reject approximate repository matches."""
    assert resolve_repo(kernel_backend=kernel_backend, tracked=TRACKED) == ("", REASON_REPO_UNRESOLVED)


def test_git_remote_is_the_second_link_in_the_chain():
    repo, reason = resolve_repo(kernel_backend="ck", git_remote="git@github.com:ROCm/ATOM.git", tracked=TRACKED)
    assert (repo, reason) == ("ROCm/ATOM", "")


def test_kernel_backend_mapping_wins_over_the_git_remote():
    repo, reason = resolve_repo(kernel_backend="aiter", git_remote="git@github.com:ROCm/ATOM.git", tracked=TRACKED)
    assert (repo, reason) == ("ROCm/aiter", "")


def test_fork_falls_back_to_upstream_when_the_fork_is_untracked():
    repo, reason = resolve_repo(git_remote="git@github.com:ROCm/vllm.git", tracked=("vllm-project/vllm",))
    assert (repo, reason) == ("vllm-project/vllm", "")


def test_untracked_repo_does_not_degrade_to_another_repo():
    repo, reason = resolve_repo(git_remote="git@github.com:AMD-AGI/Primus-Turbo.git", tracked=TRACKED)
    assert reason == REASON_REPO_UNTRACKED
    assert repo == "AMD-AGI/Primus-Turbo"
    assert repo not in TRACKED


def test_nothing_to_resolve_from_is_unresolved():
    assert resolve_repo() == ("", REASON_REPO_UNRESOLVED)


def test_unknown_tracked_set_skips_the_whitelist_gate():
    """Skip tracked-repository validation when the set is unavailable."""
    assert resolve_repo(kernel_backend="aiter", tracked=None) == ("ROCm/aiter", "")


def test_absolute_path_becomes_repo_relative():
    workspace = "/work/repo"
    absolute = os.path.join(workspace, "kernels/moe/gemm2.py")

    assert normalize_file_path(absolute, workspace=workspace) == "kernels/moe/gemm2.py"


def test_relative_path_passes_through_unchanged():
    assert normalize_file_path("kernels/moe/gemm2.py") == "kernels/moe/gemm2.py"
    assert normalize_file_path("./kernels/gemm2.py") == "kernels/gemm2.py"


def test_backslashes_are_normalized_to_posix():
    assert normalize_file_path("kernels\\moe\\gemm2.py") == "kernels/moe/gemm2.py"


@pytest.mark.parametrize(
    "raw",
    ["../outside.py", "kernels/../../escape.py", "/etc/passwd", "", "   "],
)
def test_escaping_paths_are_rejected(raw):
    assert normalize_file_path(raw) == ""


def test_absolute_path_outside_the_workspace_is_rejected():
    assert normalize_file_path("/elsewhere/x.py", workspace="/work/repo") == ""


def test_absolute_path_without_a_workspace_is_rejected():
    assert normalize_file_path("/work/repo/x.py") == ""


def test_existence_check_is_applied_when_supplied():
    present = {"kernels/moe/gemm2.py"}

    assert normalize_file_path("kernels/moe/gemm2.py", exists=present.__contains__) == "kernels/moe/gemm2.py"
    assert normalize_file_path("kernels/gone.py", exists=present.__contains__) == ""


def test_keywords_are_short_phrases_never_a_sentence():
    """Emit only short phrases for whole-string matching."""
    keywords = extract_keywords(
        operator_name="mxfp8_grouped_gemm",
        target_functions=["fused_add_rmsnorm"],
        bottleneck="memory bound on vectorized global loads",
    )

    assert keywords
    for phrase in keywords:
        assert 1 <= len(phrase.split()) <= 2


def test_keywords_split_identifiers_and_camel_case():
    keywords = extract_keywords(operator_name="fusedAddRmsNorm")

    assert "fused add" in keywords
    assert "rms" in keywords or "norm" in keywords or "rms norm" in keywords


def test_bigrams_never_splice_across_a_dropped_stopword():
    """Keep bigrams adjacent in the original identifier."""
    keywords = extract_keywords(operator_name="fused_kernel_gemm", limit=10)

    assert "fused gemm" not in keywords
    assert "fused kernel" in keywords
    assert "kernel gemm" in keywords


def test_operator_name_tokens_survive_as_a_phrase():
    keywords = extract_keywords(operator_name="fused_add_rmsnorm", limit=10)

    assert "fused add" in keywords
    assert "add rmsnorm" in keywords


def test_pure_digit_tokens_are_dropped():
    keywords = extract_keywords(operator_name="gemm_128_256", limit=10)

    assert all(not phrase.split()[0].isdigit() for phrase in keywords)
    assert "gemm" in keywords


def test_keywords_drop_generic_terms():
    keywords = extract_keywords(operator_name="kernel_support_test_gemm")

    assert "gemm" in keywords
    assert all("kernel" not in phrase.split() for phrase in keywords)


def test_keywords_are_capped_and_deduplicated():
    keywords = extract_keywords(
        operator_name="moe_gemm",
        target_functions=["moe_gemm", "moe_gemm", "attention_decode"],
        limit=3,
    )

    assert len(keywords) <= 3
    assert len(set(keywords)) == len(keywords)


def test_keywords_are_empty_without_input():
    assert extract_keywords() == ()


def test_whitelist_drift_is_clean_for_the_expected_set():
    payload = [{"repo_name": name, "is_active": True} for name in PR_REPOS_EXPECTED]

    assert check_whitelist(payload).clean


def test_wishlist_absence_never_counts_as_drift():
    """Exclude known-unindexed repositories from drift."""
    payload = [{"repo_name": name, "is_active": True} for name in PR_REPOS_EXPECTED]
    drift = check_whitelist(payload)

    assert drift.missing == ()
    assert not set(PR_REPOS_WISHLIST) & set(drift.missing + drift.unexpected)


def test_whitelist_reports_missing_expected_and_new_repos():
    payload = [{"repo_name": name, "is_active": True} for name in PR_REPOS_EXPECTED if name != "ROCm/ATOM"]
    payload.append({"repo_name": "ROCm/brand-new", "is_active": True})
    drift = check_whitelist(payload)

    assert drift.missing == ("ROCm/ATOM",)
    assert drift.unexpected == ("ROCm/brand-new",)
    assert not drift.clean


def test_whitelist_flags_a_registered_but_inactive_repo():
    payload = [{"repo_name": name, "is_active": name != "ROCm/hip"} for name in PR_REPOS_EXPECTED]

    assert check_whitelist(payload).inactive == ("ROCm/hip",)


def test_wishlist_repo_appearing_later_is_not_flagged_as_unexpected():
    payload = [{"repo_name": name, "is_active": True} for name in PR_REPOS_EXPECTED]
    payload.append({"repo_name": "ROCm/rccl", "is_active": True})

    assert check_whitelist(payload).unexpected == ()


def test_build_context_assembles_repo_paths_and_keywords():
    context = build_context(
        kernel_backend="flydsl",
        tracked=TRACKED,
        source_files=["kernels/moe/mxfp_moe/gemm2.py"],
        operator_name="mxfp_moe_gemm",
    )

    assert context.repo == "ROCm/FlyDSL"
    assert context.file_paths == ("kernels/moe/mxfp_moe/gemm2.py",)
    assert context.keywords
    assert context.usable


def test_build_context_caps_file_paths():
    context = build_context(
        kernel_backend="aiter",
        tracked=TRACKED,
        source_files=[f"a/f{i}.py" for i in range(10)],
    )

    assert len(context.file_paths) == 3


def test_build_context_drops_unusable_paths_but_keeps_the_repo():
    context = build_context(
        kernel_backend="aiter",
        tracked=TRACKED,
        source_files=["../escape.py"],
        operator_name="rmsnorm",
    )

    assert context.repo == "ROCm/aiter"
    assert context.file_paths == ()
    assert context.usable


def test_build_context_without_any_query_source_is_not_usable():
    context = build_context(kernel_backend="aiter", tracked=TRACKED)

    assert context.repo == "ROCm/aiter"
    assert not context.usable


def test_build_context_propagates_untracked_reason():
    context = build_context(
        git_remote="git@github.com:AMD-AGI/Primus-Turbo.git",
        tracked=TRACKED,
        operator_name="mxfp8_grouped_gemm",
    )

    assert context.reason == REASON_REPO_UNTRACKED
    assert not context.usable
