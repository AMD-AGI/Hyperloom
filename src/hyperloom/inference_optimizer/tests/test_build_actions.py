# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the targeted-build data model."""

from __future__ import annotations

from hyperloom.orchestrator.framework.build_actions import (
    FAILURE_CLASSES,
    BuildResult,
    FrameworkRuntime,
    TargetedBuildAction,
    build_novelty_key,
    normalize_failure_class,
)


# ---------------------------------------------------------------------------
# TargetedBuildAction round-trip
# ---------------------------------------------------------------------------

def test_targeted_build_action_round_trip():
    a = TargetedBuildAction(
        gap_id="gap.enablement.hip_kernel_missing",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        repo_url="https://github.com/ROCm/aiter",
        ref="v0.1.0",
        gpu_arch="gfx950",
        max_jobs=8,
        expected_symbols=("fp4_moe_op",),
        expected_artifacts=("module_aiter_core*.so",),
        build_command=("python", "setup.py", "develop"),
        build_budget_sec=2400,
        envs={"AITER_REBUILD": "1"},
        attempt_root="/s/enablement/builds/a1",
    )
    assert TargetedBuildAction.from_state(a.to_state()) == a


def test_targeted_build_action_defaults():
    b = TargetedBuildAction.from_state({"framework": "VLLM", "gap_id": "g"})
    assert b.framework == "vllm"
    assert b.component == "aiter"  # invalid/absent coerced
    assert b.torch_constraint_mode == "constraint_file"
    assert b.build_command == ()
    assert b.max_jobs == 0


def test_targeted_build_action_invalid_component_coerced():
    b = TargetedBuildAction.from_state({"framework": "vllm", "component": "bogus"})
    assert b.component == "aiter"


def test_targeted_build_action_invalid_torch_constraint_mode_coerced():
    """An unknown torch_constraint_mode falls back to the constraint_file default."""
    b = TargetedBuildAction.from_state({"framework": "vllm", "torch_constraint_mode": "bogus_mode"})
    assert b.torch_constraint_mode == "constraint_file"


def test_targeted_build_action_invalid_ints_coerced():
    b = TargetedBuildAction.from_state({"framework": "vllm", "max_jobs": "x", "build_budget_sec": None})
    assert b.max_jobs == 0
    assert b.build_budget_sec == 0


def test_targeted_build_action_from_none():
    b = TargetedBuildAction.from_state(None)
    assert b.gap_id == ""
    assert b.component == "aiter"


# ---------------------------------------------------------------------------
# BuildResult round-trip + failure-class
# ---------------------------------------------------------------------------

def test_build_result_ok_round_trip():
    rt = FrameworkRuntime(bin_path="/b", pythonpath_prefixes=("/pkg",))
    r = BuildResult(
        ok=True,
        attempt_root="/a",
        runtime=rt,
        built_artifacts=("/a/lib.so",),
        installed_versions={"torch": "2.10.0+git8514f05", "arch": "gfx950"},
        build_log_path="/a/build.log",
    )
    r2 = BuildResult.from_state(r.to_state())
    assert r2 == r
    assert r2.runtime.pythonpath_prefixes == ("/pkg",)


def test_build_result_failure_round_trip():
    r = BuildResult(ok=False, failure_class="timeout", failure_summary="ran out of time", error="SIGKILL")
    r2 = BuildResult.from_state(r.to_state())
    assert r2.ok is False
    assert r2.failure_class == "timeout"
    assert r2.failure_summary == "ran out of time"


def test_normalize_failure_class():
    assert normalize_failure_class("compile_error") == "compile_error"
    assert normalize_failure_class("bogus") == "ok"
    assert normalize_failure_class(None) == "ok"
    for fc in FAILURE_CLASSES:
        assert normalize_failure_class(fc) == fc


def test_build_result_from_state_normalizes_bad_failure_class():
    r = BuildResult.from_state({"ok": False, "failure_class": "not_a_real_class"})
    assert r.failure_class == "ok"


# ---------------------------------------------------------------------------
# build_novelty_key — repeat vs novel
# ---------------------------------------------------------------------------

def _action(**kw):
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe")
    base.update(kw)
    return TargetedBuildAction(**base)


def test_novelty_key_repeat_is_equal():
    a = _action(ref="v1", gpu_arch="gfx950", build_command=("make",))
    b = _action(ref="v1", gpu_arch="gfx950", build_command=("make",), reason="differs but irrelevant")
    assert build_novelty_key(a) == build_novelty_key(b)


def test_novelty_key_novel_ref_differs():
    a = _action(ref="v1", gpu_arch="gfx950")
    b = _action(ref="v2", gpu_arch="gfx950")
    assert build_novelty_key(a) != build_novelty_key(b)


def test_novelty_key_novel_arch_and_command_differ():
    a = _action(ref="v1", gpu_arch="gfx942", build_command=("make",))
    b = _action(ref="v1", gpu_arch="gfx950", build_command=("make",))
    c = _action(ref="v1", gpu_arch="gfx942", build_command=("ninja",))
    assert build_novelty_key(a) != build_novelty_key(b)
    assert build_novelty_key(a) != build_novelty_key(c)


def test_novelty_key_novel_repo_and_capability_differ():
    a = _action(repo_url="https://github.com/ROCm/aiter", capability="decode")
    b = _action(repo_url="https://github.com/example/aiter", capability="decode")
    c = _action(repo_url="https://github.com/ROCm/aiter", capability="prefill")
    assert build_novelty_key(a) != build_novelty_key(b)
    assert build_novelty_key(a) != build_novelty_key(c)


# ---------------------------------------------------------------------------
# resolve_build_ref
# ---------------------------------------------------------------------------

from hyperloom.orchestrator.framework.build_actions import resolve_build_ref


def test_resolve_github_pr_url():
    repo, ref, pr_url = resolve_build_ref(
        "https://github.com/ROCm/aiter/pull/42",
        "https://github.com/ROCm/aiter",
    )
    assert repo == "https://github.com/ROCm/aiter"
    assert ref == "PR:42"
    assert pr_url == "https://github.com/ROCm/aiter/pull/42"


def test_resolve_github_pr_url_ignores_default_repo():
    repo, ref, _ = resolve_build_ref(
        "https://github.com/sgl-project/sglang/pull/7",
        "https://github.com/ROCm/aiter",
    )
    assert repo == "https://github.com/sgl-project/sglang"
    assert ref == "PR:7"


def test_resolve_bare_pr_ref_uses_default_repo():
    repo, ref, pr_url = resolve_build_ref("PR:123", "https://github.com/ROCm/aiter")
    assert repo == "https://github.com/ROCm/aiter"
    assert ref == "PR:123"
    assert pr_url == ""


def test_resolve_plain_tag():
    repo, ref, pr_url = resolve_build_ref("v0.4.0", "https://github.com/ROCm/aiter")
    assert repo == "https://github.com/ROCm/aiter"
    assert ref == "v0.4.0"
    assert pr_url == ""


def test_resolve_non_pr_url_skipped():
    repo, ref, pr_url = resolve_build_ref("https://example.com/foo", "https://github.com/ROCm/aiter")
    assert repo == ""
    assert ref == ""
    assert pr_url == ""


def test_resolve_empty_candidate_skipped():
    assert resolve_build_ref("", "https://github.com/ROCm/aiter") == ("", "", "")


# An issue is a discussion thread, not a branch. GitHub publishes
# refs/pull/{n}/head for a PR but nothing checkoutable for an issue, so an
# ``issue:{n}`` string reaching ``git worktree add`` verbatim aborts the build
# with ``fatal: invalid reference``. Observed live: an enablement specialist
# cited upstream vllm issue 41292 as the rationale for a from-source build and
# the build died in ~2 minutes during workspace preparation. Resolution must
# strip the issue number to an empty ref (falling back to tag autoselect) while
# keeping the citation as provenance.


def test_resolve_bare_issue_ref_falls_back_to_autoselect():
    repo, ref, pr_url = resolve_build_ref("issue:41292", "https://github.com/ROCm/aiter")
    assert repo == "https://github.com/ROCm/aiter"
    assert ref == "", "an issue number is not a checkoutable git ref"
    assert pr_url == ""


def test_resolve_plural_issues_ref_also_handled():
    _repo, ref, _pr = resolve_build_ref("issues:41292", "https://github.com/ROCm/aiter")
    assert ref == ""


def test_resolve_github_issue_url_keeps_repo_and_provenance():
    url = "https://github.com/vllm-project/vllm/issues/41292"
    repo, ref, pr_url = resolve_build_ref(url, "https://github.com/ROCm/aiter")
    assert repo == "https://github.com/vllm-project/vllm"
    assert ref == ""
    assert pr_url == url, "the citation is kept for the audit trail"


def test_resolve_pr_still_wins_over_issue_shapes():
    """A PR remains checkoutable; the issue branch must not shadow it."""
    _repo, ref, _pr = resolve_build_ref(
        "https://github.com/vllm-project/vllm/pull/33291",
        "https://github.com/ROCm/aiter",
    )
    assert ref == "PR:33291"


def test_source_pr_url_round_trip():
    a = _action(ref="PR:99", source_pr_url="https://github.com/ROCm/aiter/pull/99")
    a2 = TargetedBuildAction.from_state(a.to_state())
    assert a2.source_pr_url == "https://github.com/ROCm/aiter/pull/99"
    assert a2 == a


def test_source_pr_url_not_in_novelty_key():
    a = _action(ref="v1", gpu_arch="gfx950", source_pr_url="https://github.com/x/y/pull/1")
    b = _action(ref="v1", gpu_arch="gfx950", source_pr_url="")
    assert build_novelty_key(a) == build_novelty_key(b)
