"""Unit tests for :class:`FrameworkPRExecutor` (the new fa-as-arm path).

These tests exercise the executor's selection / KEEP / DISCARD / rollback
branches with the heavy dependencies (``fa candidates`` subprocess,
``apply_to_sglang`` git checkout, Magpie sub-baseline) stubbed via
monkeypatch. They are pure-Python, deterministic, and run in <1s.

Integration with a real fa binary + a real sglang worktree is covered by
``test_framework_pr_discover.py`` (binary smoke) and the
``scripts/run_iofa_e2e.sh`` end-to-end harness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors._framework_gap_composer import (
    compose_gap,
)
from inference_optimizer.orchestrator.action_executors.framework_pr import (
    FrameworkPRExecutor,
    _pick_candidate,
)
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    task: Task
    lease: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def _ctx(session_dir: Path, params: dict[str, Any], task_id: str = "t-fpr-1") -> _Ctx:
    workspace = session_dir / "runs" / "framework_pr" / task_id
    return _Ctx(
        task=Task(
            task_id=task_id,
            kind="framework_pr",
            params=dict(params),
            requires_lanes=(),
            state="running",
            idempotency_key=task_id,
        ),
        extra={"workspace": str(workspace)},
    )


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "sess"
    sd.mkdir()
    return sd


@pytest.fixture
def fake_executor(session_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a FrameworkPRExecutor with the BaselineExecutor stubbed.

    Returns ``(executor, fake_state)`` where ``fake_state`` is a dict the
    individual tests pre-populate with the synthetic behaviour they want
    (candidates list, sub-baseline result, head-sha evolution, apply
    failure, rollback failure, ...). Every stub reads from this dict so
    each test can keep its setup local.
    """
    state: dict[str, Any] = {
        "candidates": [],
        "sub_baseline_result": {"status": "succeeded", "output_throughput": 1.0},
        "head_sha_seq": ["headA"],
        "apply_raises": None,
        "rollback_raises": None,
        "apply_calls": [],
        "rollback_calls": [],
    }

    async def _fake_sub_baseline(self, *, ctx, workspace, params):
        return dict(state["sub_baseline_result"])

    def _fake_enumerate(**kwargs):
        from inference_optimizer.orchestrator.framework_pr_discover import (
            FrameworkPRError,
        )
        if state.get("enumerate_raises"):
            raise FrameworkPRError(str(state["enumerate_raises"]))
        return list(state["candidates"])

    def _fake_current_head(sglang_path=None):
        seq = state["head_sha_seq"]
        if not seq:
            return ""
        return seq[0] if len(seq) == 1 else seq.pop(0)

    def _fake_apply(head_sha, *, pr_number, sglang_path=None,
                    pip_reinstall=False, auto_stash=True):
        state["apply_calls"].append({"head_sha": head_sha, "pr_number": pr_number})
        if state.get("apply_raises"):
            from inference_optimizer.orchestrator.framework_pr_discover import (
                FrameworkPRError,
            )
            raise FrameworkPRError(str(state["apply_raises"]))

    def _fake_rollback(target_sha, *, sglang_path=None):
        state["rollback_calls"].append({"target_sha": target_sha})
        if state.get("rollback_raises"):
            from inference_optimizer.orchestrator.framework_pr_discover import (
                FrameworkPRError,
            )
            raise FrameworkPRError(str(state["rollback_raises"]))

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors.framework_pr"
        ".enumerate_candidates_via_fa",
        _fake_enumerate,
    )
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors.framework_pr"
        ".current_head_sha",
        _fake_current_head,
    )
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors.framework_pr"
        ".apply_to_sglang",
        _fake_apply,
    )
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors.framework_pr"
        ".rollback_to",
        _fake_rollback,
    )
    monkeypatch.setattr(
        FrameworkPRExecutor,
        "_run_sub_baseline",
        _fake_sub_baseline,
    )

    executor = FrameworkPRExecutor(session_dir=session_dir)
    return executor, state


def _base_params(**overrides) -> dict[str, Any]:
    params = {
        "base_tput": 1000.0,
        "min_gain_pct": 1.0,
        "max_candidates": 5,
        "framework": "sglang",
        "gpu_type": "mi300x",
        "model_class": "moe_mla",
        "precision": "fp8",
    }
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# _pick_candidate (pure)
# ---------------------------------------------------------------------------


def test_pick_candidate_skips_non_pr_refs():
    cands = [
        {"ref": "v0.5.0", "head_sha": "abc"},
        {"ref": "PR:42", "head_sha": "deadbeef"},
    ]
    cand, reason = _pick_candidate(cands, current_head_sha="other")
    assert cand is not None
    assert cand["ref"] == "PR:42"
    assert reason == ""


def test_pick_candidate_skips_already_applied():
    cands = [
        {"ref": "PR:1", "head_sha": "AAAA"},
        {"ref": "PR:2", "head_sha": "BBBB"},
    ]
    cand, _ = _pick_candidate(cands, current_head_sha="aaaa")
    assert cand is not None and cand["ref"] == "PR:2"


def test_pick_candidate_all_filtered_returns_reason():
    cands = [
        {"ref": "v0.5", "head_sha": "x"},
        {"ref": "PR:1", "head_sha": "headA"},
    ]
    cand, reason = _pick_candidate(cands, current_head_sha="headA")
    assert cand is None
    assert "head_eq_current" in reason
    assert "non_pr_ref" in reason


# ---------------------------------------------------------------------------
# Executor branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keep_path_lifts_output_throughput(fake_executor, session_dir):
    executor, state = fake_executor
    state["candidates"] = [
        {"ref": "PR:123", "head_sha": "newhead", "score": 0.9, "title": "fast attn"},
    ]
    state["head_sha_seq"] = ["headA", "newhead"]  # before apply, after apply
    state["sub_baseline_result"] = {
        "status": "succeeded",
        "output_throughput": 1100.0,
        "accuracy": 0.83,
    }

    result = await executor(_ctx(session_dir, _base_params(min_gain_pct=1.0)))

    assert result["status"] == "succeeded"
    assert result["decision"] == "kept"
    assert result["applied_ref"] == "PR:123"
    assert result["output_throughput"] == pytest.approx(1100.0)
    assert result["delta_pct"] == pytest.approx(10.0)
    assert result["accuracy"] == pytest.approx(0.83)
    assert state["apply_calls"] == [{"head_sha": "newhead", "pr_number": 123}]
    assert state["rollback_calls"] == []
    # selected_score is forwarded from fa candidate for downstream visibility.
    assert result["selected_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_discard_path_rolls_back(fake_executor, session_dir):
    executor, state = fake_executor
    state["candidates"] = [{"ref": "PR:7", "head_sha": "newhead"}]
    state["head_sha_seq"] = ["headA", "newhead"]
    state["sub_baseline_result"] = {
        "status": "succeeded",
        "output_throughput": 1005.0,  # +0.5% < default min_gain_pct=1.0
    }

    result = await executor(_ctx(session_dir, _base_params(min_gain_pct=1.0)))

    assert result["status"] == "succeeded"
    assert result["decision"] == "discarded"
    assert result["applied_ref"] == "PR:7"
    assert "output_throughput" not in result  # NOT lifted -> no current_best update
    assert result["rollback_done"] is True
    assert state["rollback_calls"] == [{"target_sha": "headA"}]


@pytest.mark.asyncio
async def test_no_base_tput_fails_fast(fake_executor, session_dir):
    executor, _state = fake_executor
    params = _base_params(base_tput=0.0)
    result = await executor(_ctx(session_dir, params))
    assert result["status"] == "failed"
    assert result["error_class"] == "no_base_tput"


@pytest.mark.asyncio
async def test_unsupported_framework_fails_fast(fake_executor, session_dir):
    executor, _state = fake_executor
    params = _base_params(framework="vllm")
    result = await executor(_ctx(session_dir, params))
    assert result["status"] == "failed"
    assert result["error_class"] == "unsupported_framework"


@pytest.mark.asyncio
async def test_no_applicable_candidate_when_all_already_applied(
    fake_executor, session_dir,
):
    executor, state = fake_executor
    state["candidates"] = [{"ref": "PR:1", "head_sha": "headA"}]
    state["head_sha_seq"] = ["headA"]  # current HEAD == only candidate
    result = await executor(_ctx(session_dir, _base_params()))
    assert result["status"] == "failed"
    assert result["error_class"] == "no_applicable_candidate"
    assert state["apply_calls"] == []  # nothing applied


@pytest.mark.asyncio
async def test_apply_failure_does_not_attempt_rollback(
    fake_executor, session_dir,
):
    executor, state = fake_executor
    state["candidates"] = [{"ref": "PR:99", "head_sha": "newhead"}]
    state["head_sha_seq"] = ["headA"]  # current HEAD probed once, then apply raises
    state["apply_raises"] = "git fetch failed"
    result = await executor(_ctx(session_dir, _base_params()))
    assert result["status"] == "failed"
    assert result["error_class"] == "apply_failed"
    # No rollback: apply never mutated the worktree (git failed before checkout).
    assert state["rollback_calls"] == []


@pytest.mark.asyncio
async def test_sub_baseline_failure_triggers_rollback(
    fake_executor, session_dir,
):
    executor, state = fake_executor
    state["candidates"] = [{"ref": "PR:9", "head_sha": "newhead"}]
    state["head_sha_seq"] = ["headA", "newhead"]
    state["sub_baseline_result"] = {
        "status": "failed",
        "error_class": "subprocess_nonzero",
        "error": "server crashed",
    }
    result = await executor(_ctx(session_dir, _base_params()))
    assert result["status"] == "failed"
    assert result["error_class"] == "subprocess_nonzero"
    assert result["rollback_done"] is True
    assert state["rollback_calls"] == [{"target_sha": "headA"}]


@pytest.mark.asyncio
async def test_fa_candidates_failure_propagates_error_class(
    fake_executor, session_dir,
):
    executor, state = fake_executor
    state["enumerate_raises"] = "primus 503"
    result = await executor(_ctx(session_dir, _base_params()))
    assert result["status"] == "failed"
    assert result["error_class"] == "fa_candidates_failed"
    assert "primus 503" in result["error"]
    assert state["apply_calls"] == []  # nothing applied on enumeration failure


@pytest.mark.asyncio
async def test_dry_run_skips_apply_and_bench(fake_executor, session_dir):
    executor, state = fake_executor
    state["candidates"] = [
        {"ref": "PR:5", "head_sha": "X", "score": 0.42, "title": "fp8 moe"},
    ]
    state["head_sha_seq"] = ["headA"]
    result = await executor(_ctx(session_dir, _base_params(dry_run=True)))
    assert result["status"] == "succeeded"
    assert result["decision"] == "dry_run"
    assert result["selected_ref"] == "PR:5"
    assert result["applied_ref"] == ""
    assert state["apply_calls"] == []
    assert state["rollback_calls"] == []


@pytest.mark.asyncio
async def test_rollback_failure_surfaces_in_result(fake_executor, session_dir):
    executor, state = fake_executor
    state["candidates"] = [{"ref": "PR:7", "head_sha": "newhead"}]
    state["head_sha_seq"] = ["headA", "newhead"]
    state["sub_baseline_result"] = {
        "status": "succeeded",
        "output_throughput": 1005.0,
    }
    state["rollback_raises"] = "git checkout aborted"
    result = await executor(_ctx(session_dir, _base_params(min_gain_pct=1.0)))
    assert result["status"] == "succeeded"
    assert result["decision"] == "discarded"
    assert result["rollback_done"] is False
    assert "git checkout aborted" in result["rollback_error"]


@pytest.mark.asyncio
async def test_keyword_override_replaces_composed_keywords(
    fake_executor, session_dir,
):
    """Operator-supplied keyword_override must reach enumerate_candidates_via_fa."""
    executor, state = fake_executor
    state["candidates"] = []  # short-circuits via no_applicable_candidate later
    state["head_sha_seq"] = ["headA"]
    captured: dict[str, Any] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return []

    import inference_optimizer.orchestrator.action_executors.framework_pr as fpr_mod
    fpr_mod.enumerate_candidates_via_fa = _capture  # type: ignore[assignment]

    params = _base_params(keyword_override=["mla", "fp8"])
    result = await executor(_ctx(session_dir, params))

    assert captured["keywords"] == ["mla", "fp8"]
    # No PR returned by fa → no_applicable_candidate failure path.
    assert result["status"] == "failed"
    assert result["error_class"] == "no_applicable_candidate"


# ---------------------------------------------------------------------------
# compose_gap (pure)
# ---------------------------------------------------------------------------


def test_compose_gap_dense_no_profile():
    gap, kw = compose_gap(
        framework="sglang",
        gpu_type="mi300x",
        model_class="dense",
        precision="bf16",
    )
    assert "sglang" in gap and "bf16" in gap and "dense" in gap and "mi300x" in gap
    assert kw == sorted(["sglang", "mi300x", "dense", "bf16"])


def test_compose_gap_moe_class_normalises_to_moe():
    gap, kw = compose_gap(
        framework="sglang", model_class="moe_mla", precision="fp8",
    )
    assert "moe" in gap and "moe_mla" not in gap
    assert "moe" in kw and "moe_mla" not in kw


def test_compose_gap_bottleneck_from_breakdown(tmp_path):
    import json
    bp = tmp_path / "kernels.json"
    bp.write_text(json.dumps({
        "top_kernels": [
            {"name": "ck_attention_forward_kernel"},
            {"name": "flash_attn_v2"},
        ],
    }))
    gap, kw = compose_gap(
        framework="sglang", model_class="moe_mla",
        profile_kernel_breakdown_path=bp,
    )
    assert "attention" in gap
    assert "attention" in kw


def test_compose_gap_handles_missing_breakdown_silently():
    gap, kw = compose_gap(
        framework="sglang", model_class="dense",
        profile_kernel_breakdown_path="/no/such/file.json",
    )
    # Falls back to no-bottleneck composition; still produces non-empty output.
    assert gap
    assert "dense" in kw


def test_compose_gap_empty_inputs_returns_minimal_gap():
    gap, kw = compose_gap()
    assert gap == "improve throughput"
    assert kw == []
