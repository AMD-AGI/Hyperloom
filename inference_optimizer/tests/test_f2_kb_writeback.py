"""F2-5 — KB writeback for framework-PR provenance.

Covers two surfaces:

* The :func:`kb_writeback.write_framework_pr_record` adapter writes a
  canonical JSONL record under ``KB_ROOT / lessons.jsonl``.
* The :class:`IntegratePatchExecutor` hook detects a framework-PR
  provenance on the specialist's done payload and routes through the
  adapter — but stays a strict no-op for kernel / legacy proposals.

Reference: ``plan_roofline_framework/F2_framework_agent.MD`` §F2-5.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kb_writeback
from inference_optimizer.orchestrator.action_executors.integrate_patch import (
    IntegratePatchExecutor,
)


# ---------------------------------------------------------------------------
# kb_writeback adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_framework_pr_record_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)

    written = await kb_writeback.write_framework_pr_record(
        pr_url="https://github.com/sgl-project/sglang/pull/1234",
        pr_sha="abc123",
        patch_path="patches/fa-1234.patch",
        outcome=kb_writeback.OUTCOME_INTEGRATED,
        tps_delta_pct=4.5,
        session_id="20260524T140800Z",
    )
    assert written == tmp_path / kb_writeback.LESSONS_FILE
    assert written.exists()
    record = json.loads(written.read_text(encoding="utf-8").strip())
    assert record["pr_url"] == "https://github.com/sgl-project/sglang/pull/1234"
    assert record["pr_sha"] == "abc123"
    assert record["patch_path"] == "patches/fa-1234.patch"
    assert record["outcome"] == "integrated"
    assert record["tps_delta_pct"] == pytest.approx(4.5)
    assert record["session_id"] == "20260524T140800Z"
    assert "ts" in record


@pytest.mark.asyncio
async def test_write_framework_pr_record_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    for sha, outcome in (
        ("aaa", kb_writeback.OUTCOME_INTEGRATED),
        ("bbb", kb_writeback.OUTCOME_REVERTED_SMOKE_FAIL),
    ):
        await kb_writeback.write_framework_pr_record(
            pr_url=f"https://github.com/x/y/pull/{sha}",
            pr_sha=sha,
            patch_path=f"patches/{sha}.patch",
            outcome=outcome,
            tps_delta_pct=1.0,
            session_id="s",
        )
    lines = (tmp_path / kb_writeback.LESSONS_FILE).read_text().splitlines()
    assert len(lines) == 2
    rec0, rec1 = (json.loads(line) for line in lines)
    assert rec0["pr_sha"] == "aaa"
    assert rec0["outcome"] == "integrated"
    assert rec1["pr_sha"] == "bbb"
    assert rec1["outcome"] == "reverted_smoke_fail"


@pytest.mark.asyncio
async def test_write_framework_pr_record_rejects_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outcome"):
        await kb_writeback.write_framework_pr_record(
            pr_url="x", pr_sha="y", patch_path="z",
            outcome="some_made_up_outcome", tps_delta_pct=0.0,
            session_id="s",
        )


def test_kb_root_env_override(monkeypatch: pytest.MonkeyPatch):
    """``INFERENCE_OPTIMIZER_FA_KB_PATH`` redirects KB_ROOT for shared
    KB mounts across many sessions."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", "/tmp/fa_kb_test")
    from inference_optimizer.orchestrator.kb_writeback import _default_kb_root
    assert _default_kb_root() == Path("/tmp/fa_kb_test/framework_optimization")


# ---------------------------------------------------------------------------
# IntegratePatchExecutor hook — _maybe_write_framework_pr_kb_record
# ---------------------------------------------------------------------------


def _executor() -> IntegratePatchExecutor:
    return IntegratePatchExecutor(session_dir=Path("/tmp"))


def test_find_framework_pr_proposal_picks_first_match():
    payload = {
        "proposal_set": [
            {"provenance": "specialist:kernel:rocm", "name": "p0"},
            {
                "provenance": "specialist:serving:framework_pr",
                "name": "p1",
                "fa_pr_url": "u",
                "fa_pr_sha": "s",
            },
            {
                "provenance": "specialist:serving:framework_pr",
                "name": "p2",
            },
        ],
    }
    found = IntegratePatchExecutor._find_framework_pr_proposal(payload)
    assert found is not None and found["name"] == "p1"


def test_find_framework_pr_proposal_none_when_no_match():
    assert IntegratePatchExecutor._find_framework_pr_proposal(
        {"proposal_set": [{"provenance": "specialist:kernel:rocm"}]},
    ) is None
    assert IntegratePatchExecutor._find_framework_pr_proposal({}) is None
    assert IntegratePatchExecutor._find_framework_pr_proposal(None) is None


@pytest.mark.asyncio
async def test_executor_hook_writes_on_integrated_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    done_payload = {
        "proposal_set": [{
            "provenance": "specialist:serving:framework_pr",
            "fa_pr_url": "https://github.com/sgl-project/sglang/pull/9",
            "fa_pr_sha": "deadbeef",
            "patches_written": ["patches/fa-9.patch"],
        }],
    }
    await _executor()._maybe_write_framework_pr_kb_record(
        done_payload=done_payload,
        outcome="integrated",
        tps_delta_pct=3.7,
        extra={},
    )
    record = json.loads(
        (tmp_path / kb_writeback.LESSONS_FILE).read_text().strip()
    )
    assert record["pr_sha"] == "deadbeef"
    assert record["outcome"] == "integrated"
    assert record["tps_delta_pct"] == pytest.approx(3.7)
    assert record["patch_path"] == "patches/fa-9.patch"


@pytest.mark.asyncio
async def test_executor_hook_noop_for_kernel_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    done_payload = {
        "proposal_set": [{
            "provenance": "specialist:kernel:rocm",
            "patches_written": ["patches/k1.patch"],
        }],
    }
    await _executor()._maybe_write_framework_pr_kb_record(
        done_payload=done_payload,
        outcome="integrated",
        tps_delta_pct=3.7,
        extra={},
    )
    assert not (tmp_path / kb_writeback.LESSONS_FILE).exists()


@pytest.mark.asyncio
async def test_executor_hook_skips_proposal_missing_both_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A framework_pr proposal lacking both fa_pr_url AND fa_pr_sha
    is useless for cross-session dedup; the hook logs + skips."""
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    done_payload = {
        "proposal_set": [{
            "provenance": "specialist:serving:framework_pr",
            "patches_written": ["patches/missing.patch"],
        }],
    }
    await _executor()._maybe_write_framework_pr_kb_record(
        done_payload=done_payload,
        outcome="integrated",
        tps_delta_pct=2.0,
        extra={},
    )
    assert not (tmp_path / kb_writeback.LESSONS_FILE).exists()


@pytest.mark.asyncio
async def test_executor_hook_writes_on_revert_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    done_payload = {
        "proposal_set": [{
            "provenance": "specialist:serving:framework_pr",
            "fa_pr_url": "u",
            "fa_pr_sha": "s",
            "patches_written": ["p"],
        }],
    }
    await _executor()._maybe_write_framework_pr_kb_record(
        done_payload=done_payload,
        outcome="reverted_smoke_fail",
        tps_delta_pct=-1.5,
        extra={},
    )
    record = json.loads(
        (tmp_path / kb_writeback.LESSONS_FILE).read_text().strip()
    )
    assert record["outcome"] == "reverted_smoke_fail"
    assert record["tps_delta_pct"] == pytest.approx(-1.5)
