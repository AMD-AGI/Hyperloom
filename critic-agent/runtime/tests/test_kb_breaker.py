"""Circuit-breaker behaviour for KB unreachability.

These tests cover the contract:

* When the KB transport fails, ``KBWriter.list_priors`` short-circuits
  on subsequent calls within the cooldown window, returning an empty
  ``priors`` list with ``cache="kb_unreachable"`` and never raising.
* Writes (``write_verdict`` / ``write_kb_drafts`` / ``add_contradiction``)
  honour the open breaker and refuse to make remote calls.
* A successful KB call resets the breaker.
* ``DecisionReviewer.prepare_review`` reflects the breaker state in
  ``judge_bundle.kb_read_skipped_reason`` so the SKILL knows priors are
  missing because of an outage rather than a clean miss.
"""

from __future__ import annotations

import pytest

from runtime.dead_letter import DeadLetter
from runtime.decision_reviewer import DecisionReviewer
from runtime.errors import KBTransportError, KBValidationError
from runtime.in_memory_kb_client import InMemoryKBClient
from runtime.kb_writer import KBWriter, WriteContext
from runtime.session_memory import SessionMemory


class _FlakyKBClient(InMemoryKBClient):
    """InMemoryKBClient with deterministic transport failure injection."""

    def __init__(self):
        super().__init__()
        self._fail_count: dict[str, int] = {}

    def fail_next(self, endpoint: str, times: int = 1) -> None:
        self._fail_count[endpoint] = self._fail_count.get(endpoint, 0) + times

    def _consume(self, endpoint: str) -> None:
        remaining = self._fail_count.get(endpoint, 0)
        if remaining > 0:
            self._fail_count[endpoint] = remaining - 1
            raise KBTransportError(f"{endpoint}: simulated transport error")

    def list(self, **kwargs):
        self._consume("list")
        return super().list(**kwargs)

    def upsert(self, payload):
        self._consume("upsert")
        return super().upsert(payload)

    def batch_insert(self, items, *, on_conflict="upsert"):
        self._consume("batch_insert")
        return super().batch_insert(items, on_conflict=on_conflict)

    def add_edges(self, edges):
        self._consume("edges/add")
        return super().add_edges(edges)


@pytest.fixture()
def writer(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    dlq = DeadLetter(root=tmp_path / "dlq")
    kb = _FlakyKBClient()
    w = KBWriter(kb, session_memory=sm, dead_letter=dlq)
    return w, kb, sm, dlq


def _scope():
    return {
        "org": "hyperloom",
        "framework": "sglang",
        "model": "deepseek-r1",
        "model_family": "deepseek",
        "workload": "decode",
        "precision": "fp8",
    }


def test_list_priors_short_circuits_after_first_transport_error(writer):
    w, kb, _, _ = writer
    kb.fail_next("list", times=1)
    out1 = w.list_priors(scope=_scope())
    assert out1["cache"] == "kb_unreachable"
    assert out1["priors"] == []
    assert "error" in out1
    assert w.is_kb_unreachable() is True

    # Second call should not even hit the transport because the breaker is
    # open. We assert this by clearing the failure queue: if the call
    # actually went through, list() would succeed and cache would be 'miss'.
    out2 = w.list_priors(scope=_scope())
    assert out2["cache"] == "kb_unreachable"
    assert "error" not in out2  # no transport call made
    assert out2["breaker"]["open"] is True


def test_list_priors_validation_error_does_not_open_breaker(writer):
    w, kb, _, _ = writer

    def boom(**kwargs):
        raise KBValidationError("422 bad scope")

    kb.list = boom  # type: ignore[method-assign]
    out = w.list_priors(scope=_scope())
    assert out["cache"] == "miss"
    assert out["priors"] == []
    assert "error" in out
    assert w.is_kb_unreachable() is False


def test_list_priors_resets_breaker_on_recovery(writer):
    w, kb, _, _ = writer
    kb.fail_next("list", times=1)
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
    # Force the breaker closed manually so we can re-attempt within cooldown.
    w._unreachable_until = 0.0
    w._consecutive_failures = 0
    out = w.list_priors(scope=_scope())
    assert out["cache"] == "miss"
    assert w._consecutive_failures == 0


def test_breaker_threshold_higher_than_one(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_KB_BREAKER_THRESHOLD", "3")
    sm = SessionMemory(root=tmp_path / "sm")
    kb = _FlakyKBClient()
    w = KBWriter(kb, session_memory=sm, dead_letter=DeadLetter(root=tmp_path / "dlq"))
    kb.fail_next("list", times=2)
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
    assert w.is_kb_unreachable() is False  # only 1 failure recorded
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
    assert w.is_kb_unreachable() is False  # 2 failures, still under threshold
    kb.fail_next("list", times=1)
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
    assert w.is_kb_unreachable() is True  # 3rd failure trips it


def test_write_verdict_disabled_when_breaker_open(writer):
    w, _, _, _ = writer
    w.force_kb_unreachable()
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context={
            "framework": "sglang", "model": "deepseek-r1",
            "model_family": "deepseek", "workload": "decode", "precision": "fp8",
        },
        ctx=WriteContext(session_id="s1", review_id="rev"),
    )
    assert res.status == "disabled"
    assert res.detail["reason"] == "kb_unreachable"
    assert res.detail["breaker"]["open"] is True


def test_write_kb_drafts_disabled_when_breaker_open(writer):
    w, _, _, _ = writer
    w.force_kb_unreachable()
    res = w.write_kb_drafts(
        kb_drafts=[{
            "category": "kernel_optimization",
            "action": "patch the active dispatch path",
            "lesson": "active dispatch path must stay in sync",
            "tags": [],
        }],
        packet_context={
            "framework": "sglang", "model": "deepseek-r1",
            "model_family": "deepseek", "workload": "decode", "precision": "fp8",
        },
        ctx=WriteContext(session_id="s2"),
    )
    assert res.status == "disabled"
    assert res.detail["reason"] == "kb_unreachable"


def test_write_verdict_dead_letters_then_opens_breaker(writer):
    w, kb, _, dlq = writer
    kb.fail_next("upsert", times=1)
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context={
            "framework": "sglang", "model": "deepseek-r1",
            "model_family": "deepseek", "workload": "decode", "precision": "fp8",
        },
        ctx=WriteContext(session_id="s_dlq", review_id="rev"),
    )
    assert res.status == "dead_lettered"
    assert res.detail["reason"] == "transport_error"
    assert dlq.files()
    # Subsequent write would short-circuit on the breaker.
    assert w.is_kb_unreachable() is True


def test_decision_reviewer_marks_bundle_when_breaker_open(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    kb = _FlakyKBClient()
    writer = KBWriter(kb, session_memory=sm)
    rev = DecisionReviewer(session_memory=sm, kb_writer=writer)
    writer.force_kb_unreachable()

    bundle = rev.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_breaker",
        "raw_prompt": (
            "=== Shared session state ===\n"
            "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
            "=== Inbox for critic ===\n"
            "  seq=1 msg_id=mmm from=orchestration topic=proposal payload={'action_name': 'kernel_opt'}\n"
        ),
    })
    assert bundle.kb_read_skipped_reason == "kb_unreachable"
    assert any("KB service unreachable" in n for n in bundle.notes)
    assert bundle.review_constraints["kb_breaker"]["open"] is True


def test_decision_reviewer_marks_bundle_when_kb_read_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_READ_ENABLED", "false")
    sm = SessionMemory(root=tmp_path / "sm")
    writer = KBWriter(InMemoryKBClient(), session_memory=sm)
    rev = DecisionReviewer(session_memory=sm, kb_writer=writer)
    bundle = rev.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_disabled",
        "raw_prompt": (
            "=== Shared session state ===\n"
            "model=qwen3-14b framework=sglang\n"
            "=== Inbox for critic ===\n"
            "  seq=1 msg_id=mmm from=orchestration topic=proposal payload={}\n"
        ),
    })
    assert bundle.kb_read_skipped_reason == "kb_read_disabled"


def test_breaker_reflects_per_request_failures_in_bundle(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    kb = _FlakyKBClient()
    writer = KBWriter(kb, session_memory=sm)
    rev = DecisionReviewer(session_memory=sm, kb_writer=writer)
    kb.fail_next("list", times=1)
    bundle = rev.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_per_request",
        "raw_prompt": (
            "=== Shared session state ===\n"
            "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
            "=== Inbox for critic ===\n"
            "  seq=1 msg_id=mmm from=orchestration topic=proposal payload={'action_name': 'baseline'}\n"
        ),
    })
    assert bundle.kb_read_skipped_reason == "kb_unreachable"
    assert bundle.kb_priors_by_proposal["mmm"] == []
