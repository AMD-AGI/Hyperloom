"""End-to-end tests for :class:`runtime.kb_writer.KBWriter`, including
circuit-breaker behaviour on KB unreachability.

We use :class:`runtime.in_memory_kb_client.InMemoryKBClient` so the tests
exercise the same surface a production HTTP client would, but without
network IO. Dead-letter and session-memory paths use the temp fixtures.

The breaker tests live in :class:`TestKbBreaker` below — they cover:

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

import json

import pytest

from hyperloom.agents.critic.runtime.dead_letter import DeadLetter
from hyperloom.agents.critic.runtime.decision_reviewer import DecisionReviewer
from hyperloom.agents.critic.runtime.errors import KBTransportError, KBValidationError
from hyperloom.agents.critic.runtime.in_memory_kb_client import InMemoryKBClient
from hyperloom.agents.critic.runtime.kb_writer import KBWriter, WriteContext
from hyperloom.agents.critic.runtime.session_memory import SessionMemory


# ---------------------------------------------------------------------------
# KBWriter happy-path / dead-letter
# ---------------------------------------------------------------------------


@pytest.fixture()
def packet_context():
    return {
        "framework": "sglang",
        "model": "deepseek-r1-0528-fp8",
        "model_family": "deepseek",
        "workload": "decode",
        "precision": "fp8",
    }


@pytest.fixture()
def writer(tmp_path):
    kb = InMemoryKBClient()
    sm = SessionMemory(root=tmp_path / "sm")
    dlq = DeadLetter(root=tmp_path / "dlq")
    return KBWriter(kb, session_memory=sm, dead_letter=dlq), kb, sm, dlq


def test_write_verdict_skipped_for_advise(writer, packet_context):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={
            "verdict": "advise",
            "reasoning": "small concurrency mismatch",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "skipped"
    assert kb.all_rows() == []


def test_write_verdict_reject_creates_pitfall(writer, packet_context):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
            "kb_evidence": [],
            "confidence": "high",
            "risks": [{"type": "active_path_unproven", "severity": "blocker"}],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev_1", topic="active dispatch path unproven"),
    )
    assert res.status == "ok"
    rows = kb.all_rows()
    assert len(rows) == 1
    assert rows[0]["kind"] == "pitfall"
    assert rows[0]["importance"] <= 0.84


def test_write_verdict_skipped_when_disabled(monkeypatch, writer, packet_context):
    w, kb, _, _ = writer
    monkeypatch.setenv("KB_WRITE_ENABLED", "false")
    w.write_enabled = False  # honour env at runtime
    res = w.write_verdict(
        verdict={"verdict": "reject", "reasoning": "x"},
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "disabled"


def test_write_verdict_dead_letters_on_validation_error(writer, packet_context):
    w, kb, sm, dlq = writer
    kb.simulate_failure(endpoint="upsert", times=1, error={"code": 422})
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev"),
    )
    assert res.status == "dead_lettered"
    files = dlq.files()
    assert len(files) == 1
    line = files[0].read_text("utf-8").splitlines()[0]
    record = json.loads(line)
    assert record["endpoint"] == "upsert"
    assert "session_id" in record["context"]


def test_write_kb_drafts_batch_inserts_and_filters_unknown_categories(writer, packet_context):
    w, kb, _, _ = writer
    drafts = [
        {
            "category": "kernel_optimization",
            "action": "Patch fused attention kernel for Qwen3-14B on MI355X.",
            "lesson": "Active dispatch path must be updated jointly.",
            "tags": ["attention"],
            "result": {"status": "KEEP", "gain_pct": 4.2},
            "confidence": 0.9,
        },
        {
            "category": "definitely_not_a_real_category",
            "action": "x",
        },
    ]
    res = w.write_kb_drafts(
        kb_drafts=drafts,
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev_2"),
    )
    assert res.status == "ok"
    assert len(kb.all_rows()) == 1
    assert kb.all_rows()[0]["kind"] == "technique"
    rejected = res.detail["rejected"]
    assert rejected and rejected[0]["draft"]["category"] == "definitely_not_a_real_category"


def test_list_priors_uses_session_memory_cache(writer, packet_context):
    w, kb, sm, _ = writer
    kb.upsert({
        "scope": {
            "org": "hyperloom",
            **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
        },
        "kind": "pitfall",
        "slug": "active-path-unproven-pitfall",
        "importance": 0.5,
        "metadata": {"topic": "active path"},
    })
    scope = {
        "org": "hyperloom",
        **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
    }
    ctx = WriteContext(session_id="s_cache")
    first = w.list_priors(scope=scope, kind="pitfall", topic="active path", ctx=ctx)
    assert first["cache"] == "miss"
    assert first["priors"]
    second = w.list_priors(scope=scope, kind="pitfall", topic="active path", ctx=ctx)
    assert second["cache"] == "hit"
    assert second["priors"] == first["priors"]


def test_add_contradiction_writes_edge(writer, packet_context):
    w, kb, _, _ = writer
    a = kb.upsert({
        "scope": {
            "org": "hyperloom",
            **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
        },
        "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5,
    })["row"]["id"]
    b = kb.upsert({
        "scope": {
            "org": "hyperloom",
            **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
        },
        "kind": "pitfall", "slug": "abcdef-2", "importance": 0.5,
    })["row"]["id"]
    res = w.add_contradiction(
        new_id=a, old_ids=[b], ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "ok"
    rows = {r["id"]: r for r in kb.all_rows()}
    assert b in rows[a]["edges"]["contradicts"]
    assert a in rows[b]["edges"]["contradicts"]


def test_write_verdict_with_missing_critical_scope_skipped(writer):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={"verdict": "reject", "reasoning": "x"},
        packet_context={"framework": "sglang"},  # model missing
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "skipped"
    assert "scope_construction_failed" in res.detail.get("reason", "")


# ===========================================================================
# Circuit-breaker behaviour (formerly test_kb_breaker.py)
# ===========================================================================


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
def breaker_writer(tmp_path):
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


def test_list_priors_short_circuits_after_first_transport_error(breaker_writer):
    w, kb, _, _ = breaker_writer
    kb.fail_next("list", times=1)
    out1 = w.list_priors(scope=_scope())
    assert out1["cache"] == "kb_unreachable"
    assert out1["priors"] == []
    assert "error" in out1
    assert w.is_kb_unreachable() is True

    out2 = w.list_priors(scope=_scope())
    assert out2["cache"] == "kb_unreachable"
    assert "error" not in out2  # no transport call made
    assert out2["breaker"]["open"] is True


def test_list_priors_validation_error_does_not_open_breaker(breaker_writer):
    w, kb, _, _ = breaker_writer

    def boom(**kwargs):
        raise KBValidationError("422 bad scope")

    kb.list = boom  # type: ignore[method-assign]
    out = w.list_priors(scope=_scope())
    assert out["cache"] == "miss"
    assert out["priors"] == []
    assert "error" in out
    assert w.is_kb_unreachable() is False


def test_list_priors_resets_breaker_on_recovery(breaker_writer):
    w, kb, _, _ = breaker_writer
    kb.fail_next("list", times=1)
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
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
    assert w.is_kb_unreachable() is False
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
    assert w.is_kb_unreachable() is False
    kb.fail_next("list", times=1)
    assert w.list_priors(scope=_scope())["cache"] == "kb_unreachable"
    assert w.is_kb_unreachable() is True


def test_write_verdict_disabled_when_breaker_open(breaker_writer):
    w, _, _, _ = breaker_writer
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


def test_write_kb_drafts_disabled_when_breaker_open(breaker_writer):
    w, _, _, _ = breaker_writer
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


def test_write_verdict_dead_letters_then_opens_breaker(breaker_writer):
    w, kb, _, dlq = breaker_writer
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
