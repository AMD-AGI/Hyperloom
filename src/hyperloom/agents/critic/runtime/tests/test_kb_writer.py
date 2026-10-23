# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end tests for :class:`runtime.kb_writer.KBWriter`, including circuit-breaker behaviour on KB unreachability."""

from __future__ import annotations

import json

import pytest

from hyperloom.agents.critic.runtime.dead_letter import DeadLetter
from hyperloom.agents.critic.runtime.decision_reviewer import DecisionReviewer
from hyperloom.agents.critic.runtime.errors import KBError, KBTransportError, KBValidationError
from hyperloom.agents.critic.runtime.in_memory_kb_client import InMemoryKBClient
from hyperloom.agents.critic.runtime.kb_client import KBListQuery
from hyperloom.agents.critic.runtime.kb_writer import KBWriter, WriteContext
from hyperloom.agents.critic.runtime.session_memory import SessionMemory


def _rows(kb: InMemoryKBClient) -> list[dict]:
    return kb.list(KBListQuery(scope_filter={}, limit=0))["entries"]


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
    assert _rows(kb) == []


def test_write_verdict_reject_creates_pitfall(writer, packet_context):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "topic": "active dispatch path unproven",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
            "kb_evidence": [],
            "confidence": "high",
            "risks": [{"type": "active_path_unproven", "severity": "blocker"}],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev_1"),
    )
    assert res.status == "ok"
    rows = _rows(kb)
    assert len(rows) == 1
    assert rows[0]["kind"] == "pitfall"
    assert rows[0]["importance"] <= 0.84


def test_write_verdict_skipped_when_disabled(monkeypatch, writer, packet_context):
    w, kb, _, _ = writer
    monkeypatch.setenv("KB_WRITE_ENABLED", "false")
    w.write_enabled = False
    res = w.write_verdict(
        verdict={"verdict": "reject", "reasoning": "x"},
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "disabled"


def test_write_verdict_dead_letters_on_validation_error(writer, packet_context):
    w, kb, sm, dlq = writer

    def boom(payload):
        raise KBValidationError("422 bad payload")

    kb.upsert = boom  # type: ignore[method-assign]
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
    assert res.detail["reason"] == "validation_error"


def test_list_priors_uses_session_memory_cache(writer, packet_context):
    w, kb, sm, _ = writer
    kb.upsert(
        {
            "scope": {
                "org": "hyperloom",
                **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
            },
            "kind": "pitfall",
            "slug": "active-path-unproven-pitfall",
            "importance": 0.5,
            "metadata": {"topic": "active path"},
        }
    )
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


def test_write_verdict_with_missing_critical_scope_skipped(writer):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={"verdict": "reject", "reasoning": "x"},
        packet_context={"framework": "sglang"},  # model missing
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "skipped"
    assert "scope_construction_failed" in res.detail.get("reason", "")


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

    def list(self, query):
        self._consume("list")
        return super().list(query)

    def upsert(self, payload):
        self._consume("upsert")
        return super().upsert(payload)


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
    assert "error" not in out2
    assert out2["breaker"]["open"] is True


def test_list_priors_validation_error_does_not_open_breaker(breaker_writer):
    w, kb, _, _ = breaker_writer

    def boom(_query):
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
    w._unreachable_until = w._time_fn() + w._breaker_cooldown
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context={
            "framework": "sglang",
            "model": "deepseek-r1",
            "model_family": "deepseek",
            "workload": "decode",
            "precision": "fp8",
        },
        ctx=WriteContext(session_id="s1", review_id="rev"),
    )
    assert res.status == "disabled"
    assert res.detail["reason"] == "kb_unreachable"
    assert res.detail["breaker"]["open"] is True


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
            "framework": "sglang",
            "model": "deepseek-r1",
            "model_family": "deepseek",
            "workload": "decode",
            "precision": "fp8",
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
    rev = DecisionReviewer(kb_writer=writer)
    writer._unreachable_until = writer._time_fn() + writer._breaker_cooldown

    bundle = rev.prepare_review(
        {
            "session_id": "sess_breaker",
            "raw_prompt": (
                "=== Shared session state ===\n"
                "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
                "=== Inbox for critic ===\n"
                "  seq=1 msg_id=mmm from=orchestration topic=proposal payload={'action_name': 'kernel_opt'}\n"
            ),
        }
    )
    assert bundle.kb_read_skipped_reason == "kb_unreachable"
    assert any("KB service unreachable" in n for n in bundle.notes)
    assert bundle.review_constraints["kb_breaker"]["open"] is True


def test_decision_reviewer_marks_bundle_when_kb_read_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_READ_ENABLED", "false")
    sm = SessionMemory(root=tmp_path / "sm")
    writer = KBWriter(InMemoryKBClient(), session_memory=sm)
    rev = DecisionReviewer(kb_writer=writer)
    bundle = rev.prepare_review(
        {
            "session_id": "sess_disabled",
            "raw_prompt": (
                "=== Shared session state ===\n"
                "model=qwen3-14b framework=sglang\n"
                "=== Inbox for critic ===\n"
                "  seq=1 msg_id=mmm from=orchestration topic=proposal payload={}\n"
            ),
        }
    )
    assert bundle.kb_read_skipped_reason == "kb_read_disabled"


def test_write_verdict_dead_letters_on_generic_kb_error(writer, packet_context):
    """A bare KBError (not transport/validation) dead-letters the upsert as a transport error without the breaker."""
    w, kb, _, dlq = writer

    def boom(payload):
        raise KBError("generic kb failure")

    kb.upsert = boom  # type: ignore[method-assign]
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s_gen", review_id="rev_gen"),
    )
    assert res.status == "dead_lettered"
    assert res.detail["reason"] == "transport_error"
    assert dlq.files()
    assert w.is_kb_unreachable() is False


def test_breaker_reflects_per_request_failures_in_bundle(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    kb = _FlakyKBClient()
    writer = KBWriter(kb, session_memory=sm)
    rev = DecisionReviewer(kb_writer=writer)
    kb.fail_next("list", times=1)
    bundle = rev.prepare_review(
        {
            "session_id": "sess_per_request",
            "raw_prompt": (
                "=== Shared session state ===\n"
                "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
                "=== Inbox for critic ===\n"
                "  seq=1 msg_id=mmm from=orchestration topic=proposal payload={'action_name': 'baseline'}\n"
            ),
        }
    )
    assert bundle.kb_read_skipped_reason == "kb_unreachable"
    assert bundle.kb_priors_by_proposal["mmm"] == []


def test_a_topic_that_cannot_be_slugged_is_skipped(writer, packet_context):
    w, _, _, _ = writer
    topic = "\ud800 lone surrogate topic"
    error = "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed"

    verdict = {"verdict": "reject", "topic": topic, "reasoning": "r", "packet_evidence": ["e"]}
    skipped = w.write_verdict(verdict=verdict, packet_context=packet_context, ctx=WriteContext(session_id="s"))

    assert skipped.to_dict() == {"status": "skipped", "detail": {"reason": "slug_failed", "error": error}}
