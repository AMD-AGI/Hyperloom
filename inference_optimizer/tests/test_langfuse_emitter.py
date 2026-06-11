# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Live-Langfuse emitter coverage (the opt-in second trace sink).

The local jsonl ledger is always written; this module's emitter mirrors
in-process calls into Langfuse only when three gates pass
(HYPERLOOM_LANGFUSE_ENABLE + LANGFUSE_* creds + importable SDK). These tests
pin:

* default OFF -> emitter is a no-op, builds no client;
* all gates on (with a fake SDK) -> token + conversation rows pair into one
  Generation with correctly mapped usage and input/output text;
* session-end flush emits unpaired halves, backfills ext/*.jsonl
  out-of-process token rows, and turns decision_trace rows into Scores;
* every send is best-effort: a client that raises never propagates.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.trace import langfuse_mapping as lfmap
from inference_optimizer.orchestrator.trace import langfuse_emitter as lfe


# ---------------------------------------------------------------------------
# Fake langfuse SDK
# ---------------------------------------------------------------------------
class _FakeObservation:
    """A span or generation; can spawn children (nested spans/generations)."""

    def __init__(self, sink: "_FakeClient", kind: str, kwargs: dict):
        self._sink = sink
        self.kind = kind
        self.kwargs = kwargs
        self.ended = False
        self.trace_update: dict | None = None
        self.observation_scores: list[dict] = []

    def start_observation(self, *, as_type: str, **kwargs):
        return self._sink._spawn(as_type, kwargs)

    def update_trace(self, **kwargs):
        self.trace_update = kwargs
        self._sink.trace_updates.append(kwargs)

    def score(self, **kwargs):
        self.observation_scores.append(kwargs)
        self._sink.observation_scores.append(kwargs)

    def end(self, **kwargs):
        self.ended = True


class _FakeClient:
    """Records every observation / score so tests can assert on them."""

    def __init__(self, *, raise_on_generation: bool = False):
        self.observations: list[_FakeObservation] = []
        self.generations: list[_FakeObservation] = []
        self.spans: list[_FakeObservation] = []
        self.scores: list[dict] = []  # trace-level
        self.observation_scores: list[dict] = []  # span-level
        self.trace_updates: list[dict] = []
        self.flushed = 0
        self._raise_on_generation = raise_on_generation

    def _spawn(self, as_type: str, kwargs: dict) -> _FakeObservation:
        if as_type == "generation" and self._raise_on_generation:
            raise RuntimeError("boom: langfuse network down")
        obs = _FakeObservation(self, as_type, kwargs)
        self.observations.append(obs)
        if as_type == "generation":
            self.generations.append(obs)
        else:
            self.spans.append(obs)
        return obs

    def start_observation(self, *, as_type: str, **kwargs):
        return self._spawn(as_type, kwargs)

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed += 1

    def span_named(self, name: str) -> _FakeObservation | None:
        for s in self.spans:
            if s.kwargs.get("name") == name:
                return s
        return None


def _install_fake_sdk(monkeypatch, client: _FakeClient) -> None:
    """Inject a fake ``langfuse`` module exposing ``get_client``."""
    fake_mod = types.ModuleType("langfuse")
    fake_mod.get_client = lambda: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_mod)


def _enable_env(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "1")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")


def _write_manifest(session_dir: Path, **fields) -> None:
    base = {
        "session_id": session_dir.name,
        "model_name": "TestModel",
        "claw_session_id": "claw-abc-123",
    }
    base.update(fields)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(base), encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test gets a fresh emitter registry (no cross-test leakage)."""
    lfe._REGISTRY.clear()
    yield
    lfe._REGISTRY.clear()


def _llm_row(**over) -> dict:
    base = {
        "session_id": "SID", "component": "orchestration", "role": "orchestration",
        "tick": 3, "phase": "EXPLORE", "ts": "2026-06-09T15:14:54.100000+00:00",
        "model": "claude-opus-4-7",
        "input_tokens": 100, "output_tokens": 40,
        "cache_creation_input_tokens": None, "cache_read_input_tokens": 7,
    }
    base.update(over)
    return base


def _conv_row(**over) -> dict:
    base = {
        "session_id": "SID", "component": "orchestration", "role": "orchestration",
        "tick": 3, "phase": "EXPLORE", "ts": "2026-06-09T15:14:54.130000+00:00",
        "model": "claude-opus-4-7",
        "prompt": "PROMPT TEXT", "response": "RESPONSE TEXT",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------
def test_disabled_by_default(tmp_path, monkeypatch):
    """No env set -> emitter is a no-op and never builds a client."""
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    assert em.enabled is False
    # Calls are safe no-ops.
    em.record_llm_call(_llm_row())
    em.record_conversation(_conv_row())
    em.flush_session()


def test_enabled_flag_but_missing_creds_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "1")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    assert em.enabled is False


def test_enabled_but_sdk_missing_is_noop(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    # Ensure import fails.
    monkeypatch.setitem(sys.modules, "langfuse", None)
    em = lfe.LangfuseEmitter(tmp_path)
    assert em.enabled is False


def test_all_gates_pass_enables(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="claw-XYZ")
    em = lfe.LangfuseEmitter(sd)
    assert em.enabled is True
    # trace_id derives from claw_session_id (not the dir name) so live + backfill
    # of one claw session collapse onto one trace.
    assert em._trace_id == lfmap.derive_trace_id("claw-XYZ")
    assert em._session_label == "claw-XYZ"


def test_trace_id_falls_back_to_internal_id_without_claw(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="", session_id="internal-99")
    em = lfe.LangfuseEmitter(sd)
    assert em._trace_id == lfmap.derive_trace_id("internal-99")
    assert em._session_label == "internal-99"


def test_trace_session_id_set_from_claw_on_first_generation(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="claw-XYZ")
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row())
    em.record_conversation(_conv_row())
    # update_trace stamped the Langfuse session_id grouping with the claw id.
    assert client.trace_updates, "expected update_trace to be called once"
    assert client.trace_updates[0]["session_id"] == "claw-XYZ"


# ---------------------------------------------------------------------------
# Pairing + usage mapping
# ---------------------------------------------------------------------------
def test_token_and_conversation_pair_into_one_generation(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    em = lfe.LangfuseEmitter(tmp_path)

    em.record_llm_call(_llm_row())
    assert client.generations == []  # waiting for its text half
    em.record_conversation(_conv_row())

    assert len(client.generations) == 1
    g = client.generations[0]
    assert g.kwargs["model"] == "claude-opus-4-7"
    assert g.kwargs["input"] == "PROMPT TEXT"
    assert g.kwargs["output"] == "RESPONSE TEXT"
    assert g.kwargs["metadata"]["has_text"] is True
    assert g.kwargs["metadata"]["phase"] == "EXPLORE"
    # usage_details drops None cache_creation, keeps the rest.
    assert g.kwargs["usage_details"] == {
        "input": 100, "output": 40, "cache_read_input": 7,
    }
    assert g.ended is True


def test_conversation_first_then_token_also_pairs(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_conversation(_conv_row())
    assert client.generations == []
    em.record_llm_call(_llm_row())
    assert len(client.generations) == 1


# ---------------------------------------------------------------------------
# Span hierarchy: trace -> phase span -> agent span -> generation
# ---------------------------------------------------------------------------
def test_generation_nests_under_phase_and_agent_spans(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row(phase="EXPLORE", component="kernel", role="kernel"))
    em.record_conversation(_conv_row(phase="EXPLORE", component="kernel", role="kernel"))

    # Root + phase:EXPLORE + agent:kernel spans exist; the generation is a child.
    assert client.span_named("phase:EXPLORE") is not None
    assert client.span_named("agent:kernel") is not None
    assert len(client.generations) == 1
    assert client.generations[0].kwargs["metadata"]["component"] == "kernel"


def test_distinct_agents_get_distinct_spans(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row(phase="EXPLORE", component="orchestration",
                                role="orchestration", tick=1))
    em.record_conversation(_conv_row(phase="EXPLORE", component="orchestration",
                                     role="orchestration", tick=1))
    em.record_llm_call(_llm_row(phase="EXPLORE", component="kernel",
                                role="kernel", tick=2))
    em.record_conversation(_conv_row(phase="EXPLORE", component="kernel",
                                     role="kernel", tick=2))
    agent_spans = [s for s in client.spans if s.kwargs.get("name", "").startswith("agent:")]
    names = sorted(s.kwargs["name"] for s in agent_spans)
    assert names == ["agent:kernel", "agent:orchestration"]
    # One shared phase span reused across both agents.
    phase_spans = [s for s in client.spans if s.kwargs.get("name") == "phase:EXPLORE"]
    assert len(phase_spans) == 1


def test_flush_closes_all_spans(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row())
    em.record_conversation(_conv_row())
    em.flush_session()
    for s in client.spans:
        assert s.ended is True, f"span {s.kwargs.get('name')} not ended"


# ---------------------------------------------------------------------------
# flush_session: leftovers + ext shards + decision scores
# ---------------------------------------------------------------------------
def _seed_trace_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "SID"
    (sd / "reports" / "trace" / "ext").mkdir(parents=True)
    _write_manifest(sd)
    return sd


def test_flush_emits_unpaired_token_only_generation(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row())  # never gets a conversation half
    assert client.generations == []
    em.flush_session()
    assert len(client.generations) == 1
    assert client.generations[0].kwargs["metadata"]["has_text"] is False
    assert client.flushed == 1


def test_flush_backfills_ext_shards(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    shard = sd / "reports" / "trace" / "ext" / "geak-123.jsonl"
    shard.write_text(
        json.dumps(_llm_row(component="geak", role=None, input_tokens=500,
                            output_tokens=60)) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    geak_gens = [
        g for g in client.generations
        if g.kwargs["metadata"]["component"] == "geak"
    ]
    assert len(geak_gens) == 1
    assert geak_gens[0].kwargs["usage_details"]["input"] == 500
    assert geak_gens[0].kwargs["metadata"]["has_text"] is False


def test_flush_session_is_idempotent_no_duplicate_reemit(tmp_path, monkeypatch):
    """A second flush_session() must NOT re-scan ext shards / decision_trace
    and re-emit -- otherwise Langfuse gets duplicate Generations/Scores."""
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    (sd / "reports" / "trace" / "ext" / "geak-1.jsonl").write_text(
        json.dumps(_llm_row(component="geak", role=None, input_tokens=500,
                            output_tokens=60)) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)

    em.flush_session()
    gens_after_first = len(client.generations)
    assert gens_after_first >= 1

    # Second call: receipt re-written, but no new Generation emitted.
    em.flush_session()
    assert len(client.generations) == gens_after_first


def test_flush_creates_decision_scores(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    dtrace = sd / "reports" / "trace" / "decision_trace.jsonl"
    dtrace.write_text(
        json.dumps({
            "decision": {"change": "tp_sweep", "component": "kernel",
                         "outcome": "KEEP", "gain_pct": 12.5, "task_id": "k1"},
            "phase": "KERNEL", "tick": 5, "ts": "2026-06-09T16:00:00Z",
        }) + "\n"
        + json.dumps({
            "decision": {"change": "radix", "component": "orchestration",
                         "outcome": "REVERT", "gain_pct": None, "task_id": "t2"},
            "phase": "EXPLORE", "tick": 6, "ts": "2026-06-09T16:05:00Z",
        }) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    # Create an agent span for (KERNEL, kernel) so its decision attaches there;
    # the (EXPLORE, orchestration) decision has no span -> trace-level fallback.
    em.record_llm_call(_llm_row(phase="KERNEL", component="kernel", role="kernel"))
    em.record_conversation(_conv_row(phase="KERNEL", component="kernel", role="kernel"))
    em.flush_session()

    span_score_names = sorted(s["name"] for s in client.observation_scores)
    trace_score_names = sorted(s["name"] for s in client.scores)
    # KERNEL/kernel KEEP w/ gain -> 2 span-level scores.
    assert span_score_names == ["decision_outcome", "gain_pct"]
    # EXPLORE/orchestration REVERT w/o gain -> 1 trace-level fallback score.
    assert trace_score_names == ["decision_outcome"]
    gain = [s for s in client.observation_scores if s["name"] == "gain_pct"][0]
    assert gain["value"] == 12.5
    assert gain["data_type"] == "NUMERIC"


# ---------------------------------------------------------------------------
# Best-effort fault posture
# ---------------------------------------------------------------------------
def test_send_exception_is_swallowed(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient(raise_on_generation=True)
    _install_fake_sdk(monkeypatch, client)
    em = lfe.LangfuseEmitter(tmp_path)
    # Must not raise even though start_observation blows up.
    em.record_llm_call(_llm_row())
    em.record_conversation(_conv_row())
    em.flush_session()  # also exercises flush path with a raising client


def test_get_emitter_is_cached_per_session(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    a = lfe.get_emitter(tmp_path)
    b = lfe.get_emitter(tmp_path)
    assert a is b


# ---------------------------------------------------------------------------
# Receipt (session_breakdown ``langfuse`` section)
# ---------------------------------------------------------------------------
def test_receipt_disabled_records_reason_and_redacts(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    r = em.receipt()
    assert r["enabled"] is False
    assert r["disabled_reason"] == "disabled"
    assert r["config"]["enable_flag"] is False
    # No secret material ever appears in the receipt.
    blob = json.dumps(r)
    assert "sk-" not in blob and "pk-" not in blob


def test_receipt_no_credentials_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "1")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    r = em.receipt()
    assert r["enabled"] is False
    assert r["disabled_reason"] == "no_credentials"
    assert r["config"]["enable_flag"] is True
    assert r["config"]["public_key_set"] is False


def test_receipt_counts_and_redaction_when_enabled(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="claw-XYZ")
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row())
    em.record_conversation(_conv_row())

    r = em.receipt()
    assert r["enabled"] is True
    assert r["disabled_reason"] is None
    assert r["config"]["host"] == "https://lf.test"
    assert r["config"]["public_key_set"] is True
    assert r["config"]["secret_key_set"] is True
    assert r["correlated_on"] == "claw_session_id"
    assert r["counts"]["generations_sent"] == 1
    assert r["counts"]["generations_paired"] == 1
    # Pre-flush: not final yet.
    assert r["counts_final"] is False
    # host is a URL (not secret); raw keys never present.
    blob = json.dumps(r)
    assert "pk-test" not in blob and "sk-test" not in blob


def test_flush_writes_receipt_file_with_final_counts(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    # one ext shard (out-of-process) + one decision so flush counts move.
    (sd / "reports" / "trace" / "ext" / "geak-1.jsonl").write_text(
        json.dumps(_llm_row(component="geak", role=None)) + "\n", encoding="utf-8",
    )
    (sd / "reports" / "trace" / "decision_trace.jsonl").write_text(
        json.dumps({
            "decision": {"change": "x", "component": "kernel", "outcome": "KEEP",
                         "gain_pct": 5.0, "task_id": "k1"},
            "phase": "KERNEL", "tick": 1, "ts": "2026-06-09T16:00:00Z",
        }) + "\n", encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()

    persisted = lfe.read_receipt(sd)
    assert persisted is not None
    assert persisted["counts_final"] is True
    assert persisted["counts"]["ext_shards_read"] == 1
    assert persisted["counts"]["generations_sent"] >= 1
    assert persisted["counts"]["scores_sent"] >= 1


def test_read_receipt_absent_returns_none(tmp_path):
    assert lfe.read_receipt(tmp_path / "nope") is None


def test_disabled_flush_still_writes_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    persisted = lfe.read_receipt(sd)
    assert persisted is not None
    assert persisted["enabled"] is False
    assert persisted["disabled_reason"] == "disabled"


# ---------------------------------------------------------------------------
# pair_key: token <-> conversation join identity
# ---------------------------------------------------------------------------
def test_pair_key_distinguishes_same_second_burst():
    """Two calls in the same (component, tick, role) and same UTC second but
    different turns must NOT collide -- otherwise a token row pairs with the
    wrong conversation row in a burst (e.g. multi-turn specialist/critic)."""
    base = {
        "component": "specialist", "tick": 3, "role": "assistant",
        "task_id": "t1", "dyn_id": "d1", "ts": "2026-06-11T10:00:00.100Z",
    }
    turn_a = {**base, "turn": 1}
    turn_b = {**base, "turn": 2, "ts": "2026-06-11T10:00:00.800Z"}  # same second
    assert lfmap.pair_key(turn_a) != lfmap.pair_key(turn_b)


def test_pair_key_matches_token_and_text_halves_of_one_call():
    """The two streams of the SAME logical call (a few ms apart) still pair:
    identical identity fields + same UTC second => equal key."""
    token = {
        "component": "critic", "tick": 5, "role": "assistant", "turn": 2,
        "task_id": "tk", "dyn_id": "dy", "ts": "2026-06-11T10:00:01.020Z",
    }
    text = {**token, "ts": "2026-06-11T10:00:01.450Z"}  # same second, +430ms
    assert lfmap.pair_key(token) == lfmap.pair_key(text)


def test_pair_key_distinguishes_concurrent_models_same_second():
    """ProposalScorer fires several models via asyncio.gather; their rows land
    in the same UTC second with identical keys except model -> must not collide
    (otherwise usage of model A pairs with the prompt/response of model B)."""
    base = {
        "component": "proposal_scorer", "tick": None, "role": "proposal_scorer",
        "task_id": None, "dyn_id": None, "turn": None,
        "ts": "2026-06-11T10:00:00.300Z",
    }
    a = {**base, "model": "qwen-32b"}
    b = {**base, "model": "llama-70b", "ts": "2026-06-11T10:00:00.700Z"}
    assert lfmap.pair_key(a) != lfmap.pair_key(b)


def test_pair_key_scorer_token_and_text_pair_when_roles_match():
    """The scorer's token row and conversation row must share role+model so
    their pair_key matches (the bug: token row had role=None)."""
    token = {
        "component": "proposal_scorer", "role": "proposal_scorer",
        "model": "qwen-32b", "ts": "2026-06-11T10:00:00.100Z",
    }
    text = {**token, "ts": "2026-06-11T10:00:00.900Z"}  # same second
    assert lfmap.pair_key(token) == lfmap.pair_key(text)


def test_pair_key_degrades_when_turn_absent():
    """Legacy rows without turn/task_id/dyn_id still produce a stable key
    (all the new slots are None) rather than raising."""
    row = {"component": "oob", "tick": 1, "role": None,
           "ts": "2026-06-11T10:00:00Z"}
    k = lfmap.pair_key(row)
    assert lfmap.pair_key(dict(row)) == k  # stable / deterministic
