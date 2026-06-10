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
    def __init__(self, sink: "_FakeClient", kind: str, kwargs: dict):
        self._sink = sink
        self.kind = kind
        self.kwargs = kwargs
        self.ended = False

    def end(self, **kwargs):
        self.ended = True


class _FakeClient:
    """Records every observation / score so tests can assert on them."""

    def __init__(self, *, raise_on_generation: bool = False):
        self.generations: list[_FakeObservation] = []
        self.scores: list[dict] = []
        self.flushed = 0
        self._raise_on_generation = raise_on_generation

    def start_observation(self, *, as_type: str, **kwargs):
        if as_type == "generation" and self._raise_on_generation:
            raise RuntimeError("boom: langfuse network down")
        obs = _FakeObservation(self, as_type, kwargs)
        if as_type == "generation":
            self.generations.append(obs)
        return obs

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed += 1


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
    em = lfe.LangfuseEmitter(tmp_path)
    assert em.enabled is True
    assert em._trace_id == lfmap.derive_trace_id(tmp_path.name)


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
    em = lfe.LangfuseEmitter(tmp_path)
    em.record_conversation(_conv_row())
    assert client.generations == []
    em.record_llm_call(_llm_row())
    assert len(client.generations) == 1


# ---------------------------------------------------------------------------
# flush_session: leftovers + ext shards + decision scores
# ---------------------------------------------------------------------------
def _seed_trace_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "SID"
    (sd / "reports" / "trace" / "ext").mkdir(parents=True)
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
    em.flush_session()
    names = sorted(s["name"] for s in client.scores)
    # KEEP w/ gain -> 2 scores; REVERT w/o gain -> 1 score.
    assert names == ["decision_outcome", "decision_outcome", "gain_pct"]
    gain = [s for s in client.scores if s["name"] == "gain_pct"][0]
    assert gain["value"] == 12.5
    assert gain["data_type"] == "NUMERIC"
    assert gain["trace_id"] == em._trace_id


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
