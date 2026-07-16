# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Live-Langfuse emitter coverage (the opt-in second trace sink).

The local jsonl ledger is always written; this module's emitter mirrors
in-process calls into Langfuse only when three gates pass
(HYPERLOOM_LANGFUSE_ENABLE + LANGFUSE_* creds + importable SDK). These tests
pin:

* default OFF -> emitter is a no-op, builds no client;
* all gates on (with a fake SDK) -> token + conversation rows pair into one
  Generation with correctly mapped usage and input/output text;
* session-end flush emits unpaired halves, backfills the recipe-KB /
  specialist-intel audit spans, and turns decision_trace rows into Scores;
* every send is best-effort: a client that raises never propagates.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace import langfuse_mapping as lfmap
from hyperloom.orchestrator.trace import langfuse_emitter as lfe


class _FakeObservation:
    """A span or generation; can spawn children (nested spans/generations)."""

    def __init__(self, sink: "_FakeClient", kind: str, kwargs: dict):
        self._sink = sink
        self.kind = kind
        self.kwargs = kwargs
        self.ended = False
        self.end_time = None
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
        self.end_time = kwargs.get("end_time")


class _FakeClient:
    """Records every observation / score so tests can assert on them."""

    def __init__(self, *, raise_on_generation: bool = False):
        self.observations: list[_FakeObservation] = []
        self.generations: list[_FakeObservation] = []
        self.spans: list[_FakeObservation] = []
        self.scores: list[dict] = []
        self.observation_scores: list[dict] = []
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
        json.dumps(base),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test gets a fresh emitter registry (no cross-test leakage)."""
    lfe._REGISTRY.clear()
    yield
    lfe._REGISTRY.clear()


def _llm_row(**over) -> dict:
    base = {
        "session_id": "SID",
        "component": "orchestration",
        "role": "orchestration",
        "tick": 3,
        "phase": "EXPLORE",
        "ts": "2026-06-09T15:14:54.100000+00:00",
        "model": "claude-opus-4-7",
        "input_tokens": 100,
        "output_tokens": 40,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": 7,
    }
    base.update(over)
    return base


def _conv_row(**over) -> dict:
    base = {
        "session_id": "SID",
        "component": "orchestration",
        "role": "orchestration",
        "tick": 3,
        "phase": "EXPLORE",
        "ts": "2026-06-09T15:14:54.130000+00:00",
        "model": "claude-opus-4-7",
        "prompt": "PROMPT TEXT",
        "response": "RESPONSE TEXT",
    }
    base.update(over)
    return base


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
    # trace_id derives from claw_session_id so live + backfill collapse onto one trace.
    assert em._trace_id == lfmap.derive_trace_id("claw-XYZ")
    assert em._session_label == "claw-XYZ"


def test_enabling_seeds_default_flush_interval(tmp_path, monkeypatch):
    # With no pinned flush cadence, building the client tightens the SDK
    # auto-flush interval so a killed run still lands its latest observations.
    _enable_env(monkeypatch)
    monkeypatch.delenv("LANGFUSE_FLUSH_INTERVAL", raising=False)
    monkeypatch.delenv("LANGFUSE_FLUSH_AT", raising=False)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    em = lfe.LangfuseEmitter(tmp_path / "SID")
    assert em.enabled is True
    import os

    assert os.environ["LANGFUSE_FLUSH_INTERVAL"] == "1"
    # flush_at is left to the SDK default (no per-event HTTP).
    assert "LANGFUSE_FLUSH_AT" not in os.environ


def test_operator_flush_interval_is_respected(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "10")
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    lfe.LangfuseEmitter(tmp_path / "SID")
    import os

    assert os.environ["LANGFUSE_FLUSH_INTERVAL"] == "10"


def test_session_start_emits_marker_with_provenance(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    monkeypatch.setenv("USER_DATA_PATH", "/weka/users/alice")
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    monkeypatch.setenv("HYPERLOOM_CUSTOM_FLAG", "on")
    monkeypatch.setenv("MY_API_TOKEN", "super-secret-value")
    sd = tmp_path / "SID"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "SID",
                "model_name": "TestModel",
                "claw_session_id": "claw-XYZ",
                "sandbox_user_id": "alice",
                "code_revision": "abc1234",
                "session_dir": str(sd),
                "host": "node-7",
                "image": "registry/hyperloom:tag",
                "pid": 4242,
                "stack_fingerprint": {"rocm": "6.2", "vllm": "0.6"},
                "workload": {"isl": 128, "osl": 256},
                "dependencies": {"magpie": {"path": "/weka/magpie", "commit": "deadbee"}},
            }
        ),
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.record_session_start()

    marker = client.span_named("session_start")
    assert marker is not None
    out = marker.kwargs.get("output") or {}
    assert out["claw_session_id"] == "claw-XYZ"
    assert out["sandbox_user_id"] == "alice"
    assert out["code_revision"] == "abc1234"
    assert out["session_dir"] == str(sd)
    assert out["host"] == "node-7"
    assert out["image"] == "registry/hyperloom:tag"
    assert out["pid"] == 4242
    assert out["stack_fingerprint"] == {"rocm": "6.2", "vllm": "0.6"}
    assert out["workload"] == {"isl": 128, "osl": 256}
    assert out["dependencies"]["magpie"]["commit"] == "deadbee"
    assert out["user_data_path"] == "/weka/users/alice"
    # Environment snapshot is attached, with secret-looking values redacted.
    env = out["env"]
    assert env["HYPERLOOM_CUSTOM_FLAG"] == "on"
    assert env["USER_DATA_PATH"] == "/weka/users/alice"
    assert env["MY_API_TOKEN"] == "***redacted***"
    assert env["LANGFUSE_SECRET_KEY"] == "***redacted***"
    # Scalar correlation keys also land on the observation metadata.
    meta = marker.kwargs.get("metadata") or {}
    assert meta["code_revision"] == "abc1234"
    assert meta["user_data_path"] == "/weka/users/alice"
    # Trace is grouped on the claw session id.
    assert marker.trace_update is not None
    assert marker.trace_update["session_id"] == "claw-XYZ"
    assert client.flushed >= 1


def test_redact_env_keeps_names_redacts_secret_values():
    snap = lfmap.redact_env(
        {
            "PATH": "/usr/bin",
            "HF_TOKEN": "hf_xxx",
            "AWS_SECRET_ACCESS_KEY": "abc",
            "DB_PASSWORD": "pw",
            "PLAIN": "value",
        }
    )
    assert snap["PATH"] == "/usr/bin"
    assert snap["PLAIN"] == "value"
    assert snap["HF_TOKEN"] == "***redacted***"
    assert snap["AWS_SECRET_ACCESS_KEY"] == "***redacted***"
    assert snap["DB_PASSWORD"] == "***redacted***"


def test_session_start_is_idempotent(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="claw-XYZ")
    em = lfe.LangfuseEmitter(sd)
    em.record_session_start()
    em.record_session_start()
    assert len([s for s in client.spans if s.kwargs.get("name") == "session_start"]) == 1


def test_session_start_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    assert em.enabled is False
    em.record_session_start()
    assert client.span_named("session_start") is None


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
    # update_trace stamped the session_id grouping with the claw id.
    assert client.trace_updates, "expected update_trace to be called once"
    assert client.trace_updates[0]["session_id"] == "claw-XYZ"


def test_token_and_conversation_pair_into_one_generation(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    em = lfe.LangfuseEmitter(tmp_path)

    em.record_llm_call(_llm_row())
    assert client.generations == []
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
        "input": 100,
        "output": 40,
        "cache_read_input": 7,
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


def test_generation_nests_under_phase_and_agent_spans(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row(phase="EXPLORE", component="kernel_agent", role="kernel_agent"))
    em.record_conversation(_conv_row(phase="EXPLORE", component="kernel_agent", role="kernel_agent"))

    # Root + phase:EXPLORE + agent:kernel spans exist; the generation is a child.
    assert client.span_named("phase:EXPLORE") is not None
    assert client.span_named("agent:kernel_agent") is not None
    assert len(client.generations) == 1
    assert client.generations[0].kwargs["metadata"]["component"] == "kernel_agent"


def test_distinct_agents_get_distinct_spans(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row(phase="EXPLORE", component="orchestration", role="orchestration", tick=1))
    em.record_conversation(_conv_row(phase="EXPLORE", component="orchestration", role="orchestration", tick=1))
    em.record_llm_call(_llm_row(phase="EXPLORE", component="kernel_agent", role="kernel_agent", tick=2))
    em.record_conversation(_conv_row(phase="EXPLORE", component="kernel_agent", role="kernel_agent", tick=2))
    agent_spans = [s for s in client.spans if s.kwargs.get("name", "").startswith("agent:")]
    names = sorted(s.kwargs["name"] for s in agent_spans)
    assert names == ["agent:kernel_agent", "agent:orchestration"]
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


def test_record_session_breakdown_attaches_full_json(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="claw-XYZ")
    em = lfe.LangfuseEmitter(sd)

    breakdown = {
        "schema_version": "hyperloom.session_breakdown.v3.0",
        "exporter_version": "session-breakdown-1.0.0",
        "session": {"stop_reason": "budget_exhausted"},
        "kernel_journey": {"kernels": []},
    }
    em.record_session_breakdown(breakdown)

    span = client.span_named("session_breakdown")
    assert span is not None
    assert span.kwargs["output"] == breakdown
    assert span.kwargs["trace_context"] == {"trace_id": em._trace_id}
    assert span.kwargs["metadata"]["schema_version"] == breakdown["schema_version"]
    assert span.ended is True
    assert client.flushed >= 1
    assert em._counts["breakdown_recorded"] == 1


def test_record_session_breakdown_is_idempotent(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_session_breakdown({"schema_version": "v3.0"})
    em.record_session_breakdown({"schema_version": "v3.0"})
    spans = [s for s in client.spans if s.kwargs.get("name") == "session_breakdown"]
    assert len(spans) == 1


def test_record_session_breakdown_cross_process_idempotent(tmp_path, monkeypatch):
    # A second, fresh emitter skips re-attaching because the persisted receipt
    # records breakdown_recorded=1.
    _enable_env(monkeypatch)
    client1 = _FakeClient()
    _install_fake_sdk(monkeypatch, client1)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em1 = lfe.LangfuseEmitter(sd)
    em1.record_session_breakdown({"schema_version": "v3.0"})
    assert em1._counts["breakdown_recorded"] == 1
    assert (lfe.read_receipt(sd) or {}).get("counts", {}).get("breakdown_recorded")

    client2 = _FakeClient()
    _install_fake_sdk(monkeypatch, client2)
    em2 = lfe.LangfuseEmitter(sd)
    em2.record_session_breakdown({"schema_version": "v3.0"})
    assert [s for s in client2.spans if s.kwargs.get("name") == "session_breakdown"] == []
    assert em2._counts["breakdown_recorded"] == 1


def test_record_session_breakdown_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    em.record_session_breakdown({"schema_version": "v3.0"})
    assert em._counts["breakdown_recorded"] == 0


def test_module_record_session_breakdown_reads_file(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    (sd / "session_breakdown.json").write_text(
        json.dumps({"schema_version": "v3.0", "session": {"stop_reason": "ok"}}),
        encoding="utf-8",
    )
    lfe.record_session_breakdown(sd)
    span = client.span_named("session_breakdown")
    assert span is not None
    assert span.kwargs["output"]["schema_version"] == "v3.0"


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
    em.record_llm_call(_llm_row())
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
    shard = sd / "reports" / "trace" / "ext" / "forge-123.jsonl"
    shard.write_text(
        json.dumps(_llm_row(component="forge", role=None, input_tokens=500, output_tokens=60)) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    geak_gens = [g for g in client.generations if g.kwargs["metadata"]["component"] == "forge"]
    assert len(geak_gens) == 1
    assert geak_gens[0].kwargs["usage_details"]["input"] == 500
    assert geak_gens[0].kwargs["metadata"]["has_text"] is False


def test_flush_session_is_idempotent_no_duplicate_reemit(tmp_path, monkeypatch):
    """A second flush_session() must NOT re-scan leftovers / decision_trace and
    re-emit (would duplicate Generations/Scores)."""
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    (sd / "reports" / "trace" / "ext" / "forge-1.jsonl").write_text(
        json.dumps(_llm_row(component="forge", role=None, input_tokens=500, output_tokens=60)) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row())  # unpaired token half

    em.flush_session()
    gens_after_first = len(client.generations)
    assert gens_after_first >= 1

    # Second call: no new Generation emitted.
    em.flush_session()
    assert len(client.generations) == gens_after_first


def test_flush_creates_decision_scores(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    dtrace = sd / "reports" / "trace" / "decision_trace.jsonl"
    dtrace.write_text(
        json.dumps(
            {
                "decision": {
                    "change": "tp_sweep",
                    "component": "kernel_agent",
                    "operation_kind": "kernel_opt",
                    "outcome": "KEEP",
                    "gain_pct": 12.5,
                    "task_id": "k1",
                },
                "phase": "KERNEL",
                "tick": 5,
                "ts": "2026-06-09T16:00:00Z",
            }
        )
        + "\n"
        + json.dumps(
            {
                "decision": {
                    "change": "radix",
                    "component": "grid",
                    "operation_kind": "param",
                    "provenance": "default_grid",
                    "outcome": "REVERT",
                    "gain_pct": None,
                    "task_id": "t2",
                },
                "phase": "EXPLORE",
                "tick": 6,
                "ts": "2026-06-09T16:05:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    # Create an agent span so its step span parents there.
    em.record_llm_call(_llm_row(phase="KERNEL_AGENT", component="kernel_agent", role="kernel_agent"))
    em.record_conversation(_conv_row(phase="KERNEL_AGENT", component="kernel_agent", role="kernel_agent"))
    em.flush_session()

    # Each decision opens an optimization_step:<operation_kind> span whose
    # scores attach to it.
    kernel_step = client.span_named("optimization_step:kernel_opt")
    param_step = client.span_named("optimization_step:param")
    assert kernel_step is not None
    assert kernel_step.kwargs["metadata"]["operation_kind"] == "kernel_opt"
    assert param_step is not None
    assert param_step.kwargs["metadata"]["operation_kind"] == "param"
    assert param_step.kwargs["metadata"]["provenance"] == "default_grid"

    span_score_names = sorted(s["name"] for s in client.observation_scores)
    assert span_score_names == ["decision_outcome", "decision_outcome", "gain_pct"]
    gain = [s for s in client.observation_scores if s["name"] == "gain_pct"][0]
    assert gain["value"] == 12.5
    assert gain["data_type"] == "NUMERIC"
    assert gain["metadata"]["operation_kind"] == "kernel_opt"


def test_flush_emits_proposal_score_calibration(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    dtrace = sd / "reports" / "trace" / "decision_trace.jsonl"
    dtrace.write_text(
        json.dumps({
            "decision": {
                "change": "tp_sweep", "component": "specialist:perf",
                "operation_kind": "param", "outcome": "KEEP",
                "gain_pct": 8.0, "task_id": "spec-1",
                "variant_name": "v1",
                "proposal_scores": [
                    {"rater": "gpt-5.5", "score": 7.0, "reason": "ok"},
                    {"rater": "claude", "score": 9.0, "reason": "good"},
                ],
            },
            "phase": "EXPLORE", "tick": 5, "ts": "2026-06-09T16:00:00Z",
        }) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    # Seed a specialist agent span so the decision step span has a parent.
    em.record_llm_call(
        _llm_row(phase="EXPLORE", component="specialist", role="specialist")
    )
    em.flush_session()
    names = sorted(s["name"] for s in client.observation_scores)
    assert "proposal_score" in names
    pscore = [s for s in client.observation_scores
              if s["name"] == "proposal_score"][0]
    assert pscore["value"] == 8.0           # mean(7, 9)
    assert pscore["data_type"] == "NUMERIC"
    # Realized gain emitted alongside for calibration error.
    assert "gain_pct" in names


def test_flush_emits_predicted_gain_calibration(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    dtrace = sd / "reports" / "trace" / "decision_trace.jsonl"
    dtrace.write_text(
        json.dumps({
            "decision": {
                "change": "tp_sweep", "component": "specialist:perf",
                "operation_kind": "param", "outcome": "KEEP",
                "gain_pct": 6.0, "predicted_gain_pct": 12.5,
                "task_id": "spec-1",
            },
            "phase": "EXPLORE", "tick": 5, "ts": "2026-06-09T16:00:00Z",
        }) + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(
        _llm_row(phase="EXPLORE", component="specialist", role="specialist")
    )
    em.flush_session()
    by_name = {s["name"]: s for s in client.observation_scores}
    # predicted + realized both emitted as NUMERIC on the same decision.
    assert "predicted_gain_pct" in by_name and "gain_pct" in by_name
    assert by_name["predicted_gain_pct"]["value"] == 12.5
    assert by_name["gain_pct"]["value"] == 6.0


def test_record_kb_span_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    em.record_kb_span(name="kb_assess:iter_0", agent="critic", output={"x": 1})
    assert em._counts["kb_spans_sent"] == 0


def test_record_kb_span_nests_under_agent(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    em.record_kb_span(
        name="kb_assess:iter_2",
        agent="critic",
        output={"mode": "dry_run", "verdict_count": 1},
        metadata={"kind": "kb_assess", "iter": 2},
        ts="2026-06-09T15:14:54.100000+00:00",
    )
    span = client.span_named("kb_assess:iter_2")
    assert span is not None
    assert span.kwargs["output"] == {"mode": "dry_run", "verdict_count": 1}
    assert span.kwargs["metadata"]["kind"] == "kb_assess"
    assert client.span_named("agent:critic") is not None
    assert span.ended is True
    assert em._counts["kb_spans_sent"] == 1


def test_flush_backfills_recipe_snapshot_audit(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    from hyperloom.inference_optimizer.session.session_paths import recipe_snapshot_audit_jsonl

    audit = recipe_snapshot_audit_jsonl(sd)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "ts": "2026-06-09T15:14:54Z",
                    "method": "get_recipe",
                    "remote": "gbrain",
                    "resolution": "remote",
                    "hit": True,
                },
                {
                    "ts": "2026-06-09T15:14:55Z",
                    "method": "search",
                    "remote": "cortex",
                    "resolution": "local",
                    "hit": False,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    assert client.span_named("kb:recipe_snapshot:get_recipe") is not None
    assert client.span_named("kb:recipe_snapshot:search") is not None
    assert client.span_named("agent:recipe_kb") is not None
    persisted = lfe.read_receipt(sd)
    assert persisted["counts"]["recipe_audit_read"] == 2
    assert persisted["counts"]["kb_spans_sent"] == 2


def test_flush_recipe_audit_idempotent(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    from hyperloom.inference_optimizer.session.session_paths import recipe_snapshot_audit_jsonl

    audit = recipe_snapshot_audit_jsonl(sd)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps(
            {
                "ts": "2026-06-09T15:14:54Z",
                "method": "get_recipe",
                "remote": "gbrain",
                "resolution": "remote",
                "hit": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    n = len([s for s in client.spans if s.kwargs.get("name", "").startswith("kb:recipe_snapshot")])
    em.flush_session()
    n2 = len([s for s in client.spans if s.kwargs.get("name", "").startswith("kb:recipe_snapshot")])
    assert n == 1 and n2 == 1


def test_flush_backfills_specialist_intel(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    from hyperloom.inference_optimizer.session.session_paths import specialist_intel_path
    intel = specialist_intel_path(sd)
    intel.parent.mkdir(parents=True, exist_ok=True)
    intel.write_text("\n".join(json.dumps(r) for r in [
        {"ts": "2026-06-09T15:14:54Z", "tool": "WebSearch",
         "task_id": "t1", "turn": 1, "query": "rocm flash attn"},
        {"ts": "2026-06-09T15:14:55Z", "tool": "mcp__cortex_kb__lookup",
         "task_id": "t1", "turn": 1, "query": "x"},
    ]) + "\n", encoding="utf-8")
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    assert client.span_named("intel:WebSearch") is not None
    assert client.span_named("intel:mcp__cortex_kb__lookup") is not None
    assert client.span_named("agent:specialist") is not None
    persisted = lfe.read_receipt(sd)
    assert persisted["counts"]["specialist_intel_read"] == 2


def test_flush_backfills_forge_steps(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    from hyperloom.inference_optimizer.session.session_paths import forge_steps_path
    steps = forge_steps_path(sd)
    steps.parent.mkdir(parents=True, exist_ok=True)
    steps.write_text("\n".join(json.dumps(r) for r in [
        {"ts": "2026-06-09T15:14:54Z", "kind": "iteration", "kernel_id": "k1",
         "iteration": 1, "decision": "KEEP", "wall_ms": 88.1},
        {"ts": "2026-06-09T15:14:55Z", "kind": "iteration", "kernel_id": "k1",
         "iteration": 2, "decision": "REVERT", "wall_ms": 90.0},
        {"ts": "2026-06-09T15:14:56Z", "kind": "summary", "kernel_id": "k1",
         "iterations": 2, "kept": 1, "termination_reason": "plateaued"},
    ]) + "\n", encoding="utf-8")
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    assert client.span_named("forge:iter:1") is not None
    assert client.span_named("forge:iter:2") is not None
    assert client.span_named("forge:summary") is not None
    assert client.span_named("agent:forge") is not None
    persisted = lfe.read_receipt(sd)
    assert persisted["counts"]["forge_steps_read"] == 3


def test_flush_backfills_gemm_tuning(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path
    steps = gemm_tuning_steps_path(sd)
    steps.parent.mkdir(parents=True, exist_ok=True)
    steps.write_text("\n".join(json.dumps(r) for r in [
        {"ts": "2026-06-09T15:14:54Z", "kind": "gemm_tuning", "engine": "forge",
         "backend": "forge", "decision": "KEEP", "best_speedup": 1.12},
        {"ts": "2026-06-09T15:14:56Z", "kind": "gemm_tuning", "engine": "geak",
         "backend": "geak", "decision": "REVERT"},
    ]) + "\n", encoding="utf-8")
    em = lfe.LangfuseEmitter(sd)
    em.flush_session()
    forge_span = client.span_named("gemm_tuning:forge")
    assert forge_span is not None
    assert forge_span.kwargs["metadata"]["engine"] == "forge"
    assert forge_span.kwargs["metadata"]["best_speedup"] == 1.12
    assert client.span_named("gemm_tuning:geak") is not None
    # Attributed to its own gemm_tuning source agent span.
    assert client.span_named("agent:gemm_tuning") is not None
    persisted = lfe.read_receipt(sd)
    assert persisted["counts"]["gemm_tuning_read"] == 2


def test_generation_uses_latency_for_duration(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)
    row = _llm_row()
    row["latency_ms"] = 2000
    em.record_llm_call(row)
    em.flush_session()
    gen = client.generations[0] if client.generations else None
    assert gen is not None
    # start = ts - latency; end = ts -> non-zero duration.
    start = gen.kwargs.get("start_time")
    end = getattr(gen, "end_time", None)
    assert start is not None and end is not None
    assert (end - start).total_seconds() == pytest.approx(2.0, abs=0.01)


def test_send_exception_is_swallowed(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient(raise_on_generation=True)
    _install_fake_sdk(monkeypatch, client)
    em = lfe.LangfuseEmitter(tmp_path)
    # Must not raise even though start_observation blows up.
    em.record_llm_call(_llm_row())
    em.record_conversation(_conv_row())
    em.flush_session()


def test_get_emitter_is_cached_per_session(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    a = lfe.get_emitter(tmp_path)
    b = lfe.get_emitter(tmp_path)
    assert a is b


def test_receipt_disabled_records_reason_and_redacts(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    r = em.receipt()
    assert r["enabled"] is False
    assert r["disabled_reason"] == "disabled"
    assert r["config"]["enable_flag"] is False
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
    blob = json.dumps(r)
    assert "pk-test" not in blob and "sk-test" not in blob


def test_flush_writes_receipt_file_with_final_counts(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = _seed_trace_dir(tmp_path)
    # one ext shard + one unpaired token half + one decision so flush counts move.
    (sd / "reports" / "trace" / "ext" / "forge-1.jsonl").write_text(
        json.dumps(_llm_row(component="forge", role=None)) + "\n",
        encoding="utf-8",
    )
    (sd / "reports" / "trace" / "decision_trace.jsonl").write_text(
        json.dumps(
            {
                "decision": {"change": "x", "component": "kernel_agent", "outcome": "KEEP", "gain_pct": 5.0, "task_id": "k1"},
                "phase": "KERNEL",
                "tick": 1,
                "ts": "2026-06-09T16:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    em = lfe.LangfuseEmitter(sd)
    em.record_llm_call(_llm_row())  # unpaired token half
    em.flush_session()

    persisted = lfe.read_receipt(sd)
    assert persisted is not None
    assert persisted["counts_final"] is True
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


def test_pair_key_distinguishes_same_second_burst():
    """Two calls in the same (component, tick, role) and same UTC second but
    different turns must NOT collide (otherwise a token row pairs with the wrong
    conversation row in a burst)."""
    base = {
        "component": "specialist",
        "tick": 3,
        "role": "assistant",
        "task_id": "t1",
        "dyn_id": "d1",
        "ts": "2026-06-11T10:00:00.100Z",
    }
    turn_a = {**base, "turn": 1}
    turn_b = {**base, "turn": 2, "ts": "2026-06-11T10:00:00.800Z"}
    assert lfmap.pair_key(turn_a) != lfmap.pair_key(turn_b)


def test_pair_key_matches_token_and_text_halves_of_one_call():
    """The two streams of the SAME logical call (a few ms apart) still pair:
    identical identity fields + same UTC second => equal key."""
    token = {
        "component": "critic",
        "tick": 5,
        "role": "assistant",
        "turn": 2,
        "task_id": "tk",
        "dyn_id": "dy",
        "ts": "2026-06-11T10:00:01.020Z",
    }
    text = {**token, "ts": "2026-06-11T10:00:01.450Z"}
    assert lfmap.pair_key(token) == lfmap.pair_key(text)


def test_pair_key_distinguishes_concurrent_models_same_second():
    """Concurrent models land in the same UTC second with identical keys except
    model -> must not collide (usage of model A pairing with model B's text)."""
    base = {
        "component": "proposal_scorer",
        "tick": None,
        "role": "proposal_scorer",
        "task_id": None,
        "dyn_id": None,
        "turn": None,
        "ts": "2026-06-11T10:00:00.300Z",
    }
    a = {**base, "model": "qwen-32b"}
    b = {**base, "model": "llama-70b", "ts": "2026-06-11T10:00:00.700Z"}
    assert lfmap.pair_key(a) != lfmap.pair_key(b)


def test_pair_key_scorer_token_and_text_pair_when_roles_match():
    """The scorer's token row and conversation row must share role+model so
    their pair_key matches."""
    token = {
        "component": "proposal_scorer",
        "role": "proposal_scorer",
        "model": "qwen-32b",
        "ts": "2026-06-11T10:00:00.100Z",
    }
    text = {**token, "ts": "2026-06-11T10:00:00.900Z"}
    assert lfmap.pair_key(token) == lfmap.pair_key(text)


def test_pair_key_degrades_when_turn_absent():
    """Rows without turn/task_id/dyn_id still produce a stable key rather than
    raising."""
    row = {"component": "forge", "tick": 1, "role": None, "ts": "2026-06-11T10:00:00Z"}
    k = lfmap.pair_key(row)
    assert lfmap.pair_key(dict(row)) == k


def _status(**over) -> dict:
    base = {"phase": "EXPLORE", "stop_reason": "", "cumulative_gain": 12.5}
    base.update(over)
    return base


def test_record_status_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    em = lfe.LangfuseEmitter(tmp_path)
    assert em.enabled is False
    em.record_status(_status())  # must not raise / must not build a client
    assert em._client is None


def test_record_status_emits_observation_and_trace_metadata(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd, claw_session_id="claw-XYZ")
    em = lfe.LangfuseEmitter(sd)

    em.record_status(_status(phase="PRELUDE"))

    # A session_status observation was appended.
    status_spans = [s for s in client.spans if s.kwargs.get("name") == "session_status"]
    assert len(status_spans) == 1
    assert status_spans[0].kwargs.get("output", {}).get("phase") == "PRELUDE"
    # Trace-level metadata was upserted.
    assert client.trace_updates, "expected update_trace to stamp trace metadata"
    assert client.trace_updates[-1]["metadata"]["phase"] == "PRELUDE"
    assert em._counts["status_updates_sent"] == 1


def test_record_status_throttles_unchanged_snapshots(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)

    em.record_status(_status(phase="EXPLORE"))
    em.record_status(_status(phase="EXPLORE"))  # throttled
    assert em._counts["status_updates_sent"] == 1

    em.record_status(_status(phase="SWEEP"))  # changed -> sent
    assert em._counts["status_updates_sent"] == 2


def test_record_status_refreshes_after_min_interval(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)

    # min_refresh_sec=0 makes even an unchanged snapshot re-send.
    em.record_status(_status(), min_refresh_sec=0.0)
    em.record_status(_status(), min_refresh_sec=0.0)
    assert em._counts["status_updates_sent"] == 2


def test_record_status_send_failure_is_swallowed(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    em = lfe.LangfuseEmitter(sd)

    def _boom(**kwargs):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(client, "start_observation", _boom)
    em.record_status(_status())  # must not raise
    assert em._counts["errors"] >= 1
    assert em._counts["status_updates_sent"] == 0
