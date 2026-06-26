# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Substrate↔gbrain dual-read: assess client, comparator, warmup + tracing.

Covers the deterministic cross-check that folds the cortex substrate's
directional levers and the gbrain warm-start recipe into one digest, the
Coordinator conflict guardrail, and the SpecialistRunner dual-read trace +
prompt section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator import substrate_dual_read as sdr
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.specialist_runner import SpecialistRunner
from inference_optimizer.orchestrator.specialist_domains import get_domain
from inference_optimizer.orchestrator.substrate_dual_read import (
    SubstrateAssessClient,
    build_dual_read,
    compute_dual_read,
    focus_from,
    recipe_assess_inputs,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_substrate_dual_read,
    build_specialist_prompts,
)


# ── shared fixtures ──────────────────────────────────────────────────
_LEVERS = {
    "focus": {"model": "qwen3-32b", "hardware": "mi300x"},
    "seed": "model:qwen3-32b",
    "summary": {"beneficial": 1, "neutral": 0, "harmful": 0, "calibrated": 1},
    "levers": [{"knob": "knob:kv_cache_fp8", "direction": "beneficial",
                "confidence": 0.9, "calibrated": True, "verdict": "beneficial",
                "confirmed": 5, "deviated": 0, "evidence_count": 5, "scope": "struct_class"}],
}

_RECIPE = {
    "tier": "exact",
    "confidence": 0.85,
    "recipe": {
        "canonical_id": "inference:qwen3-32b:mi300x:sglang:v1:fp8",
        "best_config": {
            "extra_server_args": "--kv-cache-dtype fp8 --quantization fp8",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        },
    },
}

_ASSESS_CONFLICT = {
    "focus": {"model": "qwen3-32b", "hardware": "mi300x"},
    "seed": "model:qwen3-32b",
    "reasonable": "contested",
    "rationale": "1 lever(s) contradict KB evidence",
    "summary": {"deviated": 1, "confirmed": 1},
    "verdicts": [
        {"lever": "kv_cache_dtype=fp8", "knob": "knob:kv_cache_fp8", "polarity": "enable",
         "status": "confirmed", "predicted_factor": 0.5, "note": "calibrated"},
        {"lever": "VLLM_ROCM_USE_AITER=1", "knob": "knob:aiter", "polarity": "enable",
         "status": "deviated", "predicted_factor": 1.1, "note": "measured harmful"},
    ],
}

_ASSESS_SUPPORTED = {
    "focus": {"model": "qwen3-32b", "hardware": "mi300x"},
    "seed": "model:qwen3-32b",
    "reasonable": "supported",
    "rationale": "2 lever(s) backed by KB evidence",
    "summary": {"confirmed": 2},
    "verdicts": [
        {"lever": "kv_cache_dtype=fp8", "knob": "knob:kv_cache_fp8", "polarity": "enable",
         "status": "confirmed", "note": "calibrated"},
        {"lever": "VLLM_ROCM_USE_AITER=1", "knob": "knob:aiter", "polarity": "enable",
         "status": "confirmed", "note": "calibrated"},
    ],
}


class _StubAssess:
    """Records assess() inputs and returns a canned assessment."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def assess(self, *, focus, params=None, envs=None, args=""):
        self.calls.append({"focus": focus, "params": params, "envs": envs, "args": args})
        return self.result


# ── assess client transport ──────────────────────────────────────────
def test_assess_from_env_none_without_url(monkeypatch):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    assert SubstrateAssessClient.from_env() is None


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload.encode("utf-8")


def test_assess_posts_proposal_and_decodes(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(json.dumps(_ASSESS_SUPPORTED))

    monkeypatch.setattr(sdr.urllib.request, "urlopen", fake_urlopen)
    client = SubstrateAssessClient("http://kb.svc:8080/")
    out = client.assess(focus={"model": "qwen3-32b"}, envs={"X": "1"}, args="--y")
    assert out["reasonable"] == "supported"
    assert captured["url"].endswith("/v2/reasoning/assess")
    assert captured["body"]["focus"] == {"model": "qwen3-32b"}
    assert captured["body"]["envs"] == {"X": "1"}
    assert captured["body"]["args"] == "--y"


def test_assess_without_model_returns_none(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network attempted without model")

    monkeypatch.setattr(sdr.urllib.request, "urlopen", boom)
    assert SubstrateAssessClient("http://kb.svc:8080").assess(focus={"hardware": "x"}) is None


def test_assess_swallows_errors(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(sdr.urllib.request, "urlopen", fake_urlopen)
    assert SubstrateAssessClient("http://kb.svc:8080").assess(focus={"model": "m"}) is None


# ── helpers ──────────────────────────────────────────────────────────
def test_recipe_assess_inputs_projects_best_config():
    inputs = recipe_assess_inputs(_RECIPE)
    assert inputs["args"] == "--kv-cache-dtype fp8 --quantization fp8"
    assert inputs["envs"] == {"VLLM_ROCM_USE_AITER": "1"}


def test_recipe_assess_inputs_empty_without_recipe():
    assert recipe_assess_inputs({}) == {}
    assert recipe_assess_inputs({"recipe": {}}) == {}


def test_focus_prefers_levers_then_recipe():
    assert focus_from(_LEVERS, None)["model"] == "qwen3-32b"
    assert focus_from({}, {"recipe": {"model": "m2", "hw": "h2"}}) == {"model": "m2", "hardware": "h2"}
    assert focus_from({}, {}) == {}


# ── comparator (pure) ────────────────────────────────────────────────
def test_build_dual_read_conflict():
    dig = build_dual_read(substrate_levers=_LEVERS, warm_start_recipe=_RECIPE, assess=_ASSESS_CONFLICT)
    assert dig["verdict"] == "conflict"
    assert dig["selected_source"] == "substrate"
    assert len(dig["conflicts"]) == 1
    assert dig["conflicts"][0]["knob"] == "knob:aiter"
    assert len(dig["confirmations"]) == 1


def test_build_dual_read_agree():
    dig = build_dual_read(substrate_levers=_LEVERS, warm_start_recipe=_RECIPE, assess=_ASSESS_SUPPORTED)
    assert dig["verdict"] == "agree"
    assert dig["selected_source"] == "both_agree"
    assert not dig["conflicts"]
    assert len(dig["confirmations"]) == 2


def test_build_dual_read_gbrain_only_when_no_substrate():
    dig = build_dual_read(
        substrate_levers={}, warm_start_recipe=_RECIPE,
        assess={"reasonable": "insufficient_basis", "verdicts": []},
    )
    assert dig["verdict"] == "gbrain_only"
    assert dig["selected_source"] == "gbrain"


def test_build_dual_read_substrate_only_when_no_recipe():
    dig = build_dual_read(substrate_levers=_LEVERS, warm_start_recipe={}, assess=None)
    assert dig["verdict"] == "substrate_only"
    assert dig["selected_source"] == "substrate"


def test_build_dual_read_no_data():
    dig = build_dual_read(substrate_levers={}, warm_start_recipe={}, assess=None)
    assert dig["verdict"] == "no_data"
    assert dig["selected_source"] == "none"


def test_compute_dual_read_uses_client_and_inputs():
    stub = _StubAssess(_ASSESS_CONFLICT)
    dig = compute_dual_read(substrate_levers=_LEVERS, warm_start_recipe=_RECIPE, client=stub)
    assert dig["verdict"] == "conflict"
    # the recipe's best_config was projected into the assess call
    assert stub.calls[0]["envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert "--kv-cache-dtype fp8" in stub.calls[0]["args"]
    assert stub.calls[0]["focus"]["model"] == "qwen3-32b"


def test_compute_dual_read_failsoft_on_assess_none():
    # assess returns None (call failed) but levers + recipe still present →
    # degrades to a non-empty digest, no_basis (not a crash, not no_data).
    stub = _StubAssess(None)
    dig = compute_dual_read(substrate_levers=_LEVERS, warm_start_recipe=_RECIPE, client=stub)
    assert dig  # non-empty
    assert dig["verdict"] in ("no_basis", "gbrain_only")


def test_compute_dual_read_empty_without_signal():
    assert compute_dual_read(substrate_levers={}, warm_start_recipe={}, client=_StubAssess(None)) == {}


# ── coordinator conflict guardrail ───────────────────────────────────
@dataclass
class _BareState:
    model_name: str = "qwen3-32b"
    gpu_type: str = "mi300x"
    framework: str = "sglang"
    precision: str = "fp8"


class _FakeEmitter:
    def __init__(self):
        self.enabled = True
        self.spans: list[dict[str, Any]] = []

    def record_kb_span(self, *, name, agent, output, phase=None, metadata=None, ts=None):
        self.spans.append({"name": name, "agent": agent, "output": output, "metadata": metadata})


def _make_coord(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.knowledge_plane = None
    return c


def test_warm_dual_read_injects_caches_and_traces_guardrail(tmp_path, monkeypatch):
    stub = _StubAssess(_ASSESS_CONFLICT)
    monkeypatch.setattr(SubstrateAssessClient, "from_env", classmethod(lambda cls: stub))
    fake = _FakeEmitter()
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_emitter.get_emitter",
        lambda _sd: fake,
    )
    coord = _make_coord(tmp_path)

    params = {"domain": "serving_specialist", "substrate_levers": _LEVERS, "warm_start_recipe": _RECIPE}
    dig = coord._warm_substrate_dual_read(params)
    assert dig["verdict"] == "conflict"

    # guardrail span emitted under the orchestrator agent, flagging the conflict
    assert len(fake.spans) == 1
    span = fake.spans[0]
    assert span["name"] == "kb_guardrail:substrate_vs_recipe"
    assert span["agent"] == "orchestrator"
    assert span["metadata"]["conflict"] is True
    assert span["metadata"]["conflict_count"] == 1

    # second dispatch with same focus hits cache (no 2nd assess call, no 2nd span)
    coord._warm_substrate_dual_read(dict(params))
    assert len(stub.calls) == 1
    assert len(fake.spans) == 1


def test_warm_dual_read_noop_without_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(SubstrateAssessClient, "from_env", classmethod(lambda cls: _StubAssess(None)))
    coord = _make_coord(tmp_path)
    assert coord._warm_substrate_dual_read({"domain": "serving_specialist"}) == {}


# ── specialist runner trace ──────────────────────────────────────────
def test_specialist_compute_and_trace_dual_read(tmp_path, monkeypatch):
    stub = _StubAssess(_ASSESS_CONFLICT)
    monkeypatch.setattr(SubstrateAssessClient, "from_env", classmethod(lambda cls: stub))
    fake = _FakeEmitter()
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_emitter.get_emitter",
        lambda _sd: fake,
    )
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.session_dir = tmp_path

    dig = runner._compute_substrate_dual_read(
        {"substrate_levers": _LEVERS, "warm_start_recipe": _RECIPE}
    )
    assert dig["verdict"] == "conflict"

    runner._trace_substrate_dual_read("task-7", dig)
    assert len(fake.spans) == 1
    span = fake.spans[0]
    assert span["name"] == "kb_dual_read:task-7"
    assert span["agent"] == "specialist"
    assert span["metadata"]["verdict"] == "conflict"
    assert span["metadata"]["conflict_count"] == 1


def test_specialist_trace_noop_when_emitter_disabled(tmp_path, monkeypatch):
    class _Off:
        enabled = False

        def record_kb_span(self, **k):
            raise AssertionError("must not emit when disabled")

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_emitter.get_emitter",
        lambda _sd: _Off(),
    )
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.session_dir = tmp_path
    runner._trace_substrate_dual_read("t", {"verdict": "conflict", "conflicts": []})


# ── prompt rendering ─────────────────────────────────────────────────
def _make_inp(dual: dict[str, Any]) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        gap_canonical_id="gap.x",
        substrate_dual_read=dual,
    )


def test_section_renders_conflict_table():
    dig = build_dual_read(substrate_levers=_LEVERS, warm_start_recipe=_RECIPE, assess=_ASSESS_CONFLICT)
    text = "\n".join(_section_substrate_dual_read(_make_inp(dig)))
    assert "## 4c. SUBSTRATE × WARM-RECIPE CROSS-CHECK (advisory)" in text
    assert "CONFLICTS" in text
    assert "knob:aiter" in text
    assert "deviated" in text


def test_section_omitted_for_no_data_or_substrate_only():
    assert _section_substrate_dual_read(_make_inp({})) == []
    assert _section_substrate_dual_read(_make_inp({"verdict": "no_data"})) == []
    assert _section_substrate_dual_read(_make_inp({"verdict": "substrate_only"})) == []


def test_build_inserts_dual_read_after_recipe():
    dig = build_dual_read(substrate_levers=_LEVERS, warm_start_recipe=_RECIPE, assess=_ASSESS_SUPPORTED)
    inp = SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        gap_canonical_id="gap.x",
        substrate_levers=_LEVERS,
        warm_start_recipe=_RECIPE,
        substrate_dual_read=dig,
    )
    _system, user = build_specialist_prompts(inp)
    recipe_idx = user.index("## 5. WARM-START RECIPE SUMMARY")
    cross_idx = user.index("## 4c. SUBSTRATE × WARM-RECIPE CROSS-CHECK (advisory)")
    assert recipe_idx < cross_idx
