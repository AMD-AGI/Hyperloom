# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SubstrateLeversClient + substrate_levers warmup + SUBSTRATE EVIDENCE section."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator import substrate_levers_client as slc
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.specialist_domains import get_domain
from inference_optimizer.orchestrator.substrate_levers_client import (
    SubstrateLeversClient,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_substrate_levers,
    build_specialist_prompts,
)


# ── client config + transport ────────────────────────────────────────
def test_from_env_returns_none_without_url(monkeypatch):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    assert SubstrateLeversClient.from_env() is None


def test_from_env_builds_client_with_url(monkeypatch):
    monkeypatch.setenv("CORTEX_KB_URL", "http://kb.svc:8080/")
    monkeypatch.setenv("CORTEX_KB_HTTP_TIMEOUT_SEC", "5")
    client = SubstrateLeversClient.from_env()
    assert client is not None
    assert client.base_url == "http://kb.svc:8080"  # trailing slash stripped
    assert client.timeout_sec == 5.0


class _FakeResp:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload.encode("utf-8")


def test_recommend_posts_focus_and_decodes(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResp(json.dumps({"focus": {"model": "m"}, "seed": "model:m",
                                     "summary": {"beneficial": 1}, "levers": []}))

    monkeypatch.setattr(slc.urllib.request, "urlopen", fake_urlopen)
    client = SubstrateLeversClient("http://kb.svc:8080")
    out = client.recommend(focus={"model": "m", "hardware": "mi300x"})
    assert out["seed"] == "model:m"
    assert captured["url"].endswith("/v2/reasoning/levers")
    assert captured["body"] == {"focus": {"model": "m", "hardware": "mi300x"}}


def test_recommend_without_model_returns_none(monkeypatch):
    def boom(*a, **k):  # must not be reached
        raise AssertionError("network attempted without model")

    monkeypatch.setattr(slc.urllib.request, "urlopen", boom)
    client = SubstrateLeversClient("http://kb.svc:8080")
    assert client.recommend(focus={"hardware": "mi300x"}) is None


def test_recommend_swallows_errors(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(slc.urllib.request, "urlopen", fake_urlopen)
    client = SubstrateLeversClient("http://kb.svc:8080")
    assert client.recommend(focus={"model": "m"}) is None


# ── coordinator warmup ───────────────────────────────────────────────
@dataclass
class _BareState:
    model_name: str = "qwen3-32b"
    gpu_type: str = "mi300x"
    framework: str = "sglang"
    precision: str = "fp8"
    tp: int = 0
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def find_gap(self, _cid: str):
        return None


def _make_coord(tmp_path: Path, *, state: _BareState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


class _StubClient:
    calls = 0

    def recommend(self, *, focus):
        type(self).calls += 1
        return {"focus": focus, "seed": "model:qwen3-32b",
                "summary": {"beneficial": 1, "neutral": 0, "harmful": 0, "calibrated": 1},
                "levers": [{"knob": "knob:kv_cache_fp8", "direction": "beneficial",
                            "confidence": 0.9, "calibrated": True, "verdict": "beneficial",
                            "confirmed": 5, "deviated": 0, "evidence_count": 5,
                            "scope": "struct_class"}]}


def test_warm_substrate_levers_injects_and_caches(tmp_path, monkeypatch):
    _StubClient.calls = 0
    monkeypatch.setattr(SubstrateLeversClient, "from_env", classmethod(lambda cls: _StubClient()))
    coord = _make_coord(tmp_path, state=_BareState())

    params: dict[str, Any] = {"domain": "serving_specialist"}
    digest = coord._warm_substrate_levers(params)
    assert digest["seed"] == "model:qwen3-32b"
    assert digest["levers"][0]["knob"] == "knob:kv_cache_fp8"

    # second dispatch with the same focus hits the per-focus cache (no 2nd call)
    coord._warm_substrate_levers({"domain": "serving_specialist"})
    assert _StubClient.calls == 1


def test_warm_substrate_levers_noop_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(SubstrateLeversClient, "from_env", classmethod(lambda cls: _StubClient()))
    coord = _make_coord(tmp_path, state=_BareState(model_name=""))
    assert coord._warm_substrate_levers({"domain": "serving_specialist"}) == {}


def test_warm_substrate_levers_noop_without_kb(tmp_path, monkeypatch):
    monkeypatch.setattr(SubstrateLeversClient, "from_env", classmethod(lambda cls: None))
    coord = _make_coord(tmp_path, state=_BareState())
    assert coord._warm_substrate_levers({"domain": "serving_specialist"}) == {}


# ── prompt rendering ─────────────────────────────────────────────────
def _make_inp(substrate_levers: dict[str, Any]) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        gap_canonical_id="gap.x",
        substrate_levers=substrate_levers,
    )


def test_section_renders_directional_table_sorted():
    inp = _make_inp({
        "seed": "model:qwen3-32b",
        "summary": {"beneficial": 1, "neutral": 1, "harmful": 1, "calibrated": 1},
        "levers": [
            {"knob": "knob:neutral_one", "direction": "neutral", "confidence": 0.6,
             "verdict": "neutral", "evidence_count": 3, "scope": "global"},
            {"knob": "knob:harmful_one", "direction": "harmful", "confidence": 0.8,
             "verdict": "harmful", "evidence_count": 4, "scope": "struct_class"},
            {"knob": "knob:good_one", "direction": "beneficial", "confidence": 0.95,
             "calibrated": True, "confirmed": 6, "deviated": 1, "scope": "struct_class"},
        ],
    })
    text = "\n".join(_section_substrate_levers(inp))
    assert "## 4b. SUBSTRATE EVIDENCE (advisory)" in text
    assert "model:qwen3-32b" in text
    assert "beneficial=1" in text and "calibrated=1" in text
    # beneficial sorts before harmful before neutral
    assert text.index("knob:good_one") < text.index("knob:harmful_one") < text.index("knob:neutral_one")
    # calibrated evidence renders as +confirmed/-deviated; global tagged
    assert "+6/-1" in text
    assert "(global)" in text


def test_section_renders_none_when_empty():
    text = "\n".join(_section_substrate_levers(_make_inp({})))
    assert "## 4b. SUBSTRATE EVIDENCE (advisory)" in text
    assert "(none" in text


def test_build_inserts_substrate_section_between_roofline_and_recipe():
    inp = _make_inp({
        "seed": "model:m", "summary": {"beneficial": 1},
        "levers": [{"knob": "knob:x", "direction": "beneficial", "confidence": 0.7}],
    })
    _system, user = build_specialist_prompts(inp)
    roof_idx = user.index("## 4a. ROOFLINE EVIDENCE")
    sub_idx = user.index("## 4b. SUBSTRATE EVIDENCE (advisory)")
    recipe_idx = user.index("## 5. WARM-START RECIPE SUMMARY")
    assert roof_idx < sub_idx < recipe_idx
