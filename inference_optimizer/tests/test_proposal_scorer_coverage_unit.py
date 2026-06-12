# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplementary coverage for ProposalScorer helpers + client construction."""

from __future__ import annotations

import asyncio

import pytest

from inference_optimizer.orchestrator import proposal_scorer as ps
from inference_optimizer.orchestrator.proposal_scorer import (
    ProposalScorer,
    _clip,
    _coerce_score,
    _extract_scores_json,
    _normalise_model_scores,
)


# ---- _extract_scores_json edges ----
def test_extract_scores_json_empty():
    assert _extract_scores_json("") is None


def test_extract_scores_json_bare_with_trailing():
    # bare object followed by trailing prose -> shrink loop trims to valid JSON.
    text = 'here are scores {"scores": {"a": {"score": 1, "reason": "x"}}} thanks'
    out = _extract_scores_json(text)
    assert out["scores"]["a"]["score"] == 1


def test_extract_scores_json_wrong_shape():
    # parses but no "scores" key -> None
    assert _extract_scores_json('{"foo": 1, "scores_x": 2}') is None


# ---- _coerce_score / _clip ----
def test_coerce_score_edges():
    assert _coerce_score("nan") is None
    assert _coerce_score("bad") is None
    assert _coerce_score(99) == 10.0
    assert _coerce_score(-5) == 0.0


def test_clip_truncates():
    assert _clip(None) == ""
    assert _clip("x" * 10, limit=3) == "xxx…"


# ---- _normalise_model_scores ----
def test_normalise_scores_not_dict():
    assert _normalise_model_scores({"scores": "x"}, proposal_names=["a"]) == {}


def test_normalise_drops_unknown_and_bad_score():
    parsed = {"scores": {
        "a": {"score": 5, "reason": "ok"},
        "ghost": {"score": 3},      # unknown name -> dropped
        "b": {"score": "x"},        # bad score -> dropped
    }}
    out = _normalise_model_scores(parsed, proposal_names=["a", "b"])
    assert "a" in out
    assert "ghost" not in out
    assert "b" not in out


# ---- _ensure_client ----
def test_ensure_client_no_api_key(monkeypatch):
    for var in ("ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    scorer = ProposalScorer(models=("m",))
    with pytest.raises(RuntimeError):
        scorer._ensure_client()


def test_ensure_client_builds_with_key(monkeypatch):
    import openai

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy")
    sentinel = object()
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: sentinel)
    scorer = ProposalScorer(models=("m",))
    assert scorer._ensure_client() is sentinel
    # cached on second call
    assert scorer._ensure_client() is sentinel


# ---- score: proposal cap + timeout + no-usable ----
class _FakeCompletions:
    def __init__(self, behaviour):
        self._b = behaviour
        self.calls = []

    async def create(self, *, model, messages, max_completion_tokens):
        self.calls.append(model)
        r = self._b.get(model)
        if isinstance(r, BaseException):
            raise r

        class _Msg:
            content = r or ""

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        return _Resp()


class _FakeClient:
    def __init__(self, behaviour):
        class _Chat:
            completions = _FakeCompletions(behaviour)
        self.chat = _Chat()


@pytest.mark.asyncio
async def test_score_caps_proposals():
    client = _FakeClient({"m": '{"scores": {}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    proposals = [{"name": f"p{i}"} for i in range(30)]
    await scorer.score(gap={"domain": "d"}, proposals=proposals)
    # only one model call, but prompt built from capped list (<=16)
    assert scorer._client.chat.completions.calls == ["m"]


@pytest.mark.asyncio
async def test_score_timeout_recorded_as_error():
    client = _FakeClient({"m": asyncio.TimeoutError()})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    out = await scorer.score(gap={"domain": "d"}, proposals=[{"name": "p"}])
    assert "m" in out["errors"]
    assert "timed out" in out["errors"]["m"]


@pytest.mark.asyncio
async def test_score_no_usable_scores():
    # model returns valid JSON but only unknown names -> empty after normalise
    client = _FakeClient({"m": '{"scores": {"ghost": {"score": 5, "reason": "x"}}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    out = await scorer.score(gap={"domain": "d"}, proposals=[{"name": "p"}])
    assert out["models"] == {}
    assert out["errors"]["m"] == "no usable scores returned"


@pytest.mark.asyncio
async def test_build_prompt_includes_evidence():
    client = _FakeClient({"m": '{"scores": {}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    gap = {"domain": "d", "gap_evidence": {"e": 1}, "gap_symptom": "s"}
    prompt = scorer._build_prompt(gap=gap, proposals=[
        {"name": "p", "extra_args": "--x", "extra_envs": {"E": "1"},
         "reason": "r", "kb_evidence": ["k"]},
    ])
    assert "evidence:" in prompt
    assert "extra_envs:" in prompt
    assert "kb_evidence:" in prompt
