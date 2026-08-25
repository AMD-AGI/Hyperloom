# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplementary coverage for ProposalScorer helpers + client construction."""

from __future__ import annotations

import asyncio
import sys

import pytest

from hyperloom.orchestrator.scoring import proposal_scorer as proposal_scorer_module
from hyperloom.orchestrator.scoring.proposal_scorer import (
    ProposalScorer,
    _ScoringProposal,
    _clip,
    _coerce_score,
    _extract_scores_json,
    _normalise_model_scores,
    _prepare_scoring_proposals,
)


# ---- _extract_scores_json edges ----
def test_extract_scores_json_empty():
    assert _extract_scores_json("") is None


def test_extract_scores_json_bare_with_trailing():
    # bare object with trailing prose -> shrink loop trims to valid JSON.
    text = 'here are scores {"scores": {"proposal_0": {"score": 1, "reason": "x"}}} thanks'
    out = _extract_scores_json(text)
    assert out["scores"]["proposal_0"]["score"] == 1


def test_extract_scores_json_prefers_last_envelope():
    text = (
        'draft {"scores": {"proposal_0": {"score": 1, "reason": "old"}}} '
        'final {"scores": {"proposal_0": {"score": 8, "reason": "new"}}}'
    )
    out = _extract_scores_json(text)
    assert out["scores"]["proposal_0"]["score"] == 8


def test_extract_scores_json_wrong_shape():
    # parses but no "scores" key -> None
    assert _extract_scores_json('{"foo": 1, "scores_x": 2}') is None


# ---- _coerce_score / _clip ----
def test_coerce_score_edges():
    assert _coerce_score("nan") is None
    assert _coerce_score("bad") is None
    assert _coerce_score(True) is None
    assert _coerce_score(False) is None
    assert _coerce_score(99) == 10.0
    assert _coerce_score(-5) == 0.0


def test_clip_truncates():
    assert _clip(None) == ""
    assert _clip("x" * 10, limit=3) == "xxx…"


# ---- _normalise_model_scores ----
def test_normalise_scores_not_dict():
    assert _normalise_model_scores({"scores": "x"}, scoring_entries=[]) == {}


def test_normalise_drops_unknown_and_bad_score():
    entries = [
        _ScoringProposal(
            stable_id="proposal_0",
            output_name="a",
            label_name="a",
            proposal={"name": "a"},
        ),
        _ScoringProposal(
            stable_id="proposal_1",
            output_name="b",
            label_name="b",
            proposal={"name": "b"},
        ),
    ]
    parsed = {
        "scores": {
            "proposal_0": {"score": 5, "reason": "ok"},
            "ghost": {"score": 3},
            "proposal_1": {"score": "x"},
        }
    }
    out = _normalise_model_scores(parsed, scoring_entries=entries)
    assert "a" in out
    assert "ghost" not in out
    assert "b" not in out


def test_prepare_scoring_proposals_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate proposal name"):
        _prepare_scoring_proposals(
            [
                {"name": "same"},
                {"name": "same"},
            ]
        )


# ---- _ensure_client ----
def test_ensure_client_no_api_key(monkeypatch):
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "LLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    scorer = ProposalScorer(models=("m",))
    with pytest.raises(RuntimeError):
        scorer._ensure_client()


def test_ensure_client_builds_with_key(monkeypatch):
    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "tok")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy")
    sentinel = object()
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: sentinel)
    scorer = ProposalScorer(models=("m",))
    assert scorer._ensure_client() is sentinel
    # cached on second call
    assert scorer._ensure_client() is sentinel


def test_ensure_client_prefers_explicit_openai_key(monkeypatch):
    """Explicit OPENAI_API_KEY wins over SAFE-filled ANTHROPIC_AUTH_TOKEN."""
    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "safe-filled")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy")
    captured: dict = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())
    scorer = ProposalScorer(models=("m",))
    scorer._ensure_client()
    assert captured["api_key"] == "openai-user-key"


def test_ensure_client_works_in_an_anthropic_only_deployment(monkeypatch):
    """The scorer used to raise here, which disabled scoring for Anthropic-only users."""
    import openai

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-api.amd.com/Anthropic")
    captured: dict = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())
    scorer = ProposalScorer(models=("m",))
    assert scorer._ensure_client() is not None
    assert captured["api_key"] == "anthropic-token"
    assert captured["base_url"] == "https://llm-api.amd.com/Unified/v1"


def test_ensure_client_comes_from_the_shared_llm_gateway(monkeypatch):
    """The scorer owns no credentials: it forwards its env-var contract to llm_config."""
    captured: dict = {}
    sentinel = object()
    monkeypatch.setattr(
        proposal_scorer_module,
        "get_async_openai_client",
        lambda **kwargs: captured.update(kwargs) or sentinel,
    )
    scorer = ProposalScorer(models=("m",), api_key_env="SCORER_KEY", base_url_env="SCORER_URL")
    assert scorer._ensure_client() is sentinel
    assert captured == {"api_key_env": "SCORER_KEY", "base_url_env": "SCORER_URL"}


def test_ensure_client_without_openai_sdk_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "tok")
    monkeypatch.setitem(sys.modules, "openai", None)
    scorer = ProposalScorer(models=("m",))
    with pytest.raises(RuntimeError, match="openai SDK not installed"):
        scorer._ensure_client()


# ---- score: proposal cap + timeout + no-usable ----
class _FakeCompletions:
    def __init__(self, behaviour):
        self._b = behaviour
        self.calls = []
        self.params = []

    async def create(self, **params):
        model = params["model"]
        self.calls.append(model)
        self.params.append(params)
        r = self._b.get(model)
        if isinstance(r, BaseException):
            raise r

        text = r or ""

        class _Delta:
            content = text

        class _Choice:
            delta = _Delta()

        class _Chunk:
            choices = [_Choice()]
            usage = None

        class _Stream:
            def __aiter__(self):
                self._done = False
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                return _Chunk()

        return _Stream()


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
    # one model call; prompt built from capped list (<=16)
    assert scorer._client.chat.completions.calls == ["m"]


@pytest.mark.asyncio
async def test_score_streams_with_usage_and_keeps_the_token_cap():
    """The shared streaming helper adds stream + include_usage; the 4096 cap survives."""
    client = _FakeClient({"m": '{"scores": {"proposal_0": {"score": 5, "reason": "ok"}}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    out = await scorer.score(gap={"domain": "d"}, proposals=[{"name": "p"}])
    assert out["models"]["m"]["p"]["score"] == 5.0
    params = client.chat.completions.params[0]
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
    assert params["max_completion_tokens"] == 4096


@pytest.mark.asyncio
async def test_score_timeout_recorded_as_error():
    client = _FakeClient({"m": asyncio.TimeoutError()})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    out = await scorer.score(gap={"domain": "d"}, proposals=[{"name": "p"}])
    assert "m" in out["errors"]
    assert "timed out" in out["errors"]["m"]


@pytest.mark.asyncio
async def test_score_no_usable_scores():
    # valid JSON but only unknown names -> empty after normalise
    client = _FakeClient({"m": '{"scores": {"ghost": {"score": 5, "reason": "x"}}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    out = await scorer.score(gap={"domain": "d"}, proposals=[{"name": "p"}])
    assert out["models"] == {}
    assert out["errors"]["m"] == "no usable scores returned"


@pytest.mark.asyncio
async def test_score_duplicate_names_recorded_as_input_error():
    client = _FakeClient({"m": '{"scores": {}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    out = await scorer.score(
        gap={"domain": "d"},
        proposals=[{"name": "dup"}, {"name": "dup"}],
    )
    assert out["models"] == {}
    assert "duplicate proposal name" in out["errors"]["input"]


@pytest.mark.asyncio
async def test_score_duplicate_model_slugs_rejected_at_construction():
    with pytest.raises(ValueError, match="duplicate scorer model slug"):
        ProposalScorer(models=("m", "m"), client_factory=lambda: object())


@pytest.mark.asyncio
async def test_build_prompt_includes_evidence():
    client = _FakeClient({"m": '{"scores": {}}'})
    scorer = ProposalScorer(models=("m",), client_factory=lambda: client)
    gap = {"domain": "d", "gap_evidence": {"e": 1}, "gap_symptom": "s"}
    entries = _prepare_scoring_proposals(
        [
            {"name": "p", "extra_args": "--x", "extra_envs": {"E": "1"}, "reason": "r", "kb_evidence": ["k"]},
        ]
    )
    prompt = scorer._build_prompt(gap=gap, scoring_entries=entries)
    assert "evidence:" in prompt
    assert "extra_envs:" in prompt
    assert "kb_evidence:" in prompt
    assert "id: proposal_0" in prompt
