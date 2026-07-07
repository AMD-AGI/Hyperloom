# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the advisory specialist-proposal scorer (ProposalScorer)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.scoring.proposal_scorer import (
    DEFAULT_SCORER_MODELS,
    ProposalScorer,
    _extract_scores_json,
)
from hyperloom.orchestrator.policy.gate import SPECIALIST_FROM_AGENT_PREFIX
from inference_optimizer.session_paths import (
    conversations_path,
    llm_calls_path,
)


# Fake OpenAI client (test seam)
@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeDelta:
    content: str | None


@dataclass
class _FakeStreamChoice:
    delta: _FakeDelta


@dataclass
class _FakeChunk:
    choices: list[_FakeStreamChoice]
    usage: _FakeUsage | None = None


class _FakeStream:
    """Async iterator emulating an OpenAI streaming response: content deltas
    followed by a final usage-only chunk (``include_usage``)."""

    def __init__(self, text: str, usage: _FakeUsage):
        self._chunks = [
            _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(text))]),
            _FakeChunk(choices=[], usage=usage),
        ]

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeCompletions:
    """Per-model scripted behaviour keyed by ``model=`` (string reply or Exception)."""

    def __init__(self, behaviour: dict[str, Any]):
        self._behaviour = behaviour
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model: str, messages, max_completion_tokens, stream=False, stream_options=None):
        self.calls.append({"model": model, "messages": messages, "stream": stream})
        result = self._behaviour.get(model)
        if isinstance(result, BaseException):
            raise result
        return _FakeStream(
            result or "",
            _FakeUsage(prompt_tokens=120, completion_tokens=30),
        )


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    def __init__(self, behaviour: dict[str, Any]):
        self.chat = _FakeChat(_FakeCompletions(behaviour))


def _make_scorer(behaviour: dict[str, Any], *, models=None) -> ProposalScorer:
    client = _FakeClient(behaviour)
    return ProposalScorer(
        models=tuple(models or behaviour.keys()),
        client_factory=lambda: client,
    )


_GAP = {
    "domain": "serving_specialist",
    "gap_canonical_id": "gap.framework.cuda_graph.session-1",
    "gap_symptom": "cuda graph capture stalls at high concurrency",
    "summary": "specialist explored cuda graph bracket",
}

_PROPOSALS = [
    {
        "name": "cuda_graph_bs_512",
        "extra_args": "--cuda-graph-max-bs 512",
        "extra_envs": {"FOO": "1"},
        "reason": "bracket the live CONC",
        "kb_evidence": ["kb-123"],
    },
    {
        "name": "disable_radix",
        "extra_args": "--disable-radix-cache",
        "reason": "reduce scheduler overhead",
    },
]


def _scores_json(*pairs: tuple[str, float, str]) -> str:
    body = ", ".join(f'"{name}": {{"score": {score}, "reason": "{reason}"}}' for name, score, reason in pairs)
    return f'```json\n{{"scores": {{{body}}}}}\n```'


# 1. Scorer unit
@pytest.mark.asyncio
async def test_score_two_models_happy_path():
    behaviour = {
        "claude-opus-4-7": _scores_json(
            ("cuda_graph_bs_512", 8.0, "strong fit"),
            ("disable_radix", 4.0, "marginal"),
        ),
        "gpt-5.4": _scores_json(
            ("cuda_graph_bs_512", 6.5, "plausible"),
            ("disable_radix", 5.0, "ok"),
        ),
    }
    scorer = _make_scorer(behaviour)
    out = await scorer.score(gap=_GAP, proposals=_PROPOSALS)

    assert out["scale"] == "0-10"
    assert set(out["models"]) == {"claude-opus-4-7", "gpt-5.4"}
    assert out["models"]["claude-opus-4-7"]["cuda_graph_bs_512"]["score"] == 8.0
    assert out["models"]["gpt-5.4"]["disable_radix"]["score"] == 5.0
    assert out["errors"] == {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_scoring_call_writes_full_conversation_trace(tmp_path: Path):
    """With ``session_dir`` set, each scoring call records both a token row
    (component=proposal_scorer) and a full prompt/reply conversation row."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    client = _FakeClient(
        {
            "claude-opus-4-7": _scores_json(("cuda_graph_bs_512", 8.0, "fit")),
        }
    )
    scorer = ProposalScorer(
        models=("claude-opus-4-7",),
        client_factory=lambda: client,
        session_dir=session_dir,
    )
    await scorer.score(gap=_GAP, proposals=_PROPOSALS, task_id="spec-7")

    conv_rows = _read_jsonl(conversations_path(session_dir))
    assert len(conv_rows) == 1
    row = conv_rows[0]
    assert row["component"] == "proposal_scorer"
    assert row["role"] == "proposal_scorer"
    assert row["model"] == "claude-opus-4-7"
    # task_id threads onto the conversation row so its pair key matches the
    # token row (and the collector can attribute the scoring spend).
    assert row["task_id"] == "spec-7"
    # The prompt carries the scoring instructions + proposals; the reply is
    # the model's verbatim (fenced) scores JSON.
    assert "cuda_graph_bs_512" in row["prompt"]
    assert "cuda_graph_bs_512" in row["response"]

    token_rows = _read_jsonl(llm_calls_path(session_dir))
    scorer_rows = [r for r in token_rows if r["component"] == "proposal_scorer"]
    assert scorer_rows
    r = scorer_rows[0]
    assert r["input_tokens"] == 120 and r["output_tokens"] == 30
    # Attribution key + measured latency now land on the token row.
    assert r["task_id"] == "spec-7"
    assert r["latency_ms"] is not None and r["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_scoring_without_session_dir_writes_no_trace(tmp_path: Path):
    """The default (tests / no full-trace) path writes no conversation file."""
    scorer = _make_scorer({"m1": _scores_json(("cuda_graph_bs_512", 7.0, "x"))})
    await scorer.score(gap=_GAP, proposals=_PROPOSALS)
    # session_dir is None → nothing on disk (and no crash).
    assert scorer.session_dir is None


@pytest.mark.asyncio
async def test_group_prompt_contains_all_proposals_and_gap():
    behaviour = {"m1": _scores_json(("cuda_graph_bs_512", 7.0, "x"))}
    scorer = _make_scorer(behaviour)
    await scorer.score(gap=_GAP, proposals=_PROPOSALS)
    client = scorer._client
    sent = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "cuda_graph_bs_512" in sent
    assert "disable_radix" in sent
    assert "cuda graph capture stalls" in sent
    assert "0 to 10" in sent


@pytest.mark.asyncio
async def test_per_model_degrade_one_raises_other_survives():
    behaviour = {
        "good": _scores_json(("cuda_graph_bs_512", 9.0, "great")),
        "bad": RuntimeError("gateway 404 unknown model"),
    }
    scorer = _make_scorer(behaviour)
    out = await scorer.score(gap=_GAP, proposals=_PROPOSALS)
    assert "good" in out["models"]
    assert out["models"]["good"]["cuda_graph_bs_512"]["score"] == 9.0
    assert "bad" in out["errors"]
    assert "404" in out["errors"]["bad"]


@pytest.mark.asyncio
async def test_unparseable_reply_recorded_as_error():
    behaviour = {"m1": "I cannot produce JSON, sorry."}
    scorer = _make_scorer(behaviour)
    out = await scorer.score(gap=_GAP, proposals=_PROPOSALS)
    assert out["models"] == {}
    assert "m1" in out["errors"]


@pytest.mark.asyncio
async def test_scores_clamped_and_unknown_names_dropped():
    behaviour = {
        "m1": _scores_json(
            ("cuda_graph_bs_512", 99.0, "over"),  # clamp to 10
            ("disable_radix", -3.0, "under"),  # clamp to 0
            ("ghost_variant", 5.0, "not in set"),  # dropped
        ),
    }
    scorer = _make_scorer(behaviour)
    out = await scorer.score(gap=_GAP, proposals=_PROPOSALS)
    m = out["models"]["m1"]
    assert m["cuda_graph_bs_512"]["score"] == 10.0
    assert m["disable_radix"]["score"] == 0.0
    assert "ghost_variant" not in m


@pytest.mark.asyncio
async def test_empty_proposals_returns_empty_envelope():
    scorer = _make_scorer({"m1": "irrelevant"})
    out = await scorer.score(gap=_GAP, proposals=[])
    assert out["models"] == {}
    assert out["errors"] == {}
    assert scorer._client.chat.completions.calls == []


def test_extract_scores_json_bare_and_fenced():
    fenced = _scores_json(("a", 1.0, "x"))
    assert _extract_scores_json(fenced)["scores"]["a"]["score"] == 1.0
    bare = '{"scores": {"a": {"score": 2, "reason": "y"}}}'
    assert _extract_scores_json(bare)["scores"]["a"]["score"] == 2
    assert _extract_scores_json("no json here") is None


def test_default_models_constant():
    assert DEFAULT_SCORER_MODELS == (
        "claude-opus-4-8",
        "gpt-5.5",
        "dvue-aoai-005-Kimi-K2.6",
        "gemini/gemini-3.1-pro-preview",
    )


# 2. Coordinator wiring
@dataclass
class _StubTask:
    task_id: str
    kind: str = "specialist"
    params: dict[str, Any] = field(default_factory=dict)


class _StubSharedState:
    def __init__(self):
        self.specialist_rounds: list[dict[str, Any]] = []
        self.specialist_domain_empty_streak: dict[str, int] = {}
        self.last_specialist: dict[str, Any] = {}
        self.saved: int = 0

    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        round_id = str(entry.get("round_id") or "").strip()
        if round_id:
            for i, prev in enumerate(self.specialist_rounds):
                if str(prev.get("round_id") or "") == round_id:
                    self.specialist_rounds[i] = dict(entry)
                    return
        self.specialist_rounds.append(dict(entry))

    def bump_specialist_domain_empty_streak(self, domain, *, empty) -> int:
        d = domain or "unknown"
        self.specialist_domain_empty_streak[d] = 0 if not empty else self.specialist_domain_empty_streak.get(d, 0) + 1
        return self.specialist_domain_empty_streak[d]

    def update_last_specialist(self, snapshot) -> None:
        self.last_specialist = dict(snapshot)

    def save(self, _sd) -> None:
        self.saved += 1


def _coord(tmp_path: Path, scorer):
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _StubSharedState()
    c._proposal_scorer = scorer
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    return c


def _done():
    return {
        "domain": "serving_specialist",
        "gap_canonical_id": "gap.framework.cuda_graph.session-1",
        "proposal_set": list(_PROPOSALS),
        "empty": False,
        "summary": "explored",
        "reason": "kb_evidence",
        "confidence": 0.7,
        "new_findings": [],
        "residual_questions": [],
    }


@pytest.mark.asyncio
async def test_coordinator_attaches_ensemble_scores(tmp_path):
    scorer = _make_scorer(
        {
            "claude-opus-4-7": _scores_json(("cuda_graph_bs_512", 8.0, "fit")),
        }
    )
    c = _coord(tmp_path, scorer)
    task = _StubTask(task_id="t1", params={"gap_symptom": "cuda stalls"})
    await c._record_specialist_result(
        task=task,
        done_payload=_done(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    row = c.shared_state.specialist_rounds[0]
    assert "ensemble_scores" in row
    assert row["ensemble_scores"]["models"]["claude-opus-4-7"]["cuda_graph_bs_512"]["score"] == 8.0


@pytest.mark.asyncio
async def test_coordinator_no_scorer_no_key(tmp_path):
    c = _coord(tmp_path, None)
    task = _StubTask(task_id="t1", params={})
    await c._record_specialist_result(
        task=task,
        done_payload=_done(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    assert "ensemble_scores" not in c.shared_state.specialist_rounds[0]


@pytest.mark.asyncio
async def test_coordinator_scorer_exception_still_records(tmp_path):
    class _BoomScorer:
        async def score(self, **_kw):
            raise RuntimeError("scorer blew up")

    c = _coord(tmp_path, _BoomScorer())
    task = _StubTask(task_id="t1", params={})
    await c._record_specialist_result(
        task=task,
        done_payload=_done(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    assert len(c.shared_state.specialist_rounds) == 1
    assert "ensemble_scores" not in c.shared_state.specialist_rounds[0]


@pytest.mark.asyncio
async def test_coordinator_empty_proposals_not_scored(tmp_path):
    scorer = _make_scorer({"m1": _scores_json(("x", 1.0, "y"))})
    c = _coord(tmp_path, scorer)
    payload = _done()
    payload["proposal_set"] = []
    payload["empty"] = True
    task = _StubTask(task_id="t1", params={})
    await c._record_specialist_result(
        task=task,
        done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    assert "ensemble_scores" not in c.shared_state.specialist_rounds[0]
    assert scorer._client.chat.completions.calls == []


# 3. Renderer
def _real_state():
    from hyperloom.orchestrator.state.shared_state import SharedState

    return SharedState()


def test_render_omits_section_when_no_scores():
    st = _real_state()
    st.specialist_rounds = [
        {"round_id": "r1", "domain": "serving_specialist", "proposal_set": [{"name": "v1"}]},
    ]
    assert st.to_proposal_scores_summary() == ""


def test_render_shows_per_model_side_by_side():
    st = _real_state()
    st.specialist_rounds = [
        {
            "round_id": "r1",
            "domain": "serving_specialist",
            "proposal_set": [{"name": "cuda_graph_bs_512"}, {"name": "disable_radix"}],
            "ensemble_scores": {
                "scale": "0-10",
                "models": {
                    "claude-opus-4-7": {
                        "cuda_graph_bs_512": {"score": 8.0, "reason": "fit"},
                        "disable_radix": {"score": 4.0, "reason": "marginal"},
                    },
                    "gpt-5.4": {
                        "cuda_graph_bs_512": {"score": 6.5, "reason": "plausible"},
                    },
                },
                "errors": {},
            },
        }
    ]
    text = st.to_proposal_scores_summary()
    assert "Advisory only" in text
    assert "cuda_graph_bs_512" in text
    # Model slugs are anonymized to rater_N (sorted, stable).
    assert "claude-opus-4-7" not in text
    assert "gpt-5.4" not in text
    assert "rater_1=8.0" in text
    assert "rater_2=6.5" in text
    assert "rater_2=n/a" in text


def test_render_reports_unavailable_models():
    st = _real_state()
    st.specialist_rounds = [
        {
            "round_id": "r1",
            "domain": "comm_specialist",
            "proposal_set": [{"name": "v1"}],
            "ensemble_scores": {
                "scale": "0-10",
                "models": {"good": {"v1": {"score": 7.0, "reason": "ok"}}},
                "errors": {"bad": "timed out"},
            },
        }
    ]
    text = st.to_proposal_scores_summary()
    # Anonymized: failing slug "bad" -> rater_1, must not leak.
    assert "bad" not in text
    assert "raters unavailable this round: rater_1" in text


def test_render_rater_labels_stable_across_rounds():
    """The same model maps to the same rater_N across rounds; no slug leaks."""
    st = _real_state()
    st.specialist_rounds = [
        {
            "round_id": "r1",
            "domain": "serving_specialist",
            "proposal_set": [{"name": "v1"}],
            "ensemble_scores": {
                "scale": "0-10",
                "models": {
                    "claude-opus-4-8": {"v1": {"score": 8.0, "reason": "a"}},
                    "gpt-5.5": {"v1": {"score": 6.0, "reason": "b"}},
                },
                "errors": {},
            },
        },
        {
            "round_id": "r2",
            "domain": "kernel_switch_specialist",
            "proposal_set": [{"name": "v2"}],
            "ensemble_scores": {
                "scale": "0-10",
                "models": {
                    # Different dict order: the label keys on slug, not order.
                    "gpt-5.5": {"v2": {"score": 5.0, "reason": "c"}},
                    "claude-opus-4-8": {"v2": {"score": 9.0, "reason": "d"}},
                },
                "errors": {},
            },
        },
    ]
    text = st.to_proposal_scores_summary(max_rounds=2)
    for slug in ("claude-opus-4-8", "gpt-5.5", "claude", "gpt"):
        assert slug not in text, f"model slug {slug!r} leaked into prompt"
    # claude-opus-4-8 sorts first → rater_1, stable across rounds.
    assert "rater_1=8.0" in text
    assert "rater_2=6.0" in text
    assert "rater_2=5.0" in text
    assert "rater_1=9.0" in text


# 4. Resume idempotency
@pytest.mark.asyncio
async def test_resume_idempotent_on_round_id(tmp_path):
    scorer = _make_scorer(
        {
            "m1": _scores_json(("cuda_graph_bs_512", 8.0, "fit")),
        }
    )
    c = _coord(tmp_path, scorer)
    task = _StubTask(
        task_id="t1",
        params={},
    )
    for _ in range(2):
        await c._record_specialist_result(
            task=task,
            done_payload=_done(),
            source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
        )
    assert len(c.shared_state.specialist_rounds) == 1


# ---------------------------------------------------------------------------
# Streaming behaviour: the Primus-Safe proxy requires stream=True, and the
# per-call deadline must cover the full stream body (not just creation).
# ---------------------------------------------------------------------------


class _ScriptedStream:
    """Async iterator yielding a caller-supplied list of chunks. If
    ``stall`` is set, it hangs forever after the scripted chunks instead of
    stopping — emulating a proxy that opens the stream then stalls mid-body."""

    def __init__(self, chunks: list[Any], *, stall: bool = False):
        self._chunks = chunks
        self._stall = stall

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            if self._stall:
                await asyncio.sleep(3600)  # never completes within the deadline
            raise StopAsyncIteration


class _ScriptedCompletions:
    def __init__(self, chunks: list[Any], *, stall: bool = False):
        self._chunks = chunks
        self._stall = stall
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model, messages, max_completion_tokens, stream=False, stream_options=None):
        self.calls.append({"stream": stream, "stream_options": stream_options})
        return _ScriptedStream(self._chunks, stall=self._stall)


class _ScriptedClient:
    def __init__(self, chunks: list[Any], *, stall: bool = False):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(chunks, stall=stall))


def _content_chunk(text: str) -> _FakeChunk:
    return _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(text))])


def _usage_chunk(usage: _FakeUsage) -> _FakeChunk:
    return _FakeChunk(choices=[], usage=usage)


def _scripted_scorer(chunks: list[Any], *, stall: bool = False, call_timeout_s: float = 30.0):
    client = _ScriptedClient(chunks, stall=stall)
    return ProposalScorer(
        models=("m",),
        client_factory=lambda: client,
        call_timeout_s=call_timeout_s,
    ), client


@pytest.mark.asyncio
async def test_stream_flag_is_passed():
    chunks = [_content_chunk(_scores_json(("p", 5, "ok"))), _usage_chunk(_FakeUsage(10, 20))]
    scorer, client = _scripted_scorer(chunks)
    await scorer._score_one_model("m", "prompt", ["p"])
    call = client.chat.completions.calls[0]
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_multiple_content_chunks_are_accumulated():
    # Split a valid scores JSON across several content deltas.
    body = _scores_json(("p", 7, "good"))
    third = len(body) // 3
    chunks = [
        _content_chunk(body[:third]),
        _content_chunk(body[third : 2 * third]),
        _content_chunk(body[2 * third :]),
        _usage_chunk(_FakeUsage(11, 22)),
    ]
    scorer, _ = _scripted_scorer(chunks)
    out = await scorer._score_one_model("m", "prompt", ["p"])
    assert out["p"]["score"] == 7.0


@pytest.mark.asyncio
async def test_stalled_stream_body_times_out():
    # One content chunk, then the stream stalls forever: the deadline must
    # cover the consumption loop, not just stream creation.
    chunks = [_content_chunk('{"scores": {')]
    scorer, _ = _scripted_scorer(chunks, stall=True, call_timeout_s=0.05)
    with pytest.raises(RuntimeError, match="timed out"):
        await scorer._score_one_model("m", "prompt", ["p"])


@pytest.mark.asyncio
async def test_missing_usage_chunk_degrades_cleanly():
    # No usage-bearing chunk at all: scoring still succeeds.
    chunks = [_content_chunk(_scores_json(("p", 4, "fine")))]
    scorer, _ = _scripted_scorer(chunks)
    out = await scorer._score_one_model("m", "prompt", ["p"])
    assert out["p"]["score"] == 4.0
