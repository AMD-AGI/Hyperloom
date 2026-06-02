"""Tests for the advisory specialist-proposal scorer (ProposalScorer).

Covers:

* Scorer unit — group prompt shape, JSON parse, per-model independent
  degrade (timeout / bad JSON / one model raises / unknown-slug 404),
  score clamping + unknown-name drop.
* Coordinator wiring — ``_record_specialist_result`` attaches
  ``ensemble_scores`` to the round entry when a scorer is present;
  absent scorer → no key; scorer exception → entry still recorded.
* Renderer — ``SharedState.to_proposal_scores_summary`` text + the
  section is omitted when no round carries scores.
* Resume idempotency — re-recording the same ``round_id`` does not
  duplicate (delegated to existing ``record_specialist_round``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from inference_optimizer.orchestrator.proposal_scorer import (
    DEFAULT_SCORER_MODELS,
    ProposalScorer,
    _extract_scores_json,
)
from inference_optimizer.orchestrator.policy import SPECIALIST_FROM_AGENT_PREFIX


# ===========================================================================
# Fake OpenAI client (test seam)
# ===========================================================================
@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResp:
    choices: list[_FakeChoice]


class _FakeCompletions:
    """Per-model scripted behaviour keyed by ``model=``.

    ``behaviour`` maps a model slug to either a string (returned as the
    reply content) or an Exception instance (raised to simulate a
    gateway error / 404 / timeout).
    """

    def __init__(self, behaviour: dict[str, Any]):
        self._behaviour = behaviour
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model: str, messages, max_completion_tokens):
        self.calls.append({"model": model, "messages": messages})
        result = self._behaviour.get(model)
        if isinstance(result, BaseException):
            raise result
        return _FakeResp(choices=[_FakeChoice(_FakeMessage(result or ""))])


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
    body = ", ".join(
        f'"{name}": {{"score": {score}, "reason": "{reason}"}}'
        for name, score, reason in pairs
    )
    return f'```json\n{{"scores": {{{body}}}}}\n```'


# ===========================================================================
# 1. Scorer unit
# ===========================================================================
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


@pytest.mark.asyncio
async def test_group_prompt_contains_all_proposals_and_gap():
    behaviour = {"m1": _scores_json(("cuda_graph_bs_512", 7.0, "x"))}
    scorer = _make_scorer(behaviour)
    await scorer.score(gap=_GAP, proposals=_PROPOSALS)
    client = scorer._client
    sent = client.chat.completions.calls[0]["messages"][0]["content"]
    # group scoring → one prompt names every proposal + the gap.
    assert "cuda_graph_bs_512" in sent
    assert "disable_radix" in sent
    assert "cuda graph capture stalls" in sent
    assert "0 to 10" in sent  # instruction present


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
            ("cuda_graph_bs_512", 99.0, "over"),       # clamp to 10
            ("disable_radix", -3.0, "under"),          # clamp to 0
            ("ghost_variant", 5.0, "not in set"),      # dropped
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
    # No model call should have been made.
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


# ===========================================================================
# 2. Coordinator wiring
# ===========================================================================
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
        self.specialist_domain_empty_streak[d] = (
            0 if not empty
            else self.specialist_domain_empty_streak.get(d, 0) + 1
        )
        return self.specialist_domain_empty_streak[d]

    def update_last_specialist(self, snapshot) -> None:
        self.last_specialist = dict(snapshot)

    def save(self, _sd) -> None:
        self.saved += 1


def _coord(tmp_path: Path, scorer):
    from inference_optimizer.orchestrator.coordinator import Coordinator

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
    scorer = _make_scorer({
        "claude-opus-4-7": _scores_json(("cuda_graph_bs_512", 8.0, "fit")),
    })
    c = _coord(tmp_path, scorer)
    task = _StubTask(task_id="t1", params={"gap_symptom": "cuda stalls"})
    await c._record_specialist_result(
        task=task, done_payload=_done(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    row = c.shared_state.specialist_rounds[0]
    assert "ensemble_scores" in row
    assert row["ensemble_scores"]["models"]["claude-opus-4-7"][
        "cuda_graph_bs_512"
    ]["score"] == 8.0


@pytest.mark.asyncio
async def test_coordinator_no_scorer_no_key(tmp_path):
    c = _coord(tmp_path, None)
    task = _StubTask(task_id="t1", params={})
    await c._record_specialist_result(
        task=task, done_payload=_done(),
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
        task=task, done_payload=_done(),
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    # Round still recorded, just without scores.
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
        task=task, done_payload=payload,
        source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
    )
    assert "ensemble_scores" not in c.shared_state.specialist_rounds[0]
    assert scorer._client.chat.completions.calls == []


# ===========================================================================
# 3. Renderer
# ===========================================================================
def _real_state():
    from inference_optimizer.orchestrator.shared_state import SharedState
    return SharedState()


def test_render_omits_section_when_no_scores():
    st = _real_state()
    st.specialist_rounds = [
        {"round_id": "r1", "domain": "serving_specialist",
         "proposal_set": [{"name": "v1"}]},
    ]
    assert st.to_proposal_scores_summary() == ""


def test_render_shows_per_model_side_by_side():
    st = _real_state()
    st.specialist_rounds = [{
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
    }]
    text = st.to_proposal_scores_summary()
    assert "Advisory only" in text
    assert "cuda_graph_bs_512" in text
    # Identities are anonymized: real model slugs MUST NOT leak into the
    # orchestration-facing render. Sorted slugs map claude-opus-4-7 ->
    # rater_1, gpt-5.4 -> rater_2 (stable within a render).
    assert "claude-opus-4-7" not in text
    assert "gpt-5.4" not in text
    assert "rater_1=8.0" in text
    assert "rater_2=6.5" in text
    # rater_2 omitted disable_radix → rendered as n/a.
    assert "rater_2=n/a" in text


def test_render_reports_unavailable_models():
    st = _real_state()
    st.specialist_rounds = [{
        "round_id": "r1",
        "domain": "comm_specialist",
        "proposal_set": [{"name": "v1"}],
        "ensemble_scores": {
            "scale": "0-10",
            "models": {"good": {"v1": {"score": 7.0, "reason": "ok"}}},
            "errors": {"bad": "timed out"},
        },
    }]
    text = st.to_proposal_scores_summary()
    # Anonymized: the failing model slug "bad" must not leak; sorted
    # slugs map bad -> rater_1, good -> rater_2.
    assert "bad" not in text
    assert "raters unavailable this round: rater_1" in text


def test_render_rater_labels_stable_across_rounds():
    """The same model maps to the same rater_N in every rendered round,
    and no real model slug ever appears in the orchestration text."""
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
                    # Deliberately different dict order to prove the label
                    # is keyed on the slug, not on iteration order.
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
    # claude-opus-4-8 sorts before gpt-5.5 → rater_1 / rater_2, stable
    # across both rounds: rater_1 carries claude's 8.0 then 9.0.
    assert "rater_1=8.0" in text
    assert "rater_2=6.0" in text
    assert "rater_2=5.0" in text
    assert "rater_1=9.0" in text


# ===========================================================================
# 4. Resume idempotency
# ===========================================================================
@pytest.mark.asyncio
async def test_resume_idempotent_on_round_id(tmp_path):
    scorer = _make_scorer({
        "m1": _scores_json(("cuda_graph_bs_512", 8.0, "fit")),
    })
    c = _coord(tmp_path, scorer)
    task = _StubTask(task_id="t1", params={}, )
    # round_id defaults to task_id; record twice.
    for _ in range(2):
        await c._record_specialist_result(
            task=task, done_payload=_done(),
            source=f"{SPECIALIST_FROM_AGENT_PREFIX}t1",
        )
    # Idempotent on round_id → exactly one row.
    assert len(c.shared_state.specialist_rounds) == 1
    assert "ensemble_scores" in c.shared_state.specialist_rounds[0]
