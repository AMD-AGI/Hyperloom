# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that configures no predictor must behave exactly as before.

Most users will never run one: it is a separate 27B service on its own host,
not something a single-GPU install can stand up. So every seam the integration
touches has to be inert without an endpoint -- and inert here means *identical*,
not merely harmless. The prompt is the seam that matters most, because a
sentence about a proposer that will never appear is a permanent tax on every
orchestration turn in every session that does not have one.

The byte-for-byte assertions below are deliberately literal. A queue header
rewritten "harmlessly" is exactly the change that would slip through a test
asserting only that the block is non-empty.
"""

from __future__ import annotations

import asyncio

import pytest

from hyperloom.orchestrator.predictor import config as cfg
from hyperloom.orchestrator.predictor import pump as pp
from hyperloom.orchestrator.state.shared_state import SharedState

#: The queue header as it read before the predictor existed. Duplicated here on
#: purpose: comparing the renderer against itself would pass no matter what it
#: emitted, which is the whole failure this guards.
_BASELINE_HEADER = (
    "Executable specialist proposals from this cycle that no explore round has benched.\n"
    "Ranked by gap severity, then most recent. Compose the next `explore` grid from these;\n"
    "dispatch an ATOMIC entry verbatim as one variant — never split or re-derive its flags.\n"
)


@pytest.fixture(autouse=True)
def _no_predictor(monkeypatch):
    """Unset the predictor env, whatever the operator's shell carries."""
    for name in (cfg.ENV_ENDPOINT, cfg.ENV_MODE):
        monkeypatch.delenv(name, raising=False)


def _specialist_state() -> SharedState:
    """A queue holding one ordinary specialist proposal."""
    state = SharedState()
    state.macro_cycle = 0
    state.gaps = [{"canonical_id": "gap.x", "severity": "high"}]
    state.specialist_rounds = [
        {
            "cycle": 0,
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.x",
            "task_id": "abcdef1234",
            "proposal_set": [{"name": "s1", "extra_args": "--max-num-seqs 512"}],
        }
    ]
    return state


def test_the_config_reports_itself_disabled():
    conf = cfg.load()
    assert not conf.enabled
    assert not conf.enqueues


def test_the_pump_sends_no_request(monkeypatch):
    sent: list[dict] = []

    def _fake(request, *, endpoint, timeout_sec):
        sent.append(request)
        raise AssertionError("no request may be sent without an endpoint")

    monkeypatch.setattr(pp, "predict", _fake)

    class _Phase:
        def __init__(self):
            self.shared_state = SharedState()
            self.shared_state.phase = "FRAMEWORK_AGENT"
            self.shared_state.framework = "vllm"

    phase = _Phase()
    asyncio.run(pp.pump(phase, caller="entry"))
    assert sent == []
    assert phase.shared_state.specialist_rounds == []
    assert phase.shared_state.predictor_asked_keys == []


def test_the_queue_header_is_byte_identical():
    block = _specialist_state().to_untested_proposals_summary()
    assert block.startswith(_BASELINE_HEADER)


def test_the_queue_says_nothing_about_a_predictor():
    block = _specialist_state().to_untested_proposals_summary()
    lowered = block.lower()
    for token in ("first-pass", "primatune", "votes=", "mandate"):
        assert token not in lowered, token


def test_the_orchestration_system_prompt_says_nothing_about_a_predictor():
    """The static asset carries no predictor-specific wording at all.

    Anything the LLM needs to know about a first-pass row travels with the row,
    in the rendered block header, so this stays true whether or not a predictor
    is configured.
    """
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir

    text = (asset_system_prompts_dir() / "orchestration.md").read_text(encoding="utf-8")
    assert "primatune" not in text.lower()
    assert "first-pass" not in text.lower()


def test_a_constant_priority_leaves_the_ranking_untouched():
    """The new primary sort key must be a no-op when nothing sets it.

    Two gaps of different severity plus two rounds of different age exercise
    both surviving keys; a constant primary cannot reorder them.
    """
    state = SharedState()
    state.macro_cycle = 0
    state.gaps = [
        {"canonical_id": "gap.hi", "severity": "high"},
        {"canonical_id": "gap.lo", "severity": "low"},
    ]
    state.specialist_rounds = [
        {
            "cycle": 0,
            "domain": "a_specialist",
            "gap_canonical_id": "gap.lo",
            "task_id": "t1",
            "proposal_set": [{"name": "old-low", "extra_args": "--a 1"}],
        },
        {
            "cycle": 0,
            "domain": "b_specialist",
            "gap_canonical_id": "gap.hi",
            "task_id": "t2",
            "proposal_set": [{"name": "new-high", "extra_args": "--b 1"}],
        },
        {
            "cycle": 0,
            "domain": "c_specialist",
            "gap_canonical_id": "gap.lo",
            "task_id": "t3",
            "proposal_set": [{"name": "new-low", "extra_args": "--c 1"}],
        },
    ]
    names = [row["name"] for row in state._untested_proposal_rows()]
    # Severity first, then recency: high beats both lows, and the newer low
    # beats the older one.
    assert names == ["new-high", "new-low", "old-low"]
    assert all(row["priority"] == 0 for row in state._untested_proposal_rows())


def test_no_variant_is_relabelled_without_a_predictor():
    """The fingerprint stamp has nothing to match, so provenance survives."""
    from hyperloom.orchestrator.loop.proposals import ProposalsCollaborator

    obj = ProposalsCollaborator.__new__(ProposalsCollaborator)
    obj.shared_state = _specialist_state()
    params = {
        "grid": [
            {"name": "a", "extra_args": "--max-num-seqs 512", "provenance": "llm_direct"},
            {"name": "b", "extra_args": "--kv-cache-dtype fp8", "provenance": "specialist:moe"},
        ]
    }
    obj._stamp_first_pass_provenance(params)
    assert [v["provenance"] for v in params["grid"]] == ["llm_direct", "specialist:moe"]


def test_a_specialist_dispatch_is_untouched_without_a_mandate():
    from hyperloom.orchestrator.loop.intent_router import IntentRouter

    router = IntentRouter.__new__(IntentRouter)
    object.__setattr__(router, "_coord", type("C", (), {"shared_state": SharedState()})())
    params = {"scope": "freeform", "task_description": "an ordinary LLM mandate"}
    before = dict(params)
    router._resolve_first_pass_mandate(params)
    assert params == before
