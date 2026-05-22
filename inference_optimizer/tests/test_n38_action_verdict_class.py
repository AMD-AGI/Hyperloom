"""N38 (May 2026) — structural fix: per-action verdict_class metadata.

Background — the cumulative N33 / N35 / N37 saga:

Each time we found a Critic ``needs_review`` / ``advise`` deadlock for
a particular action class (archival, exploration, validate_stack), the
fix was the SAME shape: edit ``critic.md`` to add the action name to a
hard-coded carve-out list. That meant every newly registered action
risked re-introducing the same deadlock if someone forgot to update
the prompt — and indeed N37 was a direct consequence of N35
mis-classifying ``validate_stack``.

N38 makes the fix structural: each action declares its verdict class
(``archival`` / ``exploration`` / ``promotion``) in its yaml metadata,
the CriticAgentBackend injects ``action_verdict_policy`` into the
runtime's ``judge_bundle.review_constraints``, and ``critic.md``'s
primary rule becomes a lookup into that table. New actions opt into
the right policy by setting one field in their yaml; no prompt edits
needed.

Class semantics:

* ``archival`` — transcribes existing state to disk. Produces no new
  measurements. Examples: ``report``, ``session_breakdown``,
  ``target_analysis``. ALWAYS approve.
* ``exploration`` — runs benchmarks / variants to GENERATE before/after
  data. Examples: ``baseline``, ``profile``, ``roofline``, ``params``,
  ``backends``, ``sweep``, ``kernel_opt``, ``validate_stack``, etc.
  Approve unless the proposal itself is structurally invalid; the
  measurement IS the evidence.
* ``promotion`` — mutates ``optimization_stack`` with a KEEP entry
  that claims an E2E gain. Only ``integrate`` currently. Requires
  comparable before/after benchmark + accuracy gate + rollback.

Tests pin:
* ActionMetadata declares ``verdict_class`` field.
* Default classifier produces correct mapping for every currently
  registered action (so new yaml files inherit safe defaults).
* Loaded ActionRegistry covers every action with a non-empty
  verdict_class.
* CriticAgentBackend accepts an ``action_verdict_policy`` constructor
  param and injects it into ``judge_bundle.review_constraints``
  after the runtime's ``prepare-review`` returns the bundle.
* ``critic.md`` mentions ``action_verdict_policy`` as the primary
  per-proposal lookup (so the LLM reads it instead of hard-coded
  carve-outs).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# ActionMetadata.verdict_class
# ---------------------------------------------------------------------------
def test_action_metadata_has_verdict_class_field():
    from inference_optimizer.orchestrator.action_registry import (
        ActionMetadata,
    )
    fields = {f.name for f in ActionMetadata.__dataclass_fields__.values()}
    assert "verdict_class" in fields, (
        "ActionMetadata must declare verdict_class field so per-action "
        "policy can be looked up in critic review_constraints"
    )


def test_default_classifier_covers_all_registered_actions():
    """Default ``verdict_class`` must be filled in for every action
    currently in ``actions/_meta``. This means a newly added yaml that
    forgets to set ``verdict_class`` still gets a safe default rather
    than an empty string the critic doesn't know how to interpret."""
    from inference_optimizer.orchestrator.action_registry import (
        ActionRegistry,
    )
    reg = ActionRegistry().load()
    all_actions = reg.all()
    assert all_actions, "expected ActionRegistry to load >= 1 action"
    missing = [a.name for a in all_actions if not a.verdict_class]
    assert not missing, (
        f"actions missing verdict_class default: {missing} -- update the "
        f"default classifier in action_registry.py or add the field to "
        f"the yaml"
    )


def test_default_classifier_matches_expected_buckets():
    """Pin the expected default mapping for the actions we have rules
    about. Newly added actions get a default from the classifier but
    these specific ones MUST map to specific buckets to avoid
    reintroducing the N33/N35/N37 deadlocks."""
    from inference_optimizer.orchestrator.action_registry import (
        ActionRegistry,
    )
    reg = ActionRegistry().load()

    def klass(name: str) -> str:
        a = reg.get(name)
        assert a is not None, f"action {name!r} not registered"
        return a.verdict_class

    # Promotion bucket — ONLY integrate. These actually mutate the
    # optimization stack with a KEEP claim and need before/after
    # evidence.
    assert klass("integrate") == "promotion"

    # Archival bucket.
    for n in ("report", "session_breakdown", "target_analysis"):
        assert klass(n) == "archival", n

    # Exploration / measurement bucket — every other registered action
    # that runs benchmarks or variants to produce data.
    for n in (
        "baseline", "profile", "roofline", "params", "backends",
        "sweep", "kernel_opt", "pmc_roofline", "compiler_tuning",
        "comm_optimization", "operator_tuning", "vendor_kernel_config",
        "deep_kernel_analysis", "recover", "validate_stack",
    ):
        assert klass(n) == "exploration", n


# ---------------------------------------------------------------------------
# CriticAgentBackend injects action_verdict_policy
# ---------------------------------------------------------------------------
def test_critic_agent_backend_accepts_action_verdict_policy(tmp_path):
    """The constructor must accept ``action_verdict_policy`` so the
    coordinator can plumb a registry-derived mapping through."""
    from inference_optimizer.orchestrator.backends.critic_agent import (
        CriticAgentBackend,
    )
    # Minimal critic-agent root for the constructor's smoke check.
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub")
    sd = tmp_path / "session"
    sd.mkdir()

    def _fake_client_factory():
        class _C: pass
        return _C()

    def _fake_runtime_caller_factory():
        def _caller(call): return None
        return _caller

    backend = CriticAgentBackend(
        critic_agent_root=root,
        session_dir=sd,
        codex_client_factory=_fake_client_factory,
        runtime_caller_factory=_fake_runtime_caller_factory,
        static_context={"model": "m", "framework": "sglang"},
        action_verdict_policy={"baseline": "exploration", "integrate": "promotion"},
    )
    assert backend.action_verdict_policy == {
        "baseline": "exploration", "integrate": "promotion",
    }


def test_critic_agent_backend_injects_policy_into_judge_bundle(tmp_path):
    """When ``action_verdict_policy`` is non-empty the backend must
    enrich ``judge_bundle.review_constraints.action_verdict_policy``
    with it before the bundle is JSON-serialized into the LLM prompt."""
    import asyncio
    from inference_optimizer.orchestrator.backends.critic_agent import (
        CriticAgentBackend,
        RuntimeCall,
    )
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub")
    sd = tmp_path / "session"
    sd.mkdir()

    captured_bundle: dict = {}

    class _FakeAsyncOpenAI:
        def __init__(self): self.chat = _FakeChat(captured_bundle)
    class _FakeChat:
        def __init__(self, bucket): self.completions = _FakeCompletions(bucket)
    class _FakeCompletions:
        def __init__(self, bucket): self._b = bucket
        async def create(self, *, model, messages, max_completion_tokens):
            # User prompt contains the JSON-serialized judge_bundle.
            user_msg = messages[-1]["content"]
            self._b["user_prompt"] = user_msg
            class _Choice:
                message = type("M", (), {"content": json.dumps({
                    "review_verdicts": [],
                })})()
                finish_reason = "stop"
            return type("R", (), {"choices": [_Choice()]})()

    def _fake_runtime_caller_factory():
        def _caller(call: RuntimeCall) -> None:
            # Simulate runtime materializing a judge_bundle with
            # review_constraints already set. The backend should
            # ENRICH this with action_verdict_policy.
            if call.phase == "prepare-review":
                bundle = {
                    "kind": "coordinator_inbox",
                    "session_id": "test",
                    "proposals": [{"msg_id": "abc", "action_name": "params"}],
                    "review_constraints": {
                        "allowed_verdicts": ["approve", "advise"],
                    },
                }
                call.out_path.write_text(json.dumps(bundle), encoding="utf-8")
            else:
                call.out_path.write_text(json.dumps({
                    "intent_envelope": {"intents": []},
                }), encoding="utf-8")
        return _caller

    backend = CriticAgentBackend(
        critic_agent_root=root,
        session_dir=sd,
        codex_client_factory=_FakeAsyncOpenAI,
        runtime_caller_factory=_fake_runtime_caller_factory,
        static_context={"model": "m", "framework": "sglang"},
        action_verdict_policy={
            "params": "exploration", "integrate": "promotion",
        },
    )
    asyncio.run(backend.run(prompt="hello"))

    assert "action_verdict_policy" in captured_bundle.get("user_prompt", ""), (
        "action_verdict_policy must appear in the JSON prompt sent to "
        "the LLM-critic so it can look up each proposal's class"
    )
    assert "promotion" in captured_bundle["user_prompt"]


# ---------------------------------------------------------------------------
# critic.md primary lookup
# ---------------------------------------------------------------------------
def test_critic_md_mentions_action_verdict_policy_lookup():
    """``critic.md`` must instruct the LLM-critic to use
    ``review_constraints.action_verdict_policy`` as the primary
    per-proposal rule. The hard-coded carve-out lists can stay as
    belt-and-suspenders docs but the lookup must be the canonical
    source of truth so newly added actions don't deadlock."""
    p = (
        Path(__file__).resolve().parent.parent
        / "orchestrator" / "system_prompts" / "critic.md"
    )
    text = p.read_text(encoding="utf-8")
    assert "action_verdict_policy" in text, (
        "critic.md must mention action_verdict_policy so the LLM-critic "
        "treats it as the primary per-proposal lookup; otherwise newly "
        "added actions will hit the same N33/N35/N37 chicken-and-egg "
        "deadlock"
    )
    # Reinforce: each of the 3 verdict classes must be named so the
    # LLM knows what action they map to.
    for klass in ("archival", "exploration", "promotion"):
        assert klass in text.lower(), klass
