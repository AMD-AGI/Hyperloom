# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hyperloom.inference_optimizer.fleet_kb import (
    FleetKBEvidence,
    FleetKBIntegration,
)
from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
from hyperloom.orchestrator.loop.intent_router import _stamp_fleet_kb_exposure


class FakeRef:
    def to_dict(self):
        return {
            "id": "exp-00000000000000000000000000000001",
            "purpose": "representative",
        }


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def read(self, decision, context, **kwargs):
        self.calls.append((decision, context, kwargs))
        return SimpleNamespace(
            read_id=kwargs["operation_id"],
            status="completed",
            prompt_block="=== Relevant Experience KB ===\nprior evidence",
            rendered_refs=(FakeRef(),),
            warnings=(),
        )


def _state(tick: int = 7):
    return SimpleNamespace(
        tick=tick,
        macro_cycle=2,
        session_id="session-1",
        model_name="Qwen3-8B",
        phase="FRAMEWORK_AGENT",
        baseline_tput=100.0,
        current_best="stack-a",
        current_action="explore",
        to_prompt_summary=lambda: "analysis_md=decode is bandwidth bound",
    )


def test_fleet_read_context_is_runtime_shaped_and_cached_per_tick(tmp_path) -> None:
    client = FakeClient()
    integration = FleetKBIntegration(client, tmp_path)
    state = _state()

    first = integration.read_for_framework(
        state,
        untested_proposals="fp8 KV cache",
    )
    second = integration.read_for_framework(
        state,
        untested_proposals="ignored on cached read",
    )

    assert first == second
    assert len(client.calls) == 1
    _, context, kwargs = client.calls[0]
    assert context["model"] == "Qwen3-8B"
    assert context["baseline_tput"] == 100.0
    assert context["untested_proposals"] == "fp8 KV cache"
    assert kwargs["operation_id"] == "fleet-read-session-1-2-7"


def test_conversation_fleet_block_is_fail_open_and_records_exposure(tmp_path) -> None:
    evidence = FleetKBEvidence(
        tick=7,
        read_id="read-1",
        status="completed",
        prompt_block="=== Relevant Experience KB ===\nprior evidence",
        rendered_refs=(
            {
                "id": "exp-00000000000000000000000000000001",
                "purpose": "representative",
            },
        ),
        warnings=(),
    )

    class Integration:
        def read_for_framework(self, *_args, **_kwargs):
            return evidence

    coordinator = SimpleNamespace(
        session_dir=tmp_path,
        shared_state=_state(),
        _fleet_kb_integration=Integration(),
    )
    collaborator = ConversationCollaborator(coordinator)

    block = asyncio.run(collaborator._fleet_kb_prompt_block("proposal"))

    assert block == evidence.prompt_block
    assert coordinator._fleet_kb_last_read == evidence


def test_proposal_exposure_only_uses_current_orchestration_tick() -> None:
    evidence = FleetKBEvidence(
        tick=7,
        read_id="read-1",
        status="completed",
        prompt_block="evidence",
        rendered_refs=(
            {
                "id": "exp-00000000000000000000000000000001",
                "purpose": "representative",
            },
        ),
        warnings=(),
    )
    router = SimpleNamespace(
        _fleet_kb_last_read=evidence,
        shared_state=SimpleNamespace(tick=7),
    )
    payload = {}

    _stamp_fleet_kb_exposure(router, payload, source="orchestration")

    assert payload == {
        "fleet_kb_read_id": "read-1",
        "fleet_kb_rendered_refs": list(evidence.rendered_refs),
    }
    stale = {}
    router.shared_state.tick = 8
    _stamp_fleet_kb_exposure(router, stale, source="orchestration")
    assert stale == {}
