# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
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
        current_best={"action": "explore", "tput": 112.0},
        current_action="explore",
        attempts_history=[{"throughput": 111.0, "decision": "keep"}],
        optimization_stack=["stack-a"],
        current_top_bottleneck=lambda: "decode bandwidth",
        to_prompt_summary=lambda: "analysis_md=decode is bandwidth bound",
    )


def test_fleet_read_context_is_runtime_shaped_and_cached_per_decision(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "task_config": {
                        "model_name": "manifest-model",
                        "gpu_type": "mi355x",
                        "framework_name": "sglang",
                        "framework_version": "0.5.18",
                        "precision": "bf16",
                        "architecture": {
                            "model_type": "qwen3",
                            "architectures": ["Qwen3ForCausalLM"],
                        },
                        "tp": 8,
                        "conc": 64,
                        "isl": 1024,
                        "osl": 128,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client = FakeClient()
    integration = FleetKBIntegration(client, tmp_path)
    state = _state()

    first = integration.read_for_framework(
        state,
        untested_proposals="fp8 KV cache",
    )
    second = integration.read_for_framework(
        state,
        untested_proposals="fp8 KV cache",
    )
    state.tick = 8
    third = integration.read_for_framework(
        state,
        untested_proposals="fp8 KV cache",
    )
    changed = integration.read_for_framework(
        state,
        untested_proposals="paged attention",
    )

    assert first == second
    assert third.read_id != first.read_id
    assert third.tick == 8
    assert changed.read_id != first.read_id
    assert changed.read_id != third.read_id
    assert len(client.calls) == 3
    _, context, kwargs = client.calls[0]
    assert context["identity"]["model"] == "Qwen3-8B"
    assert context["identity"]["gpu"] == "mi355x"
    assert context["identity"]["architecture"] == "Qwen3ForCausalLM"
    assert context["workload"] == {
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 128,
    }
    assert context["benchmark_baseline"]["throughput"] == 100.0
    assert context["current_best"]["throughput"] == 112.0
    assert context["observations"]["untested_directions"] == "fp8 KV cache"
    assert context["recent_results"] == [{"throughput": 111.0, "decision": "keep"}]
    assert "candidate_change" not in context
    assert "question" not in context
    assert kwargs["operation_id"].startswith("fleet-read-session-1-")


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

    assert evidence.prompt_block in block
    assert "original Recipe benchmark measurement remains" in block
    assert "never replace benchmark_baseline" in block
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
