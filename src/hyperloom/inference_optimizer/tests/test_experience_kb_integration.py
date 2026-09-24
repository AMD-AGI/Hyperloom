# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

from hyperloom.inference_optimizer.experience_kb import (
    ExperienceKBEvidence,
    ExperienceKBIntegration,
)
from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
from hyperloom.orchestrator.loop.intent_router import _stamp_kb_exposure
from hyperloom.orchestrator.policy.gate import CORE_STATE_FIELDS
from hyperloom.orchestrator.state.shared_state import _KB_INJECTIONS_CAP, SharedState


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
            experiences=({"experience_id": "exp-00000000000000000000000000000001", "decision": "keep"},),
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


def test_kb_read_context_is_runtime_shaped_and_cached_per_decision(tmp_path) -> None:
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
    integration = ExperienceKBIntegration(client, tmp_path)
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
    assert first.experiences == ({"experience_id": "exp-00000000000000000000000000000001", "decision": "keep"},)
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
    assert kwargs["operation_id"].startswith("kb-read-session-1-")


def test_bootstrap_uses_only_service_url_and_token(tmp_path, monkeypatch) -> None:
    seen = {}

    class Config:
        @staticmethod
        def from_env(values):
            seen["env"] = dict(values)
            return SimpleNamespace(base_url=values["HYPERLOOM_KB_URL"], token=values["HYPERLOOM_KB_TOKEN"])

    module = ModuleType("hyperloom_kb")
    module.RemoteConfig = Config
    module.RemoteClient = lambda config: SimpleNamespace(config=config)
    monkeypatch.setitem(sys.modules, "hyperloom_kb", module)

    assert ExperienceKBIntegration.from_env(tmp_path, {}) is None
    assert "env" not in seen

    env = {"HYPERLOOM_KB_URL": "https://kb.example", "HYPERLOOM_KB_TOKEN": "service-token"}
    integration = ExperienceKBIntegration.from_env(tmp_path, env)

    assert integration is not None
    assert seen["env"] == env
    assert integration.client.config.base_url == "https://kb.example"
    assert integration.client.config.token == "service-token"


def _evidence(*experience_ids: str, tick: int = 7) -> ExperienceKBEvidence:
    return ExperienceKBEvidence(
        tick=tick,
        read_id=f"read-{tick}",
        status="completed",
        prompt_block="=== Relevant Experience KB ===\n" + "\n".join(experience_ids),
        rendered_refs=tuple({"id": item, "purpose": "representative"} for item in experience_ids),
        warnings=(),
        experiences=tuple({"experience_id": item, "decision": "keep"} for item in experience_ids),
    )


class _Integration:
    def __init__(self, evidence: ExperienceKBEvidence) -> None:
        self.evidence = evidence

    def read_for_framework(self, *_args, **_kwargs):
        return self.evidence


def test_conversation_kb_block_is_fail_open_and_records_exposure(tmp_path) -> None:
    evidence = _evidence("exp-00000000000000000000000000000001")
    state = SharedState(tick=7, phase="FRAMEWORK_AGENT")
    coordinator = SimpleNamespace(
        session_dir=tmp_path,
        shared_state=state,
        _kb_integration=_Integration(evidence),
    )
    collaborator = ConversationCollaborator(coordinator)

    block = asyncio.run(collaborator._kb_prompt_block("proposal"))

    assert evidence.prompt_block in block
    assert "original Recipe benchmark measurement remains" in block
    assert "never replace benchmark_baseline" in block
    assert coordinator._kb_last_read == evidence
    [record] = state.experience_kb_injections
    assert record["tick"] == 7
    assert record["phase"] == "FRAMEWORK_AGENT"
    assert record["read_id"] == "read-7"
    assert record["experience_ids"] == ["exp-00000000000000000000000000000001"]
    assert record["experiences"] == [{"experience_id": "exp-00000000000000000000000000000001", "decision": "keep"}]
    assert record["prompt_block"] == block


def test_injection_is_recorded_once_per_injected_experience_set(tmp_path) -> None:
    first, second = "exp-00000000000000000000000000000001", "exp-00000000000000000000000000000002"
    state = SharedState()
    integration = _Integration(_evidence(first, second))
    coordinator = SimpleNamespace(session_dir=tmp_path, shared_state=state, _kb_integration=integration)
    collaborator = ConversationCollaborator(coordinator)

    for tick, evidence in (
        (1, _evidence(first, second, tick=1)),
        (2, _evidence(second, first, tick=2)),
        (3, _evidence(first, tick=3)),
    ):
        state.tick = tick
        integration.evidence = evidence
        asyncio.run(collaborator._kb_prompt_block("proposal"))
    integration.evidence = ExperienceKBEvidence(
        tick=4, read_id="read-4", status="unavailable", prompt_block="", rendered_refs=(), warnings=("offline",)
    )
    state.tick = 4
    assert asyncio.run(collaborator._kb_prompt_block("proposal")) == ""

    assert [row["tick"] for row in state.experience_kb_injections] == [1, 3]
    assert [row["experience_ids"] for row in state.experience_kb_injections] == [[first, second], [first]]


def test_injection_record_is_capped_and_survives_resume(tmp_path) -> None:
    state = SharedState()
    for index in range(_KB_INJECTIONS_CAP + 5):
        state.tick = index
        state.record_experience_kb_injection(
            read_id=f"read-{index}", experience_ids=[f"exp-{index}"], experiences=[], prompt_block="x"
        )
    state.save(tmp_path)

    restored = SharedState.load_or_init(tmp_path)

    assert len(restored.experience_kb_injections) == _KB_INJECTIONS_CAP
    assert restored.experience_kb_injections[-1]["read_id"] == f"read-{_KB_INJECTIONS_CAP + 4}"
    assert restored.experience_kb_injections == state.experience_kb_injections
    assert "experience_kb_injections" in CORE_STATE_FIELDS


def test_proposal_exposure_only_uses_current_orchestration_tick() -> None:
    evidence = ExperienceKBEvidence(
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
        _kb_last_read=evidence,
        shared_state=SimpleNamespace(tick=7),
    )
    payload = {}

    _stamp_kb_exposure(router, payload, source="orchestration")

    assert payload == {
        "kb_read_id": "read-1",
        "kb_rendered_refs": list(evidence.rendered_refs),
    }
    stale = {}
    router.shared_state.tick = 8
    _stamp_kb_exposure(router, stale, source="orchestration")
    assert stale == {}


def test_state_reader_prints_each_injected_experience_with_its_summary(tmp_path, monkeypatch, capsys) -> None:
    from hyperloom.inference_optimizer.tools import read_optimizer_state

    first, second = "exp-00000000000000000000000000000001", "exp-00000000000000000000000000000002"
    state = SharedState(tick=5, phase="FRAMEWORK_AGENT")
    state.record_experience_kb_injection(
        read_id="read-5",
        experience_ids=[first, second],
        experiences=[
            {"experience_id": second, "decision": "revert", "change_summary": "compile", "source_run_id": "run-a"},
            {"experience_id": first, "decision": "keep", "change_summary": "chunk-8192", "source_run_id": "run-a"},
        ],
        prompt_block=f"Experience {first}\n...\nExperience {second}\n...",
    )
    state.save(tmp_path)
    monkeypatch.setattr(sys, "argv", ["read_optimizer_state.py", str(tmp_path)])

    assert read_optimizer_state.main() == 0

    lines = capsys.readouterr().out.splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("experience_kb_injection:"))
    assert "latest tick=5 read_id=read-5" in lines[header]
    assert lines[header + 1].startswith(f"  {first}: decision=keep change='chunk-8192'")
    assert lines[header + 2].startswith(f"  {second}: decision=revert change='compile'")
