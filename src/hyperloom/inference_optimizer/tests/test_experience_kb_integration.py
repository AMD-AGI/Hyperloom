# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

from hyperloom.inference_optimizer.experience_collect import mapping_schema_ref
from hyperloom.inference_optimizer.experience_kb import (
    CONTENT_INLINE_LIMIT,
    ExperienceKBEvidence,
    ExperienceKBIntegration,
    integration_for,
)
from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.intent_router import _stamp_kb_exposure
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.state.shared_state import _KB_INJECTIONS_CAP, SharedState
from hyperloom_kb import RemoteClient, RemoteConfig

_FIRST = "exp-00000000000000000000000000000001"
_SECOND = "exp-00000000000000000000000000000002"
_SCHEMA = "schema:sha256:" + "a" * 64


class FakeRef:
    def to_dict(self):
        return {"id": _FIRST, "purpose": "representative"}


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def read(self, decision, context, *, schema_ref, content_inline_limit):
        assert schema_ref == _SCHEMA
        assert content_inline_limit == CONTENT_INLINE_LIMIT
        self.calls.append((decision, context))
        return SimpleNamespace(
            read_id=f"read-{len(self.calls)}",
            status="completed",
            prompt_block="=== Relevant Experience KB ===\nprior evidence",
            rendered_refs=(FakeRef(),),
            warnings=(),
            experiences=({"experience_id": _FIRST, "decision": "keep"},),
            contents=(),
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
        attempts=[{"lever_kind": "config", "outcome": "KEEP", "gain_pct": 12.0}],
        optimization_stack=[{"variant_name": "stack-a", "gain_pct": 12.0}],
        current_top_bottleneck=lambda: "decode bandwidth",
    )


def _write_manifest(tmp_path) -> None:
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


def test_kb_read_context_is_runtime_shaped_and_cached_per_decision(tmp_path) -> None:
    _write_manifest(tmp_path)
    client = FakeClient()
    integration = ExperienceKBIntegration(client, tmp_path, _SCHEMA)
    state = _state()

    first = integration.read_for_framework(state, untested_proposals="fp8 KV cache")
    second = integration.read_for_framework(state, untested_proposals="fp8 KV cache")
    state.tick = 8
    third = integration.read_for_framework(state, untested_proposals="fp8 KV cache")
    changed = integration.read_for_framework(state, untested_proposals="paged attention")

    assert first == second
    assert first.experiences == ({"experience_id": _FIRST, "decision": "keep"},)
    assert third.read_id != first.read_id
    assert third.tick == 8
    assert changed.read_id not in {first.read_id, third.read_id}
    assert len(client.calls) == 3
    decision, context = client.calls[0]
    assert decision == "Select the next framework optimization to benchmark."
    assert context["identity"]["model"] == "Qwen3-8B"
    assert context["identity"]["gpu"] == "mi355x"
    assert context["identity"]["architecture"] == "Qwen3ForCausalLM"
    assert context["workload"] == {"tp": 8, "conc": 64, "isl": 1024, "osl": 128}
    assert context["benchmark_baseline"]["throughput"] == 100.0
    assert context["current_best"]["throughput"] == 112.0
    assert context["observations"] == {"bottleneck": "decode bandwidth", "untested_directions": "fp8 KV cache"}
    # The run's own history is already in its prompt; querying with it only steers retrieval back to it.
    for own_history in ("recent_results", "already_tried"):
        assert own_history not in context


def test_specialist_read_context_describes_the_dispatch(tmp_path) -> None:
    _write_manifest(tmp_path)
    client = FakeClient()
    integration = ExperienceKBIntegration(client, tmp_path, _SCHEMA)
    params = {
        "domain": "kernel_switch_specialist",
        "gap_symptom": "VLLM_ROCM_USE_AITER defaults False",
        "task_description": "bridge the AITER linear path",
        "pr_lead": {"title": "Enable AITER linear on MI300X"},
    }

    evidence = integration.read_for_specialist(_state(), params)
    integration.read_for_specialist(_state(), params)

    assert evidence.read_id == "read-1"
    assert len(client.calls) == 1
    decision, context = client.calls[0]
    assert decision == "Propose framework optimizations for this specialist investigation."
    assert context["identity"]["model"] == "Qwen3-8B"
    assert context["observations"] == {
        "bottleneck": "decode bandwidth",
        "specialist_domain": "kernel_switch_specialist",
        "investigation": "VLLM_ROCM_USE_AITER defaults False",
        "task": "bridge the AITER linear path",
        "upstream_pr": "Enable AITER linear on MI300X",
    }


def test_reads_speak_the_service_read_contract_through_the_real_sdk(tmp_path) -> None:
    requests = []
    response = {
        "read_id": "read-06b67219b0b8426593afc9914838a09c",
        "status": "completed",
        "outcome": "mixed",
        "limit": 10,
        "prompt_block": f"=== Relevant Experience KB ===\nExperience {_FIRST}\nRecord:\n{{}}",
        "rendered_refs": [{"id": _FIRST, "purpose": "representative"}],
        "experiences": [{"experience_id": _FIRST, "decision": "keep", "score": 1.5, "why_matched": []}],
        "eligible_count": 1,
        "rendered_count": 1,
        "warnings": ["capability_unavailable:semantic"],
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps(response).encode()

    def opener(request, timeout):
        requests.append(request)
        return _Response()

    config = RemoteConfig(base_url="https://kb.example", token="service-token")
    integration = ExperienceKBIntegration(RemoteClient(config, opener=opener), tmp_path, _SCHEMA)

    evidence = integration.read_for_specialist(_state(), {"domain": "serving_specialist"})

    [request] = requests
    assert request.full_url == "https://kb.example/v1/read"
    assert request.get_header("Authorization") == "Bearer service-token"
    body = json.loads(request.data)
    assert set(body) == {"decision", "context", "schema_ref", "content_inline_limit"}
    assert (body["schema_ref"], body["content_inline_limit"]) == (_SCHEMA, CONTENT_INLINE_LIMIT)
    assert evidence.status == "completed"
    assert evidence.read_id == response["read_id"]
    assert evidence.prompt_block == response["prompt_block"]
    assert evidence.rendered_refs == ({"id": _FIRST, "purpose": "representative"},)
    assert evidence.experiences == tuple(response["experiences"])


def test_a_referenced_change_reaches_the_prompt_as_session_files(tmp_path) -> None:
    patch = "--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n" * 60
    content = json.dumps({"patches": [{"path": "patches/fuse attn.diff", "sha256": "0" * 64, "content": patch}]})
    ref = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    block = f"=== Relevant Experience KB ===\nExperience {_FIRST}\nRecord:\n<external content {ref}, 9 bytes>"
    contents = ({"ref": ref, "bytes": len(content.encode()), "content": content},)
    reads = []

    class _Client:
        def read(self, decision, context, *, schema_ref, content_inline_limit):
            reads.append(content_inline_limit)
            return SimpleNamespace(
                read_id=f"read-{len(reads)}",
                status="completed",
                prompt_block=block,
                rendered_refs=(FakeRef(),),
                warnings=(),
                experiences=(),
                contents=contents,
            )

    integration = ExperienceKBIntegration(_Client(), tmp_path, _SCHEMA)
    evidence = integration.read_for_framework(_state(tick=1))
    again = integration.read_for_framework(_state(tick=2))

    content_file = tmp_path / "experience_kb" / "contents" / f"{ref.removeprefix('sha256:')}.txt"
    patch_file = content_file.with_suffix("") / "1-fuse_attn.diff"
    assert content_file.read_text(encoding="utf-8") == content
    assert patch_file.read_text(encoding="utf-8") == patch
    assert evidence.prompt_block.startswith(block)
    assert f"- {ref} ({len(content.encode())} bytes): {content_file}" in evidence.prompt_block
    assert f"  - patch: {patch_file}" in evidence.prompt_block
    assert patch not in evidence.prompt_block
    assert again.prompt_block == evidence.prompt_block
    assert reads == [CONTENT_INLINE_LIMIT, CONTENT_INLINE_LIMIT]


def test_bootstrap_uses_only_service_url_and_token(tmp_path) -> None:
    assert ExperienceKBIntegration.from_env(tmp_path, {}) is None
    assert ExperienceKBIntegration.from_env(tmp_path, {"HYPERLOOM_KB_URL": "https://kb.example"}) is None

    env = {"HYPERLOOM_KB_URL": "https://kb.example/", "HYPERLOOM_KB_TOKEN": "service-token"}
    integration = ExperienceKBIntegration.from_env(tmp_path, env)

    assert integration is not None
    assert isinstance(integration.client, RemoteClient)
    assert integration.client.config.base_url == "https://kb.example"
    assert integration.client.config.token == "service-token"
    # A run reads the schema its packaged mapping writes.
    assert integration.schema_ref == mapping_schema_ref()


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
        self.specialist_params = []

    def read_for_framework(self, *_args, **_kwargs):
        return self.evidence

    def read_for_specialist(self, _state, params):
        self.specialist_params.append(dict(params))
        return self.evidence


def test_conversation_kb_block_is_fail_open_and_records_exposure(tmp_path) -> None:
    evidence = _evidence(_FIRST)
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
    assert record["consumer"] == "orchestration"
    assert record["domain"] == ""
    assert record["gap_canonical_id"] == ""
    assert record["read_id"] == "read-7"
    assert record["experience_ids"] == [_FIRST]
    assert record["experiences"] == [{"experience_id": _FIRST, "decision": "keep"}]
    assert record["prompt_block"] == block


def test_orchestration_injection_is_recorded_once_per_injected_experience_set(tmp_path) -> None:
    state = SharedState()
    integration = _Integration(_evidence(_FIRST, _SECOND))
    coordinator = SimpleNamespace(session_dir=tmp_path, shared_state=state, _kb_integration=integration)
    collaborator = ConversationCollaborator(coordinator)

    for tick, evidence in (
        (1, _evidence(_FIRST, _SECOND, tick=1)),
        (2, _evidence(_SECOND, _FIRST, tick=2)),
        (3, _evidence(_FIRST, tick=3)),
    ):
        state.tick = tick
        integration.evidence = evidence
        asyncio.run(collaborator._kb_prompt_block("proposal"))
        state.record_experience_kb_injection(
            consumer="specialist",
            domain="serving_specialist",
            gap_canonical_id="gap.x",
            read_id=f"specialist-{tick}",
            experience_ids=[_FIRST],
            experiences=[],
            prompt_block="x",
        )
    integration.evidence = ExperienceKBEvidence(
        tick=4, read_id="read-4", status="unavailable", prompt_block="", rendered_refs=(), warnings=("offline",)
    )
    state.tick = 4
    assert asyncio.run(collaborator._kb_prompt_block("proposal")) == ""

    orchestration = [row for row in state.experience_kb_injections if row["consumer"] == "orchestration"]
    specialist = [row for row in state.experience_kb_injections if row["consumer"] == "specialist"]
    assert [row["tick"] for row in orchestration] == [1, 3]
    assert [row["experience_ids"] for row in orchestration] == [[_FIRST, _SECOND], [_FIRST]]
    assert [row["read_id"] for row in specialist] == ["specialist-1", "specialist-2", "specialist-3"]


def test_injection_record_is_capped_and_survives_resume(tmp_path) -> None:
    state = SharedState()
    for index in range(_KB_INJECTIONS_CAP + 5):
        state.tick = index
        state.record_experience_kb_injection(
            consumer="orchestration",
            read_id=f"read-{index}",
            experience_ids=[f"exp-{index}"],
            experiences=[],
            prompt_block="x",
        )
    state.save(tmp_path)

    restored = SharedState.load_or_init(tmp_path)

    assert len(restored.experience_kb_injections) == _KB_INJECTIONS_CAP
    assert restored.experience_kb_injections[-1]["read_id"] == f"read-{_KB_INJECTIONS_CAP + 4}"
    assert restored.experience_kb_injections == state.experience_kb_injections
    assert "experience_kb_injections" not in SharedState.AGENT_UPDATE_FIELDS


def _dispatch_coordinator(tmp_path, state: SharedState, integration: _Integration) -> Coordinator:
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.session_dir = tmp_path
    coordinator.shared_state = state
    coordinator.knowledge_plane = None
    coordinator._kb_integration = integration
    return coordinator


def test_specialist_dispatch_injects_its_experience_block_and_records_it(tmp_path) -> None:
    state = SharedState(tick=9, phase="FRAMEWORK_AGENT")
    state.upsert_gap(
        {
            "canonical_id": "gap.static_recon.aiter_master",
            "symptom": "VLLM_ROCM_USE_AITER defaults False",
            "layer": "static_recon",
            "domain_hint": "kernel_switch_specialist",
        }
    )
    evidence = _evidence(_FIRST, _SECOND, tick=9)
    integration = _Integration(evidence)
    coordinator = _dispatch_coordinator(tmp_path, state, integration)
    params = {"gap_canonical_id": "gap.static_recon.aiter_master"}

    asyncio.run(coordinator._warm_specialist_params(params))

    assert params["experience_kb_block"] == evidence.prompt_block
    assert params["kb_read_id"] == "read-9"
    assert params["kb_rendered_refs"] == [
        {"id": _FIRST, "purpose": "representative"},
        {"id": _SECOND, "purpose": "representative"},
    ]
    [read_params] = integration.specialist_params
    assert read_params["domain"] == "kernel_switch_specialist"
    assert read_params["gap_symptom"] == "VLLM_ROCM_USE_AITER defaults False"
    [record] = state.experience_kb_injections
    assert record["consumer"] == "specialist"
    assert record["domain"] == "kernel_switch_specialist"
    assert record["gap_canonical_id"] == "gap.static_recon.aiter_master"
    assert record["read_id"] == "read-9"
    assert record["experience_ids"] == [_FIRST, _SECOND]
    assert record["prompt_block"] == evidence.prompt_block

    _system, user = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="t-1",
            domain=get_domain("kernel_switch_specialist"),
            gap_canonical_id="gap.static_recon.aiter_master",
            experience_kb_block=params["experience_kb_block"],
        )
    )
    assert "## 4b. EXPERIENCE KB" in user
    assert evidence.prompt_block in user
    assert "COLD-START MODE" not in user


def test_specialist_dispatch_reads_only_in_framework_agent_and_fails_open(tmp_path) -> None:
    state = SharedState(tick=3, phase="PRELUDE")
    integration = _Integration(_evidence(_FIRST, tick=3))
    coordinator = _dispatch_coordinator(tmp_path, state, integration)

    prelude_params = {"domain": "research_scout_specialist"}
    asyncio.run(coordinator._warm_specialist_params(prelude_params))
    state.phase = "FRAMEWORK_AGENT"
    integration.evidence = ExperienceKBEvidence(
        tick=3, read_id="", status="unavailable", prompt_block="", rendered_refs=(), warnings=("offline",)
    )
    offline_params = {"domain": "serving_specialist"}
    asyncio.run(coordinator._warm_specialist_params(offline_params))

    for params in (prelude_params, offline_params):
        assert not {"experience_kb_block", "kb_read_id", "kb_rendered_refs"} & set(params)
    assert len(integration.specialist_params) == 1
    assert state.experience_kb_injections == []

    _system, user = build_specialist_prompts(
        SpecialistPromptInputs(task_id="t-2", domain=get_domain("serving_specialist"), gap_canonical_id="gap.y")
    )
    assert "EXPERIENCE KB" not in user


def test_an_agentx_run_reads_no_experience_since_none_of_its_own_is_published(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("HYPERLOOM_KB_URL", "https://kb.example")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "service-token")
    synthetic = SimpleNamespace(shared_state=SharedState(phase="FRAMEWORK_AGENT"))
    assert isinstance(integration_for(synthetic, tmp_path), ExperienceKBIntegration)

    state = SharedState(tick=4, phase="FRAMEWORK_AGENT", benchmark_mode="agentx")
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.session_dir = tmp_path
    coordinator.shared_state = state
    coordinator.knowledge_plane = None
    params = {"domain": "serving_specialist"}

    assert asyncio.run(ConversationCollaborator(coordinator)._kb_prompt_block("proposal")) == ""
    asyncio.run(coordinator._warm_specialist_params(params))

    assert coordinator._kb_integration is None
    assert not {"experience_kb_block", "kb_read_id", "kb_rendered_refs"} & set(params)
    assert state.experience_kb_injections == []


def test_proposal_exposure_only_uses_current_orchestration_tick() -> None:
    evidence = ExperienceKBEvidence(
        tick=7,
        read_id="read-1",
        status="completed",
        prompt_block="evidence",
        rendered_refs=({"id": _FIRST, "purpose": "representative"},),
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


def _grid_citing(*citations: dict) -> dict:
    return {"params": {"grid": [{"name": "v1", "experience_citations": list(citations)}, {"name": "v2"}]}}


def test_a_grid_keeps_only_citations_of_experiences_its_tick_showed() -> None:
    evidence = ExperienceKBEvidence(
        tick=7,
        read_id="read-1",
        status="completed",
        prompt_block="evidence",
        rendered_refs=({"id": _FIRST, "purpose": "representative"},),
        warnings=(),
    )
    router = SimpleNamespace(_kb_last_read=evidence, shared_state=SimpleNamespace(tick=7))
    shown = {"id": _FIRST, "stance": "ADOPT", "claim": "  Kept   twice on this model. "}
    unshown = {"id": _SECOND, "stance": "adopt", "claim": "Never rendered."}
    unknown_stance = {"id": _FIRST, "stance": "trust", "claim": "Not a stance."}
    payload = _grid_citing(shown, unshown, unknown_stance, shown)

    _stamp_kb_exposure(router, payload, source="orchestration")

    [cited, uncited] = payload["params"]["grid"]
    assert cited["experience_citations"] == [{"id": _FIRST, "stance": "adopt", "claim": "Kept twice on this model."}]
    assert uncited == {"name": "v2"}

    router.shared_state.tick = 8
    later = _grid_citing(shown)
    _stamp_kb_exposure(router, later, source="orchestration")
    assert later["params"]["grid"][0]["experience_citations"] == []


def test_state_reader_prints_each_consumers_latest_injection(tmp_path, monkeypatch, capsys) -> None:
    from hyperloom.inference_optimizer.tools import read_optimizer_state

    state = SharedState(tick=5, phase="FRAMEWORK_AGENT")
    state.record_experience_kb_injection(
        consumer="orchestration",
        read_id="read-5",
        experience_ids=[_FIRST, _SECOND],
        experiences=[
            {"experience_id": _SECOND, "decision": "revert", "change_summary": "compile", "source_run_id": "run-a"},
            {"experience_id": _FIRST, "decision": "keep", "change_summary": "chunk-8192", "source_run_id": "run-a"},
        ],
        prompt_block=f"Experience {_FIRST}\n...\nExperience {_SECOND}\n...",
    )
    state.record_experience_kb_injection(
        consumer="specialist",
        domain="kernel_switch_specialist",
        gap_canonical_id="gap.x",
        read_id="read-6",
        experience_ids=[_SECOND],
        experiences=[{"experience_id": _SECOND, "decision": "revert", "change_summary": "compile"}],
        prompt_block=f"Experience {_SECOND}\n...",
    )
    state.save(tmp_path)
    monkeypatch.setattr(sys, "argv", ["read_optimizer_state.py", str(tmp_path)])

    assert read_optimizer_state.main() == 0

    lines = capsys.readouterr().out.splitlines()
    orchestration = next(index for index, line in enumerate(lines) if line.startswith("experience_kb_injection[orch"))
    assert "latest tick=5 read_id=read-5" in lines[orchestration]
    assert lines[orchestration + 1].startswith(f"  {_FIRST}: decision=keep change='chunk-8192'")
    assert lines[orchestration + 2].startswith(f"  {_SECOND}: decision=revert change='compile'")
    specialist = next(index for index, line in enumerate(lines) if line.startswith("experience_kb_injection[spec"))
    assert "read_id=read-6 domain=kernel_switch_specialist gap=gap.x" in lines[specialist]
    assert lines[specialist + 1].startswith(f"  {_SECOND}: decision=revert change='compile'")
