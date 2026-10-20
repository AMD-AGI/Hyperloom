# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``speculative_decoding_specialist`` domain: catalogue entry, focus prompt, and the target model it is told about."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.prompts.speculative_decoding_focus import speculative_decoding_focus
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, build_specialist_prompts
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.specialists.domains import get_domain, normalize_dispatch_tags
from hyperloom.orchestrator.specialists.runner import SpecialistRunner, SpecialistSubprocessConfig
from hyperloom.orchestrator.state.shared_state import SharedState

DOMAIN_KEY = "speculative_decoding_specialist"
_TARGET = {"model_name": "org/Target-7B", "model_path": "/models/Target-7B", "model_type": "llama"}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _system_prompt(model_info: dict[str, Any] | None = None) -> str:
    domain = get_domain(DOMAIN_KEY)
    assert domain is not None
    system, _ = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="t-1",
            domain=domain,
            gap_canonical_id="gap.decode",
            framework="vllm",
            model_info=_TARGET if model_info is None else model_info,
        )
    )
    return _flat(system)


def test_orchestration_may_delegate_to_the_domain_under_its_own_kb_anchor():
    domain = get_domain(DOMAIN_KEY)
    assert domain is not None
    assert domain.llm_selectable
    assert normalize_dispatch_tags({"domain": DOMAIN_KEY}) == ["speculative_decoding"]
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = SharedState()
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "specialist", "params": {"domain": DOMAIN_KEY, "gap_canonical_id": "gap.x"}},
        ),
    )


def test_focus_names_the_target_model():
    assert "`org/Target-7B`, launched from `/models/Target-7B`" in _system_prompt()


def test_focus_sends_the_specialist_to_the_serving_config_when_the_model_is_not_recorded():
    text = _system_prompt({})
    assert "Target model: not recorded for this session" in text
    assert "MTP first.** The target's config.json could not be read" in text


def test_focus_states_the_mtp_layer_count_and_still_asks_for_the_tensors():
    text = _system_prompt({**_TARGET, "mtp_layers": 1})
    assert "**MTP first.** The target's config.json declares 1 MTP layer(s)." in text
    assert "Confirm the weight index carries those tensors" in text


def test_focus_skips_mtp_when_the_config_declares_none():
    text = _system_prompt()
    assert "**No MTP.**" in text
    assert "MTP first" not in text


def test_focus_looks_for_drafters_locally_before_downloading_one():
    text = _system_prompt()
    assert text.index("look locally first") < text.index("search Hugging Face and download")
    assert "Skip this step when `HF_HUB_OFFLINE` is set" in text


def test_focus_downloads_safely_into_the_hub_cache_by_default():
    text = _system_prompt()
    assert "Download into the Hugging Face cache by default" in text
    assert "must outlive this dispatch — not your worktree or workspace" in text
    assert "Take safetensors weights only; never enable `trust_remote_code`" in text
    assert "Pin the revision" in text


def test_focus_names_no_framework_flags_and_sends_the_specialist_to_the_installed_source():
    text = "\n".join(speculative_decoding_focus({**_TARGET, "mtp_layers": 1}))
    for version_specific in ("--", "SGLANG_", "VLLM_", "synthetic_acceptance_length", "rejection_sample_method"):
        assert version_specific not in text
    assert "from the installed framework's source" in _flat(text)


def test_a_patch_mode_prompt_gets_no_coda_about_flags_it_never_listed():
    assert "Altitude —" not in _system_prompt()


@dataclass
class _StubTask:
    task_id: str
    params: dict[str, Any]
    kind: str = "specialist"


async def test_dispatch_tells_the_specialist_the_session_model(tmp_path):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan

    SharedState(model_info={**_TARGET, "mtp_layers": 2}).save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    coord = Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )
    params = {"domain": DOMAIN_KEY, "gap_canonical_id": "gap.decode", "mode": "research"}
    await coord.specialist_dispatch.warm_specialist_params(params)
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig(), session_dir=tmp_path)

    prep = await runner._prepare(
        RunnerContext(task=_StubTask("task-spec", params), lease=None, extra={"shared_state": coord.shared_state}),
        prompt_inputs=None,
    )

    text = _flat(prep.system_prompt)
    assert "`org/Target-7B`, launched from `/models/Target-7B`" in text
    assert "declares 2 MTP layer(s)" in text
