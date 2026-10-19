# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``speculative_decoding_specialist`` domain: catalogue entry, focus prompt, and the target model it is told about."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, build_specialist_prompts
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.specialists.domains import get_domain, normalize_dispatch_tags
from hyperloom.orchestrator.specialists.runner import SpecialistRunner, SpecialistSubprocessConfig
from hyperloom.orchestrator.state.shared_state import SharedState

DOMAIN_KEY = "speculative_decoding_specialist"


def _system_prompt(**kw: Any) -> str:
    domain = get_domain(DOMAIN_KEY)
    assert domain is not None
    system, _ = build_specialist_prompts(
        SpecialistPromptInputs(task_id="t-1", domain=domain, gap_canonical_id="gap.decode", framework="vllm", **kw)
    )
    return system


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


@pytest.mark.parametrize(
    "marker",
    [
        "memory-bound",
        "Small batch",
        "Long context",
        "profiling and roofline evidence",
        "do not propose speculative decoding",
    ],
)
def test_focus_says_when_speculative_decoding_fits(marker):
    assert marker in _system_prompt()


@pytest.mark.parametrize("method", ["MTP", "EAGLE-3", "DFlash / DSpark", "Separate draft model", "suffix decoding"])
def test_focus_says_where_each_method_fits(method):
    assert method in _system_prompt()


def test_focus_puts_mtp_first_and_checks_its_weights_exist():
    text = _system_prompt()
    assert "**MTP first.**" in text
    for field in ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_modules"):
        assert field in text
    assert "weight index actually contains the MTP layer tensors" in text
    assert text.index("**MTP first.**") < text.index("look locally first")


def test_focus_looks_for_drafters_locally_before_downloading_one():
    text = _system_prompt(model_name="org/Target-7B", model_path="/models/Target-7B")
    assert "`org/Target-7B`, launched from `/models/Target-7B`" in text
    assert text.index("look locally first") < text.index("search Hugging Face and download")
    for rule in ("HF_HUB_OFFLINE", "only public repos", "safetensors", "never enable `trust_remote_code`"):
        assert rule in text


def test_focus_leaves_the_download_location_open_but_requires_it_to_outlive_the_dispatch():
    text = _system_prompt()
    assert "Download to a location that outlives this dispatch" in text
    assert "not your\n     worktree or workspace" in text


def test_focus_names_no_framework_flags_and_sends_the_specialist_to_the_installed_source():
    from hyperloom.orchestrator.prompts.speculative_decoding_focus import speculative_decoding_focus

    text = "\n".join(speculative_decoding_focus(model_name="m", model_path="/m"))
    for version_specific in ("--", "SGLANG_", "VLLM_", "synthetic_acceptance_length", "rejection_sample_method"):
        assert version_specific not in text
    assert "from the installed\n  framework's source" in text


def test_a_patch_mode_prompt_gets_no_coda_about_flags_it_never_listed():
    assert "Altitude —" not in _system_prompt()


def test_focus_carries_the_constraints_the_runner_enforces():
    text = _system_prompt()
    assert "**Never propose\n  these on the synthetic benchmark**" in text
    assert "the draft is pinned" in text
    assert "do not switch it to simulated" in text
    assert "absolute local path, never a repo id" in text
    assert "**atomic** proposal" in text


@dataclass
class _StubTask:
    task_id: str
    params: dict[str, Any]
    kind: str = "specialist"


async def test_dispatch_tells_the_specialist_the_target_model(tmp_path):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan

    SharedState(model_name="org/Target-7B", model_path="/models/Target-7B").save(tmp_path)
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

    assert "`org/Target-7B`, launched from `/models/Target-7B`" in prep.system_prompt
