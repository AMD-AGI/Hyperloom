# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The section that tells a specialist of a checkout-built server how to find its gains."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common.agent_isolation import ACCOUNT, BWRAP, NONE, REFUSED, IsolationReport
from hyperloom.common.outside_facts import NATIVE_AUX_MODEL_FLAGS, NATIVE_SERVING
from hyperloom.common.tests.handed_names import (
    identifiers_named,
    knob_words_named,
    pull_requests_named,
    server_options_named,
)
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.inference_optimizer.session.session_paths import runs_root
from hyperloom.orchestrator.prompts import native_server_blocks
from hyperloom.orchestrator.prompts.native_discovery import OptionList, gpu_kernel_dirs, native_discovery
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists.domains import SPECIALIST_DOMAINS, get_domain
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs

_CARRIED = "your task ends with them."
_LOGS = "under /session/runs"

#: A line llama-server logged when a request ended, at its default verbosity.
_LLAMA_SERVER_LOG = (
    "10.44.529.863 I slot print_timing: id  0 | task 6888 |        eval time =   15449.58 ms /   512 tokens "
    "(   30.23 ms per token,    33.08 tokens per second)"
)


#: The option list of a session whose server takes 212 options, 15 of them set by its launch.
_OPTIONS = OptionList(Path("/session/server_options.json"), total=212, unset=197)


def _section(options: OptionList | None = None) -> str:
    return native_discovery("llamacpp", carried=_CARRIED, logs=_LOGS, options=options)


def test_the_section_names_the_directories_of_the_files_the_component_s_kernel_knobs_are_set_in():
    dirs = gpu_kernel_dirs("llamacpp")

    assert dirs and all(directory in _section() for directory in dirs)


#: What the rule pack states of the server, which the section repeats.
_STATED = " ".join((*NATIVE_SERVING["llamacpp"].values(), *NATIVE_AUX_MODEL_FLAGS["llamacpp"]))


def test_the_section_names_of_the_server_only_what_the_rule_pack_states_of_it():
    """An option, a kernel constant or a pull request the section named would be the answer handed over."""
    section = _section(_OPTIONS)

    assert server_options_named(section) == server_options_named(_STATED)
    assert identifiers_named(section) <= identifiers_named(_STATED) | {_OPTIONS.file.stem}
    assert not knob_words_named(section) and not pull_requests_named(section)


def test_a_single_branch_specialist_of_a_source_built_framework_reads_the_section_with_where_its_rounds_log(
    monkeypatch,
):
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, "/session")
    inputs = SpecialistPromptInputs(
        domain=get_domain("serving_specialist"), max_turns=4, framework="llamacpp", worktree_base="/f"
    )

    system, _ = build_specialist_prompts(inputs)

    assert "### Finding gains in a server built from its checkout" in system
    assert str(runs_root(Path("/session"))) in system


_FINDINGS = "runtime findings for /s/server.log [llamacpp]" + chr(10) + "    " + _LLAMA_SERVER_LOG
#: Every domain but the one dispatched before a round is measured, whose task has no findings to carry.
_MEASURED_DOMAINS = sorted(domain.key for domain in SPECIALIST_DOMAINS if domain.key != "enablement_specialist")
_PATCH_WITH_CARDS = {"worktree_base": "/f", "workspace_path": "/ws/worktree", "allocated_gpu_ids": [3]}


@pytest.mark.parametrize("domain", _MEASURED_DOMAINS)
@pytest.mark.parametrize("dispatch", [_PATCH_WITH_CARDS, {"mode": "research"}])
def test_a_single_branch_prompt_of_a_source_built_framework_names_no_option_but_the_stated_ones_and_no_knob(
    monkeypatch, domain, dispatch
):
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, "/session")
    inputs = SpecialistPromptInputs(domain=get_domain(domain), max_turns=4, framework="llamacpp", **dispatch)

    prompts = " ".join(build_specialist_prompts(inputs))

    assert server_options_named(prompts) <= server_options_named(_STATED)
    assert not knob_words_named(prompts) and not pull_requests_named(prompts)


@pytest.mark.parametrize("domain", ["custom_specialist", "serving_specialist"])
def test_a_specialist_told_its_timing_lines_are_handed_to_it_has_them_in_its_task(monkeypatch, domain):
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, "/session")
    common = {"domain": get_domain(domain), "max_turns": 4, "runtime_findings": _FINDINGS, "mode": "research"}

    system, user = build_specialist_prompts(SpecialistPromptInputs(framework="llamacpp", **common))
    image_system, image_user = build_specialist_prompts(SpecialistPromptInputs(framework="sglang", **common))

    assert "Those lines are handed to you" in system and _LLAMA_SERVER_LOG in user
    assert "## 4c. RUNTIME FINDINGS" in user
    assert "Those lines are handed to you" not in image_system
    if domain == "custom_specialist":
        assert "## 4c. RUNTIME FINDINGS" not in image_user, "told of no lines, an operator's specialist is handed none"


_KNOB_TUNE = "python -m hyperloom.measure.runner.native.knob_tune"


def _prompts(monkeypatch, isolation: str = NONE, **dispatch) -> str:
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, "/session")
    report = IsolationReport(specialists=isolation, sandboxes=NONE, required=False, bwrap=None, reasons=())
    monkeypatch.setattr(native_server_blocks, "agent_isolation_report", lambda: report)
    for name in ("HYPERLOOM_GFX_ARCH", "GFX_ARCH"):
        monkeypatch.delenv(name, raising=False)
    given = {"framework": "llamacpp", "gpu_type": "w7900d", **_PATCH_WITH_CARDS, **dispatch}
    inputs = SpecialistPromptInputs(domain=get_domain("serving_specialist"), max_turns=4, **given)
    return " ".join(build_specialist_prompts(inputs))


def test_a_patch_specialist_with_cards_is_offered_the_knob_sweep_of_its_session_on_those_cards(monkeypatch):
    prompts = _prompts(monkeypatch)

    assert f"ROCR_VISIBLE_DEVICES=3 {_KNOB_TUNE}" in prompts and "--session /session --knob" in prompts


@pytest.mark.parametrize(
    "dispatch", [{"mode": "research"}, {"allocated_gpu_ids": []}, {"framework": "sglang"}, {"gpu_type": "r9700"}]
)
def test_the_knob_sweep_is_not_offered_where_it_cannot_be_timed_or_applied_or_the_board_has_no_knob(
    monkeypatch, dispatch
):
    assert _KNOB_TUNE not in _prompts(monkeypatch, **dispatch)


@pytest.mark.parametrize("isolation", [BWRAP, ACCOUNT, REFUSED])
def test_the_knob_sweep_is_not_offered_to_a_confined_specialist_which_is_not_shown_the_session_s_base(
    monkeypatch, isolation
):
    assert _KNOB_TUNE not in _prompts(monkeypatch, isolation=isolation)
