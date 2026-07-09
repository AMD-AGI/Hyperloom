# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist prompt hint blocks switch to atom paths when framework == 'atom'."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)


def _render(domain_key: str, *, framework: str = "") -> str:
    domain = get_domain(domain_key)
    assert domain is not None, domain_key
    inp = SpecialistPromptInputs(
        task_id=f"task-{domain_key}-{framework or 'none'}",
        domain=domain,
        max_turns=4,
        gap_canonical_id=f"gap.{domain_key}.atomhint",
        gap_symptom="example symptom",
        gap_layer=domain.layer,
        framework=framework,
        workspace_path=f"/tmp/test/{domain_key}",
    )
    system, user = build_specialist_prompts(inp)
    return system + "\n" + user


# 1. Atom hint blocks: serving / kernels / dist render atom paths
def test_focus_serving_renders_atom_paths_when_framework_atom():
    text = _render("serving_specialist", framework="atom")
    for marker in (
        "atom/entrypoints/openai_server.py",
        "atom/model_engine/engine_core.py",
        "atom/model_engine/model_runner.py",
        "atom/model_engine/arg_utils.py",
    ):
        assert marker in text, f"missing atom serving hint: {marker!r}"


def test_focus_kernels_mentions_aiter_shared_with_sglang_vllm_under_atom():
    text = _render("kernel_switch_specialist", framework="atom")
    assert "atom/model_ops/" in text
    assert "atom/quantization/" in text
    lower = text.lower()
    assert "aiter" in lower
    assert "shared" in lower


def test_focus_dist_notes_single_node_only_under_atom():
    text = _render("comm_specialist", framework="atom")
    assert "single-node" in text.lower()
    assert "atom/utils/distributed/utils.py" in text


# 2. Atom hint blocks DROP literal sglang/vllm paths
@pytest.mark.parametrize(
    "domain_key",
    ["serving_specialist", "kernel_switch_specialist", "comm_specialist"],
)
def test_no_sglang_or_vllm_paths_in_atom_focus_blocks(domain_key):
    """The atom focus block must not mention sglang/vllm path literals."""
    text = _render(domain_key, framework="atom")
    head = text.find("**What to read first**")
    tail = text.find("**Pitfalls", head)
    assert head >= 0 and tail > head, "focus block boundaries not found"
    block = text[head:tail]
    assert "vllm/v1/" not in block, f"atom {domain_key} block mentions vllm/v1/: {block!r}"
    assert "sglang/python/sglang/srt/" not in block, f"atom {domain_key} block mentions sglang srt: {block!r}"


# 3. Cross-framework regression guard — non-atom still renders canonical hints.
@pytest.mark.parametrize("framework", ["", "sglang", "vllm"])
def test_focus_serving_renders_canonical_paths_under_non_atom(framework):
    text = _render("serving_specialist", framework=framework)
    assert "vllm/v1/" in text or "sglang/python/sglang/srt/" in text, (
        f"non-atom framework={framework!r} dropped canonical sglang/vllm serving hints"
    )


@pytest.mark.parametrize(
    "domain_key",
    ["serving_specialist", "kernel_switch_specialist", "comm_specialist"],
)
@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom", ""])
def test_specialist_focus_renders_non_empty_for_all_frameworks(
    domain_key,
    framework,
):
    text = _render(domain_key, framework=framework)
    assert "**What to read first**" in text
    assert "**Pitfalls" in text


# 4. Option A: dedicated cross-framework rewrite domain + porting focus.
def test_cross_framework_rewrite_domain_registered():
    domain = get_domain("cross_framework_rewrite_specialist")
    assert domain is not None
    assert domain.kb_anchor == "framework"


@pytest.mark.parametrize("framework", ["", "sglang", "vllm"])
def test_cross_framework_rewrite_focus_renders_porting_methodology(framework):
    text = _render("cross_framework_rewrite_specialist", framework=framework)
    lower = text.lower()
    # Rewrite (not git apply) methodology + self-check + unified-diff deliverable.
    assert "port" in lower
    assert "git apply" in lower
    assert "self-check" in lower
    assert "unified-diff" in lower or "unified diff" in lower
    # Provenance echo so the KB ledger records the cross-framework outcome.
    assert "provenance" in lower
