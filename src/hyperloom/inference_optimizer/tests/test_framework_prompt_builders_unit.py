# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the FRAMEWORK prompt builder functions."""

from __future__ import annotations

import json

from hyperloom.orchestrator.prompts.framework_ranker_prompt import build_framework_ranker_prompt


def test_ranker_prompt_contains_workload_fields():
    prompt = build_framework_ranker_prompt(
        model="meta-llama/Llama-3-70b",
        framework="sglang",
        gpu_type="MI300X",
        precision="fp8",
        tp=8,
        best_throughput=3500.0,
        candidate_rows=["0. id=PR:100 repo=sgl-project/sglang title='MoE dispatch'"],
        has_local_explore=False,
        memory_block="",
    )
    assert "meta-llama/Llama-3-70b" in prompt
    assert "sglang" in prompt
    assert "MI300X" in prompt
    assert "fp8" in prompt
    assert "3500.0" in prompt


def test_ranker_prompt_includes_candidate_rows():
    rows = [
        "0. id=PR:1 repo=sgl-project/sglang title='fast path'",
        "1. id=PR:2 repo=ROCm/vllm title='moe opt'",
    ]
    prompt = build_framework_ranker_prompt(
        model="m", framework="f", gpu_type="g", precision="fp16",
        tp=1, best_throughput=None,
        candidate_rows=rows, has_local_explore=False, memory_block="",
    )
    for row in rows:
        assert row in prompt


def test_ranker_prompt_local_explore_note_when_flagged():
    prompt = build_framework_ranker_prompt(
        model="m", framework="f", gpu_type="g", precision="fp16",
        tp=1, best_throughput=None,
        candidate_rows=["0. id=local_explore:1 repo='' title=''"],
        has_local_explore=True, memory_block="",
    )
    assert "LOCAL-EXPLORATION" in prompt


def test_ranker_prompt_no_local_explore_note_when_absent():
    prompt = build_framework_ranker_prompt(
        model="m", framework="f", gpu_type="g", precision="fp16",
        tp=1, best_throughput=None,
        candidate_rows=["0. id=PR:1 repo=r title='t'"],
        has_local_explore=False, memory_block="",
    )
    assert "LOCAL-EXPLORATION" not in prompt


def test_ranker_prompt_memory_block_appears():
    prompt = build_framework_ranker_prompt(
        model="m", framework="f", gpu_type="g", precision="fp16",
        tp=1, best_throughput=None,
        candidate_rows=[],
        has_local_explore=False,
        memory_block="Already tried THIS session: PR:999",
    )
    assert "Already tried THIS session" in prompt


def test_ranker_prompt_footer_contains_json_hint():
    prompt = build_framework_ranker_prompt(
        model="m", framework="f", gpu_type="g", precision="fp16",
        tp=1, best_throughput=None,
        candidate_rows=[], has_local_explore=False, memory_block="",
    )
    assert '"candidate_id"' in prompt
    assert '"reason"' in prompt


def test_audit_refine_prompt_structure():
    from hyperloom.agents.framework.audit import build_audit_refine_prompt

    static = {
        "semantic_status": "not_present",
        "applicability": "direct_apply",
        "confidence": 0.2,
        "evidence": [],
        "risks": [],
    }
    prompt = build_audit_refine_prompt(static, "diff content here")
    assert json.dumps(static, ensure_ascii=False) in prompt
    assert "diff content here" in prompt
    assert "STRICT JSON" in prompt
    assert "semantic_status" in prompt


def test_audit_refine_prompt_truncates_long_diff():
    from hyperloom.agents.framework.audit import build_audit_refine_prompt

    long_diff = "x" * 10000
    prompt = build_audit_refine_prompt({}, long_diff)
    assert long_diff[:6000] in prompt
    assert long_diff not in prompt  # full 10 000-char string must not appear
