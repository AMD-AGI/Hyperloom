# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-derived prompts retain the existing ordered optimization contract."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json

import pytest

from ave.optimizer import AveLoop
from ave.prompts import build_prompts
from ave.task import read_prompts
from ave.tests.test_loop import FakeClaude


def test_recipe_runs_through_the_existing_prompt_loop(prepared_run, monkeypatch):
    recipe = {
        "objective": "Optimize the fixed workload without changing VALUE.",
        "knowledge": ["avelang"],
        "steps": [
            {
                "name": step.id,
                "knowledge": ["avelang", "register_tiling"],
                "files": list(step.files),
                "required": [step.prompt],
                "avoid": ["Do not change the reference."],
            }
            for step in prepared_run.steps
        ],
    }
    steps = build_prompts(recipe, context="Use the supplied CPU driver with decode and prefill cases.")
    prompt_file = prepared_run.result_json.parent / "generated-prompts.json"
    prompt_file.write_text(json.dumps({"steps": [asdict(step) for step in steps]}))
    run = replace(prepared_run, steps=read_prompts(prompt_file))
    assert [step.id for step in run.steps] == [step.id for step in prepared_run.steps]
    assert run.steps[0].files == prepared_run.steps[0].files
    assert run.steps[0].prompt.count("# AveLang authoring and validation") == 1
    assert "# Register tiling" in run.steps[0].prompt
    assert "Do not change the reference." in run.steps[0].prompt
    assert "decode and prefill" in run.steps[0].prompt

    claude = FakeClaude(["VALUE = 7\nLATENCY = 3.0\n", "VALUE = 7\nLATENCY = 1.0\n"])
    monkeypatch.setattr("claude_agent_sdk.query", claude.query)
    result = asyncio.run(AveLoop(run).optimize())

    assert result["status"] == "completed"
    assert result["completed_steps"] == [step.id for step in steps]
    assert [context["prompt"] for context in claude.prompts] == [step.prompt for step in steps]
    assert result["final_case_ms"] == {"decode": 1.0, "prefill": 2.0}


@pytest.mark.parametrize("topic, error", [("missing_topic", FileNotFoundError), ("../optimizer", ValueError)])
def test_unknown_knowledge_cannot_silently_produce_a_prompt(topic, error):
    recipe = {
        "objective": "Optimize the kernel.",
        "knowledge": [topic],
        "steps": [{"name": "tile", "knowledge": [], "files": ["kernel.py"], "required": ["Tile the reduction."]}],
    }
    with pytest.raises(error):
        build_prompts(recipe)
