# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The worker context contract and its failure artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict


from ave import __main__ as cli
from ave.tests.test_loop import session_result
from hyperloom_common.git_safety import repo_git


def test_cli_passes_ordered_prompts_and_preserves_prefix_when_provider_fails(prepared_run, monkeypatch, capsys, caplog):
    context = json.dumps(asdict(prepared_run), default=str)
    path = prepared_run.result_json.parent / "run.json"
    path.write_text(context)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-subscription-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    prompts = []
    sources = []

    async def query(*, prompt, options):
        prompts.append(json.loads(prompt))
        assert set(options.tools) == {"Read", "Glob", "Grep", "Edit", "Write", "Bash"}
        assert options.cwd == str(prepared_run.workspace)
        assert options.env["CLAUDE_CODE_OAUTH_TOKEN"] == "test-subscription-token"
        assert not options.env.get("ANTHROPIC_API_KEY")
        assert not options.env.get("ANTHROPIC_AUTH_TOKEN")
        source = prepared_run.workspace / "kernel.py"
        sources.append(source.read_text())
        if len(prompts) == 1:
            source.write_text("VALUE = 7\nLATENCY = 1.0\n")
            yield session_result()
        else:
            source.write_text("VALUE = 8\nLATENCY = 0.5\n")
            yield session_result("subscription unavailable", is_error=True)

    monkeypatch.setattr("claude_agent_sdk.query", query)
    assert cli.main(["optimize", str(path)]) == 1
    assert "ave: failed" in capsys.readouterr().out
    assert "subscription unavailable" in caplog.text
    result = json.loads(prepared_run.result_json.read_text())
    assert result["status"] == "failed"
    assert result["correctness_passed"] is True
    assert result["completed_steps"] == ["prepare_layout"]
    assert [context["prompt"] for context in prompts] == [
        prepared_run.steps[0].prompt,
        *[prepared_run.steps[1].prompt] * 3,
    ]
    assert all("subscription unavailable" in attempt["reason"] for attempt in result["attempts"][1:])
    assert all(source == "VALUE = 7\nLATENCY = 1.0\n" for source in sources[1:])
    assert repo_git(prepared_run.workspace, "rev-parse", "HEAD").stdout.strip() == result["final_commit"]
    assert repo_git(prepared_run.workspace, "diff", "HEAD").stdout == ""
