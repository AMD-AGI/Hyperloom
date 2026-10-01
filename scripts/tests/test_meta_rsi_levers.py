# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the findings and implement steps, with a fake agent in a throwaway repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from meta_rsi.rsi.agent import AgentResult
from meta_rsi.rsi.config import parse_config
from meta_rsi.rsi.pipeline import RoundContext, StepFailed
from meta_rsi.rsi.state import RoundState
from meta_rsi.rsi.steps import levers

LEVER = {
    "id": "trim-prompt",
    "title": "Trim the prompt",
    "problem": "The prompt repeats itself.",
    "evidence": [{"metric": "repeat share", "value": "25%", "source": "analysis/x.txt"}],
    "estimated_saving": {"basis": "orchestration weighted tokens", "pct": 12.5},
    "change": "Drop the repeat.",
    "files": ["src/lever.py"],
    "off_switch": "TRIM_PROMPT",
    "tests": ["tests/test_lever.py"],
    "risk": "low",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def ctx(rsi_config_dict, tmp_path) -> RoundContext:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "config", "user.email", "rsi@example.com")
    _git(repo, "config", "user.name", "rsi test")
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "src" / "base.py").write_text("BASE = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    rsi_config_dict["agent"]["implement_attempts"] = 2
    cfg = parse_config(rsi_config_dict)
    return RoundContext(config=cfg, state=RoundState.load(cfg.round_dir), log=lambda _m: None)


def _reply(payload: dict) -> AgentResult:
    return AgentResult(
        text=f"done\n```json\n{json.dumps(payload)}\n```", is_error=False, turns=2, cost_usd=0.1, usage={}
    )


def _committing_agent(test_body: str, commit: bool = True):
    def agent(spec):
        cand = spec.cwd
        (cand / "src" / "lever.py").write_text("VALUE = 1\n")
        (cand / "tests").mkdir(exist_ok=True)
        (cand / "tests" / "test_lever.py").write_text(
            f"# {spec.name}\nfrom lever import VALUE\n\n\ndef test_value():\n    {test_body}\n"
        )
        if commit:
            _git(cand, "add", "-A")
            _git(cand, "commit", "-qm", "perf(lever): trim the prompt")
        return _reply({"commit": "x", "tests": ["tests/test_lever.py"], "summary": "s"})

    return agent


class TestLeverProblems:
    def test_a_complete_lever_is_usable(self):
        assert levers.lever_problems({"levers": [LEVER]}, 4) == []

    def test_missing_fields_bad_ids_and_too_many_levers_are_reported(self):
        bad = {**LEVER, "id": "Not A Slug", "off_switch": ""}
        problems = levers.lever_problems({"levers": [bad, LEVER, LEVER]}, 2)
        text = "; ".join(problems)
        assert "3 levers proposed" in text and "lacks off_switch" in text and "kebab-case" in text

    def test_a_non_numeric_saving_is_rejected(self):
        bad = {**LEVER, "estimated_saving": {"basis": "x", "pct": "a lot"}}
        assert any("pct" in p for p in levers.lever_problems({"levers": [bad]}, 4))


class TestFindings:
    def test_a_usable_reply_writes_levers_and_findings(self, ctx):
        ctx.agent = lambda spec: _reply({"levers": [LEVER], "model_choices": []})
        assert levers.findings(ctx) == {"levers": ["trim-prompt"]}
        assert json.loads((ctx.round_dir / "levers.json").read_text())["levers"][0]["id"] == "trim-prompt"
        assert "Off switch: `TRIM_PROMPT`" in (ctx.round_dir / "findings.md").read_text()
        assert (ctx.worktree("base") / "src" / "base.py").exists()

    def test_one_bad_reply_is_retried_with_the_problem(self, ctx):
        replies = [_reply({"levers": []}), _reply({"levers": [LEVER]})]
        prompts_seen = []
        ctx.agent = lambda spec: prompts_seen.append(spec.prompt) or replies.pop(0)
        levers.findings(ctx)
        assert "no levers were proposed" in prompts_seen[1]

    def test_two_bad_replies_fail_the_step(self, ctx):
        ctx.agent = lambda spec: AgentResult(text="no json", is_error=False, turns=1, cost_usd=0.0, usage={})
        with pytest.raises(StepFailed, match="unusable"):
            levers.findings(ctx)


class TestImplement:
    def _start(self, ctx):
        levers.ensure_worktrees(ctx)
        cand = ctx.worktree("candidate")
        return cand, _git(cand, "rev-parse", "HEAD")

    def test_a_committed_lever_with_passing_tests_lands(self, ctx):
        cand, base = self._start(ctx)
        ctx.agent = _committing_agent("assert VALUE == 1")
        outcome = levers.implement_one(ctx, LEVER, base)
        assert outcome["status"] == "landed" and outcome["attempts"] == 1
        assert outcome["commits"][0].endswith("perf(lever): trim the prompt")
        assert _git(cand, "rev-parse", "--abbrev-ref", "HEAD") == "rsi/test"

    def test_failing_tests_drop_the_lever_and_its_commits(self, ctx):
        cand, base = self._start(ctx)
        ctx.agent = _committing_agent("assert VALUE == 2")
        outcome = levers.implement_one(ctx, LEVER, base)
        assert outcome["status"] == "dropped" and "tests failed" in outcome["reason"] and outcome["attempts"] == 2
        assert _git(cand, "rev-parse", "HEAD") == base and not (cand / "src" / "lever.py").exists()

    def test_uncommitted_work_is_not_accepted(self, ctx):
        _cand, base = self._start(ctx)
        ctx.agent = _committing_agent("assert VALUE == 1", commit=False)
        assert "no commit was made" in levers.implement_one(ctx, LEVER, base)["reason"]

    def test_the_step_resets_a_lever_left_running_by_a_dead_driver(self, ctx):
        cand, base = self._start(ctx)
        (ctx.round_dir / "levers.json").write_text(json.dumps({"levers": [LEVER]}))
        (cand / "src" / "half.py").write_text("x = 1\n")
        _git(cand, "add", "-A")
        _git(cand, "commit", "-qm", "half done")
        ctx.state.data["implement"] = {"trim-prompt": {"status": "running", "base": base}}
        ctx.agent = _committing_agent("assert VALUE == 1")
        assert levers.implement(ctx) == {"landed": ["trim-prompt"], "dropped": []}
        assert not (cand / "src" / "half.py").exists()
        assert _git(cand, "log", "--format=%s", f"{base}..HEAD") == "perf(lever): trim the prompt"
