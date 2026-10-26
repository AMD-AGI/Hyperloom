# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pr-review-publish.sh applies a review-pr card's verdict only when the card is the one this run should publish.

The script runs against a stub ``gh`` that records every call and answers ``gh pr view`` with the PR open at the
reviewed head; a refusal is the note it posts with ``gh pr comment``, and no review is submitted.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PUBLISH = _ROOT / ".github" / "scripts" / "pr-review-publish.sh"
_PR = 500
_HEAD = "a" * 40
_BASE = "b" * 40
_SKILL = "0123456789ab"
_MODEL = "model-under-test"
_SKILL_TXT = f"skill: {_SKILL}\nmodel: {_MODEL}\n"
_ARTIFACTS = (
    "rules.txt",
    "answers.txt",
    "core_files.txt",
    "verdicts.txt",
    "ai_diagnostic.txt",
    "refutations.txt",
    "independent.txt",
)

_STUB_GH = """#!{python}
import json
import os
import sys

with open(os.environ["GH_STUB_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["pr", "view"]:
    print(os.environ["GH_STUB_PR_NOW"])
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="pr-review-publish.sh needs bash")


@dataclass(frozen=True)
class Published:
    returncode: int
    calls: list[list[str]]
    stderr: str

    @property
    def note(self) -> str:
        """The first line of the note the script posted instead of a verdict, or empty."""
        bodies = [call[call.index("--body") + 1] for call in self.calls if call[:2] == ["pr", "comment"]]
        return bodies[0].split("\n", 1)[0] if bodies else ""

    @property
    def reviewed(self) -> bool:
        return any(call[:2] == ["pr", "review"] for call in self.calls)


def _card(*, blocking: bool = True, skill: str = _SKILL, model: str = _MODEL) -> str:
    verdict = (
        "1\n\n1. [C1] x is still 1 [verified]\n   Problem: x is 1 at src/pkg/mod.py:1\n   Impact: x is wrong\n"
        "   Action: set x to 2\n"
        if blocking
        else "none\n"
    )
    return (
        f"## PR #{_PR} -- fix: change mod\n\nWhat it does: changes mod.\n\nBlocking issues: {verdict}\n"
        f"Checked: src/pkg/mod.py | Ran: none | Skill: {skill} | Model: {model} | Base: {_BASE} | Head: {_HEAD}\n"
    )


def _publish(
    tmp_path: Path, card: str, *, skill_txt: str | None = _SKILL_TXT, files: dict[str, str] | None = None
) -> Published:
    """pr-review-publish.sh on ``card``, beside every artifact a review leaves, ``skill_txt`` unless it is None, and
    ``files``."""
    work = tmp_path / "work"
    work.mkdir()
    for name in _ARTIFACTS:
        (work / name).write_text("x\n", encoding="utf-8")
    if skill_txt is not None:
        (work / "skill.txt").write_text(skill_txt, encoding="utf-8")
    for name, text in (files or {}).items():
        (work / name).write_text(text, encoding="utf-8")
    (work / "card.md").write_text(card, encoding="utf-8")
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    gh = stub_dir / "gh"
    gh.write_text(_STUB_GH.replace("{python}", sys.executable), encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "gh.log"
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_STUB_LOG": str(log),
        "GH_STUB_PR_NOW": f"OPEN {_HEAD}",
        "REPO": "org/repo",
        "PR": str(_PR),
        "REVIEWED_SHA": _HEAD,
        "CARD": str(work / "card.md"),
        "RUN_URL": "https://example.invalid/run",
        "REVIEW_RESULT": "success",
    }
    result = subprocess.run(["bash", str(_PUBLISH)], env=env, capture_output=True, text=True, check=False, timeout=60)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return Published(result.returncode, calls, result.stderr)


def test_a_card_naming_the_skill_and_model_of_this_run_gets_its_verdict(tmp_path: Path) -> None:
    run = _publish(tmp_path, _card())

    assert run.returncode == 0, run.stderr
    assert run.reviewed
    assert run.note == ""


@pytest.mark.parametrize(
    ("card", "skill_txt", "reason"),
    [
        pytest.param(
            _card(model="unknown"), f"skill: {_SKILL}\nmodel: unknown\n", "skill.txt names no model", id="model-unknown"
        ),
        pytest.param(
            _card(skill=f"{_SKILL}+dirty"),
            f"skill: {_SKILL}+dirty\nmodel: {_MODEL}\n",
            "skill.txt names no clean skill version",
            id="dirty-skill",
        ),
        pytest.param(
            _card(skill="fedcba987654"), _SKILL_TXT, "does not name this run's Skill", id="another-skill-on-the-card"
        ),
        pytest.param(_card(), None, "the review left no skill.txt", id="no-skill-txt"),
    ],
)
def test_a_card_that_does_not_name_the_skill_and_model_of_this_run_gets_no_verdict(
    tmp_path: Path, card: str, skill_txt: str | None, reason: str
) -> None:
    run = _publish(tmp_path, card, skill_txt=skill_txt)

    assert run.returncode == 1
    assert reason in run.note
    assert not run.reviewed
