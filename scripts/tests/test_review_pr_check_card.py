# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""review-pr's check-card.sh holds a card to the card contract before anyone publishes it, from CI or a local run.

These pin the parts the risk grade adds: the card's Risk never falls below risk.txt and says when it was raised, and a
medium- or high-risk card lists what a person must decide, naming every high-risk path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_CHECK = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "review-pr" / "check-card.sh"
_PR = 500
_HEAD = "a" * 40
_BASE = "b" * 40
_ARTIFACTS = ("rules.txt", "answers.txt", "core_files.txt", "verdicts.txt", "ai_diagnostic.txt", "refutations.txt")
_HIGH = (
    "risk: high\n"
    "high .github/workflows/docs.yml -- CI workflows\n"
    "high .github/workflows/lint.yml -- CI workflows\n"
    "medium src/pkg/mod.py -- runtime code\n"
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="check-card.sh needs bash")


def _check(tmp_path: Path, risk_block: str, risk_txt: str, *, skill: str = "0123456789ab+dirty") -> tuple[int, str]:
    """check-card.sh on a clean card carrying ``risk_block``, held to ``risk_txt``, as a local run leaves it: a skill
    with uncommitted edits, and a model the environment did not name but the card does."""
    work = tmp_path / "work"
    work.mkdir()
    for name in (*_ARTIFACTS, "independent.txt"):
        (work / name).write_text("x\n", encoding="utf-8")
    (work / "comments.txt").write_text("", encoding="utf-8")
    (work / "skill.txt").write_text(f"skill: {skill}\nmodel: unknown\n", encoding="utf-8")
    (work / "risk.txt").write_text(risk_txt, encoding="utf-8")
    (work / "card.md").write_text(
        f"## PR #{_PR} -- fix: change mod\n\nWhat it does: changes mod.\n\n{risk_block}\n\nBlocking issues: none\n\n"
        f"Checked: src/pkg/mod.py | Ran: none | Skill: {skill} | Model: model-named-by-the-agent | Base: {_BASE} | "
        f"Head: {_HEAD}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(_CHECK), str(work), str(_PR), _HEAD, str(work / "risk.txt")],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result.returncode, result.stdout.strip()


_MEDIUM = "risk: medium\nmedium src/pkg/mod.py -- runtime code\n"
_NONE = "Decisions for a human: none -- no operator-visible change"
_DECIDE = "Decisions for a human:\n- src/pkg/mod.py -- the default x becomes 2 -- should runs keep x = 1?"


@pytest.mark.parametrize(
    ("risk_block", "risk_txt"),
    [
        pytest.param(f"Risk: medium -- runtime code\n{_NONE}", _MEDIUM, id="medium-at-the-floor"),
        pytest.param(f"Risk: high (raised from medium) -- it widens a gate\n{_DECIDE}", _MEDIUM, id="raised"),
        pytest.param("Risk: low -- prose only", "risk: low\nlow docs/guide.md -- prose\n", id="low-without-decisions"),
        pytest.param(
            "Risk: high -- CI\nDecisions for a human:\n"
            "- .github/workflows/docs.yml, .github/workflows/lint.yml -- both gates run on PRs -- keep them required?",
            _HIGH,
            id="high-paths-named",
        ),
        pytest.param(
            "Risk: high -- CI\nDecisions for a human:\n- `.github/` -- two workflows change -- accept the new gates?",
            _HIGH,
            id="high-paths-under-a-named-directory",
        ),
    ],
)
def test_a_card_holding_its_risk_and_decisions_passes(tmp_path: Path, risk_block: str, risk_txt: str) -> None:
    assert _check(tmp_path, risk_block, risk_txt) == (0, "")


@pytest.mark.parametrize(
    ("risk_block", "risk_txt", "defect"),
    [
        pytest.param(_NONE, _MEDIUM, "no single 'Risk:' line", id="no-risk"),
        pytest.param("Risk: severe -- x\n" + _NONE, _MEDIUM, "Risk line is not", id="unknown-level"),
        pytest.param("Risk: low -- prose", _MEDIUM, "Risk low is below medium", id="below-the-floor"),
        pytest.param(f"Risk: high -- a gate\n{_DECIDE}", _MEDIUM, "must say '(raised from medium)'", id="unmarked"),
        pytest.param(
            f"Risk: medium (raised from low) -- x\n{_NONE}", _MEDIUM, "raises its Risk from low", id="no-rise"
        ),
        pytest.param("Risk: medium -- runtime code", _MEDIUM, "one 'Decisions for a human:' list", id="no-decisions"),
        pytest.param(
            f"Risk: low -- prose\n{_NONE}", "risk: low\nlow docs/guide.md -- prose\n", "low-risk card has no", id="low"
        ),
        pytest.param(f"Risk: high -- CI\n{_NONE}", _HIGH, "names at least one decision", id="high-with-none"),
        pytest.param(
            "Risk: high -- CI\nDecisions for a human:\n- .github/workflows/lint.yml -- lint changes -- keep it?",
            _HIGH,
            "path .github/workflows/docs.yml is named by no decision",
            id="high-path-unnamed",
        ),
        pytest.param(
            "Risk: high -- CI\nDecisions for a human:\n- .github -- two workflows change -- accept them?",
            _HIGH,
            "is named by no decision",
            id="directory-without-its-slash",
        ),
        pytest.param(
            "Risk: medium -- runtime code\nDecisions for a human:\n- src/pkg/mod.py -- x becomes 2",
            _MEDIUM,
            "every decision is",
            id="decision-without-a-question",
        ),
    ],
)
def test_a_card_that_does_not_hold_its_risk_or_decisions_is_refused(
    tmp_path: Path, risk_block: str, risk_txt: str, defect: str
) -> None:
    code, printed = _check(tmp_path, risk_block, risk_txt)

    assert code == 1
    assert defect in printed
