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
elif any(arg.endswith("/files") for arg in sys.argv):
    sys.stdout.write(os.environ["GH_STUB_FILES"])
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="pr-review-publish.sh needs bash")


@dataclass(frozen=True)
class Published:
    returncode: int
    calls: list[list[str]]
    stderr: str

    @property
    def comments(self) -> list[str]:
        return [call[call.index("--body") + 1] for call in self.calls if call[:2] == ["pr", "comment"]]

    @property
    def note(self) -> str:
        """The first line of the note the script posted instead of a verdict, or empty."""
        notes = [body for body in self.comments if body.startswith("Automated review: no verdict applied")]
        return notes[0].split("\n", 1)[0] if notes else ""

    @property
    def reviewed(self) -> bool:
        """Whether a request-changes review was submitted."""
        return any(call[:2] == ["pr", "review"] for call in self.calls)

    @property
    def lgtm(self) -> bool:
        return any(body.endswith("\n\nLGTM") for body in self.comments)


def _card(
    *,
    blocking: bool = True,
    skill: str = _SKILL,
    model: str = _MODEL,
    prior: str | None = None,
    head: str = _HEAD,
    trim: str | None = None,
    risk: str = "Risk: medium -- src/pkg/mod.py is runtime code\nDecisions for a human: none -- no operator-visible change",
) -> str:
    verdict = (
        "1\n\n1. [C1] x is still 1 [verified]\n   Problem: x is 1 at src/pkg/mod.py:1\n   Impact: x is wrong\n"
        "   Action: set x to 2\n"
        if blocking
        else "none\n"
    )
    trims = f"Trim in this PR:\n{trim}\n\n" if trim is not None else ""
    fields = f"Skill: {skill} | Model: {model}" + (f" | Prior: {prior}" if prior is not None else "")
    return (
        f"## PR #{_PR} -- fix: change mod\n\nWhat it does: changes mod.\n\n{risk}\n\nBlocking issues: {verdict}\n"
        f"{trims}Checked: src/pkg/mod.py | Ran: none | {fields} | Base: {_BASE} | Head: {head}\n"
    )


_TRIM = "- src/pkg/mod.py:40-58 -- delete the legacy branch -- git grep -n legacy_mode: no reader outside it"


#: comments.txt holding a card an earlier review posted on an earlier head.
_EARLIER = f"[REVIEW bob CHANGES_REQUESTED 2026-10-24T02:00:00Z]\n{_card(head='c' * 40)}\n"


def _publish(
    tmp_path: Path,
    card: str,
    *,
    skill_txt: str | None = _SKILL_TXT,
    comments: str | None = "",
    files: dict[str, str] | None = None,
    changed: str = "src/pkg/mod.py\n",
) -> Published:
    """pr-review-publish.sh on ``card``, beside every artifact a review leaves, ``skill_txt`` and ``comments`` as
    skill.txt and comments.txt unless they are None, and ``files``; GitHub lists ``changed`` as the PR's files."""
    work = tmp_path / "work"
    work.mkdir()
    for name in _ARTIFACTS:
        (work / name).write_text("x\n", encoding="utf-8")
    if skill_txt is not None:
        (work / "skill.txt").write_text(skill_txt, encoding="utf-8")
    if comments is not None:
        (work / "comments.txt").write_text(comments, encoding="utf-8")
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
        "GH_STUB_FILES": changed,
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


@pytest.mark.parametrize("skill", [_SKILL, "0123456789abc"], ids=["twelve", "longer-when-ambiguous"])
def test_a_card_naming_the_skill_and_model_of_this_run_gets_its_verdict(tmp_path: Path, skill: str) -> None:
    run = _publish(tmp_path, _card(skill=skill), skill_txt=f"skill: {skill}\nmodel: {_MODEL}\n")

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


@pytest.mark.parametrize("blocking", [True, False], ids=["blocking", "clean"])
def test_a_first_review_gets_its_verdict_beside_another_prs_card(tmp_path: Path, blocking: bool) -> None:
    other = f"[COMMENT bob 2026-10-24T02:00:00Z]\n{_card().replace(f'#{_PR} ', f'#{_PR}1 ')}\n"

    run = _publish(tmp_path, _card(blocking=blocking), comments=other)

    assert run.returncode == 0, run.stderr
    assert (run.reviewed, run.lgtm) == (blocking, not blocking)
    assert run.note == ""


@pytest.mark.parametrize(
    ("verdicts", "prior", "blocking"),
    [
        pytest.param("C1 FIXED -- src/pkg/mod.py:1 now sets x to 2\n", "C1 fixed", False, id="fixed"),
        pytest.param(
            "C1 HOLDS -- src/pkg/mod.py:1 still sets x to 1\nT2 FIXED -- tests/test_mod.py:4 fails on the base\n",
            "C1 holds, T2 fixed",
            True,
            id="holds-and-fixed",
        ),
        pytest.param("C1 MOVED -- src/pkg/other.py:3 still sets x to 1\n", "C1 moved", True, id="moved-on-the-card"),
        pytest.param("none -- the latest card reported no finding\n", "none", False, id="none"),
    ],
)
def test_a_re_review_that_settles_the_earlier_card_gets_its_verdict(
    tmp_path: Path, verdicts: str, prior: str, blocking: bool
) -> None:
    run = _publish(
        tmp_path, _card(blocking=blocking, prior=prior), comments=_EARLIER, files={"prior_verdicts.txt": verdicts}
    )

    assert run.returncode == 0, run.stderr
    assert (run.reviewed, run.lgtm) == (blocking, not blocking)


@pytest.mark.parametrize(
    ("verdicts", "prior", "reason"),
    [
        pytest.param(None, "C1 fixed", "produced no prior_verdicts.txt", id="no-verdicts"),
        pytest.param("C1 is fixed now\n", "C1 fixed", "prior_verdicts.txt is neither", id="malformed-line"),
        pytest.param(
            "none -- nothing\nC1 FIXED -- src/pkg/mod.py:1\n",
            "none, C1 fixed",
            "prior_verdicts.txt is neither",
            id="none-and-more",
        ),
        pytest.param("C1 FIXED -- src/pkg/mod.py:1\n", "C1 holds", "expected 'Prior: C1 fixed'", id="prior-disagrees"),
        pytest.param("C1 FIXED -- src/pkg/mod.py:1\n", None, "expected 'Prior: C1 fixed'", id="no-prior"),
        pytest.param(
            "C1 HOLDS -- src/pkg/mod.py:1 still sets x to 1\n",
            "C1 holds",
            "still holds but the card does not list it",
            id="holds-dropped",
        ),
        pytest.param(
            "C1 MOVED -- src/pkg/other.py:3 still sets x to 1\n",
            "C1 moved",
            "still holds but the card does not list it",
            id="moved-dropped",
        ),
    ],
)
def test_a_re_review_that_does_not_settle_the_earlier_card_gets_no_verdict(
    tmp_path: Path, verdicts: str | None, prior: str | None, reason: str
) -> None:
    files = {"prior_verdicts.txt": verdicts} if verdicts is not None else {}
    # A dropped HOLDS/MOVED key is refused even on a clean card; the other cases keep the default blocking card.
    card = _card(blocking=False, prior=prior) if reason.startswith("still holds") else _card(prior=prior)

    run = _publish(tmp_path, card, comments=_EARLIER, files=files)

    assert run.returncode == 1
    assert reason in run.note
    assert not run.reviewed


@pytest.mark.parametrize("blocking", [True, False], ids=["blocking", "clean"])
def test_a_trim_list_rides_with_the_verdict_without_counting_toward_it(tmp_path: Path, blocking: bool) -> None:
    trim = "\n".join([_TRIM, "- tests/test_mod.py:12 -- delete the repeat -- tests/test_cli.py::test_x asserts it"])

    run = _publish(tmp_path, _card(blocking=blocking, trim=trim))

    assert run.returncode == 0, run.stderr
    assert (run.reviewed, run.lgtm) == (blocking, not blocking)
    if not blocking:
        assert f"Trim in this PR:\n{trim}\n" in run.comments[0]


@pytest.mark.parametrize(
    ("trim", "reason"),
    [
        pytest.param("", "holds 0 line(s)", id="empty"),
        pytest.param(
            "\n".join(f"- src/pkg/m{i}.py:{i} -- delete f{i} -- git grep -n f{i}: no caller" for i in range(6)),
            "holds 6 line(s)",
            id="six",
        ),
        pytest.param(
            "- src/pkg/mod.py:40 -- delete the legacy branch", "every 'Trim in this PR:' line", id="no-evidence"
        ),
        pytest.param("- the legacy branch -- delete it -- no reader", "every 'Trim in this PR:' line", id="no-place"),
        pytest.param(f"{_TRIM}\n\nTrim in this PR:\n{_TRIM}", "more than one 'Trim in this PR:' list", id="twice"),
    ],
)
def test_a_malformed_trim_list_gets_no_verdict(tmp_path: Path, trim: str, reason: str) -> None:
    run = _publish(tmp_path, _card(blocking=False, trim=trim))

    assert run.returncode == 1
    assert reason in run.note
    assert not run.lgtm


def test_the_risk_floor_comes_from_the_prs_files_on_github_not_from_the_review_job(tmp_path: Path) -> None:
    low = "Risk: low -- the paths are prose"

    run = _publish(
        tmp_path,
        _card(blocking=False, risk=low),
        files={"risk.txt": "risk: low\nlow docs/guide.md -- prose\n"},
        changed=".github/workflows/lint.yml\n",
    )

    assert run.returncode == 1
    assert "the card's Risk low is below high" in run.note
    assert not run.lgtm


def test_a_card_naming_a_prior_on_a_pr_with_no_earlier_card_gets_no_verdict(tmp_path: Path) -> None:
    run = _publish(tmp_path, _card(prior="none"), files={"prior_verdicts.txt": "none -- invented\n"})

    assert run.returncode == 1
    assert "names a Prior but the PR has no earlier card" in run.note


def test_a_review_that_left_no_comments_txt_gets_no_verdict(tmp_path: Path) -> None:
    run = _publish(tmp_path, _card(), comments=None)

    assert run.returncode == 1
    assert "left no comments.txt" in run.note
