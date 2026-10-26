# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""review-pr's fetch.sh names every cut in openprs.txt instead of reading it as no overlap.

fetch.sh runs against a stub ``gh`` that answers from a fake repository: the PR under
review with a one-file diff, its open PRs, which ``gh pr list`` serves newest first and
stops at ``--limit`` the way the real one does, and closed PRs that only a GraphQL count
without the open-state filter would include.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FETCH = _ROOT / ".claude" / "skills" / "review-pr" / "fetch.sh"
_REPO = "org/repo"
_PR = 500
_HEAD = "a" * 40
_BASE = "b" * 40
_CHANGED = "src/pkg/mod.py"

_STUB_GH = """#!{python}
import json
import os
import subprocess
import sys

with open(os.environ["GH_STUB_WORLD"], encoding="utf-8") as fh:
    world = json.load(fh)
args = sys.argv[1:]
valued = {"-H", "-f", "--jq", "--json", "--limit", "--repo", "--state", "--template"}
options, positional = {}, []
i = 0
while i < len(args):
    if args[i] in valued:
        options.setdefault(args[i], []).append(args[i + 1])
        i += 2
        continue
    if not args[i].startswith("-"):
        positional.append(args[i])
    i += 1


def emit(doc):
    if "--jq" not in options:
        sys.stdout.write(json.dumps(doc))
        return
    jq = ["jq", "-rc", options["--jq"][0]]
    sys.stdout.write(subprocess.run(jq, input=json.dumps(doc), capture_output=True, text=True, check=True).stdout)


endpoint = positional[1]
if positional[:2] == ["pr", "view"] and "--template" in options:
    sys.stdout.write(world["meta"])
elif positional[:2] == ["pr", "view"]:
    emit({"body": ""})
elif positional[:2] == ["pr", "list"]:
    emit(world["open_prs"][: int(options["--limit"][0])])
elif endpoint == "graphql":
    fields = dict(field.split("=", 1) for field in options["-f"])
    if f"{fields['owner']}/{fields['name']}" != world["repo"]:
        sys.exit("GraphQL: Could not resolve to a Repository")
    if world["count"] == "fails":
        sys.exit("GraphQL: Something went wrong while executing your query")
    total = len(world["open_prs"]) + (0 if "states: OPEN" in fields["query"] else world["closed_prs"])
    emit({"data": {"repository": {"pullRequests": {"totalCount": total if world["count"] == "answers" else None}}}})
elif "-H" in options:
    sys.stdout.write(world["diff"])
elif "--slurp" in args:
    emit([[]])
elif endpoint.endswith("/check-runs"):
    emit({"check_runs": []})
elif endpoint.endswith("/status"):
    emit({"statuses": []})
elif "/compare/" in endpoint:
    emit(world["compare"])
else:
    sys.exit(f"gh stub: no answer for {args}")
"""

pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="fetch.sh needs jq")


def _open_pr(number: int, *paths: str, changed: int = 0) -> dict[str, object]:
    return {
        "number": number,
        "title": f"change {number}",
        "url": f"https://github.com/{_REPO}/pull/{number}",
        "files": [{"path": path} for path in paths],
        "changedFiles": changed or len(paths),
    }


def _open_prs(total: int) -> list[dict[str, object]]:
    """``total`` open PRs, newest first: the one under review, then others; only the oldest overlaps it."""
    others = [_open_pr(_PR - i, f"other/{i}.py") for i in range(1, total - 1)]
    return [_open_pr(_PR, _CHANGED), *others, _open_pr(_PR - total + 1, _CHANGED)]


def _run(
    tmp_path: Path,
    open_prs: list[dict[str, object]],
    count: Literal["answers", "null", "fails"] = "answers",
    *,
    env: dict[str, str] | None = None,
    fetch: Path = _FETCH,
) -> subprocess.CompletedProcess[str]:
    """``fetch`` (the skill's fetch.sh by default) for the PR under review, into ``tmp_path / "work"``; ``count`` is
    how the open-PR count replies, and ``env`` adds to fetch.sh's environment."""
    world = tmp_path / "world.json"
    world.write_text(
        json.dumps(
            {
                "repo": _REPO,
                "meta": (
                    f"number: {_PR}\ntitle: fix: change mod\nauthor: alice\nstate: OPEN\nhead: {_HEAD}\n"
                    f"base_ref: main\nbase_tip: {_BASE}\nurl: https://github.com/{_REPO}/pull/{_PR}\n"
                    "mergeable: MERGEABLE\n"
                ),
                "compare": {
                    "merge_base_commit": {"sha": _BASE},
                    "total_commits": 1,
                    "commits": [{"commit": {"message": "fix: change mod"}}],
                },
                "diff": f"diff --git a/{_CHANGED} b/{_CHANGED}\n--- a/{_CHANGED}\n+++ b/{_CHANGED}\n"
                "@@ -1 +1 @@\n-x = 1\n+x = 2\n",
                "open_prs": open_prs,
                "closed_prs": 40,
                "count": count,
            }
        ),
        encoding="utf-8",
    )
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    gh = stub_dir / "gh"
    gh.write_text(_STUB_GH.replace("{python}", sys.executable), encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    run_env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "HL_REPO": _REPO,
        "GH_STUB_WORLD": str(world),
        **(env or {}),
    }
    return subprocess.run(
        ["bash", str(fetch), str(_PR), str(tmp_path / "work")],
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _fetch(tmp_path: Path, open_prs: list[dict[str, object]]) -> str:
    """openprs.txt as fetch.sh writes it for the PR under review."""
    result = _run(tmp_path, open_prs)
    assert result.returncode == 0, result.stderr
    return (tmp_path / "work" / "openprs.txt").read_text(encoding="utf-8")


def test_an_overlap_past_the_open_pr_limit_reads_partial_not_absent(tmp_path: Path) -> None:
    assert _fetch(tmp_path, _open_prs(150)) == (
        f"# PARTIAL: {_REPO} has 150 open PRs, only the newest 100 were compared\n"
    )


@pytest.mark.parametrize("total", [3, 100])
def test_a_list_within_the_limit_is_compared_whole(tmp_path: Path, total: int) -> None:
    oldest = _PR - total + 1
    row = f"#{oldest}  1 overlapping file(s)  change {oldest}  https://github.com/{_REPO}/pull/{oldest}"
    assert _fetch(tmp_path, _open_prs(total)) == f"{row}\n    {_CHANGED}\n"


def test_a_pr_whose_file_list_was_cut_is_named_partial(tmp_path: Path) -> None:
    large = _open_pr(_PR - 1, *(f"other/{i}.py" for i in range(100)), changed=250)
    assert _fetch(tmp_path, [_open_pr(_PR, _CHANGED), large]) == (
        f"# PARTIAL: #{_PR - 1} changes 250 files, only its first 100 were compared\n"
    )


@pytest.mark.parametrize("count", ["fails", "null"])
def test_a_full_list_whose_open_total_cannot_be_read_stops_the_fetch(
    tmp_path: Path, count: Literal["fails", "null"]
) -> None:
    result = _run(tmp_path, _open_prs(150), count)
    assert result.returncode != 0
    assert not (tmp_path / "work" / "openprs.txt").exists()


def test_skill_txt_names_the_model_and_the_skill_directory_tree_that_ran(tmp_path: Path) -> None:
    result = _run(tmp_path, _open_prs(3), env={"ANTHROPIC_MODEL": "model-under-test"})
    assert result.returncode == 0, result.stderr

    skill, model = (tmp_path / "work" / "skill.txt").read_text(encoding="utf-8").splitlines()
    tree = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "--short=12", "HEAD:.claude/skills/review-pr"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert skill in (f"skill: {tree}", f"skill: {tree}+dirty")
    assert model == "model: model-under-test"


def test_a_skill_directory_git_cannot_name_stops_the_fetch(tmp_path: Path) -> None:
    copy = tmp_path / "skill" / "fetch.sh"
    copy.parent.mkdir()
    shutil.copy(_FETCH, copy)

    result = _run(tmp_path, _open_prs(3), fetch=copy)

    assert result.returncode != 0
    assert "cannot name this skill's version" in result.stderr
    assert not (tmp_path / "work" / "skill.txt").exists()
