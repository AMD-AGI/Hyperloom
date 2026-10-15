# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The check-prior-art scripts record what they searched, so a short result never reads as "none".

Each script runs against a stub ``gh`` that answers from a fake GitHub, and applies ``--jq`` with
the real ``jq`` as gh does: search results keyed by the exact query; open pull requests that
GraphQL pages by cursor the way the real API does (synthetic ones fill a repository up to its open
total); and per-PR detail for the deep read and for pull requests the open scan lists again. An
answer can fail. A stub ``sleep`` returns at once.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_SKILL = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "check-prior-art"
_UPSTREAM = "up/engine"
_KERNELS = "up/kernels"

_STUB_GH = """#!{python} -S
import json, os, re, subprocess, sys

world = json.load(open(os.environ["GH_STUB_WORLD"], encoding="utf-8"))
args = sys.argv[1:]
fields, options, positional = {}, {}, []
while args:
    flag = args.pop(0)
    if flag == "-f":
        key, _, value = args.pop(0).partition("=")
        fields[key] = value
    elif flag in ("-X", "-R", "--jq", "--json"):
        options[flag] = args.pop(0)
    elif not flag.startswith("-"):
        positional.append(flag)
endpoint = positional[1].split("?")[0]


def fail(message):
    sys.stderr.write(message + "\\n")
    sys.exit(1)


def answer(payload):
    text = json.dumps(payload)
    if "--jq" in options:
        jq = subprocess.run(["jq", "-r", options["--jq"]], input=text, capture_output=True, text=True, check=True)
        text = jq.stdout
    sys.stdout.write(text)


if positional[0] == "pr":
    pr = world["prs"][options["-R"] + "#" + positional[2]]
    if positional[1] == "diff" and pr["diff"] is None:
        fail("could not find pull request diff: HTTP 406: Sorry, the diff exceeded the maximum number of files (300).")
    sys.stdout.write(pr["diff"] if positional[1] == "diff" else json.dumps(pr["view"]))
elif endpoint == "search/issues":
    hits = world["search"].get(fields["q"], {"total_count": 0, "incomplete_results": False, "items": []})
    if "error" in hits:
        sys.stdout.write(json.dumps({"message": hits["error"]}))
        fail("gh: " + hits["error"])
    answer(hits)
elif endpoint == "graphql":
    repo = world["open"][fields["owner"] + "/" + fields["name"]]
    start = int(fields.get("cursor", "0"))
    if start == repo.get("fail_at"):
        fail("gh: Something went wrong while executing your query.")
    size = int(re.search(r"pullRequests\\([^)]*first:\\s*(\\d+)", fields["query"]).group(1))
    end = min(start + size, repo["total"])
    listed = repo["prs"]
    nodes = [
        listed[i] if i < len(listed) else {
            "number": 90000 - i, "title": "pr %d" % i, "url": "u/%d" % i, "isDraft": False,
            "updatedAt": "2026-01-01T00:00:00Z", "changedFiles": 1,
            "files": {"nodes": [{"path": "other/%d.py" % i, "changeType": "MODIFIED"}]},
        }
        for i in range(start, end)
    ]
    page = {"hasNextPage": end < repo["total"], "endCursor": str(end)}
    answer({"data": {"repository": {"pullRequests": {"totalCount": repo["total"], "pageInfo": page, "nodes": nodes}}}})
else:
    _, owner, name, kind, number, detail = endpoint.split("/")
    pr = world["prs"][owner + "/" + name + "#" + number]
    if kind + "/" + detail in pr.get("fail", ()):
        fail("gh: Not Found (HTTP 404)")
    answer(pr[kind + "/" + detail])
"""

_STUB_SLEEP = """#!/bin/sh
"""

pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="the check-prior-art scripts need jq")


def _run(
    tmp_path: Path, script: str, world: dict, inputs: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    """``script`` against the stub over ``world``, with ``inputs`` written into the work directory first."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    for name, text in inputs.items():
        (work / name).write_text(text, encoding="utf-8")
    (tmp_path / "world.json").write_text(json.dumps(world), encoding="utf-8")
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    for name, text in (("gh", _STUB_GH.replace("{python}", sys.executable)), ("sleep", _STUB_SLEEP)):
        stub = stub_dir / name
        stub.write_text(text, encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_STUB_WORLD": str(tmp_path / "world.json"),
    }
    return subprocess.run(
        ["bash", str(_SKILL / script), str(work), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _artifact(tmp_path: Path, name: str) -> str:
    return (tmp_path / "work" / name).read_text(encoding="utf-8")


def _scan(tmp_path: Path) -> dict[str, tuple[str, list[int]]]:
    """scan.txt by repository or PR: its status and the numbers its line states, whatever the wording."""
    lines = {}
    for line in _artifact(tmp_path, "scan.txt").splitlines():
        status, rest = line.split(": ", 1)
        subject, _, prose = rest.partition(" ")
        lines[subject.rstrip(":")] = (status, [int(number) for number in re.findall(r"\d+", prose)])
    return lines


def _item(number: int, *, merged: bool = False, draft: bool = False, updated: str = "2026-05-01T00:00:00Z") -> dict:
    """One search hit as the issues search returns a pull request."""
    return {
        "number": number,
        "state": "closed" if merged else "open",
        "draft": draft,
        "title": f"change\t{number}",
        "html_url": f"https://example.test/{number}",
        "updated_at": updated,
        "pull_request": {"merged_at": "2026-04-01T00:00:00Z" if merged else None},
    }


def _hits(*items: dict, total: int | None = None, incomplete: bool = False) -> dict:
    return {
        "total_count": len(items) if total is None else total,
        "incomplete_results": incomplete,
        "items": list(items),
    }


def _query(state: str, terms: str, repo: str = _UPSTREAM) -> str:
    return f"repo:{repo} is:pr is:{state} {terms}"


def test_a_pr_several_queries_found_is_one_candidate_ranked_by_how_many(tmp_path: Path) -> None:
    world = {
        "search": {
            _query("open", "moe padding"): _hits(_item(11), _item(13, draft=True, updated="2026-06-01T00:00:00Z")),
            _query("open", '"align_block"'): _hits(_item(11)),
            _query("merged", "moe padding"): _hits(_item(12, merged=True)),
        }
    }
    result = _run(
        tmp_path, "search.sh", world, {"repos.txt": f"{_UPSTREAM}\n", "terms.txt": 'moe padding\n"align_block"\n'}
    )
    assert result.returncode == 0, result.stderr
    assert _artifact(tmp_path, "candidates.tsv").splitlines() == [
        "repo\tnumber\tstate\tdraft\ttitle\turl\tupdated\thits",
        f"{_UPSTREAM}\t11\topen\tfalse\tchange\\t11\thttps://example.test/11\t2026-05-01T00:00:00Z\t2",
        f"{_UPSTREAM}\t13\topen\ttrue\tchange\\t13\thttps://example.test/13\t2026-06-01T00:00:00Z\t1",
        f"{_UPSTREAM}\t12\tmerged\tfalse\tchange\\t12\thttps://example.test/12\t2026-05-01T00:00:00Z\t1",
    ]


def test_every_query_is_recorded_with_what_it_covered_and_a_failure_does_not_stop_the_rest(tmp_path: Path) -> None:
    world = {
        "search": {
            _query("open", "moe"): _hits(_item(1), total=340),
            _query("merged", "moe"): _hits(_item(2, merged=True), incomplete=True),
            _query("open", "moe", _KERNELS): {"error": "Validation Failed (HTTP 422)"},
            _query("merged", "moe", _KERNELS): _hits(_item(3, merged=True)),
        }
    }
    result = _run(
        tmp_path, "search.sh", world, {"repos.txt": f"{_UPSTREAM}\n# skipped\n\n{_KERNELS}\n", "terms.txt": "moe\n"}
    )
    assert result.returncode == 0, result.stderr
    assert _artifact(tmp_path, "queries.tsv").splitlines() == [
        "status\tlisted\ttotal\tquery\tnote",
        f"TRUNCATED\t1\t340\t{_query('open', 'moe')}\t",
        f"INCOMPLETE\t1\t1\t{_query('merged', 'moe')}\t",
        f"FAILED\t-\t-\t{_query('open', 'moe', _KERNELS)}\tgh: Validation Failed (HTTP 422)",
        f"OK\t1\t1\t{_query('merged', 'moe', _KERNELS)}\t",
    ]


@pytest.mark.parametrize(
    ("repos", "message"),
    [("", "names no repository"), ("https://github.com/up/engine\n", "not an owner/name repository")],
)
def test_a_search_with_no_valid_repository_stops_before_any_query(tmp_path: Path, repos: str, message: str) -> None:
    result = _run(tmp_path, "search.sh", {"search": {}}, {"repos.txt": repos, "terms.txt": "moe\n"})
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "work" / "queries.tsv").exists()


def _open_pr(number: int, *paths: str, changed: int = 0, change: str = "MODIFIED") -> dict:
    """One open pull request as the GraphQL page lists it: a renamed file by its new path only."""
    return {
        "number": number,
        "title": f"open {number}",
        "url": f"u/{number}",
        "isDraft": False,
        "updatedAt": "2026-07-01T00:00:00Z",
        "changedFiles": changed or len(paths),
        "files": {"nodes": [{"path": path, "changeType": change} for path in paths]},
    }


def _overlap(tmp_path: Path, open_prs: dict, paths: str, rest: dict | None = None) -> subprocess.CompletedProcess[str]:
    """overlap.sh over ``open_prs``; ``rest`` holds the REST file lists of the PRs it lists again."""
    repos = "".join(f"{repo}\n" for repo in open_prs)
    world = {"open": open_prs, "prs": rest or {}}
    return _run(tmp_path, "overlap.sh", world, {"repos.txt": repos, "paths.txt": paths})


def test_open_prs_on_a_planned_file_or_directory_are_rows_and_every_repository_says_how_far_it_was_compared(
    tmp_path: Path,
) -> None:
    prs = [
        _open_pr(7, "pkg/moe.py", "docs/x.md"),
        _open_pr(6, "pkg/ops/a.py", "pkg/ops/b.py"),
        _open_pr(5, "pkg/other.py"),
    ]
    open_prs = {_UPSTREAM: {"total": 230, "prs": prs}, _KERNELS: {"total": 3, "prs": []}}
    result = _overlap(tmp_path, open_prs, f"{_UPSTREAM}\tpkg/moe.py\n{_UPSTREAM}\tpkg/ops/\n")
    assert result.returncode == 0, result.stderr
    assert _artifact(tmp_path, "overlap.tsv").splitlines() == [
        "repo\tnumber\tstate\tdraft\ttitle\turl\tupdated\toverlap\tpaths",
        f"{_UPSTREAM}\t7\topen\tfalse\topen 7\tu/7\t2026-07-01T00:00:00Z\t1\tpkg/moe.py",
        f"{_UPSTREAM}\t6\topen\tfalse\topen 6\tu/6\t2026-07-01T00:00:00Z\t2\tpkg/ops/a.py pkg/ops/b.py",
    ]
    assert _scan(tmp_path) == {_UPSTREAM: ("COMPLETE", [230, 2]), _KERNELS: ("SKIPPED", [])}


def test_a_repository_past_the_page_cap_reads_partial_not_absent(tmp_path: Path) -> None:
    result = _overlap(tmp_path, {_UPSTREAM: {"total": 10_050, "prs": []}}, f"{_UPSTREAM}\tpkg/moe.py\n")
    assert result.returncode == 0, result.stderr
    assert _scan(tmp_path) == {_UPSTREAM: ("PARTIAL", [10000, 10050])}


def test_a_pr_graphql_cannot_settle_is_compared_on_its_rest_file_list_old_paths_included(tmp_path: Path) -> None:
    moved = _open_pr(8, "pkg/moe/impl.py", change="RENAMED")
    large = _open_pr(9, *(f"other/{i}.py" for i in range(100)), changed=101)
    elsewhere = _open_pr(10, "docs/new.md", change="RENAMED")
    rest = {
        f"{_UPSTREAM}#8": {"pulls/files": [{"filename": "pkg/moe/impl.py", "previous_filename": "pkg/moe.py"}]},
        f"{_UPSTREAM}#9": {
            "pulls/files": [{"filename": f"other/{i}.py"} for i in range(100)] + [{"filename": "pkg/ops/late.py"}]
        },
        f"{_UPSTREAM}#10": {"pulls/files": [{"filename": "docs/new.md", "previous_filename": "docs/old.md"}]},
    }
    open_prs = {_UPSTREAM: {"total": 3, "prs": [moved, large, elsewhere]}}
    result = _overlap(tmp_path, open_prs, f"{_UPSTREAM}\tpkg/moe.py\n{_UPSTREAM}\tpkg/ops/\n", rest)
    assert result.returncode == 0, result.stderr
    rows = [row.split("\t") for row in _artifact(tmp_path, "overlap.tsv").splitlines()[1:]]
    assert [(cells[1], cells[-1]) for cells in rows] == [("8", "pkg/moe.py->pkg/moe/impl.py"), ("9", "pkg/ops/late.py")]
    assert _scan(tmp_path) == {_UPSTREAM: ("COMPLETE", [3, 2])}


def test_a_pr_whose_overlap_cannot_be_decided_is_named_partial(tmp_path: Path) -> None:
    huge = _open_pr(11, *(f"other/{i}.py" for i in range(100)), changed=3500)
    unreadable = _open_pr(12, "lib/new.py", change="RENAMED")
    rest = {
        f"{_UPSTREAM}#11": {"pulls/files": [{"filename": f"other/{i}.py"} for i in range(300)]},
        f"{_UPSTREAM}#12": {"fail": ["pulls/files"]},
    }
    open_prs = {_UPSTREAM: {"total": 2, "prs": [huge, unreadable]}}
    result = _overlap(tmp_path, open_prs, f"{_UPSTREAM}\tpkg/moe.py\n", rest)
    assert result.returncode == 0, result.stderr
    scan = _scan(tmp_path)
    assert {subject: status for subject, (status, _) in scan.items()} == {
        _UPSTREAM: "COMPLETE",
        f"{_UPSTREAM}#11": "PARTIAL",
        f"{_UPSTREAM}#12": "PARTIAL",
    }
    assert scan[_UPSTREAM][1] == [2, 1]
    assert scan[f"{_UPSTREAM}#11"][1] == [3500, 300]


def test_a_failed_page_keeps_what_was_compared_and_the_next_repository_is_still_scanned(tmp_path: Path) -> None:
    open_prs = {
        _UPSTREAM: {"total": 250, "prs": [_open_pr(4, "pkg/moe.py")], "fail_at": 100},
        _KERNELS: {"total": 1, "prs": [_open_pr(3, "csrc/k.cu")]},
    }
    result = _overlap(tmp_path, open_prs, f"{_UPSTREAM}\tpkg/moe.py\n{_KERNELS}\tcsrc/k.cu\n")
    assert result.returncode == 0, result.stderr
    assert [row.split("\t")[:2] for row in _artifact(tmp_path, "overlap.tsv").splitlines()[1:]] == [
        [_UPSTREAM, "4"],
        [_KERNELS, "3"],
    ]
    assert _scan(tmp_path) == {_UPSTREAM: ("PARTIAL", [100, 250]), _KERNELS: ("COMPLETE", [1, 1])}


def test_a_planned_path_for_a_repository_outside_repos_txt_stops_the_scan(tmp_path: Path) -> None:
    result = _overlap(tmp_path, {_UPSTREAM: {"total": 0, "prs": []}}, f"{_KERNELS}\tcsrc/k.cu\n")
    assert result.returncode != 0
    assert f"{_KERNELS}\tcsrc/k.cu" in result.stderr


def _pr_world(diff: str | None, fail: tuple[str, ...] = ()) -> dict:
    """One open PR's detail as gh pr view, gh pr diff and the REST endpoints serve it; ``fail`` answers 404."""
    view = {"number": 42, "title": "Pad the MoE block", "state": "OPEN", "isDraft": False, "mergeCommit": None}
    revert = {
        "number": 77,
        "state": "open",
        "title": "Revert pad",
        "html_url": "https://example.test/77",
        "repository": {"full_name": _UPSTREAM},
        "pull_request": {"merged_at": None},
    }
    return {
        "prs": {
            f"{_UPSTREAM}#42": {
                "fail": list(fail),
                "view": {**view, "changedFiles": 2},
                "diff": diff,
                "pulls/files": [
                    {"status": "modified", "additions": 3, "deletions": 1, "filename": "pkg/moe.py"},
                    {"status": "added", "additions": 9, "deletions": 0, "filename": "pkg/ops/pad.py"},
                ],
                "issues/timeline": [
                    {"event": "commented"},
                    {"event": "cross-referenced", "source": {"issue": revert}},
                    {
                        "event": "cross-referenced",
                        "source": {"issue": {"number": 5, "repository": {"full_name": _UPSTREAM}}},
                    },
                ],
            }
        }
    }


def test_a_deep_read_keeps_every_path_and_the_links_to_it_when_github_refuses_the_diff(tmp_path: Path) -> None:
    result = _run(tmp_path, "read.sh", _pr_world(None), {}, _UPSTREAM, "42")
    assert result.returncode == 0, result.stderr
    directory = tmp_path / "work" / "prs" / _UPSTREAM / "42"
    assert result.stdout.splitlines()[-1] == str(directory)
    assert json.loads((directory / "pr.json").read_text(encoding="utf-8"))["mergeCommit"] is None
    diff = (directory / "diff.txt").read_text(encoding="utf-8")
    assert diff.startswith("SKIPPED: diff")
    assert "HTTP 406" in diff
    assert re.findall(r"\d+", diff)[-2:] == ["2", "2"]
    files = (directory / "files.txt").read_text(encoding="utf-8")
    assert files == "modified\t3\t1\tpkg/moe.py\nadded\t9\t0\tpkg/ops/pad.py\n"
    assert (directory / "xrefs.tsv").read_text(encoding="utf-8").splitlines() == [
        "repo\tnumber\tstate\ttitle\turl",
        f"{_UPSTREAM}\t77\topen\tRevert pad\thttps://example.test/77",
    ]


def test_a_deep_read_that_cannot_collect_a_part_exits_non_zero(tmp_path: Path) -> None:
    result = _run(tmp_path, "read.sh", _pr_world("diff\n", ("issues/timeline",)), {}, _UPSTREAM, "42")
    assert result.returncode != 0
    assert "gh: Not Found (HTTP 404)" in result.stderr
