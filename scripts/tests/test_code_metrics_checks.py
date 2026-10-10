# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of scripts/code_metrics_checks.py: which refusal each check makes, and when it stays quiet.

Every case names the check and the message that fires, so a refusal from another check
(or from a broken parser) cannot stand in for the one under test.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

import code_metrics_checks as checks
from code_metrics_collect import Units

CJK_CHAR = chr(0x4E2D)


def comment_refusals(source: str, added: set[int] | None = None) -> list[tuple[str, int, str]]:
    text = textwrap.dedent(source)
    rows = set(range(1, len(text.splitlines()) + 1)) if added is None else added
    return [(p.check, p.line, p.message) for p in checks.comment_problems("m.py", text, rows)]


def block(lines: int, start: str = "") -> str:
    return start + "".join(f"# line {i}\n" for i in range(lines))


# --- comments --------------------------------------------------------------------


def test_an_added_comment_block_over_eight_lines_fails() -> None:
    assert comment_refusals(block(9, "x = 1\n")) == [("comments", 2, "adds a 9-line comment block (limit 8)")]
    assert comment_refusals(block(8, "x = 1\n")) == []


def test_only_the_added_lines_of_a_block_count() -> None:
    text = block(12)
    # Four lines added to an existing eight-line block: the change added a 4-line block.
    assert comment_refusals(text, added={9, 10, 11, 12}) == []
    assert comment_refusals(text, added=set(range(3, 12))) == [("comments", 3, "adds a 9-line comment block (limit 8)")]


def test_a_blank_line_or_code_ends_a_block_and_trailing_comments_do_not_join_one() -> None:
    assert comment_refusals(block(5) + "\n" + block(5)) == []
    assert comment_refusals(block(5) + "x = 1  # trailing\n" + block(4)) == []


def test_a_docstring_is_not_a_comment() -> None:
    doc = '"""\n' + "".join(f"line {i} #1234\n" for i in range(12)) + '"""\n'
    assert comment_refusals(doc) == []


@pytest.mark.parametrize(
    ("comment", "shown"),
    [
        ("# fixed in #1812", "#1812"),
        ("# see PR 1811 for the history", "PR 1811"),
        ("# workaround for issue #77", "issue #77"),
        ("# https://github.com/o/r/pull/12", "github.com/o/r/pull/12"),
    ],
)
def test_an_added_comment_may_not_point_at_a_pr_or_issue(comment: str, shown: str) -> None:
    assert comment_refusals(f"x = 1  {comment}\n") == [("comments", 1, f"comment points at a PR or issue (`{shown}`)")]


@pytest.mark.parametrize(
    ("comment", "shown"),
    [
        ("# guards against the outage we had", "the outage"),
        ("# added after the 2026-09-21 incident", "the 2026-09-21 incident"),
        ("# see the postmortem", "the postmortem"),
    ],
)
def test_an_added_comment_may_not_narrate_an_incident(comment: str, shown: str) -> None:
    assert comment_refusals(f"{comment}\nx = 1\n") == [("comments", 1, f"comment narrates an incident (`{shown}`)")]


@pytest.mark.parametrize(
    "comment",
    ["# step #1 of 3", "# colour #fff", "# an outage is retried", "# not a PR: 1234 widgets", "# issues 3 calls"],
)
def test_ordinary_comments_are_not_history(comment: str) -> None:
    assert comment_refusals(f"{comment}\nx = 1\n") == []


def test_a_todo_and_its_issue_link_line_are_exempt() -> None:
    source = "# TODO(ann): drop the shim (#1234)\n# https://github.com/o/r/issues/1234\n# then #1234 again\n"
    assert comment_refusals(source) == [("comments", 3, "comment points at a PR or issue (`#1234`)")]


def test_a_reference_on_an_unchanged_line_is_not_this_changes() -> None:
    assert comment_refusals("# fixed in #1812\nx = 1\n", added={2}) == []


# --- English only ----------------------------------------------------------------


@pytest.mark.parametrize("char", [chr(c) for c in (0x4E2D, 0x3002, 0xFF01, 0x3400, 0x20001)])
def test_a_cjk_character_in_a_text_file_fails(tmp_path: Path, char: str) -> None:
    (tmp_path / "a.md").write_text(f"ok\nbad {char}\n", encoding="utf-8")
    got = [(p.check, p.where, p.line, p.message) for p in checks.cjk_problems(tmp_path, ["a.md"])]
    assert got == [("english", "a.md", 2, f"CJK character U+{ord(char):04X}")]


def test_non_cjk_non_ascii_text_and_binary_files_pass(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        f"price = '{chr(0x20AC)}5'  # caf{chr(0xE9)} {chr(0x661)}{chr(0x662)}\n", encoding="utf-8"
    )
    (tmp_path / "b.bin").write_bytes(b"\0" + CJK_CHAR.encode())
    (tmp_path / "c.txt").write_bytes(CJK_CHAR.encode("utf-16"))
    assert checks.cjk_problems(tmp_path, ["a.py", "b.bin", "c.txt", "missing.txt"]) == []


def test_the_pull_request_title_body_and_commits_are_checked() -> None:
    pr = checks.PullRequest(7, [], f"t{CJK_CHAR}", "fine", [("a" * 40, "ok"), ("b" * 40, f"s\n\n{CJK_CHAR}")])
    got = [(p.where, p.line) for p in checks.pull_request_cjk(pr)]
    assert got == [("PR title", 1), (f"commit {'b' * 12} message", 3)]


def test_the_tree_is_english_only() -> None:
    root = Path(__file__).resolve().parents[2]
    assert checks.cjk_problems(root, checks.tracked_text_files(root)) == []


# --- production code does not import test code -------------------------------------


def import_refusals(tmp_path: Path, files: dict[str, str]) -> list[tuple[str, int, str]]:
    for path, source in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(source, encoding="utf-8")
    problems = checks.test_import_problems(tmp_path, sorted(files), Units(tmp_path), ["src", "scripts"])
    return [(p.where, p.line, p.message) for p in problems]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("from pkg.tests.test_x import helper\n", "pkg.tests.test_x"),
        ("import pkg.tests\n", "pkg.tests"),
        ("from pkg import tests\n", "pkg.tests"),
        ("from .tests import helper\n", ".tests"),
        ("from pkg.sub import test_helpers\n", "pkg.sub.test_helpers"),
        ("from pkg.sub.test_helpers import x\n", "pkg.sub.test_helpers"),
    ],
)
def test_production_code_importing_test_code_fails(tmp_path: Path, source: str, target: str) -> None:
    files = {"src/pkg/mod.py": source, "src/pkg/sub/test_helpers.py": "x = 1\n"}
    got = import_refusals(tmp_path, files)
    assert got == [("src/pkg/mod.py", 1, f"production code imports `{target}`")]


@pytest.mark.parametrize(
    "source",
    [
        "from kernelforge.mcp_server.tools import test\n",
        "import kernelforge.mcp_server.tools.test\n",
        "from pkg.sub import test_value\n",
        "from pkg.contest import x\n",
        "from pkg.testing import y\n",
    ],
)
def test_a_module_merely_named_test_is_not_test_code(tmp_path: Path, source: str) -> None:
    assert import_refusals(tmp_path, {"src/pkg/mod.py": source, "src/pkg/sub/value.py": "x = 1\n"}) == []


def test_test_code_may_import_test_code(tmp_path: Path) -> None:
    source = "from pkg.tests.test_x import helper\n"
    files = {"src/pkg/tests/test_y.py": source, "scripts/test_flow.py": source, "src/pkg/conftest.py": source}
    assert import_refusals(tmp_path, files) == []


def test_scripts_are_production_code(tmp_path: Path) -> None:
    got = import_refusals(tmp_path, {"scripts/tool.py": "from hyperloom.x.tests.test_y import z\n"})
    assert got == [("scripts/tool.py", 1, "production code imports `hyperloom.x.tests.test_y`")]


# --- repeated literals -------------------------------------------------------------


def literal_refusals(before: str | None, after: str) -> list[tuple[int, str]]:
    problems = checks.repeated_literal_problems("m.py", before, textwrap.dedent(after))
    assert all(p.check == "literals" for p in problems)
    return [(p.line, p.message) for p in problems]


def test_adding_the_third_occurrence_of_a_literal_fails() -> None:
    before = 'a = "region"\nb = "region"\n'
    assert literal_refusals(before, before + 'c = "region"\n') == [
        (1, "`'region'` is written 3 times (was 2); name it as a module-level constant")
    ]
    assert literal_refusals("t = 3600\n", "t = 3600\nu = 3600\nv = 3600\n") == [
        (1, "`3600` is written 3 times (was 1); name it as a module-level constant")
    ]
    assert literal_refusals(None, "x = -5\ny = -5\nz = -5\n") == [
        (1, "`-5` is written 3 times (was 0); name it as a module-level constant")
    ]


def test_a_literal_already_repeated_at_the_base_only_fails_when_the_change_adds_one() -> None:
    before = 'a = "region"\nb = "region"\nc = "region"\n'
    assert literal_refusals(before, before) == []
    assert literal_refusals(before, before.replace('c = "region"\n', "")) == []
    assert literal_refusals(before, before + 'd = "region"\n') != []


@pytest.mark.parametrize(
    "after",
    [
        'd = {"key": 1}\ne = {"key": 2}\nf = {"key": 3}\n',
        'x["key"]\ny["key"]\nz["key"]\n',
        "f(key=7)\ng(key=8)\nh(key=9)\n",
        'def f():\n    """Same doc."""\ndef g():\n    """Same doc."""\ndef h():\n    """Same doc."""\n',
        'a = f"pre {x} post"\nb = f"pre {y} post"\nc = f"pre {z} post"\n',
        '__all__ = ["name", "name2"]\nname = 1\n__all__ += ["name"]\nv = {"name": 1}\n',
        'a: "Thing" = 1\nb: "Thing" = 2\ndef f(c: "Thing") -> "Thing": ...\n',
        "a = 0\nb = 0\nc = 0\nd = 1\ne = 1\nf = 1\ng = -1\nh = -1\ni = -1\nj = 2\nk = 2\nl = 2\n",
        'a = "ab"\nb = "ab"\nc = "ab"\nd = True\ne = True\nf = True\ng = None\nh = None\ni = None\n',
    ],
)
def test_keys_names_docstrings_fstring_text_all_and_plain_values_do_not_count(after: str) -> None:
    assert literal_refusals(None, after) == []


# --- the diff-only checks in a repository --------------------------------------------


def git(root: Path, *args: str) -> None:
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "-C", str(root), *ident, *args], check=True, capture_output=True)


def test_diff_only_checks_judge_the_change_and_skip_tests(tmp_path: Path) -> None:
    (tmp_path / "src/pkg/tests").mkdir(parents=True)
    old = 'A = "seen"\nB = "seen"\n' + block(9)
    (tmp_path / "src/pkg/old.py").write_text(old, encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "src/pkg/old.py").write_text(old + "# see #4321\nC = 'seen'\n", encoding="utf-8")
    (tmp_path / "src/pkg/tests/test_new.py").write_text("x = 'abc'\ny = 'abc'\nz = 'abc'\n" + block(9))
    problems = checks.run_all(tmp_path, ["src"], [], Units(tmp_path), "HEAD")
    got = sorted((p.check, p.where, p.line) for p in problems)
    assert got == [
        ("comments", "src/pkg/old.py", 12),
        ("comments", "src/pkg/tests/test_new.py", 4),
        ("literals", "src/pkg/old.py", 1),
    ]
    # Without a base there is no diff: only the whole-tree checks run.
    assert checks.run_all(tmp_path, ["src"], [], Units(tmp_path), None) == []


def test_parse_added_lines_reads_new_side_hunks() -> None:
    diff = "@@ -1,0 +2,3 @@\n+a\n+b\n+c\n@@ -9 +12 @@\n-x\n+y\n@@ -20,2 +24,0 @@\n-z\n-w\n"
    assert checks.parse_added_lines(diff) == {2, 3, 4, 12}


def test_one_import_statement_is_one_refusal(tmp_path: Path) -> None:
    files = {"src/pkg/mod.py": "from pkg.tests import test_x\n", "src/pkg/tests/test_x.py": "x = 1\n"}
    assert import_refusals(tmp_path, files) == [("src/pkg/mod.py", 1, "production code imports `pkg.tests`")]
