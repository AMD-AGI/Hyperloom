# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of scripts/code_metrics.py: the comparison with the base, its refusals, and the report.

The gate is driven through ``main()`` in a scratch git repository; only the tool runs
(``measure`` and the version check) are replaced, so the base tree, the base-ref
comparison and the report are the real code. The parsers are tested on captured tool
output shapes, so no tool needs to be installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

import code_metrics
import code_metrics_collect as collect
import code_metrics_report

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {"ruff": "0.16.2"}
CC = "cyclomatic-complexity"
NEW = "### New violations (not at the base)"
WORSE = "### Worse than at the base"
IMPROVED = "### Improved by this change (informational)"
INVENTORY = "### Over the limit in this tree"


class Repo:
    """A git repo holding the real pyproject.toml, with the tools stubbed for this tree and for the base."""

    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = path
        self.findings: list[collect.Finding] = []
        #: What the tools would report for the tree at the base.
        self.base: list[collect.Finding] = []
        (path / "scripts").mkdir()
        shutil.copyfile(ROOT / "pyproject.toml", path / "pyproject.toml")
        shutil.copyfile(ROOT / ".importlinter", path / ".importlinter")
        self.git("init", "-q", "-b", "main")
        monkeypatch.setattr(code_metrics, "measure", self.measure)
        monkeypatch.setattr(code_metrics, "check_tools", lambda config: dict(VERSIONS))

    def measure(self, root: Path, config, units) -> dict[tuple[str, str, str], collect.Finding]:
        return collect.worst_per_unit(self.findings if root.resolve() == self.path.resolve() else self.base)

    def git(self, *args: str) -> str:
        argv = ["git", "-C", str(self.path), "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()

    def touch(self, *paths: str) -> None:
        """Change ``paths`` in the working tree, so a ``--base-ref`` run judges them."""
        for path in paths:
            (self.path / path).parent.mkdir(parents=True, exist_ok=True)
            with (self.path / path).open("a", encoding="utf-8") as handle:
                handle.write("x = 1\n")

    def commit(self) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "state")
        return self.git("rev-parse", "HEAD")

    def change(self, *paths: str) -> str:
        """Commit the base, then touch ``paths`` in the working tree; returns the base commit."""
        base = self.commit()
        self.touch(*paths)
        return base

    def gate(self, *args: str) -> tuple[int, str]:
        report = self.path / "report.md"
        code = code_metrics.main(["--root", str(self.path), "--report", str(report), *args])
        return code, report.read_text(encoding="utf-8") if report.exists() else ""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repo:
    return Repo(tmp_path, monkeypatch)


def section(report: str, title: str) -> str:
    """The body of the ``### title...`` section, or '' when the report has none."""
    start = report.find(title)
    if start < 0:
        return ""
    end = report.find("\n### ", start + 1)
    return report[start : end if end >= 0 else len(report)]


def cc(path: str, unit: str, value: int, line: int = 3) -> collect.Finding:
    return collect.Finding(CC, path, unit, value, line)


def test_a_change_that_leaves_every_unit_as_the_base_had_it_passes(repo: Repo) -> None:
    base = repo.change("src/a.py")
    repo.base = [cc("src/a.py", "f", 25)]
    repo.findings = [cc("src/a.py", "f", 25)]
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert "## Code metrics gate: PASSED\n" in report
    assert section(report, NEW) == section(report, WORSE) == section(report, IMPROVED) == ""
    assert "| Cyclomatic complexity | fails > 20 | 0 | 0 | 0 | 1 | +0 |" in report


def test_new_violation_fails_and_report_links_the_offender(repo: Repo) -> None:
    base = repo.change("src/new.py")
    repo.findings = [cc("src/new.py", "Planner.pick", 25, line=42)]
    code, report = repo.gate("--base-ref", base, "--link-base", "https://github.com/o/r/blob/abc123")
    assert code == 1
    assert "## Code metrics gate: FAILED" in report
    rows = section(report, NEW)
    assert "[`src/new.py:42`](https://github.com/o/r/blob/abc123/src/new.py#L42) `Planner.pick`" in rows
    assert "| Cyclomatic complexity | 25 | fails > 20 |" in rows
    assert section(report, WORSE) == ""


def test_a_unit_worse_than_at_the_base_fails(repo: Repo) -> None:
    base = repo.change("src/a.py")
    repo.base = [cc("src/a.py", "f", 22)]
    repo.findings = [cc("src/a.py", "f", 23)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`f` | Cyclomatic complexity | 22 | 23 |" in section(report, WORSE)
    assert section(report, NEW) == ""


@pytest.mark.parametrize(("now", "shown"), [(22, "| 25 | 22 |"), (None, "| 25 | within the limit, or removed |")])
def test_an_improved_or_removed_unit_passes_and_is_listed(repo: Repo, now, shown) -> None:
    base = repo.change("src/a.py")
    repo.base = [cc("src/a.py", "f", 25)]
    repo.findings = [] if now is None else [cc("src/a.py", "f", now)]
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert f"`src/a.py` `f` | Cyclomatic complexity {shown}" in section(report, IMPROVED)


def test_moved_unit_is_matched_with_the_unit_it_left_at_the_base(repo: Repo) -> None:
    base = repo.change("src/old.py", "src/new.py")
    repo.base = [cc("src/old.py", "Planner.pick", 24)]
    repo.findings = [cc("src/new.py", "Planner.pick", 23)]
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert section(report, NEW) == ""
    assert "| 24 | moved to `src/new.py:3` `Planner.pick` (23) |" in section(report, IMPROVED)


@pytest.mark.parametrize(("unit", "value"), [("Planner.pick", 25), ("Planner.choose", 24), ("Scheduler.pick", 24)])
def test_move_that_got_worse_or_renamed_is_a_new_violation(repo: Repo, unit: str, value: int) -> None:
    base = repo.change("src/old.py", "src/new.py")
    repo.base = [cc("src/old.py", "Planner.pick", 24)]
    repo.findings = [cc("src/new.py", unit, value)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert f"`{unit}`" in section(report, NEW)
    assert "within the limit, or removed" in section(report, IMPROVED)


@pytest.mark.parametrize(
    "at_base",
    [
        # A.run improves under the limit while B.run worsens; both move.
        [cc("src/a.py", "A.run", 24), cc("src/b.py", "B.run", 21)],
        # Two same-named units: the worse one must not borrow the other's allowance.
        [cc("src/a.py", "main", 24), cc("src/b.py", "main", 21)],
    ],
)
def test_a_worse_unit_cannot_borrow_another_units_allowance_by_moving(repo: Repo, at_base) -> None:
    base = repo.change("src/a.py", "src/b.py", "src/new.py")
    repo.base = at_base
    unit = at_base[1].unit
    repo.findings = [cc("src/new.py", unit, 23)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert f"`{unit}` | Cyclomatic complexity | 23 |" in section(report, NEW)


def test_a_symlinked_python_file_in_scope_is_refused_not_skipped(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "payload.txt").write_text("def f():\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    assert collect.list_files(tmp_path, ["src"], []) == (["src/a.py"], [])
    (tmp_path / "src/b.py").symlink_to(tmp_path / "payload.txt")
    with pytest.raises(collect.ToolError, match=r"`src/b.py` is a symlink; the gate measures Python files only"):
        collect.list_files(tmp_path, ["src"], [])


def test_a_unit_copied_to_two_places_is_not_a_move(repo: Repo) -> None:
    base = repo.change("src/old.py", "src/x.py", "src/y.py")
    repo.base = [cc("src/old.py", "A.run", 24)]
    repo.findings = [cc("src/x.py", "A.run", 22), cc("src/y.py", "A.run", 22)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert f"{NEW}: 2" in report


def test_a_copy_of_a_unit_still_over_at_its_old_place_is_new(repo: Repo) -> None:
    base = repo.change("src/old.py", "src/new.py")
    repo.base = [cc("src/old.py", "f", 30)]
    repo.findings = [cc("src/old.py", "f", 30), cc("src/new.py", "f", 30)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert f"{NEW}: 1" in report
    assert "| `src/new.py:3` `f` | Cyclomatic complexity | 30 | fails > 20 |" in section(report, NEW)
    assert section(report, IMPROVED) == ""


def test_duplication_is_compared_with_the_base_like_every_other_dimension(repo: Repo) -> None:
    dup = "duplicated-lines"
    base = repo.change("src/a.py")
    repo.base = [collect.Finding(dup, "src/a.py", collect.MODULE, 40, 1)]
    repo.findings = [collect.Finding(dup, "src/a.py", collect.MODULE, 40, 1)]
    assert repo.gate("--base-ref", base)[0] == 0
    repo.findings = [collect.Finding(dup, "src/a.py", collect.MODULE, 45, 1)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`src/a.py:1` | Duplicated lines per file | 40 | 45 |" in section(report, WORSE)


def test_loosening_the_config_relative_to_base_fails(repo: Repo) -> None:
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    text = text.replace("cyclomatic-complexity = 20\n", "cyclomatic-complexity = 25\n")
    text = text.replace(
        'exclude = [\n    "src/hyperloom/orch', 'exclude = [\n    "src/hyperloom/agents",\n    "src/hyperloom/orch', 1
    )
    pyproject.write_text(text, encoding="utf-8")
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    loosened = section(report, "### Gate configuration loosened")
    assert "threshold `cyclomatic-complexity` loosened from 20 to 25" in loosened
    assert "`exclude` gained `src/hyperloom/agents`" in loosened


def test_raising_the_tuple_limit_or_dropping_a_vocabulary_root_is_loosening(repo: Repo) -> None:
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "tuple-return-max-elements = 2\n" in text and 'vocabulary-roots = ["src"]\n' in text
    text = text.replace("tuple-return-max-elements = 2\n", "tuple-return-max-elements = 4\n")
    pyproject.write_text(text.replace('vocabulary-roots = ["src"]\n', "vocabulary-roots = []\n"), encoding="utf-8")
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    loosened = section(report, "### Gate configuration loosened")
    assert "`tuple-return-max-elements` raised from 2 to 4" in loosened
    assert "`vocabulary-roots` lost `src`" in loosened


def test_the_design_checks_run_on_the_change_and_report_in_their_own_section(repo: Repo) -> None:
    (repo.path / "src/hyperloom/common").mkdir(parents=True)
    (repo.path / "src/hyperloom/common/state.py").write_text("X = 1\n", encoding="utf-8")
    base = repo.commit()
    (repo.path / "src/hyperloom/common/state.py").write_text(
        "X = 1\n\n\ndef bump():\n    global X\n    X += 1\n", encoding="utf-8"
    )
    code, report = repo.gate("--base-ref", base)
    # A11 is advisory in the A rules: listed with its finding, never failing.
    assert code == 0, report
    rows = section(report, "### A11 Hidden global write (advisory, never fails): 1")
    assert "`src/hyperloom/common/state.py:5`" in rows and "A11 hidden global write: `global X`" in rows
    assert "| A11 Hidden global write | 1 | advisory, never fails |" in report
    # Without a base there is no change to judge.
    assert "### A11" not in repo.gate()[1]


FLAGGED = """def pick(fast: bool) -> int:
    if fast:
        return 1
    return 2


def pair(a: int, b: int) -> tuple[int, int, int]:
    return a, b, a + b
"""


def test_failing_design_checks_fail_and_the_override_label_waives_them(repo: Repo) -> None:
    (repo.path / "src/hyperloom/common").mkdir(parents=True)
    (repo.path / "src/hyperloom/common/pick.py").write_text("X = 1\n", encoding="utf-8")
    base = repo.commit()
    (repo.path / "src/hyperloom/common/pick.py").write_text(FLAGGED, encoding="utf-8")
    code, report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, []))
    assert code == 1, report
    assert "`src/hyperloom/common/pick.py:1`" in section(report, "### A8 Flag argument: 1")
    assert "| A8 Flag argument | 1 | fails; `baseline-raise` waives |" in report
    assert "`src/hyperloom/common/pick.py:7`" in section(report, "### A9 Tuple return (advisory, never fails): 1")
    code, report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, [LABEL]))
    assert code == 0, report
    assert f"## Code metrics gate: PASSED with waived findings (`{LABEL}`)" in report
    assert "`src/hyperloom/common/pick.py:1`" in section(
        report, f"### A8 Flag argument (waived by the `{LABEL}` label)"
    )


def test_a_missing_import_contracts_file_is_could_not_run(repo: Repo) -> None:
    (repo.path / ".importlinter").unlink()
    base = repo.commit()
    code, report = repo.gate("--base-ref", base)
    assert code == 2
    assert "`.importlinter` (import-contracts) does not exist" in report


def test_the_change_cannot_redraw_the_package_units_it_is_judged_by(repo: Repo) -> None:
    base = repo.commit()
    (repo.path / ".importlinter").write_text("[importlinter]\nroot_packages =\n    hyperloom\n", encoding="utf-8")
    (repo.path / "src/hyperloom/orchestrator").mkdir(parents=True, exist_ok=True)
    (repo.path / "src/hyperloom/orchestrator/reach.py").write_text(
        "from hyperloom.inference_optimizer.cli.preflight import _HIDDEN\n", encoding="utf-8"
    )
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    rows = section(report, "### A1 Private name of another package")
    assert "`src/hyperloom/orchestrator/reach.py:1`" in rows and "private to `hyperloom.inference_optimizer`" in rows


def test_moving_the_import_contracts_file_is_loosening(repo: Repo) -> None:
    base = repo.commit()
    shutil.copyfile(repo.path / ".importlinter", repo.path / "other.ini")
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'import-contracts = ".importlinter"\n' in text
    pyproject.write_text(
        text.replace('import-contracts = ".importlinter"\n', 'import-contracts = "other.ini"\n'), encoding="utf-8"
    )
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`import-contracts` moved from `.importlinter` to `other.ini`" in section(
        report, "### Gate configuration loosened"
    )


def test_base_without_the_gate_skips_the_loosening_check_with_a_note(repo: Repo) -> None:
    (repo.path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    base = repo.change("src/a.py")
    shutil.copyfile(ROOT / "pyproject.toml", repo.path / "pyproject.toml")
    repo.base = [cc("src/a.py", "f", 25)]
    repo.findings = [cc("src/a.py", "f", 26)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "has no code-metrics config yet; the loosening check starts once it does." in report
    assert "| 25 | 26 |" in section(report, WORSE)


def test_unresolvable_base_ref_or_tool_failure_cannot_pass(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    repo.commit()
    code, report = repo.gate("--base-ref", "no-such-ref")
    assert code == 2
    assert "## Code metrics gate: could not run" in report
    assert "no-such-ref does not resolve" in report

    def broken(config):
        raise collect.ToolError("tool version mismatch: ruff 0.15.0 (pinned 0.16.2)")

    monkeypatch.setattr(code_metrics, "check_tools", broken)
    code, report = repo.gate()
    assert code == 2
    assert "ruff 0.15.0 (pinned 0.16.2)" in report


def test_report_truncates_long_sections(repo: Repo) -> None:
    paths = [f"src/m{i:02d}.py" for i in range(30)]
    base = repo.change(*paths)
    repo.findings = [cc(path, "f", 21) for path in paths]
    _code, report = repo.gate("--base-ref", base)
    assert f"{NEW}: 30" in report
    assert "...and 5 more; the job log lists every row." in report
    assert "src/m24.py" in report and "src/m25.py" not in report


def test_without_a_base_every_unit_over_a_limit_is_listed_and_none_fails(repo: Repo) -> None:
    repo.findings = [cc(f"src/m{i:02d}.py", "f", 21) for i in range(30)]
    code, report = repo.gate()
    assert code == 0, report
    assert "No base to compare with: every unit over a threshold is listed below, none fails." in report
    inventory = section(report, f"{INVENTORY}: 30")
    assert all(f"`src/m{i:02d}.py:3` `f` | Cyclomatic complexity | 21 | fails > 20 |" in inventory for i in range(30))
    assert section(report, NEW) == ""
    assert "| Cyclomatic complexity | fails > 20 | 0 | 0 | 0 | 30 | n/a |" in report


def test_a_push_run_lists_every_unit_over_a_limit_and_a_pull_request_run_does_not(repo: Repo, monkeypatch) -> None:
    api_stub(monkeypatch, [MERGED], [])
    base = repo.change("src/mine.py")
    repo.base = repo.findings = [cc(f"src/m{i:02d}.py", "f", 21) for i in range(30)]
    code, report = repo.gate("--base-ref", base, "--pushed-sha", PUSHED)
    assert code == 0, report
    assert all(f"`src/m{i:02d}.py:3` `f`" in section(report, f"{INVENTORY}: 30") for i in range(30))
    code, report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, []))
    assert code == 0, report
    assert INVENTORY not in report


class FakeUnits:
    """Units over a fixed source string (``tmp_path`` stands in for the repo root)."""

    def __init__(self, root: Path, path: str, source: str) -> None:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(source, encoding="utf-8")
        self.units = collect.Units(root)


SOURCE = """\
class Planner:
    def pick(self, a):
        for x in a:
            if x:
                while x:
                    pass

    @property
    def size(self):
        return 1


def helper():
    def inner():
        return 2
    return inner
"""


def test_ruff_rows_map_to_qualified_units(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    rows = [
        {
            "code": "PLR1702",
            "message": "Too many nested blocks (6 > 5)",
            "filename": str(tmp_path / "src/p.py"),
            "location": {"row": 4},
        },
    ]
    got = [(f.metric, f.unit, f.value, f.line) for f in collect.parse_ruff(rows, tmp_path, units)]
    assert got == [("nested-blocks", "Planner.pick", 6, 4)]
    with pytest.raises(collect.ToolError, match="unexpected ruff diagnostic"):
        collect.parse_ruff([{**rows[0], "message": "reworded"}], tmp_path, units)
    # A rule this gate no longer selects is unexpected output, not a silently dropped row.
    for code in ("PLR0911", "C901", "PLR0913"):
        with pytest.raises(collect.ToolError, match=f"unexpected ruff diagnostic: {code}"):
            collect.parse_ruff([{**rows[0], "code": code}], tmp_path, units)


def test_radon_blocks_map_functions_and_closures_to_units(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    pick = {"type": "method", "name": "pick", "classname": "Planner", "lineno": 2, "complexity": 21, "closures": []}
    size = {"type": "method", "name": "size", "classname": "Planner", "lineno": 9, "complexity": 1, "closures": []}
    inner = {"type": "function", "name": "inner", "lineno": 14, "complexity": 22, "closures": []}
    rows = {
        str(tmp_path / "src/p.py"): [
            {"type": "class", "name": "Planner", "lineno": 1, "complexity": 30, "methods": [pick, size]},
            pick,
            size,
            {"type": "function", "name": "helper", "lineno": 13, "complexity": 20, "closures": [inner]},
        ]
    }
    got = [(f.metric, f.unit, f.value, f.line) for f in collect.parse_radon(rows, tmp_path, 20, units)]
    assert got == [(CC, "Planner.pick", 21, 2), (CC, "helper.inner", 22, 14)]
    with pytest.raises(collect.ToolError, match="radon could not measure"):
        collect.parse_radon({str(tmp_path / "src/p.py"): {"error": "invalid syntax"}}, tmp_path, 20, units)


@pytest.mark.skipif(shutil.which("radon") is None, reason="radon")
def test_cyclomatic_complexity_counts_boolean_operators(tmp_path: Path) -> None:
    # McCabe counts this function as 1; every `and` is a branch for the gate.
    operands = " and ".join(f"x{i}" for i in range(22))
    path = "src/a.py"
    FakeUnits(tmp_path, path, f"def f({', '.join(f'x{i}' for i in range(22))}):\n    return {operands}\n")
    got = collect.radon_findings(tmp_path, [path], 20, collect.Units(tmp_path))
    assert got == [collect.Finding(CC, path, "f", 22, 1)]


def test_complexipy_rows_keep_only_violations_with_their_line(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    rows = [
        {"complexity": 16, "function_name": "Planner::pick", "path": str(tmp_path / "src/p.py")},
        {"complexity": 15, "function_name": "helper", "path": str(tmp_path / "src/p.py")},
    ]
    got = collect.parse_complexipy(rows, tmp_path, 15, units)
    assert got == [collect.Finding("cognitive-complexity", "src/p.py", "Planner.pick", 16, 2)]


def test_vulture_lines_become_scoped_units(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    text = f"{tmp_path}/src/p.py:2: unused variable 'a' (100% confidence)\n"
    assert collect.parse_vulture(text, tmp_path, units) == [
        collect.Finding("dead-code", "src/p.py", "Planner.pick:a", 1, 2)
    ]
    with pytest.raises(collect.ToolError, match="unexpected vulture output"):
        collect.parse_vulture("garbage\n", tmp_path, units)


def test_jscpd_counts_each_files_duplicated_lines_once(tmp_path: Path) -> None:
    def side(name: str, start: int, end: int) -> dict:
        return {"name": str(tmp_path / name), "start": start, "end": end}

    data = {
        "duplicates": [
            {"firstFile": side("a.py", 10, 19), "secondFile": side("b.py", 1, 10)},
            {"firstFile": side("a.py", 15, 24), "secondFile": side("tests/c.py", 5, 14)},
        ]
    }
    got = {f.path: (f.value, f.line) for f in collect.parse_jscpd(data, tmp_path)}
    assert got == {"a.py": (15, 10), "b.py": (10, 1), "tests/c.py": (10, 5)}


def test_list_files_splits_tests_and_honours_exclude(tmp_path: Path) -> None:
    for path in ["src/a.py", "src/tests/test_a.py", "src/vendor/v.py", "src/notes.md"]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    production, tests = collect.list_files(tmp_path, ["src"], ["src/vendor"])
    assert (production, tests) == (["src/a.py"], ["src/tests/test_a.py"])


def test_repo_config_names_every_metric_with_a_threshold() -> None:
    config = code_metrics.parse_config((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config is not None
    # A pending metric (PENDING) is the one name the table may still be missing.
    assert set(collect.METRICS) - set(collect.PENDING) <= set(config.thresholds) <= set(collect.METRICS)
    assert config.tools["ruff"] in (ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8")
    assert f'"radon=={config.tools["radon"]}"' in (ROOT / ".github/workflows/code-metrics.yml").read_text(
        encoding="utf-8"
    )
    assert "maintainability-index" not in config.thresholds


def test_comment_poster_never_runs_pull_request_code() -> None:
    poster = yaml.safe_load((ROOT / ".github/workflows/code-metrics-comment.yml").read_text(encoding="utf-8"))
    assert poster["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "write"}
    steps = [step for job in poster["jobs"].values() for step in job["steps"]]
    checkouts = [step for step in steps if "checkout" in str(step.get("uses", ""))]
    assert [step["with"]["ref"] for step in checkouts] == ["${{ github.event.repository.default_branch }}"]
    gate = yaml.safe_load((ROOT / ".github/workflows/code-metrics.yml").read_text(encoding="utf-8"))
    assert gate["jobs"]["code-metrics"]["name"] == "code-metrics"
    assert gate["name"] in poster[True]["workflow_run"]["workflows"]


def test_a_pending_metric_may_lack_a_threshold_and_is_then_not_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    without_depth = text.replace("nested-blocks = 4\n", "")
    with pytest.raises(collect.ToolError, match="thresholds must name"):
        code_metrics.parse_config(without_depth)
    monkeypatch.setattr(code_metrics, "PENDING", ("nested-blocks",))
    config = code_metrics.parse_config(without_depth)
    assert config is not None and "nested-blocks" not in config.thresholds
    assert [code for code, (metric, _) in collect.RUFF_RULES.items() if metric in config.thresholds] == []
    for broken in (text.replace("function-args = 5\n", ""), text.replace("function-args =", "max-args =")):
        with pytest.raises(collect.ToolError, match="thresholds must name"):
            code_metrics.parse_config(broken)


@pytest.mark.parametrize(
    "entry",
    [":(exclude)scripts/x.py", ":!scripts/x.py", ":/scripts", "scripts/*.py", "../outside", "/abs", "a/./b", ""],
)
def test_scope_entries_must_be_plain_paths(entry: str) -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    text = text.replace('roots = ["src", "scripts"]', f'roots = ["src", "scripts", {json.dumps(entry)}]')
    assert entry in text
    with pytest.raises(collect.ToolError, match="plain repository paths"):
        code_metrics.parse_config(text)


def test_list_files_reads_a_pathspec_as_a_literal_path(tmp_path: Path) -> None:
    for path in ["scripts/a.py", "scripts/b.py"]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    assert collect.list_files(tmp_path, ["scripts", ":(exclude)scripts/a.py"], []) == (
        ["scripts/a.py", "scripts/b.py"],
        [],
    )
    assert collect.list_files(tmp_path, [":!scripts/a.py"], []) == ([], [])


def test_scope_that_measures_fewer_files_than_the_base_is_loosened(repo: Repo) -> None:
    for path in ["src/a/x.py", "src/b/y.py", "scripts/z.py"]:
        (repo.path / path).parent.mkdir(parents=True, exist_ok=True)
        (repo.path / path).write_text("x = 1\n", encoding="utf-8")
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # Replacing a root by narrower ones loses no root by name that the base listed twice,
    # yet stops measuring src/b; only the file sets show it.
    pyproject.write_text(text.replace('roots = ["src", "scripts"]', 'roots = ["src/a", "scripts", "src/c"]'))
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "scope no longer measures `src/b/y.py`" in section(report, "### Gate configuration loosened")
    assert "src/a/x.py" not in report


def test_widening_the_scope_is_not_loosening(repo: Repo) -> None:
    (repo.path / "src/a").mkdir(parents=True)
    (repo.path / "src/a/x.py").write_text("x = 1\n", encoding="utf-8")
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(text.replace('roots = ["src", "scripts"]', 'roots = ["src/a"]'))
    base = repo.commit()
    pyproject.write_text(text)
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report


def test_an_edit_to_the_gate_is_reported_for_review_without_failing(repo: Repo) -> None:
    (repo.path / "scripts/code_metrics.py").write_text("x = 1\n", encoding="utf-8")
    base = repo.commit()
    (repo.path / "scripts/code_metrics.py").write_text("x = 2\n", encoding="utf-8")
    pyproject = repo.path / "pyproject.toml"
    pyproject.write_text(pyproject.read_text(encoding="utf-8").replace('ruff = "0.16.2"', 'ruff = "0.16.1"'))
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    changed = section(report, "### Gate implementation changed (needs review)")
    assert "`scripts/code_metrics.py` changed" in changed
    assert "tool pin `ruff` changed from 0.16.2 to 0.16.1" in changed


def test_report_says_when_another_copy_of_the_gate_judged_the_tree(repo: Repo, monkeypatch) -> None:
    assert "not by this tree's copy" in repo.gate()[1]
    monkeypatch.setattr(code_metrics, "__file__", str(repo.path / "scripts" / "code_metrics.py"))
    assert "not by this tree's copy" not in repo.gate()[1]


def test_ci_reruns_when_the_label_or_the_pr_text_changes_and_keeps_every_main_run() -> None:
    gate = yaml.safe_load((ROOT / ".github/workflows/code-metrics.yml").read_text(encoding="utf-8"))
    # The label and the CJK check of the title and body are read at run time: each edit must re-run it.
    types = set(gate[True]["pull_request"]["types"])
    assert {"opened", "synchronize", "reopened", "edited", "labeled", "unlabeled"} <= types
    assert "paths" not in gate[True]["pull_request"] and "paths-ignore" not in gate[True]["pull_request"]
    assert gate["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_ci_runs_the_base_branchs_copy_of_the_gate() -> None:
    gate = yaml.safe_load((ROOT / ".github/workflows/code-metrics.yml").read_text(encoding="utf-8"))
    steps = {step.get("name"): step for step in gate["jobs"]["code-metrics"]["steps"]}
    run = steps["Code metrics gate"]["run"]
    names = ("code_metrics.py", "code_metrics_checks.py", "code_metrics_collect.py", "code_metrics_design.py")
    for name in (*names, "code_metrics_report.py"):
        assert f'git show "$base:scripts/{name}" > "$gate/{name}"' in run
    assert 'python "$impl/code_metrics.py"' in run and "python scripts/" not in run
    assert '--root "$GITHUB_WORKSPACE"' in run
    assert "RUNNER_TEMP}/code-metrics-gate/comment.js" in steps["Post the report on the PR"]["with"]["script"]


_HARNESS = r"""
const fs = require('fs');
const scenario = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const calls = [];
const record = (kind) => (message) => calls.push([kind, String(message)]);
const core = { info: record('info'), notice: record('notice'), warning: record('warning'), setFailed: record('setFailed') };
const refuse = async () => {
  const err = new Error(scenario.refuse);
  err.status = 403;
  throw err;
};
const github = {
  paginate: async () => scenario.comments,
  rest: {
    issues: {
      listComments: {},
      createComment: scenario.refuse ? refuse : async ({ body }) => (calls.push(['create', body]), { data: { html_url: 'c' } }),
      updateComment: async ({ comment_id, body }) => calls.push(['update', `${comment_id}\n${body}`]),
    },
    pulls: { get: async () => ({ data: scenario.pr }) },
  },
};
const context = { repo: { owner: 'o', repo: 'r' }, payload: scenario.payload };
const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
new AsyncFunction('github', 'context', 'core', 'require', scenario.script)(github, context, core, require)
  .then(() => console.log(JSON.stringify(calls)))
  .catch((err) => { console.log(JSON.stringify([...calls, ['threw', err.message]])); process.exit(1); });
"""
SHA = "a" * 40
MARKER = "<!-- code-metrics-report -->"


def workflow_steps(name: str) -> dict:
    workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))
    job = next(iter(workflow["jobs"].values()))
    return {step.get("name") or step.get("uses"): step for step in job["steps"]}


def run_github_script(tmp_path: Path, script: str, env: dict[str, str], **scenario) -> tuple[int, list[list[str]]]:
    """Run a workflow's actions/github-script body under node with a recorded GitHub client."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    (tmp_path / "harness.js").write_text(_HARNESS, encoding="utf-8")
    (tmp_path / "scenario.json").write_text(
        json.dumps({"script": script, "comments": [], **scenario}), encoding="utf-8"
    )
    proc = subprocess.run(
        [node, str(tmp_path / "harness.js"), str(tmp_path / "scenario.json")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", **env},
    )
    assert proc.stdout.strip(), proc.stderr
    return proc.returncode, json.loads(proc.stdout.strip().splitlines()[-1])


def fork_scenario(tmp_path: Path, report: str, conclusion: str) -> tuple[str, dict[str, str], dict]:
    report_dir = tmp_path / "artifact"
    report_dir.mkdir()
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    (report_dir / "pr-number.txt").write_text("7\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    (workspace / ".github/scripts").mkdir(parents=True)
    shutil.copy(ROOT / ".github/scripts/code_metrics_comment.js", workspace / ".github/scripts")
    head = {"full_name": "fork/r"}
    run = {
        "conclusion": conclusion,
        "head_sha": SHA,
        "html_url": "https://github.com/o/r/actions/runs/9",
        "head_repository": head,
    }
    env = {"GITHUB_WORKSPACE": str(workspace), "REPORT_DIR": str(report_dir), "REPORT_OUTCOME": "success"}
    script = workflow_steps("code-metrics-comment.yml")["actions/github-script@v9"]["with"]["script"]
    return script, env, {"payload": {"workflow_run": run}, "pr": {"number": 7, "head": {"sha": SHA, "repo": head}}}


def test_fork_comment_opens_with_the_conclusion_github_reported(tmp_path: Path) -> None:
    forged = f"{MARKER}\n## Code metrics gate: PASSED\n\n**Gate job conclusion: success**\n"
    script, env, scenario = fork_scenario(tmp_path, forged, "failure")
    code, calls = run_github_script(tmp_path, script, env, **scenario)
    assert code == 0, calls
    [(kind, body)] = [call for call in calls if call[0] in ("create", "update")]
    assert kind == "create"
    assert body.startswith(f"{MARKER}\n**Gate job conclusion: failure** ([run](https://github.com/o/r/actions/runs/9))")
    assert f"`{SHA}`" in body.split("---", 1)[0]
    assert body.index("conclusion: failure") < body.index("PASSED")


def test_fork_comment_without_an_artifact_is_a_notice_not_a_failure(tmp_path: Path) -> None:
    script, env, scenario = fork_scenario(tmp_path, f"{MARKER}\n", "failure")
    code, calls = run_github_script(tmp_path, script, {**env, "REPORT_OUTCOME": "failure"}, **scenario)
    assert code == 0
    assert [kind for kind, _ in calls] == ["notice"]
    assert "uploaded no code-metrics report" in calls[0][1]


def test_fork_comment_skips_cancelled_runs_one_poster_per_branch() -> None:
    poster = yaml.safe_load((ROOT / ".github/workflows/code-metrics-comment.yml").read_text(encoding="utf-8"))
    condition = poster["jobs"]["comment"]["if"]
    assert "github.event.workflow_run.conclusion != 'cancelled'" in condition
    group = poster["concurrency"]["group"]
    assert "workflow_run.head_repository.full_name" in group and "workflow_run.head_branch" in group
    download = workflow_steps("code-metrics-comment.yml")["actions/download-artifact@v8"]
    assert download["continue-on-error"] is True and download["id"] == "report"


def test_the_report_is_written_and_uploaded_outside_the_checkout() -> None:
    # A PR controls every path in the checkout; one made a symlink into the runner temp
    # directory would upload the persisted git credential with the artifact.
    text = (ROOT / ".github/workflows/code-metrics.yml").read_text(encoding="utf-8")
    steps = workflow_steps("code-metrics.yml")
    assert 'out="$RUNNER_TEMP/code-metrics-report"' in steps["Code metrics gate"]["run"]
    assert '--report "$out/report.md"' in steps["Code metrics gate"]["run"]
    assert steps["actions/upload-artifact@v7"]["with"]["path"] == "${{ runner.temp }}/code-metrics-report/"
    assert "code-metrics/" not in text.replace("code-metrics-report/", "").replace("code-metrics-gate/", "")


@pytest.mark.parametrize("refusal", ["Resource not accessible by integration", "API rate limit exceeded"])
def test_a_refused_comment_is_a_warning_not_a_red_job(tmp_path: Path, refusal: str) -> None:
    gate_dir = tmp_path / "temp/code-metrics-gate"
    gate_dir.mkdir(parents=True)
    shutil.copy(ROOT / ".github/scripts/code_metrics_comment.js", gate_dir / "comment.js")
    (tmp_path / "temp/code-metrics-report").mkdir()
    (tmp_path / "temp/code-metrics-report/report.md").write_text(
        f"{MARKER}\n## Code metrics gate: PASSED\n", encoding="utf-8"
    )
    script = workflow_steps("code-metrics.yml")["Post the report on the PR"]["with"]["script"]
    payload = {"pull_request": {"number": 7}}
    code, calls = run_github_script(
        tmp_path, script, {"RUNNER_TEMP": str(tmp_path / "temp")}, payload=payload, refuse=refusal
    )
    assert code == 0, calls
    assert [kind for kind, _ in calls] == ["warning"]
    assert f"403: {refusal}" in calls[0][1]


@pytest.mark.parametrize(("gate_code", "job_code"), [("0", 0), ("1", 1), ("2", 1), ("", 1)])
def test_only_the_gate_step_decides_the_job(gate_code: str, job_code: int) -> None:
    steps = workflow_steps("code-metrics.yml")
    final = steps["Fail when the gate failed"]
    assert "if" not in final and final["env"] == {"CODE": "${{ steps.gate.outputs.code }}"}
    proc = subprocess.run(["bash", "-c", final["run"]], env={"CODE": gate_code}, capture_output=True, text=True)
    assert proc.returncode == job_code
    # Reporting steps cannot redden a passing verdict; the gate step itself carries no escape hatch.
    assert steps["actions/upload-artifact@v7"]["continue-on-error"] is True
    assert "continue-on-error" not in steps["Code metrics gate"]


def test_jscpd_reads_a_defused_copy_and_vulture_the_real_file(tmp_path, monkeypatch) -> None:
    source = "import os  # noqa: F401\n# jscpd:ignore-start\nx = 1\n# jscpd:ignore-end\n"
    (tmp_path / "src").mkdir()
    (tmp_path / "src/p.py").write_text(source, encoding="utf-8")
    seen: dict[str, tuple[Path, str]] = {}

    def fake_run(argv, cwd, ok=(0,)):
        name = Path(argv[0]).name
        path = Path(next(a for a in argv if a.endswith("p.py")))
        seen[name] = (path, path.read_text(encoding="utf-8"))
        if name == "vulture":
            # vulture prints a path under its cwd relative to it, any other path as given.
            shown = path.relative_to(cwd) if path.is_relative_to(cwd) else path
            return f"{shown}:3: unused variable 'x' (100% confidence)\n"
        Path(cwd, "jscpd-report.json").write_text('{"duplicates": []}', encoding="utf-8")
        return ""

    monkeypatch.setattr(collect, "_run", fake_run)
    monkeypatch.setattr(collect, "jscpd_command", lambda: ["jscpd"])
    units = collect.Units(tmp_path)
    found = collect.vulture_findings(tmp_path, ["src/p.py"], 80, units)
    assert found == [collect.Finding("dead-code", "src/p.py", "<module>:x", 1, 3)]
    # Ruff owns the noqa marker (F401 for a side-effect import, RUF100 for an unused one), so
    # vulture reads the real file and honours it; jscpd's own marker is defused.
    assert seen["vulture"] == (tmp_path / "src/p.py", source)
    collect.jscpd_findings(tmp_path, ["src/p.py"], 100, 10)
    jscpd_path, jscpd_text = seen["jscpd"]
    assert jscpd_path != tmp_path / "src/p.py"
    assert "jscpd:ignore" not in jscpd_text and "# noqa: F401" in jscpd_text
    assert len(jscpd_text.splitlines()) == 4


def test_vulture_honours_noqa_on_a_real_run(tmp_path: Path) -> None:
    if shutil.which("vulture") is None:
        pytest.skip("vulture is not installed")
    (tmp_path / "src").mkdir()
    source = "import os  # noqa: F401\nimport re\n"
    (tmp_path / "src/p.py").write_text(source, encoding="utf-8")
    found = collect.vulture_findings(tmp_path, ["src/p.py"], 80, collect.Units(tmp_path))
    assert [f.unit for f in found] == ["<module>:re"]


def test_every_package_under_src_is_measured() -> None:
    config = code_metrics.parse_config((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config is not None
    production, _tests = collect.list_files(ROOT, config.roots, config.exclude)
    packages = {p.parent.name for p in (ROOT / "src").glob("*/__init__.py")}
    assert {"hyperloom", "kernelforge", "hyperloom_kb"} <= packages
    assert packages <= {Path(path).parts[1] for path in production if path.startswith("src/")}


# --- touched-file scope, the comparison with the base, the override label ----

OUTSIDE = "### Outside this change (informational)"
LABEL = "baseline-raise"


def test_a_violation_in_an_untouched_file_never_fails_the_change(repo: Repo) -> None:
    base = repo.change("src/mine.py")
    repo.base = [cc("src/old.py", "f", 21)]
    # A new and a worse unit, both in files this change leaves alone.
    repo.findings = [cc("src/old.py", "f", 25), cc("src/other.py", "h", 40)]
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert section(report, NEW) == section(report, WORSE) == ""
    outside = section(report, OUTSIDE)
    assert "`src/other.py` `h` | Cyclomatic complexity | absent | 40 |" in outside
    assert "`src/old.py` `f` | Cyclomatic complexity | 21 | 25 |" in outside
    assert "Judged: the 1 files this change touches." in report


@pytest.mark.parametrize(
    ("finding", "title"),
    [
        (cc("src/mine.py", "f", 25), NEW),
        (cc("src/old.py", "f", 22), WORSE),
    ],
)
def test_a_new_or_worse_unit_in_a_touched_file_fails(repo: Repo, finding: collect.Finding, title: str) -> None:
    base = repo.change(finding.path)
    repo.base = [cc("src/old.py", "f", 21)]
    repo.findings = [finding]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert f"`{finding.path}:3` `{finding.unit}`" in section(report, title)
    assert finding.path not in section(report, OUTSIDE)


@pytest.mark.parametrize(("at_base", "now", "code"), [(None, 21, 1), (24, 25, 1), (25, 25, 0), (30, 25, 0)])
def test_the_complexity_ceiling_compares_head_with_the_base(repo: Repo, at_base, now, code) -> None:
    """A unit the change adds or raises fails; one the base already had at that value or worse passes."""
    base = repo.change("src/a.py")
    repo.findings = [cc("src/a.py", "f", now)]
    repo.base = [] if at_base is None else [cc("src/a.py", "f", at_base)]
    got, report = repo.gate("--base-ref", base)
    assert got == code, report
    if at_base is None:
        assert f"| Cyclomatic complexity | {now} | fails > 20 |" in section(report, NEW)
    elif code:
        assert f"`f` | Cyclomatic complexity | {at_base} | {now} |" in section(report, WORSE)
    elif at_base > now:
        assert f"`f` | Cyclomatic complexity | {at_base} | {now} |" in section(report, IMPROVED)
    else:
        assert section(report, NEW) == section(report, WORSE) == section(report, IMPROVED) == ""


def write_pr(repo: Repo, labels: list[str], **fields) -> str:
    path = repo.path.parent / "pr.json"
    path.write_text(json.dumps({"number": 7, "labels": labels, "title": "t", "body": "b", **fields}), encoding="utf-8")
    return str(path)


def test_the_override_label_waives_new_and_worse_but_lists_them_all(repo: Repo, capsys) -> None:
    base = repo.change("src/a.py", *(f"src/m{i:02d}.py" for i in range(30)))
    repo.base = [cc("src/a.py", "f", 21)]
    repo.findings = [cc("src/a.py", "f", 23)] + [cc(f"src/m{i:02d}.py", "f", 22) for i in range(30)]
    code, _report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, []))
    assert code == 1
    capsys.readouterr()
    code, report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, ["size-exception", LABEL]))
    assert code == 0, report
    assert f"## Code metrics gate: PASSED with waived findings (`{LABEL}`)" in report
    waived = f" (waived by the `{LABEL}` label)"
    assert "`src/a.py:3` `f` | Cyclomatic complexity | 21 | 23 |" in section(report, WORSE + waived)
    assert f"{NEW}{waived}: 30" in report and "...and 5 more" in report
    # The job log is the full list: every waived row, untruncated.
    log = capsys.readouterr().out
    assert all(f"src/m{i:02d}.py" in log for i in range(30))


def test_the_override_label_waives_nothing_else(repo: Repo) -> None:
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("cyclomatic-complexity = 20\n", "cyclomatic-complexity = 25\n")
    )
    pr = write_pr(repo, [LABEL], title=f"Fix {chr(0x4E2D)}")
    code, report = repo.gate("--base-ref", base, "--pull-request-json", pr)
    assert code == 1
    assert "threshold `cyclomatic-complexity` loosened from 20 to 25" in section(
        report, "### Gate configuration loosened"
    )
    assert "PR title, line 1 | CJK character U+4E2D" in section(report, "### English only")


def test_pr_title_body_and_commit_messages_must_be_english(repo: Repo) -> None:
    commits = [["a" * 40, "fine"], ["b" * 40, f"subject\n\nbody {chr(0x3002)}"]]
    pr = write_pr(repo, [], body=f"ok\n{chr(0xFF01)}", commits=commits)
    code, report = repo.gate("--pull-request-json", pr)
    assert code == 1
    english = section(report, "### English only")
    assert "PR body, line 2 | CJK character U+FF01" in english
    assert f"commit {'b' * 12} message, line 3 | CJK character U+3002" in english
    assert "PR title" not in english and "a" * 12 not in english


def test_the_label_comes_from_the_api_at_run_time(repo: Repo, monkeypatch) -> None:
    calls = []

    def fake_get(url: str, token: str):
        calls.append((url, token))
        if url.endswith("/pulls/7"):
            return {"labels": [{"name": LABEL}], "title": "t", "body": None}
        return [{"sha": "c" * 40, "commit": {"message": "m"}}]

    monkeypatch.setattr(code_metrics.checks, "_get_json", fake_get)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    # An event payload is never read: pointing GITHUB_EVENT_PATH at a forged one changes nothing.
    forged = repo.path.parent / "event.json"
    forged.write_text(json.dumps({"pull_request": {"labels": []}}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(forged))
    base = repo.change("src/a.py")
    repo.findings = [cc("src/a.py", "f", 30)]
    code, report = repo.gate("--base-ref", base, "--pr-number", "7")
    assert code == 0, report
    assert f"{NEW} (waived by the `{LABEL}` label): 1" in report
    assert calls == [
        ("https://api.github.com/repos/o/r/pulls/7", "tok"),
        ("https://api.github.com/repos/o/r/pulls/7/commits?per_page=100&page=1", "tok"),
    ]


PUSHED = "d" * 40


def api_stub(monkeypatch: pytest.MonkeyPatch, pulls_of_commit, labels: list[str]) -> list[str]:
    """Stand the GitHub API in with ``pulls_of_commit`` for ``PUSHED``; returns the URLs read."""
    calls: list[str] = []

    def fake_get(url: str, token: str):
        calls.append(url)
        if f"/commits/{PUSHED}/pulls" in url:
            if isinstance(pulls_of_commit, Exception):
                raise pulls_of_commit
            return pulls_of_commit
        if url.endswith("/pulls/7"):
            return {"labels": [{"name": name} for name in labels], "title": "t", "body": "b"}
        return []

    monkeypatch.setattr(code_metrics.checks, "_get_json", fake_get)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    return calls


MERGED = {"number": 7, "merged_at": "2026-10-12T00:00:00Z", "merge_commit_sha": PUSHED}


def test_a_push_applies_the_label_of_the_pr_whose_merge_it_is(repo: Repo, monkeypatch) -> None:
    other = {"number": 9, "merged_at": "2026-10-11T00:00:00Z", "merge_commit_sha": "e" * 40}
    calls = api_stub(monkeypatch, [other, MERGED], [LABEL])
    base = repo.change("src/a.py")
    repo.findings = [cc("src/a.py", "f", 30)]
    code, report = repo.gate("--base-ref", base, "--pushed-sha", PUSHED)
    assert code == 0, report
    assert "is the merge of #7; its labels apply" in report
    assert calls[:2] == [
        f"https://api.github.com/repos/o/r/commits/{PUSHED}/pulls?per_page=100",
        "https://api.github.com/repos/o/r/pulls/7",
    ]


@pytest.mark.parametrize(
    "pulls",
    [
        [],
        # A PR that only contains the commit, or was closed unmerged, did not produce this push.
        [{"number": 7, "merged_at": "2026-10-12T00:00:00Z", "merge_commit_sha": "e" * 40}],
        [{"number": 7, "merged_at": None, "merge_commit_sha": PUSHED}],
    ],
)
def test_a_push_no_pr_made_waives_nothing(repo: Repo, monkeypatch, pulls) -> None:
    calls = api_stub(monkeypatch, pulls, [LABEL])
    base = repo.change("src/a.py")
    repo.findings = [cc("src/a.py", "f", 30)]
    code, report = repo.gate("--base-ref", base, "--pushed-sha", PUSHED)
    assert code == 1, report
    assert "No merged pull request produced" in report
    assert all("/pulls/7" not in url for url in calls)


def test_a_push_whose_pr_cannot_be_read_is_could_not_run(repo: Repo, monkeypatch) -> None:
    api_stub(monkeypatch, collect.ToolError("could not read the API: HTTP Error 502"), [LABEL])
    repo.findings = [cc("src/a.py", "f", 30)]
    code, report = repo.gate("--pushed-sha", PUSHED)
    assert code == 2 and "HTTP Error 502" in report
    api_stub(monkeypatch, {"message": "Not Found"}, [LABEL])
    code, report = repo.gate("--pushed-sha", PUSHED)
    assert code == 2 and "unexpected reply" in report


def test_an_unreadable_api_is_could_not_run_not_a_pass(repo: Repo, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    code, report = repo.gate("--pr-number", "7")
    assert code == 2 and "needs GITHUB_REPOSITORY" in report
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_API_URL", "http://api.example.com")
    code, report = repo.gate("--pr-number", "7")
    assert code == 2 and "the API URL must be https" in report


@pytest.mark.parametrize(("warn", "fail"), [(800, 1200), (700, 1000)])
def test_module_length_between_warning_and_limit_is_reported_not_failed(repo: Repo, warn: int, fail: int) -> None:
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "module-lines-warning = 800\n" in text and "module-lines = 1200\n" in text
    text = text.replace("module-lines-warning = 800\n", f"module-lines-warning = {warn}\n")
    pyproject.write_text(text.replace("module-lines = 1200\n", f"module-lines = {fail}\n"), encoding="utf-8")
    base = repo.commit()
    for path, lines in [("src/warn.py", warn + 1), ("src/ok.py", warn), ("src/far.py", warn + 100)]:
        (repo.path / path).parent.mkdir(parents=True, exist_ok=True)
        (repo.path / path).write_text("x = 1\n" * lines, encoding="utf-8")
    repo.git("add", "src/far.py")
    repo.commit()
    (repo.path / "src/warn.py").write_text("x = 1\n" * (warn + 1), encoding="utf-8")
    code, report = repo.gate("--base-ref", base)
    assert f"\n| Module length (lines) | warns > {warn}, fails > {fail} | 0 | 0 | 0 | 0 | +0 |\n" in report
    title = f"### Module length warning: over {warn} lines, within the {fail}-line failure limit (not failing): 2\n"
    assert title in report
    warned = section(report, title)
    assert f"\n| `src/warn.py` | {warn + 1} | warns > {warn} |\n" in warned
    assert "fails >" not in warned
    assert "src/ok.py" not in warned
    assert "src/far.py" in warned  # committed after the base: touched too
    assert code == 0, report


@pytest.mark.parametrize("warn", [700, 699])
def test_a_warning_level_at_or_above_the_failing_length_is_not_reported(warn: int) -> None:
    # module_warnings lists lengths in (warn, fail]: empty once warn >= fail, so no warning level exists.
    thresholds = dict.fromkeys(collect.METRICS, 0) | {collect.MODULE_LINES: 700}
    rendered = code_metrics_report.render(code_metrics_report.Outcome(thresholds=thresholds, module_warning=warn))
    shown = "fails > 700" if warn == 700 else "warns > 699, fails > 700"
    assert f"\n| Module length (lines) | {shown} | 0 | 0 | 0 | 0 | n/a |\n" in rendered


def test_only_module_length_has_a_warning_level_in_the_dimension_table(repo: Repo) -> None:
    _code, report = repo.gate()
    config = code_metrics.parse_config((repo.path / "pyproject.toml").read_text(encoding="utf-8"))
    rows = [line for line in report.splitlines() if line.startswith("| ") and "fails > " in line]
    # A pending metric has no threshold yet, so the table has no limit to state for it.
    measured = [metric for metric in collect.METRICS.values() if metric.name in config.thresholds]
    assert len(rows) == len(measured)
    for metric in measured:
        limit = f"fails > {config.thresholds[metric.name]}"
        if metric.name == collect.MODULE_LINES:
            limit = f"warns > {config.params['module-lines-warning']}, {limit}"
        assert any(row.startswith(f"| {metric.label} | {limit} | ") for row in rows), (metric.name, rows)


def test_function_length_counts_body_code_lines_and_nested_functions_alone(tmp_path: Path) -> None:
    source = (
        "@decorator\n@another\n"
        'def outer(\n    a,\n):\n    """Doc.\n\n    More doc.\n    """\n    # comment\n\n'
        + "    x = 1\n" * 50
        + '    text = """\n\n    keep\n    """\n'
        + '    def inner():\n        """Inner doc."""\n'
        + "        y = 2\n" * 51
        + "    return inner\n"
    )
    units = FakeUnits(tmp_path, "src/f.py", source).units
    got = {f.unit: (f.value, f.line) for f in collect.function_line_findings(["src/f.py"], 50, units)}
    # outer: 50 assignments, the string's 3 non-blank lines and the return; inner is its own unit.
    assert got == {"outer": (54, 3), "outer.inner": (51, 66)}
    assert collect.function_line_findings(["src/f.py"], 54, units) == []


def test_parameter_count_adds_varargs_and_keyword_only_and_skips_self_and_cls(tmp_path: Path) -> None:
    source = (
        "def named(a, b, c, d, e):\n    pass\n"
        "def star(a, b, c, d, e, *args):\n    pass\n"
        "def keywords(a, b, c, d, e, **kwargs):\n    pass\n"
        "async def both(a, /, b, *args, c, d, e, **kwargs):\n    pass\n"
        "class Planner:\n"
        "    def method(self, a, b, c, d, e):\n        pass\n"
        "    @classmethod\n    def make(cls, a, b, c, d, e):\n        pass\n"
        "    def wide(self, a, b, c, d, *, e, f):\n        pass\n"
        "def outer():\n    def inner(a, b, c, d, e, *rest):\n        pass\n    return inner\n"
        "def hook(a, b, c, d, e, _unused):\n    pass\n"
        "@overload\ndef over(a, b, c, d, e, f): ...\n"
    )
    units = FakeUnits(tmp_path, "src/a.py", source).units
    got = {f.unit: (f.value, f.line) for f in collect.function_arg_findings(["src/a.py"], 5, units)}
    assert got == {
        "star": (6, 3),
        "keywords": (6, 5),
        "both": (7, 7),
        "Planner.wide": (6, 15),
        "outer.inner": (6, 18),
        "hook": (6, 21),
        "over": (6, 24),
    }


@pytest.mark.skipif(any(shutil.which(tool) is None for tool in ("ruff", "radon", "complexipy")), reason="tools")
def test_the_gate_counts_varargs_in_the_parameter_count(tmp_path: Path) -> None:
    FakeUnits(tmp_path, "src/a.py", "def f(a, b, c, d, e, *args, **kwargs):\n    pass\n")
    config = code_metrics.parse_config((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config is not None
    got = code_metrics.per_file_findings(tmp_path, ["src/a.py"], config.thresholds, collect.Units(tmp_path))
    assert got == [collect.Finding(collect.FUNCTION_ARGS, "src/a.py", "f", 7, 1)]


def test_directory_entries_count_direct_modules_and_subdirectories() -> None:
    files = ["src/a/x.py", "src/a/y.py", "src/a/b/z.py", "src/a/b/c/w.py", "src/q.py"]
    assert collect.directory_findings(files, 2) == [
        collect.Finding(collect.DIRECTORY_ENTRIES, "src/a", collect.MODULE, 3, 1)
    ]


def test_a_directory_is_judged_when_the_change_touches_a_file_under_it() -> None:
    outcome = code_metrics_report.Outcome(touched={"src/a/b/z.py"})
    assert [outcome.judged(p) for p in ("src/a", "src/a/b", "src/a/b/z.py", "src/ab", "src/a/x.py")] == [
        True,
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(("at_base", "title"), [(51, WORSE), (None, NEW)])
def test_directory_size_is_compared_with_the_base(repo: Repo, at_base, title: str) -> None:
    base = repo.change("src/a/new.py")
    if at_base is not None:
        repo.base = [collect.Finding(collect.DIRECTORY_ENTRIES, "src/a", collect.MODULE, at_base, 1)]
    repo.findings = [collect.Finding(collect.DIRECTORY_ENTRIES, "src/a", collect.MODULE, 52, 1)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    shown = "| 51 | 52 |" if at_base else "| 52 | fails > 50 |"
    assert f"`src/a:1` | Directory size (entries) {shown}" in section(report, title)


def test_the_base_is_the_tree_at_the_merge_base_measured_with_this_trees_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True
        )

    (tmp_path / "src/tests").mkdir(parents=True)
    (tmp_path / "src/a.py").write_text("base = 1\n", encoding="utf-8")
    (tmp_path / "src/tests/test_a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "docs.md").write_text("not in scope\n", encoding="utf-8")
    # A committed file an ignore rule names is measured in this tree, so the base measures it too.
    (tmp_path / "src/.gitignore").write_text("gen.py\n", encoding="utf-8")
    (tmp_path / "src/gen.py").write_text("x = 1\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("add", "-f", "src/gen.py")
    git("commit", "-q", "-m", "base")
    (tmp_path / "src/a.py").write_text("head = 1\n", encoding="utf-8")
    (tmp_path / "src/new.py").write_text("x = 1\n", encoding="utf-8")
    config = code_metrics.parse_config((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config is not None and config.roots == ("src", "scripts")
    seen = {}

    def measure(root: Path, config, units) -> dict:
        seen["files"] = collect.list_files(root, config.roots, config.exclude)
        seen["a"] = (root / "src/a.py").read_text(encoding="utf-8")
        seen["docs"] = (root / "docs.md").exists()
        return {"measured": root != tmp_path}

    monkeypatch.setattr(code_metrics, "measure", measure)
    # ``scripts`` is a root this commit does not have: the base is the roots it has.
    assert code_metrics.measure_base(tmp_path, "HEAD", config) == {"measured": True}
    assert collect.list_files(tmp_path, config.roots, config.exclude)[0] == ["src/a.py", "src/gen.py", "src/new.py"]
    assert seen == {"files": (["src/a.py", "src/gen.py"], ["src/tests/test_a.py"]), "a": "base = 1\n", "docs": False}
    # A base the tree cannot be read from is "could not run", not a failed verdict.
    with pytest.raises(collect.ToolError, match=r"^git ls-tree -z --name-only exited 128: fatal: "):
        code_metrics.measure_base(tmp_path, "0" * 40, config)
    with pytest.raises(collect.ToolError, match=r"^git -C .+ archive 0+ exited 128: fatal: "):
        code_metrics._checked(["git", "-C", str(tmp_path), "archive", "0" * 40])


def test_ci_reads_the_pr_from_the_api_and_logs_the_full_report() -> None:
    steps = workflow_steps("code-metrics.yml")
    gate = steps["Code metrics gate"]
    assert gate["env"]["PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    assert gate["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    run = gate["run"]
    pull_request_branch = run.split('if [ "$EVENT" = pull_request ]; then', 1)[1].split("else", 1)[0]
    assert 'args+=(--pr-number "$PR_NUMBER")' in pull_request_branch
    push_branch = run.split('if [ "$EVENT" = pull_request ]; then', 1)[1].split("else", 1)[1].split("fi", 1)[0]
    assert 'args+=(--pushed-sha "$GITHUB_SHA")' in push_branch
    assert "--pull-request-json" not in run
    # stdout is the untruncated report (waived rows included) for the job log.
    assert 'python "$impl/code_metrics.py" "${args[@]}"\n' in run and "/dev/null" not in run.split("set +e", 1)[1]


def test_the_local_install_line_names_every_pinned_python_tool() -> None:
    outcome = code_metrics_report.Outcome()
    outcome.versions = {"ruff": "0.16.2", "complexipy": "8.0.1", "radon": "6.0.1", "vulture": "2.16", "jscpd": "5.4.1"}
    fix = code_metrics_report._how_to_fix(outcome)
    assert "`pip install ruff==0.16.2 complexipy==8.0.1 radon==6.0.1 vulture==2.16`" in fix
