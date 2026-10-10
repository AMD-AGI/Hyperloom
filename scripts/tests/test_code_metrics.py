# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of scripts/code_metrics.py: the ratchet, its refusals, and the report.

The gate is driven through ``main()`` in a scratch git repository; only the tool runs
(``measure`` and the version check) are replaced, so baseline I/O, the base-ref
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
from code_metrics_collect import Finding

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "scripts/code_metrics_baseline.txt"
VERSIONS = {"ruff": "0.16.2"}
CC = "cyclomatic-complexity"
OUT_OF_DATE = "### Baseline out of date (run `python scripts/code_metrics.py --update-baseline`)"
GREW = "### Baseline grew relative to the base branch"
NEW = "### New violations (not in the baseline)"
WORSE = "### Worse than the baseline"


class Repo:
    """A git repo holding the real pyproject.toml and a baseline, with the tools stubbed."""

    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = path
        self.findings: list[Finding] = []
        #: What the per-file tools would report for the touched files at the base.
        self.at_base: dict[tuple[str, str, str], int] = {}
        (path / "scripts").mkdir()
        shutil.copyfile(ROOT / "pyproject.toml", path / "pyproject.toml")
        self.git("init", "-q", "-b", "main")
        monkeypatch.setattr(code_metrics, "measure", lambda root, config, units: collect.worst_per_unit(self.findings))
        monkeypatch.setattr(code_metrics, "base_values", lambda root, since, paths, config: dict(self.at_base))
        monkeypatch.setattr(code_metrics, "check_tools", lambda config: dict(VERSIONS))

    def git(self, *args: str) -> str:
        argv = ["git", "-C", str(self.path), "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()

    def write_baseline(self, entries: dict[tuple[str, str, str], int]) -> None:
        (self.path / BASELINE).write_text(code_metrics.dump_baseline(entries), encoding="utf-8")

    def baseline(self) -> dict[tuple[str, str, str], int]:
        return code_metrics.load_baseline((self.path / BASELINE).read_text(encoding="utf-8"))

    def touch(self, *paths: str) -> None:
        """Change ``paths`` in the working tree, so a ``--base-ref`` run judges them."""
        for path in paths:
            (self.path / path).parent.mkdir(parents=True, exist_ok=True)
            with (self.path / path).open("a", encoding="utf-8") as handle:
                handle.write("x = 1\n")

    def commit(self) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "state")
        return self.git("rev-parse", "HEAD")

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


def cc(path: str, unit: str, value: int, line: int = 3) -> Finding:
    return Finding(CC, path, unit, value, line)


def key(path: str, unit: str, metric: str = CC) -> tuple[str, str, str]:
    return (metric, path, unit)


def test_clean_tree_with_matching_baseline_passes(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    repo.findings = [cc("src/a.py", "f", 12)]
    code, report = repo.gate()
    assert code == 0
    assert "## Code metrics gate: PASSED" in report


def test_new_violation_fails_and_report_links_the_offender(repo: Repo) -> None:
    repo.write_baseline({})
    repo.findings = [cc("src/new.py", "Planner.pick", 15, line=42)]
    code, report = repo.gate("--link-base", "https://github.com/o/r/blob/abc123")
    assert code == 1
    assert "## Code metrics gate: FAILED" in report
    rows = section(report, NEW)
    assert "[`src/new.py:42`](https://github.com/o/r/blob/abc123/src/new.py#L42) `Planner.pick`" in rows
    assert "| Cyclomatic complexity | 15 | fails > 20 |" in rows
    assert section(report, WORSE) == section(report, OUT_OF_DATE) == ""


def test_baselined_unit_that_got_worse_fails(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    repo.findings = [cc("src/a.py", "f", 13)]
    code, report = repo.gate()
    assert code == 1
    assert "`f` | Cyclomatic complexity | 12 | 13 |" in section(report, WORSE)
    assert section(report, NEW) == ""


@pytest.mark.parametrize(("now", "shown"), [(11, "| 12 | 11 |"), (None, "| 12 | within the limit, or removed |")])
def test_improved_or_removed_unit_fails_as_out_of_date_naming_the_command(repo: Repo, now, shown) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    repo.findings = [] if now is None else [cc("src/a.py", "f", now)]
    code, report = repo.gate()
    assert code == 1
    assert shown in section(report, OUT_OF_DATE)
    assert "Run `python scripts/code_metrics.py --update-baseline`" in report


def test_update_baseline_tightens_and_then_the_gate_passes_against_the_base(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12, key("src/a.py", "gone"): 20})
    base = repo.commit()
    repo.findings = [cc("src/a.py", "f", 11)]
    assert repo.gate("--update-baseline")[0] == 0
    assert repo.baseline() == {key("src/a.py", "f"): 11}
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert "| Cyclomatic complexity | fails > 20 | 0 | 0 | 0 | 1 | -1 |" in report


def test_update_baseline_never_absorbs_a_worse_or_new_unit(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    repo.findings = [cc("src/a.py", "f", 14), cc("src/b.py", "g", 30)]
    assert repo.gate("--update-baseline")[0] == 0
    assert repo.baseline() == {key("src/a.py", "f"): 12}
    code, report = repo.gate()
    assert code == 1
    assert "`g`" in section(report, NEW)
    assert "| 12 | 14 |" in section(report, WORSE)


def test_baseline_entry_added_relative_to_base_fails(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    base = repo.commit()
    repo.write_baseline({key("src/a.py", "f"): 12, key("src/b.py", "g"): 30})
    repo.findings = [cc("src/a.py", "f", 12), cc("src/b.py", "g", 30)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`src/b.py` `g` | Cyclomatic complexity | absent | 30 |" in section(report, GREW)
    assert section(report, NEW) == ""


def test_baseline_value_raised_relative_to_base_fails(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    base = repo.commit()
    repo.write_baseline({key("src/a.py", "f"): 14})
    repo.findings = [cc("src/a.py", "f", 14)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`src/a.py` `f` | Cyclomatic complexity | 12 | 14 |" in section(report, GREW)
    assert section(report, WORSE) == ""


def test_moved_unit_is_matched_not_new_and_its_update_passes_against_the_base(repo: Repo) -> None:
    repo.write_baseline({key("src/old.py", "Planner.pick"): 14})
    base = repo.commit()
    repo.findings = [cc("src/new.py", "Planner.pick", 13)]
    code, report = repo.gate()
    assert code == 1
    assert section(report, NEW) == ""
    assert "moved to `src/new.py:3` `Planner.pick` (13)" in section(report, OUT_OF_DATE)
    assert repo.gate("--update-baseline")[0] == 0
    assert repo.baseline() == {key("src/new.py", "Planner.pick"): 13}
    assert repo.gate("--base-ref", base)[0] == 0


@pytest.mark.parametrize(("unit", "value"), [("Planner.pick", 15), ("Planner.choose", 14), ("Scheduler.pick", 14)])
def test_move_that_got_worse_or_renamed_is_a_new_violation(repo: Repo, unit: str, value: int) -> None:
    repo.write_baseline({key("src/old.py", "Planner.pick"): 14})
    repo.findings = [cc("src/new.py", unit, value)]
    code, report = repo.gate()
    assert code == 1
    assert f"`{unit}`" in section(report, NEW)
    assert "within the limit, or removed" in section(report, OUT_OF_DATE)


@pytest.mark.parametrize(
    "baseline",
    [
        # A.run improves under the limit while B.run worsens; both move.
        {key("src/a.py", "A.run"): 14, key("src/b.py", "B.run"): 11},
        # Two same-named units: the worse one must not borrow the other's allowance.
        {key("src/a.py", "main"): 14, key("src/b.py", "main"): 11},
    ],
)
def test_a_worse_unit_cannot_borrow_another_units_allowance_by_moving(repo: Repo, baseline) -> None:
    repo.write_baseline(baseline)
    base = repo.commit()
    unit = sorted(baseline)[1][2]
    repo.findings = [cc("src/new.py", unit, 13)]
    code, report = repo.gate()
    assert code == 1
    assert f"`{unit}` | Cyclomatic complexity | 13 |" in section(report, NEW)
    assert repo.gate("--update-baseline")[0] == 0
    assert key("src/new.py", unit) not in repo.baseline()
    repo.touch("src/new.py")
    assert repo.gate("--base-ref", base)[0] == 1


def test_a_baseline_entry_cannot_move_away_from_a_unit_that_is_still_over(repo: Repo) -> None:
    repo.write_baseline({key("src/old.py", "f"): 30})
    base = repo.commit()
    # The entry is edited over to a new same-named unit while the old one stays at 30.
    repo.write_baseline({key("src/new.py", "f"): 30})
    repo.findings = [cc("src/old.py", "f", 30), cc("src/new.py", "f", 30)]
    repo.touch("src/new.py")
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`src/new.py` `f` | Cyclomatic complexity | absent | 30 |" in section(report, GREW)
    # A real move (the old unit is gone) still keeps the entry.
    repo.findings = [cc("src/new.py", "f", 30)]
    code, report = repo.gate("--base-ref", base)
    assert section(report, GREW) == ""


SECRET = "GITHUB_TOKEN=ghs_not_a_real_token"


@pytest.mark.parametrize("baseline", ["/proc/self/environ", "../outside.txt", "scripts/"])
def test_the_baseline_path_must_be_a_plain_repository_file(repo: Repo, baseline: str) -> None:
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(text.replace(f'baseline = "{BASELINE}"', f'baseline = "{baseline}"'), encoding="utf-8")
    code, report = repo.gate()
    assert code == 2
    assert "`baseline` takes a plain repository file path" in report


def test_a_symlinked_baseline_is_refused_without_reading_it(repo: Repo, tmp_path_factory) -> None:
    secret = tmp_path_factory.mktemp("outside") / "environ"
    secret.write_text(SECRET + "\n", encoding="utf-8")
    (repo.path / BASELINE).symlink_to(secret)
    code, report = repo.gate()
    assert code == 2
    assert f"`{BASELINE}` must be a regular file in the repository, not a symlink" in report
    assert SECRET not in report


def test_a_malformed_baseline_line_is_not_echoed(repo: Repo) -> None:
    (repo.path / BASELINE).write_text(SECRET + "\n", encoding="utf-8")
    code, report = repo.gate()
    assert code == 2
    assert "baseline line 1 is not `<metric> <path>[::<unit>] <value>`" in report
    assert SECRET not in report


def test_list_files_skips_symlinks(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src/b.py").symlink_to(tmp_path / "src/a.py")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    assert collect.list_files(tmp_path, ["src"], []) == (["src/a.py"], [])


def test_a_unit_copied_to_two_places_is_not_a_move(repo: Repo) -> None:
    repo.write_baseline({key("src/old.py", "A.run"): 14})
    repo.findings = [cc("src/x.py", "A.run", 12), cc("src/y.py", "A.run", 12)]
    code, report = repo.gate()
    assert code == 1
    assert "### New violations (not in the baseline): 2" in report


def test_loosening_the_config_relative_to_base_fails(repo: Repo) -> None:
    repo.write_baseline({})
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


def test_moving_the_baseline_file_does_not_escape_the_base_comparison(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(f'baseline = "{BASELINE}"', 'baseline = "other.txt"'),
        encoding="utf-8",
    )
    (repo.path / "other.txt").write_text(code_metrics.dump_baseline({key("src/a.py", "f"): 20}), encoding="utf-8")
    repo.findings = [cc("src/a.py", "f", 20)]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "| 12 | 20 |" in section(report, GREW)


def test_base_without_the_gate_skips_growth_with_a_note(repo: Repo) -> None:
    (repo.path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    base = repo.commit()
    shutil.copyfile(ROOT / "pyproject.toml", repo.path / "pyproject.toml")
    repo.write_baseline({key("src/a.py", "f"): 12})
    repo.findings = [cc("src/a.py", "f", 12)]
    code, report = repo.gate("--base-ref", base)
    assert code == 0
    assert "has no code-metrics config yet" in report


def test_unresolvable_base_ref_or_tool_failure_cannot_pass(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    repo.write_baseline({})
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


def test_seed_refuses_to_overwrite_an_existing_baseline(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    repo.findings = [cc("src/a.py", "f", 40), cc("src/b.py", "g", 40)]
    assert code_metrics.main(["--root", str(repo.path), "--seed-baseline"]) == 2
    assert repo.baseline() == {key("src/a.py", "f"): 12}


def test_report_truncates_long_sections(repo: Repo) -> None:
    repo.write_baseline({})
    repo.findings = [cc(f"src/m{i:02d}.py", "f", 11) for i in range(30)]
    _code, report = repo.gate()
    assert "### New violations (not in the baseline): 30" in report
    assert "...and 5 more; the job log lists every row." in report
    assert "src/m24.py" in report and "src/m25.py" not in report


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
            "code": "C901",
            "message": "`pick` is too complex (11 > 10)",
            "filename": str(tmp_path / "src/p.py"),
            "location": {"row": 2},
        },
        {
            "code": "PLR1702",
            "message": "Too many nested blocks (6 > 5)",
            "filename": str(tmp_path / "src/p.py"),
            "location": {"row": 4},
        },
        {
            "code": "C901",
            "message": "`inner` is too complex (21 > 20)",
            "filename": str(tmp_path / "src/p.py"),
            "location": {"row": 14},
        },
    ]
    got = [(f.metric, f.unit, f.value, f.line) for f in collect.parse_ruff(rows, tmp_path, units)]
    assert got == [
        (CC, "Planner.pick", 11, 2),
        ("nested-blocks", "Planner.pick", 6, 4),
        (CC, "helper.inner", 21, 14),
    ]
    with pytest.raises(collect.ToolError, match="unexpected ruff diagnostic"):
        collect.parse_ruff([{**rows[0], "message": "reworded"}], tmp_path, units)
    # A rule this gate no longer selects is unexpected output, not a silently dropped row.
    with pytest.raises(collect.ToolError, match="unexpected ruff diagnostic: PLR0911"):
        collect.parse_ruff([{**rows[0], "code": "PLR0911"}], tmp_path, units)


def test_complexipy_rows_keep_only_violations_with_their_line(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    rows = [
        {"complexity": 16, "function_name": "Planner::pick", "path": str(tmp_path / "src/p.py")},
        {"complexity": 15, "function_name": "helper", "path": str(tmp_path / "src/p.py")},
    ]
    got = collect.parse_complexipy(rows, tmp_path, 15, units)
    assert got == [Finding("cognitive-complexity", "src/p.py", "Planner.pick", 16, 2)]


def test_vulture_lines_become_scoped_units(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    text = f"{tmp_path}/src/p.py:2: unused variable 'a' (100% confidence)\n"
    assert collect.parse_vulture(text, tmp_path, units) == [Finding("dead-code", "src/p.py", "Planner.pick:a", 1, 2)]
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
    assert set(config.thresholds) == set(collect.METRICS)
    assert config.tools["ruff"] in (ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8")
    baseline = code_metrics.load_baseline((ROOT / config.baseline).read_text(encoding="utf-8"))
    assert not [path for _metric, path, _unit in baseline if Path(path).name.startswith("code_metrics")]
    assert "radon" not in config.tools and "maintainability-index" not in config.thresholds


def test_comment_poster_never_runs_pull_request_code() -> None:
    poster = yaml.safe_load((ROOT / ".github/workflows/code-metrics-comment.yml").read_text(encoding="utf-8"))
    assert poster["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "write"}
    steps = [step for job in poster["jobs"].values() for step in job["steps"]]
    checkouts = [step for step in steps if "checkout" in str(step.get("uses", ""))]
    assert [step["with"]["ref"] for step in checkouts] == ["${{ github.event.repository.default_branch }}"]
    gate = yaml.safe_load((ROOT / ".github/workflows/code-metrics.yml").read_text(encoding="utf-8"))
    assert gate["jobs"]["code-metrics"]["name"] == "code-metrics"
    assert gate["name"] in poster[True]["workflow_run"]["workflows"]


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
    repo.write_baseline({})
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
    repo.write_baseline({})
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(text.replace('roots = ["src", "scripts"]', 'roots = ["src/a"]'))
    base = repo.commit()
    pyproject.write_text(text)
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report


def test_an_edit_to_the_gate_is_reported_for_review_without_failing(repo: Repo) -> None:
    (repo.path / "scripts/code_metrics.py").write_text("x = 1\n", encoding="utf-8")
    repo.write_baseline({})
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
    repo.write_baseline({})
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
    for name in ("code_metrics.py", "code_metrics_checks.py", "code_metrics_collect.py", "code_metrics_report.py"):
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


@pytest.mark.parametrize("refusal", ["Resource not accessible by integration", "API rate limit exceeded"])
def test_a_refused_comment_is_a_warning_not_a_red_job(tmp_path: Path, refusal: str) -> None:
    gate_dir = tmp_path / "temp/code-metrics-gate"
    gate_dir.mkdir(parents=True)
    shutil.copy(ROOT / ".github/scripts/code_metrics_comment.js", gate_dir / "comment.js")
    (tmp_path / "code-metrics").mkdir()
    (tmp_path / "code-metrics/report.md").write_text(f"{MARKER}\n## Code metrics gate: PASSED\n", encoding="utf-8")
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
    assert found == [Finding("dead-code", "src/p.py", "<module>:x", 1, 3)]
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


# --- touched-file scope, base backlog, the override label --------------------

OUTSIDE = "### Outside this change (informational)"
AT_BASE = "### Already over at the base (backlog, not this change's)"
LABEL = "baseline-raise"


def test_a_violation_in_an_untouched_file_never_fails_the_change(repo: Repo) -> None:
    repo.write_baseline({key("src/old.py", "f"): 21, key("src/gone.py", "g"): 30})
    base = repo.commit()
    repo.touch("src/mine.py")
    # Drift on main: new and worse units, and an entry already out of date, all in files this change leaves alone.
    repo.findings = [cc("src/old.py", "f", 25), cc("src/other.py", "h", 40)]
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert section(report, NEW) == section(report, WORSE) == section(report, OUT_OF_DATE) == ""
    outside = section(report, OUTSIDE)
    assert "`src/other.py` `h`" in outside and "`src/old.py` `f`" in outside and "`src/gone.py` `g`" in outside
    assert "Judged: the 1 files this change touches." in report


@pytest.mark.parametrize(
    ("finding", "title"),
    [
        (cc("src/mine.py", "f", 25), NEW),
        (cc("src/old.py", "f", 22), WORSE),
    ],
)
def test_a_new_or_worse_unit_in_a_touched_file_fails(repo: Repo, finding: Finding, title: str) -> None:
    repo.write_baseline({key("src/old.py", "f"): 21})
    base = repo.commit()
    repo.touch(finding.path)
    repo.findings = [finding]
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert f"`{finding.path}:3` `{finding.unit}`" in section(report, title)
    assert finding.path not in section(report, OUTSIDE)


def test_a_fixed_unit_in_a_touched_file_must_leave_the_baseline(repo: Repo) -> None:
    repo.write_baseline({key("src/old.py", "f"): 21})
    base = repo.commit()
    repo.touch("src/old.py")
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    assert "`src/old.py` `f` | Cyclomatic complexity | 21 | within the limit, or removed |" in section(
        report, OUT_OF_DATE
    )
    assert repo.gate("--update-baseline")[0] == 0
    assert repo.gate("--base-ref", base)[0] == 0


@pytest.mark.parametrize(("at_base", "now", "code"), [(None, 21, 1), (24, 25, 1), (25, 25, 0), (30, 25, 0)])
def test_the_complexity_ceiling_compares_head_with_the_base(repo: Repo, at_base, now, code) -> None:
    """A unit the change adds or raises fails; one the base already had at that value or worse is backlog."""
    repo.write_baseline({})
    base = repo.commit()
    repo.touch("src/a.py")
    repo.findings = [cc("src/a.py", "f", now)]
    repo.at_base = {} if at_base is None else {key("src/a.py", "f"): at_base}
    got, report = repo.gate("--base-ref", base)
    assert got == code, report
    if code:
        assert f"| Cyclomatic complexity | {now} | fails > 20 |" in section(report, NEW)
    else:
        assert f"`f` | Cyclomatic complexity | {at_base} | {now} |" in section(report, AT_BASE)


def test_a_worse_than_baseline_unit_the_base_already_had_is_backlog(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 22})
    base = repo.commit()
    repo.touch("src/a.py")
    repo.findings = [cc("src/a.py", "f", 24)]
    repo.at_base = {key("src/a.py", "f"): 24}
    code, report = repo.gate("--base-ref", base)
    assert code == 0, report
    assert "| 24 | 24 |" in section(report, AT_BASE)
    repo.at_base = {key("src/a.py", "f"): 23}
    assert repo.gate("--base-ref", base)[0] == 1


def write_pr(repo: Repo, labels: list[str], **fields) -> str:
    path = repo.path.parent / "pr.json"
    path.write_text(json.dumps({"number": 7, "labels": labels, "title": "t", "body": "b", **fields}), encoding="utf-8")
    return str(path)


def test_the_override_label_waives_new_worse_and_growth_but_lists_them_all(repo: Repo, capsys) -> None:
    repo.write_baseline({key("src/a.py", "f"): 21})
    base = repo.commit()
    repo.write_baseline({key("src/a.py", "f"): 21, key("src/b.py", "g"): 30})
    repo.touch("src/a.py", "src/b.py", *(f"src/m{i:02d}.py" for i in range(30)))
    repo.findings = [cc("src/a.py", "f", 23), cc("src/b.py", "g", 30)]
    repo.findings += [cc(f"src/m{i:02d}.py", "f", 22) for i in range(30)]
    code, _report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, []))
    assert code == 1
    capsys.readouterr()
    code, report = repo.gate("--base-ref", base, "--pull-request-json", write_pr(repo, ["size-exception", LABEL]))
    assert code == 0, report
    assert f"## Code metrics gate: PASSED with waived findings (`{LABEL}`)" in report
    waived = f" (waived by the `{LABEL}` label)"
    assert "`src/a.py:3` `f` | Cyclomatic complexity | 21 | 23 |" in section(report, WORSE + waived)
    assert "`src/b.py` `g` | Cyclomatic complexity | absent | 30 |" in section(report, GREW + waived)
    assert f"{NEW}{waived}: 30" in report and "...and 5 more" in report
    # The job log is the full list: every waived row, untruncated.
    log = capsys.readouterr().out
    assert all(f"src/m{i:02d}.py" in log for i in range(30))


def test_the_override_label_waives_nothing_else(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 21})
    base = repo.commit()
    repo.touch("src/a.py")
    pr = write_pr(repo, [LABEL], title=f"Fix {chr(0x4E2D)}")
    code, report = repo.gate("--base-ref", base, "--pull-request-json", pr)
    assert code == 1
    assert "within the limit, or removed" in section(report, OUT_OF_DATE)
    assert "PR title, line 1 | CJK character U+4E2D" in section(report, "### English only")


def test_pr_title_body_and_commit_messages_must_be_english(repo: Repo) -> None:
    repo.write_baseline({})
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
    repo.write_baseline({})
    repo.findings = [cc("src/a.py", "f", 30)]
    code, report = repo.gate("--pr-number", "7")
    assert code == 0, report
    assert calls == [
        ("https://api.github.com/repos/o/r/pulls/7", "tok"),
        ("https://api.github.com/repos/o/r/pulls/7/commits?per_page=100&page=1", "tok"),
    ]


def test_an_unreadable_api_is_could_not_run_not_a_pass(repo: Repo, monkeypatch) -> None:
    repo.write_baseline({})
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    code, report = repo.gate("--pr-number", "7")
    assert code == 2 and "needs GITHUB_REPOSITORY" in report
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_API_URL", "http://api.example.com")
    code, report = repo.gate("--pr-number", "7")
    assert code == 2 and "the API URL must be https" in report


def test_module_length_between_warning_and_limit_is_reported_not_failed(repo: Repo) -> None:
    repo.write_baseline({})
    base = repo.commit()
    for path, lines in [("src/warn.py", 801), ("src/ok.py", 800), ("src/far.py", 900)]:
        (repo.path / path).parent.mkdir(parents=True, exist_ok=True)
        (repo.path / path).write_text("x = 1\n" * lines, encoding="utf-8")
    repo.git("add", "src/far.py")
    repo.commit()
    (repo.path / "src/warn.py").write_text("x = 1\n" * 801, encoding="utf-8")
    code, report = repo.gate("--base-ref", base)
    warned = section(report, "### Module length warning (not failing)")
    assert "`src/warn.py` | 801 | fails > 1200 |" in warned
    assert "src/ok.py" not in warned
    assert "src/far.py" in warned  # committed after the base: touched too
    assert code == 0, report


def test_function_length_counts_def_to_last_line_and_nested_functions_alone(tmp_path: Path) -> None:
    body = "    x = 1\n" * 77
    source = (
        "@decorator\n@another\n"
        f'def outer():\n    """Doc."""\n    # comment\n{body}'
        "    def inner():\n" + "        y = 2\n" * 80 + "    return inner\n"
    )
    units = FakeUnits(tmp_path, "src/f.py", source).units
    got = {f.unit: (f.value, f.line) for f in collect.function_line_findings(["src/f.py"], 80, units)}
    # outer: def line, docstring, comment, 77 lines, return: 1 + 1 + 1 + 77 + 1; inner is its own unit.
    assert got == {"outer": (81, 3), "outer.inner": (81, 83)}
    assert collect.function_line_findings(["src/f.py"], 81, units) == []


@pytest.mark.skipif(shutil.which("ruff") is None or shutil.which("complexipy") is None, reason="ruff/complexipy")
def test_base_values_measures_the_files_as_they_were_at_the_base(tmp_path: Path) -> None:
    branches = "".join(f"    if x == {i}:\n        return {i}\n" for i in range(21))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text(f"def f(x):\n{branches}    return -1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    git_id = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "-C", str(tmp_path), *git_id, "commit", "-q", "-m", "base"], check=True)
    (tmp_path / "src/a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    config = code_metrics.parse_config((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    got = code_metrics.base_values(tmp_path, "HEAD", ["src/a.py", "src/new.py"], config)
    assert got == {(CC, "src/a.py", "f"): 22}


def test_ci_reads_the_pr_from_the_api_and_logs_the_full_report() -> None:
    steps = workflow_steps("code-metrics.yml")
    gate = steps["Code metrics gate"]
    assert gate["env"]["PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    assert gate["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    run = gate["run"]
    pull_request_branch = run.split('if [ "$EVENT" = pull_request ]; then', 1)[1].split("else", 1)[0]
    assert 'args+=(--pr-number "$PR_NUMBER")' in pull_request_branch
    assert "--pull-request-json" not in run
    # stdout is the untruncated report (waived rows included) for the job log.
    assert 'python "$impl/code_metrics.py" "${args[@]}"\n' in run and "/dev/null" not in run.split("set +e", 1)[1]
