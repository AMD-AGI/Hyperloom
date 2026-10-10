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
from code_metrics_collect import MODULE, Finding

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "scripts/code_metrics_baseline.json"
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
        (path / "scripts").mkdir()
        shutil.copyfile(ROOT / "pyproject.toml", path / "pyproject.toml")
        self.git("init", "-q", "-b", "main")
        monkeypatch.setattr(code_metrics, "measure", lambda root, config: collect.worst_per_unit(self.findings))
        monkeypatch.setattr(code_metrics, "check_tools", lambda config: dict(VERSIONS))

    def git(self, *args: str) -> str:
        argv = ["git", "-C", str(self.path), "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()

    def write_baseline(self, entries: dict[tuple[str, str, str], int]) -> None:
        (self.path / BASELINE).write_text(code_metrics.dump_baseline(entries), encoding="utf-8")

    def baseline(self) -> dict[tuple[str, str, str], int]:
        return code_metrics.load_baseline((self.path / BASELINE).read_text(encoding="utf-8"))

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
    assert "| Cyclomatic complexity | 15 | fails > 10 |" in rows
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
    assert "| Cyclomatic complexity | fails > 10 | 0 | 0 | 0 | 1 | -1 |" in report


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


@pytest.mark.parametrize(("unit", "value"), [("Planner.pick", 15), ("Planner.choose", 14)])
def test_move_that_got_worse_or_renamed_is_a_new_violation(repo: Repo, unit: str, value: int) -> None:
    repo.write_baseline({key("src/old.py", "Planner.pick"): 14})
    repo.findings = [cc("src/new.py", unit, value)]
    code, report = repo.gate()
    assert code == 1
    assert f"`{unit}`" in section(report, NEW)
    assert "within the limit, or removed" in section(report, OUT_OF_DATE)


def test_maintainability_index_ratchets_upward(repo: Repo) -> None:
    mi = "maintainability-index"
    repo.write_baseline({key("src/a.py", MODULE, mi): 8, key("src/b.py", MODULE, mi): 8})
    repo.findings = [Finding(mi, "src/a.py", MODULE, 7, 1), Finding(mi, "src/b.py", MODULE, 9, 1)]
    code, report = repo.gate()
    assert code == 1
    assert "`src/a.py:1` | Maintainability index | 8 | 7 |" in section(report, WORSE)
    assert "`src/b.py` | Maintainability index | 8 | 9 |" in section(report, OUT_OF_DATE)
    assert "| Maintainability index | fails < 10 |" in report


def test_loosening_the_config_relative_to_base_fails(repo: Repo) -> None:
    repo.write_baseline({})
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    text = text.replace("cyclomatic-complexity = 10\n", "cyclomatic-complexity = 25\n")
    text = text.replace(
        'exclude = [\n    "src/hyperloom/orch', 'exclude = [\n    "src/hyperloom/agents",\n    "src/hyperloom/orch', 1
    )
    pyproject.write_text(text, encoding="utf-8")
    code, report = repo.gate("--base-ref", base)
    assert code == 1
    loosened = section(report, "### Gate configuration loosened")
    assert "threshold `cyclomatic-complexity` loosened from 10 to 25" in loosened
    assert "`exclude` gained `src/hyperloom/agents`" in loosened


def test_moving_the_baseline_file_does_not_escape_the_base_comparison(repo: Repo) -> None:
    repo.write_baseline({key("src/a.py", "f"): 12})
    base = repo.commit()
    pyproject = repo.path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(f'baseline = "{BASELINE}"', 'baseline = "other.json"'),
        encoding="utf-8",
    )
    (repo.path / "other.json").write_text(code_metrics.dump_baseline({key("src/a.py", "f"): 20}), encoding="utf-8")
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
    assert "...and 5 more" in report
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
            "code": "PLR0904",
            "message": "Too many public methods (21 > 20)",
            "filename": str(tmp_path / "src/p.py"),
            "location": {"row": 1},
        },
        {
            "code": "PLR0911",
            "message": "Too many return statements (7 > 6)",
            "filename": str(tmp_path / "src/p.py"),
            "location": {"row": 14},
        },
    ]
    got = [(f.metric, f.unit, f.value, f.line) for f in collect.parse_ruff(rows, tmp_path, units)]
    assert got == [
        (CC, "Planner.pick", 11, 2),
        ("nested-blocks", "Planner.pick", 6, 4),
        ("public-methods", "Planner", 21, 1),
        ("returns", "helper.inner", 7, 14),
    ]
    with pytest.raises(collect.ToolError, match="unexpected ruff diagnostic"):
        collect.parse_ruff([{**rows[0], "message": "reworded"}], tmp_path, units)


def test_complexipy_rows_keep_only_violations_with_their_line(tmp_path: Path) -> None:
    units = FakeUnits(tmp_path, "src/p.py", SOURCE).units
    rows = [
        {"complexity": 16, "function_name": "Planner::pick", "path": str(tmp_path / "src/p.py")},
        {"complexity": 15, "function_name": "helper", "path": str(tmp_path / "src/p.py")},
    ]
    got = collect.parse_complexipy(rows, tmp_path, 15, units)
    assert got == [Finding("cognitive-complexity", "src/p.py", "Planner.pick", 16, 2)]


def test_radon_rank_c_is_exactly_the_violation(tmp_path: Path) -> None:
    data = {str(tmp_path / f"{name}.py"): {"mi": mi, "rank": "x"} for name, mi in [("a", 9.0), ("b", 9.01), ("c", 0.0)]}
    got = {f.path: f.value for f in collect.parse_radon(data, tmp_path, 10)}
    assert got == {"a.py": 9, "c.py": 0}
    with pytest.raises(collect.ToolError, match="could not measure"):
        collect.parse_radon({str(tmp_path / "d.py"): {"error": "bad"}}, tmp_path, 10)


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
    baseline = json.loads((ROOT / config.baseline).read_text(encoding="utf-8"))
    assert not [path for path in baseline["files"] if Path(path).name.startswith("code_metrics")]


def test_comment_poster_never_runs_pull_request_code() -> None:
    poster = yaml.safe_load((ROOT / ".github/workflows/code-metrics-comment.yml").read_text(encoding="utf-8"))
    assert poster["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "write"}
    steps = [step for job in poster["jobs"].values() for step in job["steps"]]
    checkouts = [step for step in steps if "checkout" in str(step.get("uses", ""))]
    assert [step["with"]["ref"] for step in checkouts] == ["${{ github.event.repository.default_branch }}"]
    gate = yaml.safe_load((ROOT / ".github/workflows/code-metrics.yml").read_text(encoding="utf-8"))
    assert gate["jobs"]["code-metrics"]["name"] == "code-metrics"
    assert gate["name"] in poster[True]["workflow_run"]["workflows"]
