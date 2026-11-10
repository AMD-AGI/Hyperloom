# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The documentation-only classification that decides whether the required test and lint checks measure anything."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("ci_code_changed", ROOT / "scripts" / "ci_code_changed.py")
assert _SPEC is not None and _SPEC.loader is not None
ci_code_changed = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_code_changed)

WORKFLOWS = ROOT / ".github" / "workflows"
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/index.rst",
        "docs/a/b/c.png",
        "src/pkg/notes.md",
        "LICENSE",
        "LICENSE-MIT",
        "COPYRIGHT",
        ".github/CODEOWNERS",
        "CODEOWNERS",
        ".gitattributes",
    ],
)
def test_documentation_paths_are_docs_only(path):
    assert ci_code_changed.is_docs_only(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/pkg/mod.py",
        "pyproject.toml",
        ".github/workflows/lint.yml",
        "scripts/ci_code_changed.py",
        "src/LICENSE",
        "sub/COPYRIGHT",
        "sub/.gitattributes",
        "notes.mdx",
        "mydocs/x.md.py",
        "docsrc/a.txt",
    ],
)
def test_code_and_nested_metadata_paths_are_code(path):
    assert not ci_code_changed.is_docs_only(path)


@pytest.mark.parametrize(
    ("paths", "expected"),
    [(["README.md", "docs/x.rst"], False), (["README.md", "src/a.py"], True), ([], True)],
)
def test_one_code_path_makes_the_event_code(paths, expected):
    assert ci_code_changed.code_changed(paths) is expected


def _repo(tmp_path: Path) -> Path:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "ci@example.com")
    git("config", "user.name", "ci")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    return tmp_path


def _commit(repo: Path, rel: str) -> None:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text("change\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", rel], cwd=repo, check=True)


@pytest.mark.parametrize(("rel", "expected"), [("docs/guide.md", "code=false"), ("src/b.py", "code=true")])
def test_main_classifies_the_commits_it_is_given(tmp_path, monkeypatch, capsys, rel, expected):
    repo = _repo(tmp_path)
    _commit(repo, rel)
    monkeypatch.chdir(repo)
    assert ci_code_changed.main(["HEAD^1", "HEAD"]) == 0
    assert capsys.readouterr().out.strip() == expected


def test_a_branch_creating_push_counts_as_code(capsys):
    assert ci_code_changed.main(["0" * 40, "HEAD"]) == 0
    assert capsys.readouterr().out.strip() == "code=true"


def test_an_unknown_base_fails_rather_than_skipping(tmp_path, monkeypatch):
    monkeypatch.chdir(_repo(tmp_path))
    with pytest.raises(subprocess.CalledProcessError):
        ci_code_changed.main(["1" * 40, "HEAD"])


def test_contributing_lists_the_same_patterns():
    for pattern in ci_code_changed.DOCS_ONLY_PATTERNS:
        assert f"- `{pattern}`" in CONTRIBUTING


@pytest.mark.parametrize("name", ["tests-coverage.yml", "lint.yml"])
def test_required_check_workflows_always_start(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    triggers = workflow[True]  # PyYAML reads the bare ``on:`` key as True.
    assert "paths-ignore" not in (triggers.get("pull_request") or {})
    assert "paths-ignore" not in (triggers.get("push") or {})
    assert workflow["jobs"]["changes"]["uses"] == "./.github/workflows/code-changes.yml"


def _jobs(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))["jobs"]


CODE = "needs.changes.outputs.code == 'true'"


def test_coverage_reports_on_every_event_and_measures_only_code():
    coverage = _jobs("tests-coverage.yml")["coverage"]
    assert coverage["if"] == "always()"
    assert coverage["needs"] == ["changes", "test"]
    guard, notice, *measuring = coverage["steps"]
    assert "needs.changes.result != 'success'" in guard["if"] and "exit 1" in guard["run"]
    assert notice["if"] == "needs.changes.outputs.code == 'false'"
    assert measuring and all(CODE in step["if"] for step in measuring)


def test_shards_run_only_for_code():
    jobs = _jobs("tests-coverage.yml")
    assert jobs["prepare-durations"]["if"] == CODE
    assert jobs["test"]["needs"] == "prepare-durations"


def test_ruff_runs_on_every_event_and_the_other_lint_jobs_only_for_code():
    jobs = _jobs("lint.yml")
    assert "if" not in jobs["ruff"] and "needs" not in jobs["ruff"]
    others = {name: job for name, job in jobs.items() if name not in {"changes", "ruff"}}
    assert others and all(job["if"] == CODE and job["needs"] == "changes" for job in others.values())


def test_a_rename_out_of_code_still_counts_as_code(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "docs").mkdir()
    subprocess.run(["git", "mv", "a.py", "docs/a.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "move"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    assert ci_code_changed.main(["HEAD^1", "HEAD"]) == 0
    assert capsys.readouterr().out.strip() == "code=true"


@pytest.mark.parametrize("pattern", ["doc?.md", "[ab].md", "!x.md"])
def test_unsupported_glob_syntax_is_refused(pattern):
    with pytest.raises(ValueError, match="unsupported glob syntax"):
        ci_code_changed._glob_regex(pattern)


def test_codeql_skips_on_the_same_list():
    triggers = yaml.safe_load((WORKFLOWS / "codeql.yml").read_text(encoding="utf-8"))[True]
    for event in ("push", "pull_request"):
        assert tuple(triggers[event]["paths-ignore"]) == ci_code_changed.DOCS_ONLY_PATTERNS
