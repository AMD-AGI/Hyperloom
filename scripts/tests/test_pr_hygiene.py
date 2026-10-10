# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/pr_hygiene.py and the workflows that call it."""

from __future__ import annotations

import json
import subprocess
import urllib.error
from pathlib import Path

import pytest
import yaml

import pr_hygiene
from pr_hygiene import check_exemption, is_production_python, main, parse_numstat_z, template_findings

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (ROOT / ".github/PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- size


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/hyperloom/cli.py", True),
        ("scripts/pr_hygiene.py", True),
        ("src/hyperloom/orchestrator/tests/test_x.py", False),
        ("scripts/tests/test_pr_hygiene.py", False),
        ("src/kernelforge/test/helpers.py", False),
        ("scripts/test_conc_sweep_flow.py", False),
        ("src/hyperloom/conftest.py", False),
        ("src/hyperloom/x_test.py", False),
        ("docs/conf.py", False),
        ("src/hyperloom/README.md", False),
        ("tests/src/x.py", False),
    ],
)
def test_production_python_predicate(path, expected):
    assert is_production_python(path) is expected


def test_numstat_rename_counts_only_edits_under_the_new_path():
    raw = b"3\t1\tsrc/a.py\x000\t0\t\x00src/old.py\x00src/new.py\x002\t2\t\x00src/b.py\x00src/c.py\x00-\t-\tbin.png\x00"
    assert parse_numstat_z(raw) == [
        (None, "src/a.py", 4),
        ("src/old.py", "src/new.py", 0),
        ("src/b.py", "src/c.py", 4),
        (None, "bin.png", 0),
    ]


def test_size_counts_a_test_file_moved_into_production_in_full(repo, monkeypatch, capsys):
    (repo / "src/tests").mkdir()
    (repo / "src/tests/test_big.py").write_text("".join(f"v{i} = {i}\n" for i in range(1001)), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "test file")
    _git(repo, "mv", "src/tests/test_big.py", "src/runtime.py")
    _git(repo, "commit", "-qm", "move into production")
    rc, out = _size(repo, monkeypatch, capsys)
    assert rc == 1
    assert "diff budget exceeded: 1001 production Python lines changed (limit 1000)" in out


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/keep.py").write_text("x = 1\n" * 50, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _change(repo: Path, prod: int, test: int = 0, move: bool = False) -> None:
    if move:
        _git(repo, "mv", "src/keep.py", "src/moved.py")
    (repo / "src/new.py").write_text("y = 2\n" * prod, encoding="utf-8")
    (repo / "src/tests").mkdir(exist_ok=True)
    (repo / "src/tests/test_new.py").write_text("z = 3\n" * test, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")


def _size(repo: Path, monkeypatch, capsys, labels: list[str] | None = None) -> tuple[int, str]:
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PR_LABELS", json.dumps(labels or []))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rc = main(["size", "--base", "HEAD^1", "--head", "HEAD"])
    return rc, capsys.readouterr().out


def test_size_over_limit_refuses_with_budget_message(repo, monkeypatch, capsys):
    _change(repo, prod=1001)
    rc, out = _size(repo, monkeypatch, capsys)
    assert rc == 1
    assert "::error title=diff-budget::diff budget exceeded: 1001 production Python lines changed (limit 1000)" in out
    assert "(no 'size-exception' label)" in out


def test_size_at_limit_passes_with_warning(repo, monkeypatch, capsys):
    _change(repo, prod=1000, test=5000, move=True)
    rc, out = _size(repo, monkeypatch, capsys)
    assert rc == 0
    assert "::warning title=diff-budget::1000 production Python lines changed is above the 400-line" in out
    assert "::error" not in out


def test_size_small_change_is_silent(repo, monkeypatch, capsys):
    _change(repo, prod=400)
    rc, out = _size(repo, monkeypatch, capsys)
    assert rc == 0
    assert "::" not in out
    assert "diff budget: 400 production Python lines changed (size/L)" in out


def test_size_label_without_verification_still_refuses(repo, monkeypatch, capsys):
    _change(repo, prod=1500)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    rc, out = _size(repo, monkeypatch, capsys, labels=["size-exception"])
    assert rc == 1
    assert "'size-exception' is present but GITHUB_REPOSITORY/PR_NUMBER/GH_TOKEN is unset" in out


def test_size_verified_label_exempts(repo, monkeypatch, capsys):
    _change(repo, prod=1500)
    monkeypatch.setattr(
        pr_hygiene, "check_exemption", lambda *a, **k: pr_hygiene.Exemption(True, "'size-exception' added by m (write)")
    )
    rc, out = _size(repo, monkeypatch, capsys, labels=["size-exception"])
    assert rc == 0
    assert "::notice title=diff-budget::1500 production Python lines changed exceeds" in out


def _fake_get(events: list, perm: dict | Exception):
    def get(url: str, token: str):
        if "/timeline" in url:
            if "page=2" in url:
                return events[1:], ""
            return events[:1], f'<{url}&page=2>; rel="next"'
        if isinstance(perm, Exception):
            raise perm
        return perm, ""

    return get


def _labeled(login: str, name: str = "size-exception") -> dict:
    return {"event": "labeled", "label": {"name": name}, "actor": {"login": login}}


@pytest.mark.parametrize(
    ("events", "perm", "granted", "reason"),
    [
        ([_labeled("writer")], {"permission": "write", "role_name": "write"}, True, "added by writer (write)"),
        ([_labeled("m")], {"permission": "write", "role_name": "maintain"}, True, "added by m (maintain)"),
        ([_labeled("t")], {"permission": "read", "role_name": "triage"}, False, "whose role 'triage' lacks write"),
        # The most recent labeler decides, across pages.
        ([_labeled("writer"), _labeled("t")], {"permission": "read", "role_name": "triage"}, False, "added by t"),
        ([_labeled("x", "other")], {}, False, "no labeled event names who added it"),
        ([_labeled("bot[bot]")], urllib.error.URLError("404"), False, "could not be verified"),
    ],
)
def test_exemption_requires_write_access_of_last_labeler(events, perm, granted, reason):
    got = check_exemption(["size-exception"], "o/r", "7", "tok", get=_fake_get(events, perm))
    assert got.granted is granted
    assert reason in got.reason


def test_no_label_never_calls_api():
    def boom(*_a):
        raise AssertionError("API called without the label")

    assert check_exemption(["other"], "o/r", "7", "tok", get=boom).granted is False


# ----------------------------------------------------------------------------- template

ANSWERED = """\
- Description: adds a thing
- Linked issue(s): none
- Tests: added/updated? commands run? ([what to test](../docs/contributing/style-guide.md#tests-pytest); \
if a test was replaced, where its assertions live now): pytest scripts/tests
- [Size/complexity](../docs/contributing/style-guide.md#size-and-complexity) triggers crossed: n/a
- Observable effect:
  operators see a new summary table
- Breaking changes: no
- PR addresses single concern: yes
"""


def test_template_fields_all_exist_in_the_real_template():
    prompts = pr_hygiene.template_prompts(TEMPLATE)
    assert sorted(prompts) == list(range(len(pr_hygiene.TEMPLATE_FIELDS)))


def test_untouched_template_reports_every_required_field():
    assert template_findings(TEMPLATE, TEMPLATE) == [
        "'Tests' field is not answered",
        "'Size/complexity' field is not answered",
        "'Observable effect' field is not answered",
        "'Breaking changes' field is not answered",
        "'Single concern' field is not answered",
    ]


def test_answered_body_passes():
    assert template_findings(ANSWERED, TEMPLATE) == []


@pytest.mark.parametrize(
    ("edit", "finding"),
    [
        (("- PR addresses single concern: yes\n", ""), "'Single concern' field is missing"),
        (("concern: yes", "concern: maybe"), "'Single concern' must be answered yes or no"),
        (("concern: yes", "concern: yes/no (details if no): "), "'Single concern' field is not answered"),
        (("concern: yes", "concern: yes/no"), "'Single concern' must be answered yes or no"),
        (("- Breaking changes: no\n", "- Breaking changes:\n"), "'Breaking changes' field is not answered"),
    ],
)
def test_template_refusal_names_the_field(edit, finding):
    assert template_findings(ANSWERED.replace(*edit), TEMPLATE) == [finding]


def test_nested_and_colonless_bullets_do_not_start_a_field():
    body = ANSWERED.replace("- Description: adds a thing\n", "- Description: x\n- Tests pass now\n  - Tests: fake\n")
    assert template_findings(body, TEMPLATE) == []


def test_empty_body_reports_missing_fields():
    assert len(template_findings("", TEMPLATE)) == 5


@pytest.mark.parametrize(("enforce", "rc", "level"), [(False, 0, "warning"), (True, 1, "error")])
def test_template_command_is_advisory_unless_enforced(monkeypatch, capsys, enforce, rc, level):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PR_BODY", "%0A::error::injected\n- Tests:")
    monkeypatch.delenv("PR_AUTHOR_TYPE", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["template", *(["--enforce"] if enforce else [])]) == rc
    out = capsys.readouterr().out
    assert f"::{level} title=pr-template::'Tests' field is not answered" in out
    assert "injected" not in out


def test_bot_prs_skip_template(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PR_BODY", "")
    monkeypatch.setenv("PR_AUTHOR_TYPE", "Bot")
    assert main(["template", "--enforce"]) == 0


# ----------------------------------------------------------------------------- workflows


def test_pr_hygiene_workflow_always_runs_read_only():
    wf = _workflow("pr-hygiene.yml")
    on = wf[True]  # YAML 1.1 reads the bare ``on`` key as True
    assert set(on) == {"pull_request"}
    assert "paths" not in on["pull_request"] and "paths-ignore" not in on["pull_request"]
    assert {"labeled", "unlabeled", "edited", "synchronize"} <= set(on["pull_request"]["types"])
    assert wf["permissions"] == {"contents": "read", "pull-requests": "read"}
    steps = {s.get("name"): s for s in wf["jobs"]["pr-hygiene"]["steps"]}
    tpl = steps["PR template completeness"]
    # The body reaches the script only as data, never inside the shell text.
    assert tpl["env"]["PR_BODY"] == "${{ github.event.pull_request.body }}"
    assert "github.event" not in tpl["run"]
    assert tpl["env"]["PR_TEMPLATE_ENFORCE"] == "false"
    cli = steps["Check CLI references in agent docs"]
    assert cli["if"] == "always()"
    # PR-controlled paths are printed only while workflow commands are suspended.
    run = cli["run"]
    assert run.index("::stop-commands::") < run.index("check_cli_references.py") < run.index('echo "::${resume}::"')
    names = list(steps)
    assert names.index("Force text diffs for Python files") < names.index("Diff budget")
    assert "'*.py diff'" in steps["Force text diffs for Python files"]["run"]


def test_cli_reference_check_moved_out_of_docs_workflow():
    wf = _workflow("docs.yml")
    runs = [s.get("run", "") for job in wf["jobs"].values() for s in job["steps"]]
    assert not any("check_cli_references" in r for r in runs)
    assert not any("check_cli_references" in p for p in wf[True]["pull_request"]["paths"])


def test_diff_coverage_step_is_pr_only_and_pinned():
    job = _workflow("tests-coverage.yml")["jobs"]["coverage"]
    assert job["steps"][0]["with"]["fetch-depth"] == 2
    step = next(s for s in job["steps"] if s.get("name") == "Diff coverage (changed lines >= 80%)")
    assert "github.event_name == 'pull_request'" in step["if"]
    assert '"diff-cover==9.7.1"' in step["run"]
    assert step["run"].index("'*.py diff'") < step["run"].index("diff-cover coverage.xml")
    assert "--fail-under=80" in step["run"]
    assert "GITHUB_STEP_SUMMARY" in step["run"]
