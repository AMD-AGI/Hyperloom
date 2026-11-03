# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""review-pr's risk.py grades a PR from the paths it changes, with references/tiers.md as the only source of data."""

from __future__ import annotations

import fnmatch
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RISK = _ROOT / ".claude" / "skills" / "review-pr" / "risk.py"


@pytest.fixture
def risk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("review_pr_risk", _RISK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _files(tmp_path: Path, *paths: str) -> Path:
    files = tmp_path / "files.txt"
    files.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    return files


def _printed(risk: ModuleType, capsys: pytest.CaptureFixture[str], files: Path) -> list[str]:
    assert risk.main([str(files)]) == 0
    return capsys.readouterr().out.splitlines()


def test_every_row_of_the_tables_names_a_tracked_file(risk: ModuleType) -> None:
    tracked = subprocess.run(["git", "-C", str(_ROOT), "ls-files"], capture_output=True, text=True, check=True)
    paths = tracked.stdout.splitlines()

    stale = [rule.glob for rule in risk.load_rules(risk.TIERS) if not fnmatch.filter(paths, rule.glob)]

    assert stale == []


def test_each_path_takes_the_risk_tiers_md_gives_it(
    risk: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    expected = {
        "src/hyperloom/orchestrator/loop/coordinator.py": "high",
        ".github/workflows/lint.yml": "high",
        ".claude/skills/review-pr/SKILL.md": "high",
        "docs/compatibility.rst": "high",
        "src/hyperloom/orchestrator/actions/executors/baseline.py": "medium",
        "src/hyperloom/orchestrator/phases/prelude.py": "medium",
        "src/hyperloom/orchestrator/prompts/explore.md": "medium",
        "docs/reference/operations.md": "medium",
        "scripts/code_metrics_baseline.txt": "medium",
        "examples/hyperloom-qwen3-8b-3h/SKILL.md": "medium",
        "docs/conceptual/optimization-loop.md": "low",
        "README.md": "low",
        "src/kernelforge/cli/tests/test_parse.py": "low",
    }

    printed = _printed(risk, capsys, _files(tmp_path, *expected))

    graded = dict(line.split(" -- ")[0].split(" ", 1)[::-1] for line in printed[1:])
    assert graded == expected


@pytest.mark.parametrize(
    ("paths", "level"),
    [
        pytest.param(("README.md", "src/hyperloom/orchestrator/loop/coordinator.py"), "high", id="high-wins"),
        pytest.param(("docs/guide.md", "tests/test_mod.py"), "low", id="docs-and-tests"),
        pytest.param(("tools/unlisted.py",), "medium", id="unlisted-is-medium"),
    ],
)
def test_the_pr_takes_the_risk_of_its_riskiest_path(
    risk: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, paths: tuple[str, ...], level: str
) -> None:
    assert _printed(risk, capsys, _files(tmp_path, *paths))[0] == f"risk: {level}"


def test_no_changed_path_or_no_table_row_stops_the_grade(risk: ModuleType, tmp_path: Path) -> None:
    tiers = tmp_path / "tiers.md"
    tiers.write_text("# no tables\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        risk.main([str(_files(tmp_path))])
    with pytest.raises(SystemExit):
        risk.main([str(_files(tmp_path, "README.md")), "--tiers", str(tiers)])
