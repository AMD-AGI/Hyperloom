# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for sharded tests and their coverage gates."""

from __future__ import annotations

from collections import Counter
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/tests-coverage.yml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def step(job: str, name: str) -> dict:
    return next(item for item in WORKFLOW["jobs"][job]["steps"] if item.get("name") == name)


def test_test_extra_includes_ci_regression_dependencies():
    extras = CONFIG["project"]["optional-dependencies"]
    assert "hyperloom-inference_optimizer[ci]" in extras["test"]
    assert all("[test]" not in dependency for dependency in extras["ci"])


def test_matrix_matches_configured_shards_and_workers():
    total = CONFIG["tool"]["hyperloom"]["tests_coverage"]["total_shards"]
    assert total == 6
    matrix = WORKFLOW["jobs"]["test"]["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.10", "3.11"]
    assert matrix["shard"] == list(range(1, total + 1))
    assert WORKFLOW["jobs"]["test"]["name"].endswith(f"/{total})")
    assert CONFIG["tool"]["hyperloom"]["tests_coverage"]["xdist_workers"] == 2
    assert WORKFLOW["jobs"]["test"]["strategy"]["fail-fast"] is False
    assert "max-parallel" not in WORKFLOW["jobs"]["test"]["strategy"]


def test_matrix_uses_one_resolved_duration_seed():
    prepare = WORKFLOW["jobs"]["prepare-durations"]
    resolve = step("prepare-durations", "Resolve test duration seed")
    select = step("prepare-durations", "Publish selected duration seed")
    test_job = WORKFLOW["jobs"]["test"]
    restore = step("test", "Restore selected test durations")
    require = step("test", "Require selected test durations")

    assert prepare["outputs"] == {"key": "${{ steps.select.outputs.key }}"}
    assert resolve["with"]["lookup-only"] is True
    assert resolve["with"]["restore-keys"].strip() == "test-durations-v1-"
    assert "cache-matched-key" in select["env"]["MATCHED_KEY"]
    assert test_job["needs"] == "prepare-durations"
    assert restore["if"] == "needs.prepare-durations.outputs.key != 'cold-cache'"
    assert restore["with"]["key"] == "${{ needs.prepare-durations.outputs.key }}"
    assert "restore-keys" not in restore["with"]
    assert "cache-hit" in require["env"]["EXACT_HIT"]
    assert require["env"]["SEED_KEY"] == "${{ needs.prepare-durations.outputs.key }}"
    assert "Re-run all jobs" in require["run"]

    rolling_restores = [
        item
        for job in WORKFLOW["jobs"].values()
        for item in job.get("steps", [])
        if item.get("uses") == "actions/cache/restore@v6" and "restore-keys" in item.get("with", {})
    ]
    assert rolling_restores == [resolve]


@pytest.mark.parametrize(
    ("key", "hit", "expected"),
    [
        ("cold-cache", "", 0),
        ("test-durations-v1-123", "true", 0),
        ("test-durations-v1-123", "false", 1),
        ("test-durations-v1-123", "", 1),
        ("", "", 1),
    ],
)
def test_selected_seed_requires_exact_hit(key, hit, expected, tmp_path):
    command = step("test", "Require selected test durations")["run"]
    env = dict(os.environ, SEED_KEY=key, EXACT_HIT=hit)
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
    if expected:
        assert "Re-run all jobs" in result.stdout


@pytest.mark.parametrize("matched", ["", "test-durations-v1-123"])
def test_seed_selection_records_cold_miss_separately(matched, tmp_path):
    output = tmp_path / "output"
    command = step("prepare-durations", "Publish selected duration seed")["run"]
    env = dict(os.environ, MATCHED_KEY=matched, GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text() == f"key={matched or 'cold-cache'}\n"


def test_pinned_seed_prevents_mixed_generation_partition_gaps():
    from pytest_split.algorithms import LeastDurationAlgorithm

    items = [SimpleNamespace(nodeid=name) for name in "ABCDEFGHIJKL"]
    old_seed = {item.nodeid: 12 - index for index, item in enumerate(items)}
    new_seed = dict(old_seed, A=11, B=12)
    split = LeastDurationAlgorithm()
    old_groups = split(6, items, old_seed)
    new_groups = split(6, items, new_seed)
    mixed = Counter(item.nodeid for group in [old_groups[0], *new_groups[1:]] for item in group.selected)
    assert mixed["A"] == 2 and mixed["B"] == 0
    assert sum(mixed.values()) == len(items)
    for groups in (old_groups, new_groups):
        actual = Counter(item.nodeid for group in groups for item in group.selected)
        assert actual == Counter(item.nodeid for item in items)


def test_all_six_shards_select_the_complete_test_collection(tmp_path):
    import json

    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    source = "\n".join(f"def test_{i:02d}():\n    assert True\n" for i in range(24))
    (tmp_path / "test_cases.py").write_text(source, encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\n"
        "def pytest_collection_finish(session):\n"
        "    Path(os.environ['SELECTED']).write_text(json.dumps([i.nodeid for i in session.items]))\n",
        encoding="utf-8",
    )
    expected = {f"test_cases.py::test_{i:02d}" for i in range(24)}
    seed = {nodeid: i + 1 for i, nodeid in enumerate(sorted(expected))}
    (tmp_path / ".test_durations").write_text(json.dumps(seed), encoding="utf-8")
    selected = tmp_path / "selected.json"
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", SELECTED=str(selected), PYTHONPATH=str(ROOT))
    env.pop("PYTEST_ADDOPTS", None)
    seen = set()
    total = CONFIG["tool"]["hyperloom"]["tests_coverage"]["total_shards"]
    for group in range(1, total + 1):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "pytest_split.plugin",
                "--collect-only",
                "-q",
                "--splits",
                str(total),
                "--group",
                str(group),
                "--splitting-algorithm=least_duration",
                "--durations-path",
                ".test_durations",
                "test_cases.py",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        nodes = json.loads(selected.read_text(encoding="utf-8"))
        assert nodes and len(nodes) == len(set(nodes))
        assert not seen.intersection(nodes)
        seen.update(nodes)
    assert seen == expected


def test_shards_measure_without_generating_reports():
    args = CONFIG["tool"]["hyperloom"]["tests_coverage"]["pytest_ci_args"]
    assert "--cov=." in args
    assert [arg for arg in args if arg.startswith("--cov-report")] == ["--cov-report="]
    run = next(item for item in WORKFLOW["jobs"]["test"]["steps"] if item.get("id") == "pytest")
    assert "--cov-fail-under=0" in run["run"]
    assert "-rfE" in run["run"]
    assert "--junitxml=" in run["run"]
    assert "--no-cov" not in run["run"]


@pytest.mark.parametrize("report", ["", "term"])
def test_report_mode_keeps_parallel_coverage_data(tmp_path, report):
    from coverage import CoverageData

    (tmp_path / "sample.py").write_text("def value():\n    return 42\n", encoding="utf-8")
    (tmp_path / "test_sample.py").write_text(
        "from sample import value\n\ndef test_value():\n    assert value() == 42\n", encoding="utf-8"
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    data_file = tmp_path / ".coverage.shard1"
    env = dict(
        os.environ,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        COVERAGE_FILE=str(data_file),
        PYTHONPATH=str(tmp_path),
    )
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(config),
            "-p",
            "pytest_cov.plugin",
            "-p",
            "xdist.plugin",
            "-n",
            "2",
            "--cov=sample",
            f"--cov-report={report}",
            "--cov-fail-under=0",
            "-q",
            "test_sample.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert data_file.is_file()
    data = CoverageData(basename=str(data_file))
    data.read()
    files = list(data.measured_files())
    assert len(files) == 1
    assert set(data.lines(files[0])) == {1, 2}
    assert ("TOTAL" in result.stdout) is bool(report)


def test_coverage_installs_only_reporting_dependencies():
    install = step("coverage", "Install dependencies")["run"]
    assert "coverage[toml]" in install
    assert ".[test,ci]" not in install
    assert "./OOB" not in install
    assert "-e " not in install


def test_summary_and_gate_reuse_one_report():
    summary = step("coverage", "Coverage summary")
    gate = step("coverage", "Enforce coverage fail_under (strict mode)")
    assert summary["id"] == "coverage_report"
    assert summary["if"] == "always()"
    assert "scripts/ci_coverage_report.py" in summary["run"]
    assert "scripts/ci_coverage_report.py gate" in gate["run"]
    assert gate["env"]["REPORT_EXIT_CODE"] == "${{ steps.coverage_report.outputs.report_exit_code }}"
    assert "complete == 'true'" in gate["if"]
    assert "tests_ok == 'true'" in gate["if"]
    assert "${{ vars.COVERAGE_RELAX_FAIL_UNDER }}" == gate["env"]["COVERAGE_RELAX_FAIL_UNDER"]
    assert "coverage report" not in gate["run"]


def test_failed_shards_and_missing_data_still_fail():
    gate = step("coverage", "Enforce shard results")
    assert "always()" in gate["if"]
    assert "needs.test.result != 'success'" in gate["if"]
    assert "complete != 'true'" in gate["if"]
    assert "tests_ok != 'true'" in gate["if"]
    assert gate["env"]["UPSTREAM_RESULT"] == "${{ needs.test.result }}"
    assert "test-shards-unavailable" in gate["run"]
    assert "exit 1" in gate["run"]
    assert CONFIG["tool"]["coverage"]["report"]["fail_under"] == 90


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
@pytest.mark.parametrize("failure", [None, "coverage", "outcome", "test"])
def test_aggregation_requires_every_configured_shard(tmp_path, python_version, failure):
    total = CONFIG["tool"]["hyperloom"]["tests_coverage"]["total_shards"]
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8")
    statuses = tmp_path / "shard-status"
    statuses.mkdir()
    for shard in range(1, total + 1):
        if not (shard == total and failure == "coverage"):
            (tmp_path / f".coverage.shard{shard}").touch()
        if not (shard == total and failure == "outcome"):
            outcome = "failure" if shard == total and failure == "test" else "success"
            (statuses / f"py{python_version}-shard{shard}.outcome").write_text(outcome, encoding="utf-8")
    command = step("coverage", "Aggregate shard results")["run"]
    script = command.split("python3 <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    script = script.replace("${{ matrix.python-version }}", python_version)
    output = tmp_path / "outputs"
    summary = tmp_path / "summary"
    env = dict(os.environ, GITHUB_OUTPUT=str(output), GITHUB_STEP_SUMMARY=str(summary), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert values["complete"] == ("false" if failure in {"coverage", "outcome"} else "true")
    assert values["tests_ok"] == ("false" if failure == "test" else "true")
    assert values["missing_shards"] == (f"shard{total}" if failure in {"coverage", "outcome"} else "")
    assert values["failed_shards"] == (f"shard{total}" if failure == "test" else "")
    assert f"shard{total}" in summary.read_text(encoding="utf-8")
