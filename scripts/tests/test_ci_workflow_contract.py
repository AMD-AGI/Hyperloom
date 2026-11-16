# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for sharded tests and their coverage gates."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def test_the_matrix_lists_every_configured_shard():
    total = CONFIG["tool"]["hyperloom"]["tests_coverage"]["total_shards"]
    assert WORKFLOW["jobs"]["test"]["strategy"]["matrix"]["shard"] == list(range(1, total + 1))


@pytest.mark.parametrize(
    ("upload_name", "artifact_name"),
    [
        ("Upload shard coverage data", "covdata-py${{ matrix.python-version }}-shard${{ matrix.shard }}"),
        ("Upload shard durations", "durations-shard${{ matrix.shard }}"),
    ],
)
def test_reruns_replace_only_their_own_shard_artifact(upload_name, artifact_name):
    upload = step("test", upload_name)
    assert upload["with"]["name"] == artifact_name
    assert upload["with"].get("overwrite") is True


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


def test_a_failing_shard_exits_with_pytests_code_through_tee(tmp_path):
    command = next(item for item in WORKFLOW["jobs"]["test"]["steps"] if item.get("id") == "pytest")["run"]
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text("#!/bin/sh\necho 'FAILED test_x.py::test_y'\nexit 3\n", encoding="utf-8")
    (bin_dir / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    for shim in bin_dir.iterdir():
        shim.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", SHARD="1")
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "FAILED test_x.py::test_y" in (tmp_path / "pytest-output.log").read_text(encoding="utf-8")


def test_failed_shards_and_missing_data_still_fail():
    gate = step("coverage", "Enforce shard results")
    assert "always()" in gate["if"]
    assert "needs.test.result != 'success'" in gate["if"]
    assert "complete != 'true'" in gate["if"]
    assert "tests_ok != 'true'" in gate["if"]
    assert "exit 1" in gate["run"]


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
