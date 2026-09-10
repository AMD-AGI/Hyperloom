# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression coverage for CI timing collection and publication."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "tests-coverage.yml"
_HELPER = _ROOT / "scripts" / "ci_durations.py"


def _run_shard(directory: Path, seed: dict[str, float] | None, *, workers: int = 2, group: int = 1):
    directory.mkdir(parents=True, exist_ok=True)
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    if seed is not None:
        (directory / ".test_durations").write_text(json.dumps(seed), encoding="utf-8")
        if 'cp .test_durations ".test_durations.shard${SHARD}"' in workflow:
            shutil.copyfile(directory / ".test_durations", directory / f".test_durations.shard{group}")
    duration_args = re.search(r"^\s+(--store-durations.*?)--junitxml", workflow, re.DOTALL | re.MULTILINE)
    assert duration_args
    args = shlex.split(duration_args.group(1).replace("\\\n", "").replace("${SHARD}", str(group)))
    plugins = ["-p", "pytest_split.plugin", "-p", "xdist.plugin"]
    if "-p scripts.ci_durations" in workflow:
        plugins += ["-p", "scripts.ci_durations"]
    env = {key: value for key, value in os.environ.items() if not key.startswith(("COVERAGE", "COV_CORE", "PYTEST_"))}
    env.update(PYTHONPATH=str(_ROOT), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    command = [sys.executable, "-m", "pytest", *plugins, "-n", str(workers), "--splits", "2", "--group", str(group)]
    if seed is not None:
        command += ["--splitting-algorithm=least_duration"]
    return subprocess.run(
        [*command, *args, "--junitxml=results.xml", "-rfE", "--color=no"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def _suite(directory: Path, *, failure: bool = False, count: int = 8) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    nodeids = [f"test_sample.py::test_{index:02}" for index in range(count)]
    source = "\n\n".join(
        f"def test_{index:02}():\n    assert {not (failure and index == 0)!r}" for index in range(count)
    )
    (directory / "test_sample.py").write_text(source, encoding="utf-8")
    return nodeids


@pytest.mark.parametrize("workers", [0, 2])
def test_seed_selects_unequal_shards_and_only_current_timings_are_written(tmp_path: Path, workers: int):
    nodeids = _suite(tmp_path)
    seed = dict.fromkeys(nodeids, 1.0) | {nodeids[0]: 100.0, "deleted.py::test_old": 999.0}
    for group, expected in ((1, {nodeids[0]}), (2, set(nodeids[1:]))):
        result = _run_shard(tmp_path, seed, workers=workers, group=group)
        assert result.returncode == 0, result.stdout + result.stderr
        timings = json.loads((tmp_path / f".test_durations.shard{group}").read_text())
        assert set(timings) == expected
        assert all(0 <= value < 100 for value in timings.values())
        assert json.loads((tmp_path / ".test_durations").read_text()) == seed
        assert (tmp_path / f".test_durations.shard{group}.complete").exists()


def test_cold_cache_records_disjoint_complete_shards(tmp_path: Path):
    nodeids = _suite(tmp_path)
    shards = []
    for group in (1, 2):
        result = _run_shard(tmp_path, None, group=group)
        assert result.returncode == 0, result.stdout + result.stderr
        shards.append(set(json.loads((tmp_path / f".test_durations.shard{group}").read_text())))
    assert not shards[0] & shards[1]
    assert shards[0] | shards[1] == set(nodeids)
    assert not (tmp_path / ".test_durations").exists()


def test_failed_tests_keep_exit_code_summary_and_junit(tmp_path: Path):
    nodeids = _suite(tmp_path, failure=True)
    seed = dict.fromkeys(nodeids, 1.0) | {nodeids[0]: 100.0, "deleted.py::test_old": 999.0}
    result = _run_shard(tmp_path, seed)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAILED {nodeids[0]}" in result.stdout
    assert "<failure" in (tmp_path / "results.xml").read_text()
    assert set(json.loads((tmp_path / ".test_durations.shard1").read_text())) == {nodeids[0]}
    assert (tmp_path / ".test_durations.shard1.complete").exists()


@pytest.mark.parametrize(("workers", "exit_code"), [(0, 2), (2, 1)])
def test_collection_error_does_not_publish_stale_timings(tmp_path: Path, workers: int, exit_code: int):
    (tmp_path / "test_sample.py").write_text("raise RuntimeError('collection sentinel')\n", encoding="utf-8")
    result = _run_shard(tmp_path, {"deleted.py::test_old": 10.0}, workers=workers)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert "ERROR test_sample.py" in result.stdout
    assert "collection sentinel" in (tmp_path / "results.xml").read_text()
    assert json.loads((tmp_path / ".test_durations.shard1").read_text()) == {}
    assert (tmp_path / ".test_durations.shard1.complete").exists()


def test_startup_failure_cannot_upload_seed_as_fresh_timings(tmp_path: Path):
    seed = {"deleted.py::test_old": 10.0}
    marker = tmp_path / ".test_durations.shard1.complete"
    marker.touch()
    (tmp_path / "conftest.py").write_text(
        "import pytest\ndef pytest_configure(config):\n    raise pytest.UsageError('startup sentinel')\n",
        encoding="utf-8",
    )
    result = _run_shard(tmp_path, seed)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "startup sentinel" in result.stdout + result.stderr
    assert json.loads((tmp_path / ".test_durations.shard1").read_text()) == seed
    assert not marker.exists()


def test_restarted_worker_reads_original_seed_after_other_worker_finishes(tmp_path: Path):
    nodeids = _suite(tmp_path, count=12)
    seed = {nodeid: 5.0 if index % 2 == 0 else 1.0 for index, nodeid in enumerate(nodeids)}
    seed["deleted.py::test_old"] = 999.0
    (tmp_path / ".test_durations.shard1").write_text(json.dumps(seed), encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import json\nimport os\nimport time\nfrom pathlib import Path\nimport pytest\n"
        "def pytest_collection_finish(session):\n"
        "    worker = session.config.workerinput['workerid']\n"
        "    Path(f'{worker}.collection').write_text(json.dumps([item.nodeid for item in session.items]))\n"
        "    Path(f'{worker}.seed').write_bytes(Path('.test_durations.shard1').read_bytes())\n"
        "@pytest.hookimpl(trylast=True)\n"
        "def pytest_sessionfinish(session):\n"
        "    if hasattr(session.config, 'workerinput'):\n"
        "        Path(session.config.workerinput['workerid'] + '.finished').touch()\n"
        "@pytest.fixture(autouse=True)\n"
        "def crash_once(request):\n"
        "    if request.config.workerinput['workerid'] == 'gw1':\n"
        "        deadline = time.monotonic() + 10\n"
        "        while not Path('gw0.finished').exists() and time.monotonic() < deadline:\n"
        "            time.sleep(0.02)\n"
        "        assert Path('gw0.finished').exists(), 'other worker did not finish'\n"
        "        os._exit(7)\n",
        encoding="utf-8",
    )
    result = _run_shard(tmp_path, seed)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "replacing crashed worker" in result.stdout
    assert "Different tests were collected" not in result.stdout
    assert (tmp_path / "gw0.finished").exists()
    assert json.loads((tmp_path / "gw2.seed").read_text()) == seed
    collections = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("gw*.collection"))]
    assert len(collections) == 3
    assert collections[0] == collections[1] == collections[2]
    assert set(json.loads((tmp_path / ".test_durations.shard1").read_text())) == set(collections[0])
    assert "deleted.py::test_old" not in json.loads((tmp_path / ".test_durations.shard1").read_text())


def _merge(directory: Path):
    return subprocess.run(
        [
            sys.executable,
            str(_HELPER),
            "--config",
            "pyproject.toml",
            "--artifacts",
            "durations",
            "--output",
            ".test_durations",
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _artifacts(directory: Path) -> tuple[Path, Path]:
    (directory / "pyproject.toml").write_text("[tool.hyperloom.tests_coverage]\ntotal_shards = 2\n", encoding="utf-8")
    paths = tuple(
        directory / "durations" / f"durations-shard{group}" / f".test_durations.shard{group}" for group in (1, 2)
    )
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({f"test_sample.py::test_{index}": float(index)}), encoding="utf-8")
    return paths


def test_merge_publishes_complete_disjoint_timings(tmp_path: Path):
    _artifacts(tmp_path)
    result = _merge(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / ".test_durations").read_text()) == {
        "test_sample.py::test_0": 0.0,
        "test_sample.py::test_1": 1.0,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        "{",
        "{}",
        "[]",
        "null",
        '{"": 1}',
        '{"test::a": true}',
        '{"test::a": -1}',
        '{"test::a": "1"}',
        '{"test::a": null}',
        '{"test::a": NaN}',
        '{"test::a": Infinity}',
        '{"test::a": 1e999}',
        '{"test::a": 1, "test::a": 2}',
        '{"test_sample.py::test_0": 1}',
    ],
)
def test_merge_rejects_invalid_or_duplicate_timings_without_overwriting(tmp_path: Path, invalid: str):
    _, second = _artifacts(tmp_path)
    second.write_text(invalid, encoding="utf-8")
    output = tmp_path / ".test_durations"
    output.write_text("original", encoding="utf-8")
    result = _merge(tmp_path)
    assert result.returncode != 0
    assert "::error::" in result.stdout + result.stderr
    assert output.read_text() == "original"


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "wrong-name", "empty", "unreadable"])
def test_merge_requires_exact_expected_artifacts(tmp_path: Path, problem: str):
    first, second = _artifacts(tmp_path)
    if problem == "missing":
        second.unlink()
    elif problem == "extra":
        extra = first.parent.parent / "durations-shard3"
        extra.mkdir()
        (extra / ".test_durations.shard3").write_text('{"test::extra": 1}')
    elif problem == "duplicate":
        (second.parent / first.name).write_bytes(first.read_bytes())
    elif problem == "wrong-name":
        first.rename(first.with_name(".test_durations.shard01"))
    elif problem == "empty":
        first.write_text("")
    else:
        first.unlink()
        first.mkdir()
    result = _merge(tmp_path)
    assert result.returncode != 0
    assert "::error::" in result.stdout + result.stderr
    assert not (tmp_path / ".test_durations").exists()


def test_workflow_keeps_pytest_exit_diagnostics_and_fail_closed_publication():
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert 'cp .test_durations ".test_durations.shard${SHARD}"' in workflow
    assert "python -m pytest -p scripts.ci_durations" in workflow
    assert "--store-durations --clean-durations" in workflow
    assert 'PYTEST_RC="${PIPESTATUS[0]}"' in workflow
    assert 'exit "$PYTEST_RC"' in workflow
    assert '-rfE "${EXTRA[@]}" 2>&1 | tee pytest-output.log' in workflow
    update = workflow.split("  update-durations:", 1)[1]
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main' && success()" in update
    assert "needs: [test, coverage]" in update
    assert "merge-multiple: false" in update
    assert "python scripts/ci_durations.py" in update
    assert "if: steps.merge.outcome == 'success'" in update
    assert "test-durations-v1-" in update
