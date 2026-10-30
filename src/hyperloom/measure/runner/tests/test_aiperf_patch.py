# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The replay client of an AgentX test is patched to keep the output token count of a reply it found nothing in.

What is pinned: both files of the client's source change and nothing else; the change applies twice without a second
effect; a client that is not the pinned one is changed in neither file and the refusal says which; a checkout that
declares the client as a submodule gets it fetched first; and, when a checkout of the real client is given, the real
parser exports the count the server reported and the success line and every other failure are untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hyperloom.measure.runner.aiperf_patch import ReplayClientPatchError, patch_replay_client
from hyperloom.measure.runner.tests import aiperf_source as source

PARSER, METRIC = source.PARSER_FILE, source.METRIC_FILE
SOURCE_ENV = "HYPERLOOM_TEST_AIPERF_SOURCE"
PYTHON_ENV = "HYPERLOOM_TEST_AIPERF_PYTHON"
DRIVER = Path(__file__).with_name("aiperf_driver.py")
ARTIFACTS = "aiperf_artifacts"


def _both(client: Path) -> tuple[str, str]:
    return (client / PARSER).read_text(encoding="utf-8"), (client / METRIC).read_text(encoding="utf-8")


def test_the_parser_keeps_the_token_counts_of_an_invalid_record_and_a_metric_puts_the_output_one_on_its_line(
    tmp_path: Path,
) -> None:
    client = source.write_client(tmp_path)
    patch_replay_client(tmp_path)
    parser, metric = _both(client)
    assert "token_counts=record.token_counts or TokenCounts()," in parser
    assert "input=record.token_counts.input" not in parser
    assert metric.startswith(source.OUTPUT_LENGTH_METRIC), "the metric module only gains a class at its end"
    added = metric.removeprefix(source.OUTPUT_LENGTH_METRIC)
    assert "class ErrorOutputSequenceLengthMetric(OutputSequenceLengthMetric):" in added
    assert 'tag = "error_osl"' in added and "MetricFlags.ERROR_ONLY" in added
    changed = [line for line in parser.splitlines() if line not in source.PARSER.splitlines()]
    assert 3 <= len(changed) <= 6, "the parser changes in the one branch and nowhere else"


def test_the_patch_applies_once_and_a_second_time_changes_nothing(tmp_path: Path) -> None:
    client = source.write_client(tmp_path)
    patch_replay_client(tmp_path)
    once = _both(client)
    patch_replay_client(tmp_path)
    assert _both(client) == once
    assert once[1].count("class ErrorOutputSequenceLengthMetric") == 1


def test_a_file_keeps_its_mode_and_no_scratch_file_is_left(tmp_path: Path) -> None:
    client = source.write_client(tmp_path)
    (client / PARSER).chmod(0o755)
    patch_replay_client(tmp_path)
    assert (client / PARSER).stat().st_mode & 0o777 == 0o755
    assert not [path for path in client.rglob("*") if path.name.endswith(".tmp")]


def test_a_parser_that_is_not_the_pinned_one_leaves_both_files_as_they_were_and_names_the_parser(
    tmp_path: Path,
) -> None:
    moved = source.PARSER.replace("# Parsed record was invalid, return as error record", "# the branch moved")
    client = source.write_client(tmp_path, parser=moved)
    with pytest.raises(ReplayClientPatchError, match="inference_result_parser.py does not hold the branch"):
        patch_replay_client(tmp_path)
    assert _both(client) == (moved, source.OUTPUT_LENGTH_METRIC), "the metric must not be added without the parser"


def test_a_metric_module_without_the_output_length_metric_is_refused_and_the_parser_is_left_alone(
    tmp_path: Path,
) -> None:
    other = source.OUTPUT_LENGTH_METRIC.replace("class OutputSequenceLengthMetric(", "class Renamed(")
    client = source.write_client(tmp_path, metric=other)
    with pytest.raises(ReplayClientPatchError, match="output_sequence_length_metric.py does not define"):
        patch_replay_client(tmp_path)
    assert _both(client) == (source.PARSER, other)


def test_a_checkout_with_no_client_is_refused_and_says_so(tmp_path: Path) -> None:
    with pytest.raises(ReplayClientPatchError, match="has no utils/aiperf submodule"):
        patch_replay_client(tmp_path)


def test_a_declared_submodule_that_cannot_be_fetched_is_refused_with_git_s_reason(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text('[submodule "utils/aiperf"]\n\tpath = utils/aiperf\n\turl = /nowhere\n')
    with pytest.raises(ReplayClientPatchError, match=r"the utils/aiperf submodule could not be fetched \(.+\)"):
        patch_replay_client(tmp_path)


def _git(cwd: Path, *args: str) -> None:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("where", ["utils/aiperf", "inferencex-e2e/utils/aiperf"])
def test_a_checkout_that_declares_the_client_as_a_submodule_gets_it_fetched_before_it_is_patched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """The way a runner's checkout is: the commit names the client, nothing has fetched it yet. The submodule is found by
    its name, ``utils/aiperf``, wherever the checkout puts it (InferenceX moved it under ``inferencex-e2e/``)."""
    for key, value in (
        ("GIT_CONFIG_COUNT", "1"),
        ("GIT_CONFIG_KEY_0", "protocol.file.allow"),
        ("GIT_CONFIG_VALUE_0", "always"),
    ):
        monkeypatch.setenv(key, value)  # the test's submodule is a local path, which git refuses by default
    for name in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(name, "t")
    for name in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(name, "t@example.invalid")
    upstream = tmp_path / "aiperf"
    upstream.mkdir()
    source.write_client(upstream)
    for name in ("src", "pyproject.toml"):
        shutil.move(str(upstream / "utils" / "aiperf" / name), upstream / name)
    shutil.rmtree(upstream / "utils")
    _git(upstream, "init", "-q")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "aiperf")
    full = tmp_path / "full"
    full.mkdir()
    _git(full, "init", "-q")
    _git(full, "submodule", "add", "-q", "--name", "utils/aiperf", str(upstream), where)
    _git(full, "commit", "-q", "-m", "inferencex")
    runner = tmp_path / "runner"
    _git(tmp_path, "clone", "-q", str(full), str(runner))
    assert not (runner / where / "pyproject.toml").exists(), "the clone has not fetched the submodule"

    patch_replay_client(runner)

    assert "token_counts=record.token_counts or TokenCounts()," in (runner / where / PARSER).read_text()


# --- against the real client ----------------------------------------------------------------------------------------


def _drive(python: str, client: Path, artifacts: Path) -> dict[str, dict]:
    """The real parser, metric processor and export writer of ``client`` on one reply of each kind: each one's line."""
    artifacts.mkdir(parents=True)
    done = subprocess.run(
        [python, "-I", str(DRIVER), str(client), str(artifacts)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=artifacts,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = dict(line.split("\t", 1) for line in done.stdout.splitlines() if "\t" in line)
    module = Path(json.loads(lines.pop("module")))
    assert module.is_relative_to(client), "the driver ran another copy of the client"
    return {name: json.loads(raw) for name, raw in lines.items()}


needs_client = pytest.mark.skipif(
    not (os.environ.get(SOURCE_ENV) and os.environ.get(PYTHON_ENV)),
    reason=f"needs {SOURCE_ENV} (aiperf at the pinned commit) and {PYTHON_ENV} (an interpreter with its dependencies)",
)


@needs_client
def test_the_real_client_exports_the_count_the_server_reported_and_changes_nothing_else(tmp_path: Path) -> None:
    client = tmp_path / "ix" / "utils" / "aiperf"
    shutil.copytree(os.environ[SOURCE_ENV], client, ignore=shutil.ignore_patterns(".git"))
    python = os.environ[PYTHON_ENV]

    before = _drive(python, client, tmp_path / "before" / ARTIFACTS)
    patch_replay_client(tmp_path / "ix")
    after = _drive(python, client, tmp_path / "after" / ARTIFACTS)

    assert not [name for name, line in before.items() if "error_osl" in line["metrics"]], "the client as released"
    counted = {name: line["metrics"]["error_osl"] for name, line in after.items() if "error_osl" in line["metrics"]}
    assert counted == {
        "no_content_usage_zero": {"value": 0, "unit": "tokens"},
        "no_content_usage_three": {"value": 3, "unit": "tokens"},
    }, "only a reply the client found nothing in, with a count from the server, gains the field"
    for name, line in after.items():
        line["metrics"].pop("error_osl", None)
        assert line == before[name], f"{name}: the line is what it was, the added field aside"
