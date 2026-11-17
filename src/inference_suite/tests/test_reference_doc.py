# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator reference ``docs/reference/inference-suite.md`` names the contract the package ships, and only that.

Operators read the closed lists there -- run statuses, failed steps, lifecycle modes, run_dir files, CLI commands
and their flags -- without reading the code, so a value added to or removed from one of them fails here.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import pytest

from inference_suite import cli
from inference_suite.contracts import (
    BASE_LAUNCH_SCHEMA,
    CONFIG_SCHEMA,
    REPORT_SCHEMA,
    FailedStep,
    LifecycleMode,
    RunLayout,
    RunStatus,
)
from inference_suite.contracts.launch import LAUNCH_SCHEMA
from inference_suite.contracts.layout import HOST_DERIVED

DOC = Path(__file__).resolve().parents[3] / "docs" / "reference" / "inference-suite.md"
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_CLI_ROW = re.compile(r"^\| `(run|stop|resolve) ([^`]*)` \|", re.MULTILINE)
_FLAG = re.compile(r"--[a-z][a-z-]*")


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _code_spans(text: str) -> set[str]:
    return set(_CODE_SPAN.findall(text))


@pytest.mark.parametrize("vocabulary", [RunStatus, FailedStep, LifecycleMode])
def test_every_value_of_a_closed_vocabulary_is_documented(doc: str, vocabulary: type[Enum]) -> None:
    documented = _code_spans(doc)
    assert {member.value for member in vocabulary} <= documented


def test_the_failed_step_table_lists_exactly_the_closed_list(doc: str) -> None:
    section = doc.split("`failed_step`:", 1)[1]
    table = "\n".join(line for line in section.splitlines() if line.startswith("| ") and "Stage" not in line)
    assert set(re.findall(r"`([a-z_]+)`", table)) == {step.value for step in FailedStep}


def test_every_schema_id_is_documented(doc: str) -> None:
    for schema in (BASE_LAUNCH_SCHEMA, CONFIG_SCHEMA, LAUNCH_SCHEMA, REPORT_SCHEMA):
        assert f"`{schema}`" in doc


def test_every_run_dir_file_is_documented(doc: str) -> None:
    layout = RunLayout(Path("/run"))
    paths = [getattr(layout, name) for name, member in vars(RunLayout).items() if isinstance(member, property)]
    paths.extend(layout.profile_trace_dirs())
    missing = {path.name for path in paths if path.name not in doc}
    assert not missing


def test_host_derived_files_are_documented_outside_the_run_dir(doc: str) -> None:
    run_dir_table = doc.split("## Run directory", 1)[1].split("## report.json", 1)[0]
    for name in HOST_DERIVED:
        assert f"`{name}` from `report.json` into its own round" in run_dir_table


def test_the_cli_table_names_every_command_with_its_flags(doc: str, capsys: pytest.CaptureFixture[str]) -> None:
    rows = dict(_CLI_ROW.findall(doc))
    assert set(rows) == {"run", "stop", "resolve"}
    for command, usage in rows.items():
        with pytest.raises(SystemExit) as exited:
            cli.main([command, "--help"])
        assert exited.value.code == cli.EXIT_OK
        accepted = set(_FLAG.findall(capsys.readouterr().out)) - {"--help"}
        assert set(_FLAG.findall(usage)) == accepted, command


def test_the_documented_exit_codes_are_the_cli_s(doc: str) -> None:
    prose = " ".join(doc.split())
    for code in (cli.EXIT_OK, cli.EXIT_NOT_COMPLETED, cli.EXIT_REFUSED):
        assert f"`{code}` when" in prose
