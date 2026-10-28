# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A runner that exits without writing ``status.json`` ends its test at once as ``runner_exited``, an infrastructure
failure, rather than leaving the host to poll until the test's timeout."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.failures import RUNNER_EXITED, Disposition, classify_test_failure
from hyperloom.measure.hosted_log import FAILURE_DETAIL_CHARS
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC
from hyperloom.measure.tests.hosted_fixtures import make_call, make_rig
from hyperloom.sandbox.templates import LocalTemplate

#: What the runner did on an image whose ROCm libraries were not in the loader's path: it died importing torch.
_DIES_AT_IMPORT = "echo \"KeyError: 'libamd_smi.so'\" >&2; exit 3"
#: A runner that reports its failure and then exits: the report wins over the exit.
_REPORTS_THEN_EXITS = (
    'printf \'{"state": "failed", "error_class": "no_spec", "message": "no sheet"}\' > "$HYPERLOOM_SANDBOX_ROOT/out/s";'
    ' mv "$HYPERLOOM_SANDBOX_ROOT/out/s" "$HYPERLOOM_SANDBOX_ROOT/out/status.json"'
)
#: What the runner's start step said in joint run 15 when the cache entry did not verify (the sandbox then went away).
_VERIFY_MESSAGE = (
    "bootstrap.sh or manifest of the cache entry is missing or not the one HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256 pins"
)
_TAIL = "z" * (FAILURE_DETAIL_CHARS + 500)  # a runaway message: only its beginning is kept
_START_FAILED = (
    f'printf \'{{"state": "failed", "error_class": "runner_start_failed", "message": "{_VERIFY_MESSAGE} {_TAIL}"}}\''
    ' > "$HYPERLOOM_SANDBOX_ROOT/out/s"; mv "$HYPERLOOM_SANDBOX_ROOT/out/s" "$HYPERLOOM_SANDBOX_ROOT/out/status.json";'
    " exit 1"
)
#: A runner that leaves a ``status.json`` nobody can parse, then exits.
_UNREADABLE_STATUS = 'printf "not json" > "$HYPERLOOM_SANDBOX_ROOT/out/status.json"; exit 1'


def _rig(tmp_path: Path, script: str):
    template = LocalTemplate("runner-1", "measurement", command=("/bin/sh", "-c", script), sealed=True, signing=True)
    return make_rig(tmp_path, templates=[template])


@pytest.fixture
def rigs(tmp_path: Path):
    made = []

    def make(script: str):
        made.append(_rig(tmp_path, script))
        return made[-1]

    yield make
    for rig in made:
        rig.backend.close()
        rig.provider.close()


def test_runner_exited_is_an_infrastructure_failure() -> None:
    verdict = classify_test_failure(returncode=0, error_class=RUNNER_EXITED)
    assert (verdict.disposition, verdict.failure_class) == (Disposition.FAILED, FailureClass.INFRA)


def test_a_runner_dead_at_import_ends_the_test_at_once_naming_its_exit(rigs, tmp_path: Path) -> None:
    rig = rigs(_DIES_AT_IMPORT)
    started = time.monotonic()
    outcome = rig.backend.run_test(make_call(tmp_path, timeout_sec=30))
    assert time.monotonic() - started < 15, "the host stopped waiting long before the test's timeout"
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC and outcome.failure_class is FailureClass.INFRA
    assert outcome.failure_reason == RUNNER_EXITED
    place = f"sandbox {outcome.sandbox.sandbox_id} on node {outcome.sandbox.node_name}"
    said = f"{RUNNER_EXITED}: the runner exited with code 3 without writing status.json"
    assert outcome.stderr == f"{said} ({place}; status_json missing)"
    line = rig.backend.journal.records()[-1]
    assert line.extra["runner_status"] == {"status_json": "missing"}, "a missing status.json is recorded as missing"
    assert line.extra["detail"] == outcome.stderr and line.extra["node"] == outcome.sandbox.node_name
    assert rig.provider.find({}, role="measurement") == [], "the dead runner's sandbox is closed"


def test_a_runner_that_wrote_its_status_before_exiting_is_judged_by_that_status(rigs, tmp_path: Path) -> None:
    outcome = rigs(_REPORTS_THEN_EXITS).backend.run_test(make_call(tmp_path, timeout_sec=30))
    assert outcome.failure_reason == "no_spec"
    assert outcome.stderr.startswith("no_spec: no sheet")


def test_the_start_steps_own_message_outlives_the_sandbox_in_the_outcome_and_the_journal(rigs, tmp_path: Path) -> None:
    """Joint run 15: three runners failed their start step in seconds and only ``runner_start_failed`` was kept."""
    rig = rigs(_START_FAILED)
    outcome = rig.backend.run_test(make_call(tmp_path, timeout_sec=30))
    kept = f"{_VERIFY_MESSAGE} {_TAIL}"[:FAILURE_DETAIL_CHARS]
    sandbox, node = outcome.sandbox.sandbox_id, outcome.sandbox.node_name
    assert outcome.failure_class is FailureClass.INFRA and outcome.failure_reason == "runner_start_failed"
    assert outcome.stderr == f"runner_start_failed: {kept} (sandbox {sandbox} on node {node})"
    line = rig.backend.journal.records()[-1]
    assert (line.sandbox_id, line.extra["node"]) == (sandbox, node) and node
    assert line.extra["runner_status"] == {
        "status_json": "read",
        "state": "failed",
        "error_class": "runner_start_failed",
        "message": kept,
    }
    assert line.extra["detail"] == outcome.stderr[:FAILURE_DETAIL_CHARS]
    assert rig.provider.find({}, role="measurement") == [], "and the sandbox, with its /out, is gone"


def test_an_unreadable_status_is_recorded_as_unreadable_not_as_nothing(rigs, tmp_path: Path) -> None:
    rig = rigs(_UNREADABLE_STATUS)
    outcome = rig.backend.run_test(make_call(tmp_path, timeout_sec=30))
    assert outcome.failure_reason == RUNNER_EXITED
    assert outcome.stderr.endswith("; status_json unreadable: JSONDecodeError)")
    assert rig.backend.journal.records()[-1].extra["runner_status"] == {"status_json": "unreadable: JSONDecodeError"}
