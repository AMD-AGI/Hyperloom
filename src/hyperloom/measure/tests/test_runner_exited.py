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
    assert outcome.stderr == f"{RUNNER_EXITED}: the runner exited with code 3 without writing status.json"
    assert rig.provider.find({}, role="measurement") == [], "the dead runner's sandbox is closed"


def test_a_runner_that_wrote_its_status_before_exiting_is_judged_by_that_status(rigs, tmp_path: Path) -> None:
    outcome = rigs(_REPORTS_THEN_EXITS).backend.run_test(make_call(tmp_path, timeout_sec=30))
    assert outcome.failure_reason == "no_spec"
    assert outcome.stderr.startswith("no_spec: no sheet")
