# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner of a sandbox proves which rule pack it read, and a result of a pack the session did not read is refused.

The shipped runner flow runs in a local sandbox, which starts from an emptied environment; only the template names the
pack, as the toolbox's template does for the pack its session reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.common.outside_facts.rule_pack import DEFAULT_PACK_PATH, PACK_ENV
from hyperloom.measure.returncodes import HOSTED_RESULT_REJECTED_RC
from hyperloom.measure.tests.hosted_fixtures import Rig, make_call
from hyperloom.measure.tests.rule_pack_fixtures import identity_of, other_pack
from hyperloom.measure.tests.test_hosted_real_runner import REAL_RUNNER, _point, _rigs

REFERENCE = identity_of(DEFAULT_PACK_PATH)


@pytest.fixture
def real(tmp_path: Path):
    yield from _rigs(tmp_path, REAL_RUNNER)


def _recorded(rig: Rig) -> dict:
    """The rule pack the archived, signed manifest says the runner read."""
    manifest = json.loads((Path(rig.record_archive("req-1")) / "manifest.json").read_text(encoding="utf-8"))
    return manifest["runner"]["rule_pack"]


def _run(rig: Rig, tmp_path: Path):
    call = make_call(tmp_path, point=_point(tmp_path), timeout_sec=120)
    return call, rig.backend.run_test(call)


def test_a_runner_reads_the_pack_its_template_names_and_records_its_identity(real, tmp_path: Path) -> None:
    named = identity_of(other_pack(tmp_path))
    assert named != REFERENCE
    rig = real(envs={PACK_ENV: str(tmp_path / "other-pack.yaml")}, config={"rule_pack_digest": named["digest"]})
    _, outcome = _run(rig, tmp_path)
    assert outcome.returncode == 0, outcome.stderr
    assert rig.backend.journal.latest("req-1").extra["verified"] is True
    assert _recorded(rig) == named


def test_a_session_that_names_no_pack_runs_the_reference_pack_on_both_sides(real, tmp_path: Path) -> None:
    rig = real(config={"rule_pack_digest": REFERENCE["digest"]})
    _, outcome = _run(rig, tmp_path)
    assert outcome.returncode == 0, outcome.stderr
    assert rig.backend.journal.latest("req-1").extra["verified"] is True
    assert _recorded(rig) == REFERENCE


def test_a_runner_that_did_not_get_the_sessions_pack_is_refused_not_measured_with_another(real, tmp_path: Path) -> None:
    """The failure this guards: the session names a pack, the sandbox's environment does not, the runner reads the
    reference one and every classification of the test follows other facts than the manifest of the session records."""
    named = identity_of(other_pack(tmp_path))
    rig = real(config={"rule_pack_digest": named["digest"]})
    call, outcome = _run(rig, tmp_path)
    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert outcome.failure_class is None and outcome.failure_reason == "result_rejected"
    assert "(rule_pack_mismatch)" in outcome.stderr and named["digest"] in outcome.stderr, outcome.stderr
    assert REFERENCE["digest"] in outcome.stderr
    record = rig.backend.journal.latest("req-1")
    assert record.extra["verdict"] == "result_rejected" and record.extra["reject"] == "rule_pack_mismatch"
    assert "verified" not in record.extra
    assert not call.output_dir.exists() or not any(call.output_dir.iterdir()), "nothing landed"
    assert not (tmp_path / "archive").exists() or not any((tmp_path / "archive").iterdir()), "nothing was archived"


def test_a_backend_that_names_no_pack_digest_accepts_the_pack_the_runner_read(real, tmp_path: Path) -> None:
    named = identity_of(other_pack(tmp_path))
    rig = real(envs={PACK_ENV: str(tmp_path / "other-pack.yaml")})
    _, outcome = _run(rig, tmp_path)
    assert outcome.returncode == 0, outcome.stderr
    assert _recorded(rig) == named
