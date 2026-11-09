# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every row of the fake-gain catalog is refused by its real enforcement point with exactly the row's code."""

from __future__ import annotations

from pathlib import Path

from hyperloom.contracts.measurement import Purpose
from hyperloom.measure.fake_gain import FAKE_GAINS, FakeGain
from hyperloom.measure.tests.hosted_fixtures import make_rig
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, WorkloadKind
from hyperloom_multibranch.toolbox.tests.test_acceptance_fake_gain import (
    NGRAM,
    SIM_ENV,
    SIMULATING,
    _refused_with,
    _repo,
)
from hyperloom_multibranch.toolbox.tests.test_hosted_executor_cacheless import SGLANG_OFF, _run

ROWS = {row.name: row for row in FAKE_GAINS}


def test_the_catalog_has_the_three_known_rows_with_a_known_enforcement_point() -> None:
    assert set(ROWS) == {
        "context_copy_draft_on_synthetic",
        "speculative_decoding_pinned",
        "agentx_baseline_prefix_cache_off",
    }
    assert {row.where for row in FAKE_GAINS} <= {"acceptance", "executor"}
    assert all(row.shows_as and callable(row.check) for row in FAKE_GAINS)


def test_an_acceptance_row_code_is_a_rejection_code_and_an_executor_row_code_is_not() -> None:
    for row in FAKE_GAINS:
        assert (row.refusal_code in REJECTION_CODES) is (row.where == "acceptance"), row.name


def _row(name: str) -> FakeGain:
    return ROWS[name]


async def test_the_context_copy_row_is_refused_by_acceptance(tmp_path: Path) -> None:
    row = _row("context_copy_draft_on_synthetic")
    assert await _refused_with(tmp_path, _repo(), WorkloadKind.INFERENCEX, NGRAM) == row.refusal_code


async def test_the_pinned_speculative_row_is_refused_by_acceptance(tmp_path: Path) -> None:
    row = _row("speculative_decoding_pinned")
    code = await _refused_with(tmp_path, _repo(SIMULATING, SIM_ENV), WorkloadKind.AGENTX, "--speculative-num-steps 5")
    assert code == row.refusal_code


async def test_the_cacheless_baseline_row_is_refused_by_the_executor(tmp_path: Path) -> None:
    row = _row("agentx_baseline_prefix_cache_off")
    made = make_rig(tmp_path)
    try:
        result = await _run(tmp_path, made, SGLANG_OFF, WorkloadKind.AGENTX, Purpose.BASELINE)
    finally:
        made.backend.close()
        made.provider.close()
    assert (result.failure_reason or "").startswith(f"{row.refusal_code}: ")
