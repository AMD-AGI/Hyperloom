# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Numerical regressions cannot hide behind a passing scalar SNR or stale evidence."""

from __future__ import annotations

import asyncio
import copy
import json
import sys

import pytest
import yaml

from kernelforge.loop.canonical_correctness import accept_candidate
from kernelforge.loop.numerical import EVIDENCE_PREFIX, judge_evidence, measure_outputs, validate_contract


def contract():
    return {
        "schema_version": 1,
        "repetitions": 5,
        "cases": {
            "moe/output/graph": {"max_oracle_error": 10 ** (-30 / 20), "max_error_ratio": 1.25, "error_floor": 1e-6}
        },
    }


def evidence(*, candidate_oracle_db=45, candidate_repeat_db=45):
    source = {"finite": True, "oracle_errors": [10 ** (-45 / 20)] * 5, "repeat_errors": [10 ** (-45 / 20)] * 4}
    candidate = {
        "finite": True,
        "oracle_errors": [10 ** (-candidate_oracle_db / 20)] * 5,
        "repeat_errors": [10 ** (-candidate_repeat_db / 20)] * 4,
    }
    return {
        "schema_version": 1,
        "request_id": "fresh",
        "cases": [
            {
                "id": "moe/output/graph",
                "source_before": copy.deepcopy(source),
                "candidate": candidate,
                "source_after": copy.deepcopy(source),
            }
        ],
    }


def judge(record, policy=None):
    return judge_evidence(EVIDENCE_PREFIX + json.dumps(record), policy or contract(), "fresh")


def test_stage1_math_snr_pass_does_not_mask_repeatability_regression():
    passed, detail, result = judge(evidence(candidate_oracle_db=31, candidate_repeat_db=27))
    assert not passed
    assert "repeat_errors" in detail and "oracle_errors" in detail
    assert "contract_sha256" in result


def test_stability_regression_is_rejected_even_with_unchanged_oracle_error():
    passed, detail, _ = judge(evidence(candidate_oracle_db=45, candidate_repeat_db=27))
    assert not passed and "repeat_errors" in detail


def test_source_must_pass_its_absolute_math_tolerance():
    record = evidence()
    record["cases"][0]["source_before"]["oracle_errors"][0] = 1
    passed, detail, _ = judge(record)
    assert not passed and "source_before: mathematical" in detail


def test_matching_nondeterministic_source_is_allowed():
    assert judge(evidence())[0]


def test_deterministic_contract_rejects_any_unpermitted_error():
    policy = contract()
    policy["cases"]["moe/output/graph"] = {"max_oracle_error": 0, "max_error_ratio": 1, "error_floor": 0}
    record = evidence(candidate_oracle_db=999, candidate_repeat_db=999)
    for value in record["cases"][0].values():
        if isinstance(value, dict):
            value.update(oracle_errors=[0] * 5, repeat_errors=[0] * 4)
    assert judge(record, policy)[0]
    record["cases"][0]["candidate"]["repeat_errors"][0] = 1e-12
    assert not judge(record, policy)[0]


@pytest.mark.parametrize(
    "corruption",
    ["missing_case", "duplicate_case", "stale", "missing_metric", "short_repeat", "nan", "negative", "boolean"],
)
def test_incomplete_or_malformed_evidence_cannot_pass(corruption):
    record = evidence()
    item = record["cases"][0]["candidate"]
    if corruption == "missing_case":
        record["cases"] = []
    elif corruption == "duplicate_case":
        record["cases"] *= 2
    elif corruption == "stale":
        record["request_id"] = "old"
    elif corruption == "missing_metric":
        del item["repeat_errors"]
    elif corruption == "short_repeat":
        item["repeat_errors"].pop()
    else:
        item["oracle_errors"][0] = {"nan": float("nan"), "negative": -1, "boolean": True}[corruption]
    with pytest.raises(ValueError):
        judge(record)


def test_nonfinite_output_fails_despite_zero_error_placeholders():
    record = evidence()
    record["cases"][0]["candidate"]["finite"] = False
    assert not judge(record)[0]


@pytest.mark.parametrize("output", ["SNR: 40 dB\nallclose: True", EVIDENCE_PREFIX + "{}\n" + EVIDENCE_PREFIX + "{}"])
def test_textual_success_or_multiple_records_are_not_numerical_evidence(output):
    with pytest.raises(ValueError):
        judge_evidence(output, contract(), "fresh")


@pytest.mark.parametrize("change", [{"repetitions": 1}, {"schema_version": 2}, {"cases": {}}, {"repetitions": True}])
def test_contract_cannot_omit_coverage_or_repetitions(change):
    with pytest.raises(ValueError):
        validate_contract({**contract(), **change})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_tolerances_are_rejected(value):
    policy = contract()
    policy["cases"]["moe/output/graph"]["max_oracle_error"] = value
    with pytest.raises(ValueError):
        validate_contract(policy)


def _task(tmp_path, record=None, *, declared=True):
    task = {
        "compile_command": [f'"{sys.executable}" compile.py'],
        "correctness_command": [f'"{sys.executable}" driver.py'],
    }
    if declared:
        task["numerical_validation"] = contract()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(task))
    (tmp_path / "compile.py").write_text("pass\n")
    text = "print('SNR: 31 dB')\n"
    if record is not None:
        text += (
            "import json,os\nrecord="
            + repr(record)
            + "\nrecord['request_id']=os.environ['FORGE_NUMERICAL_REQUEST']\nprint('__FORGE_NUMERICAL__'+json.dumps(record))\n"
        )
    (tmp_path / "driver.py").write_text(text)


def _accept(tmp_path):
    return asyncio.run(
        accept_candidate(str(tmp_path), timeout_cap_sec=10, candidate_label="ASM", kernel_backend="assembly")
    )


@pytest.mark.parametrize("kind", ["missing_config", "missing_contract", "missing_evidence", "unstable", "valid"])
def test_real_acceptance_enforces_the_contract(tmp_path, kind):
    if kind != "missing_config":
        record = (
            evidence(candidate_repeat_db=27 if kind == "unstable" else 45) if kind in ("unstable", "valid") else None
        )
        _task(tmp_path, record, declared=kind != "missing_contract")
    result = _accept(tmp_path)
    assert result.passed is (kind == "valid")
    if kind == "unstable":
        assert result.outcome == "numerical_correctness_failure"
        assert result.numerical_evidence
    elif kind.startswith("missing"):
        assert result.outcome in ("unverified", "invalid_result")


@pytest.mark.parametrize("kind", ["numerical_failure", "text_only_failure", "missing_evidence", "crash"])
def test_reported_failure_never_hides_numerical_evidence_or_authorizes_acceptance(tmp_path, kind):
    record = (
        None if kind == "missing_evidence" else evidence(candidate_repeat_db=27 if kind == "numerical_failure" else 45)
    )
    _task(tmp_path, record)
    path = tmp_path / "driver.py"
    path.write_text(
        path.read_text() + "\nprint('FAIL')\n" + ("raise RuntimeError('crash')\n" if kind == "crash" else "")
    )
    result = _accept(tmp_path)
    assert not result.passed
    if kind == "numerical_failure":
        assert result.outcome == "numerical_correctness_failure"
        assert result.numerical_evidence
    else:
        assert not result.numerical_evidence


def test_measurements_own_output_storage_and_detect_oscillation():
    torch = pytest.importorskip("torch")
    reference = torch.ones(4)
    output = torch.empty_like(reference)
    count = 0

    def run():
        nonlocal count
        count += 1
        output.fill_(1 + 0.01 * (count % 2))
        return output

    result = measure_outputs(run, reference, repetitions=5)
    assert all(x > 0.009 for x in result["repeat_errors"])
    assert result["oracle_errors"][1] == 0


def test_zero_reference_and_nonfinite_outputs():
    torch = pytest.importorskip("torch")
    result = measure_outputs(lambda: torch.ones(4), torch.zeros(4), repetitions=3)
    assert result["oracle_errors"] == [2.0] * 3
    invalid = measure_outputs(lambda: torch.full((4,), float("nan")), torch.ones(4), repetitions=3)
    assert not invalid["finite"]
