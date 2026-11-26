# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Speculative decoding a candidate turns on is graded at the acceptance InferenceX records for its draft (synthetic).

Covers the rule (a real model-based draft, a forced value that differs from the recorded one, and a draft InferenceX
records nothing for are refused; the recorded value forced is not; bases that draft, and context-copying drafts, are
left to their own checks), the lookup a session binds, and explore's filter, which drops such a variant before it runs
under its own reason beside a context-copying one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from inference_suite import GoldenLookupError, SourceGolden
from inference_suite.contracts.base_launch import GoldenCurve

from hyperloom.inference_optimizer.tests.test_explore_executor import _noop_filter_kwargs
from hyperloom.orchestrator.actions.executors.grid._grid_runner import GridVariant
from hyperloom.orchestrator.actions.executors.framework_change._spec_decode_context_copy import CONTEXT_COPY_REFUSAL
from hyperloom.orchestrator.actions.executors.explore_prefilter.spec_decode_golden import (
    GOLDEN_ACCEPTANCE_REFUSAL,
    GoldenLookup,
    golden_acceptance_refusal,
    session_golden_lookup,
    synthetic_draft_refusal,
)
from hyperloom.orchestrator.actions.executors.explore_prefilter.launch_filters import (
    filter_synthetic_drafts,
    launch_forces_acceptance,
)

_CURVE = GoldenCurve(
    repo="https://github.com/SemiAnalysisAI/InferenceX",
    commit="4" * 40,
    path="inferencex-e2e/infx/golden_al_distribution/m__mtp__native__default-default.yaml",
    sha256="a" * 64,
)
_MTP = '--speculative-config {"method":"mtp","num_speculative_tokens":3}'
_FORCED = (
    '--speculative-config {"method":"mtp","num_speculative_tokens":3,'
    '"rejection_sample_method":"synthetic","synthetic_acceptance_length":%s}'
)
_SGLANG = "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4"
_SGLANG_FORCED = {
    "SGLANG_SIMULATE_ACC_LEN": "2.32",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}
_ATOM = "--method mtp --num-speculative-tokens 3"
_NGRAM = '--speculative-config {"method":"ngram","num_speculative_tokens":7}'
_DOTTED_FORCED = (
    "--speculative-config.method mtp --speculative-config.num_speculative_tokens 3 "
    "--speculative-config.rejection_sample_method synthetic --speculative-config.synthetic_acceptance_length 2.32"
)


_RECORDED = SourceGolden(2.32, curve=_CURVE)


class _Lookup:
    """Stands in for the session's lookup: answers ``golden`` and records the argv it was asked about."""

    def __init__(self, golden: SourceGolden | None = _RECORDED, error: str | None = None) -> None:
        self.golden = golden
        self.error = error
        self.asked: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> SourceGolden | None:
        self.asked.append(tuple(argv))
        if self.error is not None:
            raise GoldenLookupError(self.error)
        return self.golden


def _vllm(answer: _Lookup | None = None) -> GoldenLookup:
    return GoldenLookup("vllm", answer or _Lookup())


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def test_a_real_draft_turned_on_is_refused_with_the_value_to_force_and_its_curve() -> None:
    lookup = _Lookup()

    reason = golden_acceptance_refusal("", {}, _MTP, {}, _vllm(lookup))

    assert "verifies a real draft" in reason and "repetition loops" in reason
    assert '"synthetic_acceptance_length": 2.32' in reason and "SGLANG_SIMULATE_ACC_LEN=2.32" in reason
    assert _CURVE.path in reason
    assert lookup.asked == [("--speculative-config", '{"method":"mtp","num_speculative_tokens":3}')]


@pytest.mark.parametrize(
    ("framework", "argv", "envs"),
    [
        pytest.param("vllm", _FORCED % "2.32", {}, id="vllm"),
        pytest.param("vllm", _FORCED % "2.320", {}, id="vllm-same-number-spelled-otherwise"),
        pytest.param("vllm", _DOTTED_FORCED, {}, id="vllm-per-key-flags"),
        pytest.param("sglang", _SGLANG, _SGLANG_FORCED, id="sglang"),
        pytest.param("atom", f"{_ATOM} --spec-decode-acceptance-length 2.32", {}, id="atom"),
    ],
)
def test_the_recorded_value_forced_is_not_refused(framework: str, argv: str, envs: dict[str, str]) -> None:
    assert golden_acceptance_refusal("", {}, argv, envs, GoldenLookup(framework, _Lookup())) == ""


@pytest.mark.parametrize(
    ("framework", "argv", "envs"),
    [
        pytest.param("vllm", _MTP, _SGLANG_FORCED, id="sglangs-env-on-vllm"),
        pytest.param("vllm", _FORCED.replace('"synthetic"', '"standard"') % "2.32", {}, id="vllm-without-its-sampler"),
        pytest.param("sglang", _FORCED % "2.32", {}, id="vllms-flags-on-sglang"),
        pytest.param("atom", _ATOM, _SGLANG_FORCED, id="atom-real-draft-with-sglangs-env"),
    ],
)
def test_a_length_the_engine_does_not_read_leaves_a_real_draft(framework: str, argv: str, envs: dict) -> None:
    reason = golden_acceptance_refusal("", {}, argv, envs, GoldenLookup(framework, _Lookup()))

    assert "verifies a real draft" in reason


def test_another_forced_value_is_refused_naming_both() -> None:
    reason = golden_acceptance_refusal("", {}, _FORCED % "3.5", {}, _vllm())

    assert "forces an acceptance length of 3.5" in reason and "records 2.32" in reason
    assert "not a knob" in reason


def test_a_draft_inferencex_records_nothing_for_is_refused_with_why() -> None:
    lookup = _Lookup(SourceGolden(None, reason="InferenceX names no model prefix for Qwen/Qwen3.8-27B"))

    reason = golden_acceptance_refusal("", {}, _FORCED % "2.32", {}, _vllm(lookup))

    assert "records no acceptance length" in reason and "Qwen/Qwen3.8-27B" in reason
    assert "SPEED-Bench" in reason and "Leave speculative decoding off" in reason


def test_a_lookup_that_cannot_answer_refuses_naming_why() -> None:
    reason = golden_acceptance_refusal(
        "", {}, _MTP, {}, _vllm(_Lookup(error="the staged InferenceX exited with code 1"))
    )

    assert "could not be read" in reason and "exited with code 1" in reason


@pytest.mark.parametrize(
    ("base", "base_envs", "candidate"),
    [
        pytest.param(_MTP, {}, _MTP.replace(":3", ":4"), id="a-base-that-verifies-a-real-draft"),
        pytest.param(_FORCED % "2.32", {}, _FORCED % "3.5", id="a-base-that-simulates-acceptance"),
        pytest.param(_SGLANG, _SGLANG_FORCED, _SGLANG, id="a-base-whose-simulation-is-in-its-env"),
        pytest.param("", {}, "--max-num-seqs 64", id="no-draft"),
        pytest.param(_MTP, {}, "--max-num-seqs 64", id="drops-the-draft"),
    ],
)
def test_bases_that_draft_and_candidates_that_draft_nothing_are_left_alone(
    base: str, base_envs: dict[str, str], candidate: str
) -> None:
    lookup = _Lookup()

    assert golden_acceptance_refusal(base, base_envs, candidate, {}, _vllm(lookup)) == ""
    assert lookup.asked == []


def test_a_launch_inferencex_reads_no_draft_from_is_not_refused() -> None:
    assert golden_acceptance_refusal("", {}, _MTP, {}, _vllm(_Lookup(golden=None))) == ""


def test_a_context_copying_draft_is_the_copy_checks_and_a_model_draft_the_golden_checks() -> None:
    assert synthetic_draft_refusal("", {}, _NGRAM, {}, _vllm())[0] == CONTEXT_COPY_REFUSAL
    assert synthetic_draft_refusal("", {}, _MTP, {}, _vllm())[0] == GOLDEN_ACCEPTANCE_REFUSAL
    assert synthetic_draft_refusal("", {}, _FORCED % "2.32", {}, _vllm()) is None


def test_a_session_without_a_frozen_base_cannot_look_a_draft_up(tmp_path: Path) -> None:
    for session in (None, tmp_path):
        with pytest.raises(GoldenLookupError, match="froze no base launch"):
            session_golden_lookup(session, "vllm").answer(("--speculative-config", "{}"))


# --------------------------------------------------------------------------
# explore
# --------------------------------------------------------------------------
def test_explore_drops_each_variant_under_its_own_reason_and_keeps_the_forced_draft(tmp_path: Path) -> None:
    recipe = _noop_filter_kwargs(tmp_path)["recipe"]
    grid = [
        GridVariant("real", extra_server_args=_SGLANG),
        GridVariant("forced", extra_server_args=_SGLANG, extra_envs=_SGLANG_FORCED),
        GridVariant("copying", extra_server_args="--speculative-algorithm NGRAM"),
        GridVariant("lever", extra_server_args="--max-running-requests 64"),
    ]

    kept, dropped = filter_synthetic_drafts(grid, recipe=recipe, lookup=GoldenLookup("sglang", _Lookup()))

    assert [gv.name for gv in kept] == ["forced", "lever"]
    assert {name: reason for name, reason, _ in dropped} == {
        "real": GOLDEN_ACCEPTANCE_REFUSAL,
        "copying": CONTEXT_COPY_REFUSAL,
    }
    assert "SGLANG_SIMULATE_ACC_LEN=2.32" in {name: detail for name, _, detail in dropped}["real"]


@pytest.mark.parametrize(
    ("variant", "forces"),
    [
        pytest.param(GridVariant("forced", extra_server_args=_SGLANG, extra_envs=_SGLANG_FORCED), True, id="forced"),
        pytest.param(GridVariant("real", extra_server_args=_SGLANG), False, id="real-draft"),
        pytest.param(GridVariant("lever", extra_server_args="--max-running-requests 64"), False, id="no-draft"),
    ],
)
def test_explore_knows_which_launch_would_eval_under_a_forced_acceptance(tmp_path: Path, variant, forces) -> None:
    recipe = _noop_filter_kwargs(tmp_path)["recipe"]

    assert launch_forces_acceptance(recipe, variant) is forces
