# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for GEAK acceptance identity in the breakdown collectors.

Each test here is anchored to a case measured on the recorded campaign at
``/shared_nfs/hyperloom-claw``. The behaviours under test are the ones a
reviewer flagged on Hyperloom PR #1209:

* **Two lanes.** An acceptance lands in ``accepted_kernels`` *or*
  ``accepted_heads``. Measured: 8 of 11 sessions with an acceptance carry it in
  ``accepted_heads`` alone, 3 in ``accepted_kernels`` alone, 0 in both. Reading
  one lane loses most of them.
* **Alias twins.** GEAK records one acceptance as two journey rows: the
  candidate-slot row carries the measurement, the resolved-symbol row does not.
  The symbol is on the *unmeasured* twin, and ``name`` holds it while
  ``kernel_id`` holds an underscore-stripped slug.
* **Shared admission.** A journey row declares no ``kind``, so a library
  selection (``kind="env"``) would be credited as an authored kernel. The kind
  is recovered by joining the symbol back to the run's ``result.json`` lane, and
  what the join cannot resolve is recorded as unresolved rather than guessed.
"""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_accepted_kernel_specs,
    geak_is_cand_tag,
    geak_spec_is_env,
    geak_spec_kind,
    geak_spec_name,
)


def _spec(name: str, delta: float, **extra: Any) -> dict[str, Any]:
    return {"short_name": name, "e2e_delta_pct": delta, **extra}


# --------------------------------------------------------------------------
# B2 — the alias twin collapses onto the resolved symbol
# --------------------------------------------------------------------------










def test_specs_keeps_two_distinct_kernels_that_share_op_kind_and_gain() -> None:
    result = {
        "accepted_kernels": [
            _spec("kernel_a", 12.31, op_kind="prefill_attn"),
            _spec("kernel_b", 12.31, op_kind="prefill_attn"),
        ]
    }
    assert [geak_spec_name(s) for s in _geak_accepted_kernel_specs(result)] == [
        "kernel_a",
        "kernel_b",
    ]








def _acceptance_specs(result: dict[str, Any]) -> list[dict[str, Any]]:
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    return KernelPhase._geak_acceptance_specs(result)


def test_acceptance_specs_collapse_the_alias_twin_onto_the_kernel_symbol() -> None:
    result = {
        "accepted_kernels": [_spec("c0_triton", 12.31, op_kind="prefill_attn")],
        "accepted_heads": [_spec("_dsa_prefill_kernel", 12.31, op_kind="prefill_attn")],
    }
    out = _acceptance_specs(result)
    assert [geak_spec_name(row) for row in out] == ["_dsa_prefill_kernel"]
    assert out[0]["alias_collapsed"] is True


def test_acceptance_specs_keep_the_collapsed_twins_name_as_an_alias() -> None:
    """The collapsed name is the one a reader may hold, so it is carried over."""
    result = {
        "accepted_kernels": [_spec("c0_triton", 12.31, op_kind="prefill_attn")],
        "accepted_heads": [_spec("_dsa_prefill_kernel", 12.31, op_kind="prefill_attn")],
    }
    out = _acceptance_specs(result)
    assert out[0]["aliases"] == ["c0_triton"]


def test_acceptance_specs_do_not_alias_a_row_to_its_own_name() -> None:
    """A survivor listing itself would make the alias set useless for joining."""
    result = {
        "accepted_kernels": [_spec("_dsa_prefill_kernel", 12.31, op_kind="prefill_attn")],
        "accepted_heads": [_spec("c0_triton", 12.31, op_kind="prefill_attn")],
    }
    out = _acceptance_specs(result)
    assert geak_spec_name(out[0]) == "_dsa_prefill_kernel"
    assert out[0]["aliases"] == ["c0_triton"]


def test_acceptance_specs_keep_two_rows_that_merely_both_lack_a_delta() -> None:
    """No measured delta is no evidence of twinning.

    Reading the delta as ``float(x or 0.0)`` mapped absent and zero onto the
    same number, so two unrelated env selections on one op_kind collapsed into
    one acceptance for carrying no measurement at all.
    """
    result = {
        "accepted_heads": [
            {"short_name": "ck_gemm_a8w8", "kind": "env", "op_kind": "gemm"},
            {"short_name": "hipblaslt_gemm", "kind": "env", "op_kind": "gemm"},
        ]
    }
    assert sorted(geak_spec_name(row) for row in _acceptance_specs(result)) == [
        "ck_gemm_a8w8",
        "hipblaslt_gemm",
    ]


def test_acceptance_specs_keep_a_row_whose_delta_is_not_a_number() -> None:
    """An unreadable delta is a row with no delta, not an absent acceptance."""
    result = {"accepted_heads": [_spec("odd_kernel", "n/a", op_kind="gemm")]}  # type: ignore[arg-type]
    out = _acceptance_specs(result)
    assert [geak_spec_name(row) for row in out] == ["odd_kernel"]


def test_acceptance_specs_still_distinguish_a_measured_zero_from_an_absent_delta() -> None:
    result = {
        "accepted_heads": [
            _spec("measured_zero", 0.0, op_kind="gemm"),
            {"short_name": "no_delta", "op_kind": "gemm"},
        ]
    }
    assert len(_acceptance_specs(result)) == 2


def test_cand_tag_recognises_slot_tags_only() -> None:
    assert geak_is_cand_tag("c0_triton")
    assert geak_is_cand_tag("cand_c1_flydsl")
    assert not geak_is_cand_tag("_mxfp8_linear_kernel")
    assert not geak_is_cand_tag("")


# --------------------------------------------------------------------------
# B3 — one admission test, shared with the ledger
# --------------------------------------------------------------------------
















# --------------------------------------------------------------------------
# The shared helpers the two collectors now agree on
# --------------------------------------------------------------------------


def test_spec_kind_returns_none_for_absent_and_for_empty() -> None:
    # None and "" must not be distinguishable downstream: both mean undeclared.
    assert geak_spec_kind({"short_name": "k"}) is None
    assert geak_spec_kind({"short_name": "k", "kind": ""}) is None
    assert geak_spec_kind({"short_name": "k", "kind": "  "}) is None
    assert geak_spec_kind({"short_name": "k", "kind": "AUTHORED"}) == "authored"
    assert geak_spec_kind("bare_string") is None


def test_spec_is_env_is_true_only_for_a_declared_env() -> None:
    assert geak_spec_is_env({"short_name": "k", "kind": "env"})
    assert not geak_spec_is_env({"short_name": "k", "kind": "authored"})
    assert not geak_spec_is_env({"short_name": "k"})
    assert not geak_spec_is_env("bare_string")


def test_spec_name_reads_every_spelling_including_a_bare_string() -> None:
    assert geak_spec_name({"short_name": "s", "kernel_id": "k"}) == "s"
    assert geak_spec_name({"kernel_id": "k"}) == "k"
    assert geak_spec_name({"cand_tag": "c0_triton"}) == "c0_triton"
    assert geak_spec_name("  bare_string  ") == "bare_string"
    assert geak_spec_name(None) == ""
