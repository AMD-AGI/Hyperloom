# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the two units that turn an acceptance into a named kernel.

``_geak_accepted_kernel_specs`` decides WHICH acceptances count as a kernel;
``KernelPhase._record_geak_adopted_kernels`` writes them into
``state.kernel_integrate_attempts``, the per-kernel ledger that ``by_kernel``,
``kernel_lifecycle.adopted`` and the attribution split all read. GEAK wrote only
the per-ACTION headline, so an adopted kernel existed in the headline and
nowhere a report could name it.

Both are exercised end to end by the collector tests, but only through inputs
that reach the early returns. These cover the selection rules and the ledger row
itself, which are the parts a downstream report depends on being exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hyperloom.inference_optimizer.breakdown.collectors.kernels import (
    _collect_adopted_kernels,
    collect_collective,
)
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_accepted_kernel_specs,
    _geak_overlay_digest,
    _geak_overlay_is_loadable,
    geak_spec_name,
)
from hyperloom.orchestrator.phases.kernel import KernelPhase


def _spec(name: str, delta: float, **extra: Any) -> dict[str, Any]:
    return {"short_name": name, "e2e_delta_pct": delta, **extra}


# --------------------------------------------------------------------------
# Which acceptances count as a kernel
# --------------------------------------------------------------------------


def test_non_dict_result_yields_no_specs() -> None:
    assert _geak_accepted_kernel_specs(None) == []
    assert _geak_accepted_kernel_specs("no_gain") == []


def test_both_lanes_are_read() -> None:
    # 11 campaign runs carry an acceptance: 8 in accepted_heads only, 3 in
    # accepted_kernels only. Reading one lane loses most of them.
    result = {
        "accepted_kernels": [_spec("k_from_kernels", 4.0)],
        "accepted_heads": [_spec("k_from_heads", 6.0)],
    }
    assert [geak_spec_name(s) for s in _geak_accepted_kernel_specs(result)] == [
        "k_from_kernels",
        "k_from_heads",
    ]


def test_env_selections_are_excluded() -> None:
    # ``kind: env`` selects an existing library (ck_gemm_a8w8_blockscale_...);
    # no kernel was authored, so it belongs in the config half of the gain.
    result = {
        "accepted_kernels": [
            _spec("ck_gemm_a8w8_blockscale_bpreshuffle", 14.9, kind="env"),
            _spec("_mxfp8_linear_kernel", 13.9, kind="authored"),
        ]
    }
    assert [geak_spec_name(s) for s in _geak_accepted_kernel_specs(result)] == ["_mxfp8_linear_kernel"]


def test_non_positive_and_unparseable_deltas_are_dropped() -> None:
    result = {
        "accepted_kernels": [
            _spec("regression", -1.2),
            _spec("flat", 0.0),
            _spec("unparseable", "not-a-number"),
            _spec("real", 2.5),
        ]
    }
    assert [geak_spec_name(s) for s in _geak_accepted_kernel_specs(result)] == ["real"]


def test_unnamed_and_non_dict_rows_are_skipped() -> None:
    result = {
        "accepted_kernels": [
            "bare_string",
            {"e2e_delta_pct": 9.0},
            {"short_name": "   ", "e2e_delta_pct": 9.0},
            _spec("named", 9.0),
        ]
    }
    assert [geak_spec_name(s) for s in _geak_accepted_kernel_specs(result)] == ["named"]


def test_alias_twin_collapses_to_the_kernel_symbol() -> None:
    # One acceptance written under both the candidate tag and the symbol. The
    # tag says which slot proposed it; only the symbol can be named in a report.
    result = {
        "accepted_kernels": [_spec("cand_c0_triton", 29.994, op_kind="sparse_attn")],
        "accepted_heads": [_spec("dsa_sparse_attn_prefill_main_kernel", 29.994, op_kind="sparse_attn")],
    }
    specs = _geak_accepted_kernel_specs(result)
    assert [geak_spec_name(s) for s in specs] == ["dsa_sparse_attn_prefill_main_kernel"]


def test_alias_twin_collapse_keeps_the_symbol_when_it_arrives_first() -> None:
    result = {
        "accepted_kernels": [
            _spec("dsa_sparse_attn_prefill_main_kernel", 29.994, op_kind="sparse_attn"),
            _spec("cand_c0_triton", 29.994, op_kind="sparse_attn"),
        ]
    }
    specs = _geak_accepted_kernel_specs(result)
    assert [geak_spec_name(s) for s in specs] == ["dsa_sparse_attn_prefill_main_kernel"]


def test_same_delta_on_a_different_op_kind_is_not_a_twin() -> None:
    result = {
        "accepted_kernels": [
            _spec("a", 12.31, op_kind="decode_attn"),
            _spec("b", 12.31, op_kind="prefill_attn"),
        ]
    }
    assert len(_geak_accepted_kernel_specs(result)) == 2


# --------------------------------------------------------------------------
# Overlay evidence on malformed manifests
# --------------------------------------------------------------------------


def test_manifest_that_is_not_an_object_is_not_loadable(tmp_path: Path) -> None:
    root = tmp_path / "ov"
    root.mkdir()
    (root / "sitecustomize.py").write_text("# loads\n")
    (root / "_overlay_manifest.json").write_text(json.dumps(["rebinds"]))
    assert _geak_overlay_is_loadable(str(root)) is False


def test_unparseable_manifest_still_digests_its_bytes(tmp_path: Path) -> None:
    # A digest is a comparison key, not a verdict: unreadable JSON must still
    # produce a stable value so two runs of the same overlay compare equal.
    root = tmp_path / "ov"
    root.mkdir()
    (root / "_overlay_manifest.json").write_text("{not json")
    digest = _geak_overlay_digest(str(root))
    assert digest and digest == _geak_overlay_digest(str(root))


# --------------------------------------------------------------------------
# The per-kernel ledger row
# --------------------------------------------------------------------------


def _phase() -> SimpleNamespace:
    return SimpleNamespace(shared_state=SimpleNamespace(kernel_integrate_attempts={}, macro_cycle=3))


def _record(phase: SimpleNamespace, result: Any, **kw: Any) -> None:
    KernelPhase._record_geak_adopted_kernels(
        phase,
        result,
        measured_tput=kw.get("measured_tput", 120.0),
        baseline_tput=kw.get("baseline_tput", 100.0),
        provenance=kw.get("provenance", "geak_revalidate"),
        overlay_loaded=kw.get("overlay_loaded", True),
    )


def test_no_ledger_row_without_an_accepted_kernel() -> None:
    phase = _phase()
    _record(phase, {"status": "no_gain"})
    _record(phase, None)
    assert phase.shared_state.kernel_integrate_attempts == {}


def test_single_kernel_with_a_loaded_overlay_is_attributable() -> None:
    phase = _phase()
    result = {
        "accepted_kernels": [_spec("_mxfp8_linear_kernel", 13.87, kind="authored", isolated=2.39)],
        "alignment_metrics": {"final_basis": "cold"},
        "baseline_alignment": {"status": "aligned"},
    }
    _record(phase, result, measured_tput=120.0, baseline_tput=100.0)
    entry = phase.shared_state.kernel_integrate_attempts["_mxfp8_linear_kernel"]
    assert entry["validated"] is True
    assert entry["best_gain_pct"] == 20.0  # orchestrator rebench, not GEAK's 13.87
    assert entry["geak_same_config_delta_pct"] == 13.87
    assert entry["geak_isolated_speedup"] == 2.39
    assert entry["source"] == "geak_e2e"
    assert entry["basis"] == "cold"
    assert entry["alignment_status"] == "aligned"
    assert entry["attempt_count"] == 1
    attempt = entry["attempts"][0]
    assert attempt["decision"] == "KEEP"
    assert attempt["gain_pct"] == 20.0
    assert attempt["artifact_kind"] == "authored"
    assert attempt["cycle"] == 3


def test_an_unproven_overlay_records_the_row_without_a_gain() -> None:
    # The kernel is still named -- withholding the row loses it entirely -- but
    # a gain measured without proof the kernel ran is not the kernel's.
    phase = _phase()
    _record(
        phase,
        {"accepted_kernels": [_spec("k", 5.0)]},
        overlay_loaded=False,
    )
    entry = phase.shared_state.kernel_integrate_attempts["k"]
    assert entry["validated"] is False
    assert entry["best_gain_pct"] is None
    assert entry["overlay_loaded"] is False
    assert entry["last_decision"] == "UNATTRIBUTED"
    assert entry["last_status"] == "unvalidated"
    assert entry["attempts"][0]["gain_pct"] is None
    assert entry["attempts"][0]["decision"] == "UNATTRIBUTED"


def test_unproven_overlay_geak_row_is_not_adopted() -> None:
    phase = _phase()
    _record(
        phase,
        {"accepted_kernels": [_spec("k", 5.0)]},
        overlay_loaded=False,
    )
    adopted = _collect_adopted_kernels({"kernel_integrate_attempts": phase.shared_state.kernel_integrate_attempts})
    assert adopted == []


def test_joint_rebench_with_proven_overlay_is_still_adopted() -> None:
    phase = _phase()
    result = {"accepted_kernels": [_spec("k_one", 5.0), _spec("k_two", 7.0)]}
    _record(phase, result)
    adopted = _collect_adopted_kernels({"kernel_integrate_attempts": phase.shared_state.kernel_integrate_attempts})
    assert {r["kernel_id"] for r in adopted} == {"k_one", "k_two"}
    assert all(r["validated"] is False for r in adopted)


def test_historical_keep_survives_a_later_unproven_rebench() -> None:
    phase = _phase()
    result = {"accepted_kernels": [_spec("k", 5.0)]}
    _record(phase, result, measured_tput=150.0)
    _record(phase, result, measured_tput=150.0, overlay_loaded=False)
    adopted = _collect_adopted_kernels({"kernel_integrate_attempts": phase.shared_state.kernel_integrate_attempts})
    assert len(adopted) == 1
    assert adopted[0]["kernel_id"] == "k"
    assert adopted[0]["e2e_gain_pct"] == 50.0
    assert adopted[0]["validated"] is False


def test_reverted_forge_kernel_with_prior_keep_is_not_adopted() -> None:
    state = {
        "kernel_integrate_attempts": {
            "my_kernel": {
                "kernel_id": "my_kernel",
                "attempts": [
                    {"decision": "KEEP", "gain_pct": 8.0},
                    {"decision": "REVERT", "gain_pct": -2.0},
                ],
                "last_decision": "REVERT",
                "best_gain_pct": 8.0,
                "validated": True,
            }
        }
    }
    assert _collect_adopted_kernels(state) == []


def test_two_kernels_on_one_rebench_share_no_invented_split() -> None:
    phase = _phase()
    result = {
        "accepted_kernels": [_spec("k_one", 5.0), _spec("k_two", 7.0)],
    }
    _record(phase, result)
    ledger = phase.shared_state.kernel_integrate_attempts
    assert set(ledger) == {"k_one", "k_two"}
    assert all(e["best_gain_pct"] is None and e["validated"] is False for e in ledger.values())


def test_a_missing_measurement_leaves_the_gain_null() -> None:
    phase = _phase()
    _record(phase, {"accepted_kernels": [_spec("k", 5.0)]}, measured_tput=0.0)
    assert phase.shared_state.kernel_integrate_attempts["k"]["best_gain_pct"] is None


def test_a_second_promotion_appends_rather_than_replaces() -> None:
    phase = _phase()
    result = {"accepted_kernels": [_spec("k", 5.0)]}
    _record(phase, result)
    _record(phase, result, measured_tput=150.0)
    entry = phase.shared_state.kernel_integrate_attempts["k"]
    assert entry["attempt_count"] == 2
    assert [a["new_tput"] for a in entry["attempts"]] == [120.0, 150.0]
    assert entry["best_gain_pct"] == 50.0


def test_best_gain_is_the_max_over_attempts_not_the_last_one() -> None:
    # ``by_kernel`` and ``kernel_lifecycle`` read ``best_gain_pct`` from this
    # writer and from ``_kernel_decisions.py`` alike, and that one is a max. A
    # second, worse rebench must not lower the kernel's best.
    phase = _phase()
    result = {"accepted_kernels": [_spec("k", 5.0)]}
    _record(phase, result, measured_tput=150.0)  # +50%
    _record(phase, result, measured_tput=110.0)  # +10%
    entry = phase.shared_state.kernel_integrate_attempts["k"]
    assert [a["gain_pct"] for a in entry["attempts"]] == [50.0, 10.0]
    assert entry["best_gain_pct"] == 50.0


def test_an_unattributable_second_attempt_does_not_erase_the_first_gain() -> None:
    phase = _phase()
    result = {"accepted_kernels": [_spec("k", 5.0)]}
    _record(phase, result, measured_tput=150.0)
    _record(phase, result, measured_tput=150.0, overlay_loaded=False)
    entry = phase.shared_state.kernel_integrate_attempts["k"]
    assert entry["attempts"][1]["gain_pct"] is None
    assert entry["best_gain_pct"] == 50.0
    assert entry["validated"] is False


def test_a_non_dict_ledger_is_left_alone() -> None:
    phase = SimpleNamespace(shared_state=SimpleNamespace(kernel_integrate_attempts=None, macro_cycle=0))
    _record(phase, {"accepted_kernels": [_spec("k", 5.0)]})
    assert phase.shared_state.kernel_integrate_attempts is None


def test_a_non_dict_result_records_no_candidate() -> None:
    phase = SimpleNamespace(
        shared_state=SimpleNamespace(kernel_integrate_attempts={}, macro_cycle=0),
        session_dir=None,
    )
    # Returns without touching state rather than raising on a malformed payload.
    KernelPhase._record_geak_candidate(phase, None)
    assert phase.shared_state.kernel_integrate_attempts == {}


# --------------------------------------------------------------------------
# The top-level ``collective`` breakdown section
# --------------------------------------------------------------------------


def test_a_lane_that_never_ran_yields_an_empty_section() -> None:
    # No ``collective_attempts`` and no ``last_collective`` means the lane
    # never ran; the section must be omitted (``{}``), not a hollow envelope.
    assert collect_collective({}) == {}
    assert collect_collective({"collective_attempts": [], "last_collective": {}}) == {}


def test_malformed_lane_fields_are_treated_as_absent() -> None:
    # A corrupted ``state.json`` (wrong types) must not raise; it degrades to
    # "the lane never ran" rather than surfacing a stale/garbage envelope.
    assert collect_collective({"collective_attempts": "not-a-list", "last_collective": "not-a-dict"}) == {}


def test_attempts_are_normalized_and_non_dict_rows_are_dropped() -> None:
    section = collect_collective(
        {
            "collective_only_mode": True,
            "collective_attempts": [
                {
                    "collective_attempt_id": "att-1",
                    "kernel_id": "all_reduce_k",
                    "collective_op": "all_reduce",
                    "world_size": "8",
                    "engine": "rccl",
                    "kept": True,
                    "kernel_speedup": "1.5",
                },
                "not-a-dict-row",
            ],
        }
    )
    assert section["only_mode"] is True
    assert len(section["attempts"]) == 1
    row = section["attempts"][0]
    assert row["collective_attempt_id"] == "att-1"
    assert row["world_size"] == 8  # coerced from the string in state.json
    assert row["kept"] is True
    assert row["kernel_speedup"] == 1.5
    assert "last" not in section  # no last_collective was supplied
