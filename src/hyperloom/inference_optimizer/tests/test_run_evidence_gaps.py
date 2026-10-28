# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gaps shaped like a cause, derived from what a run was measured doing."""

from __future__ import annotations

import json

from hyperloom.inference_optimizer.session.session_paths import jit_cache_dir
from hyperloom.orchestrator.state import run_evidence_gaps as reg
from kernelforge.tests.trace_fixtures import eager_step, kernel, step, write_trace


def _jit_leak(session_dir, kernel_name: str, count: int) -> None:
    cache = jit_cache_dir(session_dir) / "sglang_cache_dir"
    for i in range(count):
        entry = cache / f"H{i:032d}"
        entry.mkdir(parents=True)
        (entry / f"{kernel_name}.json").write_text(
            json.dumps({"name": f"{kernel_name}_BLOCK_M_64_GRID_MN_{i * 7 + 1}"}), encoding="utf-8"
        )


def _trace(tmp_path, events, *, name: str = "t.json"):
    return write_trace(tmp_path / name, events)


def _requests(tmp_path, *, ms_per_100k: float, name: str = "ws"):
    artifacts = tmp_path / name / "b1" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "metadata": {"benchmark_phase": "profiling"},
                "metrics": {
                    "input_sequence_length": {"value": isl},
                    "time_to_first_token": {"value": 200.0 + isl / 100_000 * ms_per_100k},
                },
            }
        )
        for isl in range(10_000, 10_000 + 200 * 2_000, 2_000)
    ]
    (artifacts / "profile_export.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / name


def test_a_recompiled_kernel_becomes_a_gap_naming_the_constant(tmp_path):
    _jit_leak(tmp_path, "_fused_fp8_bmm_rope", 40)

    (gap,) = reg.run_evidence_gaps(session_dir=tmp_path, trace_path="", workspace="")

    assert gap["layer"] == "kernel_agent"
    assert gap["source"] == "run_evidence"
    assert gap["provenance"] == "recompile"
    assert gap["severity"] == "high"
    assert gap["domain_hint"] == reg.KERNEL_DOMAIN
    assert "_GRID_MN_" in gap["symptom"] and "40 variants" in gap["symptom"]
    assert "runtime rather than constexpr" in gap["symptom"]


def test_an_eager_step_class_becomes_a_gap_that_says_what_to_establish(tmp_path):
    events = eager_step("step[EXTEND bs=1 toks=900]", start=0.0, count=10)

    (gap,) = reg.run_evidence_gaps(session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace="")

    assert gap["provenance"] == "ungraphed_step"
    assert gap["domain_hint"] == reg.SERVING_DOMAIN
    assert "EXTEND bs=1 runs 100%" in gap["symptom"]
    assert "the two have different" in gap["symptom"]


def test_a_foldable_launch_names_the_neighbour_to_fold_into(tmp_path):
    events = [step("step[VERIFY bs=3]", 0.0, 2000.0)]
    ts = 1.0
    for i in range(60):
        events.append(kernel("split_k", ts, 14.0, correlation=900 + i))
        events.append(kernel("reduce", ts + 14.0, 5.0, correlation=1900 + i))
        ts += 30.0

    gaps = reg.run_evidence_gaps(session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace="")

    (gap,) = [g for g in gaps if g["provenance"] == "foldable_launch"]
    assert "reduce runs 60 times per step" in gap["symptom"]
    assert "always beside split_k (100%)" in gap["symptom"]
    assert gap["severity"] == "medium"


def test_a_cheap_launch_is_not_worth_a_gap(tmp_path):
    """Below the per-step cost floor, folding it away does not pay for the round that would do it."""
    events = [step("step[VERIFY bs=3]", 0.0, 2000.0)]
    ts = 1.0
    for i in range(10):
        events.append(kernel("split_k", ts, 14.0, correlation=900 + i))
        events.append(kernel("reduce", ts + 14.0, 1.0, correlation=1900 + i))
        ts += 30.0

    gaps = reg.run_evidence_gaps(session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace="")

    assert [g for g in gaps if g["provenance"] == "foldable_launch"] == []


def test_a_front_end_cost_becomes_a_gap(tmp_path):
    gaps = reg.run_evidence_gaps(session_dir=tmp_path, trace_path="", workspace=_requests(tmp_path, ms_per_100k=200.0))

    (gap,) = gaps
    assert gap["provenance"] == "front_end_input_cost"
    assert gap["layer"] == "framework"
    assert gap["domain_hint"] == reg.SERVING_DOMAIN
    assert "per 100k input tokens" in gap["symptom"]
    assert "in front of it" in gap["symptom"]


def test_every_gap_names_a_lane_a_phase_reads_and_a_domain_a_specialist_answers_to(tmp_path):
    """A phase asks whether its lane has a gap; a dispatcher asks which specialist owns it. Both have to match."""
    _jit_leak(tmp_path, "_fused_fp8_bmm_rope", 40)
    events = eager_step("step[EXTEND bs=1 toks=900]", start=0.0, count=10)

    gaps = reg.run_evidence_gaps(
        session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace=_requests(tmp_path, ms_per_100k=200.0)
    )

    assert len(gaps) >= 3
    for gap in gaps:
        assert gap["domain_hint"] in {reg.KERNEL_DOMAIN, reg.SERVING_DOMAIN}
        expected = "kernel_agent" if gap["domain_hint"] == reg.KERNEL_DOMAIN else "framework"
        assert gap["layer"] == expected, gap["canonical_id"]


def test_one_census_cannot_bury_the_others(tmp_path):
    events = [step("step[VERIFY bs=3]", 0.0, 100_000.0)]
    ts = 1.0
    for i in range(40):
        events.append(kernel(f"glue_{i}", ts, 5.0, correlation=900 + i))
        ts += 20.0
    # Each distinct glue kernel has one neighbour, so every one of them qualifies.
    gaps = reg.run_evidence_gaps(session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace="")

    assert len([g for g in gaps if g["provenance"] == "foldable_launch"]) <= 3


def test_the_costliest_fold_is_raised_whichever_step_class_the_trace_opens_with(tmp_path):
    """A prefill step opening the capture with four modest folds cannot take every slot from decode."""
    events = [step("step[EXTEND bs=1 toks=900]", 0.0, 2000.0)]
    ts = 1.0
    for _ in range(60):
        for name, dur in (("gemm_e", 14.0), ("glue_a", 5.0), ("glue_b", 5.0), ("glue_c", 5.0), ("glue_d", 5.0)):
            events.append(kernel(name, ts, dur))
            ts += dur
    start = 3000.0
    for _ in range(10):
        events.append(step("step[VERIFY bs=3]", start, 2000.0))
        ts = start + 1.0
        for _ in range(50):
            events.append(kernel("split_k", ts, 14.0))
            events.append(kernel("reduce", ts + 14.0, 5.0))
            ts += 30.0
        start += 2500.0

    gaps = reg.run_evidence_gaps(session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace="")

    folds = [g["symptom"] for g in gaps if g["provenance"] == "foldable_launch"]
    assert len(folds) == 3
    assert folds[0].startswith("In VERIFY bs=3, reduce runs 50 times per step")


def test_one_launch_seen_at_several_batch_widths_is_one_fold(tmp_path):
    """A steady-state capture spans several widths; the same fold must not take a slot per width."""
    events = []
    start = 0.0
    for width in (7, 8, 9):
        events.append(step(f"step[VERIFY bs={width}]", start, 2000.0))
        ts = start + 1.0
        for _ in range(50):
            events += [kernel("split_k", ts, 14.0), kernel("reduce", ts + 14.0, 5.0), kernel("norm", ts + 19.0, 4.0)]
            events += [kernel("gemm", ts + 23.0, 14.0)]
            ts += 40.0
        start += 2500.0

    gaps = reg.run_evidence_gaps(session_dir=tmp_path, trace_path=_trace(tmp_path, events), workspace="")

    folds = [g["symptom"] for g in gaps if g["provenance"] == "foldable_launch"]
    assert len(folds) == 2
    assert sum(" reduce runs 50 times per step" in f for f in folds) == 1
    assert sum(" norm runs 50 times per step" in f for f in folds) == 1
    assert all("at 3 batch width(s)" in f for f in folds)


def test_the_same_evidence_upserts_rather_than_multiplying(tmp_path):
    _jit_leak(tmp_path, "_fused_fp8_bmm_rope", 40)

    first = reg.run_evidence_gaps(session_dir=tmp_path, trace_path="", workspace="")
    second = reg.run_evidence_gaps(session_dir=tmp_path, trace_path="", workspace="")

    assert [g["canonical_id"] for g in first] == [g["canonical_id"] for g in second]


def test_a_run_with_no_evidence_raises_no_gaps(tmp_path):
    assert reg.run_evidence_gaps(session_dir=tmp_path, trace_path="", workspace="") == []
    assert reg.run_evidence_gaps(session_dir=tmp_path, trace_path=tmp_path / "absent", workspace="") == []


def test_one_unreadable_artifact_does_not_cost_the_other_censuses(tmp_path, monkeypatch):
    events = eager_step("step[EXTEND bs=1 toks=900]", start=0.0, count=10)

    def _denied(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(reg, "variant_counts", _denied)

    gaps = reg.run_evidence_gaps(
        session_dir=tmp_path,
        trace_path=_trace(tmp_path, events),
        workspace=_requests(tmp_path, ms_per_100k=200.0),
    )

    provenances = {g["provenance"] for g in gaps}
    assert "recompile" not in provenances
    assert {"ungraphed_step", "front_end_input_cost"} <= provenances
