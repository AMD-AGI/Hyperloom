# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernels that cost a launch rather than their work, and the neighbour each could fold into."""

from __future__ import annotations

import json

from kernelforge import foldable_launches as fl


def _step(name: str, ts: float, dur: float) -> dict:
    return {"cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur, "pid": 2}


def _kernel(name: str, ts: float, dur: float) -> dict:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": 2}


def _chain(start: float, count: int, trio: tuple[str, str, str], *, glue_us: float = 4.0) -> list[dict]:
    """``producer -> glue -> consumer`` repeated, as a split-K GEMM and its reduction run."""
    producer, glue, consumer = trio
    events: list[dict] = []
    ts = start
    for _ in range(count):
        events.append(_kernel(producer, ts, 14.0))
        events.append(_kernel(glue, ts + 14.0, glue_us))
        events.append(_kernel(consumer, ts + 14.0 + glue_us, 12.0))
        ts += 40.0
    return events


def _write(tmp_path, events):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def test_the_glue_between_a_producer_and_its_consumer_is_named_with_both(tmp_path):
    events = [_step("step[VERIFY bs=3]", 0.0, 1000.0), *_chain(1.0, 20, ("split_k", "reduce", "consumer"))]

    (row,) = [r for r in fl.foldable_launches(_write(tmp_path, events))["VERIFY bs=3"] if r["kernel"] == "reduce"]

    assert row["launches_per_step"] == 20
    assert row["us_per_call"] == 4.0
    assert row["after"] == {"name": "split_k", "share": 1.0}
    assert row["before"] == {"name": "consumer", "share": 1.0}


def test_a_kernel_doing_real_work_is_not_glue(tmp_path):
    # The producer runs 14 us a call: above the floor a launch costs, so it is the work, not the glue.
    events = [_step("step[VERIFY bs=3]", 0.0, 1000.0), *_chain(1.0, 20, ("split_k", "reduce", "consumer"))]

    named = [r["kernel"] for r in fl.foldable_launches(_write(tmp_path, events))["VERIFY bs=3"]]

    assert "reduce" in named
    assert "split_k" not in named and "consumer" not in named


def test_a_kernel_launched_a_few_times_is_not_what_the_step_spends_itself_on(tmp_path):
    events = [_step("step[VERIFY bs=3]", 0.0, 1000.0), *_chain(1.0, 3, ("split_k", "reduce", "consumer"))]

    assert fl.foldable_launches(_write(tmp_path, events)) == {}


def test_glue_without_one_neighbour_is_listed_but_names_none(tmp_path):
    events = [_step("step[VERIFY bs=3]", 0.0, 2000.0)]
    ts = 1.0
    for i in range(20):
        events.append(_kernel(f"producer_{i % 5}", ts, 14.0))
        events.append(_kernel("glue", ts + 14.0, 4.0))
        ts += 40.0

    (row,) = fl.foldable_launches(_write(tmp_path, events))["VERIFY bs=3"]

    assert row["kernel"] == "glue"
    assert row["after"] is None


def test_a_name_the_categoriser_misreads_is_still_found(tmp_path):
    """``_sparse_mla_reduce_kernel`` reads as attention to the categoriser; the measurement finds it anyway."""
    events = [
        _step("step[VERIFY bs=3]", 0.0, 2000.0),
        *_chain(1.0, 20, ("_sparse_mla_split_k_kernel", "_sparse_mla_reduce_kernel", "_batched_gemm_a8w8")),
    ]

    (row,) = [
        r
        for r in fl.foldable_launches(_write(tmp_path, events))["VERIFY bs=3"]
        if r["kernel"] == "_sparse_mla_reduce_kernel"
    ]

    assert row["category"] == "attention"
    assert row["after"]["name"] == "_sparse_mla_split_k_kernel"


def test_classes_are_counted_apart(tmp_path):
    events = [
        _step("step[VERIFY bs=3]", 0.0, 1000.0),
        *_chain(1.0, 20, ("split_k", "reduce", "consumer")),
        _step("step[EXTEND bs=1 toks=900]", 2000.0, 1000.0),
        *_chain(2001.0, 20, ("prefill_gemm", "prefill_glue", "prefill_consumer")),
    ]

    out = fl.foldable_launches(_write(tmp_path, events))

    assert [r["kernel"] for r in out["VERIFY bs=3"]] == ["reduce"]
    assert [r["kernel"] for r in out["EXTEND bs=1"]] == ["prefill_glue"]


def test_a_capture_without_steps_or_a_missing_one_reports_nothing(tmp_path):
    assert fl.foldable_launches(_write(tmp_path, [_kernel("k", 1.0, 4.0)])) == {}
    assert fl.foldable_launches("") == {}
    assert fl.foldable_launches(None) == {}
    assert fl.foldable_launches(tmp_path / "absent.json") == {}


def test_the_rendering_names_what_to_fold_into_and_what_cannot(tmp_path):
    events = [_step("step[VERIFY bs=3]", 0.0, 1000.0), *_chain(1.0, 20, ("split_k", "reduce", "consumer"))]

    body = "\n".join(fl.render_foldable_launches(fl.foldable_launches(_write(tmp_path, events))))

    assert "### VERIFY bs=3" in body
    assert "after `split_k` (100%)" in body
    assert "before `consumer` (100%)" in body
    assert "folded into either of them without changing what is computed" in body


def test_nothing_renders_without_candidates():
    assert fl.render_foldable_launches({}) == []
    assert fl.render_foldable_launches(None) == []
