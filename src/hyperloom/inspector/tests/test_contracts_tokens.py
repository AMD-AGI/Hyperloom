# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 4 (throughput vs latency, uplift): the benchmark points and the accuracy the Inspector plots
come from the files Hyperloom writes, read under the names Hyperloom writes them.

Each test writes a session artifact with the real producer (see :mod:`.producers`) or compares the Inspector's
names with the producer's schema. A failure means a producer changed what it writes: update the Inspector reader the
failure message names, then this test. A test marked ``xfail(strict=True)`` pins a known break; it starts failing
("XPASS(strict)") once the Inspector is fixed, and then the marker comes off.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hyperloom.inspector import derive
from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract


_BENCH = "derive.py (benchmark_point)"


_ACCURACY = "derive.py (accuracy_results)"


# A measured baseline result, as the baseline executor returns it to its recorder.
_MEASURED = {"status": "succeeded", "output_throughput": 1500.0, "ttft_mean_ms": 100.0, "e2el_mean_ms": 2000.0}
_MEASURED["tpot_mean_ms"] = 10.0


def test_baseline_timeline_event_gives_a_benchmark_point(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    producers.record_baseline(session, "t-base", _MEASURED)
    points = producers.watch(session).streams["benchmark"].items
    assert [(p["kind"], p["output_throughput"]) for p in points] == [("baseline", 1500.0)], contract_message(
        "baseline_event._measurement / event_timeline.build_envelope", "derive.py (baseline_point)", points
    )
    assert points[0]["latency"]["ttft"]["mean"] == 100.0, contract_message("_measurement", "derive.py", points)


def test_baseline_raw_result_is_read_from_its_measure_round(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session.session_paths import runs_dir
    from hyperloom.orchestrator.actions.executors.benchmark_run.records import _MEASURE_ROUND_DIR

    session = producers.new_session(tmp_path)
    raw = runs_dir(session, "baseline", "t-base") / _MEASURE_ROUND_DIR / "benchmark_vllm_1" / "inferencex_result.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"output_throughput": 1.0}), encoding="utf-8")
    found = producers.watch(session)._baseline_raw("t-base")
    assert found == {"output_throughput": 1.0}, contract_message("records._MEASURE_ROUND_DIR", "watcher.py", found)


def _result_path(session: Path, task_id: str, name: str, *below: str) -> Path:
    from hyperloom.inference_optimizer.session.session_paths import runs_dir
    from hyperloom.orchestrator.actions.executors._suite.rounds import RUN_DIR_PREFIX
    from hyperloom.orchestrator.actions.executors.grid._grid_base import _safe, variant_slot

    # explore.py names its slot f"v{idx:02d}_{_safe(gv.name)}" inline; run_grid puts the variant's own slot below it.
    slot = runs_dir(session, "explore", task_id) / f"v00_{_safe(name)}"
    return variant_slot(slot, 0, name).joinpath(*below, f"{RUN_DIR_PREFIX}0", "inferencex_result.json")


def test_benchmark_result_written_by_the_dataset_client_is_a_point(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors.grid._grid_base import _safe

    session = producers.new_session(tmp_path)
    path = _result_path(session, "t-exp", "tp=2 mnbt")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(producers.dataset_client_result()), encoding="utf-8")
    points = producers.watch(session).streams["benchmark"].items
    where = "derive.py (benchmark_point)"
    assert len(points) == 1 and points[0]["variant"] == _safe("tp=2 mnbt"), contract_message(
        "_grid_base._safe / variant_slot", where, points
    )
    missing = [f"{s}_{k}" for k, stats in points[0]["latency"].items() for s, v in stats.items() if v is None]
    assert not missing and points[0]["ts"] == 1004.0, contract_message(
        "dataset_client.summarize / _result", where, f"latency stats not read: {missing}"
    )


def _discarded_rounds() -> list:
    return [
        pytest.param("grid._grid_base", "WARMUP_ROUND_DIR", id="grid-warmup"),
        pytest.param("grid._grid_base", "MN_WARMUP_DIR", id="grid-mn-warmup"),
    ]


@pytest.mark.parametrize(("module", "constant"), _discarded_rounds())
def test_discarded_rounds_are_not_points(tmp_path: Path, module: str, constant: str) -> None:
    import importlib

    discarded = getattr(importlib.import_module(f"hyperloom.orchestrator.actions.executors.{module}"), constant)
    session = producers.new_session(tmp_path)
    path = _result_path(session, "t-exp", "v", discarded)
    parts = path.relative_to(session / "runs").parts
    point = derive.benchmark_point(parts, producers.dataset_client_result(), None)
    assert point is None, contract_message(f"{module}.{constant}", _BENCH, f"{'/'.join(parts)} is plotted")


def test_baseline_rounds_are_left_to_the_baseline_point() -> None:
    from hyperloom.orchestrator.actions.executors.benchmark_run import records

    for name in (records._WARMUP_ROUND_DIR, records._MEASURE_ROUND_DIR, records._ACCURACY_ROUND_DIR):
        parts = ("baseline", "t-base", name, "benchmark_vllm_1", "inferencex_result.json")
        assert derive.benchmark_point(parts, producers.dataset_client_result(), None) is None, contract_message(
            "benchmark_run.records", _BENCH, f"baseline round {name} plotted twice"
        )


def test_accuracy_recorded_by_shared_state_is_read(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    state = producers.new_state(model_name="m")
    producers.promote_baseline(state, {"accuracy": 0.968})
    before = time.time()
    producers.accept_variant(state, "fp8_kv", 0.962)
    producers.accept_variant(state, "ungated", None)
    state.save(session)
    accuracy = producers.watch(session).snapshot()["summary"]["accuracy"]
    assert accuracy["baseline"] == 0.968, contract_message(
        "SharedState.baseline_accuracy (_promote_anchor_run_facts)", _ACCURACY, accuracy
    )
    (variant,) = accuracy["variants"]
    assert variant["variant"] == "fp8_kv" and variant["value"] == 0.962, contract_message(
        "SharedState.record_explore_accepted (explore_search.accepted name / accuracy)", _ACCURACY, accuracy
    )
    assert before - 1 <= variant["t"] <= time.time() + 1, contract_message(
        "explore_search.accepted ts", _ACCURACY, variant
    )
