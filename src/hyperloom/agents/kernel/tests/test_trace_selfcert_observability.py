###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Coverage for what the self-certification record says about its own basis.

The subject here is observability rather than analysis: whether a capture sidecar was read at all, whether a
metric states the terms it was computed from, and whether the verdict carries the continuous values behind its
categorical answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The tools tree supports both a package import and a bare sys.path import; the bare one has to resolve for the
# package import to succeed, the same way ``profile.py`` arranges it before certifying.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from hyperloom.agents.kernel.tools.trace_selfcert import (  # noqa: E402
    build_verdict,
    certify_capture_sidecars,
    certify_trace_dir,
    effective_thresholds,
)

_SOURCE_EVENTS = [
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200, "Input Dims": [[4, 4]]}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 7, "External id": 200}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1300, "dur": 200, "args": {"correlation": 7}},
]


def _write(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))
    return path


def _capture_events(*, with_meta: int, without_meta: int) -> list[dict]:
    events: list[dict] = []
    for i in range(with_meta):
        events.append({"cat": "cpu_op", "name": "aten::mm", "args": {"External id": i, "Input Dims": [[4, 4]]}})
    for i in range(without_meta):
        events.append({"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 1000 + i}})
    events.append({"cat": "kernel", "ph": "X", "name": "gemm", "ts": 10, "dur": 5, "args": {"correlation": 1}})
    return events


def test_capture_sidecars_report_their_operator_metadata_coverage(tmp_path):
    """The sidecars were never opened before; the point of the scan is that this number exists at all."""
    _write(tmp_path / "capture_traces" / "bs_1.trace.json", _capture_events(with_meta=3, without_meta=1))
    _write(tmp_path / "capture_traces" / "bs_2.trace.json", _capture_events(with_meta=1, without_meta=3))

    probe = certify_capture_sidecars(sorted((tmp_path / "capture_traces").glob("*.json")))

    assert probe["files_present"] == 2
    assert probe["files_scanned"] == 2
    assert probe["truncated_scan"] is False
    assert probe["cpu_op_total"] == 8
    assert probe["cpu_op_with_meta"] == 4
    assert probe["op_meta_coverage"] == 0.5
    assert probe["kernel_count"] == 2
    assert probe["files_with_errors"] == 0
    assert [Path(f["path"]).name for f in probe["files"]] == ["bs_1.trace.json", "bs_2.trace.json"]


def test_capture_sidecar_scan_admits_when_it_did_not_read_everything(tmp_path):
    """A partial scan must not read as a complete one, so the count present travels with the count scanned."""
    for i in range(4):
        _write(tmp_path / "capture_traces" / f"bs_{i}.trace.json", _capture_events(with_meta=1, without_meta=0))

    probe = certify_capture_sidecars(sorted((tmp_path / "capture_traces").glob("*.json")), limit=2)

    assert probe["files_present"] == 4
    assert probe["files_scanned"] == 2
    assert probe["truncated_scan"] is True
    assert probe["scan_limit"] == 2


def test_certify_trace_dir_attaches_the_capture_probe(tmp_path):
    _write(tmp_path / "rank0.pt.trace.json", _SOURCE_EVENTS)
    _write(tmp_path / "capture_traces" / "bs_1.trace.json", _capture_events(with_meta=1, without_meta=1))

    record = certify_trace_dir(tmp_path, framework="sglang")

    probe = record["trace_dir_level"]["capture_sidecar_probe"]
    assert probe["files_present"] == 1
    assert probe["op_meta_coverage"] == 0.5
    # The sidecar measurement reaches the verdict as a stated fact.
    assert record["verdict"]["measures"]["capture_op_meta_coverage"] == 0.5


def test_a_capture_only_directory_still_gets_its_sidecars_read(tmp_path):
    """The case with no usable trace is the one where knowing what the sidecars hold matters most."""
    _write(tmp_path / "capture_traces" / "bs_1.trace.json", _capture_events(with_meta=2, without_meta=0))

    record = certify_trace_dir(tmp_path, framework="sglang")

    assert record["trace_dir_level"]["capture_sidecar_probe"]["cpu_op_with_meta"] == 2


def test_metrics_state_the_terms_they_were_computed_from(tmp_path):
    """A ratio whose basis is missing cannot be judged, so the record carries the basis beside the ratio."""
    _write(tmp_path / "rank0.pt.trace.json", _SOURCE_EVENTS)

    rank = certify_trace_dir(tmp_path, framework="sglang")["rank_level"][0]

    assert rank["parse"]["aggregation_scope"] is not None
    assert rank["parse"]["truncated"] is False
    assert rank["parse"]["truncation_reason"] == ""
    assert rank["attribution"]["gpu_kernel_sum_ms"] is not None
    assert rank["attribution"]["attributed_gpu_ms"] is not None
    assert rank["attribution"]["op_meta_basis"] == {"cpu_op_total": 1, "cpu_op_with_meta": 1}
    # The boolean is a comparison, so it travels with the number it compared against.
    assert rank["density"]["graph_under_recorded_threshold"] == effective_thresholds()["graph_launch_coverage_max"]


def _verdict(**overrides):
    kwargs = {
        "parse_ok": True,
        "kernel_count": 12,
        "capture_fragment": False,
        "attributed_pct": 91.0,
        "step_roots_sufficient": True,
        "forecast_modelled": True,
        "viable_modes": ["decode_only"],
        "idle": {},
        "graph_under_recorded": False,
        "thresholds": effective_thresholds(),
    }
    kwargs.update(overrides)
    return build_verdict(**kwargs)


def test_verdict_severity_ranks_a_wrong_answer_above_a_doubtful_one():
    """An analysis that runs clean and concludes wrongly is worse than one that announces its own doubt."""
    assert _verdict()["severity"] == "ok"
    assert _verdict(graph_under_recorded=True)["severity"] == "silently_wrong"
    assert _verdict(graph_under_recorded=None, forecast_modelled=False)["severity"] == "warn"
    assert _verdict(parse_ok=False)["severity"] == "blocked"


def test_verdict_carries_the_numbers_behind_its_categorical_answer():
    """A boolean cannot tell a near miss from an order-of-magnitude miss; the measures block can."""
    verdict = _verdict(graph_launch_coverage=0.12, op_meta_coverage=0.4, capture_op_meta_coverage=0.02)

    measures = verdict["measures"]
    assert measures["graph_launch_coverage"] == 0.12
    assert measures["graph_launch_coverage_max"] == effective_thresholds()["graph_launch_coverage_max"]
    assert measures["op_meta_coverage"] == 0.4
    assert measures["capture_op_meta_coverage"] == 0.02
    assert measures["attributed_pct"] == 91.0
    assert measures["kernel_count"] == 12
    assert measures["viable_mode_count"] == 1


def test_measures_do_not_change_any_decision():
    """The continuous values are reported, never compared: the same inputs must still resolve the same way."""
    plain = _verdict()
    annotated = _verdict(graph_launch_coverage=0.01, op_meta_coverage=0.0, capture_op_meta_coverage=0.0)

    for key in ("usable_by", "decode_conclusions_valid", "silently_wrong", "blocking_reasons", "warnings"):
        assert plain[key] == annotated[key], key
