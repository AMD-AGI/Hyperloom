# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the predictor request body: the fields the service reads, and where they come from."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.orchestrator.predictor import evidence
from hyperloom.orchestrator.predictor.payload import build_request as _build_request
from hyperloom.orchestrator.predictor.sidecars import load_sidecars
from hyperloom.orchestrator.predictor.source_sites import load_source_sites
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.trace_analysis import _analysis_md


def build_request(state: SharedState, *, session_id: str) -> dict:
    """The body the pump sends: source sites and sidecars loaded from the session's own analysis run."""
    path = (state.last_trace_analyze if isinstance(state.last_trace_analyze, dict) else {}).get("analysis_md_path")
    return _build_request(state, session_id=session_id, sites=load_source_sites(path), sidecars=load_sidecars(path))


_P_HEADER = _analysis_md.P_ITEM_COLUMNS + "\n|---|---|---|---|---|---|---|---|---|---|---|"

REPORT = f"""# TraceLens Analysis

## Executive Summary

| Metric | Value |
|--------|-------|
| Total GPU Time | 263.980 ms |
| GPU Busy % | 71.40% |
| GPU Idle % | 28.60% |
| Top Bottleneck Category | gemm |
| Op-attribution Coverage | \u2014 |

## System-Level Signals

| Signal | Value | Note |
|--------|-------|------|
| Exposed communication | 3.20% | - |

### P1: gemm kernels

{_P_HEADER}
| aten::mm | 1200.5 | 40.10% | 30.0% | 1440 | 120.0 | 55.0% | compute | (4096, 4096)<br>(4096, 11008) | tuned_gemm.py | x |
| aten::addmm | 300.0 | 10.00% | 8.0% | 200 | 80.0 | 40.0% | memory | \u2014 | linear.py | y |

### P2: attention kernels

{_P_HEADER}
| flash_attn | 800.0 | 25.00% | 20.0% | 64 | 10.0 | 70.0% | memory | (1, 32, 8192, 128) | attention.py | z |
"""


def _state(tmp_path: Path) -> SharedState:
    report_path = tmp_path / "tracelens" / "analysis.md"
    report_path.parent.mkdir()
    report_path.write_text(REPORT)
    entries = [
        {"kernel_id": "k1", "name": "aten::mm", "source_file": "/x/tuned_gemm.py", "method": "symbol_index",
         "source_line": 395, "source_function": "torch_gemm"},
        {"kernel_id": "k2", "name": "flash_attn", "source_file": "/x/attention.py", "method": "unresolved"},
    ]  # fmt: skip
    (tmp_path / "kernel_source_resolution.json").write_text(json.dumps({"schema_version": "1.1.0", "entries": entries}))
    state = SharedState()
    state.model_name, state.framework, state.gpu_type, state.tp = "Qwen3-8B", "vllm", "mi355x", 8
    state.model_info = {"model_type": "qwen3", "num_hidden_layers": 36, "vocab_size": 151936}
    state.isl, state.osl, state.conc, state.max_model_len = 8192, 1024, 64, 9216
    state.baseline_tput, state.current_best = 4000.0, {"tput": 5000.0}
    state.optimization_stack = [
        {"candidate_extra_server_args": "--kv-cache-dtype fp8", "extra_envs": {}, "tput": 5000.0}
    ]
    state.roofline_snapshots = [
        {
            "roofline_mem_ceiling_tok_per_sec": 9000,
            "roofline_cmp_ceiling_tok_per_sec": 12000,
            "roofline_bound_kind": "memory",
            "achieved_tok_per_sec": 5000,
            "gap_to_roofline_pct": 44.4,
            "perfmodel_breakdown": {"hbm_bw_gbps": 8000, "ops": [{"bound": "memory"}, {"bound": "compute"}]},
        }
    ]
    state.last_trace_analyze = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "analysis_md_text": REPORT,
        "analysis_md_path": str(report_path),
        "hot_kernels_top15": [
            {"kernel_id": "k1", "name": "aten::mm", "gpu_pct": 40.1, "efficiency_percent": 55.0,
             "arithmetic_intensity": 120.0, "bound_type": "compute", "kernel_category": "gemm",
             "source_file": "tuned_gemm.py"},
            {"kernel_id": "k2", "name": "flash_attn", "gpu_pct": 25.0, "bound_type": "unknown",
             "source_file": "attention.py"},
        ],
    }  # fmt: skip
    return state


def test_the_body_carries_exactly_the_fields_the_service_reads(tmp_path):
    body = build_request(_state(tmp_path), session_id="s1")

    assert set(body) == {"schema", "session_id", "identification", "workload", "phase", "performance", "evidence"}
    assert body["schema"].startswith("hyperloom.predictor_request.")
    assert set(body["identification"]) == {
        "model_name", "model_class", "gpu_type", "framework", "framework_version", "precision", "tp", "ep",
        "model_info",
    }  # fmt: skip
    assert body["identification"]["model_info"] == {
        "model_type": "qwen3", "attention_type": None, "num_hidden_layers": 36, "num_experts": None,
        "hidden_size": None, "head_dim": None,
    }  # fmt: skip
    assert body["workload"] == {"isl": 8192, "osl": 1024, "conc": 64, "max_model_len": 9216}
    assert set(body["phase"]) == {"phase", "phase_reason", "phase_elapsed_seconds"}
    assert body["phase"]["phase"] == "EXPLORE"
    assert set(body["performance"]) == {
        "baseline_tput", "current_best_tput", "cumulative_gain_validated", "keep_threshold_pct", "optimization_stack",
    }  # fmt: skip
    assert body["performance"]["optimization_stack"] == [
        {"candidate_extra_server_args": "--kv-cache-dtype fp8", "extra_envs": None, "tput": 5000.0}
    ]
    json.dumps(body)


def test_evidence_blocks_come_from_the_report_the_roofline_and_the_source_artifact(tmp_path):
    ev = build_request(_state(tmp_path), session_id="s1")["evidence"]

    assert ev["profile_available"] and 0 <= ev["profile_age_sec"] < 60
    assert ev["window"] == {
        "total_gpu_time_ms": 263.98,
        "gpu_busy_pct": 71.4,
        "gpu_idle_pct": 28.6,
        "exposed_comm_pct": 3.2,
    }
    assert ev["operators"] == {
        "top_bottleneck_category": "gemm",
        "attribution_pct": None,
        "category_pct": {"gemm": 50.1, "attention": 25.0},
        "top3_cumulative_pct": 75.1,
    }
    assert ev["roofline"]["n_ops_total"] == 2 and ev["roofline"]["n_ops_memory_bound"] == 1
    assert ev["roofline"]["peak_achievable_tflops"] is None
    mm, attn = ev["hot_kernels"]
    assert mm["args"] == "(4096, 4096), (4096, 11008)" and mm["call_count"] == 1440 and mm["time_us"] == 1200.5
    assert (mm["source_file"], mm["source_line"], mm["source_function"]) == ("/x/tuned_gemm.py", 395, "torch_gemm")
    # An unresolved entry leaves the row's own file and no frame; an unclassified bound is dropped.
    assert (attn["source_file"], attn["source_line"], attn["bound_type"]) == ("attention.py", None, None)


def test_a_source_artifact_of_another_major_is_ignored(tmp_path):
    state = _state(tmp_path)
    (tmp_path / "kernel_source_resolution.json").write_text(json.dumps({"schema_version": "2.0.0", "entries": []}))
    mm = build_request(state, session_id="")["evidence"]["hot_kernels"][0]
    assert (mm["source_file"], mm["source_line"]) == ("tuned_gemm.py", None)


AGENTIC_REPORT = """# Qwen3 (LLM) - MI355X Standalone Analysis
<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 340.93 ms |
| Compute % | 97.28% |
| Idle % | 0.51% |
| Exposed Communication % | 0.92% |
| Top Bottleneck Category | InferenceAttention (39.55%) |

### \U0001f534 P1: Prefix-prefill forward kernel runs far below the memory roofline
"""

ANALYSIS_JSON = {
    "report_info": {"mode": "agentic"},
    "executive_summary": {
        "metrics": {"total_time_ms": 340.93, "compute_pct": 97.28, "idle_pct": 0.51, "exposed_communication_pct": 0.92}
    },
}

SUMMARY_TASKS = [
    {"kernel_id": "k1", "name": "aten::mm", "kernel_category": "GEMM", "gpu_pct": 40.1, "call_count": 1440,
     "duration_us": 1200.5},
    {"kernel_id": "k2", "name": "flash_attn", "kernel_category": "InferenceAttention", "gpu_pct": 25.0,
     "call_count": 64, "duration_us": 800.0},
    {"kernel_id": "k3", "name": "rms_norm", "kernel_category": "LayerNorm", "gpu_pct": 4.0, "call_count": 72,
     "duration_us": 50.0},
    {"kernel_id": "k4", "name": "aten::addmm", "kernel_category": "GEMM", "gpu_pct": 10.0, "call_count": 200,
     "duration_us": 300.0},
]  # fmt: skip


def _write_sidecars(report_dir: Path) -> None:
    (report_dir / "analysis.json").write_text(json.dumps(ANALYSIS_JSON))
    (report_dir / "summary.json").write_text(json.dumps({"tasks": SUMMARY_TASKS}))


def test_a_tracelens_route_report_takes_window_operators_and_counts_from_the_sidecars(tmp_path):
    state = _state(tmp_path)
    report_path = Path(state.last_trace_analyze["analysis_md_path"])
    report_path.write_text(AGENTIC_REPORT)
    state.last_trace_analyze["analysis_md_text"] = AGENTIC_REPORT
    _write_sidecars(report_path.parent)

    ev = build_request(state, session_id="s1")["evidence"]

    assert ev["window"] == {"total_gpu_time_ms": 340.93, "gpu_busy_pct": 99.49, "gpu_idle_pct": 0.51,
                            "exposed_comm_pct": 0.92}  # fmt: skip
    assert ev["operators"] == {
        "top_bottleneck_category": "GEMM",
        "attribution_pct": None,
        "category_pct": {"GEMM": 50.1, "SDPA": 25.0, "Normalization": 4.0},
        "top3_cumulative_pct": 79.1,
    }
    mm, attn = ev["hot_kernels"]
    assert (mm["call_count"], mm["time_us"], mm["args"]) == (1440, 1200.5, None)
    assert (attn["call_count"], attn["time_us"]) == (64, 800.0)


def test_a_report_in_the_bypass_layout_wins_over_the_sidecars(tmp_path):
    state = _state(tmp_path)
    _write_sidecars(Path(state.last_trace_analyze["analysis_md_path"]).parent)

    ev = build_request(state, session_id="s1")["evidence"]

    assert ev["window"]["total_gpu_time_ms"] == 263.98
    assert ev["operators"]["category_pct"] == {"gemm": 50.1, "attention": 25.0}


def test_missing_or_malformed_sidecars_add_nothing(tmp_path):
    report_dir = tmp_path / "tracelens"
    report_dir.mkdir()
    assert load_sidecars(report_dir / "analysis.md") == {}
    (report_dir / "analysis.json").write_text("{not json")
    (report_dir / "summary.json").write_text(json.dumps({"tasks": {"k1": {}}}))
    assert load_sidecars(report_dir / "analysis.md") == {}
    assert load_sidecars(None) == {}


def test_no_profile_sends_only_the_flag():
    assert build_request(SharedState(), session_id="")["evidence"] == {"profile_available": False}


def test_the_parser_reads_the_renderer_layout():
    columns = [cell.strip() for cell in _analysis_md.P_ITEM_COLUMNS.strip().strip("|").split("|")]
    assert len(columns) == evidence._P_ITEM_CELLS
    assert [columns[i] for i in (0, 1, 2, 4, 8)] == ["Operation", "Time (us)", "GPU%", "Count", "Args"]
    source = inspect.getsource(_analysis_md)
    for label in ("Total GPU Time", "GPU Busy %", "GPU Idle %", "Top Bottleneck Category", "Op-attribution Coverage",
                  "Exposed communication"):  # fmt: skip
        assert f"| {label} |" in source
