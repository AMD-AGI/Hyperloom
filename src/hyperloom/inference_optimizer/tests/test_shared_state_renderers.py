# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for SharedState prompt-rendering helpers (populated branches)."""

from __future__ import annotations

from hyperloom.orchestrator.state.shared_state import SharedState


def test_format_discovered_flags():
    st = SharedState()
    assert "first backends" in st._format_discovered_flags()
    st.discovered_flags = {
        "sglang": {"backend_flags": ["a", "b"], "param_flags": ["c"]},
        "bad": "not-a-dict",
    }
    out = st._format_discovered_flags()
    assert "sglang:backend=2/param=1" in out


def test_format_variant_line():
    line = SharedState._format_variant_line(
        {
            "name": "v1",
            "gain_pct": 5.0,
            "tput": 120.0,
            "extra_server_args": "--x",
            "extra_envs": {"E": "1"},
        }
    )
    assert "v1" in line
    assert "+5.00%" in line
    assert "tput=120.0" in line
    assert "E=1" in line
    line2 = SharedState._format_variant_line({"name": "v2"})
    assert "no_meas" in line2
    assert "(no-flag)" in line2


def test_format_variant_line_no_measurement_carries_reason():
    line = SharedState._format_variant_line(
        {
            "name": "slow",
            "reason": "killed_overtime",
            "wall_clock_ratio_vs_baseline": 2.06,
        }
    )
    assert "no_meas" in line
    assert "killed_overtime" in line
    assert "2.06x" in line
    # A measured row needs no reason: the gain already carries the verdict.
    measured = SharedState._format_variant_line(
        {"name": "low", "gain_pct": 0.4, "reason": "gain_below_threshold"},
    )
    assert "gain_below_threshold" not in measured


def test_enrich_with_tested_gain():
    entry = {"fingerprint": "fp1"}
    tested = {"fp1": {"gain_pct": 3.0, "result": {"output_throughput": 99.0}}}
    out = SharedState._enrich_with_tested_gain(entry, tested)
    assert out["gain_pct"] == 3.0
    assert out["tput"] == 99.0
    # already-populated returns unchanged
    full = {"gain_pct": 1.0, "tput": 1.0}
    assert SharedState._enrich_with_tested_gain(full, tested) is full
    # missing fingerprint snapshot -> unchanged
    assert SharedState._enrich_with_tested_gain({"fingerprint": "none"}, tested) == {"fingerprint": "none"}


def test_format_search_state():
    assert SharedState._format_search_state(None) == "(none)"
    search = {
        "cursor": 3,
        "accepted": [{"name": "a", "fingerprint": "f1"}],
        "rejected": [{"name": "r", "gain_pct": -1.0}],
        "tested": {"f1": {"gain_pct": 4.0}},
    }
    out = SharedState._format_search_state(search)
    assert "cursor=3" in out
    assert "accepted:" in out
    assert "rejected (last 15):" in out
    assert "killed_overtime" not in out


def test_format_search_state_head_reports_killed_overtime():
    search = {
        "cursor": 8,
        "tested": {"f1": {}, "f2": {}},
        "last_round": {"round_id": 3, "killed_overtime": ["f1", "f2"]},
    }
    out = SharedState._format_search_state(search)
    assert "killed_overtime(last_round)=2" in out
    # A round with no kill must not add noise to the head line.
    search["last_round"] = {"round_id": 4, "killed_overtime": []}
    assert "killed_overtime" not in SharedState._format_search_state(search)
    search["last_round"] = "not-a-dict"
    assert "killed_overtime" not in SharedState._format_search_state(search)


def test_format_optimization_stack():
    st = SharedState()
    assert st._format_optimization_stack() == "(none)"
    st.optimization_stack = [{"action": "explore", "variant_name": "v1"}, "bad"]
    parts = st._format_optimization_stack()
    assert "explore:v1" in parts


def test_format_last_trace_analyze():
    st = SharedState()
    assert st._format_trace_analyze_blob(st.last_trace_analyze) == "(none)"
    st.last_trace_analyze = {
        "trace_input": "t",
        "candidates_path": "c",
        "hot_kernels_top15": [{"kernel_id": "k1"}],
        "reusable_native_kernel_ids": ["k1"],
        "trace_health_warnings": [
            {"code": "high_idle", "idle_pct": 40, "threshold_pct": 20},
            {"code": "crash", "returncode": 1},
        ],
    }
    out = st._format_trace_analyze_blob(st.last_trace_analyze)
    assert "k1" in out
    assert "high_idle" in out


def test_format_trace_analyze_skipped_kernels():
    st = SharedState()
    blob = {
        "trace_input": "t",
        "candidates_path": "c",
        "hot_kernels_top15": [],
        "skipped_kernels_top": [
            {"kernel_id": "s1", "name": "n1", "skip_reason": "tiny"},
        ],
    }
    out = st._format_trace_analyze_blob(blob)
    assert "skipped_kernels_top" in out


def test_format_analysis_md_full(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE", "1")
    st = SharedState()
    assert "no TraceLens snapshot" in st._format_analysis_md_full()
    st.last_trace_analyze = {
        "analysis_md_text": "# Analysis\nbody",
        "roofline_snapshot_id": 5,
        "roofline_baseline_gain_at_snapshot": 12.5,
    }
    out = st._format_analysis_md_full()
    assert "snapshot #5" in out
    assert "12.50%" in out
    assert "body" in out


def test_format_last_action_failures_multiline_and_log_path():
    st = SharedState()
    st.last_action_failures = [
        {
            "action": "explore",
            "error_class": "server_init_dead",
            "ts": "2026-01-01T00:00:00+00:00",
            "error_excerpt": (
                "RuntimeError: Worker failed with error\n"
                "  'mla_gluon[bh16bn128] requires batch_size=1, got 512'\n"
                "Engine core initialization failed.\n"
            ),
            "workspace": "/runs/explore/t1/benchmark_sglang_001",
            "stderr_log_path": "/runs/explore/t1/server.log",
            "variant_name": "fp8_kv",
        }
    ]
    out = st._format_last_action_failures()
    assert "mla_gluon" in out
    assert "log=/runs/explore/t1/server.log" in out
    assert "variant=fp8_kv" in out


def test_format_variant_line_shows_error_class_and_reason():
    line = SharedState._format_variant_line(
        {
            "name": "fp8_kv",
            "gain_pct": None,
            "extra_server_args": "--kv-cache-dtype fp8",
            "error_class": "server_init_dead",
            "reason": "mla_gluon requires batch_size=1",
        }
    )
    assert "err=server_init_dead" in line
    assert "reason=mla_gluon" in line


def test_format_last_action_failures_suffix_when_over_window():
    st = SharedState()
    for i in range(12):
        st.record_action_failure(
            action="explore",
            task_id=f"t-{i}",
            result={"error_class": "server_init_dead", "error": f"crash {i}"},
        )
    out = st._format_last_action_failures()
    assert "[+2 earlier failures]" in out
