# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for SharedState prompt-rendering helpers (populated branches)."""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


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
    line = SharedState._format_variant_line({
        "name": "v1", "gain_pct": 5.0, "tput": 120.0,
        "extra_server_args": "--x", "extra_envs": {"E": "1"},
    })
    assert "v1" in line
    assert "+5.00%" in line
    assert "tput=120.0" in line
    assert "E=1" in line
    # no-measurement / no-flag branch
    line2 = SharedState._format_variant_line({"name": "v2"})
    assert "no_meas" in line2
    assert "(no-flag)" in line2


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


def test_format_backend_winners_history():
    st = SharedState()
    assert "no explore rounds" in st._format_backend_winners_history()
    st.backend_winners_history = [
        {
            "round_id": f"r{i}", "action": "explore", "base_tput": 100.0,
            "best": {"name": "w", "gain_pct": 2.0},
            "winners": [{"name": "w", "gain_pct": 2.0, "extra_server_args": "--x"}],
        }
        for i in range(7)
    ]
    # one round (within the last-5 window) with no winners
    st.backend_winners_history[-1]["winners"] = []
    out = st._format_backend_winners_history()
    assert "earlier rounds elided" in out
    assert "no winners this round" in out


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
    assert "rejected (last 5):" in out


def test_format_optimization_stack():
    st = SharedState()
    assert st._format_optimization_stack() == "(none)"
    st.optimization_stack = [{"action": "explore", "variant_name": "v1"}, "bad"]
    parts = st._format_optimization_stack()
    assert "explore:v1" in parts


def test_format_last_trace_analyze():
    st = SharedState()
    assert st._format_last_trace_analyze() == "(none)"
    st.last_trace_analyze = {
        "trace_input": "t", "candidates_path": "c",
        "hot_kernels_top15": [{"kernel_id": "k1"}],
        "reusable_native_kernel_ids": ["k1"],
        "trace_health_warnings": [
            {"code": "high_idle", "idle_pct": 40, "threshold_pct": 20},
            {"code": "crash", "returncode": 1},
        ],
    }
    out = st._format_last_trace_analyze()
    assert "k1" in out
    assert "high_idle" in out


def test_format_trace_analyze_skipped_kernels():
    st = SharedState()
    blob = {
        "trace_input": "t", "candidates_path": "c",
        "hot_kernels_top15": [],
        "skipped_kernels_top": [
            {"kernel_id": "s1", "name": "n1", "skip_reason": "tiny"},
        ],
    }
    out = st._format_trace_analyze_blob(blob)
    assert "skipped_kernels_top" in out


def test_format_analysis_md_full():
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


def test_format_last_sweep():
    st = SharedState()
    assert st._format_last_sweep() == "(none)"
    st.last_sweep = {"grid_size": 4}
    assert "best=(none)" in st._format_last_sweep()
    st.last_sweep = {
        "grid_size": 4,
        "best_overall": {"name": "b", "tput": 100.0, "conc": 8,
                         "isl": 1024, "osl": 512},
    }
    out = st._format_last_sweep()
    assert "grid_size=4" in out
    assert "best=b" in out
