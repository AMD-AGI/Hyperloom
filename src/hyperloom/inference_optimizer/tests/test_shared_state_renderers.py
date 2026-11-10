# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for SharedState prompt-rendering helpers (populated branches)."""

from __future__ import annotations

from hyperloom.orchestrator.state.shared_state import SharedState


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
        }
    )
    assert "no_meas" in line
    assert "killed_overtime" in line
    # A measured row needs no reason: the gain already carries the verdict.
    measured = SharedState._format_variant_line(
        {"name": "low", "gain_pct": 0.4, "reason": "gain_below_threshold"},
    )
    assert "gain_below_threshold" not in measured


def test_format_variant_line_refused_variant_is_not_unmeasured():
    """A REVERT has no ``gain_pct`` by design; its ``graded_gain_pct`` is what it measured."""
    line = SharedState._format_variant_line(
        {
            "name": "spec",
            "outcome": "REVERT",
            "gain_pct": None,
            "graded_gain_pct": 7.2,
            "graded_objective": "e2e_norm_intvty_p50",
            "tput": 1074.9,
            "reason": "p90_regressed (intvty 85.1->91.3 tput 168970.6->169070.5)",
        }
    )
    assert "no_meas" not in line
    assert "+7.20%" in line
    assert "[e2e_norm_intvty_p50]" in line
    assert "p90_regressed" in line
    # The refused gain never displaces a recorded one, and a bool is not a measurement.
    assert "+1.00%" in SharedState._format_variant_line({"name": "k", "gain_pct": 1.0, "graded_gain_pct": 9.0})
    assert "no_meas" in SharedState._format_variant_line({"name": "b", "graded_gain_pct": True})


def test_format_search_state_enumerates_every_tested_entry():
    """No truncation: a variant missing from the render reads as untried."""
    search = {"cursor": 40, "tested": {f"fp{i:02d}": {"outcome": "REJECT", "name": f"v{i}"} for i in range(40)}}
    out = SharedState._format_search_state(search)
    assert "tested=40" in out
    for i in range(40):
        assert f"fp{i:02d}" in out


def test_format_search_state_puts_keeps_first():
    search = {
        "tested": {
            "fp-reject": {"outcome": "REJECT", "name": "r"},
            "fp-keep": {"outcome": "KEEP", "name": "k"},
        }
    }
    body = SharedState._format_search_state(search).splitlines()
    rows = [ln for ln in body if ln.strip().startswith("fp-")]
    assert rows[0].strip().startswith("fp-keep")


def test_format_search_state():
    assert SharedState._format_search_state(None) == "(none)"
    assert SharedState._format_search_state({"tested": {}}).strip().startswith("cursor=0")
    search = {
        "cursor": 3,
        "accepted": [{"name": "a", "fingerprint": "f1"}],
        "rejected": [{"name": "r", "gain_pct": -1.0}],
        "tested": {"f1": {"outcome": "KEEP", "name": "a", "gain_pct": 4.0}},
    }
    out = SharedState._format_search_state(search)
    assert "cursor=3  accepted=1  rejected=1  tested=1" in out
    assert "f1" in out and "KEEP" in out


def test_format_optimization_stack():
    st = SharedState()
    assert st._format_optimization_stack() == "(none)"
    st.optimization_stack = [{"action": "explore", "variant_name": "v1"}, "bad"]
    parts = st._format_optimization_stack()
    assert "explore:v1" in parts


def test_prompt_summary_renders_current_best_without_stack_or_measurement():
    st = SharedState()
    st.current_best = {
        "variant_name": "k23",
        "tput": 2049.9,
        "optimization_stack": [{"action": "explore", "variant_name": "stacked"}],
        "measurement": {"launch_evidence_path": "/runs/launch_evidence.json"},
    }
    line = next(ln for ln in st.to_prompt_summary().splitlines() if ln.startswith("current_best="))
    assert "k23" in line and "2049.9" in line
    assert "stacked" not in line and "launch_evidence" not in line


def _launch_flags_line(state: SharedState) -> str:
    return next(ln for ln in state.to_prompt_summary().splitlines() if ln.startswith("current_launch_flags"))


def test_prompt_summary_renders_the_flags_the_current_best_launched_with():
    """Orchestration cannot read the session config, so the summary is its only account of the running launch."""
    st = SharedState()
    assert _launch_flags_line(st) == "current_launch_flags(already passed by the running stack)=(unknown)"
    st.current_best_measurement = {
        "launch_evidence": {"requested_server_args": "--max-num-seqs 64 --kv-cache-dtype fp8 --max-num-seqs 128"}
    }
    assert _launch_flags_line(st) == (
        "current_launch_flags(already passed by the running stack)=--max-num-seqs 128 --kv-cache-dtype fp8"
    )


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


def test_format_analysis_md_pointer():
    st = SharedState()
    assert "no TraceLens snapshot" in st._format_analysis_md_pointer()
    st.last_trace_analyze = {
        "analysis_md_text": "# Analysis\nbody",
        "roofline_snapshot_id": 5,
        "roofline_baseline_gain_at_snapshot": 12.5,
    }
    out = st._format_analysis_md_pointer()
    assert "snapshot #5" in out
    assert "12.50%" in out


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
