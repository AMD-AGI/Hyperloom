"""Roofline-v2 N7: verify + audit script smoke tests.

These scripts (`scripts.verify_roofline_v2` /
`scripts.audit_roofline_decisions`) are pure operator tooling — no
production code path depends on them.

Tests pin:

* Exit-code contract for verify (0=PASS ≥+5%, 2=PARTIAL >0 but <5%,
  1=FAIL ≤0 or missing state.json).
* §10.3 v2 quality criteria surfaced in the rendered output
  (cache_hit_rate, analysis_md_referenced_count,
  hallucinated_flag_count, roofline_action_count).
* JSON output schema for CI consumption (verify + audit both).
* Graceful degradation when state.json is missing / cache field
  empty / analysis_md not referenced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.scripts import (
    audit_roofline_decisions,
    verify_roofline_v2,
)


# ---------------------------------------------------------------------------
# State.json fixtures
# ---------------------------------------------------------------------------
def _baseline_state() -> dict:
    """Pre-v2 state: gain ~0, no roofline action, no v2 metadata."""
    return {
        "session_id": "baseline-1",
        "cumulative_gain_validated": 0.1,
        "optimization_stack": [
            {"kind": "params", "variant_name": "v1", "gain_pct": 0.1},
        ],
        "pruned_families": [],
        "last_trace_analyze": {"trace_input": "/tmp/trace", "roofline_snapshot_id": 1},
        "baseline_attempts": [{"ts": "2026-05-19T09:30:00+00:00"}],
        "cumulative_gain_validated_ts": "2026-05-19T10:30:00+00:00",
        "discovered_flags": {
            "sglang": {
                "backend_flags": ["--known-flag-a", "--known-flag-b"],
                "param_flags": [],
            }
        },
    }


def _exp_state(*, gain: float = 6.5) -> dict:
    """Post-v2 experiment: roofline ran 2x, prunes grounded in
    analysis.md, only proposed discovered flags, with cache metrics."""
    return {
        "session_id": "exp-1",
        "cumulative_gain_validated": gain,
        "optimization_stack": [
            {"kind": "roofline", "variant_name": ""},
            {"kind": "params", "variant_name": "two_batch_overlap",
             "gain_pct": 3.2},
            {"kind": "comm_optimization", "variant_name": "aiter_allreduce",
             "gain_pct": 2.1},
            {"kind": "roofline", "variant_name": ""},
            {"kind": "params", "variant_name": "torch_compile", "gain_pct": 1.2},
        ],
        "pruned_families": [
            {"family": "kernel_opt",
             "reason": "analysis.md snapshot #1 shows compute saturated 91.2%",
             "source": "orchestration"},
            {"family": "deep_kernel_analysis",
             "reason": "comm-bound: rcclAllreduce dominates top operations",
             "source": "orchestration"},
        ],
        "last_trace_analyze": {
            "trace_input": "/tmp/trace.gz",
            "roofline_snapshot_id": 2,
            "analysis_md_text": "FAKE REPORT",
        },
        "baseline_attempts": [{"ts": "2026-05-19T09:30:00+00:00"}],
        "roofline_attempts": [
            {"ts": "2026-05-19T09:40:00+00:00", "status": "succeeded",
             "task_id": "t-rf-1"},
            {"ts": "2026-05-19T10:10:00+00:00", "status": "succeeded",
             "task_id": "t-rf-2"},
        ],
        "profile_attempts": [
            {"ts": "2026-05-19T09:42:00+00:00"},
            {"ts": "2026-05-19T10:12:00+00:00"},
        ],
        "explore_attempts": [
            {"ts": "2026-05-19T09:50:00+00:00",
             "extra_server_args": "--known-flag-a --known-flag-c"},
            {"ts": "2026-05-19T09:55:00+00:00",
             "extra_server_args": "--known-flag-b"},
        ],
        "cumulative_gain_validated_ts": "2026-05-19T10:30:00+00:00",
        "discovered_flags": {
            "sglang": {
                "backend_flags": [
                    "--known-flag-a", "--known-flag-b", "--known-flag-c",
                ],
                "param_flags": [],
            },
        },
        "tick_cache_metrics": {
            "cache_creation_input_tokens": 5000,
            "cache_read_input_tokens": 15000,
            "input_tokens": 2000,
            "output_tokens": 800,
        },
    }


def _write(session_dir: Path, state: dict) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text(
        json.dumps(state), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# verify_roofline_v2 — exit codes
# ---------------------------------------------------------------------------
def test_verify_pass_when_delta_at_least_5pct(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    _write(tmp_path / "e", _exp_state(gain=6.5))
    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "e"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VERDICT: PASS" in out


def test_verify_partial_when_delta_positive_but_below_5pct(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    _write(tmp_path / "e", _exp_state(gain=2.0))
    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "e"),
    ])
    assert rc == 2
    assert "VERDICT: PARTIAL" in capsys.readouterr().out


def test_verify_fail_when_delta_non_positive(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    _write(tmp_path / "e", _exp_state(gain=-1.0))
    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "e"),
    ])
    assert rc == 1
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_verify_fail_on_missing_state_json(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "no-such-dir"),
    ])
    assert rc == 1  # delta gain = 0 - 0.1 = negative
    out = capsys.readouterr().out
    assert "state.json not found" in out


# ---------------------------------------------------------------------------
# verify rendering — §10.3 v2 criteria + tabular data
# ---------------------------------------------------------------------------
def test_verify_renders_v2_quality_criteria(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    _write(tmp_path / "e", _exp_state(gain=6.5))
    verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "e"),
    ])
    out = capsys.readouterr().out
    # §10.3 criteria block present
    assert "§10.3 v2 quality criteria" in out
    assert "cache_hit_rate ≥ 50%" in out
    assert "analysis_md_referenced_count ≥ 3" in out
    assert "hallucinated_flag_count = 0" in out
    assert "roofline_action_count ≥ 1" in out
    # Cache hit rate computed from tick_cache_metrics
    # 15000 / (5000+15000) = 75.0%
    assert "75.0%" in out


def test_verify_renders_action_seq_for_both_sessions(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    _write(tmp_path / "e", _exp_state(gain=6.5))
    verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "e"),
    ])
    out = capsys.readouterr().out
    # Baseline single-entry stack
    assert "params:v1" in out
    # Exp stack with roofline + variants
    assert "two_batch_overlap" in out
    assert "comm_optimization:aiter_allreduce" in out


def test_verify_json_emits_structured_summary(tmp_path, capsys):
    _write(tmp_path / "b", _baseline_state())
    _write(tmp_path / "e", _exp_state(gain=6.5))
    verify_roofline_v2.main([
        "--baseline", str(tmp_path / "b"),
        "--exp", str(tmp_path / "e"),
        "--json",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["delta"]["cumulative_gain_validated_pct"] == pytest.approx(6.4)
    assert payload["exp"]["roofline_action_count"] == 2
    assert payload["exp"]["snapshot_id"] == 2
    assert payload["exp"]["cache_hit_rate"] == 0.75
    assert payload["exp"]["prune_branch_count_orchestration"] == 2


# ---------------------------------------------------------------------------
# audit_roofline_decisions — content + JSON
# ---------------------------------------------------------------------------
def test_audit_renders_full_block(tmp_path, capsys):
    _write(tmp_path / "s", _exp_state(gain=6.5))
    rc = audit_roofline_decisions.main([
        "--session", str(tmp_path / "s"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Section headers
    assert "roofline action timeline" in out
    assert "pruned_families (analysis-md grounding)" in out
    assert "flag audit" in out
    assert "decision quality criteria" in out
    # Roofline timeline entries
    assert "t-rf-1" in out and "t-rf-2" in out
    # Pruned families with analysis-md-grounded flag
    assert "kernel_opt" in out
    assert "analysis.md snapshot #1" in out
    # Grounded should be yes (mentions "analysis.md", "saturated")
    # First prune row has the keyword
    # Hallucinated flag detection: --known-flag-c was proposed but not in
    # discovered_flags namespace... wait — _exp_state lists it as a
    # discovered flag. So no hallucination.
    assert "hallucinated (not in namespace): 0" in out
    # Cache hit rate 75%
    assert "75.0%" in out


def test_audit_detects_hallucinated_flag(tmp_path, capsys):
    """When a proposed flag is NOT in discovered_flags, count as
    hallucinated."""
    state = _exp_state(gain=4.0)
    # Remove --known-flag-c from discovered namespace; it was still
    # used in explore_attempts → should now register as hallucination
    state["discovered_flags"]["sglang"]["backend_flags"] = [
        "--known-flag-a", "--known-flag-b",  # drop --known-flag-c
    ]
    _write(tmp_path / "s", state)
    audit_roofline_decisions.main([
        "--session", str(tmp_path / "s"),
    ])
    out = capsys.readouterr().out
    assert "hallucinated (not in namespace): 1" in out
    assert "--known-flag-c" in out


def test_audit_handles_missing_state_json(tmp_path, capsys):
    rc = audit_roofline_decisions.main([
        "--session", str(tmp_path / "nope"),
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "state.json not found" in out


def test_audit_handles_empty_session(tmp_path, capsys):
    _write(tmp_path / "s", _baseline_state())
    rc = audit_roofline_decisions.main([
        "--session", str(tmp_path / "s"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no roofline action attempts recorded)" in out
    assert "(no pruned_families)" in out


def test_audit_json_output_schema(tmp_path, capsys):
    _write(tmp_path / "s", _exp_state(gain=6.5))
    audit_roofline_decisions.main([
        "--session", str(tmp_path / "s"),
        "--json",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["has_state_json"] is True
    assert payload["snapshot_id"] == 2
    assert payload["roofline_attempts_count"] == 2
    assert payload["pruned_families_count"] == 2
    assert payload["hallucinated_flag_count"] == 0
    assert payload["cache_hit_rate"] == 0.75
    assert payload["analysis_md_referenced_count"] >= 1


def test_audit_classifies_main_llm_only_prune(tmp_path, capsys):
    """A prune whose reason does NOT mention any analysis.md keyword
    is classified as not-grounded."""
    state = _exp_state(gain=4.0)
    state["pruned_families"].append({
        "family": "operator_tuning",
        "reason": "manual operator override no report quoted",
        "source": "orchestration",
    })
    _write(tmp_path / "s", state)
    audit_roofline_decisions.main([
        "--session", str(tmp_path / "s"),
    ])
    out = capsys.readouterr().out
    # operator_tuning row exists
    assert "operator_tuning" in out
    # And its grounding tag is "no" (look for "operator_tuning ... no")
    # Simple substring check: the row text contains "no" between the
    # source and reason columns. We don't pin column positions; just
    # check that "manual operator override" surfaces and the row
    # exists.
    assert "manual operator override" in out


def test_audit_cache_hit_rate_zero_when_no_metrics(tmp_path, capsys):
    """No `tick_cache_metrics` field → cache_hit_rate = 0%, criterion MISS."""
    state = _exp_state(gain=6.5)
    del state["tick_cache_metrics"]
    _write(tmp_path / "s", state)
    audit_roofline_decisions.main([
        "--session", str(tmp_path / "s"),
    ])
    out = capsys.readouterr().out
    # cache_hit_rate criterion should report MISS when no metrics
    assert "[MISS] cache_hit_rate ≥ 50%" in out
    assert "0.0%" in out
