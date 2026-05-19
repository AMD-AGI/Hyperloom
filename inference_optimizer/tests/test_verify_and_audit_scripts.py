"""Roofline-v2 C6: verify + audit script smoke tests.

These scripts (``scripts.verify_roofline_v2`` /
``scripts.audit_roofline_decisions``) are pure operator tooling — no
production code path depends on them. The tests below pin:

* Exit-code contract for verify (0 = PASS ≥+5%, 2 = PARTIAL >0% but
  <5%, 1 = FAIL ≤0% or missing state.json).
* Text rendering includes the key fields the operator needs to
  eyeball Qwen3-32B C7 results (cumulative gain delta, action_seq,
  pruned_families with analyzer cross-reference).
* JSON output is valid and contains the documented top-level keys.
* Both scripts degrade gracefully when state.json is missing
  (verify: error rendered + non-zero exit; audit: error rendered).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.scripts import audit_roofline_decisions, verify_roofline_v2


# ---------------------------------------------------------------------------
# State.json fixtures
# ---------------------------------------------------------------------------
def _baseline_state() -> dict:
    """Minimal state.json mimic — cumulative gain ≈ 0, no roofline run."""
    return {
        "session_id": "baseline-1",
        "cumulative_gain_validated": 0.1,
        "optimization_stack": [
            {"kind": "params", "variant_name": "v1", "gain_pct": 0.1},
        ],
        "pruned_families": [],
        "last_roofline_analysis": {},
        "last_select_kernels": {
            "trace_input": "/tmp/trace", "roofline_snapshot_id": 1,
        },
        "profile_attempts": [{"ts": "2026-05-19T10:00:00+00:00"}],
        "cumulative_gain_validated_ts": "2026-05-19T10:30:00+00:00",
        "baseline_attempts": [{"ts": "2026-05-19T09:30:00+00:00"}],
    }


def _exp_state(*, gain: float = 6.5) -> dict:
    """Mimic an experiment session that achieved a roofline-driven gain."""
    return {
        "session_id": "exp-1",
        "cumulative_gain_validated": gain,
        "optimization_stack": [
            {"kind": "params",
             "variant_name": "two_batch_overlap",
             "gain_pct": 3.2},
            {"kind": "comm_optimization",
             "variant_name": "aiter_allreduce",
             "gain_pct": 2.1},
            {"kind": "params",
             "variant_name": "moe_a2a_deepep",
             "gain_pct": 1.2},
        ],
        "pruned_families": [
            {"family": "kernel_opt",
             "reason": "compute saturated 91.2%",
             "source": "orchestration"},
        ],
        "last_roofline_analysis": {
            "snapshot_id": 2,
            "analyzed_at_iso": "2026-05-19T10:15:00+00:00",
            "analyzed_at_gain_pct": 3.2,
            "based_on_analysis_md": "/tmp/analysis.md",
            "primary_bottleneck": "comm",
            "bottleneck_distribution": {"comm": 0.45, "compute": 0.30},
            "suggested_prunes": [
                {"family": "kernel_opt",
                 "reason": "compute saturated 91.2%",
                 "confidence": "high"},
                {"family": "deep_kernel_analysis",
                 "reason": "comm dominates",
                 "confidence": "medium"},
            ],
            "suggested_next_actions": [
                {"kind": "params",
                 "rationale": "try comm-overlap flags",
                 "priority": "high"},
                {"kind": "comm_optimization",
                 "rationale": "rccl Allreduce dominates",
                 "priority": "high"},
            ],
            "reprofile_recommended": False,
            "reprofile_reason": "",
        },
        "last_select_kernels": {
            "trace_input": "/tmp/trace", "roofline_snapshot_id": 2,
        },
        "profile_attempts": [
            {"ts": "2026-05-19T09:35:00+00:00"},
            {"ts": "2026-05-19T10:05:00+00:00"},
        ],
        "cumulative_gain_validated_ts": "2026-05-19T10:30:00+00:00",
        "baseline_attempts": [{"ts": "2026-05-19T09:30:00+00:00"}],
    }


def _write_state(session_dir: Path, state: dict) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text(
        json.dumps(state), encoding="utf-8",
    )


# ===========================================================================
# verify_roofline_v2 — exit codes + rendered output
# ===========================================================================
def test_verify_pass_exit_zero_when_delta_meets_5pct(tmp_path, capsys):
    _write_state(tmp_path / "baseline", _baseline_state())
    _write_state(tmp_path / "exp", _exp_state(gain=6.5))

    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "baseline"),
        "--exp", str(tmp_path / "exp"),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "VERDICT: PASS" in out
    assert "+6.400" in out  # delta gain
    assert "two_batch_overlap" in out  # exp action_seq visible
    assert "kernel_opt" in out  # exp pruned visible


def test_verify_partial_exit_two_when_delta_positive_below_5pct(tmp_path, capsys):
    _write_state(tmp_path / "baseline", _baseline_state())
    _write_state(tmp_path / "exp", _exp_state(gain=2.0))

    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "baseline"),
        "--exp", str(tmp_path / "exp"),
    ])

    assert rc == 2
    out = capsys.readouterr().out
    assert "VERDICT: PARTIAL" in out


def test_verify_fail_exit_one_when_delta_non_positive(tmp_path, capsys):
    _write_state(tmp_path / "baseline", _baseline_state())
    _write_state(tmp_path / "exp", _exp_state(gain=-1.0))

    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "baseline"),
        "--exp", str(tmp_path / "exp"),
    ])

    assert rc == 1
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out


def test_verify_handles_missing_state_json(tmp_path, capsys):
    _write_state(tmp_path / "baseline", _baseline_state())
    rc = verify_roofline_v2.main([
        "--baseline", str(tmp_path / "baseline"),
        "--exp", str(tmp_path / "missing-session-dir"),
    ])
    assert rc == 1  # delta=0-0.1=negative because exp gain=0 → FAIL
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "state.json not found" in out


def test_verify_json_flag_emits_structured_summary(tmp_path, capsys):
    _write_state(tmp_path / "baseline", _baseline_state())
    _write_state(tmp_path / "exp", _exp_state(gain=6.5))

    verify_roofline_v2.main([
        "--baseline", str(tmp_path / "baseline"),
        "--exp", str(tmp_path / "exp"),
        "--json",
    ])

    captured = capsys.readouterr()
    parsed = json.loads(captured.err)
    assert "baseline" in parsed and "exp" in parsed and "delta" in parsed
    assert parsed["delta"]["cumulative_gain_validated_pct"] == pytest.approx(6.4)
    assert parsed["exp"]["pruned_families"] == ["kernel_opt"]


def test_verify_renders_optimization_stack_and_pruned_families(tmp_path, capsys):
    _write_state(tmp_path / "baseline", _baseline_state())
    _write_state(tmp_path / "exp", _exp_state(gain=6.5))

    verify_roofline_v2.main([
        "--baseline", str(tmp_path / "baseline"),
        "--exp", str(tmp_path / "exp"),
    ])

    out = capsys.readouterr().out
    # exp action_seq includes the three promoted variants
    assert "params:two_batch_overlap" in out
    assert "comm_optimization:aiter_allreduce" in out
    assert "params:moe_a2a_deepep" in out
    # baseline has just one entry
    assert "params:v1" in out


# ===========================================================================
# audit_roofline_decisions — text + JSON modes
# ===========================================================================
def test_audit_renders_full_block(tmp_path, capsys):
    _write_state(tmp_path / "session", _exp_state(gain=6.5))

    rc = audit_roofline_decisions.main([
        "--session", str(tmp_path / "session"),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "session_dir: " in out
    assert "cumulative_gain_validated_pct: 6.500" in out
    assert "snapshot_id=2" in out
    assert "primary=comm" in out
    assert "comm=45%" in out
    assert "suggested_prunes:" in out
    assert "HIGH" in out and "kernel_opt" in out
    # Pruned family table with analyzer cross-reference
    assert "[from-analyzer:HIGH]" in out
    # Action stack visible
    assert "two_batch_overlap" in out and "+3.20%" in out
    # Consumption stats
    assert "prune advice consumed: 1/2" in out
    assert "next-action advice followed: 2/2" in out


def test_audit_handles_missing_state_json(tmp_path, capsys):
    rc = audit_roofline_decisions.main([
        "--session", str(tmp_path / "nope"),
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "state.json not found" in out


def test_audit_handles_empty_roofline_cache(tmp_path, capsys):
    _write_state(tmp_path / "session", _baseline_state())
    rc = audit_roofline_decisions.main([
        "--session", str(tmp_path / "session"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(not yet run / cache empty)" in out
    assert "(no families pruned)" in out


def test_audit_json_flag_emits_structured_dict(tmp_path, capsys):
    _write_state(tmp_path / "session", _exp_state(gain=6.5))
    audit_roofline_decisions.main([
        "--session", str(tmp_path / "session"),
        "--json",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["has_state_json"] is True
    assert payload["cumulative_gain_validated_pct"] == 6.5
    assert payload["pruned_families_count"] == 1
    assert payload["advice_consumed_count"] == 1
    assert payload["advice_ignored_count"] == 1
    assert payload["next_action_followed_count"] == 2
    assert payload["next_action_ignored_count"] == 0


def test_audit_distinguishes_analyzer_advised_vs_main_llm_prunes(tmp_path, capsys):
    """A prune NOT in the analyzer's suggested_prunes list should be
    tagged [main-llm-only] so the operator can spot main-LLM autonomy."""
    state = _exp_state(gain=4.0)
    # Add a prune that wasn't in suggested_prunes
    state["pruned_families"].append({
        "family": "operator_tuning",
        "reason": "manual-override",
        "source": "orchestration",
    })
    _write_state(tmp_path / "session", state)

    audit_roofline_decisions.main([
        "--session", str(tmp_path / "session"),
    ])
    out = capsys.readouterr().out
    assert "kernel_opt" in out and "[from-analyzer:HIGH]" in out
    assert "operator_tuning" in out and "[main-llm-only]" in out
