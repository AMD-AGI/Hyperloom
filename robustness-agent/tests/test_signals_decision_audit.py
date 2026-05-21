"""Unit tests for the G1-G7 decision-audit signals."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.decision_audit import (
    DecisionAuditConfig,
    evaluate_decision_audit_signals,
)
from robustness_agent.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1.0,
    )


def _integrate_entry(
    *,
    kernel_id: str = "k1",
    decision: str = "KEEP",
    gain_pct: float | None = 5.0,
    patch_size_bytes: int | None = 1024,
    dispatched_count: int | None = 42,
    patch_path: str = "/tmp/p.diff",
    result_path: str = "/tmp/result.json",
) -> dict:
    return {
        "kernel_id": kernel_id,
        "decision": decision,
        "gain_pct": gain_pct,
        "base_tput": 100.0,
        "new_tput": 105.0,
        "patch_path": patch_path,
        "patch_size_bytes": patch_size_bytes,
        "dispatched_count": dispatched_count,
        "result_path": result_path,
        "mtime": 1.0,
    }


# ---------------------------------------------------------------------------
# G1 empty_patch_kept
# ---------------------------------------------------------------------------

def test_empty_patch_kept_fires_on_zero_patch_with_keep():
    audit = {
        "recent_integrate": [
            _integrate_entry(patch_size_bytes=0, gain_pct=0.5),
        ],
    }
    data = SourceData(local_decision_audit=audit)
    out = evaluate_decision_audit_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "empty_patch_kept")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["patch_size_bytes"] == 0
    assert sym.evidence["kernel_id"] == "k1"


def test_empty_patch_kept_silent_on_revert():
    audit = {
        "recent_integrate": [
            _integrate_entry(patch_size_bytes=0, decision="REVERT"),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit)
    )
    assert all(s.name != "empty_patch_kept" for s in out)


def test_empty_patch_kept_silent_when_patch_size_unknown():
    audit = {
        "recent_integrate": [
            _integrate_entry(patch_size_bytes=None),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit)
    )
    assert all(s.name != "empty_patch_kept" for s in out)


# ---------------------------------------------------------------------------
# G2 decision_threshold_violated
# ---------------------------------------------------------------------------

def test_decision_threshold_violated_fires_on_sub_threshold_keep():
    audit = {
        "recent_integrate": [
            _integrate_entry(gain_pct=0.07, dispatched_count=100),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "decision_threshold_violated")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["gain_pct"] == 0.07
    assert sym.evidence["min_keep_gain_pct"] == 1.0


def test_decision_threshold_violated_silent_above_threshold():
    audit = {
        "recent_integrate": [
            _integrate_entry(gain_pct=2.5, dispatched_count=100),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "decision_threshold_violated" for s in out)


# ---------------------------------------------------------------------------
# G3 kernel_dispatch_bypassed
# ---------------------------------------------------------------------------

def test_dispatch_bypassed_fires_when_dispatched_count_zero():
    audit = {
        "recent_integrate": [
            _integrate_entry(dispatched_count=0, gain_pct=2.0),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "kernel_dispatch_bypassed")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["dispatched_count"] == 0


def test_dispatch_bypassed_fires_on_near_zero_gain_with_missing_evidence():
    """K5 case — Dolphin-34B KEEP +0.07% gain, no dispatch evidence."""
    audit = {
        "recent_integrate": [
            _integrate_entry(
                dispatched_count=None,
                gain_pct=0.07,
                patch_size_bytes=512,
            ),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "kernel_dispatch_bypassed")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["dispatched_count"] is None


def test_dispatch_bypassed_silent_when_dispatched_positive():
    audit = {
        "recent_integrate": [
            _integrate_entry(dispatched_count=50, gain_pct=0.1),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "kernel_dispatch_bypassed" for s in out)


def test_dispatch_bypassed_silent_on_meaningful_gain_without_evidence():
    """Big gain with no evidence: not bypassed, just missing audit data."""
    audit = {
        "recent_integrate": [
            _integrate_entry(dispatched_count=None, gain_pct=8.0),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "kernel_dispatch_bypassed" for s in out)


# ---------------------------------------------------------------------------
# G4 kernel_negative_delta_kept
# ---------------------------------------------------------------------------

def test_negative_delta_kept_fires():
    """Qwen2.5-32B-AWQ case: ``kernels_optimized=6, delta=-0.169``."""
    audit = {
        "ci_metrics": {
            "kernels_optimized": 6,
            "optimized_kernel_delta_pct": -0.169,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "kernel_negative_delta_kept")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["optimized_kernel_delta_pct"] == -0.169


def test_negative_delta_silent_when_positive():
    audit = {
        "ci_metrics": {
            "kernels_optimized": 3,
            "optimized_kernel_delta_pct": 5.0,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "kernel_negative_delta_kept" for s in out)


def test_negative_delta_silent_when_no_kernels_optimized():
    audit = {
        "ci_metrics": {
            "kernels_optimized": 0,
            "optimized_kernel_delta_pct": -0.5,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "kernel_negative_delta_kept" for s in out)


# ---------------------------------------------------------------------------
# G5 ci_metrics_baseline_zero
# ---------------------------------------------------------------------------

def test_baseline_zero_fires_when_no_status_marker():
    audit = {
        "ci_metrics": {
            "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
            "baseline_tok_per_gpu": 0.0,
            "optimized_tok_per_gpu": 0.0,
            "gain_pct": 0.0,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "ci_metrics_baseline_zero")
    assert sym.severity is SymptomSeverity.HIGH


def test_baseline_zero_silent_when_status_baseline_failed():
    audit = {
        "ci_metrics": {
            "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
            "baseline_tok_per_gpu": 0.0,
            "optimized_tok_per_gpu": 0.0,
            "gain_pct": 0.0,
            "status": "baseline_failed",
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "ci_metrics_baseline_zero" for s in out)


def test_baseline_zero_silent_when_baseline_present():
    audit = {
        "ci_metrics": {
            "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
            "baseline_tok_per_gpu": 1500.0,
            "optimized_tok_per_gpu": 1800.0,
            "gain_pct": 20.0,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "ci_metrics_baseline_zero" for s in out)


# ---------------------------------------------------------------------------
# G6 ci_metrics_schema_drift
# ---------------------------------------------------------------------------

def test_schema_drift_fires_on_legacy_field_names():
    audit = {
        "ci_metrics": {
            # Legacy variant: uses ``baseline_throughput`` instead of
            # ``baseline_tok_per_gpu``.
            "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
            "baseline_throughput": 1500.0,
            "optimized_throughput": 1800.0,
            "gain_pct": 20.0,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "ci_metrics_schema_drift")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert "baseline_throughput" in sym.evidence["legacy_fields"]


def test_schema_drift_fires_on_missing_field():
    audit = {
        "ci_metrics": {
            "model": "X", "framework": "sglang", "gpu": "MI300X",
            # missing tp / baseline_tok_per_gpu / optimized_tok_per_gpu / gain_pct
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "ci_metrics_schema_drift")
    assert "tp" in sym.evidence["missing"]


def test_schema_drift_silent_on_canonical_schema():
    audit = {
        "ci_metrics": {
            "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
            "baseline_tok_per_gpu": 1500.0,
            "optimized_tok_per_gpu": 1800.0,
            "gain_pct": 20.0,
        },
        "ci_metrics_path": "/p/ci_metrics.json",
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "ci_metrics_schema_drift" for s in out)


# ---------------------------------------------------------------------------
# G7 oob_no_harness
# ---------------------------------------------------------------------------

def test_oob_no_harness_fires_on_expected_only_report():
    audit = {
        "oob_attempts": [
            {
                "kernel_id": "gemm_a8w8",
                "backend": "oob_claude",
                "report_text": (
                    "Hoisted loop-invariant. expected speedup ~1.1x. "
                    "No measurement performed."
                ),
                "microbench_speedup": None,
                "ts": "2026-05-11T09:30:19",
            },
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    sym = next(s for s in out if s.name == "oob_no_harness")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["kernel_id"] == "gemm_a8w8"


def test_oob_no_harness_silent_when_microbench_present():
    audit = {
        "oob_attempts": [
            {
                "kernel_id": "gemm_a8w8",
                "backend": "oob_claude",
                "report_text": "expected speedup ~1.1x — measured 1.098x.",
                "microbench_speedup": 1.098,
                "ts": "2026-05-11T09:30:19",
            },
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    assert all(s.name != "oob_no_harness" for s in out)


def test_oob_no_harness_groups_per_kernel_id():
    """Multiple offending attempts on the same kernel collapse to one."""
    audit = {
        "oob_attempts": [
            {
                "kernel_id": "gemm_a8w8",
                "backend": "oob_claude",
                "report_text": "expected speedup ~1.1x",
                "microbench_speedup": None,
                "ts": "2026-05-11T09:30:19",
            },
            {
                "kernel_id": "gemm_a8w8",
                "backend": "oob_codex",
                "report_text": "expected: 1.15x",
                "microbench_speedup": None,
                "ts": "2026-05-11T10:00:00",
            },
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit),
    )
    g7 = [s for s in out if s.name == "oob_no_harness"]
    assert len(g7) == 1


# ---------------------------------------------------------------------------
# Probe-disabled / no-data short-circuit
# ---------------------------------------------------------------------------

def test_silent_when_local_decision_audit_empty():
    out = evaluate_decision_audit_signals(_ctx(), SourceData())
    assert out == []


def test_silent_when_audit_is_not_a_dict():
    data = SourceData()
    data.local_decision_audit = []  # type: ignore[assignment]
    out = evaluate_decision_audit_signals(_ctx(), data)
    assert out == []


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

def test_custom_min_keep_gain_pct_applies():
    cfg = DecisionAuditConfig(min_keep_gain_pct=5.0)
    audit = {
        "recent_integrate": [
            _integrate_entry(gain_pct=3.0, dispatched_count=100),
        ],
    }
    out = evaluate_decision_audit_signals(
        _ctx(), SourceData(local_decision_audit=audit), config=cfg,
    )
    sym = next(s for s in out if s.name == "decision_threshold_violated")
    assert sym.evidence["min_keep_gain_pct"] == 5.0
