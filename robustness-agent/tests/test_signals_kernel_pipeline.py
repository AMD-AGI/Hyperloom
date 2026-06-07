# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""F1-F5 kernel-pipeline health signal tests."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.kernel_pipeline import (
    KernelPipelineConfig,
    RayPendingDetector,
    evaluate_kernel_pipeline_signals,
)
from robustness_agent.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(),
        inbox=[],
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# F1 — RayPendingDetector
# ---------------------------------------------------------------------------

def test_f1_ray_pending_fires_after_consecutive_ticks():
    det = RayPendingDetector(KernelPipelineConfig(
        pending_count_threshold=1, min_pending_ticks=3,
    ))
    data = SourceData(local_ray={"healthy": True, "pending_tasks": 50})
    det.evaluate(_ctx(), data)
    det.evaluate(_ctx(), data)
    out = det.evaluate(_ctx(), data)
    sym = next(s for s in out if s.name == "ray_pending_starvation")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["pending_tasks"] == 50
    assert sym.evidence["consecutive_ticks"] == 3


def test_f1_ray_pending_silent_below_threshold():
    det = RayPendingDetector(KernelPipelineConfig(
        pending_count_threshold=10, min_pending_ticks=3,
    ))
    data = SourceData(local_ray={"healthy": True, "pending_tasks": 5})
    for _ in range(5):
        out = det.evaluate(_ctx(), data)
    assert all(s.name != "ray_pending_starvation" for s in out)


def test_f1_ray_pending_resets_on_clear_tick():
    det = RayPendingDetector(KernelPipelineConfig(
        pending_count_threshold=1, min_pending_ticks=3,
    ))
    busy = SourceData(local_ray={"healthy": True, "pending_tasks": 50})
    clear = SourceData(local_ray={"healthy": True, "pending_tasks": 0})
    det.evaluate(_ctx(), busy)
    det.evaluate(_ctx(), busy)
    det.evaluate(_ctx(), clear)  # reset
    det.evaluate(_ctx(), busy)
    det.evaluate(_ctx(), busy)
    out = det.evaluate(_ctx(), busy)
    # 3 consecutive busies since reset → fire.
    assert any(s.name == "ray_pending_starvation" for s in out)


def test_f1_ray_pending_silent_when_ray_unhealthy():
    """Ray-head dead is its own signal; don't pile pending on top."""
    det = RayPendingDetector(KernelPipelineConfig(
        pending_count_threshold=1, min_pending_ticks=2,
    ))
    data = SourceData(local_ray={"healthy": False, "reason": "exit=1"})
    det.evaluate(_ctx(), data)
    out = det.evaluate(_ctx(), data)
    assert all(s.name != "ray_pending_starvation" for s in out)


# ---------------------------------------------------------------------------
# F2 — geak_budget_starvation
# ---------------------------------------------------------------------------

def test_f2_geak_budget_starvation_fires():
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "rms_norm", "backend": "geak",
             "report_text": "SIGTERM during select_patch round 3",
             "microbench_speedup": None},
            {"kernel_id": "rms_norm", "backend": "geak",
             "report_text": "killed by deadline; partial output",
             "microbench_speedup": None},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "geak_budget_starvation")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["kernel_id"] == "rms_norm"
    assert sym.evidence["attempt_count"] == 2


def test_f2_silent_when_marker_missing():
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "rms_norm", "backend": "geak",
             "report_text": "completed successfully", "microbench_speedup": 1.18},
            {"kernel_id": "rms_norm", "backend": "geak",
             "report_text": "no improvement found"},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    assert all(s.name != "geak_budget_starvation" for s in out)


# ---------------------------------------------------------------------------
# F3 (auth_proxy_unhealthy) was retired with the auth-proxy itself; the
# AMD primus-safe gateway now accepts ``x-api-key`` directly, so the
# proxy is no longer needed and the matching detector was removed.
# ---------------------------------------------------------------------------


def test_unreachable_local_servers_do_not_resurrect_auth_proxy_signal():
    """Sanity-check: no symptom named ``auth_proxy_unhealthy`` survives."""
    data = SourceData(local_server_health=[
        {"url": "http://127.0.0.1:8000/health", "reachable": False,
         "status": "error"},
    ])
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    assert all(s.name != "auth_proxy_unhealthy" for s in out)


# ---------------------------------------------------------------------------
# F4 — cursor_auth_storm
# ---------------------------------------------------------------------------

def test_f4_cursor_auth_storm_fires():
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "k1", "backend": "cursor",
             "report_text": "HTTP 401 Unauthorized: api key invalid"},
            {"kernel_id": "k2", "backend": "cursor",
             "report_text": "Primus.00009: 401"},
            {"kernel_id": "k3", "backend": "cursor",
             "report_text": "401: Unauthorized"},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "cursor_auth_storm")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["hit_count"] == 3


def test_f4_silent_below_threshold():
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "k1", "backend": "cursor",
             "report_text": "401 Unauthorized"},
            {"kernel_id": "k2", "backend": "cursor",
             "report_text": "completed"},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    assert all(s.name != "cursor_auth_storm" for s in out)


# ---------------------------------------------------------------------------
# F5 — kernel_opt_no_progress
# ---------------------------------------------------------------------------

def test_f5_kernel_opt_no_progress_fires():
    """Three kernels each tried by ≥2 backends → all REVERT/PARTIAL → fire."""
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "k1", "backend": "geak",
             "report_text": "partial", "microbench_speedup": 1.05},
            {"kernel_id": "k1", "backend": "claude",
             "report_text": "no measurable speedup",
             "microbench_speedup": 1.02},
            {"kernel_id": "k2", "backend": "geak",
             "report_text": "no speedup", "microbench_speedup": 1.0},
            {"kernel_id": "k2", "backend": "codex",
             "report_text": "no speedup", "microbench_speedup": 1.01},
            {"kernel_id": "k3", "backend": "geak",
             "report_text": "partial", "microbench_speedup": 1.1},
            {"kernel_id": "k3", "backend": "claude",
             "report_text": "no measurable speedup",
             "microbench_speedup": 1.05},
        ],
        "recent_integrate": [],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "kernel_opt_no_progress")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["kernel_count"] >= 3


def test_f5_silent_when_one_kernel_has_keep():
    """Even one ``microbench_speedup >= 1.2`` row excludes the kernel."""
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "k1", "backend": "geak", "microbench_speedup": 1.05},
            {"kernel_id": "k1", "backend": "claude", "microbench_speedup": 1.0},
            {"kernel_id": "k2", "backend": "geak", "microbench_speedup": 1.5},  # KEEP
            {"kernel_id": "k2", "backend": "claude", "microbench_speedup": 1.0},
            {"kernel_id": "k3", "backend": "geak", "microbench_speedup": 1.05},
            {"kernel_id": "k3", "backend": "claude", "microbench_speedup": 1.02},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    # Only 2 bad kernels (k1, k3) → below default threshold of 3.
    assert all(s.name != "kernel_opt_no_progress" for s in out)


def test_f5_silent_when_too_few_backends_per_kernel():
    """Single-backend attempt — not enough evidence to claim "all fail"."""
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "k1", "backend": "geak", "microbench_speedup": 1.0},
            {"kernel_id": "k2", "backend": "geak", "microbench_speedup": 1.0},
            {"kernel_id": "k3", "backend": "geak", "microbench_speedup": 1.0},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    assert all(s.name != "kernel_opt_no_progress" for s in out)


def test_f5_integrate_keep_excludes_kernel():
    """A KEEP'd integrate decision counts as progress even without oob hit."""
    data = SourceData(local_decision_audit={
        "oob_attempts": [
            {"kernel_id": "k1", "backend": "geak", "microbench_speedup": 1.0},
            {"kernel_id": "k1", "backend": "claude", "microbench_speedup": 1.0},
        ],
        "recent_integrate": [
            {"kernel_id": "k1", "decision": "KEEP", "gain_pct": 5.0},
        ],
    })
    out = evaluate_kernel_pipeline_signals(_ctx(), data)
    # Only k1, and it has a KEEP. Below threshold.
    assert all(s.name != "kernel_opt_no_progress" for s in out)
