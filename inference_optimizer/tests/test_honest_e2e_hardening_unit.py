# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the HL_HONEST_E2E v4-parity hardening helpers:
umbrella-flag resolution, VRAM util guard, import-grep source confirmation,
op-fanout de-dup in candidate batching, and umbrella-driven GEAK promotion.

Every behavior is OFF by default, so the "off" assertions below also pin the
byte-identical-to-legacy contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator import kernel_request_handlers as krh


# -- _honest_flag (umbrella + per-fix override) ---------------------------
def test_honest_flag_default_off(monkeypatch) -> None:
    monkeypatch.delenv("HL_HONEST_E2E", raising=False)
    monkeypatch.delenv("HL_KERNEL_OPFANOUT_DEDUP", raising=False)
    assert krh._honest_flag("HL_KERNEL_OPFANOUT_DEDUP") is False


def test_honest_flag_umbrella_enables(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.delenv("HL_KERNEL_OPFANOUT_DEDUP", raising=False)
    assert krh._honest_flag("HL_KERNEL_OPFANOUT_DEDUP") is True


def test_honest_flag_specific_enables(monkeypatch) -> None:
    monkeypatch.delenv("HL_HONEST_E2E", raising=False)
    monkeypatch.setenv("HL_KERNEL_OPFANOUT_DEDUP", "yes")
    assert krh._honest_flag("HL_KERNEL_OPFANOUT_DEDUP") is True


def test_honest_flag_specific_falsey_overrides_umbrella(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.setenv("HL_KERNEL_OPFANOUT_DEDUP", "off")
    assert krh._honest_flag("HL_KERNEL_OPFANOUT_DEDUP") is False


# -- _vram_guarded_server_args -------------------------------------------
def test_vram_guard_off_is_identity(monkeypatch) -> None:
    monkeypatch.delenv("HL_HONEST_E2E", raising=False)
    monkeypatch.delenv("HL_INTEGRATE_VRAM_GUARD", raising=False)
    assert krh._vram_guarded_server_args("--foo bar") == "--foo bar"
    assert krh._vram_guarded_server_args("") == ""


def test_vram_guard_appends_when_on(monkeypatch) -> None:
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_UTIL_CAP", raising=False)
    out = krh._vram_guarded_server_args("--trust-remote-code")
    assert "--trust-remote-code" in out
    assert "--gpu-memory-utilization 0.9" in out


def test_vram_guard_noop_if_already_set(monkeypatch) -> None:
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    existing = "--gpu-memory-utilization 0.7"
    assert krh._vram_guarded_server_args(existing) == existing


def test_vram_guard_umbrella_and_cap(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_GUARD", raising=False)
    monkeypatch.setenv("HL_INTEGRATE_VRAM_UTIL_CAP", "0.85")
    out = krh._vram_guarded_server_args("")
    assert out == "--gpu-memory-utilization 0.85"


def test_vram_guard_cap_clamped(monkeypatch) -> None:
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_UTIL_CAP", "5.0")
    out = krh._vram_guarded_server_args("")
    assert out == "--gpu-memory-utilization 0.99"


# -- _confirm_source_imported (tri-state) ---------------------------------
def test_confirm_source_none_inputs() -> None:
    assert krh._confirm_source_imported("", None) is None
    assert krh._confirm_source_imported("foo.py", None) is None


def test_confirm_source_no_log_is_unknown(tmp_path: Path) -> None:
    assert krh._confirm_source_imported("foo.py", tmp_path) is None


def test_confirm_source_absent_is_false(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text("nothing relevant here\n", encoding="utf-8")
    assert krh._confirm_source_imported("my_kernel.py", tmp_path) is False


def test_confirm_source_import_cue_is_true(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text("INFO importing my_kernel.py module\n", encoding="utf-8")
    assert krh._confirm_source_imported("my_kernel.py", tmp_path) is True


def test_confirm_source_present_no_cue_is_unknown(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text("my_kernel mentioned bare\n", encoding="utf-8")
    assert krh._confirm_source_imported("my_kernel.py", tmp_path) is None


# -- _kernel_result_rank: umbrella-driven GEAK promotion ------------------
def _geak_nr_result() -> dict:
    return {
        "status": "ok",
        "proposal": {"decision": "NEEDS_REVIEW"},
        "verification": {
            "best_backend": "geak",
            "correctness_passed": True,
            "micro_speedup": 1.5,
        },
    }


def test_rank_geak_nr_not_promoted_when_off(monkeypatch) -> None:
    monkeypatch.delenv("HL_HONEST_E2E", raising=False)
    monkeypatch.delenv("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW", raising=False)
    keep, verified_nr, micro = krh._kernel_result_rank(_geak_nr_result())
    assert (keep, verified_nr) == (0, 0)
    assert micro == 1.5


def test_rank_geak_nr_promoted_via_umbrella(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.delenv("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW", raising=False)
    keep, verified_nr, micro = krh._kernel_result_rank(_geak_nr_result())
    assert (keep, verified_nr) == (0, 1)


def test_rank_nongeak_nr_never_promoted(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    r = _geak_nr_result()
    r["verification"]["best_backend"] = "claude"
    keep, verified_nr, _ = krh._kernel_result_rank(r)
    assert (keep, verified_nr) == (0, 0)


# -- C2a op-fanout de-dup in _batch_kernel_candidates ---------------------
def _write_candidates(tmp_path: Path) -> str:
    """Two ungrouped reusable rows sharing one source_file (op-fanout)."""
    data = {
        "hot_kernels": [
            {
                "kernel_id": "k1",
                "reusable_native_kernel": True,
                "source_file": "/srcroot/fp8_gemm.py",
                "gpu_pct": 5.0,
            },
            {
                "kernel_id": "k2",
                "reusable_native_kernel": True,
                "source_file": "/srcroot/fp8_gemm.py",
                "gpu_pct": 4.0,
            },
        ],
        "reusable_native_kernel_ids": ["k1", "k2"],
    }
    p = tmp_path / "candidates.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_opfanout_off_keeps_both_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HL_HONEST_E2E", raising=False)
    monkeypatch.delenv("HL_KERNEL_OPFANOUT_DEDUP", raising=False)
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "1.0")
    sel = krh._batch_kernel_candidates({"candidates_path": _write_candidates(tmp_path)})
    assert sorted(c["kernel_id"] for c in sel) == ["k1", "k2"]


def test_opfanout_on_collapses_and_sums(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HL_KERNEL_OPFANOUT_DEDUP", "1")
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "1.0")
    sel = krh._batch_kernel_candidates({"candidates_path": _write_candidates(tmp_path)})
    assert len(sel) == 1
    rep = sel[0]
    assert rep["kernel_id"] == "k1"  # highest-gpu_pct representative
    assert rep["gpu_pct"] == 9.0  # summed fanned share
    assert set(rep["opfanout_collapsed_ids"]) == {"k1", "k2"}


def test_opfanout_on_via_umbrella(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.delenv("HL_KERNEL_OPFANOUT_DEDUP", raising=False)
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "1.0")
    sel = krh._batch_kernel_candidates({"candidates_path": _write_candidates(tmp_path)})
    assert len(sel) == 1


# -- high-impact infra-retry cap (the dominant root-cause fix) -------------
def _infra_entry(failure_count: int, gpu_pct: float) -> dict:
    """An infra non-finish record (no verdict, status failed) at max_failures."""
    return {
        "failure_count": failure_count,
        "last_decision": "",
        "last_status": "failed",
        "rejected_reason": "",
        "last_gpu_pct": gpu_pct,
    }


def test_infra_retry_off_retires_at_max_failures(monkeypatch) -> None:
    monkeypatch.delenv("HL_HONEST_E2E", raising=False)
    monkeypatch.delenv("HL_INFRA_RETRY_HIGH_IMPACT", raising=False)
    # failure_count == max_failures => legacy retires (cap collapses to default 1).
    cap = krh._kernel_dispatch_attempt_cap(_infra_entry(2, 26.7), max_failures=2)
    assert cap == krh._DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS


def test_infra_retry_on_widens_for_high_impact(monkeypatch) -> None:
    monkeypatch.setenv("HL_INFRA_RETRY_HIGH_IMPACT", "1")
    monkeypatch.delenv("HL_INFRA_RETRY_MIN_GPU_PCT", raising=False)
    monkeypatch.delenv("HL_INFRA_RETRY_MAX", raising=False)
    # 26.7%-GPU kernel that infra-failed twice still gets attempts up to infra_max (4).
    cap = krh._kernel_dispatch_attempt_cap(_infra_entry(2, 26.7), max_failures=2)
    assert cap == 4


def test_infra_retry_on_via_umbrella(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.delenv("HL_INFRA_RETRY_HIGH_IMPACT", raising=False)
    assert krh._kernel_dispatch_attempt_cap(_infra_entry(2, 8.0), max_failures=2) == 4


def test_infra_retry_low_impact_not_widened(monkeypatch) -> None:
    monkeypatch.setenv("HL_INFRA_RETRY_HIGH_IMPACT", "1")
    # 1%-GPU kernel below the 5% threshold: not widened, retires as legacy.
    cap = krh._kernel_dispatch_attempt_cap(_infra_entry(2, 1.0), max_failures=2)
    assert cap == krh._DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS


def test_infra_retry_does_not_touch_revert(monkeypatch) -> None:
    monkeypatch.setenv("HL_INFRA_RETRY_HIGH_IMPACT", "1")
    # A real REVERT (has a verdict) is NOT a retryable infra non-finish — unchanged.
    entry = _infra_entry(2, 26.7)
    entry["last_decision"] = "REVERT"
    cap = krh._kernel_dispatch_attempt_cap(entry, max_failures=2)
    assert cap == krh._DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS


def test_infra_retry_exhausted_at_infra_max(monkeypatch) -> None:
    monkeypatch.setenv("HL_INFRA_RETRY_HIGH_IMPACT", "1")
    # Once failure_count reaches infra_max, even a high-impact kernel retires.
    cap = krh._kernel_dispatch_attempt_cap(_infra_entry(4, 26.7), max_failures=2)
    assert cap == krh._DEFAULT_KERNEL_OPT_DISPATCH_ATTEMPTS
