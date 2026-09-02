# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the HL_HONEST_E2E hardening helpers:
umbrella-flag resolution, VRAM util guard, import-grep source confirmation,
op-fanout de-dup in candidate batching, and umbrella-driven GEAK promotion.

Honest-E2E defaults ON (umbrella); the "off" tests opt out with HL_HONEST_E2E=0.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.kernel import request_handlers as krh


# -- _honest_flag (umbrella + per-fix override) ---------------------------
def test_honest_flag_default_off(monkeypatch) -> None:
    monkeypatch.setenv("HL_HONEST_E2E", "0")
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
    monkeypatch.setenv("HL_HONEST_E2E", "0")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_GUARD", raising=False)
    assert krh._vram_guarded_server_args("--foo bar") == "--foo bar"
    assert krh._vram_guarded_server_args("") == ""


def test_vram_guard_appends_when_on(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_UTIL_CAP", raising=False)
    out = krh._vram_guarded_server_args("--trust-remote-code")
    assert "--trust-remote-code" in out
    assert "--gpu-memory-utilization 0.9" in out


def test_vram_guard_noop_if_already_set(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    existing = "--gpu-memory-utilization 0.7"
    assert krh._vram_guarded_server_args(existing) == existing


def test_vram_guard_umbrella_and_cap(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_HONEST_E2E", "1")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_GUARD", raising=False)
    monkeypatch.setenv("HL_INTEGRATE_VRAM_UTIL_CAP", "0.85")
    out = krh._vram_guarded_server_args("")
    assert out == "--gpu-memory-utilization 0.85"


def test_vram_guard_cap_clamped(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_UTIL_CAP", "5.0")
    out = krh._vram_guarded_server_args("")
    assert out == "--gpu-memory-utilization 0.99"


def test_vram_guard_sglang_is_noop(monkeypatch) -> None:
    # sglang rejects --gpu-memory-utilization; the guard must be a no-op for it.
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    assert krh._vram_guarded_server_args("--trust-remote-code") == "--trust-remote-code"
    assert "gpu-memory-utilization" not in krh._vram_guarded_server_args("")


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


# -- _confirm_sources_imported (multi-file aggregate) ----------------------
def _write_server_log(tmp_path: Path, text: str) -> None:
    (tmp_path / "server.log").write_text(text, encoding="utf-8")


def test_confirm_sources_without_paths_is_unknown(tmp_path: Path) -> None:
    assert krh._confirm_sources_imported([], tmp_path) == (None, {})
    assert krh._confirm_sources_imported(["", "   "], tmp_path) == (None, {})


def test_confirm_sources_all_imported_is_true(tmp_path: Path) -> None:
    _write_server_log(
        tmp_path,
        "INFO importing flydsl_gemm.py\nINFO loading dispatcher.py\n",
    )

    overall, per_file = krh._confirm_sources_imported(
        ["vllm/flydsl_gemm.py", "vllm/dispatcher.py"],
        tmp_path,
    )

    assert overall is True
    assert per_file == {"vllm/flydsl_gemm.py": True, "vllm/dispatcher.py": True}


def test_confirm_sources_nothing_loaded_is_false(tmp_path: Path) -> None:
    """None of the change ran, so the measured delta is unattributable."""
    _write_server_log(tmp_path, "nothing relevant here\n")

    overall, per_file = krh._confirm_sources_imported(
        ["vllm/flydsl_gemm.py", "vllm/dispatcher.py"],
        tmp_path,
    )

    assert overall is False
    assert set(per_file.values()) == {False}


def test_confirm_sources_partial_evidence_never_condemns(tmp_path: Path) -> None:
    """A module can be imported lazily, so a partial trace is audit-only."""
    _write_server_log(tmp_path, "INFO importing dispatcher.py\n")

    overall, per_file = krh._confirm_sources_imported(
        ["vllm/flydsl_gemm.py", "vllm/dispatcher.py"],
        tmp_path,
    )

    assert overall is None
    assert per_file["vllm/flydsl_gemm.py"] is False
    assert per_file["vllm/dispatcher.py"] is True


def test_confirm_sources_mixed_unknown_stays_unknown(tmp_path: Path) -> None:
    _write_server_log(
        tmp_path,
        "INFO importing dispatcher.py\nflydsl_gemm mentioned bare\n",
    )

    overall, per_file = krh._confirm_sources_imported(
        ["vllm/flydsl_gemm.py", "vllm/dispatcher.py"],
        tmp_path,
    )

    assert overall is None
    assert per_file["vllm/flydsl_gemm.py"] is None


def test_confirm_sources_deduplicates_repeated_paths(tmp_path: Path) -> None:
    _write_server_log(tmp_path, "INFO importing dispatcher.py\n")

    overall, per_file = krh._confirm_sources_imported(
        ["vllm/dispatcher.py", "vllm/dispatcher.py"],
        tmp_path,
    )

    assert overall is True
    assert list(per_file) == ["vllm/dispatcher.py"]


def test_confirm_sources_matches_single_file_semantics(tmp_path: Path) -> None:
    """A one-file bundle must grade exactly as the single-file check does."""
    _write_server_log(tmp_path, "nothing relevant here\n")

    overall, _ = krh._confirm_sources_imported(["my_kernel.py"], tmp_path)

    assert overall is krh._confirm_source_imported("my_kernel.py", tmp_path)
    assert overall is False


# -- _kernel_result_rank ---------------------------------------------------
def _needs_review_result() -> dict:
    return {
        "status": "ok",
        "proposal": {"decision": "NEEDS_REVIEW"},
        "verification": {
            "correctness_passed": True,
            "micro_speedup": 1.5,
        },
    }


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
