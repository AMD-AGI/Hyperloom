# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for C1 / C2 / C3 preflight signals."""

from __future__ import annotations

import pytest

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.preflight import (
    AmdahlCeilingConfig,
    AmdahlCeilingDetector,
    ColdStartConfig,
    HeadroomBreakdown,
    ModelGpuFitConfig,
    ModelGpuFitDetector,
    amdahl_e2e_ceiling,
    compute_headroom_gib,
    evaluate_cold_start_signals,
    extract_params_billions,
)
from robustness_agent.sources.base import SourceData


def _ctx(
    *,
    budget_minutes: float = 360.0,
    remaining_minutes: float = 60.0,
    closing_phase: bool = False,
    stop_reason: str = "",
    tick: int = 0,
) -> ReactorContext:
    return ReactorContext(
        tick_index=tick,
        shared_state=SharedStateSnapshot(
            session_id="sess-1",
            budget_minutes=budget_minutes,
            remaining_minutes=remaining_minutes,
            closing_phase=closing_phase,
            stop_reason=stop_reason,
        ),
        inbox=[],
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# helpers — pure math
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("DeepSeek-R1-0528-671B", 671.0),
        ("Qwen3-32B", 32.0),
        ("Llama-3.1-8B-Instruct-FP8", 8.0),
        ("gpt-oss-120b", 120.0),
        ("MiniMax-M2.5", None),
        ("", None),
        ("Mistral-7b-v0.3", 7.0),
        ("Mixtral-8x22B", None),  # ``8x22B`` — ``2`` is preceded by ``2`` (digit); ``x22`` rejected by ``(?<![A-Za-z0-9])`` boundary. Edge case the heuristic deliberately skips.
    ],
)
def test_extract_params_billions(name, expected):
    assert extract_params_billions(name) == expected


def test_amdahl_e2e_ceiling_well_known_values():
    # 100% optimizable at 1.5x → 1.5x E2E.
    assert amdahl_e2e_ceiling(optimizable_pct=100, single_kernel_speedup=1.5) == pytest.approx(1.5, rel=1e-3)
    # 0% optimizable → no gain regardless of single-kernel speedup.
    assert amdahl_e2e_ceiling(optimizable_pct=0, single_kernel_speedup=2.0) == 1.0
    # 30.9% Triton at 1.5x → ceiling ~ 1.117x = 11.7%.
    assert amdahl_e2e_ceiling(
        optimizable_pct=30.9, single_kernel_speedup=1.5
    ) == pytest.approx(1.117, rel=1e-2)


def test_compute_headroom_gib_basic_mi300x_fits():
    """8B BF16 model on MI300X with TP=8 fits easily."""
    manifest = {
        "model_name": "Llama-3.1-8B-Instruct",
        "model_class": "dense",
        "gpu_type": "mi300x",
        "tp": 8,
        "workload": {
            "precision": "bf16",
            "max_model_len": 4096,
            "conc": 16,
        },
    }
    br = compute_headroom_gib(manifest)
    assert isinstance(br, HeadroomBreakdown)
    assert br.required_gib < br.hbm_gib
    assert br.headroom_pct > 50


def test_compute_headroom_gib_dsr1_tp1_does_not_fit():
    """The 2026-05 DSR1 case: 671B FP8 + TP=1 on MI300X = impossible."""
    manifest = {
        "model_name": "DeepSeek-R1-0528-671B",
        "model_class": "moe_mla",
        "gpu_type": "mi300x",
        "tp": 1,
        "workload": {
            "precision": "fp8",
            "max_model_len": 4096,
            "conc": 8,
        },
    }
    br = compute_headroom_gib(manifest)
    assert isinstance(br, HeadroomBreakdown)
    # 671 * 1 * 1.05 = ~704 GB needed in single GPU; MI300X has 192 GB.
    assert br.required_gib > br.hbm_gib
    assert br.headroom_pct < 0


def test_compute_headroom_gib_returns_none_for_missing_fields():
    assert compute_headroom_gib({}) is None
    assert compute_headroom_gib({"model_name": "no-size-token"}) is None
    assert compute_headroom_gib({
        "model_name": "32B",
        "workload": {"precision": "bf16"},
        "tp": 8,
        # missing gpu_type
    }) is None


# ---------------------------------------------------------------------------
# C1 ModelGpuFitDetector
# ---------------------------------------------------------------------------

def _manifest_dsr1_tp1() -> dict:
    return {
        "model_name": "DeepSeek-R1-0528-671B",
        "model_class": "moe_mla",
        "gpu_type": "mi300x",
        "tp": 1,
        "workload": {"precision": "fp8", "max_model_len": 4096, "conc": 8},
    }


def _manifest_dsr1_tp8() -> dict:
    return {
        "model_name": "DeepSeek-R1-0528-671B",
        "model_class": "moe_mla",
        "gpu_type": "mi300x",
        "tp": 8,
        "workload": {"precision": "fp8", "max_model_len": 4096, "conc": 8},
    }


def test_model_gpu_fit_fires_high_on_infeasible():
    det = ModelGpuFitDetector()
    out = det.evaluate(
        _ctx(), SourceData(local_manifest=_manifest_dsr1_tp1())
    )
    sym = next(s for s in out if s.name == "model_gpu_infeasible")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["headroom_pct"] < 0
    assert sym.evidence["tp"] == 1
    assert sym.evidence["gpu_type"] == "mi300x"


def test_model_gpu_fit_silent_when_fits():
    det = ModelGpuFitDetector()
    out = det.evaluate(
        _ctx(), SourceData(local_manifest=_manifest_dsr1_tp8())
    )
    assert all(s.name != "model_gpu_infeasible" for s in out)


def test_model_gpu_fit_fires_once_per_fingerprint():
    """Same manifest across ticks → fire only on first hit."""
    det = ModelGpuFitDetector()
    data = SourceData(local_manifest=_manifest_dsr1_tp1())
    first = det.evaluate(_ctx(tick=0), data)
    second = det.evaluate(_ctx(tick=1), data)
    third = det.evaluate(_ctx(tick=2), data)
    assert len(first) == 1
    assert second == []
    assert third == []


def test_model_gpu_fit_re_fires_when_manifest_changes():
    """A new manifest (e.g. resume with different gpu_type) → fresh fire."""
    det = ModelGpuFitDetector()
    first = det.evaluate(_ctx(), SourceData(local_manifest=_manifest_dsr1_tp1()))
    # Operator widens TP — feasibility now passes, but manifest changed,
    # so the detector remembers a new fingerprint with no fire.
    second = det.evaluate(_ctx(), SourceData(local_manifest=_manifest_dsr1_tp8()))
    # Re-introduce the bad config (rare but possible on user override) →
    # fire again.
    third = det.evaluate(_ctx(), SourceData(local_manifest=_manifest_dsr1_tp1()))
    assert len(first) == 1
    assert second == []
    assert len(third) == 1


def test_model_gpu_fit_silent_on_missing_manifest():
    det = ModelGpuFitDetector()
    assert det.evaluate(_ctx(), SourceData()) == []


def test_model_gpu_fit_silent_on_unknown_model_name():
    """No -<N>B token in name → cannot judge feasibility → silent."""
    det = ModelGpuFitDetector()
    out = det.evaluate(_ctx(), SourceData(local_manifest={
        "model_name": "MiniMax-M2.5",
        "gpu_type": "mi300x",
        "tp": 8,
        "workload": {"precision": "bf16", "max_model_len": 4096, "conc": 8},
    }))
    assert all(s.name != "model_gpu_infeasible" for s in out)


def test_model_gpu_fit_custom_min_headroom_pct():
    """Aggressive 30% headroom requirement marks marginal fits as infeasible."""
    det = ModelGpuFitDetector(ModelGpuFitConfig(min_headroom_pct=30.0))
    # 32B BF16 on MI300X with TP=2 — ~64 GiB weights / 2 = 32 GiB per gpu
    # plus KV cache & activation — well under 192 GB, headroom ~60%+ →
    # still passes the strict 30% gate.
    out = det.evaluate(_ctx(), SourceData(local_manifest={
        "model_name": "Qwen3-32B",
        "gpu_type": "mi300x",
        "tp": 2,
        "workload": {"precision": "bf16", "max_model_len": 4096, "conc": 8},
    }))
    assert out == []


# ---------------------------------------------------------------------------
# C2 AmdahlCeilingDetector
# ---------------------------------------------------------------------------

def test_amdahl_ceiling_silent_on_dsr1_case_with_default_threshold():
    """The 2026-05 DSR1-FP8 case (30.9% Triton at 1.5x → ceiling ~11.7%):
    the *theoretical* Amdahl ceiling is above the 5% default. In practice
    GEAK delivers way less, so operators may want to tighten the threshold
    — but the *default* behaviour is intentionally conservative."""
    det = AmdahlCeilingDetector()
    breakdown = {
        "tier_pcts": {
            "triton": 30.9, "vendor": 41.6, "framework": 7.4, "comm": 20.1,
        },
        "total_kernels": 50,
        "kernel_breakdown_path": "/p/kernel_breakdown.json",
        "mtime": 1700.0,
    }
    out = det.evaluate(_ctx(), SourceData(local_kernel_breakdown=breakdown))
    # ~11.7% ceiling > default 5% → silent.
    assert all(s.name != "amdahl_kernel_ceiling_low" for s in out)


def test_amdahl_ceiling_fires_on_truly_low_optimizable():
    """20% Triton @ 1.5x → ceiling ~7.1%; above default 5%, silent.
    Try 5% Triton instead → ceiling 1.66%."""
    det = AmdahlCeilingDetector()
    breakdown = {
        "tier_pcts": {"triton": 5.0, "vendor": 70.0, "framework": 25.0},
        "total_kernels": 30,
        "kernel_breakdown_path": "/p/kernel_breakdown.json",
        "mtime": 1800.0,
    }
    out = det.evaluate(_ctx(), SourceData(local_kernel_breakdown=breakdown))
    sym = next(s for s in out if s.name == "amdahl_kernel_ceiling_low")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["optimizable_pct"] == 5.0
    assert sym.evidence["e2e_ceiling_pct"] < 5.0


def test_amdahl_ceiling_silent_when_optimizable_high():
    det = AmdahlCeilingDetector()
    breakdown = {
        "tier_pcts": {"triton": 60.0, "vendor": 20.0, "framework": 20.0},
        "total_kernels": 30,
        "kernel_breakdown_path": "/p/kernel_breakdown.json",
        "mtime": 1700.0,
    }
    out = det.evaluate(_ctx(), SourceData(local_kernel_breakdown=breakdown))
    assert all(s.name != "amdahl_kernel_ceiling_low" for s in out)


def test_amdahl_ceiling_fires_once_per_mtime():
    det = AmdahlCeilingDetector()
    breakdown = {
        "tier_pcts": {"triton": 5.0, "vendor": 90.0, "framework": 5.0},
        "total_kernels": 30,
        "kernel_breakdown_path": "/p/kernel_breakdown.json",
        "mtime": 100.0,
    }
    first = det.evaluate(_ctx(), SourceData(local_kernel_breakdown=breakdown))
    second = det.evaluate(_ctx(), SourceData(local_kernel_breakdown=breakdown))
    assert len(first) == 1
    assert second == []


def test_amdahl_ceiling_re_fires_when_breakdown_remrofiled():
    det = AmdahlCeilingDetector()
    base = {
        "tier_pcts": {"triton": 5.0, "vendor": 90.0, "framework": 5.0},
        "total_kernels": 30,
        "kernel_breakdown_path": "/p/kernel_breakdown.json",
    }
    first = det.evaluate(_ctx(), SourceData(local_kernel_breakdown={**base, "mtime": 100.0}))
    second = det.evaluate(_ctx(), SourceData(local_kernel_breakdown={**base, "mtime": 200.0}))
    assert len(first) == 1
    assert len(second) == 1


def test_amdahl_ceiling_silent_when_no_breakdown():
    det = AmdahlCeilingDetector()
    assert det.evaluate(_ctx(), SourceData()) == []


def test_amdahl_ceiling_custom_aggressive_thresholds():
    """Tighter 15% ceiling threshold catches the DSR1 30.9% Triton case."""
    det = AmdahlCeilingDetector(AmdahlCeilingConfig(
        single_kernel_speedup=1.5, min_e2e_ceiling_pct=15.0,
    ))
    breakdown = {
        "tier_pcts": {"triton": 30.9, "vendor": 41.6, "framework": 7.4, "comm": 20.1},
        "total_kernels": 50,
        "kernel_breakdown_path": "/p/k.json",
        "mtime": 200.0,
    }
    out = det.evaluate(_ctx(), SourceData(local_kernel_breakdown=breakdown))
    sym = next(s for s in out if s.name == "amdahl_kernel_ceiling_low")
    assert sym.evidence["e2e_ceiling_pct"] < 15.0


# ---------------------------------------------------------------------------
# C3 cold_start_budget_exhausted
# ---------------------------------------------------------------------------

def test_cold_start_fires_when_cold_and_short_budget():
    data = SourceData(local_aiter_jit={"so_count": 5, "jit_dir": "/x"})
    ctx = _ctx(budget_minutes=120.0, remaining_minutes=30.0)
    out = evaluate_cold_start_signals(
        ctx, data,
        config=ColdStartConfig(cold_so_count=20, cold_start_minutes=60.0),
    )
    sym = next(s for s in out if s.name == "cold_start_budget_exhausted")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["so_count"] == 5


def test_cold_start_silent_when_warm():
    data = SourceData(local_aiter_jit={"so_count": 80})
    ctx = _ctx(budget_minutes=120.0, remaining_minutes=30.0)
    out = evaluate_cold_start_signals(
        ctx, data,
        config=ColdStartConfig(cold_so_count=20, cold_start_minutes=60.0),
    )
    assert all(s.name != "cold_start_budget_exhausted" for s in out)


def test_cold_start_silent_with_ample_budget():
    data = SourceData(local_aiter_jit={"so_count": 5})
    ctx = _ctx(budget_minutes=360.0, remaining_minutes=300.0)
    out = evaluate_cold_start_signals(
        ctx, data,
        config=ColdStartConfig(cold_so_count=20, cold_start_minutes=60.0),
    )
    assert all(s.name != "cold_start_budget_exhausted" for s in out)


def test_cold_start_silent_in_closing_phase():
    data = SourceData(local_aiter_jit={"so_count": 5})
    ctx = _ctx(remaining_minutes=10.0, closing_phase=True)
    out = evaluate_cold_start_signals(
        ctx, data,
        config=ColdStartConfig(cold_so_count=20, cold_start_minutes=60.0),
    )
    assert out == []


def test_cold_start_silent_with_short_budget_below_min():
    data = SourceData(local_aiter_jit={"so_count": 5})
    ctx = _ctx(budget_minutes=20.0, remaining_minutes=10.0)
    out = evaluate_cold_start_signals(
        ctx, data,
        config=ColdStartConfig(
            cold_so_count=20, cold_start_minutes=60.0, min_budget_minutes=30.0,
        ),
    )
    assert out == []


def test_cold_start_reads_env_when_minutes_unset(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC", "5400",
    )  # 90 min
    data = SourceData(local_aiter_jit={"so_count": 5})
    ctx = _ctx(budget_minutes=180.0, remaining_minutes=80.0)
    # 80 min remaining < 90 min env cold-start → fire.
    out = evaluate_cold_start_signals(
        ctx, data, config=ColdStartConfig(cold_so_count=20),
    )
    sym = next(s for s in out if s.name == "cold_start_budget_exhausted")
    assert sym.evidence["cold_start_minutes"] == 90.0


def test_cold_start_silent_when_no_aiter_data():
    out = evaluate_cold_start_signals(
        _ctx(), SourceData(),
        config=ColdStartConfig(cold_so_count=20, cold_start_minutes=60.0),
    )
    assert out == []
