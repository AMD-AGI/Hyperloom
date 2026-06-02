"""Unit tests for the ``gpu_memory_leaked`` signal."""

from __future__ import annotations

import pytest

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import (
    Classifier,
    GpuLeakConfig,
    GpuLeakDetector,
    Symptom,
    SymptomSeverity,
)
from robustness_agent.sources.base import SourceData


def _ctx(tick: int = 0) -> ReactorContext:
    return ReactorContext(
        tick_index=tick,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1.0,
    )


def _full_gpus(n: int = 4) -> dict:
    return {
        "gpus": [
            {
                "gpu_id": i,
                "util_mem_pct": 99.7,
                "vram_used_mb": 196350.0,
                "vram_total_mb": 196608.0,
            }
            for i in range(n)
        ],
        "tool": "rocm-smi",
    }


def _half_full_gpus(n: int = 4) -> dict:
    return {
        "gpus": [
            {
                "gpu_id": i,
                "util_mem_pct": 50.0,
                "vram_used_mb": 98000.0,
                "vram_total_mb": 196608.0,
            }
            for i in range(n)
        ],
        "tool": "rocm-smi",
    }


def _data(local_gpu: dict, processes: list | None = None) -> SourceData:
    return SourceData(
        local_gpu=local_gpu,
        local_processes=list(processes or []),
    )


# ---------------------------------------------------------------------------
# Two-tick gate
# ---------------------------------------------------------------------------

def test_first_tick_with_all_full_no_owner_is_silent():
    """min_consecutive_ticks=2 means a single tick is not enough."""
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    out = det.evaluate(_ctx(), _data(_full_gpus()))
    assert out == []
    assert det.consecutive_hits == 1


def test_two_consecutive_full_ticks_with_no_owner_emits_high():
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    assert det.evaluate(_ctx(0), _data(_full_gpus())) == []
    second = det.evaluate(_ctx(1), _data(_full_gpus()))
    assert len(second) == 1
    sym = second[0]
    assert isinstance(sym, Symptom)
    assert sym.name == "gpu_memory_leaked"
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["consecutive_hits"] == 2
    assert sym.evidence["gpu_count"] == 4
    assert sym.suggestion.startswith("delegate(recover")


def test_min_consecutive_ticks_one_emits_immediately():
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=1))
    out = det.evaluate(_ctx(), _data(_full_gpus()))
    assert len(out) == 1
    assert out[0].name == "gpu_memory_leaked"


# ---------------------------------------------------------------------------
# Live owner silence
# ---------------------------------------------------------------------------

def test_full_but_live_engine_core_owner_is_silent():
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    procs = [
        {
            "pid": 1234,
            "rss_mb": 4096.0,
            "cmd": "python -m vllm.entrypoints.openai.api_server",
        },
        {
            "pid": 1235,
            "rss_mb": 8000.0,
            "cmd": "vllm-EngineCore worker rank=0",
        },
    ]
    assert det.evaluate(_ctx(0), _data(_full_gpus(), procs)) == []
    # Even a second tick keeps quiet because the owner is alive.
    assert det.evaluate(_ctx(1), _data(_full_gpus(), procs)) == []
    assert det.consecutive_hits == 0


def test_full_with_unrelated_process_still_fires():
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    procs = [{"pid": 999, "rss_mb": 10.0, "cmd": "bash -c sleep"}]
    assert det.evaluate(_ctx(0), _data(_full_gpus(), procs)) == []
    second = det.evaluate(_ctx(1), _data(_full_gpus(), procs))
    assert second and second[0].name == "gpu_memory_leaked"


# ---------------------------------------------------------------------------
# Partial-fill silence
# ---------------------------------------------------------------------------

def test_partial_full_is_silent():
    """Only some GPUs full -> not a leak."""
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    snap = _full_gpus(4)
    # Mark gpu_id=2 as not full
    snap["gpus"][2] = {
        "gpu_id": 2,
        "util_mem_pct": 25.0,
        "vram_used_mb": 49000.0,
        "vram_total_mb": 196608.0,
    }
    assert det.evaluate(_ctx(0), _data(snap)) == []
    assert det.evaluate(_ctx(1), _data(snap)) == []
    assert det.consecutive_hits == 0


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

def test_reset_after_one_clean_tick():
    """full -> clean -> full -> full: detector emits only on the 4th tick."""
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    # tick 0: full
    assert det.evaluate(_ctx(0), _data(_full_gpus())) == []
    assert det.consecutive_hits == 1
    # tick 1: clean -> reset
    assert det.evaluate(_ctx(1), _data(_half_full_gpus())) == []
    assert det.consecutive_hits == 0
    # tick 2: full -> counter back to 1, still silent
    assert det.evaluate(_ctx(2), _data(_full_gpus())) == []
    assert det.consecutive_hits == 1
    # tick 3: full again -> emit
    out = det.evaluate(_ctx(3), _data(_full_gpus()))
    assert out and out[0].name == "gpu_memory_leaked"


def test_no_gpu_data_is_silent_and_resets_counter():
    det = GpuLeakDetector(GpuLeakConfig(min_consecutive_ticks=2))
    assert det.evaluate(_ctx(0), _data(_full_gpus())) == []
    assert det.consecutive_hits == 1
    # GPU data dropped -> reset
    assert det.evaluate(_ctx(1), _data({})) == []
    assert det.consecutive_hits == 0


# ---------------------------------------------------------------------------
# Free-MB threshold path
# ---------------------------------------------------------------------------

def test_free_mb_threshold_fires_when_util_mem_pct_missing():
    """rocm-smi older builds may omit ``GPU memory use (%)`` — fall back
    on absolute free MiB."""
    det = GpuLeakDetector(GpuLeakConfig(
        util_mem_pct_threshold=200.0,  # effectively unreachable
        free_mb_threshold=500.0,
        min_consecutive_ticks=1,
    ))
    snap = {
        "gpus": [
            {"gpu_id": 0, "vram_used_mb": 196500.0, "vram_total_mb": 196608.0},
            {"gpu_id": 1, "vram_used_mb": 196500.0, "vram_total_mb": 196608.0},
        ]
    }
    out = det.evaluate(_ctx(), _data(snap))
    assert out and out[0].name == "gpu_memory_leaked"
    per = out[0].evidence["per_gpu"]
    assert per[0]["free_mb"] == pytest.approx(108.0, abs=1.0)


# ---------------------------------------------------------------------------
# Classifier integration
# ---------------------------------------------------------------------------

def test_classifier_includes_gpu_leak_rule():
    classifier = Classifier(
        gpu_leak_config=GpuLeakConfig(min_consecutive_ticks=1),
    )
    out = classifier.classify(_data(_full_gpus()), _ctx())
    names = {s.name for s in out}
    assert "gpu_memory_leaked" in names


def test_classifier_state_persists_across_ticks():
    classifier = Classifier(
        gpu_leak_config=GpuLeakConfig(min_consecutive_ticks=2),
    )
    first = classifier.classify(_data(_full_gpus()), _ctx(0))
    assert all(s.name != "gpu_memory_leaked" for s in first)
    second = classifier.classify(_data(_full_gpus()), _ctx(1))
    assert any(s.name == "gpu_memory_leaked" for s in second)
