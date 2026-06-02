"""Unit tests for the ``aiter_jit_regressed`` / ``aiter_jit_build_stuck`` signals."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.aiter_jit import (
    AiterJitConfig,
    AiterJitDetector,
)
from robustness_agent.sources.base import SourceData


def _ctx(tick: int = 0) -> ReactorContext:
    return ReactorContext(
        tick_index=tick,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1.0,
    )


def _aiter(so_count: int, build_count: int = 0) -> dict:
    return {
        "jit_dir": "/opt/aiter/jit",
        "so_count": so_count,
        "build_count": build_count,
    }


def test_empty_aiter_data_silent():
    det = AiterJitDetector()
    out = det.evaluate(_ctx(), SourceData())
    assert out == []


def test_warm_then_still_warm_silent():
    det = AiterJitDetector(AiterJitConfig(cold_so_count=20))
    det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(80)))
    out = det.evaluate(_ctx(1), SourceData(local_aiter_jit=_aiter(80)))
    assert all(s.name != "aiter_jit_regressed" for s in out)


def test_regression_from_warm_to_cold_fires_high():
    det = AiterJitDetector(AiterJitConfig(cold_so_count=20, regression_ratio=0.8))
    det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(80)))
    out = det.evaluate(_ctx(1), SourceData(local_aiter_jit=_aiter(5)))
    sym = next(s for s in out if s.name == "aiter_jit_regressed")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["previous_so_count"] == 80
    assert sym.evidence["current_so_count"] == 5


def test_first_tick_warm_does_not_fire():
    """No previous baseline → cannot detect regression."""
    det = AiterJitDetector()
    out = det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(5)))
    assert all(s.name != "aiter_jit_regressed" for s in out)


def test_drop_within_warm_does_not_fire():
    """Drop from 80 → 60: still warm, no signal."""
    det = AiterJitDetector(AiterJitConfig(cold_so_count=20))
    det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(80)))
    out = det.evaluate(_ctx(1), SourceData(local_aiter_jit=_aiter(60)))
    assert all(s.name != "aiter_jit_regressed" for s in out)


def test_drop_above_ratio_threshold_does_not_fire():
    """Drop from 30 → 18 hits the cold threshold but the ratio is
    above ``regression_ratio`` (18 > 30 * 0.8 = 24), so no fire."""
    det = AiterJitDetector(AiterJitConfig(cold_so_count=20, regression_ratio=0.5))
    det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(30)))
    out = det.evaluate(_ctx(1), SourceData(local_aiter_jit=_aiter(18)))
    assert all(s.name != "aiter_jit_regressed" for s in out)


def test_stale_build_fires_medium_after_persistence_window():
    det = AiterJitDetector(AiterJitConfig(
        stale_build_threshold=1, stale_build_persist_ticks=3,
    ))
    # 3 consecutive ticks with build_count=5, unchanged.
    det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(50, build_count=5)))
    det.evaluate(_ctx(1), SourceData(local_aiter_jit=_aiter(50, build_count=5)))
    det.evaluate(_ctx(2), SourceData(local_aiter_jit=_aiter(50, build_count=5)))
    out = det.evaluate(_ctx(3), SourceData(local_aiter_jit=_aiter(50, build_count=5)))
    sym = next(s for s in out if s.name == "aiter_jit_build_stuck")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_stale_build_resets_on_change():
    det = AiterJitDetector(AiterJitConfig(
        stale_build_threshold=1, stale_build_persist_ticks=3,
    ))
    det.evaluate(_ctx(0), SourceData(local_aiter_jit=_aiter(50, build_count=5)))
    det.evaluate(_ctx(1), SourceData(local_aiter_jit=_aiter(50, build_count=5)))
    out = det.evaluate(_ctx(2), SourceData(local_aiter_jit=_aiter(50, build_count=8)))
    assert all(s.name != "aiter_jit_build_stuck" for s in out)
