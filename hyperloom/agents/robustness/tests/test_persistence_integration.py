"""Cross-subprocess persistence integration tests.

These tests simulate the M1 transport's "fresh subprocess per tick"
behaviour by constructing N independent ``Classifier`` / ``ActionLadder``
/ ``RcaThrottle`` instances against the same on-disk state file. They
prove that consecutive-tick rules and cooldowns continue to work
across what would be subprocess restarts in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.robustness.decision.action_ladder import (
    ActionLadder,
    ActionLadderConfig,
)
from hyperloom.agents.robustness.decision.rca_engine import (
    RcaThrottle,
    RcaThrottleConfig,
)
from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals import Classifier, Symptom, SymptomSeverity
from hyperloom.agents.robustness.signals.aiter_jit import AiterJitConfig
from hyperloom.agents.robustness.signals.crash import CrashConfig
from hyperloom.agents.robustness.signals.gpu_leak import GpuLeakConfig
from hyperloom.agents.robustness.signals.kernel_pipeline import KernelPipelineConfig
from hyperloom.agents.robustness.signals.progress import ProgressConfig
from hyperloom.agents.robustness.sources.base import SourceData
from hyperloom.agents.robustness.state_store import DetectorStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_classifier(
    session_dir: Path,
    *,
    gpu_leak_min_ticks: int = 2,
    ray_min_pending_ticks: int = 3,
) -> tuple[Classifier, DetectorStateStore]:
    """Simulate one robustness-agent CLI subprocess: a brand-new
    classifier reading state from disk, then will write at tick end."""
    store = DetectorStateStore(session_dir=session_dir)
    classifier = Classifier(
        state_store=store,
        crash_config=CrashConfig(),
        gpu_leak_config=GpuLeakConfig(
            min_consecutive_ticks=gpu_leak_min_ticks,
            free_mb_threshold=500.0,
            owner_patterns=(),  # disable owner heuristic for the test
        ),
        kernel_pipeline_config=KernelPipelineConfig(
            pending_count_threshold=0,
            min_pending_ticks=ray_min_pending_ticks,
        ),
        progress_config=ProgressConfig(
            gain_window_ticks=3,
            gain_epsilon_pct=0.1,
            no_levers_min_minutes=10_000.0,  # disable B3 in this test
        ),
        aiter_jit_config=AiterJitConfig(),
    )
    return classifier, store


def _ctx_with_tick(
    tick: int,
    *,
    cumulative_gain_validated: float = 0.0,
    optimization_stack_size: int = 0,
) -> ReactorContext:
    return ReactorContext(
        tick_index=tick,
        shared_state=SharedStateSnapshot(
            session_id="sess-int",
            tick=tick,
            cumulative_gain_validated=cumulative_gain_validated,
            optimization_stack_size=optimization_stack_size,
        ),
        inbox=[],
        now_unix=float(tick),
    )


def _leak_data() -> SourceData:
    return SourceData(
        local_gpu={
            "gpus": [
                {
                    "gpu_id": 0,
                    "vram_used_mb": 70000,
                    "vram_total_mb": 70000,
                },
            ],
        },
        local_processes=[],
        sources_used=["local"],
    )


def _ray_pending_data(pending: int = 7) -> SourceData:
    return SourceData(
        local_ray={
            "healthy": True,
            "pending_tasks": pending,
        },
        sources_used=["local"],
    )


# ---------------------------------------------------------------------------
# GpuLeakDetector — consecutive-tick rule (the original motivating bug)
# ---------------------------------------------------------------------------

def test_gpu_leak_fires_only_after_2_subprocesses_see_leak(
    tmp_path: Path,
):
    # Tick 1: fresh subprocess, sees one leak tick → no symptom (need 2).
    c1, store1 = _fresh_classifier(tmp_path)
    syms1 = c1.classify(_leak_data(), _ctx_with_tick(1))
    assert all(s.name != "gpu_memory_leaked" for s in syms1)
    store1.flush_atomic()

    # Tick 2: another fresh subprocess (the original bug — counter was
    # lost). With the state store, the previous hit was preserved →
    # this tick crosses the 2-tick threshold and fires.
    c2, store2 = _fresh_classifier(tmp_path)
    syms2 = c2.classify(_leak_data(), _ctx_with_tick(2))
    assert any(s.name == "gpu_memory_leaked" for s in syms2)
    store2.flush_atomic()


def test_gpu_leak_resets_when_owner_reappears(tmp_path: Path):
    # Tick 1: leak with no owners.
    c1, store1 = _fresh_classifier(tmp_path)
    c1.classify(_leak_data(), _ctx_with_tick(1))
    store1.flush_atomic()

    # Tick 2: an owner process is present → counter resets to 0.
    data_with_owner = SourceData(
        local_gpu=_leak_data().local_gpu,
        local_processes=[
            {"pid": 4242, "cmd": "python -m sglang.launch_server"},
        ],
        sources_used=["local"],
    )
    c2, store2 = _fresh_classifier(tmp_path)
    # ``owner_patterns`` is () in the test classifier — to truly cover
    # reset behaviour we need a classifier whose patterns match.
    c2, store2 = _fresh_classifier(tmp_path)
    # Custom GpuLeakDetector with owner pattern; we'll rebuild
    # classifier with a matching pattern.
    classifier = Classifier(
        state_store=store2,
        gpu_leak_config=GpuLeakConfig(
            min_consecutive_ticks=2,
            free_mb_threshold=500.0,
            owner_patterns=("sglang.launch_server",),
        ),
    )
    syms = classifier.classify(data_with_owner, _ctx_with_tick(2))
    assert all(s.name != "gpu_memory_leaked" for s in syms)
    store2.flush_atomic()

    # Tick 3: leak again without owner. Counter starts from 1 again,
    # not 2, so it should NOT fire.
    c3, store3 = _fresh_classifier(tmp_path)
    syms3 = c3.classify(_leak_data(), _ctx_with_tick(3))
    assert all(s.name != "gpu_memory_leaked" for s in syms3)


# ---------------------------------------------------------------------------
# RayPendingDetector — ≥3 consecutive ticks
# ---------------------------------------------------------------------------

def test_ray_pending_starvation_needs_three_subprocesses(tmp_path: Path):
    for tick in (1, 2):
        c, store = _fresh_classifier(tmp_path)
        syms = c.classify(_ray_pending_data(), _ctx_with_tick(tick))
        assert all(s.name != "ray_pending_starvation" for s in syms)
        store.flush_atomic()

    c3, store3 = _fresh_classifier(tmp_path)
    syms3 = c3.classify(_ray_pending_data(), _ctx_with_tick(3))
    assert any(s.name == "ray_pending_starvation" for s in syms3)


# ---------------------------------------------------------------------------
# ProgressDetector — rolling-window plateau
# ---------------------------------------------------------------------------

def test_gain_plateau_history_survives_subprocess_restarts(
    tmp_path: Path,
):
    # Feed a flat 3-tick window across 3 fresh classifiers. PR #239
    # added a "stack_size == 0 -> defer to no_levers" early-return to
    # ``_gain_plateau_symptom`` so the two B-family symptoms can't fire
    # twice on the same condition; seed stack_size=1 here to exercise
    # the post-promotion plateau path this test targets.
    for tick in (1, 2, 3):
        c, store = _fresh_classifier(tmp_path)
        ctx = _ctx_with_tick(
            tick, cumulative_gain_validated=0.0,
            optimization_stack_size=1,
        )
        c.classify(SourceData(sources_used=["local"]), ctx)
        store.flush_atomic()
    # After three flat ticks the rolling window is full and delta is 0
    # → ``gain_plateau`` should fire on a 4th classifier instance.
    c4, store4 = _fresh_classifier(tmp_path)
    syms = c4.classify(
        SourceData(sources_used=["local"]),
        _ctx_with_tick(
            4, cumulative_gain_validated=0.0,
            optimization_stack_size=1,
        ),
    )
    assert any(s.name == "gain_plateau" for s in syms)


# ---------------------------------------------------------------------------
# ActionLadder cooldown — persists across subprocesses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_action_ladder_cooldown_persists(tmp_path: Path):
    sym = Symptom(
        name="agent_stall",
        severity=SymptomSeverity.MEDIUM,
        summary="orchestration stalled",
        subject={"agent": "orchestration"},
        evidence={},
        source="local",
    )
    cfg = ActionLadderConfig(cooldown_ticks=5)

    store1 = DetectorStateStore(session_dir=tmp_path)
    ladder1 = ActionLadder(
        config=cfg, state_view=store1.view("action_ladder"),
    )
    result1 = await ladder1.decide(
        [sym], tick_index=10, now_unix=1.0,
    )
    # MEDIUM emits an alert (not just the heartbeat).
    assert any(
        i.type.value == "alert" for i in result1.intents
    )
    store1.flush_atomic()

    # A fresh ladder instance against the same state file. Tick 11 is
    # within the 5-tick cooldown of tick 10 → no alert this time.
    store2 = DetectorStateStore(session_dir=tmp_path)
    ladder2 = ActionLadder(
        config=cfg, state_view=store2.view("action_ladder"),
    )
    result2 = await ladder2.decide(
        [sym], tick_index=11, now_unix=2.0,
    )
    intent_types = [i.type.value for i in result2.intents]
    assert "alert" not in intent_types
    # Should be heartbeat-only.
    assert intent_types == ["send_message"]


# ---------------------------------------------------------------------------
# RcaThrottle cooldown — persists across subprocesses
# ---------------------------------------------------------------------------

def test_rca_throttle_cooldown_persists(tmp_path: Path):
    cfg = RcaThrottleConfig(
        severity_min=SymptomSeverity.HIGH,
        cooldown_seconds=60.0,
        max_calls_per_tick=5,
    )
    sym = Symptom(
        name="gpu_memory_leaked",
        severity=SymptomSeverity.HIGH,
        summary="leak",
        subject={},
        evidence={},
        source="local",
    )

    store1 = DetectorStateStore(session_dir=tmp_path)
    throttle1 = RcaThrottle(cfg, state_view=store1.view("rca_throttle"))
    assert throttle1.should_call(sym, now_unix=1000.0, tick_id=1) is True
    throttle1.record(sym, now_unix=1000.0)
    store1.flush_atomic()

    # Fresh throttle, only 10s have elapsed → still within 60s cooldown.
    store2 = DetectorStateStore(session_dir=tmp_path)
    throttle2 = RcaThrottle(cfg, state_view=store2.view("rca_throttle"))
    assert throttle2.should_call(sym, now_unix=1010.0, tick_id=2) is False

    # 90s elapsed → past the 60s cooldown → may call again.
    store3 = DetectorStateStore(session_dir=tmp_path)
    throttle3 = RcaThrottle(cfg, state_view=store3.view("rca_throttle"))
    assert throttle3.should_call(sym, now_unix=1090.0, tick_id=3) is True


# ---------------------------------------------------------------------------
# state_store_enabled=False → behaviour reverts to in-memory only
# ---------------------------------------------------------------------------

def test_classifier_without_state_store_is_in_memory(tmp_path: Path):
    classifier = Classifier(
        state_store=None,
        gpu_leak_config=GpuLeakConfig(
            min_consecutive_ticks=2,
            free_mb_threshold=500.0,
            owner_patterns=(),
        ),
    )
    # First call sees 1 hit, no symptom yet.
    syms1 = classifier.classify(_leak_data(), _ctx_with_tick(1))
    assert all(s.name != "gpu_memory_leaked" for s in syms1)
    # Same classifier instance keeps state in-memory → 2nd call fires.
    syms2 = classifier.classify(_leak_data(), _ctx_with_tick(2))
    assert any(s.name == "gpu_memory_leaked" for s in syms2)
    # A fresh classifier with no store would lose the counter — that's
    # the original M1 bug. Verifying that explicitly:
    fresh = Classifier(
        state_store=None,
        gpu_leak_config=GpuLeakConfig(
            min_consecutive_ticks=2,
            free_mb_threshold=500.0,
            owner_patterns=(),
        ),
    )
    syms_fresh = fresh.classify(_leak_data(), _ctx_with_tick(99))
    assert all(s.name != "gpu_memory_leaked" for s in syms_fresh)
