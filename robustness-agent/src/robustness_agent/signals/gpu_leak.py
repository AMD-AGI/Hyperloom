# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""GPU memory leak detector.

Fires when **every** visible GPU reports near-100% memory utilization (or
near-zero free MiB) AND **no** process matching any known inference-server
owner pattern is live on the host. This is the classic ROCm KFD leak
fingerprint observed when a Magpie / vLLM EngineCore crashes mid-run and
the kernel-side driver tables keep tracking VRAM against dead PIDs.

The signal stays out of M1 :mod:`local_health` because it has two
properties the simpler rules don't:

* It is **stateful** — a single tick of "all GPUs full" is not enough.
  Baseline cold-start legitimately pegs every GPU's memory while
  ``aiter`` JIT-compiles, so we require ``min_consecutive_ticks`` clean
  hits before promoting to HIGH severity.
* It cross-correlates two different :class:`SourceData` slots
  (``local_gpu`` and ``local_processes``); the existing
  :func:`evaluate_local_health_signals` rules are single-slot.

The detector is therefore a class with internal counters; the classifier
constructs one detector instance per reactor and calls
:meth:`GpuLeakDetector.evaluate` each tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



# Default inference-server / benchmark owners whose presence proves the
# VRAM is being used legitimately rather than leaked. Mirrors the
# extended ``_DEFAULT_PROCESS_PATTERNS`` in
# :mod:`robustness_agent.sources.local_probe`. The 2026-05-18 vLLM v1
# additions (``vllm.v1.engine.core`` / ``vllm.engine.async_llm_engine``)
# close the false-fire gap where ``EngineCore-`` child PIDs hold VRAM
# but their cmdline does not contain ``vllm.entrypoints``.
_DEFAULT_OWNER_PATTERNS: tuple[str, ...] = (
    "sglang.launch_server",
    "sglang.srt",
    "vllm.entrypoints",
    "vllm serve",
    "vllm.v1.engine.core",
    "vllm.engine.async_llm_engine",
    "EngineCore",
    "Magpie",
    "inferencex",
    "ray::IDLE",
    "raylet",
    "hipcc",
    "benchmark_serving",
)


@dataclass
class GpuLeakConfig:
    """Tunables for :class:`GpuLeakDetector`.

    Either threshold is sufficient to mark a GPU as "full" — operators
    typically configure one and leave the other at a permissive default.
    ``min_consecutive_ticks`` is the anti-flap gate; the recommended
    value of 2 catches the steady-state leak without firing during the
    normal baseline VRAM ramp.
    """

    util_mem_pct_threshold: float = 99.0
    free_mb_threshold: float = 500.0
    min_consecutive_ticks: int = 2
    owner_patterns: tuple[str, ...] = _DEFAULT_OWNER_PATTERNS


class GpuLeakDetector:
    """Stateful per-tick rule that emits ``gpu_memory_leaked`` symptoms.

    Construct once per reactor and call :meth:`evaluate` each tick. The
    counter resets to zero on any tick whose conditions are not met,
    so a transient one-tick blip during baseline cold-start cannot
    accumulate toward a false positive.
    """

    def __init__(
        self,
        config: GpuLeakConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or GpuLeakConfig()
        self._state_view = state_view
        # Disk-backed counter — survives the subprocess-per-tick
        # transport. Persists `consecutive_hits` so a leak
        # crossing the 2-tick threshold is detected even when the
        # reactor is rebuilt every tick.
        loaded = state_view.load() if state_view is not None else {}
        raw_hits = loaded.get("consecutive_hits", 0)
        try:
            self._consecutive_hits: int = max(0, int(raw_hits))
        except (TypeError, ValueError):
            self._consecutive_hits = 0

    @property
    def consecutive_hits(self) -> int:
        """Visible for tests; production code should not rely on this."""
        return self._consecutive_hits

    def _persist(self) -> None:
        if self._state_view is None:
            return
        self._state_view.save({"consecutive_hits": self._consecutive_hits})

    def evaluate(self, ctx: ReactorContext, data: SourceData) -> list[Symptom]:
        gpus = self._extract_gpu_snapshots(data)
        if not gpus:
            # No GPU data → can't conclude anything; reset so partial
            # data on the next tick doesn't accumulate stale state.
            self._consecutive_hits = 0
            self._persist()
            return []

        full_gpus = [snap for snap in gpus if self._is_full(snap)]
        all_full = len(full_gpus) == len(gpus)
        if not all_full:
            self._consecutive_hits = 0
            self._persist()
            return []

        live_owners = self._live_owners(data)
        if live_owners:
            # Legitimate owner present — memory pressure isn't a leak.
            self._consecutive_hits = 0
            self._persist()
            return []

        self._consecutive_hits += 1
        self._persist()
        if self._consecutive_hits < self._config.min_consecutive_ticks:
            return []

        return [self._build_symptom(gpus, ctx)]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _extract_gpu_snapshots(self, data: SourceData) -> list[dict[str, Any]]:
        gpus = data.local_gpu.get("gpus") if isinstance(data.local_gpu, dict) else None
        if not isinstance(gpus, list):
            return []
        return [snap for snap in gpus if isinstance(snap, dict)]

    def _is_full(self, snap: dict[str, Any]) -> bool:
        cfg = self._config
        util_mem = snap.get("util_mem_pct")
        if isinstance(util_mem, (int, float)) and util_mem >= cfg.util_mem_pct_threshold:
            return True
        used = snap.get("vram_used_mb")
        total = snap.get("vram_total_mb")
        if (
            isinstance(used, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            free_mb = max(0.0, float(total) - float(used))
            if free_mb <= cfg.free_mb_threshold:
                return True
        return False

    def _live_owners(self, data: SourceData) -> list[dict[str, Any]]:
        if not data.local_processes:
            return []
        owners: list[dict[str, Any]] = []
        for proc in data.local_processes:
            if not isinstance(proc, dict):
                continue
            cmd = str(proc.get("cmd") or "")
            if not cmd:
                continue
            if any(pat in cmd for pat in self._config.owner_patterns):
                owners.append(proc)
        return owners

    def _build_symptom(
        self,
        gpus: list[dict[str, Any]],
        ctx: ReactorContext,
    ) -> Symptom:
        cfg = self._config
        per_gpu: list[dict[str, Any]] = []
        for snap in gpus:
            entry: dict[str, Any] = {"gpu_id": snap.get("gpu_id")}
            util_mem = snap.get("util_mem_pct")
            if isinstance(util_mem, (int, float)):
                entry["util_mem_pct"] = round(float(util_mem), 2)
            used = snap.get("vram_used_mb")
            total = snap.get("vram_total_mb")
            if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
                entry["vram_used_mb"] = round(float(used), 1)
                entry["vram_total_mb"] = round(float(total), 1)
                entry["free_mb"] = round(max(0.0, float(total) - float(used)), 1)
            per_gpu.append(entry)

        summary = (
            f"all {len(gpus)} GPU(s) report memory at >= "
            f"{cfg.util_mem_pct_threshold:.0f}% (or free <= "
            f"{cfg.free_mb_threshold:.0f} MiB) with no live owner "
            f"process for {self._consecutive_hits} consecutive tick(s); "
            "treating as KFD/VRAM leak from a crashed inference server"
        )
        evidence: dict[str, Any] = {
            "consecutive_hits": self._consecutive_hits,
            "util_mem_pct_threshold": cfg.util_mem_pct_threshold,
            "free_mb_threshold": cfg.free_mb_threshold,
            "gpu_count": len(gpus),
            "per_gpu": per_gpu,
            "owner_patterns": list(cfg.owner_patterns),
        }
        return Symptom(
            name="gpu_memory_leaked",
            severity=SymptomSeverity.HIGH,
            summary=summary,
            evidence=evidence,
            subject={},  # session-wide, not per-GPU
            source="local",
            suggestion=(
                "delegate(recover, params={force_gpu_cleanup: True}); "
                "if recover returns needs_review, propose `report` to "
                "finalize at the last validated gain"
            ),
        )


def evaluate_gpu_leak_signals(
    detector: GpuLeakDetector,
    ctx: ReactorContext,
    data: SourceData,
) -> list[Symptom]:
    """Module-level helper mirroring the other signal rule entry points.

    The classifier owns the :class:`GpuLeakDetector` instance because
    the signal is stateful; this wrapper just adapts the
    ``(ctx, data) -> list[Symptom]`` signature the rest of
    :mod:`signals` exposes.
    """
    return detector.evaluate(ctx, data)


__all__ = [
    "GpuLeakConfig",
    "GpuLeakDetector",
    "evaluate_gpu_leak_signals",
]
