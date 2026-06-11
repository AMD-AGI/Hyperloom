# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""GPU memory leak detector.

Fires when every visible GPU is near-full AND no known inference-server owner process
is live — the classic ROCm KFD leak fingerprint after a crashed EngineCore. Stateful
(requires ``min_consecutive_ticks`` to skip the baseline cold-start VRAM ramp) and
cross-correlates ``local_gpu`` + ``local_processes``, so it lives in its own detector
class rather than M1 :mod:`local_health`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



# Inference-server / benchmark owners whose presence proves VRAM use is legitimate,
# not leaked. Mirrors ``_DEFAULT_PROCESS_PATTERNS`` in sources.local_probe; the vLLM v1
# entries close the gap where ``EngineCore-`` child PIDs hold VRAM without ``vllm.entrypoints``.
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

    Either threshold marks a GPU "full"; ``min_consecutive_ticks`` (default 2) is the
    anti-flap gate that skips the baseline VRAM ramp.
    """

    util_mem_pct_threshold: float = 99.0
    free_mb_threshold: float = 500.0
    min_consecutive_ticks: int = 2
    owner_patterns: tuple[str, ...] = _DEFAULT_OWNER_PATTERNS


class GpuLeakDetector:
    """Stateful per-tick rule emitting ``gpu_memory_leaked``; the counter resets on any
    non-matching tick so a one-tick cold-start blip can't accumulate a false positive.
    """

    def __init__(
        self,
        config: GpuLeakConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the detector and restore the persisted hit counter.

        Args:
            config (GpuLeakConfig | None): Tunables; defaults to
                :class:`GpuLeakConfig` when ``None``.
            state_view (DetectorStateView | None): Disk-backed state view used
                to load/persist ``consecutive_hits`` across ticks.
        """
        self._config = config or GpuLeakConfig()
        self._state_view = state_view
        # Disk-backed counter so the multi-tick threshold survives the subprocess-per-tick transport.
        loaded = state_view.load() if state_view is not None else {}
        raw_hits = loaded.get("consecutive_hits", 0)
        try:
            self._consecutive_hits: int = max(0, int(raw_hits))
        except (TypeError, ValueError):
            self._consecutive_hits = 0

    @property
    def consecutive_hits(self) -> int:
        """Number of consecutive ticks the leak condition has held.

        Visible for tests; production code should not rely on this.

        Returns:
            int: The current consecutive-hit counter.
        """
        return self._consecutive_hits

    def _persist(self) -> None:
        """Write the current consecutive-hit counter to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save({"consecutive_hits": self._consecutive_hits})

    def evaluate(self, ctx: ReactorContext, data: SourceData) -> list[Symptom]:
        """Advance the leak counter and emit a symptom once it crosses threshold.

        Resets the counter on any tick whose conditions aren't met (missing GPU
        data, not all GPUs full, or a live owner process present).

        Args:
            ctx (ReactorContext): Reactor context for the current tick.
            data (SourceData): Collected source data including ``local_gpu`` and
                ``local_processes``.

        Returns:
            list[Symptom]: A single ``gpu_memory_leaked`` symptom once the
                consecutive-tick threshold is crossed, otherwise an empty list.
        """
        gpus = self._extract_gpu_snapshots(data)
        if not gpus:
            # No GPU data → reset so the next tick doesn't accumulate stale state.
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
        """Pull the list of per-GPU snapshot dicts from the source data.

        Args:
            data (SourceData): Collected source data.

        Returns:
            list[dict[str, Any]]: Per-GPU snapshot dicts, or an empty list when
                no usable GPU data is present.
        """
        gpus = data.local_gpu.get("gpus") if isinstance(data.local_gpu, dict) else None
        if not isinstance(gpus, list):
            return []
        return [snap for snap in gpus if isinstance(snap, dict)]

    def _is_full(self, snap: dict[str, Any]) -> bool:
        """Decide whether a single GPU snapshot counts as memory-full.

        A GPU is full when its memory utilization meets the configured percent
        threshold, or when its free VRAM falls to/under the free-MiB threshold.

        Args:
            snap (dict[str, Any]): A single per-GPU snapshot dict.

        Returns:
            bool: ``True`` if the GPU is considered full, otherwise ``False``.
        """
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
        """Find live processes that legitimately own GPU memory.

        Args:
            data (SourceData): Collected source data including
                ``local_processes``.

        Returns:
            list[dict[str, Any]]: Process dicts whose command line matches any
                configured owner pattern, possibly empty.
        """
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
        """Construct the ``gpu_memory_leaked`` symptom from current snapshots.

        Args:
            gpus (list[dict[str, Any]]): Per-GPU snapshot dicts for this tick.
            ctx (ReactorContext): Reactor context for the current tick.

        Returns:
            Symptom: A HIGH-severity symptom describing the suspected KFD/VRAM
                leak with per-GPU evidence and a recover suggestion.
        """
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
    """Adapt the stateful detector to the ``(ctx, data)`` entry point.

    Args:
        detector: The stateful GPU-leak detector instance.
        ctx: Reactor context for the current tick.
        data: Collected source data.

    Returns:
        The symptoms produced by the detector.
    """
    return detector.evaluate(ctx, data)


__all__ = [
    "GpuLeakConfig",
    "GpuLeakDetector",
    "evaluate_gpu_leak_signals",
]
