# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU memory leak detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData

if TYPE_CHECKING:
    from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity


# Inference-server / benchmark owners whose presence proves VRAM use is legitimate.
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
    """Tunables for :class:`GpuLeakDetector`."""

    util_mem_pct_threshold: float = 99.0
    free_mb_threshold: float = 500.0
    min_consecutive_ticks: int = 2
    owner_patterns: tuple[str, ...] = _DEFAULT_OWNER_PATTERNS


class GpuLeakDetector:
    """Stateful per-tick rule emitting ``gpu_memory_leaked``; the counter resets on any non-matching tick so a one-tick cold-start blip can't accumulate a false positive."""

    def __init__(
        self,
        config: GpuLeakConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the detector and restore the persisted hit counter."""
        self._config = config or GpuLeakConfig()
        self._state_view = state_view
        # Disk-backed counter so the multi-tick threshold survives ticks.
        loaded = state_view.load() if state_view is not None else {}
        raw_hits = loaded.get("consecutive_hits", 0)
        try:
            self._consecutive_hits: int = max(0, int(raw_hits))
        except (TypeError, ValueError):
            self._consecutive_hits = 0

    def _persist(self) -> None:
        """Write the current consecutive-hit counter to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save({"consecutive_hits": self._consecutive_hits})

    def evaluate(self, ctx: ReactorContext, data: SourceData) -> list[Symptom]:
        """Advance the leak counter and emit a symptom once it crosses threshold."""
        gpus = self._extract_gpu_snapshots(data)
        if not gpus:
            # No GPU data → reset.
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

    # internals
    def _extract_gpu_snapshots(self, data: SourceData) -> list[dict[str, Any]]:
        """Pull the list of per-GPU snapshot dicts from the source data."""
        gpus = data.local_gpu.get("gpus") if isinstance(data.local_gpu, dict) else None
        if not isinstance(gpus, list):
            return []
        return [snap for snap in gpus if isinstance(snap, dict)]

    def _is_full(self, snap: dict[str, Any]) -> bool:
        """Decide whether a single GPU snapshot counts as memory-full."""
        cfg = self._config
        util_mem = snap.get("util_mem_pct")
        if isinstance(util_mem, (int, float)) and util_mem >= cfg.util_mem_pct_threshold:
            return True
        used = snap.get("vram_used_mb")
        total = snap.get("vram_total_mb")
        if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
            free_mb = max(0.0, float(total) - float(used))
            if free_mb <= cfg.free_mb_threshold:
                return True
        return False

    def _live_owners(self, data: SourceData) -> list[dict[str, Any]]:
        """Find live processes that legitimately own GPU memory."""
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
        """Construct the ``gpu_memory_leaked`` symptom from current snapshots."""
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


__all__ = [
    "GpuLeakConfig",
    "GpuLeakDetector",
]
