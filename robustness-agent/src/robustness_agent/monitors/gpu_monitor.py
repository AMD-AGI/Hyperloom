"""GPU monitor — detects VRAM leaks, temperature spikes, ECC errors, utilization drops."""

from __future__ import annotations

import logging
import time

from ..config import Config
from ..models import Alert, GpuSnapshot, Severity
from ..providers.base import MetricsProvider

log = logging.getLogger(__name__)


class GpuMonitor:
    """Stateful GPU health monitor with trend detection."""

    def __init__(self, config: Config, provider: MetricsProvider):
        """Initialise the GPU monitor.

        Args:
            config (Config): Agent configuration carrying GPU thresholds.
            provider (MetricsProvider): Source of per-GPU metric
                snapshots.
        """
        self._config = config
        self._provider = provider
        self._prev_snapshots: dict[int, GpuSnapshot] = {}
        self._baseline_util: dict[int, float] = {}

    async def check(self) -> list[Alert]:
        """Run all GPU health checks against the latest snapshots.

        Returns:
            list[Alert]: Alerts raised for VRAM, temperature, ECC, and
            utilization-drop conditions across all GPUs.
        """
        alerts: list[Alert] = []
        snapshots = await self._provider.get_gpu_metrics()

        for snap in snapshots:
            self._check_vram(snap, alerts)
            self._check_temperature(snap, alerts)
            self._check_ecc(snap, alerts)
            await self._check_utilization_drop(snap, alerts)
            self._prev_snapshots[snap.gpu_id] = snap

        return alerts

    def set_baseline_utilization(self, gpu_id: int, util: float) -> None:
        """Record a baseline utilization for drop detection.

        Args:
            gpu_id (int): The GPU index.
            util (float): Baseline utilization percentage to compare
                future samples against.
        """
        self._baseline_util[gpu_id] = util

    def _check_vram(self, snap: GpuSnapshot, alerts: list[Alert]) -> None:
        """Append VRAM warning/critical alerts for a snapshot.

        Args:
            snap (GpuSnapshot): The current GPU snapshot.
            alerts (list[Alert]): Mutable list that matching alerts are
                appended to.
        """
        pct = snap.vram_used_pct
        if pct >= self._config.gpu_vram_crit_pct:
            alerts.append(Alert(
                check_name="gpu_vram_critical",
                severity=Severity.CRITICAL,
                summary=f"GPU {snap.gpu_id} VRAM at {pct:.1f}% "
                        f"({snap.vram_used_mb:.0f}/{snap.vram_total_mb:.0f} MB)",
                evidence={"gpu_id": snap.gpu_id, "vram_pct": pct},
                timestamp=time.time(),
            ))
        elif pct >= self._config.gpu_vram_warn_pct:
            alerts.append(Alert(
                check_name="gpu_vram_warning",
                severity=Severity.WARNING,
                summary=f"GPU {snap.gpu_id} VRAM at {pct:.1f}%",
                evidence={"gpu_id": snap.gpu_id, "vram_pct": pct},
                timestamp=time.time(),
            ))

    def _check_temperature(self, snap: GpuSnapshot, alerts: list[Alert]) -> None:
        """Append a temperature alert when the warn threshold is crossed.

        Args:
            snap (GpuSnapshot): The current GPU snapshot.
            alerts (list[Alert]): Mutable list that matching alerts are
                appended to.
        """
        if snap.temperature_c >= self._config.gpu_temp_warn_c:
            alerts.append(Alert(
                check_name="gpu_temperature_high",
                severity=Severity.WARNING,
                summary=f"GPU {snap.gpu_id} temperature at {snap.temperature_c:.0f}C",
                evidence={"gpu_id": snap.gpu_id, "temp_c": snap.temperature_c},
                timestamp=time.time(),
            ))

    def _check_ecc(self, snap: GpuSnapshot, alerts: list[Alert]) -> None:
        """Append a critical alert when new ECC errors appear.

        Compares against the previous snapshot to count only newly
        observed ECC errors.

        Args:
            snap (GpuSnapshot): The current GPU snapshot.
            alerts (list[Alert]): Mutable list that matching alerts are
                appended to.
        """
        prev = self._prev_snapshots.get(snap.gpu_id)
        if snap.ecc_errors > 0:
            new_errors = snap.ecc_errors
            if prev:
                new_errors = snap.ecc_errors - prev.ecc_errors
            if new_errors > 0:
                alerts.append(Alert(
                    check_name="gpu_ecc_error",
                    severity=Severity.CRITICAL,
                    summary=f"GPU {snap.gpu_id}: {new_errors} new ECC error(s) "
                            f"(total: {snap.ecc_errors})",
                    evidence={"gpu_id": snap.gpu_id, "new": new_errors, "total": snap.ecc_errors},
                    timestamp=time.time(),
                ))

    async def _check_utilization_drop(
        self, snap: GpuSnapshot, alerts: list[Alert],
    ) -> None:
        """Append an alert when utilization drops below the baseline.

        No-op until a meaningful baseline (>= 10%) has been recorded for
        the GPU.

        Args:
            snap (GpuSnapshot): The current GPU snapshot.
            alerts (list[Alert]): Mutable list that matching alerts are
                appended to.
        """
        baseline = self._baseline_util.get(snap.gpu_id)
        if baseline is None or baseline < 10:
            return
        drop_pct = ((baseline - snap.utilization) / baseline) * 100
        if drop_pct >= self._config.gpu_util_drop_pct:
            alerts.append(Alert(
                check_name="gpu_utilization_drop",
                severity=Severity.CRITICAL,
                summary=f"GPU {snap.gpu_id} utilization dropped {drop_pct:.0f}% "
                        f"(baseline={baseline:.0f}%, current={snap.utilization:.0f}%)",
                evidence={
                    "gpu_id": snap.gpu_id,
                    "baseline": baseline,
                    "current": snap.utilization,
                    "drop_pct": drop_pct,
                },
                timestamp=time.time(),
            ))
