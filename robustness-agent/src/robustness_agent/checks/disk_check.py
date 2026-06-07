# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Disk usage check — detects low disk space and Triton cache bloat."""

from __future__ import annotations

import logging
import time

from ..config import Config
from ..models import Alert, Severity
from ..providers.base import MetricsProvider

log = logging.getLogger(__name__)


class DiskCheck:

    def __init__(self, config: Config, provider: MetricsProvider):
        self._config = config
        self._provider = provider

    async def check(self) -> list[Alert]:
        alerts: list[Alert] = []
        disks = await self._provider.get_disk_usage(str(self._config.session_dir))
        for d in disks:
            if d.used_pct >= self._config.disk_usage_crit_pct:
                alerts.append(Alert(
                    check_name="disk_critical",
                    severity=Severity.CRITICAL,
                    summary=f"Disk {d.mount} at {d.used_pct:.1f}% "
                            f"({d.available_gb:.1f}G free)",
                    evidence={"mount": d.mount, "used_pct": d.used_pct, "avail_gb": d.available_gb},
                    timestamp=time.time(),
                ))
            elif d.used_pct >= self._config.disk_usage_warn_pct:
                alerts.append(Alert(
                    check_name="disk_warning",
                    severity=Severity.WARNING,
                    summary=f"Disk {d.mount} at {d.used_pct:.1f}%",
                    evidence={"mount": d.mount, "used_pct": d.used_pct, "avail_gb": d.available_gb},
                    timestamp=time.time(),
                ))
        return alerts
