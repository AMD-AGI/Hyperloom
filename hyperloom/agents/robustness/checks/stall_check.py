"""Agent stall detection — monitors if other agents stop producing events."""

from __future__ import annotations

import logging
import time

from ..config import Config
from ..conductor import ConductorReader
from ..models import Alert, Severity

log = logging.getLogger(__name__)


class StallCheck:

    def __init__(self, config: Config, conductor: ConductorReader):
        self._config = config
        self._conductor = conductor

    async def check(self) -> list[Alert]:
        alerts: list[Alert] = []
        activity = self._conductor.get_agent_last_activity()
        now = time.time()

        for agent_name, last_ts in activity.items():
            if agent_name == "robustness":
                continue
            elapsed = now - last_ts
            if elapsed > self._config.agent_stall_timeout_s:
                alerts.append(Alert(
                    check_name="agent_stall",
                    severity=Severity.CRITICAL,
                    summary=f"Agent '{agent_name}' has not produced events for "
                            f"{elapsed:.0f}s (timeout={self._config.agent_stall_timeout_s}s)",
                    evidence={
                        "agent": agent_name,
                        "last_event_ts": last_ts,
                        "elapsed_s": elapsed,
                    },
                    timestamp=now,
                ))
        return alerts
