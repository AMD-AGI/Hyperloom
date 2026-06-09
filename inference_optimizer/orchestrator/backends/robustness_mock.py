# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mock Robustness backend — heartbeat-only, non-intervening.

Used in P0 main-path tests. Emits a heartbeat every tick, optionally one
``alert`` after N ticks (``alert_after_ticks``) to exercise the alert pipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...protocol.intent import Intent, IntentType
from .base import BackendTurnResult


@dataclass
class MockRobustnessBackend:
    """Heartbeat-only Robustness adapter. Implements :class:`Backend`."""

    name: str = "robustness-mock"
    alert_after_ticks: int | None = None
    alert_payload: dict[str, Any] = field(default_factory=lambda: {
        "severity": "low",
        "summary": "(mock robustness scheduled alert)",
    })

    def __post_init__(self) -> None:
        self._tick_count = 0
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        self._tick_count += 1
        self.calls.append({"prompt": prompt, "tick": self._tick_count})
        intents: list[Intent] = [
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={"topic": "heartbeat", "body_md": "ok (mock robustness)"},
            ),
        ]
        if self.alert_after_ticks is not None and self._tick_count == self.alert_after_ticks:
            intents.append(Intent(type=IntentType.ALERT, payload=dict(self.alert_payload)))
        return BackendTurnResult(intents=intents, raw_text="(mock robustness)")


__all__ = ["MockRobustnessBackend"]
