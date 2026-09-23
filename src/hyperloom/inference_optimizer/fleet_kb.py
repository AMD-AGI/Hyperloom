# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-effort Fleet KB read integration for the Framework decision boundary."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DECISION = "Select the next framework optimization to benchmark."
_SUMMARY_LIMIT = 24_000


@dataclass(frozen=True)
class FleetKBEvidence:
    tick: int
    read_id: str
    status: str
    prompt_block: str
    rendered_refs: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]


class FleetKBIntegration:
    """One fail-open Fleet KB client and per-tick read cache."""

    def __init__(self, client: Any, session_dir: Path) -> None:
        self.client = client
        self.session_dir = Path(session_dir)
        self._by_tick: dict[int, FleetKBEvidence] = {}

    @classmethod
    def from_env(
        cls,
        session_dir: str | Path,
        env: dict[str, str] | None = None,
    ) -> FleetKBIntegration | None:
        values = os.environ if env is None else env
        if not str(values.get("HYPERLOOM_FLEET_KB_URL") or "").strip():
            return None
        try:
            module = import_module("hyperloom_kb")
            config = module.FleetClientConfig.from_env(values)
            if config is None:
                return None
            return cls(module.FleetKBClient(config), Path(session_dir))
        except (ImportError, RuntimeError, ValueError):
            log.exception("Fleet KB client bootstrap failed; reads are disabled")
            return None

    def _manifest_context(self) -> dict[str, Any]:
        path = self.session_dir / "manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(value, dict):
            return {}
        task = value.get("task_config")
        return dict(task) if isinstance(task, dict) else {}

    def build_context(self, state: Any, untested_proposals: str = "") -> dict[str, Any]:
        manifest = self._manifest_context()
        context: dict[str, Any] = {
            "model": str(getattr(state, "model_name", "") or manifest.get("model_name") or ""),
            "phase": str(getattr(state, "phase", "") or ""),
            "baseline_tput": float(getattr(state, "baseline_tput", 0.0) or 0.0),
            "current_best": str(getattr(state, "current_best", "") or ""),
            "current_action": str(getattr(state, "current_action", "") or ""),
            "macro_cycle": int(getattr(state, "macro_cycle", 0) or 0),
            "tick": int(getattr(state, "tick", 0) or 0),
        }
        for name in (
            "gpu_type",
            "framework_name",
            "framework_version",
            "precision",
            "tp",
            "ep",
            "conc",
            "isl",
            "osl",
            "max_model_len",
        ):
            value = manifest.get(name)
            if value not in (None, ""):
                context[name] = value
        if untested_proposals:
            context["untested_proposals"] = untested_proposals
        try:
            summary = str(state.to_prompt_summary())
        except Exception:  # noqa: BLE001 — advisory read must fail open
            summary = ""
        if summary:
            context["session_state"] = summary[:_SUMMARY_LIMIT]
        return context

    def read_for_framework(
        self,
        state: Any,
        *,
        untested_proposals: str = "",
    ) -> FleetKBEvidence:
        tick = int(getattr(state, "tick", 0) or 0)
        cached = self._by_tick.get(tick)
        if cached is not None:
            return cached
        session_id = str(getattr(state, "session_id", "") or self.session_dir.name)
        operation_id = f"fleet-read-{session_id}-{int(getattr(state, 'macro_cycle', 0) or 0)}-{tick}"
        result = self.client.read(
            _DECISION,
            self.build_context(state, untested_proposals),
            operation_id=operation_id,
            run_id=session_id,
        )
        evidence = FleetKBEvidence(
            tick=tick,
            read_id=result.read_id,
            status=result.status,
            prompt_block=result.prompt_block,
            rendered_refs=tuple(item.to_dict() for item in result.rendered_refs),
            warnings=tuple(result.warnings),
        )
        self._by_tick[tick] = evidence
        return evidence


__all__ = ["FleetKBEvidence", "FleetKBIntegration"]
