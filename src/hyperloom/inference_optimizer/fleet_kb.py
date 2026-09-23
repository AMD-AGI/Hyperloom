# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-effort Fleet KB read integration for the Framework decision boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DECISION = "Select the next framework optimization to benchmark."
_SUMMARY_LIMIT = 24_000
_RECENT_RESULT_LIMIT = 12
_ALREADY_TRIED_LIMIT = 32


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_value(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def _current_best_throughput(value: Any) -> float:
    current_best = value if isinstance(value, dict) else {}
    for name in ("tput", "output_throughput"):
        throughput = current_best.get(name)
        if not isinstance(throughput, bool) and isinstance(throughput, (int, float)) and throughput > 0:
            return float(throughput)
    return 0.0


@dataclass(frozen=True)
class FleetKBEvidence:
    tick: int
    read_id: str
    status: str
    prompt_block: str
    rendered_refs: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]


class FleetKBIntegration:
    """One fail-open Fleet KB client and decision-context read cache."""

    def __init__(self, client: Any, session_dir: Path) -> None:
        self.client = client
        self.session_dir = Path(session_dir)
        self._by_context: dict[str, FleetKBEvidence] = {}

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
        if not isinstance(task, dict):
            metadata = value.get("metadata")
            task = metadata.get("task_config") if isinstance(metadata, dict) else None
        return dict(task) if isinstance(task, dict) else {}

    def build_context(self, state: Any, untested_proposals: str = "") -> dict[str, Any]:
        manifest = self._manifest_context()
        architecture = manifest.get("architecture")
        architecture = architecture if isinstance(architecture, dict) else {}
        architectures = architecture.get("architectures")
        architecture_name = (
            architectures[0] if isinstance(architectures, list) and architectures else architecture.get("architecture")
        )
        identity_sources = {
            "model": _first_value(
                getattr(state, "model_name", None),
                manifest.get("model_name"),
            ),
            "gpu": _first_value(
                getattr(state, "gpu_type", None),
                manifest.get("gpu_type"),
            ),
            "framework": _first_value(
                getattr(state, "framework", None),
                manifest.get("framework_name"),
            ),
            "model_type": _first_value(
                getattr(state, "model_type", None),
                architecture.get("model_type"),
            ),
            "architecture": _first_value(
                getattr(state, "architecture", None),
                architecture_name,
            ),
            "framework_version": _first_value(
                getattr(state, "framework_version", None),
                manifest.get("framework_version"),
            ),
            "precision": _first_value(
                getattr(state, "precision", None),
                manifest.get("precision"),
            ),
        }
        identity = {name: _json_safe(value) for name, value in identity_sources.items() if value not in (None, "")}
        workload: dict[str, Any] = {}
        for name in (
            "tp",
            "ep",
            "conc",
            "isl",
            "osl",
            "max_model_len",
            "compute_partition_mode",
            "partitions",
        ):
            value = _first_value(getattr(state, name, None), manifest.get(name))
            if value not in (None, ""):
                workload[name] = _json_safe(value)
        baseline_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
        current_best = getattr(state, "current_best", None) or {}
        current_best_tput = float(
            getattr(state, "current_best_tput", 0.0)
            or getattr(state, "best_tput", 0.0)
            or _current_best_throughput(current_best)
            or 0.0
        )
        observations: dict[str, str] = {}
        try:
            bottleneck = str(state.current_top_bottleneck() or "").strip()
        except Exception:  # noqa: BLE001 — optional state helper
            bottleneck = ""
        if bottleneck:
            observations["bottleneck"] = bottleneck
        if untested_proposals:
            observations["untested_directions"] = untested_proposals
        try:
            summary = str(state.to_prompt_summary())
        except Exception:  # noqa: BLE001 — advisory read must fail open
            summary = ""
        if summary:
            observations["session_summary"] = summary[:_SUMMARY_LIMIT]
        recent_results = list(getattr(state, "attempts_history", None) or [])
        already_tried = list(getattr(state, "optimization_stack", None) or [])
        context: dict[str, Any] = {
            "identity": identity,
            "workload": workload,
            "objective": {
                "id": "e2e_throughput@v1",
                "direction": "higher_is_better",
            },
            "benchmark_baseline": {
                "throughput": baseline_tput,
                "source": "original_recipe_measurement",
            },
            "current_best": {
                "configuration": _json_safe(current_best),
                "throughput": current_best_tput,
            },
            "observations": observations,
            "recent_results": _json_safe(recent_results[-_RECENT_RESULT_LIMIT:]),
            "already_tried": _json_safe(already_tried[-_ALREADY_TRIED_LIMIT:]),
        }
        return context

    def read_for_framework(
        self,
        state: Any,
        *,
        untested_proposals: str = "",
    ) -> FleetKBEvidence:
        tick = int(getattr(state, "tick", 0) or 0)
        context = self.build_context(state, untested_proposals)
        context_hash = hashlib.sha256(
            json.dumps(
                {"decision": _DECISION, "context": context},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        cached = self._by_context.get(context_hash)
        if cached is not None:
            return replace(cached, tick=tick)
        session_id = str(getattr(state, "session_id", "") or self.session_dir.name)
        operation_id = f"fleet-read-{session_id}-{context_hash[:16]}"
        result = self.client.read(
            _DECISION,
            context,
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
        self._by_context[context_hash] = evidence
        return evidence


__all__ = ["FleetKBEvidence", "FleetKBIntegration"]
