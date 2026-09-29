# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-effort Experience service reads for the Framework orchestration and specialist prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_kb import RemoteClient, RemoteClientError, RemoteConfig

log = logging.getLogger(__name__)

_FRAMEWORK_DECISION = "Select the next framework optimization to benchmark."
_SPECIALIST_DECISION = "Propose framework optimizations for this specialist investigation."


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


def _bottleneck(state: Any) -> str:
    try:
        return str(state.current_top_bottleneck() or "").strip()
    except Exception:  # noqa: BLE001 — optional state helper
        return ""


@dataclass(frozen=True)
class ExperienceKBEvidence:
    tick: int
    read_id: str
    status: str
    prompt_block: str
    rendered_refs: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]
    experiences: tuple[dict[str, Any], ...] = ()


class ExperienceKBIntegration:
    """One fail-open Experience service client and decision-context read cache."""

    def __init__(self, client: Any, session_dir: Path) -> None:
        self.client = client
        self.session_dir = Path(session_dir)
        self._cache_tick: int | None = None
        self._by_context: dict[str, ExperienceKBEvidence] = {}

    @classmethod
    def from_env(
        cls,
        session_dir: str | Path,
        env: dict[str, str] | None = None,
    ) -> ExperienceKBIntegration | None:
        values = os.environ if env is None else env
        if not str(values.get("HYPERLOOM_KB_URL") or "").strip():
            return None
        try:
            config = RemoteConfig.from_env(values)
        except RemoteClientError:
            log.exception("Experience service configuration is invalid; reads are disabled")
            return None
        if config is None:
            return None
        return cls(RemoteClient(config), Path(session_dir))

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

    def _context(self, state: Any, observations: dict[str, str]) -> dict[str, Any]:
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
        return {
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
        }

    def build_context(self, state: Any, untested_proposals: str = "") -> dict[str, Any]:
        observations: dict[str, str] = {}
        bottleneck = _bottleneck(state)
        if bottleneck:
            observations["bottleneck"] = bottleneck
        if untested_proposals:
            observations["untested_directions"] = untested_proposals
        return self._context(state, observations)

    def build_specialist_context(self, state: Any, params: dict[str, Any]) -> dict[str, Any]:
        observations: dict[str, str] = {}
        bottleneck = _bottleneck(state)
        if bottleneck:
            observations["bottleneck"] = bottleneck
        pr_lead = params.get("pr_lead")
        for name, value in (
            ("specialist_domain", params.get("domain")),
            ("investigation", params.get("gap_symptom")),
            ("task", params.get("task_description")),
            ("upstream_pr", pr_lead.get("title") if isinstance(pr_lead, dict) else None),
        ):
            text = str(value or "").strip()
            if text:
                observations[name] = text
        return self._context(state, observations)

    def _read(self, state: Any, decision: str, context: dict[str, Any]) -> ExperienceKBEvidence:
        tick = int(getattr(state, "tick", 0) or 0)
        context_hash = hashlib.sha256(
            json.dumps(
                {"decision": decision, "context": context},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if tick != self._cache_tick:
            self._by_context.clear()
            self._cache_tick = tick
        cached = self._by_context.get(context_hash)
        if cached is not None:
            return cached
        result = self.client.read(decision, context)
        evidence = ExperienceKBEvidence(
            tick=tick,
            read_id=result.read_id,
            status=result.status,
            prompt_block=result.prompt_block,
            rendered_refs=tuple(item.to_dict() for item in result.rendered_refs),
            warnings=tuple(result.warnings),
            experiences=tuple(dict(item) for item in result.experiences),
        )
        self._by_context[context_hash] = evidence
        return evidence

    def read_for_framework(
        self,
        state: Any,
        *,
        untested_proposals: str = "",
    ) -> ExperienceKBEvidence:
        return self._read(state, _FRAMEWORK_DECISION, self.build_context(state, untested_proposals))

    def read_for_specialist(self, state: Any, params: dict[str, Any]) -> ExperienceKBEvidence:
        return self._read(state, _SPECIALIST_DECISION, self.build_specialist_context(state, params))


def integration_for(owner: Any, session_dir: str | Path) -> ExperienceKBIntegration | None:
    """Return ``owner``'s one Experience service integration, bootstrapping it on first use."""
    if not hasattr(owner, "_kb_integration"):
        owner._kb_integration = ExperienceKBIntegration.from_env(session_dir)
    return owner._kb_integration


__all__ = ["ExperienceKBEvidence", "ExperienceKBIntegration", "integration_for"]
