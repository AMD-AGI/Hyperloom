# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-effort Experience service reads for the Framework orchestration and specialist prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer.experience_collect import mapping_schema_ref
from hyperloom.inference_optimizer.experience_kb_service import REQUEST_TIMEOUT_SECONDS
from hyperloom_kb import ConfigurationError, RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.collect import MappingError

log = logging.getLogger(__name__)

_FRAMEWORK_DECISION = "Select the next framework optimization to benchmark."
_SPECIALIST_DECISION = "Propose framework optimizations for this specialist investigation."
# A free-text field over this many bytes reaches the prompt as a file under the session, not inline.
CONTENT_INLINE_LIMIT = 2048
# The injected records, rendered whole, stop before this many characters in every prompt that carries them.
RENDER_BUDGET_CHARS = 40_000
# A service or planner gateway this many reads in a row could not answer stays unread for the session, so a hung
# gateway costs a run a few read timeouts rather than one per orchestration turn and specialist dispatch.
READS_OFF_AFTER_FAILURES = 3
CONTENT_DIR = Path("experience_kb") / "contents"


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


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _patch_files(directory: Path, content: str) -> list[Path]:
    """Write each patch of a ``hyperloom-sbd-v6`` source change as its own file, ready to apply."""

    try:
        value = json.loads(content)
    # change.content is an opaque string from any writer or a pull, so it may be neither JSON nor shallow JSON.
    except (ValueError, RecursionError):
        return []
    patches = value.get("patches") if isinstance(value, dict) else None
    paths: list[Path] = []
    for index, patch in enumerate(patches if isinstance(patches, list) else [], start=1):
        text = patch.get("content") if isinstance(patch, dict) else None
        if not isinstance(text, str):
            continue
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(patch.get("path") or "")).name) or "change.patch"
        path = directory / f"{index}-{name}"
        _write_once(path, text)
        paths.append(path)
    return paths


def _materialize_contents(root: Path, contents: Iterable[Mapping[str, Any]]) -> str:
    """Write each ``change.content`` a read referenced instead of inlining, and return the block's file legend."""

    lines: list[str] = []
    for item in contents:
        ref, text = str(item.get("ref") or ""), item.get("content")
        digest = ref.removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not isinstance(text, str):
            continue
        path = root / f"{digest}.txt"
        try:
            _write_once(path, text)
            patches = _patch_files(root / digest, text)
        except OSError:
            log.warning("Experience KB content %s could not be written under %s", ref, root, exc_info=True)
            lines.append(f"- {ref} ({len(text.encode())} bytes): not available in this session")
            continue
        lines.append(f"- {ref} ({len(text.encode())} bytes): {path}")
        lines.extend(f"  - patch: {patch}" for patch in patches)
    if not lines:
        return ""
    return "\n".join(
        [
            "Each `<external content sha256:...>` above is that Experience's complete change.content, "
            "kept out of this prompt because of its size. Read its file when you need the change itself:",
            *lines,
        ]
    )


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

    def __init__(self, client: Any, session_dir: Path, schema_ref: str) -> None:
        self.client = client
        self.session_dir = Path(session_dir)
        # The service may hold several schemas; a run reads the one it writes.
        self.schema_ref = schema_ref
        self._cache_tick: int | None = None
        self._by_context: dict[str, ExperienceKBEvidence] = {}
        self._failed_in_a_row = 0

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
            config = RemoteConfig.from_env(values, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
            schema_ref = mapping_schema_ref()
        except (RemoteClientError, ConfigurationError, MappingError):
            log.exception("Experience service configuration is invalid; reads are disabled")
            return None
        if config is None:
            return None
        return cls(RemoteClient(config), Path(session_dir), schema_ref)

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
        if self._failed_in_a_row >= READS_OFF_AFTER_FAILURES:
            return ExperienceKBEvidence(
                tick=tick,
                read_id="",
                status="unavailable",
                prompt_block="",
                rendered_refs=(),
                warnings=("experience_kb_reads_off_for_session",),
            )
        result = self.client.read(
            decision,
            context,
            schema_ref=self.schema_ref,
            content_inline_limit=CONTENT_INLINE_LIMIT,
            render_budget_chars=RENDER_BUDGET_CHARS,
        )
        self._failed_in_a_row = 0 if result.status == "completed" else self._failed_in_a_row + 1
        if self._failed_in_a_row == READS_OFF_AFTER_FAILURES:
            log.warning(
                "Experience KB reads are off for the rest of this session after %d failed reads: %s",
                READS_OFF_AFTER_FAILURES,
                "; ".join(result.warnings) or result.status,
            )
        legend = _materialize_contents(self.session_dir / CONTENT_DIR, result.contents)
        evidence = ExperienceKBEvidence(
            tick=tick,
            read_id=result.read_id,
            status=result.status,
            prompt_block="\n\n".join(part for part in (result.prompt_block, legend) if part),
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


def _reads_off_reason(state: Any) -> str:
    """Why this run must not read Experiences, or ``""``: it reads only what the mapping would publish of it."""
    from hyperloom.common.perf_metric import GRADED_OUTPUT
    from hyperloom.inference_optimizer.breakdown.session_facts import grading_block

    # The schema cannot tell an agentic workload from a synthetic one: the mapping publishes no AgentX Experience,
    # so an AgentX run must not read the synthetic ones either.
    if agentx_active(benchmark_mode=getattr(state, "benchmark_mode", "")):
        return "its schema cannot represent an AgentX workload"
    objective = str(grading_block(state).get("objective") or "")
    if objective != GRADED_OUTPUT:
        return f"it is graded on {objective or 'no recorded objective'}, and Experiences record throughput only"
    return ""


def integration_for(owner: Any, session_dir: str | Path) -> ExperienceKBIntegration | None:
    """Return ``owner``'s one Experience service integration, bootstrapping it on first use."""
    if hasattr(owner, "_kb_integration"):
        return owner._kb_integration
    if reason := _reads_off_reason(owner.shared_state):
        log.info("Experience KB reads are off for this run: %s", reason)
        owner._kb_integration = None
    else:
        owner._kb_integration = ExperienceKBIntegration.from_env(session_dir)
    return owner._kb_integration


__all__ = ["ExperienceKBEvidence", "ExperienceKBIntegration", "integration_for"]
