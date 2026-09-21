# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project measured SBD V6 Framework attempts into canonical Experiences."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    is_secret_shaped_env_name,
    redact_secret_values,
)

log = logging.getLogger(__name__)

_ENABLE_ENV = "HYPERLOOM_KB_ENABLE"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}
_RECEIPT_REL = Path("reports") / "experience_v1_publish.json"
_MAX_PATCH_BYTES = 128 * 1024
_MAX_PATCH_TOTAL_BYTES = 256 * 1024
_MAX_BASELINE_ARGS_CHARS = 64 * 1024
_MAX_BASELINE_ENVS = 256
_MAX_BASELINE_ENV_VALUE_CHARS = 16 * 1024
_MAX_BASELINE_CONTROL_ITEMS = 256
_GENERIC_REASONING = {
    "improve performance",
    "improve throughput",
    "optimize",
    "optimization",
    "test",
    "try",
}
_TERMINAL_OUTCOMES = {
    "FAILED",
    "KEEP",
    "KEEP_UNSTABLE",
    "KEPT",
    "KILLED_OVERTIME",
    "REVERT",
    "REVERTED",
}
_FAILED_OUTCOMES = {"FAILED", "KILLED_OVERTIME"}
_CANDIDATE_FAILURE_ATTRIBUTION = "candidate_caused"
_ACTION_TIME_REASONING_PREFIXES = (
    "action_payload.",
    "action_params.",
    "candidate.",
    "generated_variant.",
    "proposal.",
)
_SECRET_ARG_RE = re.compile(
    r"(?i)(?:^|\s)--?[^\s=]*(?:api[-_]?key|token|secret|password|credential)"
    r"(?:=|\s)"
)


class ProjectionError(ValueError):
    """Raised when one attempt cannot produce a faithful complete Experience."""


@dataclass(frozen=True)
class ProjectedAttempt:
    attempt_id: str
    run_id: str
    seq: int
    created_at: datetime
    completed_at: datetime
    identity: dict[str, Any]
    baseline_identity: dict[str, Any]
    baseline_configuration: dict[str, Any]
    baseline_value: float
    reasoning: str
    reasoning_origin: str
    change_family: str
    change_fingerprint: str
    change_summary: str
    change_content: str
    resource_refs: tuple[str, ...]
    decision: str
    outcome_value: float | None
    error_class: str
    failure_attribution: str
    constraints: tuple[tuple[str, bool, Any], ...]
    reflection: str
    provenance_extra: dict[str, Any]

    def review_row(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "ready": True,
            "identity": dict(self.identity),
            "baseline_identity": dict(self.baseline_identity),
            "baseline_configuration": dict(self.baseline_configuration),
            "baseline_value": self.baseline_value,
            "reasoning_origin": self.reasoning_origin,
            "change_identity": {
                "change_family": self.change_family,
                "change_fingerprint": self.change_fingerprint,
            },
            "decision": self.decision,
            "outcome_value": self.outcome_value,
            "failure_attribution": self.failure_attribution,
            "constraints": [
                {"name": name, "passed": passed, "value": value} for name, passed, value in self.constraints
            ],
        }


def enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    value = (values.get(_ENABLE_ENV) or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    allowed = ", ".join(sorted(_TRUE | (_FALSE - {""})))
    raise ValueError(f"{_ENABLE_ENV} must be one of {allowed}; got {value!r}")


def validate_experience_config(env: dict[str, str] | None = None) -> None:
    """Fail before optimizer setup when explicit Experience config is invalid."""

    if not enabled(env):
        return
    module = import_module("hyperloom_kb")
    kb = module.experience_kb_from_env(env)
    if not kb.enabled:
        raise ValueError(f"{_ENABLE_ENV}=true requires a configured Hyperloom-KB backend")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parse_time(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _token(value: Any, *, name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:@/-]+", "_", _text(value)).strip("_")
    if not normalized:
        raise ProjectionError(f"{name} is missing")
    return normalized[:200]


def _identity(breakdown: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(breakdown.get("metadata"))
    task = _mapping(metadata.get("task_config"))
    architecture = _mapping(task.get("architecture"))
    architectures = architecture.get("architectures")
    architecture_name = (
        _text(architectures[0])
        if isinstance(architectures, list) and architectures
        else _text(architecture.get("model_class") or architecture.get("model_family"))
    )
    required = {
        "model": _text(task.get("model_name")),
        "gpu": _text(task.get("gpu_type")),
        "framework": _text(task.get("framework_name")),
        "model_type": _text(architecture.get("model_type")),
        "architecture": architecture_name,
        "framework_version": _text(task.get("framework_version")),
        "precision": _text(task.get("precision")),
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise ProjectionError(f"identity missing required fields: {', '.join(missing)}")
    identity: dict[str, Any] = dict(required)
    for name in ("tp", "conc", "isl", "osl", "max_model_len"):
        value = task.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            identity[name] = value
    ep = task.get("ep")
    if isinstance(ep, int) and not isinstance(ep, bool) and ep > 1:
        identity["ep"] = ep
    partition = _mapping(task.get("compute_partition"))
    partition_mode = _text(partition.get("mode")).upper()
    if partition_mode:
        identity["compute_partition_mode"] = partition_mode
    partitions = partition.get("partitions")
    if isinstance(partitions, int) and not isinstance(partitions, bool) and partitions > 1:
        identity["partitions"] = partitions
    return identity


def _session(breakdown: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(breakdown.get("metadata")).get("session"))


def _agentx_block_reason(breakdown: Mapping[str, Any]) -> str:
    grading = _mapping(_mapping(breakdown.get("metadata")).get("grading"))
    return (
        "agentx_experience_identity_not_supported" if _text(grading.get("benchmark_mode")).lower() == "agentx" else ""
    )


def _framework_events(breakdown: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    timeline = breakdown.get("timeline")
    if not isinstance(timeline, list):
        return ()
    return tuple(
        item for item in timeline if isinstance(item, Mapping) and _text(item.get("type")) == "framework_agent"
    )


def _event_id(event: Mapping[str, Any]) -> str:
    return _text(event.get("event") or event.get("event_id") or event.get("id"))


def _reasoning(attempt: Mapping[str, Any], proposal: Mapping[str, Any]) -> tuple[str, str]:
    attempt_value = _text(attempt.get("reasoning"))
    attempt_origin = _text(attempt.get("reasoning_origin"))
    proposal_value = _text(proposal.get("reasoning"))
    if attempt_value and any(attempt_origin.startswith(prefix) for prefix in _ACTION_TIME_REASONING_PREFIXES):
        value = attempt_value
        origin = attempt_origin
    elif proposal_value:
        value = proposal_value
        origin = "proposal.reasoning"
    else:
        value = attempt_value
        origin = attempt_origin
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) < 20 or normalized.lower().rstrip(".") in _GENERIC_REASONING:
        raise ProjectionError("decision reasoning is missing or non-specific")
    if not any(origin.startswith(prefix) for prefix in _ACTION_TIME_REASONING_PREFIXES):
        raise ProjectionError("decision reasoning is not traceable to the action-time proposal")
    return normalized, origin


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"value is not canonical JSON: {exc}") from exc


def _safe_server_args(value: Any, *, field: str) -> str:
    server_args = _text(value)
    if len(server_args) > _MAX_BASELINE_ARGS_CHARS:
        raise ProjectionError(f"{field} exceeds collection limit")
    if _SECRET_ARG_RE.search(server_args) or redact_secret_values(server_args) != server_args:
        raise ProjectionError(f"{field} contains credential material")
    return server_args


def _safe_envs(value: Any, *, field: str) -> dict[str, str]:
    raw = _mapping(value)
    if len(raw) > _MAX_BASELINE_ENVS:
        raise ProjectionError(f"{field} exceeds collection limit")
    envs: dict[str, str] = {}
    for key, value in raw.items():
        name = _text(key)
        text = str(value)
        if len(text) > _MAX_BASELINE_ENV_VALUE_CHARS:
            raise ProjectionError(f"{field} value exceeds collection limit: {name}")
        if (
            not name
            or name.upper() in BENCHMARK_SECRET_ENV_NAMES
            or is_secret_shaped_env_name(name)
            or redact_secret_values(text) != text
        ):
            raise ProjectionError(f"{field} contains unsafe key: {name or '<empty>'}")
        envs[name] = text
    return dict(sorted(envs.items()))


def _baseline_configuration(attempt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(attempt.get("measured_against"), Mapping):
        raise ProjectionError("attempt has no measured-against configuration")
    measured_against = _mapping(attempt.get("measured_against"))
    server_args = _safe_server_args(
        measured_against.get("extra_server_args"),
        field="measured-against server args",
    )
    extra_envs = _safe_envs(
        measured_against.get("extra_envs"),
        field="measured-against environment",
    )
    remove_args = sorted({_text(item) for item in (measured_against.get("remove_args") or []) if _text(item)})
    unset_envs = sorted({_text(item) for item in (measured_against.get("unset_envs") or []) if _text(item)})
    if max(len(remove_args), len(unset_envs)) > _MAX_BASELINE_CONTROL_ITEMS:
        raise ProjectionError("measured-against remove/unset controls exceed collection limit")
    args_mode = _text(measured_against.get("args_mode") or "append").lower()
    if args_mode not in {"append", "replace"}:
        raise ProjectionError(f"measured-against has unsupported args_mode: {args_mode}")
    return {
        "extra_server_args": server_args,
        "extra_envs": extra_envs,
        "remove_args": remove_args,
        "unset_envs": unset_envs,
        "args_mode": args_mode,
    }


def _baseline_identity(configuration: Mapping[str, Any]) -> dict[str, str]:
    encoded = _canonical_json(configuration)
    return {"baseline_fingerprint": hashlib.sha256(encoded.encode()).hexdigest()}


def _config_delta(attempt: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(attempt.get("config_delta"))
    server_args = _safe_server_args(
        raw.get("extra_server_args"),
        field="config delta server args",
    )
    extra_envs = _safe_envs(
        raw.get("extra_envs"),
        field="config delta environment",
    )
    remove_args = sorted({_text(item) for item in (raw.get("remove_args") or []) if _text(item)})
    unset_envs = sorted({_text(item) for item in (raw.get("unset_envs") or []) if _text(item)})
    if max(len(remove_args), len(unset_envs)) > _MAX_BASELINE_CONTROL_ITEMS:
        raise ProjectionError("config delta remove/unset controls exceed collection limit")
    args_mode = _text(raw.get("args_mode") or "append").lower()
    if args_mode not in {"append", "replace"}:
        raise ProjectionError(f"config delta has unsupported args_mode: {args_mode}")
    if not any((server_args, extra_envs, remove_args, unset_envs, args_mode == "replace")):
        raise ProjectionError("config attempt has no effective config_delta")
    return {
        "extra_server_args": server_args,
        "extra_envs": extra_envs,
        "remove_args": remove_args,
        "unset_envs": unset_envs,
        "args_mode": args_mode,
    }


def _relative_file(path_value: Any, session_dir: Path) -> tuple[str, str, str]:
    raw = _text(path_value)
    if not raw:
        return "", "", ""
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (session_dir / path).resolve()
    try:
        relative = resolved.relative_to(session_dir).as_posix()
    except ValueError:
        return "", "", ""
    try:
        if not resolved.is_file() or resolved.stat().st_size > _MAX_PATCH_BYTES:
            return "", "", ""
        payload = resolved.read_bytes()
    except OSError:
        return "", "", ""
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError:
        return "", "", ""
    if redact_secret_values(content) != content:
        return "", "", ""
    digest = hashlib.sha256(payload).hexdigest()
    return relative, digest, content


def _change(
    attempt: Mapping[str, Any],
    proposal: Mapping[str, Any],
    session_dir: Path,
) -> tuple[str, str, str, str, tuple[str, ...]]:
    arm = _text(attempt.get("arm")).lower()
    if arm == "config":
        delta = _config_delta(attempt)
        content = _canonical_json(delta)
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        summary = _text(attempt.get("variant_name")) or "Framework configuration variant"
        return "config_variant", fingerprint, summary, content, ()
    if arm != "source":
        raise ProjectionError(f"unsupported Framework attempt arm: {arm or 'missing'}")

    patch_paths = [
        attempt.get("patch_path"),
        *(attempt.get("patches_applied") or []),
        *(attempt.get("patches_reverted") or []),
    ]
    patch_material: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for path in patch_paths:
        relative, digest, content = _relative_file(path, session_dir)
        if relative and digest and relative not in seen_paths:
            patch_material.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "content": content,
                }
            )
            seen_paths.add(relative)
    if sum(len(item["content"].encode()) for item in patch_material) > _MAX_PATCH_TOTAL_BYTES:
        raise ProjectionError("source attempt patch material exceeds collection limit")
    if len(patch_material) == 1:
        fingerprint = patch_material[0]["sha256"]
    elif patch_material:
        fingerprint = hashlib.sha256(_canonical_json(patch_material).encode()).hexdigest()
    else:
        fingerprint = ""
    if not fingerprint:
        raise ProjectionError("source attempt has no durable patch material")
    details = {
        "source_ref": _text(attempt.get("source_ref") or proposal.get("source_ref")),
        "target_files": sorted(_text(item) for item in (attempt.get("target_files") or []) if _text(item)),
        "patches": patch_material,
        "patch_sha256": fingerprint,
    }
    summary = _text(proposal.get("title") or attempt.get("candidate_id"))
    if not summary:
        raise ProjectionError("source attempt has no change summary")
    return (
        "source_patch",
        fingerprint,
        summary,
        _canonical_json(details),
        tuple(item["path"] for item in patch_material),
    )


def _decision(attempt: Mapping[str, Any]) -> tuple[str, float | None, str, str]:
    outcome = _text(attempt.get("outcome") or attempt.get("decision")).upper()
    if outcome not in _TERMINAL_OUTCOMES:
        raise ProjectionError("attempt has no supported terminal outcome")
    measurement = _mapping(attempt.get("measurement"))
    before = _number(measurement.get("before_tput"))
    after = _number(measurement.get("after_tput"))
    if before is None or before <= 0:
        raise ProjectionError("attempt has no valid measured baseline")
    if outcome in _FAILED_OUTCOMES:
        failure = _mapping(attempt.get("failure"))
        error_class = _token(
            failure.get("error_class") or outcome.lower(),
            name="failure.error_class",
        )
        attribution = _text(failure.get("attribution")).lower()
        if attribution != _CANDIDATE_FAILURE_ATTRIBUTION:
            raise ProjectionError(
                f"failed attempt is not candidate-attributed (failure_attribution={attribution or 'missing'})"
            )
        return "failed", after, error_class, attribution
    if after is None or after <= 0:
        raise ProjectionError("measured attempt has no valid after_tput")
    if outcome in {"KEEP", "KEPT"}:
        accuracy = _mapping(attempt.get("accuracy"))
        if accuracy.get("required") is True and accuracy.get("passed") is not True:
            raise ProjectionError("kept attempt did not pass required accuracy")
        return "keep", after, "", ""
    return "revert", after, "", ""


def _constraints(attempt: Mapping[str, Any]) -> tuple[tuple[str, bool, Any], ...]:
    constraints: list[tuple[str, bool, Any]] = []
    for row in attempt.get("gates") or []:
        if not isinstance(row, Mapping):
            continue
        name = _text(row.get("gate"))
        passed = row.get("passed")
        if name and isinstance(passed, bool):
            constraints.append((name, passed, row.get("observed")))
    accuracy = _mapping(attempt.get("accuracy"))
    if isinstance(accuracy.get("passed"), bool) and not any(name == "accuracy" for name, _, _ in constraints):
        constraints.append(("accuracy", bool(accuracy["passed"]), accuracy.get("value")))
    return tuple(constraints)


def _project(
    *,
    breakdown: Mapping[str, Any],
    event: Mapping[str, Any],
    attempt: Mapping[str, Any],
    proposal: Mapping[str, Any],
    session_dir: Path,
) -> ProjectedAttempt:
    attempt_id = _token(attempt.get("attempt_id"), name="attempt_id")
    session = _session(breakdown)
    run_id = _token(session.get("session_id"), name="metadata.session.session_id")
    completed_at = _parse_time(attempt.get("ts"))
    if completed_at is None:
        raise ProjectionError("attempt timestamp is missing or invalid")
    created_at = _parse_time(event.get("start_time")) or completed_at
    if created_at > completed_at:
        raise ProjectionError("attempt timestamp precedes Framework event start")
    measurement = _mapping(attempt.get("measurement"))
    baseline = _number(measurement.get("before_tput"))
    if baseline is None or baseline <= 0:
        raise ProjectionError("attempt has no valid measured baseline")
    baseline_configuration = _baseline_configuration(attempt)
    reasoning, reasoning_origin = _reasoning(attempt, proposal)
    family, fingerprint, summary, content, resource_refs = _change(
        attempt,
        proposal,
        session_dir,
    )
    decision, outcome_value, error_class, failure_attribution = _decision(attempt)
    constraints = _constraints(attempt)
    facts = {
        "attempt_id": attempt_id,
        "decision": decision,
        "before_tput": baseline,
        "after_tput": outcome_value,
        "gain_pct": _number(measurement.get("gain_pct")),
        "reason": _text(attempt.get("reason")),
        "constraints": [{"name": name, "passed": passed, "value": value} for name, passed, value in constraints],
    }
    sequence_basis = f"{_event_id(event)}\0{attempt_id}".encode()
    seq = int.from_bytes(hashlib.sha256(sequence_basis).digest()[:6], "big")
    return ProjectedAttempt(
        attempt_id=attempt_id,
        run_id=run_id,
        seq=seq,
        created_at=created_at,
        completed_at=completed_at,
        identity=_identity(breakdown),
        baseline_identity=_baseline_identity(baseline_configuration),
        baseline_configuration=baseline_configuration,
        baseline_value=baseline,
        reasoning=reasoning,
        reasoning_origin=reasoning_origin,
        change_family=family,
        change_fingerprint=fingerprint,
        change_summary=summary,
        change_content=content,
        resource_refs=resource_refs,
        decision=decision,
        outcome_value=outcome_value,
        error_class=error_class,
        failure_attribution=failure_attribution,
        constraints=constraints,
        reflection="Recorded outcome: " + _canonical_json(facts),
        provenance_extra={
            "attempt_id": attempt_id,
            "framework_event_id": _event_id(event),
            "arm": _text(attempt.get("arm")),
            "proposal_ref": _text(attempt.get("proposal_ref")),
            "action_ref": _text(attempt.get("proposal_ref") or attempt.get("task_id")),
            "reasoning_origin": reasoning_origin,
            "round_id": _text(attempt.get("round_id")),
            "variant_name": _text(attempt.get("variant_name")),
            "validation_basis": _text(attempt.get("validation_basis")),
            "reflection_source": "deterministic_outcome_summary",
            "failure_attribution": failure_attribution,
            "failure_stage": _text(attempt.get("stage")),
        },
    )


def _project_all(
    session_dir: Path,
    breakdown: Mapping[str, Any],
) -> tuple[list[ProjectedAttempt], list[dict[str, Any]]]:
    projected: list[ProjectedAttempt] = []
    skipped: list[dict[str, Any]] = []
    for event in _framework_events(breakdown):
        ext = _mapping(event.get("ext"))
        proposals = {
            _text(item.get("proposal_id")): item
            for item in (ext.get("proposals") or [])
            if isinstance(item, Mapping) and _text(item.get("proposal_id"))
        }
        for item in ext.get("attempts") or []:
            if not isinstance(item, Mapping):
                continue
            attempt_id = _text(item.get("attempt_id")) or "unknown-attempt"
            proposal = proposals.get(_text(item.get("proposal_ref")), {})
            try:
                projected.append(
                    _project(
                        breakdown=breakdown,
                        event=event,
                        attempt=item,
                        proposal=proposal,
                        session_dir=session_dir,
                    )
                )
            except ProjectionError as exc:
                skipped.append({"attempt_id": attempt_id, "reason": str(exc)})
    return projected, skipped


def build_framework_experience_review(
    session_dir: Path | str,
    breakdown: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an offline fidelity review without importing or writing a KB."""

    root = Path(session_dir).resolve()
    blocked = _agentx_block_reason(breakdown)
    if blocked:
        attempts = [
            item
            for event in _framework_events(breakdown)
            for item in (_mapping(event.get("ext")).get("attempts") or [])
            if isinstance(item, Mapping)
        ]
        return {
            "schema_version": "hyperloom.framework-experience-review.v1",
            "blocked_reason": blocked,
            "framework_attempts": len(attempts),
            "ready": [],
            "skipped": [
                {
                    "attempt_id": _text(item.get("attempt_id")) or "unknown-attempt",
                    "reason": blocked,
                }
                for item in attempts
            ],
        }
    projected, skipped = _project_all(root, breakdown)
    return {
        "schema_version": "hyperloom.framework-experience-review.v1",
        "blocked_reason": "",
        "framework_attempts": len(projected) + len(skipped),
        "ready": [item.review_row() for item in projected],
        "skipped": skipped,
    }


def _publish_one(module: Any, kb: Any, projected: ProjectedAttempt) -> tuple[str, str]:
    session = kb.begin(
        run_id=projected.run_id,
        seq=projected.seq,
        identity=projected.identity,
        objective="e2e_throughput@v1",
        baseline_identity=projected.baseline_identity,
        baseline_value=projected.baseline_value,
        provenance=module.Provenance(
            producer="hyperloom-framework",
            producer_version="sbd-v6",
            snapshot_version="hyperloom.session_breakdown.v6",
            source_ref=f"session:{projected.run_id}:attempt:{projected.attempt_id}",
            extra=projected.provenance_extra,
        ),
        preconditions=(
            f"measured_baseline_tput={projected.baseline_value}",
            f"schema_identity={_canonical_json(projected.identity)}",
            (f"materialized_baseline_configuration={_canonical_json(projected.baseline_configuration)}"),
        ),
        created_at=projected.created_at,
    )
    if session.record is None:
        raise RuntimeError("Hyperloom-KB begin returned no Experience record")
    if session.record.status.value == "complete":
        return session.record.id, "unchanged"
    change = module.Change(
        identity={
            "change_family": projected.change_family,
            "change_fingerprint": projected.change_fingerprint,
        },
        summary=projected.change_summary,
        kind=projected.change_family,
        content=projected.change_content,
        resource_refs=projected.resource_refs,
    )
    if session.record.change is None:
        session.decide(
            reasoning=projected.reasoning,
            change=change,
            rendered_refs=(),
        )
    elif session.record.change != change:
        raise RuntimeError("existing in-progress Experience has a different change")
    constraints = tuple(module.ConstraintResult(name, passed, value) for name, passed, value in projected.constraints)
    session.complete(
        outcome=module.Outcome(
            decision=projected.decision,
            value=projected.outcome_value,
            constraints=constraints,
            error_class=projected.error_class,
        ),
        reflection=projected.reflection,
        completed_at=projected.completed_at,
    )
    session.publish()
    return session.record.id, "complete"


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_framework_experiences(
    session_dir: Path | str,
    breakdown: Mapping[str, Any],
) -> dict[str, Any]:
    """Best-effort publish of fidelity-qualified SBD V6 Framework attempts."""

    if not enabled():
        return {
            "schema_version": "hyperloom.experience-publish.v1",
            "enabled": False,
            "selected": 0,
            "published": 0,
            "errors": [],
        }
    root = Path(session_dir).resolve()
    review = build_framework_experience_review(root, breakdown)
    blocked = _text(review.get("blocked_reason"))
    if blocked:
        receipt = {
            "schema_version": "hyperloom.experience-publish.v1",
            "enabled": True,
            "blocked_reason": blocked,
            "selected": 0,
            "published": 0,
            "errors": [],
            "experiences": [],
            "skipped": review["skipped"],
        }
        _write_receipt(root / _RECEIPT_REL, receipt)
        return receipt

    module = import_module("hyperloom_kb")
    kb = module.experience_kb_from_env()
    projected, skipped = _project_all(root, breakdown)
    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for item in projected:
        try:
            experience_id, status = _publish_one(module, kb, item)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            log.warning(
                "Hyperloom-KB Framework attempt publish failed: attempt=%s error=%s",
                item.attempt_id,
                exc,
            )
            errors.append(
                {
                    "attempt_id": item.attempt_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "attempt_id": item.attempt_id,
                "experience_id": experience_id,
                "status": status,
            }
        )
    receipt = {
        "schema_version": "hyperloom.experience-publish.v1",
        "enabled": True,
        "blocked_reason": "",
        "source": "session_breakdown.timeline[type=framework_agent].ext.attempts",
        "selected": len(projected),
        "published": len(rows),
        "errors": errors,
        "experiences": rows,
        "skipped": skipped,
    }
    try:
        _write_receipt(root / _RECEIPT_REL, receipt)
    except OSError as exc:
        log.warning("Hyperloom-KB publish receipt write failed: %s", exc)
    return receipt


__all__ = [
    "ProjectionError",
    "build_framework_experience_review",
    "enabled",
    "publish_framework_experiences",
    "validate_experience_config",
]
