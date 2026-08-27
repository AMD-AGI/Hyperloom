# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reuse of standalone FlyDSL recipes, filed under a producer-owned identity.

A rewrite is only reusable when the contract it was written against still
holds, so a candidate is admitted on exact hashes of the source, the driver and
the builder symbol rather than on its score. The score decides ranking and the
champion pointer, nothing else: a correct port that loses to the source
baseline is still what saves the next run from repeating PORT.

Candidates that fail the gate are not discarded either. Their code goes back to
the author as reference material, which is why the reader fetches content for
rejected candidates too.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.config import Config
from kernelforge.knowledge.experience_reader import sanitize_read_error
from kernelforge.knowledge.experience_store import knowledge_config_from_runtime
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.rewrite_by_flydsl import driver_contract
from kernelforge.rewrite_by_flydsl.identity import (
    resolve_identity,
    session_id as candidate_session_id,
)
from kernelforge.rewrite_by_flydsl.port_loop import check_flydsl_port
from kernelforge.rewrite_by_flydsl.record_store import (
    RewriteRecordStore,
    create_rewrite_record_store,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

_SCHEMA_VERSION = 1
_REWRITE_KIND = "standalone_flydsl"
_KERNEL_ARTIFACT = "kernel.py"
_REFERENCE_CONTENT_CAP = 12_000


@dataclass
class RewriteKbReadResult:
    applied: bool = False
    read_reason: str = ""
    read_error: str = ""
    solution_slug: str = ""
    best_ms: float | None = None
    snr_db: float | None = None
    attempts: list[dict] = field(default_factory=list)
    reference_context: str = ""

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "read_reason": self.read_reason,
            "read_error": self.read_error,
            "solution_slug": self.solution_slug,
            "best_ms": self.best_ms,
            "snr_db": self.snr_db,
            "attempts": list(self.attempts),
            "has_reference_context": bool(self.reference_context),
        }


@dataclass(frozen=True)
class _ReadPlan:
    """What the reader resolved before it started trying candidates.

    The store is carried alongside the candidates because a candidate's code
    is an artifact fetched on demand, not a field of the document that ranked
    it.
    """

    store: RewriteRecordStore | None
    candidates: list[dict[str, Any]]
    read_reason: str
    read_error: str


def _sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _source_text(spec: RewriteSpec) -> str:
    try:
        return Path(spec.source_kernel).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""


def _secrets(config: Config) -> tuple[str, ...]:
    knowledge = knowledge_config_from_runtime(config)
    return tuple(value for value in (knowledge.kb_store_token,) if value)


def _read_top_candidates(
    spec: RewriteSpec,
    config: Config,
    *,
    framework: str,
    top_k: int,
) -> _ReadPlan:
    store = create_rewrite_record_store(config)
    if store is None:
        return _ReadPlan(None, [], "not_configured", "")
    gpu_type = str(config.gpu_type or "").strip()
    if not gpu_type:
        return _ReadPlan(None, [], "missing_gpu_type", "")
    try:
        _, canonical_id, signature, implementation = resolve_identity(
            spec,
            framework=framework,
            gpu=gpu_type,
            source_text=_source_text(spec),
        )
        candidates: list[dict[str, Any]] = []
        for candidate in store.candidates(canonical_id, limit=top_k):
            value = candidate.knowledge.get("value")
            if not isinstance(value, dict):
                continue
            recorded = str(value.get("implementation_signature") or "")
            candidates.append(
                {
                    "canonical_id": canonical_id,
                    "session_id": candidate.session_id,
                    "solution_slug": f"{canonical_id}/{candidate.session_id}",
                    "speedup": candidate.speedup,
                    "implementation_match": bool(recorded and recorded == signature),
                    "consumer_implementation_identity": implementation,
                    "attrs": value,
                }
            )
        return _ReadPlan(
            store,
            candidates,
            "hit" if candidates else "no_candidates",
            "",
        )
    except Exception as error:  # noqa: BLE001 - KB read must cold-start
        return _ReadPlan(
            None,
            [],
            "read_error",
            sanitize_read_error(error, secrets=_secrets(config)),
        )


def _candidate_content(plan: _ReadPlan, candidate: dict[str, Any]) -> bytes:
    """Fetch the referenced port artifact bytes, or ``b""`` when absent."""
    if plan.store is None:
        return b""
    rel_path = str(candidate["attrs"].get("flydsl_kernel") or "")
    if not rel_path:
        return b""
    try:
        return plan.store.read_bytes(
            candidate["canonical_id"],
            candidate["session_id"],
            rel_path,
        )
    except Exception:  # noqa: BLE001 - an unreadable artifact is just a miss
        return b""


def _reference_context(references: list[dict]) -> str:
    if not references:
        return ""
    sections = [
        "## Historical FlyDSL rewrite references",
        "",
        (
            "These top-ranked KB candidates were not accepted by the current "
            + "validation gate. Use them only as reference material; do not assume "
            + "their code is correct for the current task."
        ),
    ]
    for index, reference in enumerate(references, 1):
        content = (reference.get("content") or b"").decode(
            "utf-8",
            errors="replace",
        )
        if len(content) > _REFERENCE_CONTENT_CAP:
            content = content[:_REFERENCE_CONTENT_CAP] + "\n# ... truncated ...\n"
        sections.extend(
            [
                "",
                f"### Reference {index}: {reference.get('solution_slug', '')}",
                f"- Prior speedup: {reference.get('speedup')}",
                f"- Rejection reason: {reference.get('reason')}",
                "",
                "```python",
                content,
                "```",
            ]
        )
    return "\n".join(sections)


async def try_flydsl_kb_warmstart(
    spec: RewriteSpec,
    driver_path: str,
    config: Config,
    *,
    source_ms: float | None,
    framework: str = "",
    top_k: int = 3,
    validation_timeout_sec: int = 1800,
    stop_at_unix: float | None = None,
) -> RewriteKbReadResult:
    """Try top-3 candidates; correctness alone permits skipping PORT."""
    del source_ms  # Performance is measured for context, not used as the PORT gate.
    plan = _read_top_candidates(
        spec,
        config,
        framework=framework,
        top_k=top_k,
    )
    result = RewriteKbReadResult(
        read_reason=plan.read_reason,
        read_error=plan.read_error,
    )
    original = Path(spec.flydsl_kernel).read_bytes() if Path(spec.flydsl_kernel).is_file() else None
    source_hash = _sha256(spec.source_kernel)
    driver_hash = _sha256(driver_path)
    references: list[dict] = []

    for candidate in plan.candidates:
        remaining = stop_at_unix - time.time() if stop_at_unix and stop_at_unix > 0 else None
        if remaining is not None and remaining <= 0:
            result.read_reason = "deadline"
            break
        attrs = candidate["attrs"]
        attempt = {
            "solution_slug": candidate["solution_slug"],
            "speedup": candidate["speedup"],
        }
        reason = ""
        if candidate.get("implementation_match") is not True:
            reason = "implementation_mismatch"
        elif attrs.get("schema_version") != _SCHEMA_VERSION or attrs.get("rewrite_kind") != _REWRITE_KIND:
            reason = "wrong_solution_kind"
        elif attrs.get("source_sha256") != source_hash:
            reason = "source_changed"
        elif attrs.get("driver_sha256") != driver_hash:
            reason = "driver_contract_changed"
        elif attrs.get("builder_symbol") != spec.builder_symbol:
            reason = "builder_contract_changed"
        content = _candidate_content(plan, candidate)
        if not reason and not content.strip():
            reason = "missing_kernel_content"

        if not reason:
            try:
                Path(spec.flydsl_kernel).write_bytes(content)
                violation = check_flydsl_port(spec)
                if violation:
                    reason = f"flydsl_gate:{violation}"
                else:
                    validation = await run_validation_pipeline(
                        driver_script=driver_path,
                        snr_threshold=spec.snr_threshold,
                        timeout_per_stage=(
                            validation_timeout_sec
                            if remaining is None
                            else max(
                                1,
                                min(validation_timeout_sec, int(remaining)),
                            )
                        ),
                    )
                    if not validation.all_passed:
                        reason = "correctness_failed"
                    else:
                        remaining = stop_at_unix - time.time() if stop_at_unix and stop_at_unix > 0 else None
                        candidate_ms = None
                        if remaining is None or remaining > 0:
                            benched = driver_contract.preflight_candidate(
                                spec,
                                driver_path,
                                timeout_sec=(600 if remaining is None else max(1, min(600, int(remaining)))),
                            )
                            candidate_ms = benched.timing_ms if benched.ok else None
                        snr = validation.results[-1].snr_db if validation.results else None
                        attempt.update(
                            reason="applied",
                            best_ms=candidate_ms,
                        )
                        result.attempts.append(attempt)
                        result.applied = True
                        result.read_reason = "applied"
                        result.solution_slug = candidate["solution_slug"]
                        result.best_ms = candidate_ms
                        result.snr_db = snr
                        result.reference_context = _reference_context(references)
                        return result
            except Exception as error:  # noqa: BLE001 - candidate becomes reference
                reason = f"validation_error:{type(error).__name__}"

        attempt["reason"] = reason
        result.attempts.append(attempt)
        references.append(
            {
                "solution_slug": candidate["solution_slug"],
                "speedup": candidate["speedup"],
                "reason": reason,
                "content": content,
            }
        )

    if original is None:
        Path(spec.flydsl_kernel).unlink(missing_ok=True)
    else:
        Path(spec.flydsl_kernel).write_bytes(original)
    result.reference_context = _reference_context(references)
    if plan.candidates and result.read_reason == "hit":
        result.read_reason = "candidates_rejected"
    return result


def write_flydsl_kb_solution(
    spec: RewriteSpec,
    driver_path: str,
    config: Config,
    *,
    source_ms: float | None,
    flydsl_best_ms: float | None,
    best_commit: str = "",
    framework: str = "",
    snr_db: float | None = None,
    allow_non_improving: bool = False,
) -> dict:
    """Record a validated FlyDSL port as a candidate under its identity.

    ``allow_non_improving`` is used after a real PORT session: correctness makes
    that artifact reusable even when it does not beat the source baseline. Such
    a candidate is recorded but never promoted, so it can be replayed without
    ever being mistaken for the identity's best result.

    Never raises, and the returned reason is persisted by the rewrite runner, so
    a store exception is redacted and bounded the way the read side above does
    it. The exception type leads the message, so the cap can only cut the tail of
    a long error body.
    """
    store = create_rewrite_record_store(config)
    if store is None:
        return {"written": False, "reason": "not_configured"}
    gpu_type = str(config.gpu_type or "").strip()
    if not gpu_type:
        return {"written": False, "reason": "missing_gpu_type"}
    speedup = source_ms / flydsl_best_ms if source_ms and flydsl_best_ms else None
    if not allow_non_improving and (speedup is None or speedup <= 1.0):
        return {"written": False, "reason": "no_improvement"}
    try:
        content = Path(spec.flydsl_kernel).read_bytes()
        identity, canonical_id, signature, implementation = resolve_identity(
            spec,
            framework=framework,
            gpu=gpu_type,
            source_text=_source_text(spec),
        )
        content_hash = hashlib.sha256(content).hexdigest()
        session_id = candidate_session_id(
            canonical_id,
            identity.kernel_name,
            best_commit or content_hash,
        )
        knowledge = {
            "producer": identity.producer,
            "speedup": round(speedup, 4) if speedup is not None else None,
            "identity": asdict(identity),
            "value": {
                "id": session_id,
                "schema_version": _SCHEMA_VERSION,
                "rewrite_kind": _REWRITE_KIND,
                "flydsl_kernel": _KERNEL_ARTIFACT,
                "metric": {
                    "wall_ms": flydsl_best_ms,
                    "baseline_wall_ms": source_ms,
                    "speedup": round(speedup, 4) if speedup is not None else None,
                    "snr_db": snr_db,
                    "gpu_arch": config.gpu_target,
                    "correct": True,
                },
                "implementation_signature": signature,
                "implementation_identity": implementation,
                "source_sha256": _sha256(spec.source_kernel),
                "driver_sha256": _sha256(driver_path),
                "builder_symbol": spec.builder_symbol,
            },
        }
        with tempfile.TemporaryDirectory(prefix="flydsl-rewrite-write-") as temporary:
            staged = Path(temporary) / _KERNEL_ARTIFACT
            staged.write_bytes(content)
            store.write(canonical_id, session_id, knowledge, {_KERNEL_ARTIFACT: staged})
        # The pointer says "the best result for this identity", so a port that
        # loses to the source baseline never takes it, even when it is the only
        # one recorded. The reader enumerates candidates rather than following
        # the pointer, so staying unpromoted costs such a port nothing here.
        promoted = False
        if speedup is not None and speedup > 1.0:
            champion = store.champion_speedup(canonical_id)
            if champion is None or speedup > champion:
                store.promote(canonical_id, session_id, speedup)
                promoted = True
        return {
            "written": True,
            "kernel": canonical_id,
            "solution": f"{canonical_id}/{session_id}",
            "canonical_id": canonical_id,
            "session_id": session_id,
            "speedup": speedup,
            "champion": promoted,
        }
    except Exception as error:  # noqa: BLE001 - KB write never breaks rewrite
        return {
            "written": False,
            "reason": sanitize_read_error(error, secrets=_secrets(config)),
        }
