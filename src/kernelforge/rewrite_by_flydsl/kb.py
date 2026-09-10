# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reuse of standalone FlyDSL recipes, filed under a producer-owned identity."""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.config import Config
from kernelforge.knowledge import warmstart_policy
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

# Ceiling on the benchmark that times one candidate, when nothing tighter bounds it.
_BENCH_TIMEOUT_SEC = 600


def _stage_timeout(cap_sec: int, search_deadline: float, run_deadline_sec: float | None) -> int:
    """Seconds one measurement stage may take without outliving the search or the run.

    The search budget bounds the field, not each trial, so a stage allowed to run to its own ceiling could spend the
    whole budget on the first candidate and leave the rest of the field unmeasured -- which is the opposite of what
    reading a wide field is for.
    """
    allowance = search_deadline - time.monotonic()
    if run_deadline_sec is not None:
        allowance = min(allowance, run_deadline_sec)
    return max(1, min(cap_sec, int(allowance)))


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
    """What the reader resolved before it started trying candidates."""

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
    top_k: int | None = None,
    validation_timeout_sec: int = 1800,
    stop_at_unix: float | None = None,
) -> RewriteKbReadResult:
    """Measure the admissible candidates and skip PORT with the fastest.

    Correctness admits a candidate; a measurement on this task's own driver chooses between the admitted ones, since a
    claim was computed over whatever cases its producing task scored and so cannot order candidates for *this* task.
    ``warmstart_policy`` bounds the search on both the claim floor and wall time.
    """
    del source_ms  # The claim is not the gate; this task's own timing is.
    plan = _read_top_candidates(
        spec,
        config,
        framework=framework,
        top_k=warmstart_policy.top_k() if top_k is None else top_k,
    )
    result = RewriteKbReadResult(
        read_reason=plan.read_reason,
        read_error=plan.read_error,
    )
    original = Path(spec.flydsl_kernel).read_bytes() if Path(spec.flydsl_kernel).is_file() else None
    source_hash = _sha256(spec.source_kernel)
    driver_hash = _sha256(driver_path)
    references: list[dict] = []

    # Survivors of the whole gauntlet, with what this task's driver timed them
    # at. The winner is chosen after the field closes, not on the way through.
    measured: list[dict] = []
    search_deadline = time.monotonic() + warmstart_policy.budget_sec()

    for index, candidate in enumerate(plan.candidates):
        remaining = stop_at_unix - time.time() if stop_at_unix and stop_at_unix > 0 else None
        if remaining is not None and remaining <= 0:
            result.read_reason = "deadline"
            break
        # A trial's cost scales with how slow the candidate is -- the correctness suite and the benchmark both run the
        # kernel -- so a port claiming to be orders of magnitude off the pace can spend the whole search budget on
        # itself. The claim only has to be right about the magnitude for that to be the wrong trade.
        if warmstart_policy.below_floor(candidate["speedup"]):
            result.attempts.append(
                {
                    "solution_slug": candidate["solution_slug"],
                    "speedup": candidate["speedup"],
                    "reason": "below_claim_floor",
                }
            )
            continue
        # The candidate count does not bound wall time: one trial is minutes on the heaviest kernels. Whatever has been
        # measured already still wins below.
        if time.monotonic() >= search_deadline:
            result.attempts.append(
                {
                    "solution_slug": candidate["solution_slug"],
                    "speedup": candidate["speedup"],
                    "reason": "search_budget_spent",
                }
            )
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
                        timeout_per_stage=_stage_timeout(
                            validation_timeout_sec,
                            search_deadline,
                            remaining,
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
                                timeout_sec=_stage_timeout(_BENCH_TIMEOUT_SEC, search_deadline, remaining),
                            )
                            candidate_ms = benched.timing_ms if benched.ok else None
                        snr = validation.results[-1].snr_db if validation.results else None
                        attempt.update(reason="measured", best_ms=candidate_ms)
                        result.attempts.append(attempt)
                        measured.append(
                            {
                                "index": index,
                                "candidate": candidate,
                                "content": content,
                                "ms": candidate_ms,
                                "snr_db": snr,
                                "attempt": attempt,
                            }
                        )
                        continue
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

    if measured:
        # Fastest on this task's own driver. A survivor whose benchmark failed is still adoptable -- it passed
        # correctness -- but it ranks behind every timed one, because nothing is known about its speed.
        winner = min(
            measured,
            key=lambda item: (
                item["ms"] is None,
                item["ms"] if item["ms"] is not None else 0.0,
                item["index"],
            ),
        )
        Path(spec.flydsl_kernel).write_bytes(winner["content"])
        for item in measured:
            item["attempt"]["reason"] = "applied" if item is winner else f"outperformed_by_rank_{winner['index'] + 1}"
        result.applied = True
        result.read_reason = "applied"
        result.solution_slug = winner["candidate"]["solution_slug"]
        result.best_ms = winner["ms"]
        result.snr_db = winner["snr_db"]
        result.reference_context = _reference_context(references)
        return result

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
    session_key: str = "",
    content_override: bytes | None = None,
) -> dict:
    """Record a validated FlyDSL port as a candidate under its identity.

    Correctness alone qualifies a port; only the champion pointer is gated on speedup. ``session_key`` names the
    session this record belongs to, so a caller publishing repeatedly through one session replaces its own record
    rather than burying the identity's history under a sibling per publication. ``content_override`` supplies the
    kernel bytes for a caller publishing while an agent is still editing the workspace. ``snr_db`` is the accuracy
    measured for *this* artifact; a caller that did not measure it passes ``None``, because a reading taken from
    another kernel is not a substitute.
    """
    store = create_rewrite_record_store(config)
    if store is None:
        return {"written": False, "reason": "not_configured"}
    gpu_type = str(config.gpu_type or "").strip()
    if not gpu_type:
        return {"written": False, "reason": "missing_gpu_type"}
    speedup = source_ms / flydsl_best_ms if source_ms and flydsl_best_ms else None
    try:
        content = content_override if content_override is not None else Path(spec.flydsl_kernel).read_bytes()
        identity, canonical_id, signature, implementation = resolve_identity(
            spec,
            framework=framework,
            gpu=gpu_type,
            source_text=_source_text(spec),
        )
        content_hash = hashlib.sha256(content).hexdigest()
        # Only when the caller has no session identity at all does the artifact name the record, and then every
        # publication is a sibling. ``best_commit`` is metadata here, never a name: a caller that wants its commit to
        # name the record says so through ``session_key``.
        session_id = candidate_session_id(canonical_id, identity.kernel_name, session_key or content_hash)
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
        # The pointer says "the best result for this identity", so a port that loses to the source baseline never
        # takes it, even when it is the only one recorded.
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
