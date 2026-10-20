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
from kernelforge.knowledge.kb_store import warmstart_policy
from kernelforge.knowledge.kb_store.config import knowledge_config_from_runtime
from kernelforge.knowledge.kb_store.errors import sanitize_read_error
from kernelforge.knowledge.kb_store.identity.warmstart_rank import rank_fallback_identities
from kernelforge.knowledge.kb_store.recipe import candidate_session_id
from kernelforge.evaluation.benchmark.runner import CaseCoverageError, calculate_mean_case_speedup
from kernelforge.evaluation.correctness.pipeline import run_validation_pipeline
from kernelforge.rewrite_by_flydsl import driver_contract
from kernelforge.rewrite_by_flydsl.identity import resolve_identity
from kernelforge.rewrite_by_flydsl.measurement import FlydslMeasurement
from kernelforge.rewrite_by_flydsl.port_loop import check_flydsl_port
from kernelforge.knowledge.kb_store.record_store import (
    RewriteRecordStore,
    create_rewrite_record_store,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

_SCHEMA_VERSION = 1
_REWRITE_KIND = "standalone_flydsl"
_KERNEL_ARTIFACT = "kernel.py"
_REFERENCE_CONTENT_CAP = 12_000

# Ceilings on the correctness suite and the benchmark that try one candidate, when nothing tighter bounds them.
_VALIDATION_TIMEOUT_SEC = 1800
_BENCH_TIMEOUT_SEC = 600


def _stage_timeout(cap_sec: int, search_deadline: float, run_deadline_sec: float) -> int:
    """Seconds one measurement stage may take without outliving the search or the run.

    The search budget bounds the field, not each trial, so a stage allowed to run to its own ceiling could spend the
    whole budget on the first candidate and leave the rest of the field unmeasured -- which is the opposite of what
    reading a wide field is for.
    """
    return max(1, min(cap_sec, int(min(search_deadline - time.monotonic(), run_deadline_sec))))


@dataclass(frozen=True)
class RewriteKbTarget:
    """The rewrite a KB read or write is about: its spec, the driver measuring it, the runtime naming store and GPU."""

    spec: RewriteSpec
    driver_path: str
    config: Config


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
    top_k: int,
) -> _ReadPlan:
    store = create_rewrite_record_store(config)
    if store is None:
        return _ReadPlan(None, [], "not_configured", "")
    gpu_type = str(config.gpu_type or "").strip()
    if not gpu_type:
        return _ReadPlan(None, [], "missing_gpu_type", "")
    try:
        identity, canonical_id, _signature, _implementation = resolve_identity(
            spec,
            gpu=gpu_type,
            source_text=_source_text(spec),
        )
        resolved = [(canonical_id, candidate) for candidate in store.candidates(canonical_id, limit=top_k)]
        if not resolved:
            searched = store.search_identities(identity, limit=300)
            for fallback_id in rank_fallback_identities(identity, searched):
                fallback = store.candidates(fallback_id, limit=top_k)
                if fallback:
                    resolved = [(fallback_id, candidate) for candidate in fallback]
                    break
        candidates: list[dict[str, Any]] = []
        for candidate_canonical_id, candidate in resolved:
            value = candidate.knowledge.get("value")
            if not isinstance(value, dict):
                continue
            candidates.append(
                {
                    "canonical_id": candidate_canonical_id,
                    "session_id": candidate.session_id,
                    "solution_slug": f"{candidate_canonical_id}/{candidate.session_id}",
                    "speedup": candidate.speedup,
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


def scored_speedup(
    case_ms: dict[str, float] | None,
    baseline_case_ms: dict[str, float] | None,
) -> float | None:
    """Speedup on the arena's metric, or ``None`` when the cases cannot be paired.

    The equal-weight mean over per-case ratios is what grades the run, so it is also what orders a warm-start field
    and what a published record claims. A ratio of the two aggregate times answers a different question and disagrees
    by up to 2x on a shape sweep spanning two orders of magnitude.
    """
    try:
        return calculate_mean_case_speedup(case_ms, baseline_case_ms)
    except CaseCoverageError:
        return None


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


@dataclass
class _Trials:
    """The candidates one warm start tried on this task's driver.

    Survivors of the whole gauntlet are kept with what the driver timed them at, so the winner is chosen after the
    field closes, not on the way through. Rejects are kept as reference material for the port agent.
    """

    target: RewriteKbTarget
    source_case_ms: dict[str, float]
    stop_at_unix: float
    search_deadline: float
    measured: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)

    def reject(self, attempt: dict, content: bytes, reason: str) -> None:
        attempt["reason"] = reason
        self.references.append({**attempt, "content": content})

    async def measure(self, index: int, attempt: dict, content: bytes, remaining: float) -> None:
        """Install an admitted candidate; it survives when it passes the port gate and the suite and is scored."""
        spec = self.target.spec
        try:
            Path(spec.flydsl_kernel).write_bytes(content)
            violation = check_flydsl_port(spec)
            if violation:
                self.reject(attempt, content, f"flydsl_gate:{violation}")
                return
            validation = await run_validation_pipeline(
                driver_script=self.target.driver_path,
                snr_threshold=spec.snr_threshold,
                timeout_per_stage=_stage_timeout(_VALIDATION_TIMEOUT_SEC, self.search_deadline, remaining),
            )
            if not validation.all_passed:
                self.reject(attempt, content, "correctness_failed")
                return
            remaining = self.stop_at_unix - time.time()
            speedup = None
            if remaining > 0:
                benched = driver_contract.preflight_candidate(
                    spec,
                    self.target.driver_path,
                    timeout_sec=_stage_timeout(_BENCH_TIMEOUT_SEC, self.search_deadline, remaining),
                )
                if benched.ok:
                    speedup = scored_speedup(benched.case_ms, self.source_case_ms)
            if speedup is None:
                self.reject(attempt, content, "unscorable_measurement")
                return
            snr = validation.results[-1].snr_db if validation.results else None
            attempt.update(reason="measured", best_ms=benched.timing_ms, speedup=speedup)
            self.measured.append({"index": index, "content": content, "snr_db": snr, "attempt": attempt})
        except Exception as error:  # noqa: BLE001 - candidate becomes reference
            self.reject(attempt, content, f"validation_error:{type(error).__name__}")

    def settle(self, result: RewriteKbReadResult, original: bytes | None) -> None:
        """Adopt the best-scoring survivor, or put the original kernel back when none survived."""
        kernel = Path(self.target.spec.flydsl_kernel)
        # Best scoring on this task's own driver. A survivor whose benchmark failed never reaches this list: passing
        # correctness says the kernel computes the right answer, not that the timing behind a claim can be reproduced.
        if self.measured:
            winner = max(self.measured, key=lambda item: item["attempt"]["speedup"])
            kernel.write_bytes(winner["content"])
            outperformed = f"outperformed_by_rank_{winner['index'] + 1}"
            for item in self.measured:
                item["attempt"]["reason"] = "applied" if item is winner else outperformed
            result.applied = True
            result.read_reason = "applied"
            result.solution_slug = winner["attempt"]["solution_slug"]
            result.best_ms = winner["attempt"]["best_ms"]
            result.snr_db = winner["snr_db"]
        elif original is None:
            kernel.unlink(missing_ok=True)
        else:
            kernel.write_bytes(original)
        result.reference_context = _reference_context(self.references)
        if result.read_reason == "hit":
            result.read_reason = "candidates_rejected"


async def try_flydsl_kb_warmstart(
    target: RewriteKbTarget,
    *,
    source_case_ms: dict[str, float],
    stop_at_unix: float = float("inf"),
) -> RewriteKbReadResult:
    """Measure the admissible candidates and skip PORT with the best scoring one.

    Correctness admits a candidate; a measurement on this task's own driver chooses between the admitted ones, since a
    claim was computed over whatever cases its producing task scored and so cannot order candidates for *this* task.
    The measurement is scored the way the arena scores, so the candidate adopted here is the one the run is graded on.
    ``warmstart_policy`` bounds the search on both the claim floor and wall time.

    Admission is one question a measurement cannot answer for itself: does the artifact expose the builder symbol
    this task's driver imports. Everything else is settled by running it. The current driver validates the candidate
    against the current source and then times it, so a port that survives is reusable whatever revision, source
    layout or implementation signature produced it -- and one that does not is rejected on evidence rather than on
    a guess about its provenance. This matches the loop-layer warm start, which has always measured a candidate
    whose implementation signature disagreed instead of refusing it.
    """
    spec = target.spec
    plan = _read_top_candidates(
        spec,
        target.config,
        top_k=warmstart_policy.top_k(),
    )
    result = RewriteKbReadResult(
        read_reason=plan.read_reason,
        read_error=plan.read_error,
    )
    original = Path(spec.flydsl_kernel).read_bytes() if Path(spec.flydsl_kernel).is_file() else None
    trials = _Trials(target, source_case_ms, stop_at_unix, time.monotonic() + warmstart_policy.budget_sec())

    for index, candidate in enumerate(plan.candidates):
        remaining = stop_at_unix - time.time()
        if remaining <= 0:
            result.read_reason = "deadline"
            break
        attempt = {"solution_slug": candidate["solution_slug"], "speedup": candidate["speedup"]}
        result.attempts.append(attempt)
        # A trial's cost scales with how slow the candidate is -- the correctness suite and the benchmark both run the
        # kernel -- so a port claiming to be orders of magnitude off the pace can spend the whole search budget on
        # itself. The claim only has to be right about the magnitude for that to be the wrong trade.
        if warmstart_policy.below_floor(candidate["speedup"]):
            attempt["reason"] = "below_claim_floor"
            continue
        # The candidate count does not bound wall time: one trial is minutes on the heaviest kernels. Whatever has been
        # measured already still wins.
        if time.monotonic() >= trials.search_deadline:
            attempt["reason"] = "search_budget_spent"
            break
        builder_changed = candidate["attrs"].get("builder_symbol") != spec.builder_symbol
        content = _candidate_content(plan, candidate)
        if builder_changed:
            trials.reject(attempt, content, "builder_contract_changed")
        elif not content.strip():
            trials.reject(attempt, content, "missing_kernel_content")
        else:
            await trials.measure(index, attempt, content, remaining)
    trials.settle(result, original)
    return result


def write_flydsl_kb_solution(
    target: RewriteKbTarget,
    measurement: FlydslMeasurement,
    *,
    session_key: str = "",
    content_override: bytes | None = None,
) -> dict:
    """Record a validated FlyDSL port as a candidate under its identity.

    Correctness alone qualifies a port; only the champion pointer is gated on speedup, so a consumer ranking records
    is ranking on the number they will be graded on. ``session_key`` names the session this record belongs to, so a
    caller publishing repeatedly through one session replaces its own record rather than burying the identity's
    history under a sibling per publication. ``content_override`` supplies the kernel bytes for a caller publishing
    while an agent is still editing the workspace.
    """
    spec = target.spec
    config = target.config
    speedup = measurement.speedup
    store = create_rewrite_record_store(config)
    if store is None:
        return {"written": False, "reason": "not_configured"}
    gpu_type = str(config.gpu_type or "").strip()
    if not gpu_type:
        return {"written": False, "reason": "missing_gpu_type"}
    try:
        content = content_override if content_override is not None else Path(spec.flydsl_kernel).read_bytes()
        identity, canonical_id, signature, implementation = resolve_identity(
            spec,
            gpu=gpu_type,
            source_text=_source_text(spec),
        )
        content_hash = hashlib.sha256(content).hexdigest()
        # Only when the caller has no session identity at all does the artifact name the record, and then every
        # publication is a sibling. A caller that wants its commit to name the record says so through ``session_key``.
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
                    "wall_ms": measurement.flydsl_ms,
                    "baseline_wall_ms": measurement.source_ms,
                    "speedup": round(speedup, 4) if speedup is not None else None,
                    "snr_db": measurement.snr_db,
                    "gpu_arch": config.gpu_target,
                    "correct": True,
                },
                "implementation_signature": signature,
                "implementation_identity": implementation,
                "source_sha256": _sha256(spec.source_kernel),
                "driver_sha256": _sha256(target.driver_path),
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
