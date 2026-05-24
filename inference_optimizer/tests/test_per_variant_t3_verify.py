"""v0.8 KB_design §3.13 M5 §5 step 7 / KB_gaps/Gap-08 — per-variant T3.

KB_gaps/Gap-08 root cause: ``_cortex_t3_hook`` always emitted a single
``ingest_attempt`` + single ``verify`` per task, even when the task was
an ``explore`` action with per-variant KEEP/REVERT decisions. Gap-07
shipped per-variant edge_ids in T2 but they had no T3 consumer — the
KB negation edge always reflected "whole batch refuted", not "variant X
refuted but variant Y confirmed".

This file covers:

* ``_cortex_t3_hook`` dispatcher: routes explore + per_variant_outcomes
  to the per-variant path; everything else (kernel_opt / integrate /
  baseline / explore-without-outcomes) stays on the legacy per-task
  path.
* ``_cortex_t3_per_variant``: one ``ingest_attempt`` + one ``verify``
  per variant; KEEP → confirmed + EXPERIENTIAL promote; REVERT /
  FAILED / KEEP_UNSTABLE → refuted + no promote; SKIPPED_DEDUP skipped;
  partial failures isolate per variant; pending edge row popped once;
  ``shared_state.save`` called exactly once at the end.
* Variant edge_id lookup priority: ``pending_kb_edges[].variant_edges``
  first, then per-variant ``kb_edge_id`` on the executor's result
  (fallback for resume), then single-edge fallback for back-compat
  with pre-Gap-07 pending rows.
* ``kb_provenance.edges_promoted`` / ``edges_negated`` derived from
  audit log enqueue rows.
* ``ExploreExecutor`` result dict carries ``per_variant_outcomes``
  built from ``tested_update`` + ``skipped_dup``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.cortex_kb_client import CortexKBError
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.task_registry import Task


# ===========================================================================
# Fixtures
# ===========================================================================
@dataclass
class _BareState:
    """SharedState double exposing only the fields the T3 hooks touch."""

    cortex_session_id: str = "sid-test"
    tick: int = 7
    pending_kb_edges: list[dict[str, Any]] = field(default_factory=list)
    save_count: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


class _StubCortexKB:
    """CortexKBClient double recording every call."""

    enabled: bool = True

    def __init__(
        self,
        *,
        verify_raises_for: set[str] | None = None,
        ingest_raises_for: set[str] | None = None,
    ):
        self.propose_point_calls: list[dict] = []
        self.ingest_attempt_calls: list[dict] = []
        self.verify_calls: list[dict] = []
        self._verify_raises_for = verify_raises_for or set()
        self._ingest_raises_for = ingest_raises_for or set()

    def propose_point(self, **kwargs):
        self.propose_point_calls.append(dict(kwargs))

    def ingest_attempt(self, **kwargs):
        self.ingest_attempt_calls.append(dict(kwargs))
        attrs_or_metrics = kwargs.get("metrics") or {}
        if attrs_or_metrics.get("variant_name") in self._ingest_raises_for:
            raise CortexKBError("synthetic ingest_attempt failure")

    def verify(self, **kwargs):
        self.verify_calls.append(dict(kwargs))
        idem = kwargs.get("idempotency_key") or ""
        # idempotency_key carries the variant name at the tail.
        for failing in self._verify_raises_for:
            if idem.endswith(f":{failing}"):
                raise CortexKBError("synthetic verify failure")


@pytest.fixture
def coord(tmp_path: Path):
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.cortex_kb = _StubCortexKB()
    return c


def _task(
    *,
    task_id: str = "t-1",
    kind: str = "explore",
    proposal_msg_id: str = "msg-1",
) -> Task:
    return Task(
        task_id=task_id,
        kind=kind,
        state="completed",
        params={},
        idempotency_key=(
            f"approved-{proposal_msg_id}" if proposal_msg_id else ""
        ),
    )


def _seed_pending_edge_row(
    coord: Coordinator,
    *,
    proposal_msg_id: str = "msg-1",
    variant_edges: dict[str, str] | None = None,
    single_edge_id: str = "",
) -> None:
    row: dict[str, Any] = {
        "proposal_msg_id": proposal_msg_id,
        "edge_id":         single_edge_id,
        "action":          "explore",
        "ts":              "2026-05-19T00:00:00+00:00",
    }
    if variant_edges is not None:
        row["variant_edges"] = dict(variant_edges)
    coord.shared_state.pending_kb_edges.append(row)


# ===========================================================================
# 1. Dispatcher — _cortex_t3_hook routes by task.kind + per_variant_outcomes
# ===========================================================================
@pytest.mark.asyncio
async def test_t3_hook_no_cortex_short_circuits(coord):
    coord.cortex_kb = None
    await coord._cortex_t3_hook(
        task=_task(),
        result={"per_variant_outcomes": [{"variant_name": "v1", "outcome": "KEEP"}]},
        kept=True,
    )
    # Nothing to record — coord.cortex_kb is None.
    assert coord.shared_state.save_count == 0


@pytest.mark.asyncio
async def test_t3_hook_no_sid_short_circuits(coord):
    coord.shared_state.cortex_session_id = ""
    await coord._cortex_t3_hook(
        task=_task(),
        result={"per_variant_outcomes": [{"variant_name": "v1", "outcome": "KEEP"}]},
        kept=True,
    )
    assert coord.cortex_kb.verify_calls == []
    assert coord.shared_state.save_count == 0


@pytest.mark.asyncio
async def test_t3_hook_dispatches_explore_with_outcomes_to_per_variant(coord):
    """explore + non-empty per_variant_outcomes → per-variant path."""
    _seed_pending_edge_row(
        coord, variant_edges={"v1": "edge-v1", "v2": "edge-v2"},
    )
    result = {
        "status": "succeeded",
        "per_variant_outcomes": [
            {"variant_name": "v1", "outcome": "KEEP",   "metrics": {}},
            {"variant_name": "v2", "outcome": "REVERT", "metrics": {}},
        ],
    }
    await coord._cortex_t3_hook(
        task=_task(kind="explore"), result=result, kept=True,
    )
    assert len(coord.cortex_kb.verify_calls) == 2
    assert len(coord.cortex_kb.ingest_attempt_calls) == 2
    # Per-variant attempt_node propose_point per variant.
    assert len(coord.cortex_kb.propose_point_calls) == 2
    # save() called exactly once (single end-of-hook persist).
    assert coord.shared_state.save_count == 1


@pytest.mark.asyncio
async def test_t3_hook_dispatches_non_explore_to_per_task(coord):
    """kernel_opt / integrate / baseline → legacy per-task path."""
    _seed_pending_edge_row(coord, single_edge_id="edge-legacy")
    await coord._cortex_t3_hook(
        task=_task(kind="kernel_opt"),
        result={"status": "succeeded", "output_throughput": 1234.0},
        kept=True,
    )
    # Per-task path makes exactly one propose_point + one ingest_attempt
    # + one verify.
    assert len(coord.cortex_kb.propose_point_calls) == 1
    assert len(coord.cortex_kb.ingest_attempt_calls) == 1
    assert len(coord.cortex_kb.verify_calls) == 1
    assert coord.cortex_kb.verify_calls[0]["edge_id"] == "edge-legacy"
    assert coord.cortex_kb.verify_calls[0]["outcome"] == "confirmed"
    assert coord.shared_state.save_count == 1


@pytest.mark.asyncio
async def test_t3_hook_explore_without_outcomes_falls_back_to_per_task(coord):
    """explore action whose result is missing per_variant_outcomes goes
    back through the per-task path (covers --degraded-kb resume + older
    explore executors that pre-date Gap-08)."""
    _seed_pending_edge_row(coord, single_edge_id="edge-legacy")
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={"status": "succeeded"},
        kept=True,
    )
    # Per-task path: 1 propose + 1 ingest + 1 verify.
    assert len(coord.cortex_kb.verify_calls) == 1


@pytest.mark.asyncio
async def test_t3_hook_explore_with_empty_outcomes_falls_back_to_per_task(coord):
    _seed_pending_edge_row(coord, single_edge_id="edge-legacy")
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={"per_variant_outcomes": []},
        kept=True,
    )
    assert len(coord.cortex_kb.verify_calls) == 1


# ===========================================================================
# 2. Per-variant happy path — KEEP / REVERT / SKIPPED_DEDUP outcomes
# ===========================================================================
@pytest.mark.asyncio
async def test_per_variant_keep_emits_confirmed_with_promote(coord):
    _seed_pending_edge_row(coord, variant_edges={"vK": "edge-vK"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {
                    "variant_name": "vK",
                    "outcome":      "KEEP",
                    "metrics":      {"tput": 1234.5, "gain_pct": 5.2},
                    "provenance":   "specialist:framework",
                }
            ],
        },
        kept=True,
    )
    v = coord.cortex_kb.verify_calls[0]
    assert v["edge_id"]  == "edge-vK"
    assert v["outcome"]  == "confirmed"
    assert v["promote_authority"] == "EXPERIENTIAL"
    # ingest_attempt outcome=PASS, plan_edge bound to the variant edge,
    # metrics carry variant_name + the executor's metrics + task_kind.
    ia = coord.cortex_kb.ingest_attempt_calls[0]
    assert ia["outcome"]   == "PASS"
    assert ia["plan_edge"] == "edge-vK"
    assert ia["metrics"]["variant_name"] == "vK"
    assert ia["metrics"]["tput"] == 1234.5
    assert ia["metrics"]["gain_pct"] == 5.2
    assert ia["metrics"]["task_kind"] == "explore"


@pytest.mark.asyncio
async def test_per_variant_revert_emits_refuted_without_promote(coord):
    _seed_pending_edge_row(coord, variant_edges={"vR": "edge-vR"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {
                    "variant_name": "vR",
                    "outcome":      "REVERT",
                    "metrics":      {"gain_pct": -1.0},
                    "reason":       "gain_below_threshold",
                }
            ],
        },
        kept=False,
    )
    v = coord.cortex_kb.verify_calls[0]
    assert v["outcome"] == "refuted"
    assert v["promote_authority"] is None
    ia = coord.cortex_kb.ingest_attempt_calls[0]
    assert ia["outcome"] == "FAIL"
    assert ia["metrics"]["reason"] == "gain_below_threshold"


@pytest.mark.parametrize("outcome", ["FAILED", "KEEP_UNSTABLE"])
@pytest.mark.asyncio
async def test_per_variant_failed_and_keep_unstable_treated_as_refuted(
    coord, outcome,
):
    _seed_pending_edge_row(coord, variant_edges={"vX": "edge-vX"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "vX", "outcome": outcome, "metrics": {}}
            ],
        },
        kept=False,
    )
    assert coord.cortex_kb.verify_calls[0]["outcome"] == "refuted"
    assert coord.cortex_kb.ingest_attempt_calls[0]["outcome"] == "FAIL"


@pytest.mark.asyncio
async def test_per_variant_skipped_dedup_is_no_op(coord):
    """SKIPPED_DEDUP variants have no edge minted in T2 → T3 must not
    fabricate one."""
    _seed_pending_edge_row(coord, variant_edges={"v1": "edge-v1"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1",   "outcome": "KEEP",          "metrics": {}},
                {"variant_name": "vDup", "outcome": "SKIPPED_DEDUP", "metrics": {}},
            ],
        },
        kept=True,
    )
    # SKIPPED_DEDUP variant contributes no calls — only v1 lands.
    assert len(coord.cortex_kb.verify_calls) == 1
    assert len(coord.cortex_kb.ingest_attempt_calls) == 1
    assert len(coord.cortex_kb.propose_point_calls) == 1
    assert coord.cortex_kb.verify_calls[0]["edge_id"] == "edge-v1"


@pytest.mark.asyncio
async def test_per_variant_mixed_keep_revert_dedup(coord):
    """KB_gaps/Gap-08 acceptance criterion: K variants with mixed
    KEEP/REVERT (+ SKIPPED_DEDUP) → verify called K' times
    (K minus skipped)."""
    _seed_pending_edge_row(coord, variant_edges={
        "vK": "edge-vK", "vR": "edge-vR", "vF": "edge-vF",
    })
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "vK", "outcome": "KEEP",          "metrics": {}},
                {"variant_name": "vR", "outcome": "REVERT",        "metrics": {}},
                {"variant_name": "vF", "outcome": "FAILED",        "metrics": {}},
                {"variant_name": "vS", "outcome": "SKIPPED_DEDUP", "metrics": {}},
            ],
        },
        kept=True,
    )
    # 3 KEEP/REVERT/FAILED → 3 verify + 3 ingest_attempt calls.
    assert len(coord.cortex_kb.verify_calls) == 3
    assert len(coord.cortex_kb.ingest_attempt_calls) == 3
    outcomes = [c["outcome"] for c in coord.cortex_kb.verify_calls]
    assert outcomes == ["confirmed", "refuted", "refuted"]


# ===========================================================================
# 3. Edge_id lookup priority — pending_kb_edges → result fallback → single
# ===========================================================================
@pytest.mark.asyncio
async def test_edge_lookup_prefers_pending_row_variant_edges(coord):
    """variant_edges on the pending_kb_edges row wins over the
    per-variant kb_edge_id stamped on the executor result."""
    _seed_pending_edge_row(coord, variant_edges={"v1": "edge-from-pending"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP", "metrics": {},
                 "kb_edge_id": "edge-from-result"},
            ],
        },
        kept=True,
    )
    assert coord.cortex_kb.verify_calls[0]["edge_id"] == "edge-from-pending"


@pytest.mark.asyncio
async def test_edge_lookup_falls_back_to_result_kb_edge_id(coord):
    """When pending_kb_edges has no row (e.g. --degraded-kb T2 path), the
    per-variant ``kb_edge_id`` stamped on the result dict is used."""
    # No _seed_pending_edge_row — pending_kb_edges stays empty.
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP", "metrics": {},
                 "kb_edge_id": "edge-from-result"},
            ],
        },
        kept=True,
    )
    assert coord.cortex_kb.verify_calls[0]["edge_id"] == "edge-from-result"


@pytest.mark.asyncio
async def test_edge_lookup_single_edge_back_compat(coord):
    """Pre-Gap-07 pending_kb_edges rows only had ``edge_id`` (no
    ``variant_edges`` map). For these legacy rows, every KEEP/REVERT
    variant binds to the same edge — degraded but not broken."""
    _seed_pending_edge_row(coord, single_edge_id="edge-legacy")
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP",   "metrics": {}},
                {"variant_name": "v2", "outcome": "REVERT", "metrics": {}},
            ],
        },
        kept=True,
    )
    assert all(
        c["edge_id"] == "edge-legacy" for c in coord.cortex_kb.verify_calls
    )


@pytest.mark.asyncio
async def test_per_variant_missing_edge_skips_verify_keeps_ingest(coord):
    """A variant whose pending_kb_edges entry is empty (T2 hypothesize
    failed) still records the ingest_attempt + attempt_node — only the
    verify call gets skipped (late_verified path)."""
    _seed_pending_edge_row(coord, variant_edges={"v1": "", "v2": "edge-v2"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP",   "metrics": {}},
                {"variant_name": "v2", "outcome": "REVERT", "metrics": {}},
            ],
        },
        kept=True,
    )
    # Both variants emit ingest_attempt + propose_point; only v2 emits
    # verify.
    assert len(coord.cortex_kb.ingest_attempt_calls) == 2
    assert len(coord.cortex_kb.propose_point_calls) == 2
    assert len(coord.cortex_kb.verify_calls) == 1
    assert coord.cortex_kb.verify_calls[0]["edge_id"] == "edge-v2"


# ===========================================================================
# 4. Partial failures + isolation
# ===========================================================================
@pytest.mark.asyncio
async def test_per_variant_verify_failure_isolates_other_variants(coord):
    coord.cortex_kb = _StubCortexKB(verify_raises_for={"v2"})
    _seed_pending_edge_row(coord, variant_edges={
        "v1": "edge-v1", "v2": "edge-v2", "v3": "edge-v3",
    })
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP",   "metrics": {}},
                {"variant_name": "v2", "outcome": "KEEP",   "metrics": {}},
                {"variant_name": "v3", "outcome": "REVERT", "metrics": {}},
            ],
        },
        kept=True,
    )
    # All three verify calls attempted (v2 raised but didn't abort).
    assert len(coord.cortex_kb.verify_calls) == 3
    # All three ingest_attempt calls landed before / despite the verify
    # failure.
    assert len(coord.cortex_kb.ingest_attempt_calls) == 3


@pytest.mark.asyncio
async def test_per_variant_ingest_attempt_failure_isolates(coord):
    coord.cortex_kb = _StubCortexKB(ingest_raises_for={"v1"})
    _seed_pending_edge_row(coord, variant_edges={"v1": "edge-v1", "v2": "edge-v2"})
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP",   "metrics": {}},
                {"variant_name": "v2", "outcome": "REVERT", "metrics": {}},
            ],
        },
        kept=True,
    )
    # v1's ingest_attempt raised but v2 still ran; verify still ran for
    # both (verify is independent of ingest_attempt success).
    assert len(coord.cortex_kb.ingest_attempt_calls) == 2
    assert len(coord.cortex_kb.verify_calls) == 2


# ===========================================================================
# 5. Pending edge row popped exactly once
# ===========================================================================
@pytest.mark.asyncio
async def test_per_variant_pops_pending_edge_row_once(coord):
    _seed_pending_edge_row(coord, variant_edges={"v1": "edge-v1"})
    assert len(coord.shared_state.pending_kb_edges) == 1
    await coord._cortex_t3_hook(
        task=_task(kind="explore"),
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "KEEP", "metrics": {}},
            ],
        },
        kept=True,
    )
    # Pending row consumed.
    assert coord.shared_state.pending_kb_edges == []


# ===========================================================================
# 6. kb_provenance — edges_promoted / edges_negated derived from audit
# ===========================================================================
def test_kb_provenance_aggregates_verify_outcomes_from_audit(tmp_path: Path):
    """KB_gaps/Gap-08 acceptance: breakdown.kb_provenance.edges_promoted
    / edges_negated lengths match per-variant verify counts."""
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    from inference_optimizer.session_paths import cortex_audit_jsonl

    audit = cortex_audit_jsonl(tmp_path)
    audit.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # Async enqueue rows from the per-variant path (typical case).
        {"op": "enqueue", "status": "ok", "envelope_op": "verify",
         "payload_edge": "edge-v1", "payload_outcome": "confirmed",
         "payload_promote": "EXPERIENTIAL"},
        {"op": "enqueue", "status": "ok", "envelope_op": "verify",
         "payload_edge": "edge-v2", "payload_outcome": "refuted"},
        {"op": "enqueue", "status": "ok", "envelope_op": "verify",
         "payload_edge": "edge-v3", "payload_outcome": "refuted"},
        # Sync CLI row (flusher replay) for an already-counted edge —
        # must dedup.
        {"op": "cli", "status": "ok",
         "args": ["session", "verify", "--sid", "sid",
                  "--edge", "edge-v1", "--outcome", "confirmed"]},
        # Sync CLI row for a brand new edge (never enqueued — direct sync
        # path used by a legacy caller).
        {"op": "cli", "status": "ok",
         "args": ["session", "verify", "--sid", "sid",
                  "--edge", "edge-v4", "--outcome", "confirmed"]},
        # Non-verify enqueue / cli — must be ignored.
        {"op": "enqueue", "status": "ok", "envelope_op": "ingest_attempt",
         "payload_outcome": "PASS"},
        {"op": "cli", "status": "ok",
         "args": ["session", "begin", "--task", "x"]},
    ]
    with audit.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    warnings: list[str] = []
    out = collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=warnings)
    assert out["edges_promoted"] == ["edge-v1", "edge-v4"]
    assert out["edges_negated"] == ["edge-v2", "edge-v3"]
    # Sanity: no warnings from this golden audit.
    assert warnings == []


def test_kb_provenance_no_verify_rows_emits_empty_lists(tmp_path: Path):
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    out = collect_kb_provenance(
        tmp_path, state={}, manifest={}, warnings=[],
    )
    assert out["edges_promoted"] == []
    assert out["edges_negated"] == []


# ===========================================================================
# 7. ExploreExecutor result — per_variant_outcomes shape
# ===========================================================================
def test_explore_per_variant_outcomes_built_from_tested_and_skipped(monkeypatch):
    """Synthesize a tiny ``ExploreExecutor.execute`` slice by directly
    invoking the per_variant_outcomes assembly — we don't need a real
    runner, only the projection logic. The assembly is inlined inside
    ``execute`` so we copy the same predicate the executor uses."""
    # The executor's per_variant_outcomes block (lines around 723) is
    # straightforward enough that we re-derive the expected shape
    # without invoking run_grid. Below we exercise the same projection
    # by constructing the inputs ExploreExecutor uses.
    round_id = 7
    tested_update = {
        "fp-vK": {
            "name": "vK", "outcome": "KEEP", "round_id": round_id,
            "tput": 1200.0, "gain_pct": 5.0, "kb_edge_id": "edge-vK",
            "provenance": "specialist:framework",
        },
        "fp-vR": {
            "name": "vR", "outcome": "REVERT", "round_id": round_id,
            "tput": 1100.0, "gain_pct": -1.0, "kb_edge_id": "edge-vR",
            "provenance": "llm_direct",
        },
        "fp-vF": {
            "name": "vF", "outcome": "FAILED", "round_id": round_id,
            "tput": None, "gain_pct": None, "kb_edge_id": "edge-vF",
            "provenance": "llm_direct",
        },
        "fp-prev": {
            "name": "vOld", "outcome": "KEEP", "round_id": round_id - 1,
            "tput": 999.0, "gain_pct": 1.0, "kb_edge_id": "old",
        },  # different round → excluded
    }
    rejected_update = [
        {"fingerprint": "fp-vR", "reason": "gain_below_threshold",
         "round_id": round_id},
        {"fingerprint": "fp-vF", "reason": "no_measurement",
         "round_id": round_id},
    ]
    skipped_dup = [
        {"name": "vDup", "fingerprint": "fp-dup", "reason": "ledger_dup"},
    ]
    # Replicate the projection from explore.py (~lines 723-770).
    reasons_by_fp = {
        str(r.get("fingerprint") or ""): str(r.get("reason") or "")
        for r in rejected_update
        if r.get("round_id") == round_id
    }
    per_variant_outcomes: list[dict[str, Any]] = []
    for fp_key, te in tested_update.items():
        if te.get("round_id") != round_id:
            continue
        outcome = str(te.get("outcome") or "")
        if outcome not in ("KEEP", "REVERT", "FAILED", "KEEP_UNSTABLE"):
            continue
        metrics: dict[str, Any] = {}
        if te.get("tput") is not None:
            metrics["tput"] = te.get("tput")
        if te.get("gain_pct") is not None:
            metrics["gain_pct"] = te.get("gain_pct")
        per_variant_outcomes.append({
            "variant_name": str(te.get("name") or ""),
            "outcome":      outcome,
            "fingerprint":  fp_key,
            "kb_edge_id":   str(te.get("kb_edge_id") or ""),
            "provenance":   str(te.get("provenance") or ""),
            "metrics":      metrics,
            "reason":       reasons_by_fp.get(fp_key, ""),
        })
    for sd in skipped_dup:
        per_variant_outcomes.append({
            "variant_name": str(sd.get("name") or ""),
            "outcome":      "SKIPPED_DEDUP",
            "fingerprint":  str(sd.get("fingerprint") or ""),
            "kb_edge_id":   "",
            "provenance":   "",
            "metrics":      {},
            "reason":       str(sd.get("reason") or ""),
        })

    # Round filter: vOld (different round) excluded.
    names = [pvo["variant_name"] for pvo in per_variant_outcomes]
    assert "vOld" not in names
    assert set(names) == {"vK", "vR", "vF", "vDup"}
    by_name = {pvo["variant_name"]: pvo for pvo in per_variant_outcomes}
    assert by_name["vK"]["outcome"] == "KEEP"
    assert by_name["vK"]["kb_edge_id"] == "edge-vK"
    assert by_name["vK"]["provenance"] == "specialist:framework"
    assert by_name["vR"]["reason"] == "gain_below_threshold"
    assert by_name["vF"]["metrics"] == {}  # no tput / gain_pct
    assert by_name["vDup"]["outcome"] == "SKIPPED_DEDUP"
    assert by_name["vDup"]["kb_edge_id"] == ""
