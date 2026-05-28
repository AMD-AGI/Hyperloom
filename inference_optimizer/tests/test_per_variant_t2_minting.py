"""v0.8 KB_design §3.13 M5 §5 step 6 / KB_gaps/Gap-07 — per-variant T2.

KB_gaps/Gap-07 root cause: ``_cortex_t2_hook`` always minted one
optimization_node + one hypothesize edge per proposal, even when the
proposal was an ``explore`` action with a multi-variant ``params.grid``.
Cross-session KB queries could therefore only attribute confirms /
refutes at proposal granularity, not per variant — and Gap-11's
per-variant Critic verdicts had no edge to bind to.

This file covers:

* PendingProposal data-structure extension (``kb_edge_ids`` +
  ``kb_opt_canonicals`` per-variant maps; legacy single fields
  retained).
* ``_resolve_issue_canonical`` helper priority (top-level
  ``gap_canonical_id`` → ``params.gap_canonical_id`` → workload
  anchor fallback).
* T2 hook dispatch — explore + grid routes to per-variant path;
  non-explore / no-grid / explore with empty grid routes to single
  path (v0.8 M1 behavior preserved).
* Per-variant T2 — N optimization_nodes + N edges minted; canonical
  ids include variant name; legacy ``kb_edge_id`` / ``kb_opt_canonical``
  populated with representative (first-with-edge) variant; partial
  failures isolated; nameless variants skipped.
* ``pending_kb_edges`` row carries ``variant_edges`` +
  ``variant_canonicals`` maps so Gap-08 T3 can iterate per-variant
  without a schema bump.
* ``_materialize_approved_proposal`` stamps per-variant
  ``kb_edge_id`` into the grid so the explore executor's existing
  variant-level reader works without further changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.cortex_kb_client import CortexKBError
from inference_optimizer.orchestrator.coordinator import (
    Coordinator, PendingProposal,
)


# ===========================================================================
# Fixtures — minimal Coordinator stub for T2 hook tests
# ===========================================================================
@dataclass
class _BareState:
    """SharedState double exposing only the fields T2 +
    _materialize_approved_proposal touch."""

    cortex_session_id: str = "sid-test"
    model_name: str = "llama-3-70b"
    gpu_type: str = "MI300X"
    phase: str = "EXPLORE"
    pending_kb_edges: list[dict[str, Any]] = field(default_factory=list)
    # Materialize-path reads:
    current_best: dict[str, Any] = field(default_factory=dict)
    baseline_tput: float = 0.0
    baseline_config_path: str = ""
    synergy_attempted: list = field(default_factory=list)
    backends_search: dict = field(default_factory=dict)
    params_search: dict = field(default_factory=dict)
    # Pruned-family check.
    pruned_families: list = field(default_factory=list)
    # Pending auto-roofline gate.
    auto_roofline_pending_task_id: str = ""

    save_count: int = 0
    session_iter_index: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1

    def is_pruned(self, _action: str) -> bool:
        return False

    def reset_policy_denial_streak(self, _action: str) -> None:
        return None

    def increment_session_iter_index(self) -> int:
        self.session_iter_index += 1
        return self.session_iter_index


class _StubCortexKB:
    """CortexKBClient double recording every call. Behaviour can be
    parametrised via ``hypothesize_outcomes`` (queue of dicts) and
    ``hypothesize_raises`` (queue of exceptions to raise at index)."""

    enabled: bool = True

    def __init__(
        self,
        *,
        hypothesize_outcomes: list[dict] | None = None,
        propose_point_raises_for: set[str] | None = None,
        hypothesize_raises_for: set[str] | None = None,
    ):
        self.propose_point_calls: list[dict] = []
        self.hypothesize_calls: list[dict] = []
        self._outcomes = list(hypothesize_outcomes or [])
        self._propose_point_raises_for = propose_point_raises_for or set()
        self._hypothesize_raises_for = hypothesize_raises_for or set()

    def propose_point(self, **kwargs):
        self.propose_point_calls.append(dict(kwargs))
        # Trigger an error for variants the test wants to fail
        attrs = kwargs.get("attrs") or {}
        if attrs.get("variant_name") in self._propose_point_raises_for:
            raise CortexKBError("synthetic propose_point failure")

    def hypothesize(self, **kwargs):
        self.hypothesize_calls.append(dict(kwargs))
        attrs = kwargs.get("attrs") or {}
        if attrs.get("variant_name") in self._hypothesize_raises_for:
            raise CortexKBError("synthetic hypothesize failure")
        # Pop a pre-canned outcome if available; otherwise return a
        # synthetic edge id based on the call index.
        if self._outcomes:
            return self._outcomes.pop(0)
        idx = len(self.hypothesize_calls)
        return {"tentative_edge_id": f"edge-{idx}"}


@pytest.fixture
def coord(tmp_path: Path):
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.cortex_kb = _StubCortexKB()
    return c


def _pending(
    *,
    action_name: str = "explore",
    grid: list | None = None,
    gap_canonical_id: str = "",
    gap_canonical_id_in_params: str = "",
    reasoning: str = "test rationale",
) -> PendingProposal:
    payload: dict[str, Any] = {
        "action_name":         action_name,
        "predicted_gain_pct":  3.5,
        "reasoning":           reasoning,
    }
    if gap_canonical_id:
        payload["gap_canonical_id"] = gap_canonical_id
    params: dict[str, Any] = {}
    if grid is not None:
        params["grid"] = grid
    if gap_canonical_id_in_params:
        params["gap_canonical_id"] = gap_canonical_id_in_params
    if params:
        payload["params"] = params
    return PendingProposal(
        proposal_msg_id="msg-1",
        from_agent="orchestration",
        action_name=action_name,
        predicted_gain_pct=3.5,
        payload=payload,
    )


# ===========================================================================
# 1. PendingProposal data-structure surface
# ===========================================================================
def test_pending_proposal_has_per_variant_fields():
    """KB_gaps/Gap-07: ``kb_edge_ids`` + ``kb_opt_canonicals`` maps
    must exist alongside the legacy single-id fields."""
    fields = PendingProposal.__dataclass_fields__
    assert "kb_edge_ids" in fields
    assert "kb_opt_canonicals" in fields
    # Legacy single-id fields retained for back-compat.
    assert "kb_edge_id" in fields
    assert "kb_opt_canonical" in fields


def test_pending_proposal_per_variant_defaults_to_empty_dict():
    p = PendingProposal(
        proposal_msg_id="m", from_agent="orchestration",
        action_name="kernel_opt", predicted_gain_pct=0.0, payload={},
    )
    assert p.kb_edge_ids == {}
    assert p.kb_opt_canonicals == {}
    assert p.kb_edge_id == ""
    assert p.kb_opt_canonical == ""


# ===========================================================================
# 2. _resolve_issue_canonical helper
# ===========================================================================
def test_resolve_issue_canonical_prefers_top_level_gap_id(coord):
    pending = _pending(gap_canonical_id="issue.attention.fp8_kv")
    out = coord._resolve_issue_canonical(pending)
    assert out == "issue.attention.fp8_kv"


def test_resolve_issue_canonical_falls_back_to_params_gap_id(coord):
    pending = _pending(gap_canonical_id_in_params="issue.scheduler.moe")
    out = coord._resolve_issue_canonical(pending)
    assert out == "issue.scheduler.moe"


def test_resolve_issue_canonical_workload_anchor_fallback(coord):
    """No explicit gap_canonical_id → fall back to the registered
    ``recipe:{slug(model)}:{slug(hw)}`` anchor."""
    pending = _pending()  # no gap_canonical_id anywhere
    out = coord._resolve_issue_canonical(pending)
    assert out.startswith("recipe:")
    # ``llama-3-70b`` / ``MI300X`` from _BareState defaults.
    assert "llama-3-70b" in out
    assert "mi300x" in out


def test_resolve_issue_canonical_top_level_beats_params(coord):
    """When both are set, top-level wins (matches docstring)."""
    pending = _pending(
        gap_canonical_id="issue.top.win",
        gap_canonical_id_in_params="issue.params.lose",
    )
    out = coord._resolve_issue_canonical(pending)
    assert out == "issue.top.win"


# ===========================================================================
# 3. T2 hook dispatch — explore + grid vs single path
# ===========================================================================
@pytest.mark.asyncio
async def test_t2_hook_no_cortex_short_circuits(coord):
    coord.cortex_kb = None
    pending = _pending(grid=[{"name": "v1"}])
    await coord._cortex_t2_hook(pending)
    # Nothing minted, no per-variant maps populated.
    assert pending.kb_edge_ids == {}


@pytest.mark.asyncio
async def test_t2_hook_no_sid_short_circuits(coord):
    coord.shared_state.cortex_session_id = ""
    pending = _pending(grid=[{"name": "v1"}])
    await coord._cortex_t2_hook(pending)
    assert coord.cortex_kb.propose_point_calls == []


@pytest.mark.asyncio
async def test_t2_hook_non_explore_action_uses_single_path(coord):
    """kernel_opt / integrate / etc. still mint one edge per
    proposal — v0.8 M1 behaviour preserved."""
    pending = _pending(action_name="kernel_opt", grid=None)
    await coord._cortex_t2_hook(pending)
    assert len(coord.cortex_kb.propose_point_calls) == 1
    assert len(coord.cortex_kb.hypothesize_calls) == 1
    # Per-variant map stays empty for non-grid proposals.
    assert pending.kb_edge_ids == {}
    # Legacy fields populated.
    assert pending.kb_edge_id == "edge-1"
    assert pending.kb_opt_canonical.startswith("exp:")


@pytest.mark.asyncio
async def test_t2_hook_explore_without_grid_uses_single_path(coord):
    """explore action whose payload doesn't carry a grid (or carries
    an empty grid) still goes through the single-edge path."""
    pending = _pending(action_name="explore", grid=[])
    await coord._cortex_t2_hook(pending)
    # Single-edge path means one propose_point + one hypothesize.
    assert len(coord.cortex_kb.propose_point_calls) == 1
    assert len(coord.cortex_kb.hypothesize_calls) == 1
    assert pending.kb_edge_ids == {}


# ===========================================================================
# 4. T2 per-variant path — happy + edge cases
# ===========================================================================
@pytest.mark.asyncio
async def test_t2_hook_per_variant_mints_one_per_variant(coord):
    grid = [
        {"name": "v1", "extra_sglang_args": "--mla 1"},
        {"name": "v2", "extra_envs": {"FOO": "bar"}},
        {"name": "v3", "provenance": "specialist:framework"},
        {"name": "v4"},
    ]
    pending = _pending(grid=grid)
    await coord._cortex_t2_hook(pending)
    # 1 parent ``experiment`` anchor + 4 variant propose_point calls;
    # 4 hypothesize calls (one per variant; parent is anchor-only).
    assert len(coord.cortex_kb.propose_point_calls) == 5
    assert len(coord.cortex_kb.hypothesize_calls) == 4
    # Variant canonical_ids encode the variant name (parent does not).
    variant_calls = [
        call for call in coord.cortex_kb.propose_point_calls
        if ".variant-" in call["canonical_id"]
    ]
    assert len(variant_calls) == 4
    canonicals = [call["canonical_id"] for call in variant_calls]
    assert {c.split(".variant-")[1] for c in canonicals} == {"v1", "v2", "v3", "v4"}
    # Parent canonical is exp:{sid}:{iter:04d} (no .variant- suffix).
    parent_calls = [
        call for call in coord.cortex_kb.propose_point_calls
        if ".variant-" not in call["canonical_id"]
    ]
    assert len(parent_calls) == 1
    assert parent_calls[0]["canonical_id"].startswith("exp:")
    assert parent_calls[0]["kind"] == "experiment"
    # PendingProposal per-variant maps populated.
    assert set(pending.kb_edge_ids.keys()) == {"v1", "v2", "v3", "v4"}
    assert set(pending.kb_opt_canonicals.keys()) == {"v1", "v2", "v3", "v4"}
    # Every variant got a non-empty edge_id (stub generates sequential).
    assert all(eid for eid in pending.kb_edge_ids.values())


@pytest.mark.asyncio
async def test_t2_hook_per_variant_carries_variant_attrs(coord):
    grid = [
        {
            "name": "vA",
            "extra_sglang_args": "--mla 1",
            "extra_envs": {"FOO": "bar"},
            "provenance": "specialist:serving_specialist",
        },
    ]
    pending = _pending(grid=grid)
    await coord._cortex_t2_hook(pending)
    # Index 0 is the parent ``experiment`` anchor; variant call is index 1.
    pp_call = coord.cortex_kb.propose_point_calls[1]
    attrs = pp_call["attrs"]
    assert attrs["variant_name"] == "vA"
    assert attrs["extra_sglang_args"] == "--mla 1"
    assert attrs["extra_envs"] == {"FOO": "bar"}
    assert attrs["provenance"] == "specialist:serving_specialist"
    assert attrs["proposal_msg_id"] == "msg-1"

    # Hypothesize attrs carry variant_name + provenance + phase.
    h_call = coord.cortex_kb.hypothesize_calls[0]
    h_attrs = h_call["attrs"]
    assert h_attrs["variant_name"] == "vA"
    assert h_attrs["provenance"] == "specialist:serving_specialist"
    assert h_attrs["phase"] == "EXPLORE"
    assert h_attrs["proposal_msg_id"] == "msg-1"


@pytest.mark.asyncio
async def test_t2_hook_per_variant_skips_nameless_variants(coord):
    """A grid entry without a ``name`` field gets silently skipped —
    the executor itself rejects such variants downstream, so we don't
    even waste a propose_point on a phantom edge."""
    grid = [
        {"name": "v1"},
        {"extra_sglang_args": "--no-name"},  # nameless
        {"name": ""},                        # empty name
        {"name": "v4"},
    ]
    pending = _pending(grid=grid)
    await coord._cortex_t2_hook(pending)
    # 1 parent anchor + 2 named variants (nameless / empty-name skipped).
    assert len(coord.cortex_kb.propose_point_calls) == 3
    assert set(pending.kb_edge_ids.keys()) == {"v1", "v4"}


@pytest.mark.asyncio
async def test_t2_hook_per_variant_isolates_propose_point_failure(coord):
    """A propose_point exception on one variant does not abort the
    remaining variants. The failed variant still records a
    canonical id (we attempt hypothesize) — the test asserts the
    other variants minted successfully."""
    coord.cortex_kb = _StubCortexKB(
        propose_point_raises_for={"v2"},
    )
    grid = [{"name": "v1"}, {"name": "v2"}, {"name": "v3"}]
    pending = _pending(grid=grid)
    await coord._cortex_t2_hook(pending)
    assert {"v1", "v2", "v3"} <= set(pending.kb_opt_canonicals.keys())
    # 1 parent anchor + 3 variant propose_point calls attempted (v2 raised);
    # 3 hypothesize calls because hypothesize still runs even when
    # propose_point fails.
    assert len(coord.cortex_kb.propose_point_calls) == 4
    assert len(coord.cortex_kb.hypothesize_calls) == 3
    # All three hypothesize calls succeeded → all three edge_ids set.
    assert all(eid for eid in pending.kb_edge_ids.values())


@pytest.mark.asyncio
async def test_t2_hook_per_variant_isolates_hypothesize_failure(coord):
    """A hypothesize exception on one variant only zeroes out THAT
    variant's edge_id — others retain theirs."""
    coord.cortex_kb = _StubCortexKB(
        hypothesize_raises_for={"v2"},
    )
    pending = _pending(grid=[
        {"name": "v1"}, {"name": "v2"}, {"name": "v3"},
    ])
    await coord._cortex_t2_hook(pending)
    assert pending.kb_edge_ids["v1"]
    assert pending.kb_edge_ids["v2"] == ""  # failure path
    assert pending.kb_edge_ids["v3"]


@pytest.mark.asyncio
async def test_t2_hook_per_variant_representative_legacy_fields(coord):
    """When at least one variant minted a non-empty edge_id, the
    legacy ``kb_edge_id`` / ``kb_opt_canonical`` fields land on the
    FIRST variant with an edge so the existing per-proposal T3 hook
    has something to verify."""
    coord.cortex_kb = _StubCortexKB(
        # First variant's hypothesize returns empty edge (NDJSON path);
        # second variant returns a real one → that's the representative.
        hypothesize_outcomes=[
            {},
            {"tentative_edge_id": "edge-v2-real"},
            {"tentative_edge_id": "edge-v3-real"},
        ],
    )
    pending = _pending(grid=[
        {"name": "v1"}, {"name": "v2"}, {"name": "v3"},
    ])
    await coord._cortex_t2_hook(pending)
    assert pending.kb_edge_id == "edge-v2-real"
    assert pending.kb_opt_canonical == pending.kb_opt_canonicals["v2"]


@pytest.mark.asyncio
async def test_t2_hook_per_variant_all_failed_no_representative_edge(coord):
    """If every variant's hypothesize returns empty, legacy
    ``kb_edge_id`` is empty but ``kb_opt_canonical`` falls back to
    the FIRST variant's canonical id (so the T3 propose-edge
    fallback path still has something to anchor on)."""
    coord.cortex_kb = _StubCortexKB(
        hypothesize_outcomes=[{}, {}],
    )
    pending = _pending(grid=[{"name": "v1"}, {"name": "v2"}])
    await coord._cortex_t2_hook(pending)
    assert pending.kb_edge_id == ""
    # First variant's canonical id is the fallback representative.
    assert pending.kb_opt_canonical == pending.kb_opt_canonicals["v1"]


# ===========================================================================
# 5. pending_kb_edges row carries per-variant extension fields
# ===========================================================================
@pytest.mark.asyncio
async def test_pending_kb_edges_row_carries_variant_maps(coord):
    """The single pending_kb_edges row (back-compat with legacy T3)
    is extended with ``variant_edges`` + ``variant_canonicals`` maps
    so Gap-08 can iterate per-variant without a schema bump."""
    pending = _pending(grid=[{"name": "v1"}, {"name": "v2"}])
    await coord._cortex_t2_hook(pending)
    rows = coord.shared_state.pending_kb_edges
    assert len(rows) == 1
    row = rows[0]
    assert row["proposal_msg_id"] == "msg-1"
    assert row["action"] == "explore"
    # Per-variant maps present and consistent with PendingProposal.
    assert row["variant_edges"] == pending.kb_edge_ids
    assert row["variant_canonicals"] == pending.kb_opt_canonicals


@pytest.mark.asyncio
async def test_pending_kb_edges_row_single_path_omits_variant_maps(coord):
    """Single-path (non-grid) rows don't carry the extension fields
    — keeps the schema lean for kernel_opt / integrate / etc."""
    pending = _pending(action_name="kernel_opt")
    await coord._cortex_t2_hook(pending)
    row = coord.shared_state.pending_kb_edges[0]
    assert "variant_edges" not in row
    assert "variant_canonicals" not in row


# ===========================================================================
# 6. _materialize_approved_proposal stamps kb_edge_id into grid
# ===========================================================================
@pytest.mark.asyncio
async def test_materialize_stamps_per_variant_kb_edge_id(coord, monkeypatch):
    """When the approved proposal is explore + grid + populated
    kb_edge_ids, the task params grid carries each variant's
    kb_edge_id so the explore executor sees it."""
    # Stub TaskRegistry + bus so _materialize_approved_proposal runs
    # without a full Coordinator init.
    @dataclass
    class _StubTaskRow:
        task_id: str
        kind: str
        state: str
        params: dict
        idempotency_key: str

    class _StubTaskRegistry:
        def __init__(self):
            self.last_params: dict | None = None

        async def create_or_return_existing(self, *, kind, params, **_kw):
            self.last_params = params
            return _StubTaskRow(
                task_id="t-mat", kind=kind, state="queued",
                params=params, idempotency_key=_kw.get("idempotency_key", ""),
            ), False

    class _StubBus:
        async def append_and_seq(self, _msg):
            return None

    coord.tasks = _StubTaskRegistry()
    coord.bus = _StubBus()
    # Avoid touching the action_registry / sequence_actions gate by
    # short-circuiting _record_observation; _materialize_approved_proposal
    # uses it on the duplicate-id path.
    async def _noop_observation(*_a, **_kw):
        return None
    coord._record_observation = _noop_observation  # type: ignore[method-assign]

    pending = _pending(grid=[
        {"name": "v1", "extra_sglang_args": "--mla 1"},
        {"name": "v2", "extra_envs": {"FOO": "bar"}},
    ])
    # Pretend T2 already populated the maps:
    pending.kb_edge_ids = {"v1": "edge-v1", "v2": "edge-v2"}
    pending.kb_opt_canonicals = {
        "v1": "exp:sid:0001.variant-v1",
        "v2": "exp:sid:0001.variant-v2",
    }
    await coord._materialize_approved_proposal(pending)

    stamped_grid = coord.tasks.last_params["grid"]
    assert stamped_grid[0]["kb_edge_id"] == "edge-v1"
    assert stamped_grid[1]["kb_edge_id"] == "edge-v2"
    # Originals (extra_sglang_args / extra_envs) preserved.
    assert stamped_grid[0]["extra_sglang_args"] == "--mla 1"
    assert stamped_grid[1]["extra_envs"] == {"FOO": "bar"}


@pytest.mark.asyncio
async def test_materialize_no_stamp_when_kb_edge_ids_empty(coord):
    """When the T2 hook didn't populate kb_edge_ids (e.g. --degraded-kb
    runs), the grid stays unchanged — no spurious 'kb_edge_id=""'
    fields injected."""
    @dataclass
    class _StubTaskRow:
        task_id: str
        kind: str
        state: str
        params: dict
        idempotency_key: str

    class _StubTaskRegistry:
        def __init__(self):
            self.last_params: dict | None = None

        async def create_or_return_existing(self, *, kind, params, **_kw):
            self.last_params = params
            return _StubTaskRow(
                task_id="t-no-stamp", kind=kind, state="queued",
                params=params, idempotency_key="",
            ), False

    class _StubBus:
        async def append_and_seq(self, _msg):
            return None

    coord.tasks = _StubTaskRegistry()
    coord.bus = _StubBus()

    async def _noop_observation(*_a, **_kw):
        return None
    coord._record_observation = _noop_observation  # type: ignore[method-assign]

    pending = _pending(grid=[{"name": "v1", "extra_sglang_args": "--mla 1"}])
    # kb_edge_ids intentionally empty (e.g. --degraded-kb).
    await coord._materialize_approved_proposal(pending)
    stamped_grid = coord.tasks.last_params["grid"]
    assert "kb_edge_id" not in stamped_grid[0]


@pytest.mark.asyncio
async def test_materialize_does_not_overwrite_existing_kb_edge_id(coord):
    """If a variant already carries a ``kb_edge_id`` (e.g. an operator
    pre-populated it for a replay scenario), the materialization step
    must not clobber it with the T2 hook's value."""
    @dataclass
    class _StubTaskRow:
        task_id: str
        kind: str
        state: str
        params: dict
        idempotency_key: str

    class _StubTaskRegistry:
        def __init__(self):
            self.last_params: dict | None = None

        async def create_or_return_existing(self, *, kind, params, **_kw):
            self.last_params = params
            return _StubTaskRow(
                task_id="t-no-clobber", kind=kind, state="queued",
                params=params, idempotency_key="",
            ), False

    class _StubBus:
        async def append_and_seq(self, _msg):
            return None

    coord.tasks = _StubTaskRegistry()
    coord.bus = _StubBus()

    async def _noop_observation(*_a, **_kw):
        return None
    coord._record_observation = _noop_observation  # type: ignore[method-assign]

    pending = _pending(grid=[
        {"name": "v1", "kb_edge_id": "preset-edge-id"},
    ])
    pending.kb_edge_ids = {"v1": "t2-fresh-edge-id"}
    await coord._materialize_approved_proposal(pending)
    stamped_grid = coord.tasks.last_params["grid"]
    assert stamped_grid[0]["kb_edge_id"] == "preset-edge-id"
