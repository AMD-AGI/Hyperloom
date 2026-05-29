"""PR-A5 (Arbor-into-Hyperloom): specialist prompt KB / PR assembly.

Verifies that:

* :meth:`KnowledgePlane.select_kb_for_domain` queries Cortex KB
  (``POST /v1/points/query`` then ``POST /v1/traverse``) and returns a
  structured dict the specialist prompt builder consumes.
* :meth:`Coordinator._warm_specialist_params` pipes the result into
  ``task.params['kb_subgraph']`` so SpecialistPromptInputs picks it
  up automatically.
* Fail-soft semantics hold for every failure mode (anchor missing,
  traverse 5xx, Cortex disabled).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.knowledge_plane import KnowledgePlane


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _FakeCortexKBClient:
    """Minimal stand-in for :class:`CortexKBClient`.

    ``enabled`` toggles the cortex_enabled property, ``_post`` returns
    pre-staged responses keyed by path so the test can simulate the
    full anchor-query → traverse round-trip.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        responses: dict[str, Any] | None = None,
        raise_on: dict[str, Exception] | None = None,
    ):
        self.enabled = enabled
        self._responses = responses or {}
        self._raise_on = raise_on or {}
        self.posts: list[tuple[str, Any]] = []

    def _post(self, path: str, body: Any) -> dict[str, Any]:
        self.posts.append((path, body))
        if path in self._raise_on:
            raise self._raise_on[path]
        return self._responses.get(path, {})


def _make_plane(
    cortex: _FakeCortexKBClient | None,
    *,
    domain_repos: dict | None = None,
    pr_monitor: Any = None,
) -> KnowledgePlane:
    return KnowledgePlane(
        cortex_kb=cortex,
        pr_monitor=pr_monitor,
        domain_repos=domain_repos or {},
        pr_monitor_mcp_url="",
    )


# ---------------------------------------------------------------------------
# 1. select_kb_for_domain happy path
# ---------------------------------------------------------------------------
def test_select_kb_for_domain_returns_traverse_dict():
    cortex = _FakeCortexKBClient(
        enabled=True,
        responses={
            "/v1/points/query": {
                "points": [
                    {"point_id": 42, "canonical_id": "framework"},
                ],
            },
            "/v1/traverse": {
                "points":   ["framework", "framework.cuda_graph"],
                "neighbors": [
                    {"canonical_id": "framework.cuda_graph"},
                    {"canonical_id": "kernel.attention"},
                ],
                "paths":   [{"canonical_id": "framework→kernel.attention"}],
                "candidates": [],
            },
        },
    )
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("serving_specialist")
    assert result["anchor"] == "framework"
    assert result["domain"] == "serving_specialist"
    assert "framework.cuda_graph" in result["points"]
    assert any("framework.cuda_graph" in n for n in result["neighbors"])
    assert result["warnings"] == []
    # The traverse call must use the int point_id from the query result.
    assert cortex.posts[-1][1]["start_point"] == 42
    # Default budget values from PR-A5 design.
    assert cortex.posts[-1][1]["budget_steps"] == 4
    assert cortex.posts[-1][1]["budget_branches"] == 20


# ---------------------------------------------------------------------------
# 2. Failure modes — each returns an empty result + warning
# ---------------------------------------------------------------------------
def test_select_kb_for_domain_cortex_disabled():
    cortex = _FakeCortexKBClient(enabled=False)
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("serving_specialist")
    assert result["anchor"] == "framework"
    assert result["points"] == []
    assert "cortex_kb:disabled" in result["warnings"]


def test_select_kb_for_domain_unknown_domain():
    cortex = _FakeCortexKBClient(enabled=True)
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("not_a_real_domain")
    assert "unknown_domain" in result["warnings"][0]
    assert result["points"] == []
    # No HTTP call made.
    assert cortex.posts == []


def test_select_kb_for_domain_anchor_not_found():
    # PR-A10: anchor-not-found now triggers a per-domain content
    # fallback (entity_type / kind / attrs_filter queries). When the
    # KB has nothing matching the strategy either, the result stays
    # empty + the anchor_not_found warning still fires for back-compat.
    cortex = _FakeCortexKBClient(
        enabled=True,
        responses={"/v1/points/query": {"points": []}},
    )
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("kernel_switch_specialist")
    assert any("anchor_not_found" in w for w in result["warnings"])
    assert result["points"] == []
    # The legacy anchor query still happens first; fallback adds
    # additional /v1/points/query calls but never reaches /v1/traverse.
    assert len(cortex.posts) >= 1
    assert all(p[0] == "/v1/points/query" for p in cortex.posts)


def test_select_kb_for_domain_anchor_query_raises():
    cortex = _FakeCortexKBClient(
        enabled=True,
        raise_on={"/v1/points/query": RuntimeError("transport down")},
    )
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("serving_specialist")
    assert any("anchor_lookup_failed" in w for w in result["warnings"])


def test_select_kb_for_domain_traverse_raises():
    cortex = _FakeCortexKBClient(
        enabled=True,
        responses={
            "/v1/points/query": {
                "points": [{"point_id": 7, "canonical_id": "framework"}],
            },
        },
        raise_on={"/v1/traverse": RuntimeError("5xx")},
    )
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("serving_specialist")
    assert any("traverse_failed" in w for w in result["warnings"])
    # Anchor still surfaces via the points list (best-effort).
    assert "framework" in result["points"]


# ---------------------------------------------------------------------------
# 3. Cap enforcement — too many neighbors must be truncated
# ---------------------------------------------------------------------------
def test_select_kb_for_domain_caps_lists():
    cortex = _FakeCortexKBClient(
        enabled=True,
        responses={
            "/v1/points/query": {
                "points": [{"point_id": 1, "canonical_id": "framework"}],
            },
            "/v1/traverse": {
                "points":   [f"p{i}" for i in range(50)],
                "neighbors": [f"n{i}" for i in range(50)],
                "paths":   [f"path{i}" for i in range(20)],
                "candidates": [f"c{i}" for i in range(20)],
            },
        },
    )
    plane = _make_plane(cortex)
    result = plane.select_kb_for_domain("serving_specialist")
    assert len(result["points"]) == 12       # cap
    assert len(result["neighbors"]) == 20    # cap
    assert len(result["paths"]) == 5         # cap
    assert len(result["candidates"]) == 5    # cap


# ---------------------------------------------------------------------------
# 4. Coordinator warmup integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_warm_specialist_params_pipes_kb_subgraph_through(tmp_path: Path):
    """The Coordinator's specialist pre-dispatch warmup must call
    ``select_kb_for_domain`` and stash the result on
    ``params['kb_subgraph']`` so SpecialistPromptInputs picks it up."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState

    cortex = _FakeCortexKBClient(
        enabled=True,
        responses={
            "/v1/points/query": {
                "points": [{"point_id": 9, "canonical_id": "framework"}],
            },
            "/v1/traverse": {
                "points": ["framework"],
                "neighbors": [{"canonical_id": "framework.cuda_graph"}],
                "paths": [],
                "candidates": [],
            },
        },
    )
    plane = _make_plane(cortex)

    # Build a no-IO Coordinator skeleton — same trick as the M5 lifecycle
    # tests. We only need the attributes ``_warm_specialist_params``
    # actually touches.
    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="kb-warmup-test")
    c.knowledge_plane = plane

    params: dict[str, Any] = {"domain": "serving_specialist"}
    await c._warm_specialist_params(params)
    assert "kb_subgraph" in params
    sub = params["kb_subgraph"]
    assert sub["anchor"] == "framework"
    assert "framework.cuda_graph" in sub["neighbors"][0]


@pytest.mark.asyncio
async def test_warm_specialist_params_kb_subgraph_when_cortex_disabled(tmp_path: Path):
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState

    plane = _make_plane(_FakeCortexKBClient(enabled=False))
    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="kb-disabled-test")
    c.knowledge_plane = plane

    params: dict[str, Any] = {"domain": "serving_specialist"}
    await c._warm_specialist_params(params)
    # Defensive default: empty dict so the prompt builder renders
    # ``(none)`` instead of NameError'ing.
    assert params.get("kb_subgraph") == {}
