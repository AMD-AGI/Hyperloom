# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Directed + cross-repo + dedup coverage for the FRAMEWORK discover batch builder."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.loop.coordinator import (
    DEFAULT_FRAMEWORK_MAX_CANDIDATES,
    Coordinator,
)
from hyperloom.orchestrator.specialists.domains import PR_QUERY_REPOS


class _StateStub:
    def __init__(self) -> None:
        self.phase = "FRAMEWORK"
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.model = "test-model"
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.model_class = "dense"
        self.precision = "fp8"
        self.framework_max_candidates = 0
        self.last_profile_kernel_breakdown = None

    def save(self, _session_dir: Path) -> None:
        pass


class _CoordinatorStub:
    """Binds the real Coordinator discover + helper methods against a mocked ``phase_discover``."""

    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _build_framework_working_memory = Coordinator._build_framework_working_memory
    _FRAMEWORK_TRIED_MEMORY_CAP = Coordinator._FRAMEWORK_TRIED_MEMORY_CAP
    # cross-framework discovery lane is default-on; the real discovery merge
    # calls this reverse-lookup on every repo it queries.
    _framework_agent_repo_url_origin_framework = staticmethod(
        Coordinator._framework_agent_repo_url_origin_framework
    )

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.framework_agent_discover_timeout_sec = 0.0

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        return Coordinator._framework_agent_discover_repo_urls(self, framework)  # type: ignore[arg-type]

    def _framework_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_tried_refs(self) -> list[str]:
        return Coordinator._framework_tried_refs(self)  # type: ignore[arg-type]


def _call_discover(stub: _CoordinatorStub) -> bool:
    return asyncio.run(
        Coordinator._discover_next_framework_batch(stub)  # type: ignore[arg-type]
    )


def test_repo_urls_cover_global_allowlist_with_framework_primary():
    """The repo set leads with the framework's own repo, includes every PR_QUERY_REPOS entry, de-duplicated and order-preserving."""
    stub = _CoordinatorStub(Path("/tmp"))
    urls = stub._framework_agent_discover_repo_urls("sglang")

    assert urls[0] == _fa_client.repo_url_for_framework("sglang")
    for repo in PR_QUERY_REPOS:
        expected = f"https://github.com/{repo}.git"
        assert expected in urls or repo in "".join(urls)
    assert len(urls) == len(set(urls))


def test_discover_merges_candidates_across_repos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each repo returns a distinct candidate; the batch merges them all."""
    seen_repo_urls: list[str] = []

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        repo_url = kwargs["repo_url"]
        seen_repo_urls.append(repo_url)
        # One unique candidate per repo.
        tag = repo_url.rsplit("/", 1)[-1].replace(".git", "")
        return {
            "batch_id": "b-merge",
            "candidates": [{"pr_url": f"https://example.com/{tag}/pr/1"}],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)

    ok = _call_discover(stub)
    assert ok is True

    n_repos = len(stub._framework_agent_discover_repo_urls("sglang"))
    assert len(seen_repo_urls) == n_repos
    batch = stub.shared_state.framework_agent_batches[-1]
    # One merged candidate per repo (all distinct).
    assert batch["candidate_count"] == n_repos


def test_discover_threads_directed_gap_and_keywords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """compose_gap output (directed gap + keywords) is threaded into the
    phase_discover request."""
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": [{"pr_url": "u"}]}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)

    _call_discover(stub)

    # Directed gap leads the gaps list (model_class=dense, precision=fp8).
    gaps = captured["gaps"]
    assert gaps[0]["gap_canonical_id"] == "directed"
    assert "dense" in gaps[0]["gap_description"]
    # Keywords derived from the workload taxonomy are passed through.
    kws = captured.get("keywords") or []
    assert "dense" in kws
    assert "fp8" in kws


def test_discover_uses_max_candidates_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    stub.shared_state.framework_max_candidates = 13

    _call_discover(stub)
    assert captured["max_candidates"] == 13


def test_discover_default_max_candidates_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    stub.shared_state.framework_max_candidates = 0

    _call_discover(stub)
    assert captured["max_candidates"] == DEFAULT_FRAMEWORK_MAX_CANDIDATES


def test_discover_dedups_against_prior_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A candidate already present in an earlier batch is dropped from the
    new batch, even when a repo re-surfaces it."""

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        return {
            "batch_id": "b2",
            # Same PR every repo returns → must collapse + dedup vs prior.
            "candidates": [{"pr_url": "https://example.com/dup/pr/1"}],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    # Pre-seed a prior batch carrying that exact candidate id.
    stub.shared_state.framework_agent_batches = [
        {
            "batch_id": "b1",
            "candidate_count": 1,
            "candidates": [
                {"candidate_id": "https://example.com/dup/pr/1"},
            ],
            "max_gain_pct_observed_in_batch": 0.0,
        },
    ]

    ok = _call_discover(stub)
    # Every discovered candidate was a duplicate → no new batch appended.
    assert ok is False
    assert len(stub.shared_state.framework_agent_batches) == 1


def test_discover_fallback_filters_processed_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Step B coordinator-side backstop: even if fa re-surfaces a candidate that
    already carries a terminal progress row (but is absent from any prior
    batch's candidate list), the coordinator re-filters it against the full
    excluded set (known ∪ processed) so it is never re-queued."""

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        return {
            "batch_id": "b-proc",
            "candidates": [
                {"pr_url": "https://example.com/processed/pr/7"},  # has a terminal row
                {"pr_url": "https://example.com/fresh/pr/8"},  # genuinely new
            ],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    # A terminal progress row keyed on the processed candidate, with NO prior
    # batch carrying it (so _framework_known_candidate_ids alone wouldn't catch it).
    stub.shared_state.framework_agent_phase_progress = [
        {"candidate_id": "https://example.com/processed/pr/7", "status": "reverted"},
    ]

    ok = _call_discover(stub)
    assert ok is True
    batch = stub.shared_state.framework_agent_batches[-1]
    ids = {c["candidate_id"] for c in batch["candidates"]}
    assert ids == {"https://example.com/fresh/pr/8"}


def test_discover_intra_batch_dedup_across_repos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When multiple repos all surface the same PR in one scan, the merged
    batch keeps a single copy."""

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        return {
            "batch_id": "b-intra",
            "candidates": [{"pr_url": "https://example.com/same/pr/9"}],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)

    ok = _call_discover(stub)
    assert ok is True
    batch = stub.shared_state.framework_agent_batches[-1]
    assert batch["candidate_count"] == 1


def test_discover_survives_partial_repo_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If some repos error but at least one succeeds, the batch is built
    from the survivors and the failure counter is NOT bumped."""
    calls = {"n": 0}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("simulated repo failure")
        tag = kwargs["repo_url"].rsplit("/", 1)[-1].replace(".git", "")
        return {
            "batch_id": "b-partial",
            "candidates": [{"pr_url": f"https://example.com/{tag}/pr/1"}],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)

    ok = _call_discover(stub)
    assert ok is True
    assert stub.shared_state.framework_agent_discover_failures == 0
    batch = stub.shared_state.framework_agent_batches[-1]
    assert batch["candidate_count"] >= 1


def test_discover_keywords_dedup_in_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """phase_discover lowercases + dedups keywords while preserving order."""
    captured: dict[str, Any] = {}

    def _fake_invoke(**kwargs: Any):
        captured["request"] = kwargs["request"]

        async def _coro() -> dict[str, Any]:
            return {"batch_id": "b", "candidates": []}

        return _coro()

    monkeypatch.setattr(_fa_client, "_invoke_fa_phase", _fake_invoke)

    asyncio.run(
        _fa_client.phase_discover(
            model="m",
            framework="sglang",
            gpu_type="MI300X",
            gaps=[{"gap_canonical_id": "", "gap_description": "x"}],
            session_dir=tmp_path,
            repo_url="https://github.com/sgl-project/sglang.git",
            keywords=["Dense", "dense", "FP8", "fp8", " moe "],
        )
    )
    req = captured["request"]
    assert req["keywords"] == ["dense", "fp8", "moe"]
