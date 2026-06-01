"""Directed + cross-repo + dedup coverage for the FRAMEWORK_PR discover
batch builder.

Complements ``test_framework_pr_discover_retry.py`` (which pins discovery
to a single repo for the per-batch failure-counter semantics). Here we
exercise the *enhanced* behaviour:

  - ``compose_gap`` drives a directed gap + keywords that are threaded
    into the ``phase_discover`` request.
  - The cross-repo loop queries every repo the ``pr_intel_specialist``
    domain tracks (FRAMEWORK_PR and EXPLORE share that repo set; neither
    owns it privately) and merges the candidates into one batch.
  - Cross-batch / cross-repo de-dup: a PR already discovered earlier is
    dropped from a later batch.
  - ``framework_pr_max_candidates`` overrides the per-repo cap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator import framework_agent_client as _fa_client
from inference_optimizer.orchestrator import coordinator as _coord_mod
from inference_optimizer.orchestrator.coordinator import (
    DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES,
    Coordinator,
)
from inference_optimizer.orchestrator.specialist_domains import get_domain


class _StateStub:
    def __init__(self) -> None:
        self.phase = "FRAMEWORK_PR"
        self.framework_pr_phase_done = False
        self.framework_pr_discover_failures = 0
        self.framework_pr_batches: list[dict[str, Any]] = []
        self.framework_pr_phase_progress: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.model = "test-model"
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.model_class = "dense"
        self.precision = "fp8"
        self.framework_pr_max_candidates = 0
        self.last_profile_kernel_breakdown = None

    def save(self, _session_dir: Path) -> None:
        pass


class _CoordinatorStub:
    """Binds the *real* Coordinator discover + helper methods (including
    the cross-repo fan-out) so the enhanced behaviour is exercised end to
    end against a mocked ``phase_discover``."""

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.framework_pr_discover_timeout_sec = 0.0

    def _framework_pr_discover_repo_urls(self, framework: str) -> list[str]:
        return Coordinator._framework_pr_discover_repo_urls(self, framework)  # type: ignore[arg-type]

    def _framework_pr_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_pr_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_pr_tried_refs(self) -> list[str]:
        return Coordinator._framework_pr_tried_refs(self)  # type: ignore[arg-type]


def _call_discover(stub: _CoordinatorStub) -> bool:
    return asyncio.run(
        Coordinator._discover_next_framework_pr_batch(stub)  # type: ignore[arg-type]
    )


def test_repo_urls_cover_pr_intel_set_with_framework_primary():
    """The FRAMEWORK_PR repo set leads with the framework's own repo and
    includes every pr_intel_specialist repo (the shared discovery
    surface), de-duplicated and order-preserving."""
    stub = _CoordinatorStub(Path("/tmp"))
    urls = stub._framework_pr_discover_repo_urls("sglang")

    assert urls[0] == _fa_client.repo_url_for_framework("sglang")
    # Every pr_intel repo is represented.
    domain = get_domain("pr_intel_specialist")
    for repo in domain.pr_repos:
        expected = f"https://github.com/{repo}.git"
        assert expected in urls or repo in "".join(urls)
    # No duplicates.
    assert len(urls) == len(set(urls))


def test_discover_merges_candidates_across_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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

    n_repos = len(stub._framework_pr_discover_repo_urls("sglang"))
    assert len(seen_repo_urls) == n_repos
    batch = stub.shared_state.framework_pr_batches[-1]
    # One merged candidate per repo (all distinct).
    assert batch["candidate_count"] == n_repos


def test_discover_threads_directed_gap_and_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    stub.shared_state.framework_pr_max_candidates = 13

    _call_discover(stub)
    assert captured["max_candidates"] == 13


def test_discover_default_max_candidates_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    stub.shared_state.framework_pr_max_candidates = 0

    _call_discover(stub)
    assert captured["max_candidates"] == DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES


def test_discover_dedups_against_prior_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    stub.shared_state.framework_pr_batches = [
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
    assert len(stub.shared_state.framework_pr_batches) == 1


def test_discover_intra_batch_dedup_across_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    batch = stub.shared_state.framework_pr_batches[-1]
    assert batch["candidate_count"] == 1


def test_discover_survives_partial_repo_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    assert stub.shared_state.framework_pr_discover_failures == 0
    batch = stub.shared_state.framework_pr_batches[-1]
    assert batch["candidate_count"] >= 1


def test_discover_keywords_dedup_in_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
            model="m", framework="sglang", gpu_type="MI300X",
            gaps=[{"gap_canonical_id": "", "gap_description": "x"}],
            session_dir=tmp_path,
            repo_url="https://github.com/sgl-project/sglang.git",
            keywords=["Dense", "dense", "FP8", "fp8", " moe "],
        )
    )
    req = captured["request"]
    assert req["keywords"] == ["dense", "fp8", "moe"]
