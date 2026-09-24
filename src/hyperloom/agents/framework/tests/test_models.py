# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.models.* dataclasses. Hermetic - exercises only ``from_dict`` / derived properties; no I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.framework.models import (
    Candidate,
    ExploreRequest,
    PRMonitorConfig,
)


# Candidate.slug / pr_number --------------------------------------------


def test_candidate_slug_normalises_special_chars() -> None:
    """Slug must be filesystem-safe."""
    c = Candidate(ref="PR:123", repo="r", source="explicit")
    assert c.slug == "pr-123"
    c2 = Candidate(ref="release/v0.8.x", repo="r", source="explicit")
    assert c2.slug == "release-v0.8.x"


def test_candidate_pr_number() -> None:
    """Candidate.pr_number returns int for PR refs, None otherwise."""
    assert Candidate(ref="PR:42", repo="r", source="x").pr_number == 42
    assert Candidate(ref="main", repo="r", source="x").pr_number is None
    assert Candidate(ref="PR:not_int", repo="r", source="x").pr_number is None


# PRMonitorConfig -----------------------------------------------------


def test_pr_monitor_config_requires_base_url() -> None:
    """PRMonitorConfig.from_dict rejects empty/missing base_url."""
    with pytest.raises(ValueError, match="base_url"):
        PRMonitorConfig.from_dict({})


# ExploreRequest ---------------------------------------------------------


def _minimal_request_dict(**overrides) -> dict:
    """Return a minimal valid ExploreRequest payload for tests."""
    base = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": "/tmp/req",
        "baseline": {"throughput": 1.0, "accuracy": 0.9, "completed": "1/1"},
    }
    base.update(overrides)
    return base


def test_explore_request_minimal() -> None:
    """ExploreRequest.from_dict parses a minimal request and sets defaults."""
    r = ExploreRequest.from_dict(_minimal_request_dict())
    assert r.framework == "sglang"
    assert r.work_dir == Path("/tmp/req")
    assert r.search_modes == ("pr_monitor", "github")
    assert r.search_perf_prs is False
    assert r.gap_description == ""


def test_explore_request_derives_pr_monitor_from_kb_store(monkeypatch) -> None:
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example/knowledge-base")

    request = ExploreRequest.from_dict(_minimal_request_dict())

    assert request.pr_monitor is not None
    assert request.pr_monitor.base_url == "https://kb.example/knowledge-base/pr-monitor"


def test_explore_request_respects_runtime_pr_monitor_disable(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_PR_MONITOR_ENABLED", "0")

    request = ExploreRequest.from_dict(_minimal_request_dict())

    assert request.pr_monitor is None


def test_explore_request_requires_framework_and_repo_url() -> None:
    """from_dict rejects missing framework / repo_url."""
    with pytest.raises(ValueError, match="framework"):
        ExploreRequest.from_dict({"repo_url": "x", "baseline": {"throughput": 1}})
    with pytest.raises(ValueError, match="repo_url"):
        ExploreRequest.from_dict({"framework": "x", "baseline": {"throughput": 1}})


def test_explore_request_search_modes_validates() -> None:
    """Unknown search_modes entries should raise ValueError."""
    with pytest.raises(ValueError, match="search_modes"):
        ExploreRequest.from_dict(_minimal_request_dict(search_modes=["bad_backend"]))


def test_explore_request_search_modes_explicit_tuple() -> None:
    """Explicit search_modes is preserved in declared order."""
    r = ExploreRequest.from_dict(_minimal_request_dict(search_modes=["github", "pr_monitor"]))
    assert r.search_modes == ("github", "pr_monitor")
