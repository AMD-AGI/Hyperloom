# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.models.* dataclasses. Hermetic - exercises only ``from_dict`` / derived properties; no I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.framework.models import (
    Baseline,
    Candidate,
    CommandSpec,
    ExploreRequest,
    PrFilter,
    PRMonitorConfig,
    Thresholds,
)


# Baseline ---------------------------------------------------------------


def test_baseline_requires_positive_throughput() -> None:
    """Baseline.from_dict rejects zero/negative throughput."""
    with pytest.raises(ValueError, match="throughput"):
        Baseline.from_dict({"throughput": 0})
    with pytest.raises(ValueError, match="throughput"):
        Baseline.from_dict({"throughput": -1})


def test_baseline_accepts_throughput_alias() -> None:
    """Baseline.from_dict accepts 'output_throughput' as a fallback key."""
    b = Baseline.from_dict({"output_throughput": 12.5})
    assert b.throughput == 12.5
    assert b.accuracy is None


# Thresholds -------------------------------------------------------------


def test_thresholds_defaults() -> None:
    """Thresholds.from_dict defaults when block missing/empty."""
    t = Thresholds.from_dict(None)
    assert t.min_throughput_ratio == 1.05
    assert t.max_accuracy_drop == 0.05


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


# CommandSpec ------------------------------------------------------------


def test_command_spec_rejects_empty_command() -> None:
    """CommandSpec.from_dict rejects an empty 'command' field."""
    with pytest.raises(ValueError, match="command"):
        CommandSpec.from_dict({"command": ""})


# PrFilter ---------------------------------------------------------------


def test_pr_filter_defaults_are_empty() -> None:
    """PrFilter.from_dict(None) yields a no-op filter."""
    f = PrFilter.from_dict(None)
    assert f.is_empty


def test_pr_filter_coerces_string_to_tuple() -> None:
    """PrFilter accepts a single string for list-typed fields."""
    f = PrFilter.from_dict({"include_paths": "python/"})
    assert f.include_paths == ("python/",)


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
