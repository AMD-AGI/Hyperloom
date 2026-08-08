# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A scriptable workload does not query the serving frameworks' PRs.

``_framework_agent_discover_repo_urls`` already refuses to hand serving/infra
PRs to a scriptable model repo, because they cannot be git-applied there. The
guard requires both a repo URL and scriptability, which leaves out exactly the
framework that most needs it: an operator-supplied workload has no upstream repo
by construction, so it fell through to querying all of PR_QUERY_REPOS.

A 24h run spent about ten authoring rounds on sglang pull requests that way, and
the orchestration model wrote its own verdict into the session learnings: "any
serve-flag or cross-framework sglang PR is dead on arrival".
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator


class _Stub:
    """Minimal holder; the method under test reads nothing off self."""


def _urls(framework: str) -> list[str]:
    return Coordinator._framework_agent_discover_repo_urls(_Stub(), framework)  # type: ignore[arg-type]


def _has_serving_repos(urls: list[str]) -> bool:
    return any(name in u for u in urls for name in ("sglang", "vllm", "TensorRT-LLM", "nccl"))


def test_operator_supplied_workload_queries_no_serving_repos():
    """``custom`` has no upstream repo and must not inherit the global allowlist."""
    urls = _urls("custom")
    assert not _has_serving_repos(urls), f"serving repos leaked into a custom session: {urls}"


def test_a_scriptable_framework_without_a_repo_map_is_also_scoped():
    """The same hole applies to any scriptable framework with no repo URL."""
    urls = _urls("hunyuan_image3")
    assert not _has_serving_repos(urls), f"serving repos leaked: {urls}"


def test_scriptable_framework_with_a_repo_still_queries_only_its_own():
    """The already-correct case must not regress."""
    urls = _urls("xdit")
    assert any("xDiT" in u for u in urls)
    assert not _has_serving_repos(urls)


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_serving_frameworks_still_get_the_global_allowlist(framework: str):
    """Cross-framework PR intelligence is the point for a serving session."""
    urls = _urls(framework)
    assert _has_serving_repos(urls)
    assert len(urls) > 1, "a serving session should see more than its own repo"
