# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Directed + cross-repo + dedup coverage for the FRAMEWORK discover batch builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.loop.coordinator import (
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
        self.last_profile_kernel_breakdown = None

    def save(self, _session_dir: Path) -> None:
        pass


class _CoordinatorStub:
    """Binds the real repo-url builder and the helpers it reads."""

    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _build_framework_working_memory = Coordinator._build_framework_working_memory
    _FRAMEWORK_TRIED_MEMORY_CAP = Coordinator._FRAMEWORK_TRIED_MEMORY_CAP
    # Reverse-lookup called on every repo the discovery merge queries.

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


def test_repo_urls_cover_global_allowlist_with_framework_primary():
    """The repo set leads with the framework's own repo, includes every PR_QUERY_REPOS entry, de-duplicated and order-preserving."""
    stub = _CoordinatorStub(Path("/tmp"))
    urls = stub._framework_agent_discover_repo_urls("sglang")

    assert urls[0] == _fa_client.repo_url_for_framework("sglang")
    for repo in PR_QUERY_REPOS:
        expected = f"https://github.com/{repo}.git"
        assert expected in urls or repo in "".join(urls)
    assert len(urls) == len(set(urls))


def test_scriptable_repo_urls_skip_serving_allowlist_when_primary_repo_exists():
    """Scriptable model repos are not compatible with serving-framework PR diffs."""
    stub = _CoordinatorStub(Path("/tmp"))
    urls = stub._framework_agent_discover_repo_urls("xdit")

    assert urls == ["https://github.com/xdit-project/xDiT.git"]
