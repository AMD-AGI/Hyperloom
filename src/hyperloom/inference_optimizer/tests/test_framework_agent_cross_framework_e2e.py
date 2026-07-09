# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end contract test for the cross-framework path.

Code review flagged that the three seams were only tested in isolation:
discovery-tagging, tagging->audit detection, and audit->specialist prompt each
had a unit test, but nothing traced a single cross-repo candidate through all
of them together — so a field rename in any one seam
(``candidate["framework"]`` <-> ``audit["layer"]`` <-> the specialist
``params`` keys) would silently break the pipeline without failing a test.

This module runs the REAL coordinator methods for all three seams in sequence
(only the fa I/O boundaries — ``phase_discover``/``phase_audit`` — and the task
dispatch helpers are stubbed) and asserts the cross-framework signal survives
end to end. A same-framework control locks the "no behaviour change when not
cross-framework" contract.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.loop.coordinator import Coordinator


class _StateStub:
    def __init__(self, framework: str) -> None:
        self.phase = "FRAMEWORK"
        self.framework = framework
        self.model = "test-model"
        self.gpu_type = "MI325X"
        self.model_class = "dense"
        self.precision = "fp8"
        self.baseline_tput = 0.0
        self.gaps: list[dict[str, Any]] = []
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_discover_failures = 0
        self.framework_agent_specialist_candidate_map: dict[str, str] = {}
        self.phase_history: list[dict[str, Any]] = []
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _TasksStub:
    """Captures the specialist ``params`` the authoring dispatch builds."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create_or_return_existing(self, **kwargs: Any) -> tuple[Any, bool]:
        self.created.append(kwargs)
        task = SimpleNamespace(kind=kwargs.get("kind"), task_id=f"t-{len(self.created)}", state="")
        # existing=False so the cross-resume livelock branch is skipped.
        return task, False


class _CrossFwCoordStub:
    """Glue stub that binds the REAL methods under test for all three seams."""

    # Seam 1 — discover + tag (the real discovery-merge tagging block).
    _discover_next_framework_batch = Coordinator._discover_next_framework_batch
    _framework_agent_repo_url_origin_framework = staticmethod(
        Coordinator._framework_agent_repo_url_origin_framework
    )
    _framework_known_candidate_ids = Coordinator._framework_known_candidate_ids
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    # Seam 2 — audit cross-framework detection.
    _audit_framework_agent_candidate = Coordinator._audit_framework_agent_candidate
    # Seam 3 — specialist authoring dispatch.
    _enqueue_framework_agent_authoring_specialist = Coordinator._enqueue_framework_agent_authoring_specialist
    _framework_agent_audit_seed_lines = staticmethod(Coordinator._framework_agent_audit_seed_lines)

    def __init__(self, tmp_path: Path, *, session_framework: str, discover_repo_framework: str) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub(session_framework)
        self.tasks = _TasksStub()
        self.framework_agent_discover_timeout_sec = 0.0
        # The repo the discovery batch is queried against (drives cross-fw origin).
        self._discover_repo_url = _fa_client.repo_url_for_framework(discover_repo_framework)

    # --- overrides isolating fa/network/GPU boundaries ---
    def _framework_agent_discover_repo_urls(self, _framework: str) -> list[str]:
        return [self._discover_repo_url]

    def _build_framework_working_memory(self) -> dict[str, Any]:
        return {}

    async def _warm_specialist_params(self, _params: dict[str, Any]) -> None:
        return None

    def _framework_gpu_params(self) -> dict[str, Any]:
        return {}

    def _framework_authoring_lanes_ttl(self, _params: dict[str, Any], *, base_ttl_sec: int) -> tuple[list[str], int]:
        return [], base_ttl_sec


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _only_candidate(stub: _CrossFwCoordStub) -> dict[str, Any]:
    batches = stub.shared_state.framework_agent_batches
    assert len(batches) == 1, batches
    cands = batches[0]["candidates"]
    assert len(cands) == 1, cands
    return cands[0]


def test_cross_framework_candidate_flows_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """sglang session discovers a vllm-repo PR -> tagged -> audited cross-fw -> specialist params carry the port.

    Uses the #5-P2 DEFAULT (no env opt-in) so the whole pipeline is exercised as
    it runs in production.
    """
    monkeypatch.delenv("FRAMEWORK_AGENT_CROSS_DISCOVER_TAG", raising=False)

    # fa phase-discover returns a candidate WITHOUT a framework tag (the common
    # shape — fa never stamps origin framework).
    async def _discover(**_: Any) -> dict[str, Any]:
        return {
            "batch_id": "batch-xfw",
            "candidates": [
                {
                    "pr_url": "https://github.com/ROCm/vllm/pull/42",
                    "diff_url": "https://github.com/ROCm/vllm/pull/42.diff",
                    "repo": "ROCm/vllm",
                    "ref": "perf/paged-attn",
                    "title": "perf: paged attention prefill",
                }
            ],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)

    # Capture what the audit forwards to fa phase-audit and hand back a
    # cross_framework verdict (as cross_framework_audit would).
    audit_kwargs: dict[str, Any] = {}

    async def _audit(**kwargs: Any) -> dict[str, Any]:
        audit_kwargs.update(kwargs)
        return {
            "layer": "cross_framework",
            "semantic_status": "applicable",
            "applicability": "port_required",
            "recommended_next_step": "author",
            "confidence": 0.8,
            "evidence": [],
            "risks": [],
            "metrics": {"src_framework": "vllm", "dst_framework": "sglang"},
        }

    monkeypatch.setattr(_fa_client, "phase_audit", _audit)

    stub = _CrossFwCoordStub(tmp_path, session_framework="sglang", discover_repo_framework="vllm")

    # --- Seam 1: discover -> tag ---
    ok = _run(Coordinator._discover_next_framework_batch(stub))  # type: ignore[arg-type]
    assert ok is True
    candidate = _only_candidate(stub)
    assert candidate["framework"] == "vllm", "cross-repo candidate must be tagged with its origin framework"

    # --- Seam 2: tag -> audit cross-framework detection ---
    verdict = _run(Coordinator._audit_framework_agent_candidate(stub, candidate))  # type: ignore[arg-type]
    assert verdict["layer"] == "cross_framework"
    # The audit must recognise the framework mismatch and request a PORT into
    # the session framework: candidate's own framework as source, session
    # framework as target.
    assert audit_kwargs.get("framework") == "vllm"
    assert audit_kwargs.get("target_framework") == "sglang"
    assert audit_kwargs.get("target_framework_source_roots") is not None

    # --- Seam 3: audit -> specialist authoring params ---
    _run(
        Coordinator._enqueue_framework_agent_authoring_specialist(  # type: ignore[arg-type]
            stub, candidate, verdict
        )
    )
    assert len(stub.tasks.created) == 1
    params = stub.tasks.created[0]["params"]
    assert params["cross_framework"] is True
    assert params["source_framework"] == "vllm"
    assert params["target_framework"] == "sglang"


def test_same_framework_candidate_is_not_cross_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Control: a sglang session discovering from the sglang repo stays same-framework across all seams (default-on lane)."""
    monkeypatch.delenv("FRAMEWORK_AGENT_CROSS_DISCOVER_TAG", raising=False)

    async def _discover(**_: Any) -> dict[str, Any]:
        return {
            "batch_id": "batch-same",
            "candidates": [
                {
                    "pr_url": "https://github.com/sgl-project/sglang/pull/7",
                    "diff_url": "https://github.com/sgl-project/sglang/pull/7.diff",
                    "repo": "sgl-project/sglang",
                    "ref": "perf/moe",
                    "title": "perf: moe gemm fastpath",
                }
            ],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)

    audit_kwargs: dict[str, Any] = {}

    async def _audit(**kwargs: Any) -> dict[str, Any]:
        audit_kwargs.update(kwargs)
        return {
            "layer": "same_framework",
            "semantic_status": "applicable",
            "applicability": "apply",
            "recommended_next_step": "author",
            "confidence": 0.8,
            "evidence": [],
            "risks": [],
        }

    monkeypatch.setattr(_fa_client, "phase_audit", _audit)

    stub = _CrossFwCoordStub(tmp_path, session_framework="sglang", discover_repo_framework="sglang")

    ok = _run(Coordinator._discover_next_framework_batch(stub))  # type: ignore[arg-type]
    assert ok is True
    candidate = _only_candidate(stub)
    # Same-framework repo -> origin == session framework -> NOT stamped.
    assert not candidate.get("framework")

    verdict = _run(Coordinator._audit_framework_agent_candidate(stub, candidate))  # type: ignore[arg-type]
    # No cross-framework port requested.
    assert audit_kwargs.get("target_framework") == ""
    assert audit_kwargs.get("target_framework_source_roots") is None

    _run(
        Coordinator._enqueue_framework_agent_authoring_specialist(  # type: ignore[arg-type]
            stub, candidate, verdict
        )
    )
    assert len(stub.tasks.created) == 1
    params = stub.tasks.created[0]["params"]
    assert "cross_framework" not in params
