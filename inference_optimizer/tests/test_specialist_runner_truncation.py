# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SpecialistRunner._finalize must carry the specialist's ``proposal_set`` back unmodified (``max_proposals`` is a prompt-side target, not a runtime cap)."""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.specialist_domains import (
    SpecialistDomain,
)
from inference_optimizer.orchestrator.specialist_runner import (
    SpecialistRunner,
    _PreparedRun,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


def _make_runner() -> SpecialistRunner:
    """Build a minimum-viable runner; ``_finalize`` never calls backend_factory."""
    return SpecialistRunner(backend_factory=lambda _domain: None)


def _make_ctx(task_id: str = "spec-001") -> RunnerContext:
    return RunnerContext(
        task=Task(
            task_id=task_id,
            kind="specialist",
            state="leased",
            params={"domain": "serving_specialist"},
            idempotency_key=task_id,
        ),
        lease=None,
    )


def _make_prep(workspace: Path) -> _PreparedRun:
    domain = SpecialistDomain(
        key="serving_specialist",
        layer="sglang",
        kb_anchor="framework",
    )
    return _PreparedRun(
        domain=domain,
        gap="gap.canonical.proposal-set-test",
        max_turns=2,
        workspace=workspace,
    )


def _proposal(idx: int) -> dict:
    """A minimally well-formed proposal record."""
    return {
        "name": f"prop-{idx}",
        "kind": "param_change",
        "summary": f"proposal #{idx}",
        "args": {"foo": idx},
    }


def test_finalize_carries_full_proposal_set(tmp_path: Path):
    """A large proposal_set is carried back unmodified — no truncation, no audit note."""
    runner = _make_runner()
    ctx = _make_ctx()
    prep = _make_prep(tmp_path)

    done_payload = {
        "gap_canonical_id": prep.gap,
        "domain": prep.domain.key,
        "proposal_set": [_proposal(i) for i in range(8)],
        "summary": "eight proposals",
        "reason": "test",
        "confidence": 0.5,
        "new_findings": [],
        "residual_questions": [],
    }

    result = runner._finalize(
        ctx=ctx,
        prep=prep,
        specialist_done_payload=done_payload,
        turns_used=2,
        tool_violations=[],
        backend_error="",
        extra_notes=[],
        patches_written=[],
    )

    assert result.status == "succeeded"
    assert len(result.specialist_done["proposal_set"]) == 8
    assert result.specialist_done["empty"] is False
    assert [
        p["name"] for p in result.specialist_done["proposal_set"]
    ] == [f"prop-{i}" for i in range(8)]
    assert "proposals_truncated_from" not in result.specialist_done
    assert not any("proposal_set_truncated" in n for n in result.notes)

    # On-disk artefact matches.
    done_path = tmp_path / "specialist_done.json"
    assert done_path.exists()
    on_disk = json.loads(done_path.read_text(encoding="utf-8"))
    assert len(on_disk["proposal_set"]) == 8
    assert "proposals_truncated_from" not in on_disk


def test_finalize_empty_proposal_set_unchanged(tmp_path: Path):
    """Empty proposal_set passes through untouched, with ``empty=True`` set when not provided."""
    runner = _make_runner()
    ctx = _make_ctx(task_id="spec-003")
    prep = _make_prep(tmp_path)

    done_payload = {
        "gap_canonical_id": prep.gap,
        "domain": prep.domain.key,
        "proposal_set": [],
        "summary": "no findings",
        "reason": "test",
        "confidence": 0.0,
        "new_findings": [],
        "residual_questions": [],
    }

    result = runner._finalize(
        ctx=ctx,
        prep=prep,
        specialist_done_payload=done_payload,
        turns_used=1,
        tool_violations=[],
        backend_error="",
        extra_notes=[],
        patches_written=[],
    )

    assert result.specialist_done["proposal_set"] == []
    assert result.specialist_done["empty"] is True
    assert "proposals_truncated_from" not in result.specialist_done
