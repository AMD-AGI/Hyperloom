"""SpecialistRunner._finalize must hard-truncate ``proposal_set`` to the
single-source-of-truth cap (``DEFAULT_SPECIALIST_MAX_PROPOSALS``) before
persisting ``specialist_done.json`` and returning the SpecialistRunResult.

The prompt asks the specialist to self-curate to ≤ N proposals, but the
runtime must never trust LLM output for size limits — anything beyond the
cap is dropped before persist so the on-disk artifact, Coordinator
bookkeeping, Critic review, and explore-grid materialisation all see the
same N≤cap shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.policy import (
    DEFAULT_SPECIALIST_MAX_PROPOSALS,
)
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
    """Build a minimum-viable runner. backend_factory is never called by
    ``_finalize`` (which is downstream of the execute phase), so a trivial
    placeholder satisfies the ``exactly one of`` constructor invariant."""
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
        gap="gap.canonical.truncate-test",
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


def test_finalize_truncates_oversized_proposal_set(tmp_path: Path):
    """5 proposals in → 3 proposals on disk + in result, with the
    ``proposals_truncated_from`` field and audit note set."""
    assert DEFAULT_SPECIALIST_MAX_PROPOSALS == 3  # contract guard

    runner = _make_runner()
    ctx = _make_ctx()
    prep = _make_prep(tmp_path)

    done_payload = {
        "gap_canonical_id": prep.gap,
        "domain": prep.domain.key,
        "proposal_set": [_proposal(i) for i in range(5)],
        "summary": "five proposals",
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

    # In-result payload
    assert result.status == "succeeded"
    assert len(result.specialist_done["proposal_set"]) == 3
    assert result.specialist_done["proposals_truncated_from"] == 5
    assert result.specialist_done["empty"] is False
    # Preserved the first 3 in order
    assert [
        p["name"] for p in result.specialist_done["proposal_set"]
    ] == ["prop-0", "prop-1", "prop-2"]
    # Audit note threaded into the SpecialistRunResult.notes list
    assert "proposal_set_truncated:5->3" in result.notes

    # On-disk artefact matches
    done_path = tmp_path / "specialist_done.json"
    assert done_path.exists()
    on_disk = json.loads(done_path.read_text(encoding="utf-8"))
    assert len(on_disk["proposal_set"]) == 3
    assert on_disk["proposals_truncated_from"] == 5


def test_finalize_no_truncation_at_or_below_cap(tmp_path: Path):
    """3 proposals → no truncation, no ``proposals_truncated_from`` field,
    no audit note. Boundary case: exactly at the cap must NOT be flagged
    as truncated."""
    runner = _make_runner()
    ctx = _make_ctx(task_id="spec-002")
    prep = _make_prep(tmp_path)

    done_payload = {
        "gap_canonical_id": prep.gap,
        "domain": prep.domain.key,
        "proposal_set": [_proposal(i) for i in range(3)],
        "summary": "exactly three",
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

    assert len(result.specialist_done["proposal_set"]) == 3
    assert "proposals_truncated_from" not in result.specialist_done
    assert not any("proposal_set_truncated" in n for n in result.notes)

    on_disk = json.loads(
        (tmp_path / "specialist_done.json").read_text(encoding="utf-8")
    )
    assert "proposals_truncated_from" not in on_disk


def test_finalize_empty_proposal_set_unchanged(tmp_path: Path):
    """Empty proposal_set should pass through untouched, with
    ``empty=True`` set when not provided."""
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
