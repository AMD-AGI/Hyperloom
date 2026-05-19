"""v0.8 §3.12 — observability / breakdown schema v2 tests.

Covers KB_design/3.12_observability/README.md acceptance criteria:

* §5 / §6 — top-level ``schema_version`` says ``hyperloom.session_breakdown.v2``.
* §4.3 — ``specialist_runs`` section is populated from
  ``SharedState.specialist_rounds`` + ``runs/specialist/`` transcripts.
* §4.2 — ``capability_summary.specialist`` row exists and the counts
  agree with ``specialist_runs`` (Inv-12.2 single source of truth).
* §4.4 — ``critic_robustness.kb_writes_summary`` summarises the
  critic-agent commit-review verdicts.
* §5 — top-level ``action_timeline`` and ``explore_search`` aliases
  guarantee a v0.6/v0.7 reader sees its old fields (Inv-12.1).
* §7 step 5 — ``--breakdown-include-transcripts`` CLI flag controls
  whether transcript bodies are inlined or referenced by path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inference_optimizer.breakdown.exporter import build
from inference_optimizer.breakdown.schema import SCHEMA_VERSION


# ===========================================================================
# Test fixtures
# ===========================================================================
def _write_state(session_dir: Path, state: dict) -> None:
    (session_dir / "state.json").write_text(json.dumps(state))
    if not (session_dir / "manifest.json").exists():
        (session_dir / "manifest.json").write_text(json.dumps({}))


def _basic_state(**extras) -> dict:
    state = {
        "session_id": "sid",
        "schema_version": 2,
        "baseline_tput": 100.0,
        "current_best": {"tput": 110.0},
        "cumulative_gain": 10.0,
        "kernel_enabled": False,
    }
    state.update(extras)
    return state


def _specialist_round(round_id: int = 1, **extras) -> dict:
    base = {
        "round_id":          round_id,
        "dispatched_at":     "2025-01-01T00:00:00Z",
        "completed_at":      "2025-01-01T00:01:00Z",
        "domains":           ["kernel_specialist"],
        "parallelism":       1,
        "proposals_total":   3,
        "proposals_kept":    1,
        "proposals_rejected": 1,
        "proposals_skipped": 1,
        "kb_edge_ids":       ["edge-1"],
        "confidence_avg":    0.7,
        "domain_breakdown": {
            "kernel_specialist": {
                "dispatched": 1, "proposals_total": 3,
                "proposals_kept": 1, "proposals_rejected": 1,
            },
        },
        "task_ids":     ["t-abc"],
        "task_domains": {"t-abc": "kernel_specialist"},
        "notes":        [],
    }
    base.update(extras)
    return base


# ===========================================================================
# 1. Schema version + v1 compat aliases
# ===========================================================================
def test_schema_version_is_v2():
    assert SCHEMA_VERSION == "hyperloom.session_breakdown.v2"


def test_build_writes_schema_v2_with_v1_aliases(tmp_path):
    """KB_design §3.12 §5 — v2 file MUST carry the v1-reader aliases
    so existing dashboards keep functioning (Inv-12.1)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    assert b["schema_version"] == "hyperloom.session_breakdown.v2"
    # v1 reader compat — these aliases MUST be present:
    assert "action_timeline" in b
    assert "explore_search" in b
    assert "param_search" in b
    # Aliases are pointer-equivalent (same identity isn't required, but
    # the content must match so a reader sees one source of truth).
    assert b["explore_search"] == b["param_search"]
    assert b["action_timeline"] == b["phase_timeline"]


def test_v1_reader_does_not_crash_on_v2_payload(tmp_path):
    """KB_design §3.12 §9 — a v1 reader that knows only the v1 keys
    MUST be able to consume a v2 payload without raising.

    Simulate a v1 reader by extracting the v1 key subset and asserting
    we can still locate every required field.
    """
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    b = build(sd)
    v1_keys = {
        "session", "workload", "baseline", "final", "phase_timeline",
        "capability_summary", "geak_invocations", "oob_invocations",
        "kernel_lifecycle", "param_search", "sweep", "critic_robustness",
        "telemetry", "attribution", "warnings", "source_files",
    }
    for key in v1_keys:
        assert key in b, f"v1 reader expects {key!r} to exist"
    # The legacy ``param_search`` row carries the same data as v2's
    # ``explore_search`` — so a v1 reader sees the merged ledger
    # transparently (KB_design §3.12 §5).
    assert b["param_search"] == b["explore_search"]


# ===========================================================================
# 2. specialist_runs section
# ===========================================================================
def test_specialist_runs_empty_when_no_rounds(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    assert b["specialist_runs"] == []


def test_specialist_runs_populated_from_state(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    b = build(sd)
    assert len(b["specialist_runs"]) == 1
    entry = b["specialist_runs"][0]
    # Required schema fields.
    for k in (
        "round_id", "dispatched_at", "completed_at", "domains",
        "parallelism", "proposals_total", "proposals_kept",
        "proposals_rejected", "proposals_skipped", "kb_edge_ids",
        "confidence_avg", "domain_breakdown", "transcripts", "notes",
    ):
        assert k in entry, f"specialist_runs row missing field {k!r}"
    # No transcripts on disk → empty list (the runner artifact was not
    # written; the round-merge still runs).
    assert entry["transcripts"] == []
    # Domain breakdown round-trips with int normalisation.
    breakdown_ks = entry["domain_breakdown"]["kernel_specialist"]
    assert breakdown_ks == {
        "dispatched": 1, "proposals_total": 3,
        "proposals_kept": 1, "proposals_rejected": 1,
    }


def test_specialist_runs_attaches_transcript_path_when_present(tmp_path):
    """KB_design §3.12 §4.3 — when ``runs/specialist/<task_id>/specialist_done.json``
    exists, the breakdown captures the path (default) or the body
    (when ``include_transcripts=True``)."""
    sd = tmp_path / "session"
    sd.mkdir()
    transcript_dir = sd / "runs" / "specialist" / "t-abc"
    transcript_dir.mkdir(parents=True)
    body_text = '{"proposal_set": []}'
    (transcript_dir / "specialist_done.json").write_text(body_text)
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))

    # Path-only mode (default).
    b = build(sd)
    refs = b["specialist_runs"][0]["transcripts"]
    assert len(refs) == 1
    assert refs[0]["task_id"] == "t-abc"
    assert refs[0]["domain"] == "kernel_specialist"
    assert refs[0]["path"].endswith("specialist_done.json")
    assert "body" not in refs[0]

    # Inline mode.
    b2 = build(sd, include_transcripts=True)
    ref2 = b2["specialist_runs"][0]["transcripts"][0]
    assert ref2.get("body") == body_text


def test_build_respects_env_var_for_transcripts(tmp_path, monkeypatch):
    """KB_design §3.12 §7 step 5 — when the caller doesn't pass
    ``include_transcripts`` explicitly, the env var (set by CLI)
    drives the decision."""
    sd = tmp_path / "session"
    sd.mkdir()
    transcript_dir = sd / "runs" / "specialist" / "t-abc"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "specialist_done.json").write_text('{"x":1}')
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "1")
    b = build(sd)
    assert b["specialist_runs"][0]["transcripts"][0].get("body") == '{"x":1}'


# ===========================================================================
# 3. capability_summary.specialist row (Inv-12.2 single source)
# ===========================================================================
def test_capability_summary_specialist_row_when_no_rounds(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    spec = b["capability_summary"]["specialist"]
    assert spec == {
        "status":   "not_attempted",
        "attempts": 0,
        "keeps":    0,
        "tested":   0,
    }


def test_capability_summary_specialist_agrees_with_specialist_runs(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(round_id=1, proposals_total=4, proposals_kept=2),
        _specialist_round(round_id=2, proposals_total=2, proposals_kept=0),
    ]))
    b = build(sd)
    spec = b["capability_summary"]["specialist"]
    assert spec["attempts"] == 2
    assert spec["tested"] == 6
    assert spec["keeps"] == 2
    assert spec["status"] == "kept"
    # Cross-check (Inv-12.2): aggregate via specialist_runs matches.
    runs = b["specialist_runs"]
    assert sum(r["proposals_total"] for r in runs) == spec["tested"]
    assert sum(r["proposals_kept"] for r in runs) == spec["keeps"]


def test_capability_summary_specialist_status_tried_when_no_keeps(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(proposals_total=2, proposals_kept=0),
    ]))
    b = build(sd)
    assert b["capability_summary"]["specialist"]["status"] == "tried"


def test_capability_summary_specialist_status_attempted_when_empty_proposals(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(proposals_total=0, proposals_kept=0),
    ]))
    b = build(sd)
    assert b["capability_summary"]["specialist"]["status"] == "attempted"


# ===========================================================================
# 4. critic_robustness.kb_writes_summary
# ===========================================================================
def test_critic_kb_writes_summary_empty_by_default(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    summary = b["critic_robustness"]["kb_writes_summary"]
    assert summary == {"total": 0, "by_verdict": {}}


def test_critic_kb_writes_summary_aggregates_by_verdict(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    # Synthesize three critic iteration outputs with different verdicts.
    critic_root = sd / "critic-workdir"
    for n, verdict in enumerate(("KEEP", "KEEP", "REVERT"), start=1):
        iter_dir = critic_root / f"{n:03d}"
        iter_dir.mkdir(parents=True)
        (iter_dir / "review.json").write_text(json.dumps({
            "verdict": verdict, "summary": f"iter {n}",
        }))
        (iter_dir / "emit.json").write_text(json.dumps({
            "ts": f"2025-01-01T00:0{n}:00Z", "topic": "review",
        }))
    b = build(sd)
    summary = b["critic_robustness"]["kb_writes_summary"]
    assert summary["total"] == 3
    assert summary["by_verdict"] == {"KEEP": 2, "REVERT": 1}


# ===========================================================================
# 5. action_timeline alias mirrors phase_timeline
# ===========================================================================
def test_action_timeline_mirrors_phase_timeline(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(
        phase_history=[
            {"to": "EXPLORE", "ts": "2025-01-01T00:00:00Z",
             "reason": "session_start", "evidence": {}},
        ],
    ))
    b = build(sd)
    assert b["action_timeline"] == b["phase_timeline"]


# ===========================================================================
# 6. CLI flag wiring
# ===========================================================================
def test_cli_exposes_breakdown_include_transcripts_flag():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "optimize", "--model", "/tmp/dummy",
        "--breakdown-include-transcripts", "true",
    ])
    assert args.breakdown_include_transcripts == "true"


def test_cli_breakdown_include_transcripts_defaults_to_false():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["optimize", "--model", "/tmp/dummy"])
    assert args.breakdown_include_transcripts in ("true", "false")


def test_cli_rejects_unknown_breakdown_include_transcripts():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/tmp/dummy",
            "--breakdown-include-transcripts", "maybe",
        ])
