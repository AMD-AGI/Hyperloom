"""Unit tests for the L1 + L2 postmortem finalizer."""

from __future__ import annotations

import json
from pathlib import Path


from hyperloom.agents.robustness.finalize.postmortem import (
    PostmortemFinalizer,
    PostmortemFinalizerConfig,
    finalize_session,
)


def _write_finding(path: Path, **overrides) -> None:
    base = dict(
        tick_index=overrides.pop("tick_index", 1),
        timestamp_unix=overrides.pop("timestamp_unix", 1.0),
        symptom_name=overrides.pop("symptom_name", "x"),
        severity=overrides.pop("severity", "low"),
        summary=overrides.pop("summary", "s"),
        intents=overrides.pop("intents", []),
        evidence=overrides.pop("evidence", {}),
        rca_text=overrides.pop("rca_text", ""),
    )
    base.update(overrides)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(base) + "\n")


def _make_run_result(
    session_dir: Path,
    action: str,
    task_id: str,
    payload: dict,
) -> Path:
    task_dir = session_dir / "runs" / action / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return task_dir


def test_finalize_creates_postmortem_and_decision_trace(tmp_path: Path) -> None:
    findings_dir = tmp_path / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    findings_path = findings_dir / "sess-1.jsonl"
    _write_finding(
        findings_path,
        tick_index=2,
        symptom_name="gpu_leak_persistent",
        severity="high",
        summary="ROCm KFD leak repeated",
        evidence={"used_mb": 71000},
        rca_text="reboot mitigated last 3 times",
        intents=[{
            "intent_type": "alert",
            "payload": {"severity": "high"},
        }],
    )
    _write_finding(
        findings_path,
        tick_index=5,
        symptom_name="quota_low_hit",
        severity="medium",
    )

    _make_run_result(
        tmp_path, "integrate", "kernel_42",
        {
            "decision": "KEEP",
            "gain_pct": 12.5,
            "kernel_id": "fa_v2",
        },
    )
    _make_run_result(
        tmp_path, "integrate", "kernel_99",
        {
            "decision": "REVERT",
            "gain_pct": -3.0,
            "error_class": "regression",
        },
    )
    _make_run_result(
        tmp_path, "baseline", "task_0",
        {
            "decision": None,
            "output_throughput": 123.4,
            "status": "ok",
        },
    )

    finalizer = PostmortemFinalizer(
        session_dir=tmp_path, session_id="sess-1",
    )
    assert finalizer.finalize(stop_reason="budget_exhausted") is True
    assert finalizer.is_finalized() is True

    md_path = tmp_path / "reports" / "robustness_postmortem.md"
    trace_path = tmp_path / "reports" / "decision_trace.json"
    assert md_path.is_file()
    assert trace_path.is_file()

    md = md_path.read_text(encoding="utf-8")
    assert "Robustness postmortem" in md
    assert "budget_exhausted" in md
    assert "gpu_leak_persistent" in md
    assert "HIGH=1" in md
    # Decision-trace summary table renders per-action counts.
    assert "`integrate`" in md
    assert "`baseline`" in md

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["total_tasks"] == 3
    integrate = trace["tasks_by_action"]["integrate"]
    assert len(integrate) == 2
    decisions = {t["decision"] for t in integrate}
    assert decisions == {"KEEP", "REVERT"}
    assert any(t.get("gain_pct") == 12.5 for t in integrate)


def test_finalize_idempotent_via_marker(tmp_path: Path) -> None:
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path, session_id="sess-x",
    )
    assert finalizer.finalize(stop_reason="r1") is True
    # Second invocation must not overwrite.
    md_path = tmp_path / "reports" / "robustness_postmortem.md"
    first_body = md_path.read_text(encoding="utf-8")
    md_path.write_text("MUTATED\n", encoding="utf-8")
    assert finalizer.finalize(stop_reason="r2") is False
    assert md_path.read_text(encoding="utf-8") == "MUTATED\n"
    assert first_body  # sanity: first body was non-empty


def test_finalize_no_findings_no_runs(tmp_path: Path) -> None:
    assert finalize_session(
        tmp_path, session_id="empty", stop_reason="manual_close",
    ) is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "No HIGH-severity finding" in md
    assert "No ``runs/" in md or "No \"runs/" in md or "result.json" in md
    trace = json.loads(
        (tmp_path / "reports" / "decision_trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert trace["total_tasks"] == 0
    assert trace["tasks_by_action"] == {}


def test_finalize_skips_malformed_jsonl_lines(tmp_path: Path) -> None:
    findings_dir = tmp_path / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    findings_path = findings_dir / "sess-1.jsonl"
    findings_path.write_text(
        "{not json}\n"
        + json.dumps({
            "tick_index": 1, "timestamp_unix": 1.0,
            "symptom_name": "x", "severity": "high",
            "summary": "s", "intents": [], "evidence": {},
            "rca_text": "",
        }) + "\n"
        + "\n",
        encoding="utf-8",
    )
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path, session_id="sess-1",
    )
    assert finalizer.finalize(stop_reason="r") is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "HIGH=1" in md


def test_finalize_picks_first_high_severity_as_flashpoint(
    tmp_path: Path,
) -> None:
    findings_dir = tmp_path / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    findings_path = findings_dir / "sess-1.jsonl"
    _write_finding(
        findings_path,
        tick_index=10,
        symptom_name="late_high",
        severity="high",
        summary="later high",
    )
    _write_finding(
        findings_path,
        tick_index=3,
        symptom_name="early_high",
        severity="high",
        summary="earlier high",
    )
    _write_finding(
        findings_path,
        tick_index=1,
        severity="medium",
    )
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path, session_id="sess-1",
    )
    assert finalizer.finalize(stop_reason="r") is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "early_high" in md
    # Flashpoint section appears before catalogue.
    flash_pos = md.index("Flashpoint")
    catalogue_pos = md.index("Findings catalogue")
    assert flash_pos < catalogue_pos


def test_finalize_caps_tasks_per_action(tmp_path: Path) -> None:
    import os
    for i in range(8):
        task_dir = _make_run_result(
            tmp_path, "sweep", f"task_{i:03d}",
            {"decision": "KEEP" if i % 2 == 0 else "REVERT",
             "gain_pct": float(i)},
        )
        # Deterministic mtime: i=7 newest, i=0 oldest.
        ts = 1_700_000_000 + i * 10
        os.utime(task_dir, (ts, ts))
    cfg = PostmortemFinalizerConfig(max_tasks_per_action=3)
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path, session_id="sess-1", config=cfg,
    )
    assert finalizer.finalize(stop_reason="r") is True
    trace = json.loads(
        (tmp_path / "reports" / "decision_trace.json").read_text(
            encoding="utf-8"
        )
    )
    sweep_tasks = trace["tasks_by_action"]["sweep"]
    assert len(sweep_tasks) == 3
    task_ids = [t["task_id"] for t in sweep_tasks]
    assert task_ids == ["task_007", "task_006", "task_005"]
