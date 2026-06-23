from __future__ import annotations

import json
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_DIR = _REPO_ROOT / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

if "requests" not in sys.modules:
    try:
        import requests as _requests  # noqa: E402,F401
    except ModuleNotFoundError:
        requests_stub = types.ModuleType("requests")
        requests_stub.HTTPError = Exception
        requests_stub.exceptions = types.SimpleNamespace(RequestException=Exception)
        requests_stub.utils = types.SimpleNamespace(quote=lambda path, safe="": path)
        sys.modules["requests"] = requests_stub

import optimize_submit as opt  # noqa: E402


def _record() -> opt.SubmissionRecord:
    return opt.SubmissionRecord(
        model="Qwen/Qwen3.6-35B-A3B-Instruct",
        task_id="task-123",
        display_name="qwen36",
        model_path="/wekafs/models/Qwen3.6-35B-A3B-Instruct",
        safe_user_id="user-1",
        safe_started_at="2026-06-10T01:00:00Z",
        safe_finished_at="2026-06-10T01:05:00Z",
        final_status="Failed",
    )


def _session(tmp_path: Path, rec: opt.SubmissionRecord) -> Path:
    session = tmp_path / "users" / rec.safe_user_id / "Qwen3.6-35B-A3B-Instruct" / "20260610T010200Z"
    session.mkdir(parents=True)
    return session


def test_find_nfs_state_session_dir_matches_state_only_session(tmp_path, monkeypatch):
    rec = _record()
    session = _session(tmp_path, rec)
    (session / "state.json").write_text(
        json.dumps({"phase": "EXPLORE", "model_name": "Qwen3.6-35B-A3B-Instruct"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    assert opt._find_nfs_state_session_dir(rec) == str(session)


def test_find_nfs_state_session_dir_rejects_mismatched_state_model(tmp_path, monkeypatch):
    rec = _record()
    session = _session(tmp_path, rec)
    (session / "state.json").write_text(
        json.dumps({"phase": "EXPLORE", "model_name": "Other-Model"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    assert opt._find_nfs_state_session_dir(rec) is None


def test_reports_final_md_does_not_end_nfs_liveness_wait(tmp_path):
    rec = _record()
    session = _session(tmp_path, rec)
    (session / "state.json").write_text(
        json.dumps({"phase": "CLOSE", "model_name": "Qwen3.6-35B-A3B-Instruct"}),
        encoding="utf-8",
    )
    reports = session / "reports"
    reports.mkdir()
    (reports / "final.md").write_text("final report before breakdown", encoding="utf-8")

    assert opt._session_has_terminal_marker(session) is False

    (session / "session_breakdown.json").write_text("{}", encoding="utf-8")
    assert opt._session_has_terminal_marker(session) is True


def test_wait_for_nfs_session_delivery_waits_until_terminal_marker(
    tmp_path,
    monkeypatch,
):
    rec = _record()
    session = _session(tmp_path, rec)
    (session / "state.json").write_text(
        json.dumps({"phase": "EXPLORE", "model_name": "Qwen3.6-35B-A3B-Instruct"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    def mark_complete(_seconds):
        (session / "complete").write_text("", encoding="utf-8")

    monkeypatch.setattr(opt.time, "sleep", mark_complete)

    assert opt._wait_for_nfs_session_delivery(
        rec,
        poll_s=1,
        grace_min=1,
        idle_min=1,
    ) == str(session)
    assert (session / "complete").is_file()


def test_succeeded_task_waits_for_nfs_when_safe_breakdown_lags(
    tmp_path,
    monkeypatch,
):
    rec = _record()
    rec.final_status = None
    artifacts_dir = tmp_path / "artifacts"
    calls = {"waited": 0, "listed": 0}

    class _Safe:
        def wait_task_done(self, task_id, *, timeout_min, poll_s):
            return "Succeeded", {
                "currentPhase": 99,
                "message": "done",
                "clawSessionId": "claw-1",
                "modelPath": rec.model_path,
                "userId": rec.safe_user_id,
                "startedAt": rec.safe_started_at,
                "finishedAt": rec.safe_finished_at,
            }

        def list_artifacts(self, task_id):
            calls["listed"] += 1
            if calls["listed"] <= 3:
                return [{"path": "reports/final.md"}]
            return [{"path": "session_breakdown.json"}]

        def download_artifact_to(self, task_id, item, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_text("{}", encoding="utf-8")
            return 2

    def fake_wait_for_nfs(*args, **kwargs):
        calls["waited"] += 1
        return "/nfs/session"

    monkeypatch.setattr(opt, "_wait_for_nfs_session_delivery", fake_wait_for_nfs)
    # The artifact-retry loop in wait_and_collect_one sleeps 15s between the
    # first 3 listings (session_breakdown.json lags); stub it so the test does
    # not spend ~30-45s of real wall-clock in CI.
    monkeypatch.setattr(opt.time, "sleep", lambda *_a, **_k: None)

    out = opt.wait_and_collect_one(
        _Safe(),
        rec,
        artifacts_dir,
        task_timeout_min=1,
        poll_s=1,
        collect=True,
        all_artifacts=False,
    )

    assert out.final_status == "Succeeded"
    assert calls["waited"] == 1
    assert calls["listed"] == 4
    assert out.ci_success is True
    assert any(p.endswith("session_breakdown.json") for p in out.artifact_files)


def test_nfs_fallback_synthesizes_breakdown_for_state_only_session(tmp_path, monkeypatch):
    rec = _record()
    session = _session(tmp_path, rec)
    (session / "state.json").write_text(
        json.dumps(
            {
                "phase": "EXPLORE",
                "model_name": "Qwen3.6-35B-A3B-Instruct",
                "baseline_tput": 0.0,
                "cumulative_gain_validated": 0.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    artifacts_dir = tmp_path / "artifacts"
    added = opt._nfs_fallback_collect(rec, artifacts_dir)

    task_dir = artifacts_dir / rec.task_id
    breakdown = task_dir / "session_breakdown.json"
    report = task_dir / "optimization_report.md"
    assert added == 3
    assert breakdown.is_file()
    assert report.is_file()
    data = json.loads(breakdown.read_text(encoding="utf-8"))
    assert data["ci_emergency_artifact"] is True
    assert data["session"]["stop_reason"] == "incomplete_after_progress"
    assert data["final"]["cumulative_gain_pct_validated"] == 0.0

    opt._mark_record_delivery(rec)
    assert rec.ci_success is True
    assert rec.ci_status == "Delivered"
    assert any(src["source_type"] == "nfs_user_session_ci_emergency" for src in rec.artifact_sources)


def test_ci_emergency_artifacts_classify_bf16_abort_marker(tmp_path):
    rec = _record()
    session = tmp_path / "users" / rec.safe_user_id / "Qwen3.6-35B-A3B-Instruct" / "ABORTED_bf16_misconfig"
    session.mkdir(parents=True)
    (session / "state.json").write_text(
        json.dumps({"model_name": "Qwen3.6-35B-A3B-Instruct", "phase": "PRELUDE"}),
        encoding="utf-8",
    )

    task_dir = tmp_path / "artifacts" / rec.task_id
    added = opt._write_ci_emergency_artifacts(rec, session, task_dir)

    assert added == 3
    data = json.loads((task_dir / "session_breakdown.json").read_text(encoding="utf-8"))
    assert data["session"]["stop_reason"] == "model_config_incompatible"
    assert "BF16" in data["ci_emergency_reason"]


def test_default_artifact_filter_keeps_claw_lifecycle_jsonl():
    assert opt._is_wanted_artifact("claw-1782177641523.jsonl", all_artifacts=False) is True


def test_claw_emergency_artifacts_classify_sandbox_poll_timeout(tmp_path):
    rec = _record()
    rec.claw_session_id = "03391c9e-5dae-4d56-bdd7-4b5b697001fa"
    rec.final_message = "optimization report not found; skill may have exited early"
    task_dir = tmp_path / "artifacts" / rec.task_id
    task_dir.mkdir(parents=True)
    claw = task_dir / "claw-1782177641523.jsonl"
    claw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sessionId": rec.claw_session_id,
                        "messageId": "claw-1782177641523",
                        "eventCount": 8,
                        "failed": True,
                        "error": "Hands workload primus-claw-20260623012041-mwzck poll timeout after 3603.6s",
                        "turns": 0,
                        "elapsedMs": 3603682,
                        "failureReason": "sandbox_poll_timeout",
                    }
                ),
                json.dumps({"type": "sandboxStatus", "status": "pending", "queuePosition": 4}),
                json.dumps(
                    {
                        "type": "sandboxStatus",
                        "status": "failed",
                        "reason": "sandbox_poll_timeout",
                        "message": "Hands workload primus-claw-20260623012041-mwzck poll timeout after 3603.6s",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    rec.artifact_files.append(str(claw))
    rec.artifact_count = 1

    added = opt._write_claw_emergency_artifacts(rec, task_dir)

    assert added == 2
    data = json.loads((task_dir / "session_breakdown.json").read_text(encoding="utf-8"))
    assert data["session"]["stop_reason"] == "sandbox_start_failed"
    assert data["session"]["failure_reason"] == "sandbox_poll_timeout"
    assert data["session"]["turns"] == 0
    assert data["final"]["cumulative_gain_pct_validated"] == 0.0
    opt._mark_record_delivery(rec)
    assert rec.ci_success is True
    assert rec.ci_status == "Delivered"
    assert any(src["source_type"] == "safe_artifact_api_claw_ci_emergency" for src in rec.artifact_sources)


def test_claw_emergency_artifacts_classify_missing_breakdown_after_activity(tmp_path):
    rec = _record()
    rec.claw_session_id = "965e3cfa-1b2c-4b8b-b7c6-4a9b51a4ec1e"
    rec.final_message = "optimization report not found; skill may have exited early"
    task_dir = tmp_path / "artifacts" / rec.task_id
    task_dir.mkdir(parents=True)
    claw = task_dir / "claw-1782176365012.jsonl"
    claw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sessionId": rec.claw_session_id,
                        "messageId": "claw-1782176365012",
                        "eventCount": 300,
                        "failed": False,
                        "error": "recovered 1 files from checkpoint (failed=0)",
                        "turns": 105,
                        "elapsedMs": 14000000,
                        "failureReason": "from_inflight_checkpoint",
                    }
                ),
                json.dumps({"type": "sandboxStatus", "status": "workspace_sync_failed"}),
                json.dumps(
                    {
                        "type": "sandboxStatus",
                        "status": "workspace_sync_recovered",
                        "reason": "from_inflight_checkpoint",
                        "message": "recovered 1 files from checkpoint (failed=0)",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    rec.artifact_files.append(str(claw))
    rec.artifact_count = 1

    added = opt._write_claw_emergency_artifacts(rec, task_dir)

    assert added == 2
    data = json.loads((task_dir / "session_breakdown.json").read_text(encoding="utf-8"))
    assert data["session"]["stop_reason"] == "missing_breakdown_after_claw_activity"
    assert data["session"]["failure_reason"] == "from_inflight_checkpoint"
    assert data["session"]["turns"] == 105
