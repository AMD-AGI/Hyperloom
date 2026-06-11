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
    session = (
        tmp_path
        / "users"
        / rec.safe_user_id
        / "Qwen3.6-35B-A3B-Instruct"
        / "20260610T010200Z"
    )
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
