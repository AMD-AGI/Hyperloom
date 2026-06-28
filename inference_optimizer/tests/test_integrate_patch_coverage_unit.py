# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplementary coverage for IntegratePatchExecutor decision + KB paths."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import integrate_patch as ip
from inference_optimizer.orchestrator.action_executors.integrate_patch import (
    IntegratePatchExecutor,
    _git_checkout_clean,
    _detect_p_level,
    _read_done_payload,
)
from inference_optimizer.orchestrator.action_executors._workload_envs import (
    FrameworkScriptMismatchError,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


_VALID_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "T"
    env["GIT_AUTHOR_EMAIL"] = "t@t.local"
    env["GIT_COMMITTER_NAME"] = "T"
    env["GIT_COMMITTER_EMAIL"] = "t@t.local"
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True, env=env)


def _write_workspace(session_dir: Path, task_id: str, *, proposal_set: list | None = None) -> Path:
    workspace = session_dir / "runs" / "specialist" / task_id
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    p = workspace / "worktree" / "patches" / "001.patch"
    p.write_text(_VALID_PATCH, encoding="utf-8")
    payload: dict[str, Any] = {
        "patches_written": [f"patches/{p.name}"],
        "proposal_set": proposal_set or [],
    }
    (workspace / "specialist_done.json").write_text(json.dumps(payload), encoding="utf-8")
    return workspace


def _make_ctx(task_id: str, params: dict[str, Any], extra: dict | None = None) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra=extra or {})


def _stub_bench(result: dict, gate: dict):
    async def _b(self, **kwargs):
        return result, gate

    return _b


# ---- missing param ----
@pytest.mark.asyncio
async def test_missing_specialist_task_id(tmp_path):
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    res = await ex(_make_ctx("t", {"specialist_task_id": ""}))
    assert res["status"] == "failed"
    assert res["error_class"] == "missing_param"


# ---- no framework root ----
@pytest.mark.asyncio
async def test_no_framework_root(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    _write_workspace(session, "spec")
    monkeypatch.setattr(
        ip, "_resolve_framework_root",
        lambda explicit, patch_paths=None: None,
    )
    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(_make_ctx("t", {"specialist_task_id": "spec"}))
    assert res["status"] == "apply_failed"
    assert res["error_class"] == "no_framework_root"


# ---- rejected by critic ----
@pytest.mark.asyncio
async def test_rejected_by_critic(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    class _SS:
        def get_specialist_patch_verdict(self, tid):
            return "reject"

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo)},
            extra={"shared_state": _SS()},
        )
    )
    assert res["status"] == "rejected_by_critic"
    # patch was reverted -> tree restored
    assert (repo / "src.py").read_text().endswith("return 1\n")


# ---- KEEP path ----
@pytest.mark.asyncio
async def test_keep_path(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": None}),
    )
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "base_tput": 100.0,
                "enable_stack_rebench": False,
            },
        )
    )
    assert res["status"] == "kept"
    assert res["delta_pct"] == 100.0
    assert (repo / "src.py").read_text().endswith("return 2\n")


def _stub_confirm(result: dict):
    async def _c(self, **kwargs):
        return result

    return _c


# ---- KEEP confirmed by stack rebench ----
@pytest.mark.asyncio
async def test_keep_confirmed_by_rebench(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": None}),
    )
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_confirm_stack_rebench",
        _stub_confirm(
            {"stable": True, "tput": 190.0, "workspace": "/w", "warnings": [], "stable_floor": 100.0, "accuracy_pass": True}
        ),
    )
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo), "base_tput": 100.0},
        )
    )
    assert res["status"] == "kept"
    # headline tput becomes the confirmed rebench value
    assert res["output_throughput"] == 190.0
    assert res["delta_pct"] == 90.0


# ---- REVERT when rebench misses the stability floor ----
@pytest.mark.asyncio
async def test_revert_when_rebench_unstable(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": None}),
    )
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_confirm_stack_rebench",
        _stub_confirm(
            {"stable": False, "tput": 80.0, "workspace": "/w", "warnings": [], "stable_floor": 100.0, "accuracy_pass": None}
        ),
    )
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo), "base_tput": 100.0},
        )
    )
    assert res["status"] == "reverted"
    assert "stability floor" in res["reason"]
    assert (repo / "src.py").read_text().endswith("return 1\n")


# ---- REVERT when rebench shows an accuracy regression ----
@pytest.mark.asyncio
async def test_revert_when_rebench_accuracy_fails(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": True}),
    )
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_confirm_stack_rebench",
        _stub_confirm(
            {"stable": True, "tput": 190.0, "workspace": "/w", "warnings": [], "stable_floor": 100.0, "accuracy_pass": False}
        ),
    )
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo), "base_tput": 100.0},
        )
    )
    assert res["status"] == "reverted"
    assert "accuracy regression on rebench" in res["reason"]


# ---- REVERT when a framework-authored rebench loses its accuracy verdict ----
@pytest.mark.asyncio
async def test_revert_when_rebench_accuracy_missing_with_baseline(tmp_path, monkeypatch):
    """First bench passes accuracy, but the stable rebench produces NO accuracy
    verdict. For a framework-authored patch with a baseline accuracy, the gate
    must reject (mirrors the first-bench accuracy_keep_block) rather than KEEP on
    the stale first-bench pass."""
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": True}),
    )
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_confirm_stack_rebench",
        _stub_confirm(
            {"stable": True, "tput": 190.0, "workspace": "/w", "warnings": [], "stable_floor": 100.0, "accuracy_pass": None}
        ),
    )
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "base_tput": 100.0,
                "framework_pr_authoring": True,
                "accuracy_baseline": 0.8,
            },
        )
    )
    assert res["status"] == "accuracy_unavailable_reject"
    assert "no eval result" in res["reason"]
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_keep_when_rebench_accuracy_missing_but_not_required(tmp_path, monkeypatch):
    """A generic (non-framework-authored) integrate_patch with no baseline still
    KEEPs on a stable rebench with a missing accuracy verdict — the rebench gate
    only tightens the required+baseline case."""
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": None}),
    )
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_confirm_stack_rebench",
        _stub_confirm(
            {"stable": True, "tput": 190.0, "workspace": "/w", "warnings": [], "stable_floor": 100.0, "accuracy_pass": None}
        ),
    )
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo), "base_tput": 100.0},
        )
    )
    assert res["status"] == "kept"


# ---- REVERT path (low throughput) ----
@pytest.mark.asyncio
async def test_revert_low_throughput(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 50.0, "status": "succeeded"}, {"accuracy_pass": None}),
    )
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "base_tput": 100.0,
            },
        )
    )
    assert res["status"] == "reverted"
    assert (repo / "src.py").read_text().endswith("return 1\n")


# ---- REVERT path (accuracy fail) ----
@pytest.mark.asyncio
async def test_revert_accuracy_fail(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 500.0, "status": "succeeded"}, {"accuracy_pass": False}),
    )
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "base_tput": 100.0,
            },
        )
    )
    assert res["status"] == "reverted"
    assert "accuracy regression" in res["reason"]


# ---- bench raises generic exception ----
@pytest.mark.asyncio
async def test_bench_exception_reverts(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)

    async def _raise(self, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", _raise)
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
            },
        )
    )
    assert res["status"] == "reverted"
    assert res["error_class"] == "bench_exception"


# ---- bench raises FrameworkScriptMismatchError ----
@pytest.mark.asyncio
async def test_bench_script_mismatch_reverts(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")
    ex = IntegratePatchExecutor(session_dir=session)

    async def _raise(self, **kwargs):
        raise FrameworkScriptMismatchError("mismatch")

    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", _raise)
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
            },
        )
    )
    assert res["status"] == "reverted"
    assert res["error_class"] == "framework_script_mismatch"


# ---- framework_pr KB writeback on KEEP ----
@pytest.mark.asyncio
async def test_framework_pr_kb_writeback(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    proposal = {
        "provenance": "specialist:serving:framework_pr:x",
        "fa_pr_url": "https://example/pr/1",
        "fa_pr_sha": "deadbeef",
        "patches_written": ["patches/001.patch"],
    }
    _write_workspace(session, "spec", proposal_set=[proposal])

    calls = {}

    async def _fake_write(**kwargs):
        calls.update(kwargs)
        return "/tmp/lessons.jsonl"

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.kb_writeback.write_framework_pr_record",
        _fake_write,
    )
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _stub_bench({"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": True}),
    )
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "base_tput": 100.0,
                "enable_stack_rebench": False,
            },
        )
    )
    assert res["status"] == "kept"
    assert calls["outcome"] == "integrated"
    assert calls["pr_url"] == "https://example/pr/1"


# ---- _maybe_write helpers directly ----
@pytest.mark.asyncio
async def test_kb_writeback_skips_when_no_pr_keys(tmp_path):
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    payload = {"proposal_set": [{"provenance": "specialist:serving:framework_pr"}]}
    # No fa_pr_url / fa_pr_sha -> early return, no exception.
    await ex._maybe_write_framework_pr_kb_record(
        done_payload=payload,
        outcome="integrated",
        tps_delta_pct=1.0,
        extra={},
    )


def test_find_framework_pr_proposal():
    assert IntegratePatchExecutor._find_framework_pr_proposal(None) is None
    assert IntegratePatchExecutor._find_framework_pr_proposal({"proposal_set": "x"}) is None
    assert IntegratePatchExecutor._find_framework_pr_proposal({"proposal_set": [{"provenance": "kernel:x"}]}) is None
    found = IntegratePatchExecutor._find_framework_pr_proposal(
        {"proposal_set": [{"provenance": "specialist:serving:framework_pr:y", "id": 1}]}
    )
    assert found["id"] == 1


# ---- pure helpers ----
def test_git_checkout_clean(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "src.py").write_text("dirty\n", encoding="utf-8")
    ok, _ = _git_checkout_clean(repo)
    assert ok
    assert (repo / "src.py").read_text().endswith("return 1\n")


def test_detect_p_level_none_for_unmatched(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    bad = tmp_path / "bad.patch"
    bad.write_text(
        "diff --git a/zzz.py b/zzz.py\n--- a/zzz.py\n+++ b/zzz.py\n@@ -1,1 +1,1 @@\n-X\n+Y\n",
        encoding="utf-8",
    )
    assert _detect_p_level(repo, bad, three_way=False) is None


def test_read_done_payload_missing(tmp_path):
    assert _read_done_payload(tmp_path) is None


def test_read_done_payload_bad_json(tmp_path):
    (tmp_path / "specialist_done.json").write_text("{bad", encoding="utf-8")
    assert _read_done_payload(tmp_path) is None


# ---- _bench_patch ----
class _FakeVR:
    """Minimal VariantResult stand-in for _bench_patch."""

    def __init__(self, **kw):
        self.name = kw.get("name", "v")
        self.status = kw.get("status", "succeeded")
        self.output_throughput = kw.get("output_throughput", 123.0)
        self.ttft_ms = kw.get("ttft_ms", 10.0)
        self.itl_ms = kw.get("itl_ms", 5.0)
        # Mirror the real VariantResult attribute name (``workspace``); the
        # bench code reads ``r.workspace`` for the accuracy-gate eval dir.
        self.workspace = kw.get("workspace", "/tmp/rd")
        self.error = kw.get("error", "")
        self.nonfatal_warnings = kw.get("nonfatal_warnings", [])


@pytest.mark.asyncio
async def test_bench_patch_no_accuracy(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setattr(ip, "materialize_config_with_envs", lambda *a, **k: cfg)

    async def _fake_run_grid(**kwargs):
        return [_FakeVR(output_throughput=200.0)]

    monkeypatch.setattr(ip, "run_grid", _fake_run_grid)
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    bench, gate = await ex._bench_patch(
        params={"config_path": str(cfg)},
        output_root=tmp_path,
        config_changes_applied={"E": "1"},
        specialist_task_id="abcdef123456",
    )
    assert bench["output_throughput"] == 200.0
    assert gate["accuracy_pass"] is None


@pytest.mark.asyncio
async def test_bench_patch_with_accuracy(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setattr(ip, "materialize_config_with_envs", lambda *a, **k: cfg)

    async def _fake_run_grid(**kwargs):
        return [_FakeVR(status="succeeded", workspace=str(tmp_path))]

    monkeypatch.setattr(ip, "run_grid", _fake_run_grid)
    monkeypatch.setattr(ip, "parse_eval_results", lambda rd, framework=None: {"accuracy": 0.9})
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    bench, gate = await ex._bench_patch(
        params={"config_path": str(cfg), "accuracy_baseline": 0.8},
        output_root=tmp_path,
        config_changes_applied={},
        specialist_task_id="abcdef123456",
    )
    assert gate["accuracy_pass"] is True


@pytest.mark.asyncio
async def test_bench_patch_accuracy_regression_fails(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setattr(ip, "materialize_config_with_envs", lambda *a, **k: cfg)

    async def _fake_run_grid(**kwargs):
        return [_FakeVR(status="succeeded", workspace=str(tmp_path))]

    monkeypatch.setattr(ip, "run_grid", _fake_run_grid)
    # A large accuracy drop must fail the gate (exercises real key + arg order).
    monkeypatch.setattr(ip, "parse_eval_results", lambda rd, framework=None: {"accuracy": 0.50})
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    _, gate = await ex._bench_patch(
        params={"config_path": str(cfg), "accuracy_baseline": 0.95},
        output_root=tmp_path,
        config_changes_applied={},
        specialist_task_id="abcdef123456",
    )
    assert gate["accuracy_pass"] is False


@pytest.mark.asyncio
async def test_bench_patch_missing_baseline_skips_with_warning(tmp_path, monkeypatch, caplog):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setattr(ip, "materialize_config_with_envs", lambda *a, **k: cfg)

    async def _fake_run_grid(**kwargs):
        return [_FakeVR(status="succeeded", workspace=str(tmp_path))]

    monkeypatch.setattr(ip, "run_grid", _fake_run_grid)
    monkeypatch.setattr(ip, "parse_eval_results", lambda rd, framework=None: {"accuracy": 0.9})
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    with caplog.at_level("WARNING"):
        _, gate = await ex._bench_patch(
            params={"config_path": str(cfg)},
            output_root=tmp_path,
            config_changes_applied={},
            specialist_task_id="abcdef123456",
        )
    assert gate["accuracy_pass"] is None
    assert any("no baseline accuracy" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_bench_patch_config_not_found(tmp_path):
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    with pytest.raises(RuntimeError):
        await ex._bench_patch(
            params={"config_path": str(tmp_path / "missing.yaml")},
            output_root=tmp_path,
            config_changes_applied={},
            specialist_task_id="abc",
        )
