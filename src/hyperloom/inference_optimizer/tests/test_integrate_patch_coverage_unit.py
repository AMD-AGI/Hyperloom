# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplementary coverage for IntegratePatchExecutor decision + KB paths."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .conftest import patch_integrate_patch_allowlist

from hyperloom.orchestrator.actions.executors import integrate_patch as ip

from ._optimize_fixtures import variant_result
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
    _git_checkout_clean,
    _detect_p_level,
    _read_done_payload,
)
from hyperloom.orchestrator.actions.executors._workload_envs import (
    FrameworkScriptMismatchError,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


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


@pytest.fixture(autouse=True)
def _integrate_patch_test_framework_roots(monkeypatch, tmp_path):
    patch_integrate_patch_allowlist(monkeypatch, tmp_path)


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


@pytest.mark.asyncio
async def test_missing_specialist_task_id(tmp_path):
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    res = await ex(_make_ctx("t", {"specialist_task_id": ""}))
    assert res["status"] == "failed"
    assert res["error_class"] == "missing_param"


@pytest.mark.asyncio
async def test_no_framework_agent_root(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    _write_workspace(session, "spec")
    monkeypatch.setattr(
        ip,
        "_resolve_framework_root",
        lambda explicit, patch_paths=None, patch_texts=None, recorded_root=None: None,
    )
    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(_make_ctx("t", {"specialist_task_id": "spec"}))
    assert res["status"] == "apply_failed"
    assert res["error_class"] == "no_framework_agent_root"


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
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_forged_task_without_critic_verdict_is_rejected(tmp_path):
    # Dispatch replays PolicyGate before queued→running (see test_dispatched_task_policy).
    # This case still covers the executor-layer critic gate: with SharedState present
    # but no recorded verdict, integrate_patch must refuse to apply the patch.
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    class _SS:
        def get_specialist_patch_verdict(self, tid):
            return ""  # no verdict on record

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo)},
            extra={"shared_state": _SS()},
        )
    )
    assert res["status"] == "rejected_by_critic"
    # patch not applied: the source file is untouched.
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_slash_framework_root_override_rejected(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    class _SS:
        def get_specialist_patch_verdict(self, tid):
            return "approve"

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": "/",
                "apply_only": True,
            },
            extra={"shared_state": _SS()},
        )
    )
    assert res["status"] == "apply_failed"
    assert res["error_class"] == "framework_source_root_rejected"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_forged_task_rejected_before_any_side_effect(tmp_path, monkeypatch):
    # SWSPLAT-42420 (all-or-nothing): the Critic gate must fire BEFORE setup
    # replay / patch apply, so a forged enablement task never runs its
    # setup_commands (no pip install / live-tree mutation) before being refused.
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    called = {"setup": False}

    def _spy_setup(*args, **kwargs):
        called["setup"] = True
        return {"applied": [], "skipped": [], "failed": []}

    monkeypatch.setattr(ip, "_run_setup_commands", _spy_setup)

    class _SS:
        def get_specialist_patch_verdict(self, tid):
            return ""  # no verdict -> must reject before setup

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "enablement": True,
                "setup_commands": ["pip install evil"],
            },
            extra={"shared_state": _SS()},
        )
    )
    assert res["status"] == "rejected_by_critic"
    assert called["setup"] is False, "setup_commands ran before the Critic gate"
    assert (repo / "src.py").read_text().endswith("return 1\n")


def _capture_bench(captured: dict, result: dict, gate: dict):
    """A ``_bench_patch`` stub that records the kwargs it was handed."""

    async def _b(self, **kwargs):
        captured.update(kwargs)
        return result, gate

    return _b


@pytest.mark.asyncio
async def test_bench_is_bounded_by_the_session_budget(tmp_path, monkeypatch):
    """The patch bench is handed the session budget, as the other arms are.

    Its declared cap answers "how long before this counts as hung", not "how much
    budget is left", so without the session deadline a patch benched near the end
    of a run outlives the run itself.
    """
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    class _SS:
        baseline_runtime_sec = 600.0

        def get_specialist_patch_verdict(self, tid):
            return "approve"

        def grid_session_deadline_sec(self, **_kwargs):
            return 4242.0

    captured: dict[str, Any] = {}
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _capture_bench(captured, {"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": None}),
    )
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "framework_source_root": str(repo),
                "base_tput": 100.0,
            },
            extra={"shared_state": _SS()},
        )
    )

    assert res["status"] == "kept"
    assert captured["session_deadline_sec"] == 4242.0
    # The expected runtime, not the backstop cap: admitting on the backstop
    # abandons the tail of the budget.
    assert captured["variant_expected_sec"] == 600.0


@pytest.mark.asyncio
async def test_bench_budget_is_unbounded_without_a_session(tmp_path, monkeypatch):
    """No session context means no deadline, not a deadline of zero."""
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    captured: dict[str, Any] = {}
    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_bench_patch",
        _capture_bench(captured, {"output_throughput": 200.0, "status": "succeeded"}, {"accuracy_pass": None}),
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

    assert res["status"] == "kept"
    assert captured["session_deadline_sec"] is None
    assert captured["variant_expected_sec"] is None


@pytest.mark.asyncio
async def test_bench_patch_forwards_the_session_budget_to_the_grid(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("benchmark: {}\n", encoding="utf-8")

    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod

    captured: dict[str, Any] = {}

    async def fake_run_grid(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(ip_mod, "run_grid", fake_run_grid)
    monkeypatch.setattr(ip_mod, "materialize_config_with_envs", lambda *a, **k: config_path)

    await IntegratePatchExecutor(session_dir=session)._bench_patch(
        params={"config_path": str(config_path)},
        output_root=tmp_path / "out",
        extra_server_args_applied="",
        extra_envs_applied={},
        specialist_task_id="spec",
        session_deadline_sec=4242.0,
        variant_expected_sec=600.0,
    )

    assert captured["session_deadline_sec"] == 4242.0
    assert captured["variant_expected_sec"] == 600.0


@pytest.mark.asyncio
async def test_switch_off_parity_leg_is_bounded_by_the_session_budget(tmp_path, monkeypatch):
    """The parity leg is a second full bench, so it needs the same bound."""
    session = tmp_path / "s"
    session.mkdir()
    captured: dict[str, Any] = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return ({"output_throughput": 100.0, "status": "succeeded"}, {"accuracy_pass": None})

    ex = IntegratePatchExecutor(session_dir=session)
    monkeypatch.setattr(ex, "_bench_patch", _capture)

    await ex._switch_off_parity(
        params={},
        output_root=tmp_path,
        specialist_task_id="spec",
        switch_manifest=[{"switch": "HYPERLOOM_REWRITE_X"}],
        base_tput=100.0,
        session_deadline_sec=4242.0,
        variant_expected_sec=600.0,
    )

    assert captured["session_deadline_sec"] == 4242.0
    assert captured["variant_expected_sec"] == 600.0


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
            },
        )
    )
    assert res["status"] == "kept"
    assert res["delta_pct"] == 100.0
    assert (repo / "src.py").read_text().endswith("return 2\n")


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


@pytest.mark.asyncio
async def test_framework_kb_writeback(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    proposal = {
        "provenance": "specialist:serving:framework:x",
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
        "hyperloom.orchestrator.knowledge.kb_writeback.write_framework_record",
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
            },
        )
    )
    assert res["status"] == "kept"
    assert calls["outcome"] == "integrated"
    assert calls["pr_url"] == "https://example/pr/1"


@pytest.mark.asyncio
async def test_framework_kb_writeback_config_lever_untagged_proposal(tmp_path, monkeypatch):
    """A same-framework config-lever deliverable has no ``specialist:serving:
    framework`` provenance on its own (only the cross-framework prompt path
    emits that tag) — the FRAMEWORK dispatch context must stamp it so the KB
    write still fires.
    """
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    proposal = {"name": "cfg", "extra_envs": {"SGLANG_USE_AITER": "1"}}
    _write_workspace(session, "spec", proposal_set=[proposal])

    calls = {}

    async def _fake_write(**kwargs):
        calls.update(kwargs)
        return "/tmp/lessons.jsonl"

    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.kb_writeback.write_framework_record",
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
                "config_changes": {"SGLANG_USE_AITER": "1"},
                "framework_agent_authoring": True,
                "framework_agent_candidate_id": "https://github.com/ROCm/aiter/pull/1",
            },
        )
    )
    assert res["status"] == "kept"
    assert calls["outcome"] == "integrated"
    assert calls["pr_url"] == "https://github.com/ROCm/aiter/pull/1"


class _FakeSharedState:
    def __init__(self, framework: str = "sglang") -> None:
        self.framework = framework


def test_stamp_framework_kb_provenance_config_lever():
    payload: dict[str, Any] = {"proposal_set": [{"extra_envs": {"X": "1"}}]}
    ip._stamp_framework_kb_provenance(
        payload,
        params={"framework_agent_authoring": True, "framework_agent_candidate_id": "https://x/pr/9"},
        shared_state=_FakeSharedState("sglang"),
    )
    entry = payload["proposal_set"][0]
    assert entry["provenance"] == "specialist:serving:framework:sglang"
    assert entry["fa_pr_url"] == "https://x/pr/9"
    assert entry["framework"] == "sglang"


def test_stamp_framework_kb_provenance_noop_when_not_framework_dispatch():
    payload: dict[str, Any] = {"proposal_set": [{"extra_envs": {"X": "1"}}]}
    ip._stamp_framework_kb_provenance(payload, params={}, shared_state=_FakeSharedState())
    assert "provenance" not in payload["proposal_set"][0]


def test_stamp_framework_kb_provenance_noop_when_no_candidate_id():
    payload: dict[str, Any] = {"proposal_set": [{}]}
    ip._stamp_framework_kb_provenance(
        payload, params={"framework_agent_authoring": True}, shared_state=_FakeSharedState()
    )
    assert "provenance" not in payload["proposal_set"][0]


def test_stamp_framework_kb_provenance_leaves_cross_framework_tag_alone():
    payload: dict[str, Any] = {
        "proposal_set": [{"provenance": "specialist:serving:framework:cross_framework:sglang->vllm"}]
    }
    ip._stamp_framework_kb_provenance(
        payload,
        params={"framework_agent_authoring": True, "framework_agent_candidate_id": "https://x/pr/9"},
        shared_state=_FakeSharedState("sglang"),
    )
    assert payload["proposal_set"][0]["provenance"] == "specialist:serving:framework:cross_framework:sglang->vllm"
    assert "fa_pr_url" not in payload["proposal_set"][0]


def test_stamp_framework_kb_provenance_synthesizes_missing_proposal_set():
    payload: dict[str, Any] = {}
    ip._stamp_framework_kb_provenance(
        payload,
        params={"framework_agent_authoring": True, "framework_agent_candidate_id": "https://x/pr/9"},
        shared_state=_FakeSharedState("vllm"),
    )
    assert payload["proposal_set"][0]["provenance"] == "specialist:serving:framework:vllm"
    assert payload["proposal_set"][0]["fa_pr_url"] == "https://x/pr/9"


def test_stamp_framework_kb_provenance_noop_when_done_payload_none():
    ip._stamp_framework_kb_provenance(
        None,
        params={"framework_agent_authoring": True, "framework_agent_candidate_id": "https://x/pr/9"},
        shared_state=_FakeSharedState(),
    )  # must not raise


@pytest.mark.asyncio
async def test_kb_writeback_skips_when_no_pr_keys(tmp_path):
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    payload = {"proposal_set": [{"provenance": "specialist:serving:framework"}]}
    await ex._maybe_write_framework_kb_record(
        done_payload=payload,
        outcome="integrated",
        tps_delta_pct=1.0,
        extra={},
    )


def test_find_frameworkoposal():
    assert IntegratePatchExecutor._find_frameworkoposal(None) is None
    assert IntegratePatchExecutor._find_frameworkoposal({"proposal_set": "x"}) is None
    assert IntegratePatchExecutor._find_frameworkoposal({"proposal_set": [{"provenance": "kernel:x"}]}) is None
    found = IntegratePatchExecutor._find_frameworkoposal(
        {"proposal_set": [{"provenance": "specialist:serving:framework:y", "id": 1}]}
    )
    assert found["id"] == 1


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


def _FakeVR(**kw):
    """A real ``VariantResult`` for ``_bench_patch``.

    Was a hand-rolled stand-in carrying ``ttft_ms`` / ``itl_ms`` -- names the
    real dataclass does not have, copied from what the executor read while it
    was wrong. The stand-in made the bug untestable and then broke when it was
    fixed, so the fake is gone and the dataclass is used directly.
    """
    kw.setdefault("workspace", "/tmp/rd")
    return variant_result(**kw)


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
        extra_server_args_applied="",
        extra_envs_applied={"E": "1"},
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
        extra_server_args_applied="",
        extra_envs_applied={},
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
    monkeypatch.setattr(ip, "parse_eval_results", lambda rd, framework=None: {"accuracy": 0.50})
    ex = IntegratePatchExecutor(session_dir=tmp_path)
    _, gate = await ex._bench_patch(
        params={"config_path": str(cfg), "accuracy_baseline": 0.95},
        output_root=tmp_path,
        extra_server_args_applied="",
        extra_envs_applied={},
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
            extra_server_args_applied="",
            extra_envs_applied={},
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
            extra_server_args_applied="",
            extra_envs_applied={},
            specialist_task_id="abc",
        )


@pytest.mark.asyncio
async def test_artifact_install_failed_restores_user_stash(tmp_path, monkeypatch):
    # An artifact-install failure with a dirty tree must restore the auto-stash;
    # otherwise the untracked user file stays trapped in the git stash.
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    # Dirty the tree with an untracked file so integrate_patch auto-stashes (-u).
    scratch = repo / "user_scratch.txt"
    scratch.write_text("user work in progress\n", encoding="utf-8")

    # Force a non-empty artifact set and a failing install so the code hits the
    # artifact_install_failed return branch. It must be a real _ArtifactSpec:
    # integrate_patch inspects each spec (see _is_aiter_gemm_model_config, which
    # reads .source/.target/.kind), so a bare placeholder object raises
    # AttributeError before the branch under test is reached.
    def _fake_resolve(*args, **kwargs):
        spec = ip._ArtifactSpec(
            source=tmp_path / "tuned.json",
            target=repo / "tuned.json",
            rel_target="tuned.json",
            root=repo,
            kind="config_json",
        )
        return [spec], []

    def _fake_apply(self, specs, *, backup_root):
        return [], [{"artifact": "tuned.json", "error": "disk full"}]

    monkeypatch.setattr(ip, "_resolve_artifact_specs", _fake_resolve)
    monkeypatch.setattr(IntegratePatchExecutor, "_apply_artifacts", _fake_apply)
    monkeypatch.setattr(IntegratePatchExecutor, "_revert_artifacts", lambda self, applied: None)

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(
        _make_ctx(
            "t",
            {"specialist_task_id": "spec", "framework_source_root": str(repo)},
        )
    )

    assert res["status"] == "apply_failed"
    assert res["error_class"] == "artifact_install_failed"
    # The user's untracked file must be back in the working tree, not stranded
    # in the stash. This is the regression the fix guards against.
    assert scratch.exists(), "user auto-stash was not restored after artifact_install_failed"
    assert scratch.read_text(encoding="utf-8") == "user work in progress\n"


@pytest.mark.asyncio
async def test_cancelled_gate_reverts_the_patch_and_re_raises(tmp_path, monkeypatch):
    """A cancel unwinds the gate, so the tree it mutated must not outlive it.

    The dispatcher cancels in-flight actions when the run is shutting down or
    the session wall-clock budget is spent. ``CancelledError`` is not an
    ``Exception``, so the gate's own revert handlers never see it: the patch
    would stay in the framework tree, the operator's auto-stash would stay
    unpopped, and the session would run its CLOSE phase against a tree carrying
    an ungraded patch.

    The cancel is re-raised rather than graded as a REVERT: SubAgentRunner
    records a cancelled executor as ``cancelled``, and work the run stopped is
    not work that failed.
    """
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    scratch = repo / "user_scratch.txt"
    scratch.write_text("user work in progress\n", encoding="utf-8")

    async def _cancel(self, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", _cancel)
    ex = IntegratePatchExecutor(session_dir=session)
    with pytest.raises(asyncio.CancelledError):
        await ex(
            _make_ctx(
                "t",
                {"specialist_task_id": "spec", "framework_source_root": str(repo)},
            )
        )

    assert (repo / "src.py").read_text(encoding="utf-8").endswith("return 1\n"), (
        "the cancelled candidate was left applied in the framework tree"
    )
    assert scratch.exists(), "user auto-stash was not restored after the cancel"
    assert scratch.read_text(encoding="utf-8") == "user work in progress\n"
    stash_list = subprocess.run(
        ["git", "-C", str(repo), "stash", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert stash_list.stdout.strip() == "", "the auto-stash was left on the stack"


@pytest.mark.asyncio
async def test_a_cancel_in_the_apply_stage_still_hands_the_stash_back(tmp_path, monkeypatch):
    """The apply stage stashes and mutates the tree, then awaits, same as the gate.

    Each of its failure verdicts writes a KB record before the stash restore that
    returns it, and a cancel arrives at whatever await the action happens to be
    at -- a spent wall-clock budget is what makes it arrive at an arbitrary one.
    Only the gate was guarded, so this window left the operator's uncommitted
    work in ``git stash`` for the rest of the session.
    """
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_git_repo(repo)
    _write_workspace(session, "spec")

    scratch = repo / "user_scratch.txt"
    scratch.write_text("user work in progress\n", encoding="utf-8")

    def _fake_resolve(*args, **kwargs):
        spec = ip._ArtifactSpec(
            source=tmp_path / "tuned.json",
            target=repo / "tuned.json",
            rel_target="tuned.json",
            root=repo,
            kind="config_json",
        )
        return [spec], []

    async def _cancel(self, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(ip, "_resolve_artifact_specs", _fake_resolve)
    monkeypatch.setattr(
        IntegratePatchExecutor,
        "_apply_artifacts",
        lambda self, specs, *, backup_root: ([], [{"artifact": "tuned.json", "error": "disk full"}]),
    )
    monkeypatch.setattr(IntegratePatchExecutor, "_maybe_write_framework_kb_record", _cancel)

    ex = IntegratePatchExecutor(session_dir=session)
    with pytest.raises(asyncio.CancelledError):
        await ex(
            _make_ctx(
                "t",
                {"specialist_task_id": "spec", "framework_source_root": str(repo)},
            )
        )

    assert scratch.read_text(encoding="utf-8") == "user work in progress\n", (
        "the user's uncommitted work was left in the stash"
    )
    stash_list = subprocess.run(
        ["git", "-C", str(repo), "stash", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert stash_list.stdout.strip() == "", "the auto-stash was left on the stack"
    assert (repo / "src.py").read_text(encoding="utf-8").endswith("return 1\n"), (
        "the ungraded candidate was left applied in the framework tree"
    )


# ---- patches_dropped_by_grounding forwarding in _no_patches ----------------


@pytest.mark.asyncio
async def test_no_patches_forwards_grounding_drops(tmp_path, monkeypatch):
    """When all patches were grounding-dropped, the integrate result must carry
    ``patches_dropped_by_grounding`` so framework.py can surface it in the next
    round's mandate.  Without this forwarding the field stays in done_payload and
    is never read by _maybe_rearm_enablement."""
    session = tmp_path / "s"
    session.mkdir()
    ws = session / "runs" / "specialist" / "spec"
    ws.mkdir(parents=True)
    done = ws / "specialist_done.json"
    done.write_text(
        json.dumps(
            {
                "patches_written": [],
                "patches_dropped_by_grounding": ["target file(s) not in any framework tree: sglang_file.py"],
                "proposal_set": [{"name": "p1"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(tmp_path / "fw")])

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(
        _make_ctx(
            "t",
            {
                "specialist_task_id": "spec",
                "enablement": True,
            },
        )
    )

    assert res["status"] == "no_patches"
    assert "patches_dropped_by_grounding" in res
    drops = res["patches_dropped_by_grounding"]
    assert isinstance(drops, list) and len(drops) == 1
    assert "sglang_file.py" in drops[0]


@pytest.mark.asyncio
async def test_no_patches_without_drops_has_no_grounding_key(tmp_path, monkeypatch):
    """When there are no grounding drops the key must be absent (not an empty list)."""
    session = tmp_path / "s"
    session.mkdir()
    ws = session / "runs" / "specialist" / "spec"
    ws.mkdir(parents=True)
    done = ws / "specialist_done.json"
    done.write_text(
        json.dumps({"patches_written": [], "proposal_set": [{"name": "p1"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(tmp_path / "fw")])

    ex = IntegratePatchExecutor(session_dir=session)
    res = await ex(_make_ctx("t", {"specialist_task_id": "spec", "enablement": True}))

    assert res["status"] == "no_patches"
    assert "patches_dropped_by_grounding" not in res
