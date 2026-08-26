# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for non-diff tuned-artifact integration:
``_resolve_artifact_specs`` sandbox validation and the
``_apply_artifacts`` / ``_revert_artifacts`` backup-restore round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task

from .conftest import init_git_repo, patch_integrate_patch_allowlist
from .test_integrate_patch_executor import _VALID_PATCH, _write_specialist_workspace


def _make_workspace(tmp_path: Path) -> Path:
    """A specialist workspace whose ``worktree`` holds an authored artifact."""
    ws = tmp_path / "workspace"
    (ws / "worktree").mkdir(parents=True)
    return ws


def _make_ctx(task_id: str, params: dict) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


# ---- _resolve_artifact_specs: sandbox validation ----
def test_resolve_artifact_specs_valid(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    src = ws / "worktree" / "tuned.json"
    src.write_text('{"x": 1}', encoding="utf-8")

    fw = tmp_path / "framework"
    (fw / "vllm" / "configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[
            {
                "source": "tuned.json",
                "target": "vllm/configs/tuned.json",
                "kind": "config_json",
                "description": "tuned GEMM config",
            }
        ],
        done_payload=None,
    )

    assert errors == []
    assert len(specs) == 1
    spec = specs[0]
    assert spec.source == src.resolve()
    assert spec.target == (fw / "vllm" / "configs" / "tuned.json").resolve()
    assert spec.rel_target == "vllm/configs/tuned.json"
    assert spec.kind == "config_json"
    assert spec.description == "tuned GEMM config"


def test_resolve_artifact_specs_reads_done_payload(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    (ws / "worktree" / "a.json").write_text("{}", encoding="utf-8")
    fw = tmp_path / "framework"
    (fw / "vllm").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=None,
        done_payload={"artifacts_written": [{"source": "a.json", "target": "vllm/a.json"}]},
    )
    assert errors == []
    assert len(specs) == 1


def test_resolve_artifact_specs_source_not_found(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    fw = tmp_path / "framework"
    fw.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": "missing.json", "target": "x.json"}],
        done_payload=None,
    )
    assert specs == []
    assert errors == [{"artifact": "missing.json", "error": "source_not_found"}]


def test_resolve_artifact_specs_source_outside_workspace(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    fw = tmp_path / "framework"
    fw.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": str(outside), "target": "x.json"}],
        done_payload=None,
    )
    assert specs == []
    assert errors == [{"artifact": str(outside), "error": "source_outside_workspace"}]


def test_resolve_artifact_specs_target_escapes_root(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    (ws / "worktree" / "a.json").write_text("{}", encoding="utf-8")
    fw = tmp_path / "framework"
    fw.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": "a.json", "target": "../escape.json"}],
        done_payload=None,
    )
    assert specs == []
    assert errors == [{"artifact": "../escape.json", "error": "target_unresolved_or_escapes_root"}]


def test_resolve_artifact_specs_missing_source_or_target(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(tmp_path)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": "a.json"}],
        done_payload=None,
    )
    assert specs == []
    assert len(errors) == 1
    assert errors[0]["error"] == "missing_source_or_target"


# ---- _apply_artifacts / _revert_artifacts round-trip ----------------------
def _spec(source: Path, target: Path, rel: str) -> ip._ArtifactSpec:
    resolved = target.resolve()
    root = Path(str(resolved)[: -len(rel)]) if rel and str(resolved).endswith(rel) else resolved.parent
    return ip._ArtifactSpec(source=source.resolve(), target=resolved, rel_target=rel, root=root)


def test_apply_then_revert_restores_clobbered_target(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("NEW", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir(parents=True)
    target.write_text("ORIGINAL", encoding="utf-8")
    backup_root = tmp_path / "backups"

    applied, errors = IntegratePatchExecutor._apply_artifacts(
        IntegratePatchExecutor,
        [_spec(src, target, "fw/cfg.json")],
        backup_root=backup_root,
    )
    assert errors == []
    assert len(applied) == 1
    assert applied[0]["existed"] is True
    assert applied[0]["backup"] is not None
    assert target.read_text(encoding="utf-8") == "NEW"

    reverted = IntegratePatchExecutor._revert_artifacts(applied)
    assert reverted == ["fw/cfg.json"]
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


def test_apply_then_revert_deletes_created_target(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("NEW", encoding="utf-8")
    target = tmp_path / "fw" / "new_cfg.json"
    backup_root = tmp_path / "backups"

    applied, errors = IntegratePatchExecutor._apply_artifacts(
        IntegratePatchExecutor,
        [_spec(src, target, "fw/new_cfg.json")],
        backup_root=backup_root,
    )
    assert errors == []
    assert applied[0]["existed"] is False
    assert applied[0]["backup"] is None
    assert target.read_text(encoding="utf-8") == "NEW"

    reverted = IntegratePatchExecutor._revert_artifacts(applied)
    assert reverted == ["fw/new_cfg.json"]
    assert not target.exists()


def test_apply_keeps_artifact_when_not_reverted(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("NEW", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    backup_root = tmp_path / "backups"

    applied, errors = IntegratePatchExecutor._apply_artifacts(
        IntegratePatchExecutor,
        [_spec(src, target, "fw/cfg.json")],
        backup_root=backup_root,
    )
    assert errors == []
    # On KEEP, _revert_artifacts is never called, so the install persists.
    assert target.read_text(encoding="utf-8") == "NEW"


# ---- _resolve_artifact_target: absolute-within-allowlist (Option A) --------
def test_resolve_artifact_target_absolute_within_allowlist(tmp_path, monkeypatch):
    """An ABSOLUTE target pointing inside an allowlisted framework root (e.g.
    the installed aiter package dir) must resolve."""
    fw = tmp_path / "aiter"
    (fw / "configs" / "model_configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    abs_target = str(fw / "configs" / "model_configs" / "tuned_fmoe.csv")
    out = ip._resolve_artifact_target(abs_target)
    assert out == (
        (fw / "configs" / "model_configs" / "tuned_fmoe.csv").resolve(),
        "configs/model_configs/tuned_fmoe.csv",
        fw.resolve(),
    )


def test_resolve_artifact_target_absolute_outside_allowlist_rejected(tmp_path, monkeypatch):
    """An absolute target OUTSIDE every allowlisted root must stay rejected."""
    fw = tmp_path / "aiter"
    (fw / "configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    assert ip._resolve_artifact_target("/etc/passwd") is None


def test_resolve_artifact_target_relative_still_works(tmp_path, monkeypatch):
    """Relative targets keep resolving under an allowlisted root."""
    fw = tmp_path / "aiter"
    (fw / "configs" / "model_configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    out = ip._resolve_artifact_target("configs/model_configs/tuned_fmoe.csv")
    assert out == (
        (fw / "configs" / "model_configs" / "tuned_fmoe.csv").resolve(),
        "configs/model_configs/tuned_fmoe.csv",
        fw.resolve(),
    )


def test_resolve_artifact_target_absolute_with_dotdot_rejected(tmp_path, monkeypatch):
    """An absolute target containing ``..`` is rejected even if it would
    normalise inside a root."""
    fw = tmp_path / "aiter"
    (fw / "configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    assert ip._resolve_artifact_target(str(fw / "configs" / ".." / ".." / "x.csv")) is None


def test_resolve_artifact_specs_absolute_target_records_relative_rel_target(tmp_path, monkeypatch):
    """An absolute target inside an allowlisted root must be recorded with a
    FRAMEWORK-RELATIVE ``rel_target`` so the KEEP source-snapshot (which treats
    rel_target as framework-relative via ``snapshot_source_layer``) captures the
    installed artifact."""
    fw = tmp_path / "aiter"
    (fw / "configs" / "model_configs").mkdir(parents=True)
    ws = tmp_path / "ws"
    (ws / "worktree" / "artifacts").mkdir(parents=True)
    (ws / "worktree" / "artifacts" / "tuned.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    abs_target = str(fw / "configs" / "model_configs" / "tuned.csv")
    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": "artifacts/tuned.csv", "target": abs_target, "kind": "k"}],
        done_payload=None,
    )
    assert errors == [], errors
    assert len(specs) == 1
    assert specs[0].target == (fw / "configs" / "model_configs" / "tuned.csv").resolve()
    assert specs[0].rel_target == "configs/model_configs/tuned.csv"


# ---- _replay_base_artifacts: sandbox + stash-ordering ----------------------


def _make_executor(session_dir: Path) -> IntegratePatchExecutor:
    return IntegratePatchExecutor(session_dir=session_dir)


def _base_art_params(fw: Path, source: Path, target: Path, session_dir: Path) -> dict:
    return {
        "enablement": True,
        "enablement_base_artifacts": [
            {
                "source": str(source),
                "target": str(target),
                "rel_target": str(target.relative_to(fw)),
                "kind": "python_source",
            }
        ],
    }


def test_replay_base_artifacts_installs_into_allowlist_root(tmp_path, monkeypatch):
    """Happy path: a base artifact with a valid source and allowlisted target is installed."""
    fw = tmp_path / "sglang"
    (fw / "srt").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    session_dir = tmp_path / "session"
    source = session_dir / "runs" / "specialist" / "t1" / "artifact.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixed\n", encoding="utf-8")

    target = fw / "srt" / "artifact.py"
    ex = _make_executor(session_dir)
    ex._replay_base_artifacts(_base_art_params(fw, source, target, session_dir))

    assert target.read_text(encoding="utf-8") == "# fixed\n"


def test_replay_base_artifacts_rejects_target_outside_allowlist(tmp_path, monkeypatch):
    """A target that resolves outside any allowlisted root must be skipped silently."""
    fw = tmp_path / "sglang"
    fw.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    session_dir = tmp_path / "session"
    source = session_dir / "runs" / "specialist" / "t1" / "artifact.py"
    source.parent.mkdir(parents=True)
    source.write_text("# evil\n", encoding="utf-8")

    outside = tmp_path / "etc" / "passwd"
    params = {
        "enablement": True,
        "enablement_base_artifacts": [{"source": str(source), "target": str(outside), "rel_target": "passwd"}],
    }
    ex = _make_executor(session_dir)
    ex._replay_base_artifacts(params)

    assert not outside.exists()


def test_replay_base_artifacts_rejects_source_outside_session(tmp_path, monkeypatch):
    """A source file that lives outside the session directory must be skipped."""
    fw = tmp_path / "sglang"
    (fw / "srt").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    session_dir = tmp_path / "session"
    session_dir.mkdir()

    # Source lives next to the session dir, not inside it.
    source = tmp_path / "outside" / "artifact.py"
    source.parent.mkdir()
    source.write_text("# evil\n", encoding="utf-8")

    target = fw / "srt" / "artifact.py"
    ex = _make_executor(session_dir)
    ex._replay_base_artifacts(_base_art_params(fw, source, target, session_dir))

    assert not target.exists()


def test_replay_base_artifacts_skips_missing_source(tmp_path, monkeypatch):
    """A recorded source path that no longer exists must be skipped without error."""
    fw = tmp_path / "sglang"
    (fw / "srt").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    target = fw / "srt" / "artifact.py"

    params = {
        "enablement": True,
        "enablement_base_artifacts": [
            {
                "source": str(session_dir / "runs" / "missing.py"),
                "target": str(target),
                "rel_target": "srt/artifact.py",
            }
        ],
    }
    ex = _make_executor(session_dir)
    ex._replay_base_artifacts(params)  # must not raise

    assert not target.exists()


def test_replay_base_artifacts_noop_for_non_enablement(tmp_path, monkeypatch):
    """Must be a no-op when params['enablement'] is falsy."""
    fw = tmp_path / "sglang"
    (fw / "srt").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])

    session_dir = tmp_path / "session"
    source = session_dir / "runs" / "t1" / "artifact.py"
    source.parent.mkdir(parents=True)
    source.write_text("# should not install\n", encoding="utf-8")
    target = fw / "srt" / "artifact.py"

    ex = _make_executor(session_dir)
    ex._replay_base_artifacts({"enablement_base_artifacts": [{"source": str(source), "target": str(target)}]})
    assert not target.exists()


@pytest.mark.asyncio
async def test_stash_bookkeeping_is_published_before_the_replay_writes(tmp_path, monkeypatch):
    """The undo can only restore a stash it can see, so the replay must run after the publish.

    The replay install is unguarded on purpose, so an OSError from it unwinds through
    ``__call__``. That unwind reads the framework root off the context and skips the
    stash restore entirely when it is absent.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "repo"
    init_git_repo(repo)
    patch_integrate_patch_allowlist(monkeypatch, tmp_path)
    _write_specialist_workspace(session_dir, "t-spec-replay", patch_contents=[_VALID_PATCH])

    ex = IntegratePatchExecutor(session_dir=session_dir)
    seen: dict[str, object] = {}
    real_replay = ex._replay_base_artifacts

    def _spy(params):
        seen["framework_root"] = getattr(ex._spy_ctx, "_ip_framework_root", None)
        seen["stash_state"] = getattr(ex._spy_ctx, "_ip_stash_state", None)
        return real_replay(params)

    monkeypatch.setattr(ex, "_replay_base_artifacts", _spy)
    ctx = _make_ctx(
        "t-int-replay",
        {
            "specialist_task_id": "t-spec-replay",
            "framework_source_root": str(repo),
            "apply_only": True,
            "enablement": True,
        },
    )
    ex._spy_ctx = ctx
    await ex(ctx)

    assert seen["framework_root"] == repo.resolve()
    assert seen["stash_state"] is not None
