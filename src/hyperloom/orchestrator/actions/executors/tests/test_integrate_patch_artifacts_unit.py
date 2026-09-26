# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for non-diff tuned-artifact integration:
``_resolve_artifact_specs`` sandbox validation and the
``_backup_artifacts`` / ``_apply_artifacts`` / restore round-trip."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
)
from hyperloom.orchestrator.delivery import ledger
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


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
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])

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
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])

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
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])

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
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])

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
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": "a.json", "target": "../escape.json"}],
        done_payload=None,
    )
    assert specs == []
    assert errors == [{"artifact": "../escape.json", "error": "target_unresolved_or_escapes_root"}]


def test_resolve_artifact_specs_missing_source_or_target(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(tmp_path)])

    specs, errors = ip._resolve_artifact_specs(
        specialist_workspace=ws,
        explicit_artifacts=[{"source": "a.json"}],
        done_payload=None,
    )
    assert specs == []
    assert len(errors) == 1
    assert errors[0]["error"] == "missing_source_or_target"


# ---- _backup_artifacts / _apply_artifacts / restore round-trip ------------
def _spec(source: Path, target: Path, rel: str) -> ip._ArtifactSpec:
    resolved = target.resolve()
    root = Path(str(resolved)[: -len(rel)]) if rel and str(resolved).endswith(rel) else resolved.parent
    return ip._ArtifactSpec(source=source.resolve(), target=resolved, rel_target=rel, root=root)


def _install(tmp_path: Path, specs: list[ip._ArtifactSpec], backup_root: Path):
    """Run the two phases in their production order and return the apply result."""
    executor = IntegratePatchExecutor(session_dir=tmp_path)
    assert executor._backup_artifacts(specs, backup_root=backup_root) == []
    return executor._apply_artifacts(specs, backup_root=backup_root)


def _reverted(applied: list[dict]) -> list[str]:
    """The framework-relative targets a restore of ``applied`` actually undid."""
    restored, _errors = ledger.restore_records(applied)
    names = {str(rec["target"]): str(rec.get("rel_target") or rec["target"]) for rec in applied}
    return [names[target] for target in restored]


def _git_repo(path: Path) -> Path:
    """A one-commit git tree holding ``src.py`` at ``return 1``."""
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        GIT_AUTHOR_NAME="T",
        GIT_AUTHOR_EMAIL="t@t.local",
        GIT_COMMITTER_NAME="T",
        GIT_COMMITTER_EMAIL="t@t.local",
    )
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True, env=env)
    return path


def test_apply_then_revert_restores_clobbered_target(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("NEW", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir(parents=True)
    target.write_text("ORIGINAL", encoding="utf-8")

    applied, errors = _install(tmp_path, [_spec(src, target, "fw/cfg.json")], tmp_path / "backups")
    assert errors == []
    assert len(applied) == 1
    assert applied[0]["existed"] is True
    assert applied[0]["backup"] is not None
    assert target.read_text(encoding="utf-8") == "NEW"

    assert _reverted(applied) == ["fw/cfg.json"]
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


def test_apply_then_revert_deletes_created_target(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("NEW", encoding="utf-8")
    target = tmp_path / "fw" / "new_cfg.json"

    applied, errors = _install(tmp_path, [_spec(src, target, "fw/new_cfg.json")], tmp_path / "backups")
    assert errors == []
    assert applied[0]["existed"] is False
    assert applied[0]["backup"] is None
    assert target.read_text(encoding="utf-8") == "NEW"

    assert _reverted(applied) == ["fw/new_cfg.json"]
    assert not target.exists()


def test_live_artifact_restore_unwinds_same_target_and_is_idempotent(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("B", encoding="utf-8")
    second.write_text("C", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir()
    target.write_text("A", encoding="utf-8")
    applied, errors = _install(
        tmp_path,
        [_spec(first, target, "fw/cfg.json"), _spec(second, target, "fw/cfg.json")],
        tmp_path / "backups",
    )
    assert errors == []
    assert target.read_text(encoding="utf-8") == "C"

    ledger.restore_records(applied)
    assert target.read_text(encoding="utf-8") == "A"
    ledger.restore_records(applied)
    assert target.read_text(encoding="utf-8") == "A"


def test_artifact_backup_persists_restore_evidence_before_anything_is_installed(tmp_path):
    source = tmp_path / "new.json"
    source.write_text("B", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir()
    target.write_text("A", encoding="utf-8")
    backup_root = tmp_path / "backups"
    executor = IntegratePatchExecutor(session_dir=tmp_path)

    assert executor._backup_artifacts([_spec(source, target, "fw/cfg.json")], backup_root=backup_root) == []
    # The evidence has to be complete while the tree is still untouched.
    assert target.read_text(encoding="utf-8") == "A"
    records = ledger.load_prepared_records(backup_root)
    assert records, "artifact mutation must leave durable restore evidence"
    assert any(record["target"] == str(target.resolve()) for record in records)
    assert any(path.read_bytes() == b"A" for path in backup_root.iterdir() if path.name != ledger.LEDGER_NAME)

    applied, errors = executor._apply_artifacts([_spec(source, target, "fw/cfg.json")], backup_root=backup_root)
    assert errors == []
    assert target.read_text(encoding="utf-8") == "B"
    applied.clear()
    assert ledger.load_prepared_records(backup_root) == records


def test_live_artifact_restore_does_not_claim_missing_backup_as_success(tmp_path):
    source = tmp_path / "new.json"
    source.write_text("B", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir()
    target.write_text("A", encoding="utf-8")
    applied, errors = _install(tmp_path, [_spec(source, target, "fw/cfg.json")], tmp_path / "backups")
    assert errors == []
    backup = Path(applied[0]["backup"])
    content = backup.read_bytes()
    backup.unlink()

    assert "fw/cfg.json" not in _reverted(applied)
    assert target.read_text(encoding="utf-8") == "B"
    backup.write_bytes(content)
    assert "fw/cfg.json" in _reverted(applied)
    assert target.read_text(encoding="utf-8") == "A"


def test_recovery_deletes_an_artifact_whose_target_the_patch_created(tmp_path):
    """A preimage taken after the patches describes the patch's own output.

    The artifact target did not exist when the attempt started; the patch
    created it and the artifact then clobbered it. Recording ``existed=True``
    there makes the restore write the patch's content back and makes the git
    sweep spare the file, so the reverted tree keeps a file no accepted stack
    ever produced.
    """
    repo = _git_repo(tmp_path / "fw")
    recovery_root = tmp_path / "recovery"
    backup_root = recovery_root / "artifact_backups"
    source = tmp_path / "tuned.json"
    source.write_text("ARTIFACT", encoding="utf-8")
    target = repo / "X.json"
    spec = _spec(source, target, "X.json")
    executor = IntegratePatchExecutor(session_dir=tmp_path)

    assert executor._backup_artifacts([spec], backup_root=backup_root) == []
    # The patch runs between the two phases, creating the artifact's target.
    target.write_text("PATCH\n", encoding="utf-8")
    (repo / "src.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    applied, errors = executor._apply_artifacts([spec], backup_root=backup_root)
    assert errors == []
    assert applied[0]["existed"] is False
    assert target.read_text(encoding="utf-8") == "ARTIFACT"

    pending = {
        "patches": ["001.patch"],
        "artifacts": [{"target": str(target), "rel_target": "X.json"}],
        "framework_source_root": str(repo),
        "recovery": {
            "version": 1,
            "phase": "ready",
            "root": str(recovery_root),
            "git_head": ip._git_head_sha(repo),
            "artifacts_prepared": True,
        },
    }
    summary = ip.restore_pending_integrate(pending)
    assert summary["failed"] == []
    assert summary["artifacts_reverted"] == ["X.json"]
    assert not target.exists(), "an artifact target the attempt created was left in the tree"
    assert (repo / "src.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_recovery_restores_an_artifact_target_that_predates_the_attempt(tmp_path):
    """The mirror case: a target that was already there comes back, not away."""
    repo = _git_repo(tmp_path / "fw")
    target = repo / "cfg.json"
    target.write_text("ORIGINAL\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@t.local", "commit", "-m", "cfg"],
        check=True,
        capture_output=True,
    )
    recovery_root = tmp_path / "recovery"
    backup_root = recovery_root / "artifact_backups"
    source = tmp_path / "tuned.json"
    source.write_text("ARTIFACT", encoding="utf-8")
    spec = _spec(source, target, "cfg.json")
    executor = IntegratePatchExecutor(session_dir=tmp_path)

    assert executor._backup_artifacts([spec], backup_root=backup_root) == []
    (repo / "src.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    applied, errors = executor._apply_artifacts([spec], backup_root=backup_root)
    assert errors == []
    assert applied[0]["existed"] is True

    pending = {
        "patches": ["001.patch"],
        "artifacts": [{"target": str(target), "rel_target": "cfg.json"}],
        "framework_source_root": str(repo),
        "recovery": {
            "version": 1,
            "phase": "ready",
            "root": str(recovery_root),
            "git_head": ip._git_head_sha(repo),
            "artifacts_prepared": True,
        },
    }
    summary = ip.restore_pending_integrate(pending)
    assert summary["failed"] == []
    assert summary["artifacts_reverted"] == ["cfg.json"]
    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert (repo / "src.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_recovery_reverts_the_patches_when_the_artifact_phase_never_started(tmp_path):
    """No witness means no artifact was installed, which is nothing to undo.

    The artifact phase is planned but the apply died in the patch loop before
    reaching it, so ``artifact_backups/`` does not exist at all. Reading the
    plan as an obligation strands the patches that did land: the pending record
    stays at ``ready`` and the framework tree keeps a candidate nobody graded.
    """
    recovery_root = tmp_path / "recovery"
    backup_root = recovery_root / "patch_backups"
    backup_root.mkdir(parents=True)
    target = tmp_path / "src.py"
    backup = backup_root / "000_src.py.bak"
    backup.write_text("def f():\n    return 1\n", encoding="utf-8")
    assert ledger.append_record(
        backup_root,
        {
            "target": str(target),
            "existed": True,
            "backup_path": str(backup),
            "revert_action": "restore",
            "pre_image_sha256": ledger.file_digest(backup),
        },
    )
    assert ledger.mark_prepared(backup_root)
    target.write_text("def f():\n    return 2\n", encoding="utf-8")

    pending = {
        "patches": ["001.patch", "002.patch"],
        "artifacts": [{"target": str(tmp_path / "tuned.json"), "rel_target": "tuned.json"}],
        "framework_source_root": str(tmp_path),
        "recovery": {"version": 1, "phase": "ready", "root": str(recovery_root)},
    }
    summary = ip.restore_pending_integrate(pending)
    assert summary["failed"] == []
    assert summary["artifacts_reverted"] == []
    assert summary["reversed"] == ["002.patch", "001.patch"]
    assert pending["recovery"]["phase"] == "restored"
    assert target.read_text(encoding="utf-8") == "def f():\n    return 1\n"
    assert not (recovery_root / "artifact_backups").exists()


@pytest.mark.parametrize("damage", ["deleted", "emptied", "prefix_rewritten"])
def test_recovery_refuses_a_witnessed_artifact_ledger_that_lost_its_evidence(tmp_path, damage):
    """With a witness on record, missing evidence is damage, not absence."""
    source = tmp_path / "new.json"
    source.write_text("B", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir()
    target.write_text("A", encoding="utf-8")
    backup_root = tmp_path / "out" / "artifact_backups"
    applied, errors = _install(tmp_path, [_spec(source, target, "fw/cfg.json")], backup_root)
    assert errors == []
    backup = Path(applied[0]["backup"])
    path = ledger.ledger_path(backup_root)
    if damage == "deleted":
        path.unlink()
    elif damage == "emptied":
        path.write_text("", encoding="utf-8")
    else:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        record = json.loads(lines[0])
        record["pre_image_sha256"] = "0" * 64
        path.write_text(json.dumps(record) + "\n" + "".join(lines[1:]), encoding="utf-8")

    pending = {
        "artifacts": [{"target": str(target)}],
        "patches": [],
        "workspace": str(backup_root.parent),
        "recovery": {
            "version": 1,
            "phase": "ready",
            "root": str(backup_root.parent),
            "artifacts_prepared": True,
        },
    }
    assert ip.restore_pending_integrate(pending)["failed"]
    assert pending["recovery"]["phase"] == "ready"
    assert target.read_text(encoding="utf-8") == "B"
    assert backup.read_text(encoding="utf-8") == "A"


@pytest.mark.parametrize("damage", ["unknown_action", "bad_mode", "bad_backup", "relative_target"])
def test_recovery_refuses_an_invalid_record_inside_the_committed_prefix(tmp_path, damage):
    """A checkpoint certifies the bytes, not that they describe a restore."""
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir()
    target.write_text("B", encoding="utf-8")
    backup_root = tmp_path / "out" / "artifact_backups"
    backup_root.mkdir(parents=True)
    backup = backup_root / "000_cfg.json.bak"
    backup.write_text("A", encoding="utf-8")
    record = {
        "target": str(target),
        "rel_target": "fw/cfg.json",
        "existed": True,
        "backup": str(backup),
        "pre_image_sha256": ledger.file_digest(backup),
        "mode": 0o644,
    }
    record.update(
        {"revert_action": "erase"}
        if damage == "unknown_action"
        else {"mode": "777"}
        if damage == "bad_mode"
        else {"backup": []}
        if damage == "bad_backup"
        else {"target": "fw/cfg.json"}
    )
    assert ledger.append_record(backup_root, record)
    assert ledger.mark_prepared(backup_root)

    pending = {
        "artifacts": [{"target": str(target)}],
        "patches": [],
        "workspace": str(backup_root.parent),
        "recovery": {
            "version": 1,
            "phase": "ready",
            "root": str(backup_root.parent),
            "artifacts_prepared": True,
        },
    }
    assert ip.restore_pending_integrate(pending)["failed"]
    assert pending["recovery"]["phase"] == "ready"
    assert target.read_text(encoding="utf-8") == "B"
    assert backup.read_text(encoding="utf-8") == "A"


@pytest.mark.parametrize("tail", ["{broken\n", '{"target": "relative", "existed": true}\n'])
def test_recovery_ignores_a_tail_written_after_the_last_checkpoint(tmp_path, tail):
    """Records past the checkpoint describe mutations that never ran."""
    source = tmp_path / "new.json"
    source.write_text("B", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"
    target.parent.mkdir()
    target.write_text("A", encoding="utf-8")
    backup_root = tmp_path / "out" / "artifact_backups"
    _applied, errors = _install(tmp_path, [_spec(source, target, "fw/cfg.json")], backup_root)
    assert errors == []
    path = ledger.ledger_path(backup_root)
    path.write_text(path.read_text(encoding="utf-8") + tail, encoding="utf-8")

    pending = {
        "artifacts": [{"target": str(target)}],
        "patches": [],
        "workspace": str(backup_root.parent),
        "recovery": {
            "version": 1,
            "phase": "ready",
            "root": str(backup_root.parent),
            "artifacts_prepared": True,
        },
    }
    assert ip.restore_pending_integrate(pending)["failed"] == []
    assert pending["recovery"]["phase"] == "restored"
    assert target.read_text(encoding="utf-8") == "A"


@pytest.mark.parametrize("checkpoint", ["missing", "incomplete"])
def test_non_git_recovery_ignores_an_unproven_ledger_tail(tmp_path, checkpoint):
    """An uncommitted tail is never restored from, and never vetoes a prefix.

    ``patch_backups`` is one ledger shared by every patch of an attempt, so a
    later patch dying mid-preparation appends records with no checkpoint after
    them. Those records precede mutations that never ran, and treating them as
    proof that the whole ledger is unusable is what left the earlier, fully
    applied patch in the framework tree.
    """
    target = tmp_path / "cfg.json"
    target.write_text("B", encoding="utf-8")
    backup_root = tmp_path / "out" / "patch_backups"
    backup_root.mkdir(parents=True)
    backup = backup_root / "original.bak"
    backup.write_text("A", encoding="utf-8")
    if checkpoint == "incomplete":
        assert ledger.mark_prepared(backup_root)
    record = {
        "target": str(target),
        "existed": True,
        "backup_path": str(backup),
        "pre_image_sha256": ledger.file_digest(backup),
        "revert_action": "restore",
    }
    assert ledger.append_record(backup_root, record)
    pending = {
        "patches": ["candidate.patch"],
        "artifacts": [],
        "workspace": str(backup_root.parent),
        "framework_source_root": str(tmp_path),
        "recovery": {"version": 1, "phase": "ready", "root": str(backup_root.parent)},
    }
    assert ledger.load_prepared_records(backup_root) == []
    # Nothing committed means nothing to undo, and nothing is touched either.
    assert ip.restore_pending_integrate(pending)["failed"] == []
    assert target.read_text(encoding="utf-8") == "B"
    assert backup.read_text(encoding="utf-8") == "A"

    pending["recovery"]["phase"] = "ready"
    assert ledger.mark_prepared(backup_root)
    assert ledger.load_prepared_records(backup_root) == [record]
    assert ip.restore_pending_integrate(pending)["failed"] == []
    assert pending["recovery"]["phase"] == "restored"
    assert target.read_text(encoding="utf-8") == "A"


@pytest.mark.parametrize("existed", [False, True])
def test_restore_never_follows_a_replaced_target_symlink(tmp_path, existed):
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("DO NOT TOUCH")
    target = tmp_path / "candidate"
    try:
        target.symlink_to(unrelated)
    except OSError:
        pytest.skip("file symlinks unavailable")
    backup = tmp_path / "original.bak"
    backup.write_text("ORIGINAL")
    record = {
        "target": str(target),
        "existed": existed,
        "backup_path": str(backup) if existed else None,
        "pre_image_sha256": ledger.file_digest(backup) if existed else "",
    }
    restored, errors = ledger.restore_records([record])
    assert unrelated.read_text() == "DO NOT TOUCH"
    if existed:
        assert errors
        assert restored == []
    else:
        assert not errors
        assert restored == [str(target)]
        assert not target.is_symlink()


def test_artifact_ledger_failure_leaves_the_target_uninstallable(tmp_path, monkeypatch):
    source = tmp_path / "new.json"
    source.write_text("B", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text("A", encoding="utf-8")
    backup_root = tmp_path / "backups"
    spec = _spec(source, target, "target.json")
    executor = IntegratePatchExecutor(session_dir=tmp_path)
    monkeypatch.setattr(ip, "append_record", lambda *_args: False)

    assert executor._backup_artifacts([spec], backup_root=backup_root)
    assert target.read_text(encoding="utf-8") == "A"

    # No committed preimage, so the install must refuse rather than clobber.
    applied, errors = executor._apply_artifacts([spec], backup_root=backup_root)
    assert applied == []
    assert errors
    assert target.read_text(encoding="utf-8") == "A"


def test_apply_keeps_artifact_when_not_reverted(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("NEW", encoding="utf-8")
    target = tmp_path / "fw" / "cfg.json"

    _applied, errors = _install(tmp_path, [_spec(src, target, "fw/cfg.json")], tmp_path / "backups")
    assert errors == []
    # On KEEP nothing restores the preimage, so the install persists.
    assert target.read_text(encoding="utf-8") == "NEW"


# ---- _resolve_artifact_target: absolute-within-root (Option A) ------------
def test_resolve_artifact_target_absolute_within_root(tmp_path, monkeypatch):
    """An ABSOLUTE target pointing inside a framework search root (e.g. the
    installed aiter package dir) must resolve."""
    fw = tmp_path / "aiter"
    (fw / "configs" / "model_configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])
    abs_target = str(fw / "configs" / "model_configs" / "tuned_fmoe.csv")
    out = ip._resolve_artifact_target(abs_target)
    assert out == (
        (fw / "configs" / "model_configs" / "tuned_fmoe.csv").resolve(),
        "configs/model_configs/tuned_fmoe.csv",
        fw.resolve(),
    )


def test_resolve_artifact_target_absolute_outside_roots_rejected(tmp_path, monkeypatch):
    """An absolute target OUTSIDE every framework root must stay rejected."""
    fw = tmp_path / "aiter"
    (fw / "configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])
    assert ip._resolve_artifact_target("/etc/passwd") is None


def test_resolve_artifact_target_relative_still_works(tmp_path, monkeypatch):
    """Relative targets keep resolving under a framework search root."""
    fw = tmp_path / "aiter"
    (fw / "configs" / "model_configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])
    out = ip._resolve_artifact_target("configs/model_configs/tuned_fmoe.csv")
    assert out == (
        (fw / "configs" / "model_configs" / "tuned_fmoe.csv").resolve(),
        "configs/model_configs/tuned_fmoe.csv",
        fw.resolve(),
    )


def test_resolve_artifact_target_pip_root_does_not_double_the_package_name(tmp_path, monkeypatch):
    """A package-prefixed target must land beside the package, not under it.

    A pip-installed root IS the package dir, so joining ``vllm/...`` onto it
    yields ``.../vllm/vllm/...``. Nothing downstream catches that: the install
    site mkdirs the parent and copies, so the write reports success while the
    runtime never reads the path.
    """
    pkg = tmp_path / "dist-packages" / "vllm"
    (pkg / "model_executor" / "configs").mkdir(parents=True)
    (pkg / "__init__.py").touch()
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(pkg)])

    out = ip._resolve_artifact_target("vllm/model_executor/configs/E=8.json")
    assert out is not None
    target, rel, root = out
    assert target == (pkg / "model_executor" / "configs" / "E=8.json").resolve()
    assert rel == "model_executor/configs/E=8.json"
    assert root == pkg.resolve()


def test_resolve_artifact_target_pip_root_still_joins_a_bare_target_directly(tmp_path, monkeypatch):
    """A target that does not repeat the package name keeps the direct join."""
    pkg = tmp_path / "dist-packages" / "vllm"
    (pkg / "model_executor" / "configs").mkdir(parents=True)
    (pkg / "__init__.py").touch()
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(pkg)])

    out = ip._resolve_artifact_target("model_executor/configs/E=8.json")
    assert out is not None
    assert out[0] == (pkg / "model_executor" / "configs" / "E=8.json").resolve()


def test_resolve_artifact_target_checkout_root_is_unaffected(tmp_path, monkeypatch):
    """A checkout has no ``__init__.py`` at its root, so only the direct join applies."""
    checkout = tmp_path / "sgl-workspace" / "vllm"
    (checkout / "vllm" / "configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(checkout)])

    out = ip._resolve_artifact_target("vllm/configs/E=8.json")
    assert out is not None
    assert out[0] == (checkout / "vllm" / "configs" / "E=8.json").resolve()


def test_resolve_artifact_target_absolute_with_dotdot_rejected(tmp_path, monkeypatch):
    """An absolute target containing ``..`` is rejected even if it would
    normalise inside a root."""
    fw = tmp_path / "aiter"
    (fw / "configs").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])
    assert ip._resolve_artifact_target(str(fw / "configs" / ".." / ".." / "x.csv")) is None


def test_resolve_artifact_specs_absolute_target_records_relative_rel_target(tmp_path, monkeypatch):
    """An absolute target inside a framework root must be recorded with a
    FRAMEWORK-RELATIVE ``rel_target`` so the KEEP source-snapshot (which treats
    rel_target as framework-relative via ``snapshot_source_layer``) captures the
    installed artifact."""
    fw = tmp_path / "aiter"
    (fw / "configs" / "model_configs").mkdir(parents=True)
    ws = tmp_path / "ws"
    (ws / "worktree" / "artifacts").mkdir(parents=True)
    (ws / "worktree" / "artifacts" / "tuned.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(ip, "resolve_kernel_search_roots", lambda: [str(fw)])
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
