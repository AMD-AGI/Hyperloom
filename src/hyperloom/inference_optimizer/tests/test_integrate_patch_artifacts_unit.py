# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for non-diff tuned-artifact integration:
``_resolve_artifact_specs`` sandbox validation and the
``_apply_artifacts`` / ``_revert_artifacts`` backup-restore round-trip."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
)


def _make_workspace(tmp_path: Path) -> Path:
    """A specialist workspace whose ``worktree`` holds an authored artifact."""
    ws = tmp_path / "workspace"
    (ws / "worktree").mkdir(parents=True)
    return ws


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
    return ip._ArtifactSpec(source=source.resolve(), target=target.resolve(), rel_target=rel)


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
