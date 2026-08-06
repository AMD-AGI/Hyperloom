"""Forge keep/revert reads the published best manifest as the authority.

Forge rewrites ``forge_experiments/best_result.json`` atomically on every KEEP,
gated on correctness and pointing at a commit already in the workspace history.
It is therefore current after a clean finish, a soft budget exhaustion, or a
hard kill -- unlike the final-result sidecar, which only exists on a graceful
return. These tests pin that precedence and the lineage checks that keep a stale
manifest from being trusted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> tuple[Path, str]:
    """A workspace with one baseline commit; returns (workspace, base_commit)."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "forge@test")
    _git(workspace, "config", "user.name", "forge")
    (workspace / "kernel.py").write_text("def kernel(x):\n    return x\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "base")
    return workspace, _git(workspace, "rev-parse", "HEAD")


def _publish(workspace: Path, payload: dict) -> None:
    root = workspace / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "best_result.json").write_text(json.dumps(payload), encoding="utf-8")


def _manifest(commit_hash: str, **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "commit_hash": commit_hash,
        "correctness_passed": True,
        "baseline_wall_ms": 2.0,
        "best_wall_ms": 1.0,
        "mean_case_speedup": 2.0,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
        "speedup": 2.0,
        "iteration": 3,
        "snr_db": 45.0,
    }
    payload.update(overrides)
    return payload


def _commit_improvement(workspace: Path) -> str:
    (workspace / "kernel.py").write_text("def kernel(x):\n    return x * 1\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "faster")
    return _git(workspace, "rev-parse", "HEAD")


def test_published_best_is_accepted_after_a_keep(repo):
    workspace, base_commit = repo
    best_commit = _commit_improvement(workspace)
    _publish(workspace, _manifest(best_commit))

    validated = forge_submit._validated_forge_best_result(
        forge_submit._read_forge_best_result(str(workspace)),
        workspace=str(workspace),
        base_commit=base_commit,
    )

    assert validated is not None
    assert validated["best_commit"] == best_commit
    assert validated["baseline_ms"] == 2.0
    assert validated["best_ms"] == 1.0
    assert validated["improved"] is True


def test_missing_manifest_yields_no_evidence(repo):
    workspace, base_commit = repo

    assert forge_submit._read_forge_best_result(str(workspace)) is None
    assert (
        forge_submit._validated_forge_best_result(
            None, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"correctness_passed": False}, id="correctness_failed"),
        pytest.param({"schema_version": 2}, id="unknown_schema"),
        pytest.param({"mean_case_speedup": 1.0}, id="no_mean_case_gain"),
        pytest.param({"mean_case_speedup": None}, id="missing_mean_case_speedup"),
        pytest.param({"baseline_wall_ms": 0.0}, id="unusable_baseline"),
        pytest.param({"best_wall_ms": "fast"}, id="non_numeric_timing"),
    ],
)
def test_manifest_that_does_not_prove_a_win_is_rejected(repo, overrides):
    workspace, base_commit = repo
    best_commit = _commit_improvement(workspace)
    _publish(workspace, _manifest(best_commit, **overrides))

    assert (
        forge_submit._validated_forge_best_result(
            forge_submit._read_forge_best_result(str(workspace)),
            workspace=str(workspace),
            base_commit=base_commit,
        )
        is None
    )


def test_manifest_accepts_non_monotonic_raw_wall(repo):
    workspace, base_commit = repo
    best_commit = _commit_improvement(workspace)
    _publish(
        workspace,
        _manifest(
            best_commit,
            baseline_wall_ms=2.0,
            best_wall_ms=3.0,
            mean_case_speedup=1.5,
        ),
    )

    validated = forge_submit._validated_forge_best_result(
        forge_submit._read_forge_best_result(str(workspace)),
        workspace=str(workspace),
        base_commit=base_commit,
    )

    assert validated is not None
    assert validated["mean_case_speedup"] == 1.5
    assert validated["best_ms"] == 3.0


def test_manifest_naming_an_unknown_commit_is_rejected(repo):
    """A manifest left over from another workspace must not be trusted."""
    workspace, base_commit = repo
    _publish(workspace, _manifest("0" * 40))

    assert (
        forge_submit._validated_forge_best_result(
            forge_submit._read_forge_best_result(str(workspace)),
            workspace=str(workspace),
            base_commit=base_commit,
        )
        is None
    )


def test_manifest_off_the_base_lineage_is_rejected(repo):
    """A commit that does not descend from this run's base is stale evidence."""
    workspace, base_commit = repo
    _git(workspace, "checkout", "-q", "-b", "sidetrack", f"{base_commit}~0")
    _git(workspace, "commit", "-q", "--allow-empty", "-m", "unrelated")
    sidetrack = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "checkout", "-q", "-")
    # Re-root the run on a later commit so `sidetrack` is no longer a descendant.
    _git(workspace, "commit", "-q", "--allow-empty", "-m", "advance")
    advanced_base = _git(workspace, "rev-parse", "HEAD")
    _publish(workspace, _manifest(sidetrack))

    assert (
        forge_submit._validated_forge_best_result(
            forge_submit._read_forge_best_result(str(workspace)),
            workspace=str(workspace),
            base_commit=advanced_base,
        )
        is None
    )


def test_manifest_pointing_at_the_base_itself_is_not_a_win(repo):
    workspace, base_commit = repo
    _publish(workspace, _manifest(base_commit))

    assert (
        forge_submit._validated_forge_best_result(
            forge_submit._read_forge_best_result(str(workspace)),
            workspace=str(workspace),
            base_commit=base_commit,
        )
        is None
    )


def test_corrupt_manifest_is_ignored_rather_than_raising(repo):
    """A hard kill mid-write must degrade to "no evidence", not crash submit."""
    workspace, _base_commit = repo
    root = workspace / "forge_experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "best_result.json").write_text('{"schema_version": 1, "comm')

    assert forge_submit._read_forge_best_result(str(workspace)) is None


_APPLYBACK_REF = "refs/hyperloom/applyback/attempt-1"
_ARTIFACT_DIR = "forge_experiments/rewrite"


def _publish_applyback(
    workspace: Path,
    base_commit: str,
    *,
    manifest_overrides: dict | None = None,
    outer_overrides: dict | None = None,
    patch_body: str | None = None,
) -> dict:
    """Commit a framework apply-back and publish its canonical artifacts."""
    (workspace / "kernel.py").write_text("def kernel(x):\n    return flydsl_kernel(x)\n")
    (workspace / "flydsl_kernel.py").write_text("def flydsl_kernel(x):\n    return x\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "flydsl apply-back")
    best_commit = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "update-ref", _APPLYBACK_REF, best_commit)

    artifact_dir = workspace / _ARTIFACT_DIR
    (artifact_dir / "files").mkdir(parents=True, exist_ok=True)
    changed_files = ["flydsl_kernel.py", "kernel.py"]
    for relative in changed_files:
        (artifact_dir / "files" / relative).write_text((workspace / relative).read_text())
    patch = patch_body if patch_body is not None else _git(
        workspace, "diff", f"{base_commit}..{best_commit}"
    )
    (artifact_dir / "forge.patch").write_text(patch + "\n")

    manifest = {
        "schema_version": 2,
        "validation_scope": "reference",
        "reference_correctness_passed": True,
        "reference_snr_db": 48.5,
        "integration_validation_required": True,
        "integration_validation_status": "pending",
        "base_commit": base_commit,
        "commit_hash": best_commit,
        "commit_ref": _APPLYBACK_REF,
        "builder_symbol": "build_fused_gemm_module",
        "baseline_wall_ms": 2.0,
        "best_wall_ms": 1.25,
        "changed_files": changed_files,
    }
    manifest.update(manifest_overrides or {})
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    outer = {
        "success": True,
        "applyback_required": True,
        "applyback_ok": True,
        "artifact_kind": "framework_applyback",
        "artifact_schema_version": 2,
        "best_commit": best_commit,
        "canonical_manifest": f"{_ARTIFACT_DIR}/manifest.json",
        "canonical_patch_path": f"{_ARTIFACT_DIR}/forge.patch",
        "canonical_files_root": f"{_ARTIFACT_DIR}/files",
        "temporary_paths": [f"{_ARTIFACT_DIR}/scratch.py"],
        "logical_op_name": "vllm::fused_gemm",
        "source_entry": "matmul",
    }
    outer.update(outer_overrides or {})
    return outer


def test_canonical_applyback_is_accepted_with_both_documents_agreeing(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)

    validated = forge_submit._validated_rewrite_applyback_result(
        outer, workspace=str(workspace), base_commit=base_commit
    )

    assert validated is not None
    assert validated["best_commit"] == outer["best_commit"]
    assert validated["artifact_kind"] == "framework_applyback"
    assert validated["artifact_schema_version"] == 2
    assert validated["validation_scope"] == "reference"
    assert validated["integration_validation_status"] == "pending"
    assert validated["changed_files"] == ["flydsl_kernel.py", "kernel.py"]
    assert validated["builder_symbol"] == "build_fused_gemm_module"
    assert validated["baseline_ms"] == 2.0
    assert validated["best_ms"] == 1.25
    # Paths come back absolute and inside the workspace, ready to act on.
    assert validated["canonical_patch_path"].startswith(str(workspace))
    assert validated["temporary_paths"] == [str(workspace / _ARTIFACT_DIR / "scratch.py")]


def test_applyback_accepts_absolute_canonical_paths_inside_the_workspace(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    for key in ("canonical_manifest", "canonical_patch_path", "canonical_files_root"):
        outer[key] = str(workspace / outer[key])

    validated = forge_submit._validated_rewrite_applyback_result(
        outer, workspace=str(workspace), base_commit=base_commit
    )

    assert validated is not None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"success": False}, id="not_successful"),
        pytest.param({"applyback_ok": False}, id="applyback_failed"),
        pytest.param({"applyback_ok": None}, id="applyback_absent"),
        pytest.param({"artifact_kind": "standalone_best"}, id="wrong_artifact_kind"),
        pytest.param({"artifact_schema_version": 1}, id="schema_one_artifact"),
        pytest.param({"best_commit": ""}, id="unnamed_commit"),
        pytest.param({"canonical_manifest": ""}, id="no_manifest_path"),
        pytest.param({"canonical_patch_path": "../escape.patch"}, id="patch_escapes"),
        pytest.param({"canonical_files_root": "forge_experiments/missing"}, id="files_root_absent"),
    ],
)
def test_outer_result_that_breaks_the_contract_is_rejected(repo, overrides):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit, outer_overrides=overrides)

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"temporary_paths": None}, id="declaration_missing"),
        pytest.param({"temporary_paths": "scratch.py"}, id="declaration_not_a_list"),
        pytest.param({"temporary_paths": ["../outside.py"]}, id="declaration_escapes"),
        pytest.param({"temporary_paths": ["/etc/passwd"]}, id="declaration_absolute"),
        pytest.param({"temporary_paths": [""]}, id="declaration_empty_entry"),
    ],
)
def test_untrustworthy_temporary_path_declaration_fails_the_result(repo, overrides):
    """Reclaiming these paths deletes files, so a vague declaration is fatal."""
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit, outer_overrides=overrides)

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


def test_empty_temporary_path_declaration_is_accepted(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit, outer_overrides={"temporary_paths": []})

    validated = forge_submit._validated_rewrite_applyback_result(
        outer, workspace=str(workspace), base_commit=base_commit
    )

    assert validated is not None
    assert validated["temporary_paths"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"schema_version": 1}, id="schema_one_manifest"),
        pytest.param({"validation_scope": "integration"}, id="wrong_validation_scope"),
        pytest.param({"reference_correctness_passed": False}, id="reference_failed"),
        pytest.param({"reference_correctness_passed": None}, id="reference_unreported"),
        pytest.param({"integration_validation_required": False}, id="integration_not_required"),
        pytest.param({"integration_validation_status": "passed"}, id="integration_prematurely_passed"),
        pytest.param({"base_commit": "0" * 40}, id="base_commit_mismatch"),
        pytest.param({"best_wall_ms": 2.5}, id="no_micro_gain"),
        pytest.param({"best_wall_ms": 2.0}, id="micro_gain_is_a_tie"),
        pytest.param({"baseline_wall_ms": 0.0}, id="unusable_baseline"),
        pytest.param({"changed_files": []}, id="no_changed_files"),
        pytest.param({"changed_files": ["../outside.py"]}, id="changed_file_escapes"),
        pytest.param({"changed_files": ["kernel.py"]}, id="patch_touches_more_than_declared"),
        pytest.param({"commit_ref": ""}, id="no_pinned_ref"),
        pytest.param({"commit_ref": "refs/hyperloom/applyback/other"}, id="pinned_ref_absent"),
    ],
)
def test_manifest_that_breaks_the_contract_is_rejected(repo, overrides):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit, manifest_overrides=overrides)

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


def test_manifest_commit_disagreeing_with_the_outer_result_is_rejected(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    outer["best_commit"] = base_commit

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


def test_applyback_pinned_to_a_foreign_commit_is_rejected(repo):
    """The pinned ref must name the same commit the manifest claims."""
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    _git(workspace, "update-ref", _APPLYBACK_REF, base_commit)

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


def test_applyback_patch_disagreeing_with_changed_files_is_rejected(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(
        workspace,
        base_commit,
        patch_body="diff --git a/kernel.py b/kernel.py\n@@ -1 +1 @@\n-a\n+b\n",
    )

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


def test_applyback_off_the_base_lineage_is_rejected(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    _git(workspace, "commit", "-q", "--allow-empty", "-m", "advance")
    advanced_base = _git(workspace, "rev-parse", "HEAD")

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=advanced_base
        )
        is None
    )


def test_corrupt_applyback_manifest_is_ignored_rather_than_raising(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    (workspace / _ARTIFACT_DIR / "manifest.json").write_text('{"schema_version": 2, "comm')

    assert (
        forge_submit._validated_rewrite_applyback_result(
            outer, workspace=str(workspace), base_commit=base_commit
        )
        is None
    )


def test_nested_schema_one_best_result_is_never_consulted_for_an_applyback(repo):
    """A standalone forge-loop best must not stand in for a framework patch."""
    workspace, base_commit = repo
    best_commit = _commit_improvement(workspace)
    _publish(workspace, _manifest(best_commit))

    assert (
        forge_submit._validated_rewrite_applyback_result(
            forge_submit._read_forge_best_result(str(workspace)),
            workspace=str(workspace),
            base_commit=base_commit,
        )
        is None
    )
