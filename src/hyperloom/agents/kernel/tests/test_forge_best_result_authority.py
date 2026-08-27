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
        "schema_version": 2,
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
    assert forge_submit._validated_forge_best_result(None, workspace=str(workspace), base_commit=base_commit) is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"correctness_passed": False}, id="correctness_failed"),
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
    (root / "best_result.json").write_text('{"schema_version": 2, "comm')

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
    patch = patch_body if patch_body is not None else _git(workspace, "diff", f"{base_commit}..{best_commit}")
    (artifact_dir / "forge.patch").write_text(patch + "\n")

    manifest = {
        "schema_version": 2,
        "artifact_kind": "framework_applyback",
        "validation_scope": "reference",
        "logical_op_name": "vllm::fused_gemm",
        "operator_slug": "vllm_fused_gemm",
        "builder_symbol": "build_fused_gemm_module",
        "source_entry": "matmul",
        "reference_correctness_passed": True,
        "reference_snr_db": 48.5,
        "integration_validation_required": True,
        "integration_validation_status": "pending",
        "base_commit": base_commit,
        "commit_hash": best_commit,
        "commit_ref": _APPLYBACK_REF,
        "flydsl_best_commit": "f" * 40,
        "baseline_wall_ms": 2.0,
        "best_wall_ms": 1.25,
        "framework": "vllm",
        "changed_files": changed_files,
        "artifact_dir": "rewrite",
        "patch_path": "rewrite/forge.patch",
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


def test_a_renaming_patch_is_not_discarded_for_naming_its_source(repo):
    """The producer declares destinations; a rename header names both ends.

    ``git diff --name-only`` reports a rename as its destination alone, while the
    header reads ``diff --git a/<source> b/<destination>``. Counting the source
    too made the declared and parsed sets differ, so an artifact that had already
    run a whole campaign was thrown away. Add, modify and delete were unaffected,
    because there both ends name the same file.
    """
    workspace, base_commit = repo
    renaming_patch = (
        "diff --git a/kernel.py b/flydsl_kernel.py\n"
        "similarity index 100%\n"
        "rename from kernel.py\n"
        "rename to flydsl_kernel.py\n"
    )
    outer = _publish_applyback(
        workspace,
        base_commit,
        patch_body=renaming_patch,
        manifest_overrides={"changed_files": ["flydsl_kernel.py"]},
    )
    problems: list[str] = []

    validated = forge_submit._validated_rewrite_applyback_result(
        outer,
        workspace=str(workspace),
        base_commit=base_commit,
        problems=problems,
    )

    assert validated is not None, problems
    assert validated["changed_files"] == ["flydsl_kernel.py"]


def test_a_refused_applyback_names_the_clause_that_refused_it(repo):
    """Forty refusals used to reach an operator as one sentence.

    A campaign that spent an hour reported only that it produced nothing, which
    made every other failure in this route harder to place than it needed to be.
    """
    workspace, base_commit = repo
    outer = _publish_applyback(
        workspace,
        base_commit,
        manifest_overrides={"framework": "torch"},
    )
    problems: list[str] = []

    validated = forge_submit._validated_rewrite_applyback_result(
        outer,
        workspace=str(workspace),
        base_commit=base_commit,
        problems=problems,
    )

    assert validated is None
    assert len(problems) == 1
    assert "framework" in problems[0]
    assert "torch" in problems[0]


def test_installed_producer_contract_is_consumed_without_a_local_fixture(repo):
    """Materialize the real producer documents and pass them through this consumer.

    This used to skip unless ``$FORGE_PATH`` named a checkout, which meant the
    one test pinning both halves of the contract against each other never ran
    anywhere. KernelForge ships in this distribution, so the producer is always
    present and the test always runs.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kernelforge.cli",
            "forge-rewrite-by-flydsl",
            "--applyback-contract-json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    contract = json.loads(proc.stdout)
    manifest = dict(contract["manifest"])
    outer = dict(contract["outer_result"])

    workspace, base_commit = repo
    changed_files = list(manifest["changed_files"])
    for relative in changed_files:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def example(x):\n    return x\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "producer contract apply-back")
    best_commit = _git(workspace, "rev-parse", "HEAD")
    commit_ref = "refs/forge-rewrite/applyback/contract-example"
    _git(workspace, "update-ref", commit_ref, best_commit)

    manifest.update(
        {
            "base_commit": base_commit,
            "commit_hash": best_commit,
            "commit_ref": commit_ref,
        }
    )
    outer["best_commit"] = best_commit
    campaign = workspace / "forge_experiments"
    version = campaign / manifest["artifact_dir"]
    files_root = workspace / outer["canonical_files_root"]
    files_root.mkdir(parents=True, exist_ok=True)
    assert files_root == version / "files"
    for relative in changed_files:
        destination = files_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((workspace / relative).read_text())
    patch_path = workspace / outer["canonical_patch_path"]
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(_git(workspace, "diff", f"{base_commit}..{best_commit}") + "\n")
    manifest_path = workspace / outer["canonical_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validated = forge_submit._validated_rewrite_applyback_result(
        outer,
        workspace=str(workspace),
        base_commit=base_commit,
    )

    assert validated is not None
    assert validated["framework"] == manifest["framework"]
    assert validated["logical_op_name"] == manifest["logical_op_name"]
    assert validated["source_entry"] == manifest["source_entry"]


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
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
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
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
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
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
        is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"best_wall_ms": 2.5}, id="slower_than_the_source"),
        pytest.param({"best_wall_ms": 2.0}, id="tied_with_the_source"),
    ],
)
def test_an_apply_back_that_is_not_faster_stays_contract_valid(repo, overrides):
    """Being faster is consumer policy, not part of the producer contract.

    The producer may publish a correct-but-not-faster port, so this validator
    must still describe it. ``_run_rewrite_attempt`` is what declines it, with a
    reason that says "not faster" rather than "malformed artifact".
    """
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit, manifest_overrides=overrides)

    validated = forge_submit._validated_rewrite_applyback_result(
        outer, workspace=str(workspace), base_commit=base_commit
    )

    assert validated is not None
    assert validated["best_ms"] >= validated["baseline_ms"]
    assert validated["integration_validation_status"] == "pending"


def test_manifest_commit_disagreeing_with_the_outer_result_is_rejected(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    outer["best_commit"] = base_commit

    assert (
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
        is None
    )


def test_applyback_pinned_to_a_foreign_commit_is_rejected(repo):
    """The pinned ref must name the same commit the manifest claims."""
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    _git(workspace, "update-ref", _APPLYBACK_REF, base_commit)

    assert (
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
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
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
        is None
    )


def test_applyback_off_the_base_lineage_is_rejected(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    _git(workspace, "commit", "-q", "--allow-empty", "-m", "advance")
    advanced_base = _git(workspace, "rev-parse", "HEAD")

    assert (
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=advanced_base)
        is None
    )


def test_corrupt_applyback_manifest_is_ignored_rather_than_raising(repo):
    workspace, base_commit = repo
    outer = _publish_applyback(workspace, base_commit)
    (workspace / _ARTIFACT_DIR / "manifest.json").write_text('{"schema_version": 2, "comm')

    assert (
        forge_submit._validated_rewrite_applyback_result(outer, workspace=str(workspace), base_commit=base_commit)
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


@pytest.mark.parametrize("version", [1, 2, 3, None])
def test_a_schema_bump_alone_does_not_discard_a_proven_best(repo, version):
    """What actually broke: the producer went to 2, this stayed on 1, and every
    published best was dropped for six days. The evidence is judged on its own
    fields, so the version it is stamped with cannot decide the question."""
    workspace, base_commit = repo
    best_commit = _commit_improvement(workspace)
    _publish(workspace, _manifest(best_commit, schema_version=version))

    validated = forge_submit._validated_forge_best_result(
        forge_submit._read_forge_best_result(str(workspace)),
        workspace=str(workspace),
        base_commit=base_commit,
    )

    assert validated is not None
    assert validated["best_commit"] == best_commit
