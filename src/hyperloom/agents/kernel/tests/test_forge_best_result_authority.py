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
