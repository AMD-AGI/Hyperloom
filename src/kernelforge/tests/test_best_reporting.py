"""Tests for incremental publication of the best verified Forge result."""

from __future__ import annotations

import json

import pytest

from kernelforge.loop import reporting
from kernelforge.loop.reporting import (
    MANIFEST_SCHEMA_VERSION,
    BestResultPublisher,
)


def _publish(
    publisher: BestResultPublisher,
    *,
    iteration: int,
    wall_ms: float,
    plan: str,
    changed_files: list[str],
    mean_case_speedup: float = 2.0,
):
    return publisher.publish(
        campaign_id="campaign-1",
        session_index=1,
        experiment_id="experiment-1",
        iteration=iteration,
        commit_hash=f"commit-{iteration}",
        plan=plan,
        baseline_wall_ms=1.0,
        best_wall_ms=wall_ms,
        mean_case_speedup=mean_case_speedup,
        search_start_mean_case_speedup=1.0,
        snr_db=40.0,
        validation_text="canonical correctness passed",
        benchmark={"median_ms": wall_ms},
        changed_files=changed_files,
        patch=f"patch for iteration {iteration}\n",
    )


def test_each_keep_publishes_versioned_bundle_and_best_only_report(tmp_path):
    kernel = tmp_path / "src" / "kernel.py"
    helper = tmp_path / "src" / "helper.py"
    kernel.parent.mkdir()
    kernel.write_text("iteration one\n")
    helper.write_text("helper one\n")
    publisher = BestResultPublisher(str(tmp_path))

    first = _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="first verified improvement",
        changed_files=["src/kernel.py"],
    )
    kernel.write_text("iteration two\n")
    helper.write_text("helper two\n")
    second = _publish(
        publisher,
        iteration=2,
        wall_ms=0.8,
        plan="second verified improvement",
        changed_files=["src/kernel.py", "src/helper.py"],
    )

    root = tmp_path / "forge_experiments"
    manifest = json.loads((root / "best" / "manifest.json").read_text())
    report = (root / "optimization_report.md").read_text()
    result = json.loads((root / "best_result.json").read_text())

    assert first["iteration"] == 1
    assert second["iteration"] == 2
    assert manifest["iteration"] == 2
    assert manifest["best_wall_ms"] == 0.8
    assert manifest["speedup"] == 2.0
    assert manifest["search_start_mean_case_speedup"] == 1.0
    assert manifest["pristine_baseline_ms"] == 1.0
    assert manifest["search_start_ms"] == 1.0
    assert manifest["total_improved"] is True
    assert manifest["incremental_improved"] is True
    assert manifest["improved_during_search"] is True
    assert result == manifest
    assert "second verified improvement" in report
    assert "first verified improvement" not in report
    assert "0.8000 ms" in report


def test_report_marks_raw_wall_as_non_monotonic_diagnostic(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))

    manifest = _publish(
        publisher,
        iteration=1,
        wall_ms=0.95,
        mean_case_speedup=2.0,
        plan="improve equal-weight case score",
        changed_files=["kernel.py"],
    )
    report = (tmp_path / "forge_experiments" / "optimization_report.md").read_text()

    assert manifest["best_wall_ms"] == 0.95
    assert manifest["total_improved"] is True
    assert manifest["aggregate_regression"] == ""
    assert (
        "- Selected candidate raw mean (diagnostic; not monotonic, but it "
        "withdraws the improvement above when it contradicts the score): "
        "0.9500 ms"
    ) in report


def test_manifest_withholds_improvement_when_slower_than_baseline(tmp_path):
    """The score can rise while the aggregate wall time regresses.

    Five landed runs shipped a PASS badge that way. The manifest is what
    downstream reporting reads, so the contradiction has to be named here and
    not only in the CLI result.
    """
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))

    manifest = _publish(
        publisher,
        iteration=1,
        wall_ms=1.2,
        mean_case_speedup=2.0,
        plan="improve equal-weight case score",
        changed_files=["kernel.py"],
    )

    assert manifest["best_wall_ms"] == 1.2
    assert manifest["total_improved"] is False
    assert "is not faster than the pristine baseline" in manifest["aggregate_regression"]


def test_report_names_the_contradiction_the_manifest_withheld_the_badge_for(
    tmp_path,
):
    """optimization_report.md is the artifact a human actually opens.

    Both files are written by the same publish() call two lines apart, but the
    report listed the score, both wall times and a PASS and said nothing about
    the manifest having withdrawn the improvement over exactly those numbers.
    """
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))

    manifest = _publish(
        publisher,
        iteration=1,
        wall_ms=1.2,
        mean_case_speedup=2.0,
        plan="improve equal-weight case score",
        changed_files=["kernel.py"],
    )
    report = (tmp_path / "forge_experiments" / "optimization_report.md").read_text()

    assert "- Improved overall: no" in report
    assert f"- Aggregate regression: {manifest['aggregate_regression']}" in report


def test_a_consistent_report_states_the_improvement_without_a_regression(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))

    _publish(
        publisher,
        iteration=1,
        wall_ms=0.5,
        mean_case_speedup=2.0,
        plan="improve equal-weight case score",
        changed_files=["kernel.py"],
    )
    report = (tmp_path / "forge_experiments" / "optimization_report.md").read_text()

    assert "- Improved overall: yes" in report
    assert "Aggregate regression" not in report


def _downgrade_to_pre_badge_schema(tmp_path) -> None:
    """Rewrite a published bundle the way the workspace looked before b9825da.

    That release had no ``aggregate_regression`` key and left ``total_improved``
    derived from the score alone, so an upgraded binary republishing the same
    iteration meets a manifest whose field set it never wrote.
    """
    root = tmp_path / "forge_experiments"
    for path in (
        root / "best" / "manifest.json",
        root / "best_result.json",
        root / "best" / "iter_001" / "publication.json",
    ):
        payload = json.loads(path.read_text())
        payload.pop("aggregate_regression", None)
        payload["schema_version"] = 1
        payload["total_improved"] = payload["mean_case_speedup"] > 1.0
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_republish_over_a_pre_badge_manifest_supersedes_it(tmp_path):
    """The stale manifest is the published artifact until it is replaced.

    ``_validate_existing_bundle`` compares whole dicts, so a manifest missing a
    key the current schema writes reads as a conflicting publication of the same
    iteration. The raise is swallowed upstream as persistence_degraded, which
    leaves the pre-upgrade manifest -- and its ``total_improved: true`` over a
    slower candidate -- as the campaign's published result.
    """
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))
    _publish(
        publisher,
        iteration=1,
        wall_ms=1.2,
        mean_case_speedup=2.0,
        plan="improve equal-weight case score",
        changed_files=["kernel.py"],
    )
    _downgrade_to_pre_badge_schema(tmp_path)
    stale = json.loads((tmp_path / "forge_experiments" / "best" / "manifest.json").read_text())

    republished = _publish(
        publisher,
        iteration=1,
        wall_ms=1.2,
        mean_case_speedup=2.0,
        plan="improve equal-weight case score",
        changed_files=["kernel.py"],
    )

    published = json.loads((tmp_path / "forge_experiments" / "best" / "manifest.json").read_text())
    assert stale["total_improved"] is True
    assert published == republished
    assert published["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert published["total_improved"] is False
    assert "is not faster than the pristine baseline" in (published["aggregate_regression"])


def test_a_conflicting_publication_of_the_same_schema_still_raises(tmp_path):
    """Superseding an old schema must not turn every conflict into a rewrite."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))
    _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="first",
        changed_files=["kernel.py"],
    )
    manifest_path = tmp_path / "forge_experiments" / "best" / "manifest.json"
    diverged = json.loads(manifest_path.read_text())
    diverged["commit_hash"] = "a-different-commit"
    manifest_path.write_text(json.dumps(diverged, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="conflicts with iteration 1"):
        _publish(
            publisher,
            iteration=1,
            wall_ms=0.9,
            plan="first",
            changed_files=["kernel.py"],
        )


def test_describes_current_best_recognizes_a_complete_matching_bundle(tmp_path):
    """Reconciliation has nothing to repair when the manifest is already current.

    A resumed session rebuilds the durable best's manifest and republishes it,
    and fields it recomputes -- session_index and experiment_id among them --
    legitimately differ from the stored one, so republishing an already-current
    best raised a conflict that harmed nothing.
    """
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))
    _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="first",
        changed_files=["kernel.py"],
    )

    assert publisher.describes_current_best(iteration=1, commit_hash="commit-1")
    assert not publisher.describes_current_best(iteration=1, commit_hash="commit-2")
    assert not publisher.describes_current_best(iteration=2, commit_hash="commit-1")


def test_describes_current_best_is_false_when_manifest_or_bundle_is_missing(tmp_path):
    """A missing manifest or a partial bundle is exactly what reconcile repairs."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("selected candidate\n")
    publisher = BestResultPublisher(str(tmp_path))

    assert not publisher.describes_current_best(iteration=1, commit_hash="commit-1")

    _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="first",
        changed_files=["kernel.py"],
    )
    (tmp_path / "forge_experiments" / "best" / "iter_001" / "benchmark.json").unlink()
    assert not publisher.describes_current_best(iteration=1, commit_hash="commit-1")


def test_failed_manifest_replace_preserves_previous_best(tmp_path, monkeypatch):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("first\n")
    publisher = BestResultPublisher(str(tmp_path))
    _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="first",
        changed_files=["kernel.py"],
    )
    manifest_path = tmp_path / "forge_experiments" / "best" / "manifest.json"
    before = manifest_path.read_bytes()
    original = reporting.atomic_write_text

    def fail_manifest(path, text):
        if path == manifest_path:
            raise OSError("simulated manifest failure")
        return original(path, text)

    monkeypatch.setattr(reporting, "atomic_write_text", fail_manifest)
    kernel.write_text("second\n")

    with pytest.raises(OSError, match="simulated manifest failure"):
        _publish(
            publisher,
            iteration=2,
            wall_ms=0.8,
            plan="second",
            changed_files=["kernel.py"],
        )

    assert manifest_path.read_bytes() == before
    (tmp_path / "forge_experiments" / "best" / "iter_002" / "publication.json").unlink()
    monkeypatch.setattr(reporting, "atomic_write_text", original)

    recovered = _publish(
        publisher,
        iteration=2,
        wall_ms=0.8,
        plan="second",
        changed_files=["kernel.py"],
    )

    assert recovered["iteration"] == 2
    assert json.loads(manifest_path.read_text()) == recovered
    assert json.loads((tmp_path / "forge_experiments" / "best_result.json").read_text()) == recovered


def test_retry_repairs_partial_derived_best_views(tmp_path, monkeypatch):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("verified\n")
    publisher = BestResultPublisher(str(tmp_path))
    result_path = tmp_path / "forge_experiments" / "best_result.json"
    original = reporting.atomic_write_text

    def fail_result(path, text):
        if path == result_path:
            raise OSError("simulated result failure")
        return original(path, text)

    monkeypatch.setattr(reporting, "atomic_write_text", fail_result)
    with pytest.raises(OSError, match="simulated result failure"):
        _publish(
            publisher,
            iteration=1,
            wall_ms=0.9,
            plan="verified candidate",
            changed_files=["kernel.py"],
        )

    monkeypatch.setattr(reporting, "atomic_write_text", original)
    recovered = _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="verified candidate",
        changed_files=["kernel.py"],
    )

    assert json.loads(result_path.read_text()) == recovered
    assert "verified candidate" in (tmp_path / "forge_experiments" / "optimization_report.md").read_text()


def test_retry_repairs_incomplete_orphan_bundle(tmp_path, monkeypatch):
    """A crash between os.replace and manifest write can leave version_dir
    visible but truncated. Retry must quarantine the corrupt bundle and rewrite
    it (repairable), not wedge the iteration on a hard 'incomplete' error."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("verified\n")
    publisher = BestResultPublisher(str(tmp_path))
    best_root = tmp_path / "forge_experiments" / "best"
    manifest_path = best_root / "manifest.json"
    original = reporting.atomic_write_text

    def fail_manifest(path, text):
        if path == manifest_path:
            raise OSError("simulated manifest failure")
        return original(path, text)

    monkeypatch.setattr(reporting, "atomic_write_text", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest failure"):
        _publish(
            publisher,
            iteration=1,
            wall_ms=0.9,
            plan="verified candidate",
            changed_files=["kernel.py"],
        )

    # Simulate the post-crash truncation: a visible but incomplete bundle.
    (best_root / "iter_001" / "validation.txt").unlink()
    monkeypatch.setattr(reporting, "atomic_write_text", original)

    manifest = _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="verified candidate",
        changed_files=["kernel.py"],
    )

    # The retry published a new immutable generation rather than raising.
    version_dir = best_root.parent / manifest["artifact_dir"]
    assert (version_dir / "validation.txt").read_text() == "canonical correctness passed"
    assert (version_dir / "forge.patch").read_text() == "patch for iteration 1\n"
    assert manifest_path.is_file()
    assert manifest["patch_path"].endswith("forge.patch")


def test_retry_repairs_inconsistent_orphan_bundle(tmp_path, monkeypatch):
    """A visible-but-inconsistent orphan bundle (wrong patch bytes) is treated
    as repairable: quarantine + rewrite, not a hard 'inconsistent' error."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("verified\n")
    publisher = BestResultPublisher(str(tmp_path))
    best_root = tmp_path / "forge_experiments" / "best"
    manifest_path = best_root / "manifest.json"
    original = reporting.atomic_write_text

    def fail_manifest(path, text):
        if path == manifest_path:
            raise OSError("simulated manifest failure")
        return original(path, text)

    monkeypatch.setattr(reporting, "atomic_write_text", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest failure"):
        _publish(
            publisher,
            iteration=1,
            wall_ms=0.9,
            plan="verified candidate",
            changed_files=["kernel.py"],
        )

    (best_root / "iter_001" / "forge.patch").write_text("different patch\n")
    monkeypatch.setattr(reporting, "atomic_write_text", original)

    manifest = _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="verified candidate",
        changed_files=["kernel.py"],
    )

    version_dir = best_root.parent / manifest["artifact_dir"]
    assert (version_dir / "forge.patch").read_text() == "patch for iteration 1\n"
    assert manifest_path.is_file()
    assert manifest["patch_path"].endswith("forge.patch")


def test_history_view_contains_every_attempt_while_final_report_stays_best_only(
    tmp_path,
):
    publisher = BestResultPublisher(str(tmp_path))
    events = [
        {
            "type": "iteration_result",
            "iter": 1,
            "decision": "KEEP",
            "plan": "vectorize loads",
            "wall_ms": 0.9,
            "session_index": 1,
            "experiment_id": "exp-1",
            "session_end_reason": "candidate_submitted",
            "turns": 20,
        },
        {
            "type": "iteration_result",
            "iter": 2,
            "decision": "REVERT_PERF",
            "plan": "increase tile",
            "wall_ms": 0.95,
            "session_index": 1,
            "experiment_id": "exp-1",
            "session_end_reason": "turn_cap",
            "turns": 100,
        },
        {
            "type": "iteration_result",
            "iter": 3,
            "decision": "NO_CHANGES",
            "plan": "inspect only",
            "session_index": 2,
            "experiment_id": "exp-2",
            "session_end_reason": "agent_stopped",
            "turns": 4,
        },
    ]
    metadata = {
        1: {
            "validation_passed": True,
            "commit_hash": "best-commit",
            "best_wall_ms_before": 1.0,
            "changed_files": ["kernel.py"],
            "dir": "iter_001",
        },
        2: {
            "validation_passed": True,
            "commit_hash": "",
            "best_wall_ms_before": 0.9,
            "changed_files": ["kernel.py"],
            "dir": "iter_002",
        },
    }

    publisher.publish_history(events=events, candidate_metadata=metadata)

    history = (tmp_path / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — KEEP" in history
    assert "Iteration 2 — REVERT_PERF" in history
    assert "Iteration 3 — NO_CHANGES" in history
    assert "vectorize loads" in history
    assert "increase tile" in history
    assert "turn_cap" in history


def test_published_manifest_bytes_stay_indented_sorted_and_newline_terminated(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("verified\n")
    publisher = BestResultPublisher(str(tmp_path))
    _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="verified candidate",
        changed_files=["kernel.py"],
    )

    for path in (
        tmp_path / "forge_experiments" / "best" / "manifest.json",
        tmp_path / "forge_experiments" / "best_result.json",
    ):
        raw = path.read_bytes()
        rendered = json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
        assert raw == rendered.encode("utf-8")


def test_the_round_budget_restatement_rewrites_a_published_best_in_place(tmp_path):
    """Nothing else exercises this republish, so nothing else notices it break."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("verified\n")
    publisher = BestResultPublisher(str(tmp_path))
    _publish(
        publisher,
        iteration=1,
        wall_ms=0.9,
        plan="verified candidate",
        changed_files=["kernel.py"],
    )
    manifest_path = tmp_path / "forge_experiments" / "best" / "manifest.json"
    before = json.loads(manifest_path.read_text())
    assert "round_budget" not in before

    assert publisher.refresh_round_budget({"rounds": 3, "spent_minutes": 42.0}) is True

    after = json.loads(manifest_path.read_text())
    assert after["round_budget"] == {"rounds": 3, "spent_minutes": 42.0}
    assert after["commit_hash"] == before["commit_hash"]
    # Restated in place, so it stays the shape its readers parse.
    raw = manifest_path.read_bytes()
    assert raw == (json.dumps(after, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert json.loads((tmp_path / "forge_experiments" / "best_result.json").read_text()) == after


def test_the_restatement_declines_when_there_is_no_published_best(tmp_path):
    publisher = BestResultPublisher(str(tmp_path))

    assert publisher.refresh_round_budget({"rounds": 1}) is False
