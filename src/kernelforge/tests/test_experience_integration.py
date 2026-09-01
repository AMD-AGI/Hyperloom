"""Tests for forge-loop experience KB integration helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.knowledge import experience_integration as integ
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_KB_STORE,
    KnowledgeConfig,
)
from kernelforge.knowledge.implementation_identity import implementation_signature
from kernelforge.rewrite_by_flydsl import driver_contract, record_store, runner
from kernelforge.rewrite_by_flydsl.flydsl_rewrite_driver_preparation import (
    DriverPreflight,
)
from kernelforge.rewrite_by_flydsl.port_loop import PortResult

#: The cap :func:`sanitize_read_error` bounds a persisted store error at.
MAX_READ_ERROR_LENGTH = 240


APPLICABLE_PATCH = """diff --git a/kernel.py b/kernel.py
--- a/kernel.py
+++ b/kernel.py
@@ -1 +1 @@
-old
+new
"""

BAD_PATCH = """diff --git a/kernel.py b/kernel.py
--- a/kernel.py
+++ b/kernel.py
@@ -1 +1 @@
-missing
+new
"""

NEW_FILE_PATCH = """diff --git a/helper.py b/helper.py
new file mode 100644
--- /dev/null
+++ b/helper.py
@@ -0,0 +1 @@
+helper
"""

OUTSIDE_PATCH = """diff --git a/helper.py b/helper.py
--- a/helper.py
+++ b/helper.py
@@ -1 +1 @@
-old
+new
"""


def _run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "kernel.py").write_text("old\n")
    _run(["git", "add", "kernel.py"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


def _solution(patch: str, **overrides) -> dict:
    sol = {
        "solution_slug": "kernelforge-exp/kernel/prev",
        "speedup": 1.5,
        "patch_content": patch,
        "strategy": "vectorize loads",
        "recipe": "Apply vectorized loads.",
        "lessons": "Alignment matters.",
        "match_mode": "exact",
        "implementation_match": True,
        "implementation_signature": "producer-signature",
        "consumer_implementation_signature": "consumer-signature",
        "implementation_identity": {"source_paths": ["kernel.py"]},
        "consumer_implementation_identity": {"source_paths": ["kernel.py"]},
        "consumer_source_map": {"kernel.py": "kernel.py"},
    }
    sol.update(overrides)
    return sol


def _patch_read_solutions(monkeypatch, *sols: dict):
    """Patch the top-k reader to return the given ranked solution list."""
    monkeypatch.setattr(
        "kernelforge.knowledge.experience_reader.read_top_solutions",
        lambda **_kwargs: [dict(s) for s in sols],
    )


def _patch_read_solution(monkeypatch, patch: str):
    _patch_read_solutions(monkeypatch, _solution(patch))


def _bench(ms: float) -> dict:
    return {
        "success": True,
        "median_ms": ms,
        "case_times": {"case-1": ms},
    }


def _three_measurements(*stages: dict | None):
    return iter(measurement for stage in stages for measurement in [stage, stage, stage])


def _indexed_reference(root: Path, rank: int) -> Path:
    line = next(line for line in (root / "index.md").read_text().splitlines() if line.startswith(f"- Rank {rank}:"))
    return root / line.split("`", 2)[1]


def test_git_apply_normalizes_strip_depth_for_deeper_workspace(tmp_path):
    # Producer recorded the diff relative to a repo root ('pkg/kernel.py'), but
    # the consumer workspace root sits one level deeper (inside 'pkg/'), so the
    # file is just 'kernel.py' here. -p1 would miss; the normalizer must find the
    # right strip depth and apply cleanly.
    repo = _init_repo(tmp_path)  # repo/kernel.py == "old\n"
    deep_patch = """diff --git a/pkg/kernel.py b/pkg/kernel.py
--- a/pkg/kernel.py
+++ b/pkg/kernel.py
@@ -1 +1 @@
-old
+new
"""
    assert integ._git_apply(str(repo), deep_patch, check_only=True) is True
    assert integ._git_apply(str(repo), deep_patch) is True
    assert (repo / "kernel.py").read_text() == "new\n"


def test_git_apply_rewrites_matching_canonical_owner_paths(tmp_path):
    repo = _init_repo(tmp_path)
    consumer = repo / "src" / "aiter" / "ops" / "kernel.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text("old\n")
    _run(["git", "add", str(consumer.relative_to(repo))], repo)
    _run(["git", "commit", "-m", "consumer layout"], repo)
    producer_patch = """diff --git a/packages/src/aiter_meta/ops/kernel.py b/packages/src/aiter_meta/ops/kernel.py
--- a/packages/src/aiter_meta/ops/kernel.py
+++ b/packages/src/aiter_meta/ops/kernel.py
@@ -1 +1 @@
-old
+new
"""
    allowed = {"src/aiter/ops/kernel.py"}
    source_map = {"aiter/ops/kernel.py": "src/aiter/ops/kernel.py"}

    assert integ._git_apply(
        str(repo),
        producer_patch,
        check_only=True,
        allowed_paths=allowed,
        canonical_source_paths={"aiter/ops/kernel.py"},
        consumer_source_map=source_map,
    )
    assert integ._git_apply(
        str(repo),
        producer_patch,
        allowed_paths=allowed,
        canonical_source_paths={"aiter/ops/kernel.py"},
        consumer_source_map=source_map,
    )
    assert consumer.read_text() == "new\n"


def test_kb_warmstart_uses_canonical_path_mapping_end_to_end(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    consumer = repo / "src" / "aiter" / "ops" / "kernel.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text("old\n")
    _run(["git", "add", str(consumer.relative_to(repo))], repo)
    _run(["git", "commit", "-m", "consumer layout"], repo)
    producer_patch = """diff --git a/packages/src/aiter_meta/ops/kernel.py b/packages/src/aiter_meta/ops/kernel.py
--- a/packages/src/aiter_meta/ops/kernel.py
+++ b/packages/src/aiter_meta/ops/kernel.py
@@ -1 +1 @@
-old
+new
"""
    _patch_read_solutions(
        monkeypatch,
        _solution(
            producer_patch,
            implementation_identity={"source_paths": ["aiter/ops/kernel.py"]},
            consumer_implementation_identity={
                "source_paths": ["aiter/ops/kernel.py"],
            },
            consumer_source_map={
                "aiter/ops/kernel.py": "src/aiter/ops/kernel.py",
            },
        ),
    )
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_a, **_k: next(benches),
    )
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(consumer),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        framework="aiter",
    )

    assert warm["applied"] is True
    assert warm["read_reason"] == "hit"
    assert warm["read_error"] == ""
    assert consumer.read_text() == "new\n"


def test_git_checkout_branch_creates_and_switches_branch(tmp_path):
    repo = _init_repo(tmp_path)

    out = integ.git_checkout_branch(str(repo), "kernel-agent-optimize")
    assert "Switched to a new branch" in out
    assert _run(["git", "branch", "--show-current"], repo) == "kernel-agent-optimize"

    _run(["git", "checkout", "master"], repo)
    out = integ.git_checkout_branch(str(repo), "kernel-agent-optimize")
    assert "Switched to branch" in out
    assert _run(["git", "branch", "--show-current"], repo) == "kernel-agent-optimize"


def test_kb_read_status_is_compact_and_stable():
    status = integ.kb_read_status(
        {
            "candidate": True,
            "read_reason": "hit",
            "read_error": "",
            "applied": True,
            "match_mode": "",
            "reference_reason": "",
            "solution_slug": "solution",
            "speedup": 1.5,
            "pristine_ms": 10.0,
            "keep_baseline_ms": 5.0,
            "applied_commit": "",
            "program_md_addition": "large prompt text",
        }
    )

    assert status == {
        "measured_writebacks": 0,
        "measured_writeback_failures": [],
        "candidate": True,
        "read_reason": "hit",
        "read_error": "",
        "applied": True,
        "match_mode": "",
        "reference_reason": "",
        "solution_slug": "solution",
        "speedup": 1.5,
        "pristine_ms": 10.0,
        "keep_baseline_ms": 5.0,
        "applied_commit": "",
    }


def test_kb_read_status_keeps_a_refused_amendment():
    """A store that refused a correction must reach the persisted record.

    The KB goes on ranking a claim no consumer reproduced, so the refusal is
    exactly the outcome an operator needs to see later.
    """
    status = integ.kb_read_status(
        {
            "candidate": True,
            "applied": True,
            "measured_writebacks": [
                {"rank": 1, "recorded": True, "reason": ""},
                {"rank": 2, "recorded": False, "reason": "error:KBStoreError:refused"},
            ],
        }
    )

    assert status["measured_writebacks"] == 2
    assert status["measured_writeback_failures"] == ["error:KBStoreError:refused"]


def test_kb_warmstart_cold_starts_without_candidate(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(
        "kernelforge.knowledge.experience_reader.read_top_solutions",
        lambda **_kwargs: [],
    )

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm == {
        "candidate": False,
        "read_reason": "solution_pages_missing",
        "read_error": "",
    }


def test_kb_warmstart_supports_legacy_monkeypatched_reader_signature(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)

    def legacy_reader(
        *,
        config,
        kernel_path,
        kernel_source,
        kernel_backend,
        target_functions=None,
        framework="",
        top_k=3,
        source_files=None,
        workspace="",
        operator_name="",
    ):
        return []

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_reader.read_top_solutions",
        legacy_reader,
    )

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm == {
        "candidate": False,
        "read_reason": "solution_pages_missing",
        "read_error": "",
    }


def test_kb_warmstart_propagates_reader_no_hit_status(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)

    def no_hit(**kwargs):
        kwargs["read_status"].update(
            {
                "read_reason": "kernel_page_not_found",
                "read_error": "",
            }
        )
        return []

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_reader.read_top_solutions",
        no_hit,
    )

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm == {
        "candidate": False,
        "read_reason": "kernel_page_not_found",
        "read_error": "",
    }


@pytest.mark.parametrize("lookup_mode", ["empty", "error"])
def test_fresh_kb_lookup_clears_stale_references(
    monkeypatch,
    tmp_path,
    lookup_mode,
):
    repo = _init_repo(tmp_path)
    root = repo / "forge_experiments" / "kb_references"
    stale = root / "sets" / "stale-generation" / "reference_01.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale reference\n")
    (root / "index.md").write_text(
        "- Rank 1: `sets/stale-generation/reference_01.md` | solution `stale` | speedup 2x | status `applied`\n"
    )

    def lookup(**_kwargs):
        if lookup_mode == "error":
            raise RuntimeError("lookup failed")
        return []

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_reader.read_top_solutions",
        lookup,
    )
    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    if lookup_mode == "error":
        assert warm["read_reason"] == "warm_start_error"
        assert warm["read_error"] == "RuntimeError: lookup failed"
    else:
        assert warm == {
            "candidate": False,
            "read_reason": "solution_pages_missing",
            "read_error": "",
        }
    assert not root.exists()
    assert (repo / "kernel.py").read_text() == "old\n"


def test_clear_kb_references_does_not_follow_root_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    experiments = workspace / "forge_experiments"
    experiments.mkdir(parents=True)
    external = tmp_path / "external-references"
    external.mkdir()
    (external / "index.md").write_text("preserve\n")
    (experiments / "kb_references").symlink_to(external, target_is_directory=True)

    integ._clear_kb_references(str(workspace))

    assert not (experiments / "kb_references").exists()
    assert (external / "index.md").read_text() == "preserve\n"


def test_kb_warmstart_applies_ranked_solution_when_safe(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    _patch_read_solutions(
        monkeypatch,
        _solution(
            APPLICABLE_PATCH,
            solution_slug="kernelforge-exp/kernel/decode",
            speedup=5.0,
            match_mode="reference",
        ),
    )
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["candidate"] is True
    assert warm["applied"] is True
    assert warm["keep_baseline_ms"] == 5.0
    assert (repo / "kernel.py").read_text() == "new\n"


def test_kb_warmstart_tries_next_candidate_when_first_ranked_fails(
    monkeypatch,
    tmp_path,
):
    # The first-ranked candidate fails to apply, so the loop must fall through
    # to the next-ranked candidate and adopt it if it is safe.
    repo = _init_repo(tmp_path)
    _patch_read_solutions(
        monkeypatch,
        _solution(BAD_PATCH, solution_slug="kernelforge-exp/kernel/first"),
        _solution(APPLICABLE_PATCH, solution_slug="kernelforge-exp/kernel/second"),
    )
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is True
    assert warm["solution_slug"] == "kernelforge-exp/kernel/second"
    assert (repo / "kernel.py").read_text() == "new\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == ("kb warm-start: apply kernelforge-exp/kernel/second")
    index = (repo / "forge_experiments" / "kb_references" / "index.md").read_text()
    assert "Rank 1:" in index
    assert "rejected:patch_touches_protected_path_or_not_applicable" in index
    assert "Rank 2:" in index
    assert "status `applied`" in index


def test_kb_warmstart_persists_all_available_references_without_truncation(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    long_patch = APPLICABLE_PATCH + ("# complete patch payload\n" * 600)
    _patch_read_solutions(
        monkeypatch,
        _solution(long_patch, solution_slug="solution/fast", speedup=4.0),
        _solution(BAD_PATCH, solution_slug="solution/second", speedup=2.0),
    )
    monkeypatch.setattr(integ, "_bench_once", lambda *_args: None)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    root = repo / "forge_experiments" / "kb_references"
    assert warm["num_references"] == 2
    assert (root / "index.md").is_file()
    assert _indexed_reference(root, 1).is_file()
    assert _indexed_reference(root, 2).is_file()
    assert "reference_03.md" not in (root / "index.md").read_text()
    assert long_patch in _indexed_reference(root, 1).read_text()
    assert "strategy" in _indexed_reference(root, 1).read_text().lower()
    generation_dirs = [path for path in (root / "sets").iterdir() if path.is_dir()]
    assert len(generation_dirs) == 1
    assert long_patch not in warm["program_md_addition"]
    assert "kb_references/index.md" in warm["program_md_addition"]


def test_reference_generation_swap_removes_old_set_only_after_publish(tmp_path):
    workspace = tmp_path / "workspace"
    old_solution = _solution(BAD_PATCH, solution_slug="solution/old")
    new_solution = _solution(APPLICABLE_PATCH, solution_slug="solution/new")

    integ._persist_kb_references(
        str(workspace),
        [old_solution],
        ["rejected:old"],
    )
    root = workspace / "forge_experiments" / "kb_references"
    old_reference = _indexed_reference(root, 1)
    old_index = (root / "index.md").read_text()

    integ._persist_kb_references(
        str(workspace),
        [new_solution],
        ["applied"],
    )
    new_index = (root / "index.md").read_text()
    new_reference = _indexed_reference(root, 1)

    assert old_index != new_index
    assert new_reference.is_file()
    assert APPLICABLE_PATCH in new_reference.read_text()
    assert not old_reference.exists()
    assert list((root / "sets").iterdir()) == [new_reference.parent]


def test_reference_generation_failure_before_index_keeps_old_index_valid(
    monkeypatch,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    integ._persist_kb_references(
        str(workspace),
        [_solution(BAD_PATCH, solution_slug="solution/old")],
        ["rejected:old"],
    )
    root = workspace / "forge_experiments" / "kb_references"
    old_index = (root / "index.md").read_text()
    old_reference = _indexed_reference(root, 1)
    real_atomic_write = integ.atomic_write_text

    def fail_index(path, text):
        if path == root / "index.md":
            raise OSError("simulated index replacement failure")
        return real_atomic_write(path, text)

    monkeypatch.setattr(integ, "atomic_write_text", fail_index)
    with pytest.raises(OSError, match="index replacement"):
        integ._persist_kb_references(
            str(workspace),
            [_solution(APPLICABLE_PATCH, solution_slug="solution/new")],
            ["applied"],
        )

    assert (root / "index.md").read_text() == old_index
    assert old_reference.is_file()


def test_reference_generation_cleanup_failure_keeps_new_index_valid(
    monkeypatch,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    integ._persist_kb_references(
        str(workspace),
        [_solution(BAD_PATCH, solution_slug="solution/old")],
        ["rejected:old"],
    )
    root = workspace / "forge_experiments" / "kb_references"
    old_index = (root / "index.md").read_text()
    old_reference = _indexed_reference(root, 1)

    def fail_cleanup(_root, _current):
        raise OSError("simulated cleanup interruption")

    monkeypatch.setattr(
        integ,
        "_cleanup_old_reference_generations",
        fail_cleanup,
    )
    integ._persist_kb_references(
        str(workspace),
        [_solution(APPLICABLE_PATCH, solution_slug="solution/new")],
        ["applied"],
    )

    assert (root / "index.md").read_text() != old_index
    assert APPLICABLE_PATCH in _indexed_reference(root, 1).read_text()
    assert old_reference.is_file()
    assert "already applied" in integ.kb_reference_program_md(str(workspace))
    integ.mark_kb_reference_rejected(str(workspace), 1, "publication_failed")
    assert "rejected:publication_failed" in (root / "index.md").read_text()
    assert "already applied" not in integ.kb_reference_program_md(str(workspace))


def test_kb_warmstart_resume_skips_read_and_restores_reference_pointer(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    root = repo / "forge_experiments" / "kb_references"
    reference = root / "sets" / "generation-a" / "reference_01.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("historical solution\n")
    (root / "index.md").write_text(
        "# KernelForge KB references\n\n"
        "- Rank 1: `sets/generation-a/reference_01.md` | "
        "solution `solution/fast` | "
        "speedup 4x | status `applied`\n"
    )

    def fail_read(**_kwargs):
        raise AssertionError("resume must not query the KB")

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_reader.read_top_solutions",
        fail_read,
    )
    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        resume=True,
    )

    assert warm["skipped"] == "resume"
    assert "kb_references/index.md" in warm["program_md_addition"]
    assert "Rank 1 solution `solution/fast` is already applied" in (warm["program_md_addition"])
    assert root.is_dir()
    assert reference.is_file()


def test_kb_warmstart_applies_benches_and_commits(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["candidate"] is True
    assert warm["applied"] is True
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 5.0
    assert "kb_references/index.md" in warm["program_md_addition"]
    assert "already applied" in warm["program_md_addition"]
    assert APPLICABLE_PATCH not in warm["program_md_addition"]
    references = repo / "forge_experiments" / "kb_references"
    assert "status `applied`" in (references / "index.md").read_text()
    assert APPLICABLE_PATCH in _indexed_reference(references, 1).read_text()
    assert (repo / "kernel.py").read_text() == "new\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == ("kb warm-start: apply kernelforge-exp/kernel/prev")


def test_kb_warmstart_preserves_candidate_measurements_and_repeat(
    monkeypatch,
    tmp_path,
):
    """Forward bench repeat and retain the accepted candidate's gate evidence."""
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    pristine = {
        "success": True,
        "median_ms": 10.0,
        "case_times": {"scored": 10.0, "noisy": 4.0},
        "unscored_cases": ["noisy"],
    }
    candidate = {
        "success": True,
        "median_ms": 5.0,
        "case_times": {"scored": 5.0, "noisy": 2.0},
        "unscored_cases": ["noisy"],
    }
    benches = _three_measurements(pristine, candidate)
    received_repeats = []

    def bench(_driver, bench_repeat=1):
        """Return staged benchmark results and record the requested repeat."""
        received_repeats.append(bench_repeat)
        return next(benches)

    monkeypatch.setattr(integ, "_bench_once", bench)
    monkeypatch.setattr(integ, "_correctness_once", lambda *_args: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        bench_repeat=4,
    )

    assert warm["applied"] is True
    assert received_repeats == [4] * 6
    assert warm["case_times"] == {"scored": 5.0, "noisy": 2.0}
    assert warm["unscored_cases"] == ["noisy"]


def test_kb_warmstart_uses_three_measurement_medians(monkeypatch, tmp_path):
    """Use three-run case medians for pristine and candidate measurements.

    The candidate runs disagree by enough that the median is neither the first
    of them nor their mean, and by little enough that the adoption gate -- three
    sigma of the candidate's own scores -- still admits the 2x gain.
    """
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    benches = iter(
        [
            {
                "success": True,
                "median_ms": 10.0,
                "case_times": {"case": 10.0},
            },
            {
                "success": True,
                "median_ms": 12.0,
                "case_times": {"case": 12.0},
            },
            {
                "success": True,
                "median_ms": 9.0,
                "case_times": {"case": 9.0},
            },
            {
                "success": True,
                "median_ms": 5.2,
                "case_times": {"case": 5.2},
            },
            {
                "success": True,
                "median_ms": 5.0,
                "case_times": {"case": 5.0},
            },
            {
                "success": True,
                "median_ms": 4.9,
                "case_times": {"case": 4.9},
            },
        ]
    )
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_args, **_kwargs: next(benches),
    )
    monkeypatch.setattr(integ, "_correctness_once", lambda *_args: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is True
    assert warm["pristine_ms"] == 10.0
    assert warm["baseline_case_times"] == {"case": 10.0}
    assert warm["keep_baseline_ms"] == 5.0


@pytest.mark.parametrize(
    ("pristine_bench", "candidate_bench", "expected_applied", "expected_reason"),
    [
        (
            {
                "success": True,
                "median_ms": 2.0,
                "case_times": {"a": 1.0, "b": 1.0},
            },
            {
                "success": True,
                "median_ms": 0.4,
                "case_times": {"a": 0.2},
            },
            False,
            "case_coverage_failed",
        ),
        (
            {
                "success": True,
                "median_ms": 2.0,
                "case_times": {"a": 1.0, "b": 1.0},
            },
            {
                "success": True,
                "median_ms": 1.4,
                "case_times": {"a": 0.2, "b": 1.2},
            },
            True,
            "",
        ),
        (
            {
                "success": True,
                "median_ms": 2.0,
                "case_times": {"a": 1.0, "b": 1.0},
            },
            {
                "success": True,
                "median_ms": 1.0,
                "case_times": {"a": 0.5, "b": 0.5},
            },
            True,
            "",
        ),
    ],
)
def test_kb_warmstart_matches_mean_only_keep_policy(
    monkeypatch,
    tmp_path,
    pristine_bench,
    candidate_bench,
    expected_applied,
    expected_reason,
):
    """Require complete cases but ignore individual case and group regressions."""
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    benches = _three_measurements(pristine_bench, candidate_bench)
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_args, **_kwargs: next(benches),
    )
    monkeypatch.setattr(integ, "_correctness_once", lambda *_args: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is expected_applied
    assert warm["reference_reason"] == expected_reason


def test_kb_warmstart_is_reference_only_without_pristine_baseline(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: None)

    def fail_apply(*_args, **_kwargs):
        raise AssertionError("warm-start must not apply without a pristine baseline")

    monkeypatch.setattr(integ, "_git_apply", fail_apply)
    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is False
    assert warm["pristine_ms"] is None
    assert warm["keep_baseline_ms"] is None
    assert "kb_references/index.md" in warm["program_md_addition"]
    assert (repo / "kernel.py").read_text() == "old\n"


def test_kb_warmstart_preserves_preexisting_staged_changes(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    kernel = repo / "kernel.py"
    kernel.write_text("caller staged change\n")
    _run(["git", "add", "kernel.py"], repo)
    staged_before = _run(["git", "diff", "--cached"], repo)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)

    def fail_mutation(*_args, **_kwargs):
        raise AssertionError("dirty workspace must not be benchmarked or patched")

    monkeypatch.setattr(integ, "_bench_once", fail_mutation)
    monkeypatch.setattr(integ, "_git_apply", fail_mutation)
    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(kernel),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is False
    assert _run(["git", "diff", "--cached"], repo) == staged_before
    assert kernel.read_text() == "caller staged change\n"


def test_kb_warmstart_rollback_preserves_existing_untracked_and_removes_new(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    existing = repo / "existing.txt"
    existing.write_text("keep me\n")
    helper = repo / "helper.py"
    _patch_read_solution(monkeypatch, NEW_FILE_PATCH)
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: _bench(10.0))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: False)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        source_files=[str(helper)],
    )

    assert warm["applied"] is False
    assert existing.read_text() == "keep me\n"
    assert not helper.exists()
    status = _run(["git", "status", "--short"], repo).splitlines()
    assert "?? existing.txt" in status
    assert "?? forge_experiments/" in status


def test_kb_warmstart_commits_allowed_new_source_file(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    helper = repo / "helper.py"
    _patch_read_solution(monkeypatch, NEW_FILE_PATCH)
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        source_files=[str(helper)],
    )

    assert warm["applied"] is True
    assert helper.read_text() == "helper\n"
    assert _run(["git", "show", "--format=", "--name-only", "HEAD"], repo) == ("helper.py")


def test_kb_warmstart_mismatched_candidate_falls_back_when_patch_does_not_apply(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    _patch_read_solutions(
        monkeypatch,
        _solution(BAD_PATCH, implementation_match=False),
    )
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: _bench(10.0))

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 10.0
    assert "kb_references/index.md" in warm["program_md_addition"]
    assert BAD_PATCH not in warm["program_md_addition"]
    references = repo / "forge_experiments" / "kb_references"
    assert "patch_touches_protected_path_or_not_applicable" in (references / "index.md").read_text()
    assert BAD_PATCH in _indexed_reference(references, 1).read_text()
    assert (repo / "kernel.py").read_text() == "old\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == "initial"


def test_kb_warmstart_rolls_back_when_applied_kernel_fails_bench(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    benches = _three_measurements(_bench(10.0), None)
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["keep_baseline_ms"] == 10.0
    assert (repo / "kernel.py").read_text() == "old\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == "initial"


def test_kb_warmstart_rolls_back_when_applied_kernel_fails_correctness(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    # Only the three pristine measurements should run; correctness rejects the
    # applied patch before candidate measurement.
    benches = _three_measurements(_bench(10.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: False)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 10.0
    assert "kb_references/index.md" in warm["program_md_addition"]
    assert (repo / "kernel.py").read_text() == "old\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == "initial"


def test_rejected_warmstart_removes_only_new_untracked_probe_artifacts(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    preserved = repo / "preexisting.txt"
    preserved.write_text("keep\n")
    (repo / ".gitignore").write_text(".probe-cache/\n")
    _run(["git", "add", ".gitignore"], repo)
    _run(["git", "commit", "-m", "ignore probe cache"], repo)
    ignored_preserved = repo / ".probe-cache" / "preexisting.bin"
    ignored_preserved.parent.mkdir()
    ignored_preserved.write_text("keep ignored\n")
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    monkeypatch.setattr(integ, "_bench_once", lambda *_args: _bench(10.0))

    def fail_correctness(*_args, **_kwargs):
        probe_dir = repo / "probe-output"
        probe_dir.mkdir()
        (probe_dir / "generated.txt").write_text("remove\n")
        (repo / ".probe-cache" / "generated.bin").write_text("remove ignored\n")
        return False

    monkeypatch.setattr(integ, "_correctness_once", fail_correctness)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is False
    assert preserved.read_text() == "keep\n"
    assert ignored_preserved.read_text() == "keep ignored\n"
    assert not (repo / ".probe-cache" / "generated.bin").exists()
    assert not (repo / "probe-output").exists()
    assert (repo / "kernel.py").read_text() == "old\n"


def test_kb_warmstart_rolls_back_when_applied_kernel_is_slower(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    # Applied kernel is correct but slower than the pristine baseline (12 >= 10):
    # it must be discarded and the loop cold-started from the pristine baseline.
    benches = _three_measurements(_bench(10.0), _bench(12.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_a, **_k: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 10.0
    assert "kb_references/index.md" in warm["program_md_addition"]
    assert (repo / "kernel.py").read_text() == "old\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == "initial"


@pytest.mark.parametrize(
    "kernel_backend",
    [
        "triton",
        "hip",
        "ck",
        "flydsl",
        "aiter",
        "hipblaslt",
    ],
)
def test_kb_warmstart_applies_for_every_backend_when_driver_validates(
    monkeypatch,
    tmp_path,
    kernel_backend,
):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_a, **_k: next(benches),
    )
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend=kernel_backend,
    )

    assert warm["candidate"] is True
    assert warm["applied"] is True
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 5.0
    assert "already applied" in warm["program_md_addition"]
    assert (repo / "kernel.py").read_text() == "new\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo).startswith("kb warm-start: apply ")


def test_kb_warmstart_attempts_candidate_on_implementation_mismatch(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    _patch_read_solutions(
        monkeypatch,
        _solution(APPLICABLE_PATCH, implementation_match=False),
    )
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_a, **_k: next(benches),
    )
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is True
    assert warm["reference_reason"] == ""
    assert warm["keep_baseline_ms"] == 5.0
    assert (repo / "kernel.py").read_text() == "new\n"


def test_kb_warmstart_reference_only_when_pristine_baseline_unavailable(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    monkeypatch.setattr(integ, "_bench_once", lambda *_args: None)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is False
    assert warm["reference_reason"] == "baseline_unavailable"


def test_kb_warmstart_applies_patch_to_undeclared_non_protected_source(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    (repo / "helper.py").write_text("old\n")
    _run(["git", "add", "helper.py"], repo)
    _run(["git", "commit", "-m", "helper"], repo)
    _patch_read_solutions(monkeypatch, _solution(OUTSIDE_PATCH))
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(integ, "_bench_once", lambda *_args: next(benches))
    monkeypatch.setattr(integ, "_correctness_once", lambda *_args: True)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        source_files=[],
    )

    assert warm["applied"] is True
    assert warm["reference_reason"] == ""
    assert (repo / "helper.py").read_text() == "new\n"


def test_kb_warmstart_rejects_patch_to_protected_config(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    config = repo / "config.yaml"
    config.write_text("value: original\n")
    _run(["git", "add", "config.yaml"], repo)
    _run(["git", "commit", "-m", "config"], repo)
    patch = """diff --git a/config.yaml b/config.yaml
--- a/config.yaml
+++ b/config.yaml
@@ -1 +1 @@
-value: original
+value: forged
"""
    _patch_read_solutions(monkeypatch, _solution(patch))
    monkeypatch.setattr(integ, "_bench_once", lambda *_args: _bench(10.0))

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
        source_files=[],
    )

    assert warm["applied"] is False
    assert warm["reference_reason"] == ("patch_touches_protected_path_or_not_applicable")
    assert config.read_text() == "value: original\n"


def test_kb_warmstart_rejects_and_restores_on_commit_failure(
    monkeypatch,
    tmp_path,
):
    repo = _init_repo(tmp_path)
    _patch_read_solution(monkeypatch, APPLICABLE_PATCH)
    benches = _three_measurements(_bench(10.0), _bench(5.0))
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_a, **_k: next(benches),
    )
    monkeypatch.setattr(integ, "_correctness_once", lambda *_a, **_k: True)
    real_git = integ.git

    def fail_commit(*args, **kwargs):
        if args[:1] == ("commit",):
            return subprocess.CompletedProcess(
                ["git", *args],
                1,
                stdout="",
                stderr="hook rejected commit",
            )
        return real_git(*args, **kwargs)

    monkeypatch.setattr(integ, "git", fail_commit)

    warm = integ.kb_warmstart(
        config=object(),
        kernel=str(repo / "kernel.py"),
        driver="driver.py",
        workspace_dir=str(repo),
        kernel_backend="triton",
    )

    assert warm["applied"] is False
    assert warm["reference_reason"] == "commit_failed"
    assert (repo / "kernel.py").read_text() == "old\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == "initial"


class _FakeExperiment:
    experiment_id = "exp123"


class _FakeIC:
    baseline_wall_ms = 8.0


class _FakeArchive:
    def load_index(self):
        return [
            {"decision": "REVERT_PERF", "wall_ms": 6.0, "snr_db": 31.0},
            {
                "decision": "KEEP",
                "wall_ms": 5.0,
                "mean_case_speedup": 3.0,
                "snr_db": 42.0,
            },
        ]

    def render_digest(self):
        return "archive digest"


class _FakeLoopRunner:
    experiment = _FakeExperiment()
    ic = _FakeIC()
    best_wall_ms = 4.0
    best_mean_case_speedup = 3.0
    archive = _FakeArchive()


def test_write_experience_to_kb_extracts_run_context(monkeypatch, tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n")
    captured = {}

    def fake_write_run_experience(**kwargs):
        captured.update(kwargs)
        return {"written": True, "solution": "solution", "speedup": 3.0}

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_sink.write_run_experience",
        fake_write_run_experience,
    )
    monkeypatch.setattr(integ, "_git_cumulative_diff", lambda _workspace, _base: "diff")
    usage = object()

    status = integ.write_experience_to_kb(
        config=object(),
        loop_runner=_FakeLoopRunner(),
        workspace_dir=str(tmp_path),
        kernel=str(kernel),
        kernel_backend="triton",
        gpu_target="gfx942",
        base_sha="base",
        pristine_baseline_ms=12.0,
        usage=usage,
    )

    assert status == {"written": True, "solution": "solution", "speedup": 3.0}
    assert captured["workspace"] == str(tmp_path)
    assert captured["kernel_path"] == str(kernel)
    assert captured["kernel_source"] == kernel.read_text()
    assert captured["kernel_backend"] == "triton"
    assert captured["gpu_target"] == "gfx942"
    assert captured["experiment_id"] == "exp123"
    assert captured["baseline_wall_ms"] == 12.0
    assert captured["best_wall_ms"] == 4.0
    assert captured["mean_case_speedup"] == 3.0
    assert captured["cumulative_diff"] == "diff"
    assert captured["digest"] == "archive digest"
    assert captured["snr_db"] == 42.0
    assert "workload_key" not in captured
    assert captured["usage"] is usage


def test_write_experience_to_kb_names_the_failure_that_stopped_the_publish(
    monkeypatch,
    tmp_path,
):
    """The refusal is persisted, so it has to say which failure happened.

    This status becomes ``kb_experience.write`` in the run's result JSON. What a
    caller needs from it is that the mirror did not happen and that the text
    identifies the failure well enough to act on. The literal ``error:`` prefix
    is not part of that contract, so asserting it pinned the format instead of
    the behaviour.
    """
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n")

    def fail_write_run_experience(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_sink.write_run_experience",
        fail_write_run_experience,
    )

    status = integ.write_experience_to_kb(
        config=object(),
        loop_runner=_FakeLoopRunner(),
        workspace_dir=str(tmp_path),
        kernel=str(kernel),
        kernel_backend="triton",
        gpu_target="gfx942",
        base_sha="base",
    )

    assert status["written"] is False
    assert "RuntimeError" in status["reason"]
    assert "boom" in status["reason"]


def test_write_uses_pristine_campaign_signature_after_helper_is_added(
    monkeypatch,
    tmp_path,
):
    kernel = tmp_path / "vllm" / "ops" / "kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n@triton.jit\ndef target_kernel(x):\n    return x\n")
    pristine_signature, pristine_identity = implementation_signature(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[],
        framework="vllm",
    )
    kernel.write_text(kernel.read_text() + "\n@triton.jit\ndef optimization_helper(x):\n    return x\n")
    optimized_signature, _ = implementation_signature(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[],
        framework="vllm",
    )
    assert optimized_signature != pristine_signature
    captured = {}

    def fake_write_run_experience(**kwargs):
        captured.update(kwargs)
        return {"written": True, "solution": "solution", "speedup": 2.0}

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_sink.write_run_experience",
        fake_write_run_experience,
    )
    monkeypatch.setattr(integ, "_git_cumulative_diff", lambda *_args: "diff")

    class Runner:
        experiment = _FakeExperiment()
        best_wall_ms = 4.0
        best_mean_case_speedup = 2.0
        archive = None
        ic = type(
            "IC",
            (),
            {
                "baseline_wall_ms": 8.0,
                "pristine_baseline_wall_ms": 8.0,
                "implementation_signature": pristine_signature,
                "implementation_identity": pristine_identity,
            },
        )()

    integ.write_experience_to_kb(
        config=object(),
        loop_runner=Runner(),
        workspace_dir=str(tmp_path),
        kernel=str(kernel),
        kernel_backend="triton",
        gpu_target="gfx942",
        base_sha="base",
        target_functions=["different_caller_target"],
        framework="vllm",
    )

    assert captured["implementation_signature_override"] == pristine_signature
    assert captured["implementation_identity_override"] == pristine_identity


# --- the rewrite warm start's persisted read error --------------------------- #
def _kb_store_config(tmp_path: Path, token: str) -> Config:
    """A KB Store run configuration whose credential is a recognizable string."""
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "remote-knowledge",
        kb_store_url="http://in-memory",
        kb_store_token=token,
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )
    return Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def test_rewrite_warm_start_failure_is_persisted_without_its_credential(
    tmp_path,
    monkeypatch,
):
    """The rewrite runner persists this reason, so it may not carry a credential.

    The warm start builds a KB Store client and reads over HTTP, and this guard
    catches everything the reader's own sanitizer does not: the store client is
    constructed outside that sanitizer's ``try``, so a construction failure of
    any type other than ``KBStoreError`` arrives here verbatim. It lands in the
    run's result JSON as ``kb_experience.read.read_error``, so a KB Store
    exception quoting the bearer token it authenticated with, a credentialed URL
    and an unbounded response body is redacted and bounded at 240 characters
    exactly like every sibling write path.
    """
    token = "kb-store-secret-9f3c"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    driver = workspace / "driver.py"
    driver.write_text("print('drive')\n")
    result_json = tmp_path / "rewrite-result.json"

    def failing_warm_start(*_args, **_kwargs):
        raise TimeoutError(
            f"connect https://forge:{token}@kb.example/knowledge failed "
            f"(sent Bearer {token}); resolver said {token} is unreachable" + " and dumped an unbounded trace" * 20
        )

    monkeypatch.setattr(
        runner.flydsl_rewrite_driver_preparation,
        "preflight_rewrite_driver",
        lambda *_a, **_k: DriverPreflight(
            report=driver_contract.PreflightReport(ok=True),
            reference=driver_contract.PreflightReport(
                ok=True,
                timing_ms=1.0,
                timing_metric="median_ms",
                case_ids=("case0",),
            ),
        ),
    )
    monkeypatch.setattr(runner, "try_flydsl_kb_warmstart", failing_warm_start)

    async def port_never_runs(*_args, **_kwargs):
        return PortResult(ok=False, attempts=1, error_tail="port failed")

    monkeypatch.setattr(runner, "run_port_loop", port_never_runs)

    result = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(source),
        driver=str(driver),
        workspace=str(workspace),
        experiments_dir=str(tmp_path / "experiments"),
        target_functions=["softmax"],
        config=_kb_store_config(tmp_path, token),
        result_json=str(result_json),
    )

    read = result["kb_experience"]["read"]
    assert read["read_reason"] == "read_error"
    assert json.loads(result_json.read_text())["kb_experience"]["read"] == read
    reason = read["read_error"]
    assert token not in reason
    assert reason.startswith("TimeoutError: connect https://[REDACTED]@")
    assert "Bearer [REDACTED]" in reason
    assert "resolver said [REDACTED] is unreachable" in reason
    assert len(reason) == MAX_READ_ERROR_LENGTH


def test_a_store_failure_reaches_the_publish_status_already_redacted(
    monkeypatch,
    tmp_path,
):
    """The credential-bearing half of ``write_experience_to_kb``'s contract.

    Its own handler reports ``f"error:{e!r}"``, which redacts nothing and bounds
    nothing, and the status is persisted as ``kb_experience.write``. That is safe
    only because no configured credential can reach it: the single store call,
    ``write_run_experience``, wraps its whole body in a handler that redacts
    against ``kb_store_secrets`` and caps at 240 characters, and everything else
    in the gathering step is ``getattr`` on local run state, a local ``git diff``
    helper that returns "" on any error, and reads inside
    ``contextlib.suppress``. This drives the real store path to prove the claim
    rather than restate it, so moving a store call out of that handler's reach
    fails here.
    """
    token = "kb-store-secret-9f3c"
    kernel = tmp_path / "vllm" / "ops" / "kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n@triton.jit\ndef target_kernel(x):\n    return x\n")

    class ExplodingStore:
        """A KB Store client whose every call quotes the credential it used."""

        def __init__(self, *_args, **_kwargs):
            pass

        def __getattr__(self, name):
            def refuse(*_args, **_kwargs):
                raise record_store.KBStoreError(
                    f"{name} https://forge:{token}@kb.example/knowledge failed "
                    f"(sent Bearer {token}); the store said {token} expired" + " and returned an unbounded body" * 20
                )

            return refuse

    monkeypatch.setattr(record_store, "KBStoreClient", ExplodingStore)
    monkeypatch.setattr(integ, "_git_cumulative_diff", lambda *_args: "diff\n")

    status = integ.write_experience_to_kb(
        config=_kb_store_config(tmp_path, token),
        loop_runner=_FakeLoopRunner(),
        workspace_dir=str(tmp_path),
        kernel=str(kernel),
        kernel_backend="triton",
        gpu_target="gfx950",
        base_sha="base",
        pristine_baseline_ms=12.0,
        framework="vllm",
        llm_summary=False,
        incremental_summary={
            "category": "",
            "strategy": "vectorize loads",
            "recipe": "",
            "lessons": "",
        },
    )

    assert status["written"] is False
    reason = status["reason"]
    assert token not in reason
    assert "https://[REDACTED]@kb.example" in reason
    assert "Bearer [REDACTED]" in reason
    # Still says which failure happened, and cannot grow past the cap.
    assert reason.startswith("KBStoreError: ")
    assert len(reason) == MAX_READ_ERROR_LENGTH
