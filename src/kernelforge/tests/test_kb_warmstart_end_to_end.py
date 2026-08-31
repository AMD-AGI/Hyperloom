"""Minimum-cost end-to-end coverage for KB warm-start reuse."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from kernelforge.config import Config
from kernelforge.knowledge import experience_integration as integration
from kernelforge.knowledge import experience_sink as sink
from kernelforge.knowledge.experience_store import KnowledgeConfig
from kernelforge.loop import recovery


PRODUCER_KERNEL_PATH = Path("packages/src/aiter_meta/ops/triton/deterministic_kernel.py")
CONSUMER_KERNEL_PATH = Path("src/aiter/ops/triton/deterministic_kernel.py")
PRODUCER_HELPER_PATH = Path("packages/src/aiter_meta/ops/triton/helper.py")

PRISTINE_SOURCE = """\
import triton

BLOCK_SIZE = 32

@triton.jit
def deterministic_kernel(x):
    return x
"""

INCOMPATIBLE_PRISTINE_SOURCE = PRISTINE_SOURCE.replace(
    "BLOCK_SIZE = 32",
    "BLOCK_SIZE = 16",
)
OPTIMIZED_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 64")
INCOMPATIBLE_OPTIMIZED_SOURCE = INCOMPATIBLE_PRISTINE_SOURCE.replace(
    "BLOCK_SIZE = 16",
    "BLOCK_SIZE = 64",
)
HELPER_SOURCE = """\
import triton

@triton.jit
def deterministic_helper(x):
    return x
"""

SUMMARY = {
    "category": "elementwise",
    "strategy": "increase the deterministic block size",
    "recipe": "Use BLOCK_SIZE=64.",
    "lessons": "The larger block is faster for this workload.",
}


#: Set per test by the autouse fixture below. Producer and consumer address one
#: store, which is what makes the round trip a round trip.
_KNOWLEDGE_ROOT: Path | None = None


def _run_config(gpu_type: str = "mi355x") -> Config:
    """A runtime config pointed at this test's own on-disk KB Store."""
    knowledge = KnowledgeConfig.from_env({}, mode="local", local_root=_KNOWLEDGE_ROOT)
    return Config.from_env(
        workspace=str(_KNOWLEDGE_ROOT),
        gpu_target="gfx950",
        gpu_type=gpu_type,
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_diff(repo: Path, base_commit: str) -> str:
    result = subprocess.run(
        ["git", "diff", base_commit, "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _initialize_workspace(
    root: Path,
    name: str,
    kernel_path: Path,
    source: str,
    *,
    include_helper: bool = False,
) -> tuple[Path, Path, str, list[str]]:
    workspace = root / name
    workspace.mkdir()
    _git(workspace, "init", "-b", "main")
    _git(workspace, "config", "user.email", "kb-e2e@example.com")
    _git(workspace, "config", "user.name", "KB E2E")

    kernel = workspace / kernel_path
    kernel.parent.mkdir(parents=True)
    kernel.write_text(source)
    source_files: list[str] = []
    if include_helper:
        helper = workspace / PRODUCER_HELPER_PATH
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text(HELPER_SOURCE)
        source_files.append(str(helper))

    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "pristine")
    return workspace, kernel, _git(workspace, "rev-parse", "HEAD"), source_files


def _publish_producer_solution(
    workspace: Path,
    kernel: Path,
    base_commit: str,
    *,
    optimized_source: str,
    experiment_id: str,
    gpu_type: str = "mi355x",
    source_files: list[str] | None = None,
) -> tuple[dict, str]:
    kernel.write_text(optimized_source)
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "optimize deterministic kernel")
    patch = _git_diff(workspace, base_commit)
    assert patch

    status = sink.write_run_experience(
        config=_run_config(gpu_type),
        workspace=str(workspace),
        kernel_path=str(kernel),
        kernel_source=optimized_source,
        kernel_backend="triton",
        gpu_target="gfx950",
        experiment_id=experiment_id,
        baseline_wall_ms=10.0,
        best_wall_ms=5.0,
        mean_case_speedup=2.0,
        cumulative_diff=patch,
        digest="deterministic end-to-end producer result",
        source_files=source_files,
        summary_override=SUMMARY,
    )
    return status, patch


def _install_driver_execution_doubles(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Path,
) -> tuple[list[float], list[str]]:
    benchmark_results: list[float] = []
    correctness_sources: list[str] = []

    def benchmark(_driver: str, *_a, **_k) -> dict:
        source = kernel.read_text()
        result = 5.0 if "BLOCK_SIZE = 64" in source else 10.0
        benchmark_results.append(result)
        return {
            "success": True,
            "median_ms": result,
            "case_times": {"case-1": result},
        }

    def correctness(_driver: str, _snr_threshold: float) -> bool:
        source = kernel.read_text()
        correctness_sources.append(source)
        return "BLOCK_SIZE = 64" in source

    monkeypatch.setattr(integration, "_bench_once", benchmark)
    monkeypatch.setattr(integration, "_correctness_once", correctness)
    return benchmark_results, correctness_sources


def _warm_start(
    consumer_workspace: Path,
    consumer_kernel: Path,
    *,
    gpu_type: str,
    source_files: list[str] | None = None,
) -> dict:
    return integration.kb_warmstart(
        config=_run_config(gpu_type),
        kernel=str(consumer_kernel),
        driver="unused-driver.py",
        workspace_dir=str(consumer_workspace),
        kernel_backend="triton",
        source_files=source_files,
    )


def _reference_artifacts(workspace: Path) -> tuple[Path, Path]:
    root = workspace / "forge_experiments" / "kb_references"
    index = root / "index.md"
    references = list((root / "sets").glob("*/reference_01.md"))
    assert len(references) == 1
    return index, references[0]


def _compact_pointer(solution_slug: str = "") -> str:
    pointer = (
        "## Historical KB design references\n"
        "Read `forge_experiments/kb_references/index.md` and the referenced files "
        "on demand. These historical code solutions are design references for "
        "this search; their full metadata and diffs are stored there."
    )
    if solution_slug:
        pointer += f"\nRank 1 solution `{solution_slug}` is already applied and is the search start."
    return pointer


@pytest.fixture(autouse=True)
def kb_store_root(tmp_path_factory):
    """One empty on-disk store per test, so runs never inherit each other."""
    global _KNOWLEDGE_ROOT
    _KNOWLEDGE_ROOT = tmp_path_factory.mktemp("kb-store")
    yield _KNOWLEDGE_ROOT
    _KNOWLEDGE_ROOT = None


def test_happy_path_applies_and_publishes_iteration_zero_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    producer, producer_kernel, producer_base, producer_sources = _initialize_workspace(
        tmp_path,
        "producer",
        PRODUCER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    consumer, consumer_kernel, consumer_base, _ = _initialize_workspace(
        tmp_path,
        "consumer",
        CONSUMER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    written, patch = _publish_producer_solution(
        producer,
        producer_kernel,
        producer_base,
        optimized_source=OPTIMIZED_SOURCE,
        experiment_id="producer-happy",
        source_files=producer_sources,
    )
    benchmark_results, correctness_sources = _install_driver_execution_doubles(
        monkeypatch,
        consumer_kernel,
    )

    warm = _warm_start(consumer, consumer_kernel, gpu_type="mi355x")

    assert written["written"] is True
    assert written["speedup"] == 2.0
    # Filed under the five-tuple, with the GPU in the address.
    assert written["kernel"] == "kernel:forge-loop:deterministic:aiter:unspecified:triton:mi355x"
    assert written["solution"] == f"{written['kernel']}/{written['session_id']}"
    assert written["champion"] is True
    assert warm["candidate"] is True
    assert warm["applied"] is True
    assert warm["solution_slug"] == written["solution"]
    assert warm["num_references"] == 1
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 5.0
    assert warm["mean_case_speedup"] == 2.0
    assert warm["applied_rank"] == 1
    assert benchmark_results == [10.0] * 3 + [5.0] * 3
    assert correctness_sources == [OPTIMIZED_SOURCE]
    assert consumer_kernel.read_text() == OPTIMIZED_SOURCE
    assert warm["applied_commit"] == _git(consumer, "rev-parse", "HEAD")
    assert warm["applied_commit"] != consumer_base
    assert _git(consumer, "log", "-1", "--pretty=%s") == (f"kb warm-start: apply {written['solution']}")

    index, reference = _reference_artifacts(consumer)
    assert index.is_file()
    assert reference.is_file()
    assert "status `applied`" in index.read_text()
    assert patch in reference.read_text()
    assert warm["program_md_addition"] == _compact_pointer(written["solution"])
    assert patch not in warm["program_md_addition"]
    assert "BLOCK_SIZE = 32" not in warm["program_md_addition"]

    checkpoints: dict[str, dict] = {}

    class Tracker:
        @staticmethod
        def set_checkpoint(experiment_id: str, checkpoint: dict) -> None:
            checkpoints[experiment_id] = checkpoint

    caller_result = tmp_path / "caller-result.json"
    result = recovery.publish_warm_start_recovery(
        workspace_dir=str(consumer),
        base_commit=consumer_base,
        warm=warm,
        caller_experiment_id="consumer-run",
        experience_id="producer-happy",
        tracker=Tracker(),
        result_json=str(caller_result),
    )

    best_result = json.loads((consumer / "forge_experiments" / "best_result.json").read_text())
    best_manifest = json.loads((consumer / "forge_experiments" / "best" / "manifest.json").read_text())
    caller_payload = json.loads(caller_result.read_text())
    checkpoint = checkpoints["consumer-run"]
    assert result is not None
    assert best_manifest == best_result
    assert best_result["iteration"] == 0
    assert best_result["commit_hash"] == warm["applied_commit"]
    assert best_result["baseline_wall_ms"] == 10.0
    assert best_result["search_start_ms"] == 5.0
    assert best_result["best_wall_ms"] == 5.0
    assert best_result["mean_case_speedup"] == 2.0
    assert best_result["total_speedup"] == 2.0
    assert best_result["incremental_speedup"] == 1.0
    assert best_result["improved_during_search"] is False
    assert best_result["correctness_passed"] is True
    assert best_result["changed_files"] == [CONSUMER_KERNEL_PATH.as_posix()]
    assert caller_payload["warm_start"] is True
    assert caller_payload["best_iteration"] == 0
    assert caller_payload["next_iteration"] == 1
    assert checkpoint["decision"] == "WARM_START"
    assert checkpoint["base_commit"] == consumer_base
    assert checkpoint["best_commit"] == warm["applied_commit"]
    assert checkpoint["best_iteration"] == 0
    assert checkpoint["baseline_ms"] == 10.0
    assert checkpoint["best_ms"] == 5.0


def test_invalid_patch_persists_reference_and_preserves_pristine_consumer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    producer, producer_kernel, producer_base, producer_sources = _initialize_workspace(
        tmp_path,
        "producer",
        PRODUCER_KERNEL_PATH,
        INCOMPATIBLE_PRISTINE_SOURCE,
    )
    consumer, consumer_kernel, consumer_base, _ = _initialize_workspace(
        tmp_path,
        "consumer",
        CONSUMER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    written, patch = _publish_producer_solution(
        producer,
        producer_kernel,
        producer_base,
        optimized_source=INCOMPATIBLE_OPTIMIZED_SOURCE,
        experiment_id="producer-invalid-patch",
        source_files=producer_sources,
    )
    benchmark_results, correctness_sources = _install_driver_execution_doubles(
        monkeypatch,
        consumer_kernel,
    )

    warm = _warm_start(consumer, consumer_kernel, gpu_type="mi355x")

    assert written["written"] is True
    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["reference_reason"] == ("patch_touches_protected_path_or_not_applicable")
    assert warm["num_references"] == 1
    assert warm["pristine_ms"] == 10.0
    assert warm["keep_baseline_ms"] == 10.0
    assert benchmark_results == [10.0] * 3
    assert correctness_sources == []
    assert consumer_kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") == consumer_base
    assert _git(consumer, "status", "--porcelain", "--untracked-files=no") == ""

    index, reference = _reference_artifacts(consumer)
    assert index.is_file()
    assert reference.is_file()
    assert "rejected:patch_touches_protected_path_or_not_applicable" in index.read_text()
    assert patch in reference.read_text()
    assert warm["program_md_addition"] == _compact_pointer()
    assert patch not in warm["program_md_addition"]


def test_expanded_consumer_source_set_attempts_and_applies_solution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    producer, producer_kernel, producer_base, producer_sources = _initialize_workspace(
        tmp_path,
        "producer",
        PRODUCER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    consumer, consumer_kernel, consumer_base, consumer_sources = _initialize_workspace(
        tmp_path,
        "consumer",
        CONSUMER_KERNEL_PATH,
        PRISTINE_SOURCE,
        include_helper=True,
    )
    written, patch = _publish_producer_solution(
        producer,
        producer_kernel,
        producer_base,
        optimized_source=OPTIMIZED_SOURCE,
        experiment_id="producer-source-set-mismatch",
        source_files=producer_sources,
    )
    benchmark_results, correctness_sources = _install_driver_execution_doubles(
        monkeypatch,
        consumer_kernel,
    )

    warm = _warm_start(
        consumer,
        consumer_kernel,
        gpu_type="mi355x",
        source_files=consumer_sources,
    )

    assert written["written"] is True
    assert warm["candidate"] is True
    assert warm["solution_slug"] == written["solution"]
    assert warm["applied"] is True
    assert warm["match_mode"] == "reference"
    assert warm["reference_reason"] == ""
    assert warm["num_references"] == 1
    assert warm["keep_baseline_ms"] == 5.0
    assert benchmark_results == [10.0] * 3 + [5.0] * 3
    assert correctness_sources == [OPTIMIZED_SOURCE]
    assert consumer_kernel.read_text() == OPTIMIZED_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") != consumer_base

    index, reference = _reference_artifacts(consumer)
    reference_text = reference.read_text()
    assert "status `applied`" in index.read_text()
    assert "- Implementation match: `False`" in reference_text
    assert "aiter/ops/triton/helper.py" in reference_text
    assert patch in reference_text
    assert warm["program_md_addition"] == _compact_pointer(written["solution"])
    assert patch not in warm["program_md_addition"]


def test_gpu_model_mismatch_has_no_candidate_or_reference_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    producer, producer_kernel, producer_base, producer_sources = _initialize_workspace(
        tmp_path,
        "producer",
        PRODUCER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    consumer, consumer_kernel, consumer_base, _ = _initialize_workspace(
        tmp_path,
        "consumer",
        CONSUMER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    written, _patch = _publish_producer_solution(
        producer,
        producer_kernel,
        producer_base,
        optimized_source=OPTIMIZED_SOURCE,
        experiment_id="producer-gpu-model-mismatch",
        gpu_type="mi355x",
        source_files=producer_sources,
    )
    benchmark_results, correctness_sources = _install_driver_execution_doubles(
        monkeypatch,
        consumer_kernel,
    )

    warm = _warm_start(consumer, consumer_kernel, gpu_type="mi300x")

    assert written["written"] is True
    assert warm == {
        "candidate": False,
        "read_reason": "no_prior_record",
        "read_error": "",
    }
    assert benchmark_results == []
    assert correctness_sources == []
    assert consumer_kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") == consumer_base
    assert not (consumer / "forge_experiments" / "kb_references").exists()


def _declare_failing_task_suite(workspace: Path) -> None:
    """Give the consumer an arena task config whose own suite rejects everything.

    The mla_decode shape: the driver's SNR probe is happy, and the task's own
    tolerance is what the kernel actually breaks.
    """
    workspace.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                # Step 1 has to pass for the gate to reach the tolerance the
                # kernel actually breaks.
                "compile_command": [f"{sys.executable} -c 'pass'"],
                "correctness_command": [
                    f"{sys.executable} -c " + repr("raise AssertionError('normalized max err 0.02468 exceeds 0.02')")
                ],
            }
        )
    )
    _git(workspace, "add", "config.yaml")
    _git(workspace, "commit", "-m", "declare the task's correctness command")


def test_a_warm_start_failing_the_task_suite_is_not_adopted_or_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """A 33.4 dB kernel that breaks the task's tolerance cannot become the start.

    The SNR probe passes on this candidate, every performance gate passes, and
    the task's own suite fails: the warm start must reject it, leave the
    consumer pristine, and publish nothing. Because the CLI reaches its
    ``--return-after-read-KB`` result only through ``applied``, a rejection here
    is also what keeps such a kernel out of the run's answer -- see
    ``test_a_warm_start_rejected_by_the_task_suite_is_not_returned`` in
    tests/test_forge_loop_resume.py.
    """
    producer, producer_kernel, producer_base, producer_sources = _initialize_workspace(
        tmp_path,
        "producer",
        PRODUCER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    consumer, consumer_kernel, consumer_base, _ = _initialize_workspace(
        tmp_path,
        "consumer",
        CONSUMER_KERNEL_PATH,
        PRISTINE_SOURCE,
    )
    _declare_failing_task_suite(consumer)
    consumer_base = _git(consumer, "rev-parse", "HEAD")
    written, patch = _publish_producer_solution(
        producer,
        producer_kernel,
        producer_base,
        optimized_source=OPTIMIZED_SOURCE,
        experiment_id="producer-canonical-failure",
        source_files=producer_sources,
    )
    benchmark_results, correctness_sources = _install_driver_execution_doubles(
        monkeypatch,
        consumer_kernel,
    )

    warm = _warm_start(consumer, consumer_kernel, gpu_type="mi355x")

    assert written["written"] is True
    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["reference_reason"] == "canonical_correctness_failed"
    assert warm["applied_commit"] == ""
    assert warm["applied_rank"] is None
    # The SNR probe and the benchmark both accepted this candidate first.
    assert correctness_sources == [OPTIMIZED_SOURCE]
    assert benchmark_results == [10.0] * 3 + [5.0] * 3
    assert consumer_kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") == consumer_base
    assert _git(consumer, "status", "--porcelain", "--untracked-files=no") == ""

    index, reference = _reference_artifacts(consumer)
    assert "rejected:canonical_correctness_failed" in index.read_text()
    assert patch in reference.read_text()
    assert warm["program_md_addition"] == _compact_pointer()

    published = recovery.publish_warm_start_recovery(
        workspace_dir=str(consumer),
        base_commit=consumer_base,
        warm=warm,
        caller_experiment_id="consumer-run",
        experience_id="producer-canonical-failure",
        tracker=None,
        result_json=str(tmp_path / "caller-result.json"),
    )

    assert published is None
    assert not (consumer / "forge_experiments" / "best_result.json").exists()
    assert not (tmp_path / "caller-result.json").exists()
