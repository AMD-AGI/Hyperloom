# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for recording a forge-loop run under its kernel identity.

These run against the KB Store's on-disk backend rather than a stand-in, so the
gates, the address and the artifact round trip are exercised the way a real run
would exercise them.
"""

from __future__ import annotations

import pytest

from kernelforge.config import Config
from kernelforge.knowledge import experience_sink as sink
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_GBRAIN,
    KnowledgeConfig,
)
from kernelforge.rewrite_by_flydsl.agent_kb import KernelRecipeKB
from kernelforge.knowledge.loop_identity import (
    EXPERIENCE_ARTIFACT,
    PATCH_ARTIFACT,
    resolve_loop_identity,
)

DIFF = """diff --git a/kernel.py b/kernel.py
--- a/kernel.py
+++ b/kernel.py
@@ -1 +1 @@
-old
+new
"""
KERNEL_SOURCE = "import triton\n\n\n@triton.jit\ndef my_kernel(x):\n    return x\n"
SUMMARY = {
    "category": "gemm",
    "strategy": "vectorize loads",
    "recipe": "Use vectorized loads.",
    "lessons": "Alignment matters.",
}
#: ``my_kernel`` loses its ``_kernel`` suffix, and a file owned by no framework
#: package reports ``unknown`` with no installed version.
IDENTITY = "kernel:forge-loop:my:unknown:none:triton:mi300x"


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "kernel.py").write_text(KERNEL_SOURCE, encoding="utf-8")
    return root


@pytest.fixture()
def config(tmp_path, workspace):
    knowledge = KnowledgeConfig.from_env({}, mode="local", local_root=tmp_path / "knowledge")
    return Config.from_env(
        workspace=str(workspace),
        gpu_target="gfx942",
        gpu_type="mi300x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _write(config, workspace, **overrides):
    kwargs = {
        "config": config,
        "workspace": str(workspace),
        "kernel_path": str(workspace / "kernel.py"),
        "kernel_source": KERNEL_SOURCE,
        "kernel_backend": "triton",
        "gpu_target": "gfx942",
        "experiment_id": "exp1",
        "baseline_wall_ms": 10.0,
        "best_wall_ms": 5.0,
        "mean_case_speedup": 2.0,
        "cumulative_diff": DIFF,
        "digest": "iter 1 kept",
        "snr_db": 42.0,
        "framework": "standalone",
        "summary_override": SUMMARY,
    }
    kwargs.update(overrides)
    return sink.write_run_experience(**kwargs)


def _records(config, workspace) -> KernelRecipeKB:
    identity, _op, _fw = resolve_loop_identity(
        kernel_path=str(workspace / "kernel.py"),
        kernel_source=KERNEL_SOURCE,
        kernel_backend="triton",
        gpu_type="mi300x",
        framework="standalone",
    )
    return KernelRecipeKB.open_identity(identity, config)


# --- gates ----------------------------------------------------------------- #
def test_write_skips_when_the_store_is_not_configured(tmp_path, workspace):
    # Remote mode selected against GBrain, which holds no rewrite records, so
    # there is no backend to write to.
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "knowledge",
        gbrain_base_url="https://gbrain.invalid",
        gbrain_token="secret",
        remote_backend=REMOTE_BACKEND_GBRAIN,
    )
    config = Config.from_env(
        workspace=str(workspace),
        gpu_target="gfx942",
        gpu_type="mi300x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )

    assert _write(config, workspace) == {
        "written": False,
        "reason": "not_configured",
    }


def test_write_fails_closed_without_gpu_type(config, workspace):
    # The hardware model addresses the record. Writing without it would file the
    # run under an address no read resolves to, which is worse than not writing:
    # the loop would report success while the experience is unreachable.
    config.gpu_type = ""
    assert _write(config, workspace) == {
        "written": False,
        "reason": "missing_gpu_type",
    }


def test_write_skips_no_improvement_and_empty_diff(config, workspace):
    assert _write(config, workspace, mean_case_speedup=1.0)["reason"] == "no_improvement"
    assert _write(config, workspace, mean_case_speedup=0.9)["reason"] == "no_improvement"
    assert _write(config, workspace, cumulative_diff="")["reason"] == "empty_diff"
    assert _records(config, workspace).list_candidates() == []


def test_write_requires_explicit_mean_case_speedup(config, workspace):
    assert _write(config, workspace, mean_case_speedup=None) == {
        "written": False,
        "reason": "missing_mean_case_speedup",
    }
    assert _write(config, workspace, mean_case_speedup=float("nan")) == {
        "written": False,
        "reason": "invalid_mean_case_speedup",
    }
    assert _records(config, workspace).list_candidates() == []


# --- what a successful write records --------------------------------------- #
def test_write_files_the_run_under_its_five_tuple(config, workspace):
    status = _write(config, workspace)

    assert status["written"] is True
    assert status["kernel"] == IDENTITY
    assert status["solution"] == f"{IDENTITY}/{status['session_id']}"
    assert status["speedup"] == 2.0


def test_write_preserves_the_patch_and_the_measurements(config, workspace):
    status = _write(config, workspace)
    kb = _records(config, workspace)
    record = kb.list_candidates(limit=1)[0].value

    assert record["metric"] == {
        "wall_ms": 5.0,
        "baseline_wall_ms": 10.0,
        "speedup": 2.0,
        "snr_db": 42.0,
        "gpu_arch": "gfx942",
    }
    assert record["changed_files"] == ["kernel.py"]
    assert record["strategy"] == "vectorize loads"
    assert record["lessons"] == "Alignment matters."
    assert record["task_id"] == "exp1"
    # The diff travels as an artifact, not inside the record.
    assert "patch_content" not in record
    assert kb.prior_file(status["session_id"], PATCH_ARTIFACT) == DIFF.encode()


def test_a_crlf_patch_round_trips_byte_for_byte(config, workspace):
    """A patch is applied by matching context byte for byte.

    Folding its newlines leaves a diff that still parses, still names the right
    file, and still cannot be applied to a CRLF source -- so the solution reads
    as reusable right up to the moment git refuses it.
    """
    crlf = DIFF.replace("\n", "\r\n")
    status = _write(config, workspace, cumulative_diff=crlf)

    assert status["written"] is True
    kb = _records(config, workspace)

    # Both ways a stored patch is reached must return the same bytes: the warm
    # start reads the materialized file, and a direct fetch goes through the
    # store. A reader that folded the newlines would still return a diff that
    # parses and names the right file, so only the bytes reveal the loss.
    bundle = kb.read_top_n(workspace / "kb-candidates", limit=1)[0]
    materialized = (bundle.files_dir / PATCH_ARTIFACT).read_bytes()
    fetched = kb.prior_file(bundle.session_id, PATCH_ARTIFACT)

    assert materialized == crlf.encode()
    assert fetched == crlf.encode()
    assert materialized.count(b"\r") == crlf.count("\r")


def test_the_record_carries_a_readable_account_beside_the_patch(config, workspace):
    """A ranker reads the record's fields; a person reads this.

    It travels with the patch so a reader does not have to reconstruct the run
    from the record, and it names the patch rather than copying it.
    """
    _write(config, workspace)

    bundle = _records(config, workspace).read_top_n(workspace / "kb-candidates", limit=1)[0]
    experience = (bundle.files_dir / EXPERIENCE_ARTIFACT).read_text(encoding="utf-8")

    assert experience.startswith(f"# {IDENTITY}\n")
    assert "- Speedup: 2x (5 ms vs 10 ms)\n" in experience
    assert "- Correctness: SNR 42.0 dB\n" in experience
    assert "- Compiled for: gfx942\n" in experience
    assert "- Changed files: kernel.py\n" in experience
    assert f"- Patch: `{PATCH_ARTIFACT}`\n" in experience
    assert "## Strategy\n\nvectorize loads\n" in experience
    assert "## Lessons\n\nAlignment matters.\n" in experience
    # The diff sits beside it under its own name; copying it here would hold the
    # same bytes twice in one record.
    assert "-old" not in experience


def test_write_persists_supplied_pristine_implementation_contract(config, workspace):
    pristine_identity = {
        "source_paths": ["kernel.py"],
        "implementation_symbols": ["pristine_kernel"],
    }
    pristine_signature = sink.hash_implementation_identity(pristine_identity)

    status = _write(
        config,
        workspace,
        implementation_signature_override=pristine_signature,
        implementation_identity_override=pristine_identity,
    )

    assert status["written"] is True
    record = _records(config, workspace).list_candidates(limit=1)[0].value
    assert record["implementation_signature"] == pristine_signature
    assert record["implementation_identity"] == pristine_identity


# --- how repeated runs accumulate ------------------------------------------ #
def test_a_slower_run_is_still_recorded_but_never_takes_the_champion(config, workspace):
    """Losing to a previous run is not a reason to discard the evidence."""
    fast = _write(config, workspace, experiment_id="run-a", mean_case_speedup=3.0)
    slow = _write(
        config,
        workspace,
        experiment_id="run-b",
        mean_case_speedup=1.5,
        cumulative_diff=DIFF.replace("+new", "+other"),
    )

    assert [fast["champion"], slow["champion"]] == [True, False]
    kb = _records(config, workspace)
    assert [c.speedup for c in kb.list_candidates()] == [3.0, 1.5]
    assert kb.list_candidates(limit=1)[0].value["task_id"] == "run-a"


def test_a_warm_started_run_that_improved_nothing_records_no_second_copy(config, workspace):
    """Reproducing the solution you started from is not a new solution."""
    _write(config, workspace, experiment_id="prior", mean_case_speedup=2.0)

    same = _write(
        config,
        workspace,
        experiment_id="warm-started",
        mean_case_speedup=2.0,
        reused_speedup=2.0,
    )

    assert same == {"written": False, "reason": "no_improvement_over_reuse"}
    # The one recorded solution still stands, and still serves its patch: the
    # warm-started run keeps it as its own result.
    kb = _records(config, workspace)
    candidates = kb.list_candidates(limit=5)
    assert len(candidates) == 1
    assert kb.prior_file(candidates[0].session_id, PATCH_ARTIFACT) == DIFF.encode()


def test_a_warm_started_run_that_improved_records_the_better_result(config, workspace):
    _write(config, workspace, experiment_id="prior", mean_case_speedup=2.0)

    better = _write(
        config,
        workspace,
        experiment_id="warm-started",
        mean_case_speedup=2.5,
        reused_speedup=2.0,
        cumulative_diff=DIFF.replace("+new", "+better"),
    )

    assert better["written"] is True
    assert better["champion"] is True
    assert len(_records(config, workspace).list_candidates(limit=5)) == 2


def test_a_new_summary_for_one_solution_is_recorded_as_a_second_candidate(config, workspace):
    """Known gap, pinned so a change in it cannot pass unnoticed.

    The store names a record after its own content, so the final write's richer
    prose for a solution already recorded is filed as its own record: one patch
    at one speedup, held twice, crowding out a genuinely different approach.
    Closing it needs a way to name the record being revised, which the store
    does not expose.
    """
    _write(config, workspace, summary_override={**SUMMARY, "lessons": ""})
    _write(config, workspace, summary_override={**SUMMARY, "lessons": "later"})

    assert len(_records(config, workspace).list_candidates(limit=5)) == 2


def test_a_cold_run_is_unaffected_by_the_reuse_floor(config, workspace):
    """No warm start means no floor, so the usual gates are the only ones."""
    assert _write(config, workspace, reused_speedup=None)["written"] is True


def test_recording_the_same_result_twice_updates_one_record(config, workspace):
    first = _write(config, workspace)
    second = _write(config, workspace)

    assert first["session_id"] == second["session_id"]
    assert len(_records(config, workspace).list_candidates(limit=5)) == 1


def test_a_different_gpu_is_a_different_address(config, workspace):
    # Two cards can share one compilation target, so the target alone would pool
    # runs whose timings are not comparable. The model separates them.
    _write(config, workspace)
    config.gpu_type = "mi355x"
    _write(config, workspace)

    identity, _op, _fw = resolve_loop_identity(
        kernel_path=str(workspace / "kernel.py"),
        kernel_source=KERNEL_SOURCE,
        kernel_backend="triton",
        gpu_type="mi355x",
        framework="standalone",
    )
    other = KernelRecipeKB.open_identity(identity, config)

    assert other.canonical_id.endswith(":mi355x")
    assert other.canonical_id != IDENTITY
    # Each model holds exactly its own run, so neither can be read on the
    # other's behalf.
    assert len(other.list_candidates(limit=5)) == 1
    config.gpu_type = "mi300x"
    assert len(_records(config, workspace).list_candidates(limit=5)) == 1


def test_a_different_producer_is_a_different_address(config, workspace):
    # A pipeline built ON the loop rewires a framework rather than optimizing a
    # kernel, so its records must neither rank against the loop's own nor be
    # offered to one as a warm start.
    _write(config, workspace)
    config.producer = "fusion"
    _write(config, workspace)

    identity, _op, _fw = resolve_loop_identity(
        kernel_path=str(workspace / "kernel.py"),
        kernel_source=KERNEL_SOURCE,
        kernel_backend="triton",
        gpu_type="mi300x",
        framework="standalone",
        producer="fusion",
    )
    other = KernelRecipeKB.open_identity(identity, config)

    assert other.canonical_id.startswith("kernel:fusion:")
    assert other.canonical_id != IDENTITY
    assert len(other.list_candidates(limit=5)) == 1
    config.producer = ""
    assert len(_records(config, workspace).list_candidates(limit=5)) == 1


def test_an_unset_producer_still_files_under_the_loops_own(config, workspace):
    """Every existing caller passes nothing, and must keep its address."""
    identity, _op, _fw = resolve_loop_identity(
        kernel_path=str(workspace / "kernel.py"),
        kernel_source=KERNEL_SOURCE,
        kernel_backend="triton",
        gpu_type="mi300x",
        framework="standalone",
    )
    assert identity.producer == "forge-loop"
