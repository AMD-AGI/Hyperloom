# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the forge-loop warm-start read.

These run against the KB Store's on-disk backend and seed it through the real
write path, so a read is only ever asserted against something a run could
actually have recorded.
"""

from __future__ import annotations

import pytest

from kernelforge.config import Config
from kernelforge.knowledge import experience_reader as reader
from kernelforge.knowledge.experience_reader import (
    read_best_solution,
    read_top_solutions,
    sanitize_read_error,
)
from kernelforge.knowledge.experience_sink import (
    hash_implementation_identity,
    write_run_experience,
)
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_GBRAIN,
    KnowledgeConfig,
)
from kernelforge.rewrite_by_flydsl import record_store

DIFF = "diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n"
KERNEL_SOURCE = "import triton\n\n\n@triton.jit\ndef my_kernel(x):\n    return x\n"
SUMMARY = {
    "category": "gemm",
    "strategy": "tile",
    "recipe": "step",
    "lessons": "ok",
}


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


def _seed(config, workspace, **overrides):
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
        "digest": "d",
        "snr_db": 42.0,
        "framework": "standalone",
        "summary_override": SUMMARY,
    }
    kwargs.update(overrides)
    return write_run_experience(**kwargs)


def _read_args(config, workspace, **overrides):
    args = {
        "config": config,
        "kernel_path": str(workspace / "kernel.py"),
        "kernel_source": KERNEL_SOURCE,
        "kernel_backend": "triton",
        "framework": "standalone",
        "workspace": str(workspace),
    }
    args.update(overrides)
    return args


# --- the paths that yield no candidate ------------------------------------- #
def test_read_none_when_the_store_is_not_configured(tmp_path, workspace):
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
    status: dict[str, str] = {}

    assert read_top_solutions(**_read_args(config, workspace), read_status=status) == []
    assert status == {"read_reason": "not_configured", "read_error": ""}


def test_read_none_without_required_gpu_type(config, workspace):
    # Reading without the model would resolve a GPU-less address that no write
    # ever reached, and the empty result would look like an honest cold start.
    config.gpu_type = ""
    status: dict[str, str] = {}

    assert read_top_solutions(**_read_args(config, workspace), read_status=status) == []
    assert status == {"read_reason": "missing_gpu_type", "read_error": ""}


def test_read_none_when_nothing_was_ever_recorded(config, workspace):
    status: dict[str, str] = {}

    assert read_top_solutions(**_read_args(config, workspace), read_status=status) == []
    assert status == {"read_reason": "no_prior_record", "read_error": ""}
    assert read_best_solution(**_read_args(config, workspace)) is None


def test_a_transport_failure_is_not_reported_as_an_empty_store(config, workspace, monkeypatch):
    """Cold-starting on a broken link would hide an outage as a normal miss."""
    _seed(config, workspace)

    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(record_store.LocalRewriteRecords, "candidates", boom)
    status: dict[str, str] = {}

    assert read_top_solutions(**_read_args(config, workspace), read_status=status) == []
    assert status["read_reason"] == "read_error"
    assert "kaboom" in status["read_error"]


def test_an_unexpected_failure_cold_starts_instead_of_raising(config, workspace, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(reader, "implementation_signature", boom)

    assert read_best_solution(**_read_args(config, workspace)) is None


# --- what a hit carries ----------------------------------------------------- #
def test_read_returns_the_champion_with_its_patch(config, workspace):
    _seed(config, workspace)
    status: dict[str, str] = {}

    best = read_best_solution(**_read_args(config, workspace))
    solutions = read_top_solutions(**_read_args(config, workspace), read_status=status)

    assert status == {"read_reason": "hit", "read_error": ""}
    assert best["speedup"] == 2.0
    assert best["strategy"] == "tile"
    assert best["recipe"] == "step"
    assert best["lessons"] == "ok"
    assert best["metric"]["speedup"] == 2.0
    assert best["patch_content"] == DIFF
    assert best["kernel_slug"].startswith("kernel:forge-loop:my:")
    assert solutions[0]["solution_slug"] == best["solution_slug"]


def test_the_same_tree_matches_its_own_implementation_signature(config, workspace):
    _seed(config, workspace)

    best = read_best_solution(**_read_args(config, workspace))

    assert best["implementation_match"] is True
    assert best["match_mode"] == "exact"
    assert best["implementation_signature"] == best["consumer_implementation_signature"]


def test_a_foreign_implementation_stays_reference_only(config, workspace):
    """Only an exact signature may reach the auto-apply gate downstream."""
    foreign_identity = {
        "source_paths": ["kernel.py"],
        "implementation_symbols": ["someone_elses_kernel"],
    }
    _seed(
        config,
        workspace,
        implementation_signature_override=hash_implementation_identity(foreign_identity),
        implementation_identity_override=foreign_identity,
    )

    best = read_best_solution(**_read_args(config, workspace))

    assert best["implementation_match"] is False
    assert best["match_mode"] == "reference"


def test_candidates_come_back_ranked_by_speedup(config, workspace):
    _seed(config, workspace, experiment_id="mid", mean_case_speedup=2.0, cumulative_diff=DIFF.replace("+new", "+mid"))
    _seed(config, workspace, experiment_id="best", mean_case_speedup=5.0, cumulative_diff=DIFF.replace("+new", "+best"))
    _seed(config, workspace, experiment_id="low", mean_case_speedup=1.25, cumulative_diff=DIFF.replace("+new", "+low"))

    solutions = read_top_solutions(**_read_args(config, workspace), top_k=3)

    assert [s["speedup"] for s in solutions] == [5.0, 2.0, 1.25]
    assert read_best_solution(**_read_args(config, workspace))["speedup"] == 5.0


def test_top_k_bounds_the_result(config, workspace):
    for name, speedup in (("a", 2.0), ("b", 3.0), ("c", 4.0)):
        _seed(
            config,
            workspace,
            experiment_id=name,
            mean_case_speedup=speedup,
            cumulative_diff=DIFF.replace("+new", f"+{name}"),
        )

    assert len(read_top_solutions(**_read_args(config, workspace), top_k=2)) == 2


def test_only_the_champion_is_downloaded_at_top_1(config, workspace):
    """A bounded read must not pay for the candidates it will never look at."""
    for name, speedup in (("a", 2.0), ("b", 3.0), ("c", 4.0)):
        _seed(
            config,
            workspace,
            experiment_id=name,
            mean_case_speedup=speedup,
            cumulative_diff=DIFF.replace("+new", f"+{name}"),
        )

    solutions = read_top_solutions(**_read_args(config, workspace), top_k=1)

    assert [s["speedup"] for s in solutions] == [4.0]
    bundles = list((workspace / "forge_experiments" / "kb_candidates").iterdir())
    assert len(bundles) == 1, "only the selected candidate may be materialized"


# --- where materialized candidates are allowed to land ---------------------- #
def test_candidates_land_under_the_workspace_for_inspection(config, workspace):
    _seed(config, workspace)

    read_top_solutions(**_read_args(config, workspace))

    bundles = list((workspace / "forge_experiments" / "kb_candidates").iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "files" / "solution.patch").read_text() == DIFF


def test_a_cold_read_leaves_no_directory_behind(config, workspace):
    read_top_solutions(**_read_args(config, workspace))

    assert not (workspace / "forge_experiments" / "kb_candidates").exists()


def test_a_later_read_does_not_inherit_an_earlier_one_s_bundles(config, workspace):
    """Stale bundles beside current ones would read as if they were selected."""
    _seed(config, workspace, experiment_id="first")
    read_top_solutions(**_read_args(config, workspace))
    root = workspace / "forge_experiments" / "kb_candidates"
    stale = root / "left-over-from-a-previous-read"
    stale.mkdir()

    read_top_solutions(**_read_args(config, workspace))

    assert not stale.exists()
    assert len(list(root.iterdir())) == 1


def test_without_a_workspace_nothing_is_written_beside_the_kernel(config, workspace):
    """A kernel usually lives in site-packages; a read may not write there."""
    _seed(config, workspace)
    args = _read_args(config, workspace)
    args.pop("workspace")

    solutions = read_top_solutions(**args)

    assert solutions, "the read must still work without a workspace"
    assert solutions[0]["patch_content"] == DIFF
    assert not (workspace / "forge_experiments").exists()


# --- error message hygiene -------------------------------------------------- #
def test_read_error_is_sanitized_and_bounded():
    message = sanitize_read_error(
        RuntimeError("Authorization: Bearer super-secret password=hunter2 https://user:pw@gbrain.example " + "x" * 500),
        secrets=("super-secret",),
    )

    assert "super-secret" not in message
    assert "hunter2" not in message
    assert "user:pw@" not in message
    assert "[REDACTED]" in message
    assert len(message) <= 500
