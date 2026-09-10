# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The handoff read back through the KernelForge side that consumes it.

Kept apart from the projection's own suite because importing the controller
pulls in the repo-lock machinery, and with it POSIX ``fcntl``. Only the
round-trip needs that; the projection tests should run anywhere.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.kernel_rewrite_controller.handoff import read_handoff
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
    _additional_directories,
)

from hyperloom.orchestrator.kernel.forge_handoff import write_forge_handoff
from hyperloom.orchestrator.kernel.kernel_context import (
    ArtifactRef,
    EvidenceIndex,
    KernelContext,
    ServingFacts,
    WorkloadFacts,
)


def test_the_source_roots_the_handoff_names_are_the_trees_the_analyst_may_read(tmp_path: Path) -> None:
    """A repository the handoff names has to become a directory the agent can open."""
    framework_repo = tmp_path / "sglang"
    (framework_repo / "python" / "sglang").mkdir(parents=True)
    aiter_repo = tmp_path / "aiter"
    aiter_repo.mkdir()
    candidates = tmp_path / "artifacts" / "kernel_candidates.json"
    candidates.parent.mkdir()
    candidates.write_text("[]\n", encoding="utf-8")

    context = KernelContext(
        session_dir=tmp_path / "session",
        workload=WorkloadFacts(model_name="example/model", gpu_type="mi355x"),
        serving=ServingFacts(
            framework="sglang",
            repository_roots=(str(framework_repo), str(aiter_repo)),
        ),
        evidence=EvidenceIndex(kernel_candidates=ArtifactRef.of(candidates)),
    )

    handoff_dir = write_forge_handoff(context, tmp_path / "attempt" / "handoff")
    additional = _additional_directories(read_handoff(handoff_dir))

    assert str(framework_repo.resolve()) in additional
    assert str(aiter_repo.resolve()) in additional
    # A file reference widens to the directory holding it, so the analyst can
    # read the sibling artifacts TraceLens wrote beside it.
    assert str(candidates.parent.resolve()) in additional


def test_a_named_root_that_is_not_on_disk_is_not_offered_to_the_agent(tmp_path: Path) -> None:
    context = KernelContext(
        session_dir=tmp_path / "session",
        serving=ServingFacts(framework="sglang", repository_roots=(str(tmp_path / "never-cloned"),)),
    )

    handoff_dir = write_forge_handoff(context, tmp_path / "attempt" / "handoff")
    additional = _additional_directories(read_handoff(handoff_dir))

    assert str((tmp_path / "never-cloned").resolve()) not in additional
