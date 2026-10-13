# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The planners are told which new files the campaign admits.

Planning saw only the editable source set, so with an empty allowlist every lane of a round planned around a new
config JSON that the workspace guard then refused -- the round was lost before a candidate was measured.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernelforge.orchestrator.analysis import AnalysisBundleError
from kernelforge.tests.test_analysis_agent import _BundleBackend, _service, _SourceThenFailBackend, _workspace
from kernelforge.tests.test_analysis_agent import _context as _analysis_context
from kernelforge.tests.test_forge_orchestration import _context
from kernelforge.tests.test_loop_runner import RunState, _make_loop

ADMITTED = ("vllm_fused_moe/configs/*.json",)


def test_an_empty_allowlist_tells_the_planner_no_new_files() -> None:
    new_files = _context().to_prompt_dict()["new_files"]
    assert new_files["allowlist"] == []
    assert "No new files are admitted" in new_files["policy"]


def test_a_declared_allowlist_reaches_the_planner() -> None:
    new_files = replace(_context(), new_file_allowlist=ADMITTED).to_prompt_dict()["new_files"]
    assert new_files["allowlist"] == list(ADMITTED)
    assert "only where they match" in new_files["policy"]


def test_the_allowlist_rejects_blank_entries() -> None:
    with pytest.raises(ValueError, match="new_file_allowlist"):
        replace(_context(), new_file_allowlist=("  ",))


def test_the_runner_publishes_the_campaign_allowlist(tmp_path, monkeypatch) -> None:
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.run_state = RunState(head_commit=loop._git("rev-parse", "HEAD").splitlines()[0])
    loop.config = SimpleNamespace(experiments_dir=workspace / "forge_experiments", gpu_target="gfx942")
    loop.ic.commit_new_paths = list(ADMITTED)

    assert loop._build_orchestration_context().new_file_allowlist == ADMITTED


async def test_a_published_bundle_keeps_the_allowlist(tmp_path) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    service = _service(tmp_path, _BundleBackend())
    context = replace(_analysis_context(workspace), new_file_allowlist=ADMITTED)

    bundle = await service.ensure_bundle(
        context, kernel_file=str(kernel), driver_script=str(driver), source_files=[str(kernel)]
    )

    assert bundle.apply(context).new_file_allowlist == ADMITTED


async def test_a_partial_checkpoint_keeps_the_allowlist(tmp_path) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    service = _service(tmp_path, _SourceThenFailBackend())
    context = replace(_analysis_context(workspace), new_file_allowlist=ADMITTED)

    with pytest.raises(AnalysisBundleError):
        await service.ensure_bundle(
            context, kernel_file=str(kernel), driver_script=str(driver), source_files=[str(kernel)]
        )

    assert service.apply_checkpoint(context).new_file_allowlist == ADMITTED
