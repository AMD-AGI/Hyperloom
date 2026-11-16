# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign's gfx target reaches every forge-loop consumer that reads its ArchProfile."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernelforge.config import Config
from kernelforge.config.resources import resource_path
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.workflows.forge_loop.bootstrap.run import _profiling_plan
from kernelforge.workflows.forge_loop.roles.analysis_evidence import AnalysisEvidenceMixin

_RDNA3_NOTE = "## Target architecture: gfx1100 (RDNA 3"


@pytest.mark.parametrize(
    ("requested", "gpu_target", "analysis", "implementer"),
    [
        (True, "gfx950", True, True),
        (True, "gfx942", True, True),
        (True, "gfx1100", False, True),
        (False, "gfx950", False, False),
        (False, "gfx1100", False, False),
    ],
)
def test_a_profiling_request_follows_the_targets_profiler(requested, gpu_target, analysis, implementer, capsys):
    """The profiled Analysis tier needs rocprof-compute; the implementer self-profiles with whatever the target has."""
    plan = _profiling_plan(requested, gpu_target)

    assert (plan.analysis, plan.implementer) == (analysis, implementer)
    said = capsys.readouterr().out
    assert ("profiles with rocprofv3" in said) is (requested and gpu_target == "gfx1100")


@pytest.mark.parametrize(("gpu_target", "noted"), [("gfx1100", True), ("gfx950", False)])
def test_the_implementer_prompt_carries_the_targets_architecture_note(gpu_target, noted):
    prompt = build_single_kernel_backend_prompt(Config(gpu_target=gpu_target), "triton")

    assert (_RDNA3_NOTE in prompt) is noted
    assert ("hardware/rdna3" in prompt) is noted


@pytest.mark.parametrize(("gpu_target", "noted"), [("gfx1100", True), ("gfx950", False)])
def test_the_orchestration_knowledge_index_carries_the_targets_architecture_note(gpu_target, noted):
    owner = SimpleNamespace(
        config=SimpleNamespace(
            gpu_target=gpu_target, local_knowledge_dir=str(resource_path("local_knowledge")), defer_knowledge_maps=True
        ),
        ic=SimpleNamespace(kernel_backend="triton", backend="triton"),
        _target_source_files=lambda: ["kernel.py"],
    )

    index = AnalysisEvidenceMixin._orchestration_knowledge_index(owner)

    assert (_RDNA3_NOTE in index) is noted
