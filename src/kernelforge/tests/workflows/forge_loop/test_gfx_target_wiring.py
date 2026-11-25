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
_RDNA4_NOTE = "## Target architecture: gfx1201 (RDNA 4"


@pytest.mark.parametrize(
    ("requested", "gpu_target", "analysis", "implementer"),
    [
        (True, "gfx950", True, True),
        (True, "gfx942", True, True),
        (True, "gfx1100", False, True),
        (True, "gfx1201", False, True),
        (True, "gfx1201:sramecc-:xnack-", False, True),
        (True, "r9700", False, True),
        (False, "gfx950", False, False),
        (False, "gfx1100", False, False),
        (False, "gfx1201", False, False),
    ],
)
def test_a_profiling_request_follows_the_targets_profiler(requested, gpu_target, analysis, implementer, capsys):
    """The profiled Analysis tier needs rocprof-compute; the implementer self-profiles with whatever the target has."""
    plan = _profiling_plan(requested, gpu_target)

    assert (plan.analysis, plan.implementer) == (analysis, implementer)
    said = capsys.readouterr().out
    assert ("profiles with rocprofv3" in said) is (requested and implementer and not analysis)


@pytest.mark.parametrize(
    ("gpu_target", "note", "hardware"),
    [("gfx1100", _RDNA3_NOTE, "hardware/rdna3"), ("gfx1201", _RDNA4_NOTE, "hardware/rdna4"), ("gfx950", "", "")],
)
def test_the_implementer_prompt_carries_the_targets_architecture_note(gpu_target, note, hardware):
    prompt = build_single_kernel_backend_prompt(Config(gpu_target=gpu_target), "triton")

    if note:
        assert note in prompt
        assert hardware in prompt
    else:
        assert _RDNA3_NOTE not in prompt
        assert _RDNA4_NOTE not in prompt


@pytest.mark.parametrize(
    ("gpu_target", "note", "hardware"),
    [("gfx1100", _RDNA3_NOTE, "hardware/rdna3"), ("gfx1201", _RDNA4_NOTE, "hardware/rdna4"), ("gfx950", "", "")],
)
def test_the_evidence_knowledge_index_carries_the_targets_architecture_note(gpu_target, note, hardware):
    owner = SimpleNamespace(
        config=SimpleNamespace(
            gpu_target=gpu_target, local_knowledge_dir=str(resource_path("local_knowledge")), defer_knowledge_maps=True
        ),
        ic=SimpleNamespace(kernel_backend="triton", backend="triton"),
        _target_source_files=lambda: ["kernel.py"],
    )

    index = AnalysisEvidenceMixin._knowledge_index(owner)

    if note:
        assert note in index
        assert hardware in index
    else:
        assert _RDNA3_NOTE not in index
        assert _RDNA4_NOTE not in index
