# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROOFLINE EVIDENCE names vendor-backend substitution for hot ATen ops.

A high-GPU-share ``aten::`` op is dispatching into some backend library. The
specialist may not rewrite that library's kernel body, but the call site in its
own worktree chooses the backend, and swapping it is an ordinary source rewrite.
Without this the evidence reads as "attention is 60% of device time" with no
indication that anything may be done about it, and specialists have gone after
1%-scale glue instead.
"""

from __future__ import annotations

from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_roofline_evidence,
    _vendor_substitution_candidates,
)
from hyperloom.orchestrator.specialists.domains import get_domain


def _inp(hot_kernels: list[dict]) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("framework_rewrite_specialist"),
        gap_canonical_id="local_explore",
        roofline_evidence={
            "analysis_md_path": "/sd/analysis.md",
            "roofline_snapshot_id": 2,
            "executive_summary": {},
            "hot_kernels_top15": hot_kernels,
        },
    )


def _sdpa_row(gpu_pct: float = 29.8) -> dict:
    return {
        "kernel_id": "k2",
        "name": "aten::_scaled_dot_product_flash_attention",
        "kernel_category": "SDPA",
        "gpu_pct": gpu_pct,
        "bottleneck": "compute",
        "source_file": "hyvideo/models/transformers/modules/attention.py(243): sequence_parallel_attention_vision",
    }


def test_hot_aten_op_is_a_substitution_candidate():
    """A material ``aten::`` op is surfaced with its call site."""
    rows = _vendor_substitution_candidates([_sdpa_row()])
    assert len(rows) == 1
    assert rows[0]["name"] == "aten::_scaled_dot_product_flash_attention"
    assert "attention.py" in rows[0]["source_file"]


def test_immaterial_ops_are_not_candidates():
    """Sub-threshold ops stay out; the point is to redirect attention, not add noise."""
    glue = {
        "kernel_id": "k9",
        "name": "aten::cat",
        "kernel_category": "Other",
        "gpu_pct": 1.66,
        "source_file": "hyvideo/models/transformers/modules/attention.py(243)",
    }
    assert _vendor_substitution_candidates([glue]) == []


def test_non_aten_kernels_are_not_candidates():
    """Already-vendor kernels have no backend left to swap."""
    row = {
        "kernel_id": "k3",
        "name": "fmha_v3_fwd_kernel",
        "kernel_category": "SDPA",
        "gpu_pct": 40.0,
        "source_file": "/sgl-workspace/aiter/aiter/ops/mha.py",
    }
    assert _vendor_substitution_candidates([row]) == []


def test_candidates_are_ordered_by_gpu_share():
    """The largest share leads, so the specialist reads the biggest lever first."""
    rows = _vendor_substitution_candidates(
        [
            {"name": "aten::addmm", "gpu_pct": 6.0, "kernel_category": "GEMM", "source_file": "a.py"},
            _sdpa_row(29.8),
        ]
    )
    assert [r["name"] for r in rows] == [
        "aten::_scaled_dot_product_flash_attention",
        "aten::addmm",
    ]


def test_section_renders_substitution_directive():
    """The rendered section states the rule, names the op, and points at the install roots."""
    text = "\n".join(_section_roofline_evidence(_inp([_sdpa_row()])))

    assert "aten::_scaled_dot_product_flash_attention" in text
    assert "29.80%" in text
    # The rule the specialist was missing: the body is off-limits, the call site is not.
    assert "call site" in text.lower()
    assert "Section 7" in text
    # Discovery, not an answer key: no specific library is prescribed.
    assert "aiter" not in text.lower()


def test_section_omits_directive_without_candidates():
    """No material ATen op → no directive, so the section stays evidence-only."""
    text = "\n".join(
        _section_roofline_evidence(_inp([{"name": "fmha_v3_fwd_kernel", "gpu_pct": 40.0, "source_file": "x.py"}]))
    )
    assert "call site" not in text.lower()


def test_section_still_renders_none_when_no_evidence():
    """Unchanged behaviour when there is no snapshot at all."""
    inp = SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("framework_rewrite_specialist"),
        gap_canonical_id="local_explore",
        roofline_evidence={},
    )
    text = "\n".join(_section_roofline_evidence(inp))
    assert "(none" in text
