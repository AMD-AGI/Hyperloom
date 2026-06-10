# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SpecialistPromptInputs.roofline_evidence + ROOFLINE EVIDENCE section tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.specialist_domains import (
    get_domain,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_roofline_evidence,
    build_specialist_prompts,
)


# Coordinator-warmer integration
@dataclass
class _BareState:
    """Minimal SharedState double used by _warm_specialist_params."""

    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    gpu_type: str = ""
    tp: int = 0
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def find_gap(self, _cid: str):
        return None


def _make_coord(tmp_path: Path, *, state: _BareState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_warm_specialist_params_injects_roofline_evidence(tmp_path):
    """``_warm_specialist_params`` mirrors ``last_trace_analyze`` into ``params['roofline_evidence']``."""
    # Synthetic analysis.md so extract_workload_summary can parse the table.
    analysis_path = tmp_path / "analysis.md"
    analysis_path.write_text(
        "## Executive Summary\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Compute % | 32.1% |\n"
        "| Idle % | 17.5% |\n"
        "| Exposed Communication % | 41.8% |\n"
        "| Top Bottleneck Category | rccl_AllReduce (41.8%) |\n",
        encoding="utf-8",
    )

    state = _BareState(
        last_trace_analyze={
            "analysis_md_text": "FULL ANALYSIS TEXT",
            "analysis_md_path": str(analysis_path),
            "roofline_snapshot_id": 3,
            "hot_kernels_top15": [
                {
                    "kernel_id": "k1",
                    "name": "topk_softmax",
                    "gpu_pct": 8.4,
                    "bottleneck": "compute",
                    "source_file": "/sgl-workspace/aiter/aiter/ops/topk.py",
                },
                # 9 entries — the warmer should slice to top 8.
                *[
                    {
                        "kernel_id": f"k{i}",
                        "name": f"kernel_{i}",
                        "gpu_pct": float(i),
                        "bottleneck": "compute",
                        "source_file": "",
                    }
                    for i in range(2, 11)
                ],
            ],
        },
    )

    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "kernel_switch_specialist"}
    await coord._warm_specialist_params(params)

    assert "roofline_evidence" in params
    ev = params["roofline_evidence"]
    assert ev["analysis_md_path"] == str(analysis_path)
    assert ev["roofline_snapshot_id"] == 3
    # Sliced to top 8.
    assert len(ev["hot_kernels_top15"]) == 8
    assert ev["hot_kernels_top15"][0]["kernel_id"] == "k1"
    # Executive summary parsed from disk.
    assert ev["executive_summary"]["compute_pct"] == pytest.approx(32.1, rel=0.01)
    assert ev["executive_summary"]["idle_pct"] == pytest.approx(17.5, rel=0.01)
    assert ev["executive_summary"]["comm_pct"] == pytest.approx(41.8, rel=0.01)
    assert (
        ev["executive_summary"]["top_bottleneck"]
        == "rccl_AllReduce"
    )


@pytest.mark.asyncio
async def test_warm_specialist_params_noop_when_no_snapshot(tmp_path):
    """No ``last_trace_analyze`` → no ``roofline_evidence`` key."""
    state = _BareState(last_trace_analyze={})
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "serving_specialist"}
    await coord._warm_specialist_params(params)
    assert "roofline_evidence" not in params


@pytest.mark.asyncio
async def test_warm_specialist_params_noop_when_analysis_md_text_empty(tmp_path):
    """Empty ``analysis_md_text`` is treated as no-snapshot."""
    state = _BareState(
        last_trace_analyze={
            "analysis_md_text": "",
            "analysis_md_path": "/dev/null",
        },
    )
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "kernel_switch_specialist"}
    await coord._warm_specialist_params(params)
    assert "roofline_evidence" not in params


@pytest.mark.asyncio
async def test_warm_specialist_params_respects_existing_evidence(tmp_path):
    """A caller-supplied ``roofline_evidence`` is not overwritten (setdefault)."""
    analysis_path = tmp_path / "analysis.md"
    analysis_path.write_text("# stub\n", encoding="utf-8")
    state = _BareState(
        last_trace_analyze={
            "analysis_md_text": "stub",
            "analysis_md_path": str(analysis_path),
            "roofline_snapshot_id": 1,
            "hot_kernels_top15": [],
        },
    )
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {
        "domain": "comm_specialist",
        "roofline_evidence": {"sentinel": True},
    }
    await coord._warm_specialist_params(params)
    assert params["roofline_evidence"] == {"sentinel": True}


# Prompt rendering
def _make_inp(roofline_evidence: dict[str, Any]) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        gap_canonical_id="gap.x",
        roofline_evidence=roofline_evidence,
    )


def test_section_renders_executive_summary_and_hot_kernels():
    inp = _make_inp(
        {
            "analysis_md_path": "/sd/.../analysis.md",
            "roofline_snapshot_id": 7,
            "executive_summary": {
                "compute_pct": 30.0,
                "idle_pct": 15.0,
                "comm_pct": 40.0,
                "top_bottleneck": "MoE_fused",
            },
            "hot_kernels_top15": [
                {
                    "kernel_id": "k1",
                    "name": "topk",
                    "gpu_pct": 8.4,
                    "bottleneck": "compute",
                    "source_file": "/sgl/foo.py",
                },
            ],
        }
    )
    section = _section_roofline_evidence(inp)
    text = "\n".join(section)
    assert "## 4a. ROOFLINE EVIDENCE" in text
    assert "snapshot #7" in text
    assert "Compute %" in text and "30.0%" in text
    assert "MoE_fused" in text
    assert "k1" in text and "8.40%" in text
    assert "/sd/.../analysis.md" in text


def test_section_renders_none_when_evidence_empty():
    inp = _make_inp({})
    section = _section_roofline_evidence(inp)
    text = "\n".join(section)
    assert "## 4a. ROOFLINE EVIDENCE" in text
    assert "(none" in text


def test_build_specialist_prompts_inserts_section_between_kb_and_recipe():
    """build_specialist_prompts inserts section 4a between section 4 (KB) and section 5 (recipe)."""
    inp = _make_inp(
        {
            "analysis_md_path": "/sd/analysis.md",
            "executive_summary": {"compute_pct": 12.0},
            "hot_kernels_top15": [],
        }
    )
    _system, user = build_specialist_prompts(inp)
    kb_idx = user.index("## 4. KB CONTEXT (optional, advisory)")
    roof_idx = user.index("## 4a. ROOFLINE EVIDENCE")
    recipe_idx = user.index("## 5. WARM-START RECIPE SUMMARY")
    assert kb_idx < roof_idx < recipe_idx
