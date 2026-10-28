# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SpecialistPromptInputs.roofline_evidence + ROOFLINE EVIDENCE section tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.specialists.domains import (
    get_domain,
)
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_roofline_evidence,
    build_specialist_prompts,
)


def _make_coord(tmp_path: Path, *, state: SharedState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_warm_specialist_params_injects_roofline_evidence(tmp_path):
    """``warm_specialist_params`` mirrors ``last_trace_analyze`` into ``params['roofline_evidence']``."""
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

    state = SharedState(
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
    await coord.specialist_dispatch.warm_specialist_params(params)

    assert "roofline_evidence" in params
    ev = params["roofline_evidence"]
    assert ev["analysis_md_path"] == str(analysis_path)
    assert ev["roofline_snapshot_id"] == 3
    assert len(ev["hot_kernels_top15"]) == 8
    assert ev["hot_kernels_top15"][0]["kernel_id"] == "k1"
    assert ev["executive_summary"]["compute_pct"] == pytest.approx(32.1, rel=0.01)
    assert ev["executive_summary"]["idle_pct"] == pytest.approx(17.5, rel=0.01)
    assert ev["executive_summary"]["comm_pct"] == pytest.approx(41.8, rel=0.01)
    assert ev["executive_summary"]["top_bottleneck"] == "rccl_AllReduce"


@pytest.mark.asyncio
async def test_warm_specialist_params_noop_when_no_snapshot(tmp_path):
    """No ``last_trace_analyze`` → no ``roofline_evidence`` key."""
    state = SharedState(last_trace_analyze={})
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "serving_specialist"}
    await coord.specialist_dispatch.warm_specialist_params(params)
    assert "roofline_evidence" not in params


@pytest.mark.asyncio
async def test_warm_specialist_params_noop_when_analysis_md_text_empty(tmp_path):
    """Empty ``analysis_md_text`` is treated as no-snapshot."""
    state = SharedState(
        last_trace_analyze={
            "analysis_md_text": "",
            "analysis_md_path": "/dev/null",
        },
    )
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "kernel_switch_specialist"}
    await coord.specialist_dispatch.warm_specialist_params(params)
    assert "roofline_evidence" not in params


@pytest.mark.asyncio
async def test_warm_specialist_params_packs_hot_kernels_without_analysis_md(tmp_path):
    """Hot kernels with no analysis.md still reach the specialist."""
    state = SharedState(
        last_trace_analyze={
            "analysis_md_text": "",
            "analysis_md_path": "",
            "roofline_snapshot_id": 4,
            "hot_kernels_top15": [
                {
                    "kernel_id": "k1",
                    "name": "aten::_flash_attention_forward",
                    "gpu_pct": 40.0,
                    "bottleneck": "compute",
                    "source_file": "attention.py(243)",
                },
            ],
        },
    )
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "framework_rewrite_specialist"}
    await coord.specialist_dispatch.warm_specialist_params(params)

    ev = params["roofline_evidence"]
    assert ev["roofline_snapshot_id"] == 4
    assert ev["hot_kernels_top15"][0]["name"] == "aten::_flash_attention_forward"
    assert ev["executive_summary"] == {}


@pytest.mark.asyncio
async def test_warm_specialist_params_respects_existing_evidence(tmp_path):
    """A caller-supplied ``roofline_evidence`` is not overwritten (setdefault)."""
    analysis_path = tmp_path / "analysis.md"
    analysis_path.write_text("# stub\n", encoding="utf-8")
    state = SharedState(
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
    await coord.specialist_dispatch.warm_specialist_params(params)
    assert params["roofline_evidence"] == {"sentinel": True}


def _verify_capture(tmp_path: Path) -> Path:
    from kernelforge.tests.trace_fixtures import kernel, step, write_trace

    events = [
        step("step[VERIFY bs=2]", 0.0, 100.0),
        kernel("mfma_moe1_silu_mul_afp4_wfp4", 1.0, 60.0),
        kernel("ck_gemm_a8w8_blockscale_b_preshuffle", 62.0, 40.0),
    ]
    return write_trace(tmp_path / "r-TP-0-EP-0.trace.json.gz", events)


@pytest.mark.asyncio
async def test_warm_specialist_params_reads_the_decode_modules_off_the_raw_capture(tmp_path):
    """The VERIFY graph replay TraceLens cannot split still reaches the specialist, by module, from the capture."""
    state = SharedState(last_trace_analyze={}, last_profile_trace=str(_verify_capture(tmp_path)))
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {"domain": "framework_rewrite_specialist"}

    await coord.specialist_dispatch.warm_specialist_params(params)

    assert params["roofline_evidence"] == {
        "decode_modules": [
            "| module | VERIFY bs=2 | top kernel |",
            "|---|---:|---|",
            "| moe | 0.060 ms (60.0%) | `mfma_moe1_silu_mul_afp4_wfp4` |",
            "| gemm | 0.040 ms (40.0%) | `ck_gemm_a8w8_blockscale_b_preshuffle` |",
        ]
    }


def _kept_stack_capture(tmp_path: Path) -> Path:
    from kernelforge.tests.trace_fixtures import kernel, step, write_trace

    events = [step("step[VERIFY bs=4]", 0.0, 100.0), kernel("cross_device_reduce_1stage", 1.0, 50.0)]
    (tmp_path / "attribution").mkdir()
    return write_trace(tmp_path / "attribution" / "p-TP-0.trace.json.gz", events)


_OPENING_TABLE = [
    "| module | VERIFY bs=2 | top kernel |",
    "|---|---:|---|",
    "| moe | 0.060 ms (60.0%) | `mfma_moe1_silu_mul_afp4_wfp4` |",
    "| gemm | 0.040 ms (40.0%) | `ck_gemm_a8w8_blockscale_b_preshuffle` |",
]
_KEPT_TABLE = [
    "| module | VERIFY bs=4 | top kernel |",
    "|---|---:|---|",
    "| allreduce/comm | 0.050 ms (100.0%) | `cross_device_reduce_1stage` |",
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("held_stack_len", "readable", "table"),
    [(1, True, _KEPT_TABLE), (0, True, _OPENING_TABLE), (1, False, _OPENING_TABLE)],
)
async def test_the_module_table_comes_from_the_current_stacks_capture_before_the_opening_profile(
    tmp_path, held_stack_len, readable, table
):
    capture = str(_kept_stack_capture(tmp_path)) if readable else str(tmp_path / "missing.trace.json.gz")
    state = SharedState(
        last_trace_analyze={},
        last_profile_trace=str(_verify_capture(tmp_path)),
        optimization_stack=[{"kind": "config"}],
        last_attribution_trace={"stack_len": held_stack_len, "trace": capture},
    )
    params: dict[str, Any] = {"domain": "framework_rewrite_specialist"}

    await _make_coord(tmp_path, state=state).specialist_dispatch.warm_specialist_params(params)

    assert params["roofline_evidence"] == {"decode_modules": table}


_OWNED_MOE = {
    "group": "moe",
    "rows": [
        "| module | VERIFY bs=2 | top kernel |",
        "|---|---:|---|",
        "| moe | 0.060 ms (60.0%) | `mfma_moe1_silu_mul_afp4_wfp4` |",
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("switch", "evidence"),
    [
        ("1", {"decode_modules": _OPENING_TABLE, "owned_module": _OWNED_MOE}),
        ("0", {"decode_modules": _OPENING_TABLE}),
    ],
)
async def test_a_specialist_owning_a_module_group_is_warmed_with_that_groups_row(
    tmp_path, monkeypatch, switch, evidence
):
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", switch)
    state = SharedState(last_trace_analyze={}, last_profile_trace=str(_verify_capture(tmp_path)))
    params: dict[str, Any] = {"scope": "freeform", "task_description": "fuse the MoE sort", "module_group": "moe"}

    await _make_coord(tmp_path, state=state).specialist_dispatch.warm_specialist_params(params)

    assert params["roofline_evidence"] == evidence


def test_the_mandate_names_the_owned_module_group_with_its_figures():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import _section_mandate

    rows = _section_mandate(_make_inp({"owned_module": _OWNED_MOE}))

    start = rows.index("- module group: `moe` (no other specialist works it while you run); its decode time per step:")
    assert rows[start : start + 5] == [rows[start], "", *_OWNED_MOE["rows"]]


def test_an_owned_group_alone_is_not_roofline_evidence():
    assert _section_roofline_evidence(_make_inp({"owned_module": _OWNED_MOE})) == _section_roofline_evidence(
        _make_inp({})
    )


def test_section_renders_the_decode_modules_ahead_of_the_tracelens_kernels():
    table = ["| module | VERIFY bs=2 | top kernel |", "|---|---:|---|", "| moe | 0.060 ms (60.0%) | `m` |"]
    inp = _make_inp({"decode_modules": table, "hot_kernels_top15": [{"kernel_id": "k1", "name": "topk"}]})

    section = _section_roofline_evidence(inp)

    start = section.index(table[0])
    assert section[start : start + 3] == table
    assert start < section.index("| kernel_id | name | gpu_pct | bottleneck | source_file |")


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
                "memcpy_pct": 9.2,
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
    assert "- Exposed memcpy %: 9.2%" in section
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
