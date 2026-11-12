# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.common.tracelens_report: locating analysis.json and reading its task members."""

from __future__ import annotations

import json

import pytest

from hyperloom.common.tracelens_report import analysis_json_path, fusion_findings, member_name

_ATTENTION_GEMM = "Cijk_Alik_Bljk_B_BS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x32..."
_SOFTMAX = "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<c10::BFloat1..."


@pytest.mark.parametrize(
    ("operation", "kernel_name", "expected"),
    [
        (" aiter::gemm_a16w16 ", ["Cijk_gemm_a"], "aiter::gemm_a16w16"),
        (None, ["", "Cijk_gemm_a", "Cijk_gemm_a_tail"], "Cijk_gemm_a"),
        (None, [], ""),
    ],
)
def test_a_member_is_named_by_its_task_operation_else_by_its_first_device_symbol(operation, kernel_name, expected):
    assert member_name({"operation": operation}, {"kernel_name": kernel_name}) == expected


def test_the_agent_route_report_is_the_json_rendered_beside_its_md(tmp_path):
    assert analysis_json_path(tmp_path / "analysis.md") == tmp_path / "analysis.json"
    assert analysis_json_path(tmp_path / "run" / "analysis.json") == tmp_path / "run" / "analysis.json"


def test_fusion_findings_keep_tracelens_order_and_member_kernel_order(tmp_path):
    report = {
        "compute_optimizations": [{"operation": "aten::addmm", "members": [], "impact": {"mid": 1.65}}],
        "fusion_optimizations": [
            {
                "operation": "Unfused Attention (QK^T GEMM + Softmax + PV GEMM) in `Attention`",
                "kind": "attention",
                "members": [
                    {"kernel_name": [_ATTENTION_GEMM], "kernel_launcher_path": "models/m/modeling.py(410): forward"},
                    {"kernel_name": [_SOFTMAX], "kernel_launcher_path": "torch/nn/functional.py(2103): softmax"},
                ],
                "impact": {"mid": 13.52},
            },
            {"operation": "Unrecognized run", "kind": None, "members": [], "impact": {"mid": None}},
        ],
    }
    (tmp_path / "analysis.json").write_text(json.dumps(report), encoding="utf-8")

    findings = fusion_findings(tmp_path / "analysis.md")

    assert findings == [
        {
            "kind": "attention",
            "operation": "Unfused Attention (QK^T GEMM + Softmax + PV GEMM) in `Attention`",
            "projected_e2e_pct": 13.52,
            "members": [
                {"kernel_names": [_ATTENTION_GEMM], "kernel_launcher_path": "models/m/modeling.py(410): forward"},
                {"kernel_names": [_SOFTMAX], "kernel_launcher_path": "torch/nn/functional.py(2103): softmax"},
            ],
        },
        {"kind": "", "operation": "Unrecognized run", "projected_e2e_pct": None, "members": []},
    ]


def test_a_report_without_fusion_findings_reads_as_none(tmp_path):
    (tmp_path / "analysis.json").write_text(json.dumps({"compute_optimizations": []}), encoding="utf-8")

    assert fusion_findings(tmp_path / "analysis.json") == []
    assert fusion_findings(tmp_path / "missing" / "analysis.md") == []
