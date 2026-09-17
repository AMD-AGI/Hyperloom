# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stable comparison schemas for synthetic and agentic reference data."""

from typing import get_args, get_type_hints

from hyperloom.inference_optimizer.baseline_comparison.types import BaselinePoint, BaselineQuery, BaselineSummary


def test_reason_annotation_covers_all_summary_outcomes():
    hints = get_type_hints(BaselineSummary)
    assert set(get_args(hints["reason"])) == {
        "",
        "ok",
        "model_mapping_miss",
        "no_target_gpu_configured",
        "fetch_error",
        "no_inferencex_data",
        "unsupported_target_gpu",
        "precision_mismatch",
        "dimension_mismatch",
        "no_valid_rows",
    }
    assert hints["status"] is str


def test_query_defaults_preserve_synthetic_dimensions():
    query = BaselineQuery("MiniMax-M2.5", "b300", "vllm", "fp8", 1024, 2048)
    assert query.to_dict() == {
        "model": "MiniMax-M2.5",
        "gpu": "b300",
        "framework": "vllm",
        "precision": "fp8",
        "isl": 1024,
        "osl": 2048,
        "benchmark_mode": "synthetic",
    }


def test_agentx_query_serializes_variable_lengths():
    query = BaselineQuery("GLM-5.2", "b300", isl=None, osl=None, benchmark_mode="agentx")
    assert query.to_dict()["benchmark_mode"] == "agentx"
    assert query.to_dict()["isl"] is None
    assert query.to_dict()["osl"] is None


def test_point_defaults_do_not_reconstruct_p90_from_mean_tpot():
    point = BaselinePoint(100.0, 10.0, 4, 8, 1.0, 10.0, 100.0)
    data = point.to_dict()
    assert data["tput_per_gpu"] == 100.0
    assert data["mean_tpot_ms"] == 10.0
    assert data["e2e_norm_intvty_p90"] is None
    assert data["benchmark_id"] is None


def test_summary_preserves_reference_id_and_exact_p90_without_renormalizing():
    query = BaselineQuery("GLM-5.2", "b300", isl=None, osl=None, benchmark_mode="agentx")
    point = BaselinePoint(
        tput_per_gpu=2643.69293,
        output_tput_per_gpu=18.7487,
        conc=1,
        decode_tp=8,
        mean_ttft_ms=1.0,
        mean_tpot_ms=10.0,
        mean_e2el_ms=100.0,
        benchmark_id="439985",
        e2e_norm_intvty_p90=12.345,
    )
    summary = BaselineSummary(query, "2026-09-10T00:00:00Z", 1, point, [point])
    data = summary.to_dict()
    assert data["best"]["benchmark_id"] == "439985"
    assert data["best"]["e2e_norm_intvty_p90"] == 12.345
    assert data["best"]["tput_per_gpu"] == 2643.69293
    assert data["all_concurrencies"] == [data["best"]]
