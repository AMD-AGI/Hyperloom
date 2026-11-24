# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PD default table composes the per-role launch for a model and refuses models it has no row for."""

from __future__ import annotations

import pytest

from inference_suite.frameworks import Framework
from inference_suite.multinode.table import RoleShape, TableError, TableQuery, compose


def test_qwen35_matches_the_hyperloom_baseline() -> None:
    """The per-role args Hyperloom's ``baseline_args.py`` composed for Qwen3.5-397B-A17B-FP8 at TP 8."""
    hit = compose(Framework.SGLANG, TableQuery("Qwen3.5-397B-A17B-FP8", RoleShape(8), RoleShape(8)))
    graph = [str(n) for n in range(1, 129)]
    base = [
        *("--decode-log-interval", "1000", "--log-level", "warning", "--watchdog-timeout", "3600"),
        *("--load-balance-method", "round_robin", "--kv-cache-dtype", "fp8_e4m3", "--attention-backend", "aiter"),
        *("--disaggregation-transfer-backend", "mori", "--moe-dense-tp-size", "1"),
    ]
    assert list(hit.prefill.args) == [
        *("--tp-size", "8", *base, "--mem-fraction-static", "0.8", "--max-running-requests", "128"),
        *("--chunked-prefill-size", "262144", "--cuda-graph-bs", *graph, "--disable-radix-cache"),
    ]
    assert list(hit.decode.args) == [
        *("--tp-size", "8", *base, "--mem-fraction-static", "0.85", "--max-running-requests", "128"),
        *("--cuda-graph-bs", *graph, "--prefill-round-robin-balance"),
    ]


def test_unknown_model_is_refused() -> None:
    with pytest.raises(TableError, match="no row"):
        compose(Framework.SGLANG, TableQuery("NoSuchModel", RoleShape(8), RoleShape(8)))


def test_agentic_keeps_the_prefill_radix_cache() -> None:
    hit = compose(Framework.SGLANG, TableQuery("DeepSeek-R1", RoleShape(8), RoleShape(8), agentic=True))
    assert "--disable-radix-cache" not in hit.prefill.args
