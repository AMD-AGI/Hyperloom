# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the bypass server-log throughput parser (pure functions, no GPU)."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors import bypass_analysis as ba


def test_parse_server_log_throughput_sglang_and_vllm():
    text = "\n".join(
        [
            "Decode batch. #running-req: 4 gen throughput (token/s): 100.0",
            "Decode batch. #running-req: 4 gen throughput (token/s): 200.5",
            "Avg generation throughput: 300 tokens/s",
            "irrelevant line",
            "gen throughput (token/s): 0",  # non-positive dropped
        ]
    )
    samples = ba.parse_server_log_throughput(text)
    assert samples == [100.0, 200.5, 300.0]
