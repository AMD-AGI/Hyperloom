# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Periodic generation-throughput samples read off a bypass server log."""

from __future__ import annotations

import re

# sglang: "Decode batch. ... gen throughput (token/s): 1234.5" vllm: "Avg generation throughput: 1234.5 tokens/s"
_THROUGHPUT_PATTERNS = (
    re.compile(r"gen throughput \(token/s\):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"Avg generation throughput:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def parse_server_log_throughput(text: str) -> list[float]:
    """Extract every positive periodic generation-throughput sample."""
    samples: list[float] = []
    for line in text.splitlines():
        for pattern in _THROUGHPUT_PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                if value > 0:
                    samples.append(value)
                break
    return samples
