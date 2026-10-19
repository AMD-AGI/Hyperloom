# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compatibility CLI for the exploratory single-case benchmark command."""

from __future__ import annotations

import sys

from kernelforge.evaluation.benchmark.runner import sweep_cli


if __name__ == "__main__":
    sys.exit(sweep_cli(prog="python3 -m kernelforge.mcp_server.tools.bench"))
