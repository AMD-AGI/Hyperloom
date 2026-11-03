# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Every kernelforge command is reachable from the `kernelforge` entry point."""

from __future__ import annotations

import kernelforge.cli as cli

EXPECTED_COMMANDS = {
    "export-runs",
    "forge-fuse",
    "forge-loop",
    "forge-rewrite-by-flydsl",
    "gemm-tune",
    "kernel-rewrite-controller",
    "roofline-ceiling",
}


def test_every_command_is_registered():
    assert set(cli.main.commands) == EXPECTED_COMMANDS
