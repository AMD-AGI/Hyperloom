# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pinned TraceLens revision must name split files without a colon.

Older TraceLens revisions wrote ``<base>_iteration_<start>:<end>.json.gz`` when the
splitter extracted several iterations. A colon is rejected by some network
filesystems (``OSError: [Errno 22]``), which fails ``trace_split`` and with it the
whole trace analysis. TraceLens PR #1120 switched the separator to a hyphen.
"""

from __future__ import annotations

import re
from pathlib import Path

from hyperloom.orchestrator.trace_analysis import tracelens_analysis


REPO_ROOT = Path(__file__).resolve().parents[5]
INSTALL_KERNEL_TOOLS = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install_kernel_tools.sh"

# TraceLens commit that switched the split separator from a colon to a hyphen.
TRACELENS_PIN = "c2e65473cdac421d2905b5960532f73cfcc94786"


def _install_script_pin() -> str:
    text = INSTALL_KERNEL_TOOLS.read_text(encoding="utf-8")
    match = re.search(r'^TRACELENS_REF="([0-9a-f]{40})"$', text, re.M)
    assert match, "could not locate the TRACELENS_REF pin in install_kernel_tools.sh"
    return match.group(1)


def test_install_script_pins_the_hyphen_split_filename_revision() -> None:
    assert _install_script_pin() == TRACELENS_PIN


def test_python_default_matches_the_install_script_pin() -> None:
    assert tracelens_analysis._TRACELENS_REF_DEFAULT == _install_script_pin()
