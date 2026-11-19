# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch (``qwen3_seam``) and its host side (``benchmark/forge_mono``) for the tests."""

import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK), str(TASK / "benchmark" / "forge_mono")]
