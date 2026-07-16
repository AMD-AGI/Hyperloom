# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Module entry-point so ``python3 -m hyperloom.inference_optimizer.multi_node`` works.

All real logic lives in :mod:`hyperloom.inference_optimizer.multi_node.cli`.
"""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
