# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Module entry-point so ``python3 -m inference_optimizer.multi_node`` works.

All real logic lives in :mod:`inference_optimizer.multi_node.cli`.
"""
from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
