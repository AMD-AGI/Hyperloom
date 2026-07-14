# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Package execution entry point for ``python -m hyperloom.inference_optimizer.cli``."""

from __future__ import annotations

import sys

from . import main


if __name__ == "__main__":
    sys.exit(main())
