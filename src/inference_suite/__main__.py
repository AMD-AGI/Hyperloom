# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m inference_suite``; see :mod:`inference_suite.cli`."""

from __future__ import annotations

import sys

from inference_suite.cli import main

if __name__ == "__main__":
    sys.exit(main())
