# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Entry point for the packaged launch collector deployed beside this asset."""

from _hyperloom_launch.serving_launch import main

if __name__ == "__main__":
    raise SystemExit(main())
