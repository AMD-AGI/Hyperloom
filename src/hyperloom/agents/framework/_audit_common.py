# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-export shim; canonical location is ``hyperloom.common.unified_diff``."""

from __future__ import annotations

from hyperloom.common.unified_diff import FileChange, _strip_diff_path, parse_unified_diff  # noqa: F401

__all__ = ["FileChange", "parse_unified_diff"]
