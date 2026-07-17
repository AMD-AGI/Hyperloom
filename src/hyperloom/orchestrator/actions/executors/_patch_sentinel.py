# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared sentinel-substring "already patched?" probe for the in-place patchers."""

from __future__ import annotations

import logging
from pathlib import Path


def file_contains_sentinel(src: Path, sentinel: str, log: logging.Logger, label: str) -> bool:
    """Return whether *src* contains *sentinel*; warn + return False on read error."""
    try:
        return sentinel in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("%s: cannot read %s: %s", label, src, e)
        return False
