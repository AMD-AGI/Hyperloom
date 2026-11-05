# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where an installed aiter keeps its device sources, found the way ``aiter/jit/core.py`` finds them, without importing it.

Importing ``aiter.jit.core`` probes the GPU with ``rocminfo``, opens every card the process can see and raises on a host
with none, so a process that only needs these paths must not import it.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.outside_facts import AITER_DEVELOP_MODE, AITER_INSTALL_MODE_FILE, AITER_META_WHEEL_DIR

__all__ = ["image_aiter_meta"]


def image_aiter_meta(package: Path) -> Path:
    """The meta directory the image's own aiter reads ``csrc`` from, resolved as ``aiter/jit/core.py`` resolves it.

    ``package`` is the aiter package directory the image's interpreter finds. A develop install (what the
    ``install_mode`` file beside the package says, and what a missing file means) keeps ``csrc`` in the directory around
    the package; a wheel install keeps it in the ``aiter_meta`` package beside it.
    """
    try:
        develop = (package / AITER_INSTALL_MODE_FILE).read_text(encoding="utf-8").strip() == AITER_DEVELOP_MODE
    except FileNotFoundError:
        develop = True
    return package.parent if develop else package.parent / AITER_META_WHEEL_DIR
