# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The refusals of the plug kernel."""

from __future__ import annotations


class PlugError(Exception):
    """A socket cannot be filled: the manifest names something missing, cyclic or unknown."""


class ManifestError(PlugError):
    """A manifest, override file, command-line setting or resolved file is not acceptable."""
