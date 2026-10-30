# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile-trace analysis: the agent and non-llm TraceLens routes, roofline, and trace self-certification."""

from __future__ import annotations

from ._capture_shapes import is_capture_fragment

__all__ = ["is_capture_fragment"]
