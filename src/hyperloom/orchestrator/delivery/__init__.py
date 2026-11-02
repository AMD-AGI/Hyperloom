# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delivery layer: what a round declared, and what its apply can undo."""

from __future__ import annotations

from hyperloom.orchestrator.delivery.ledger import file_digest

__all__ = ["file_digest"]
