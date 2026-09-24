# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared types and helpers across PR source backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GitHubPr:
    """Lightweight PR record returned by any PR source backend."""

    number: int
    title: str
    html_url: str

    @property
    def ref(self) -> str:
        """Stable candidate ref used downstream (`Candidate.ref`)."""
        return f"PR:{self.number}"


__all__ = ["GitHubPr"]
