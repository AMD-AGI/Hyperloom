# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR/ref candidate records and the search request that enumerates them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRMonitorConfig:
    """Connection settings for the pr_monitor service."""

    base_url: str
    timeout_sec: float = 10.0


@dataclass(frozen=True)
class Candidate:
    """A single PR or git ref candidate (explicit, pr_monitor, or GitHub)."""

    ref: str
    repo: str
    source: str = "explicit"
    title: str = ""
    html_url: str = ""
    score: float = 0.0

    @property
    def slug(self) -> str:
        """Filesystem-safe slug derived from ref (used for candidate_dir name)."""
        out = []
        for ch in self.ref.lower():
            if ch.isalnum():
                out.append(ch)
            elif ch in (".", "-", "_"):
                out.append(ch)
            else:
                out.append("-")
        slug = "".join(out).strip("-")
        return slug or "candidate"


@dataclass(frozen=True)
class CandidateSearchRequest:
    """What :func:`sources.enumerate_candidates` looks for in one repo."""

    repo_url: str
    max_search_candidates: int = 5
    pr_monitor: PRMonitorConfig | None = None
    # Title keywords for the pr_monitor search and rerank; empty takes the unranked listing.
    keywords: tuple[str, ...] = ()
    search_modes: tuple[str, ...] = ("pr_monitor", "github")
    # PR states to include in discovery.
    pr_states: tuple[str, ...] = ("open",)
