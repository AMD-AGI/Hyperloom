# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR/ref candidate records and the search request that enumerates them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A single PR or git ref candidate."""

    ref: str
    title: str = ""
    html_url: str = ""

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
    """What a pull-request search looks for in one repo."""

    repo_url: str
    max_search_candidates: int = 5
    keywords: tuple[str, ...] = ()
