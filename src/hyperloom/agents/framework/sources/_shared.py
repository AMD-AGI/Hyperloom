# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared types and helpers across PR source backends."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

_GITHUB_HOST = "github.com"


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


def _repo_slug(repo_url: str) -> str:
    """Parse ``owner/name`` from a GitHub-style git URL."""
    raw = repo_url.strip()
    if raw.endswith(".git"):
        raw = raw[:-4]

    path: str
    if raw.startswith("git@github.com:"):
        path = raw.split(":", 1)[1]
    elif raw.startswith("ssh://git@github.com/"):
        parsed = urlparse(raw)
        if (parsed.hostname or "").lower() != _GITHUB_HOST:
            raise ValueError(f"cannot derive GitHub repo from repo_url={repo_url!r}")
        path = parsed.path
    else:
        candidate = raw if "://" in raw else f"https://{raw}"
        parsed = urlparse(candidate)
        if (parsed.hostname or "").lower() != _GITHUB_HOST:
            raise ValueError(f"cannot derive GitHub repo from repo_url={repo_url!r}")
        path = parsed.path

    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"cannot derive GitHub repo from repo_url={repo_url!r}")
    return f"{parts[0]}/{parts[1]}"


__all__ = ["GitHubPr", "_repo_slug"]
