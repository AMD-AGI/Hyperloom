"""Shared types and helpers across PR source backends.

Defines `GitHubPr` (a lightweight result record) and `_repo_slug` (a
repo_url -> "owner/name" parser) so that Primus Cortex and GitHub backends
can produce uniform candidate records without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GitHubPr:
    """Lightweight PR record returned by any PR source backend.

    All source backends (primus_cortex, github) map their native payload
    into this shape so the dispatcher can union them uniformly.
    """

    number: int
    title: str
    html_url: str

    @property
    def ref(self) -> str:
        """Stable candidate ref used downstream (`Candidate.ref`)."""
        return f"PR:{self.number}"


def _repo_slug(repo_url: str) -> str:
    """Parse ``owner/name`` from a GitHub-style git URL.

    Accepts the common forms (https + .git, https without .git, ssh).
    Raises ValueError on a non-GitHub or malformed URL so callers can
    surface a clean error instead of guessing.
    """
    raw = repo_url.strip()
    if raw.endswith(".git"):
        raw = raw[:-4]
    if raw.startswith("git@github.com:"):
        raw = raw.split(":", 1)[1]
    elif "github.com/" in raw:
        raw = raw.split("github.com/", 1)[1]
    parts = [p for p in raw.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"cannot derive GitHub repo from repo_url={repo_url!r}")
    return f"{parts[0]}/{parts[1]}"


__all__ = ["GitHubPr", "_repo_slug"]
