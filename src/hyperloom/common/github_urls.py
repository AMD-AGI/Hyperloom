# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GitHub URL helpers shared across packages."""

from __future__ import annotations

from urllib.parse import urlparse

_GITHUB_HOST = "github.com"


def repo_slug(repo_url: str) -> str:
    """Parse ``owner/name`` from a github.com HTTPS or SSH URL; raise ``ValueError`` for anything else."""
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


__all__ = ["repo_slug"]
