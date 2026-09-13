# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""URL scheme validation for fetch call-sites. Standard library only."""

from __future__ import annotations

import urllib.parse


def require_http_url(
    url: str,
    *,
    error: type[Exception] = ValueError,
    context: str = "",
) -> None:
    """Raise ``error`` unless ``url`` is http or https."""
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        prefix = f"{context}: " if context else ""
        raise error(f"{prefix}unsupported URL scheme: {scheme!r} (only http/https allowed)")


__all__ = ["require_http_url"]
