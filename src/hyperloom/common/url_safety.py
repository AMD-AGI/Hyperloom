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
    """Raise ``error`` unless ``url`` is http or https.

    Rejects ``file://`` and every other scheme, so a URL reaching a fetch from
    remote or model-supplied data cannot read the local filesystem.

    Args:
        url: The URL to validate.
        error: Exception class to raise; callers with a domain-specific error
            type pass their own.
        context: Names the URL's source in the message (e.g. ``"diff_url"``).

    Raises:
        ``error``: When the URL scheme is not ``http`` or ``https``.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        prefix = f"{context}: " if context else ""
        raise error(f"{prefix}unsupported URL scheme: {scheme!r} (only http/https allowed)")


__all__ = ["require_http_url"]
