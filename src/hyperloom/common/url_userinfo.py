# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strip the ``user[:password]@`` part from URLs (stdlib only)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

VCS_SCHEME_PREFIXES: tuple[str, ...] = ("git+", "hg+", "svn+")

CREDENTIALED_URL_RE = re.compile(
    r"(?P<scheme>(?:(?:git|hg|svn)\+)?[A-Za-z][A-Za-z0-9+.-]*://)"
    r"(?P<userinfo>[^\s/?#\"']+)@(?P<location>[^\s\"']+)"
)


def strip_url_userinfo(url: str) -> str:
    """Return ``url`` with any userinfo component removed."""
    text = str(url or "")
    prefix = ""
    for candidate in VCS_SCHEME_PREFIXES:
        if text.startswith(candidate):
            prefix, text = candidate, text[len(candidate) :]
            break
    if "://" not in text:
        return str(url or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        parts = None
    if parts is not None and "@" in parts.netloc:
        host = parts.netloc.rsplit("@", 1)[1]
        return prefix + urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return prefix + CREDENTIALED_URL_RE.sub(r"\g<scheme>\g<location>", text)
