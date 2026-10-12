"""Credential detection enforced on every collected Experience.

A mapping cannot switch these checks off: an Experience that carries a
credential-shaped value is skipped rather than published. Free-text fields get
only the high-confidence token formats, because prose such as "per token: ..."
would otherwise trip the assignment rules written for configuration text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from hyperloom_kb.schema import JsonValue

_TOKEN_FORMATS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    ),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")),
    ("api key", re.compile(r"\b(?:ak|sk|pk)-(?:lf-)?[A-Za-z0-9_-]{6,}")),
    ("github token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{3,}|github_pat_[A-Za-z0-9_]{10,})")),
    ("aws access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    (
        "presigned url",
        re.compile(r"(?i)https?://\S+[?&](?:X-Amz-Signature|Signature|sig)="),
    ),
)

_CONFIGURATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "authorization header",
        re.compile(r"(?i)\b(?:authorization|ocp-apim-subscription-key)\s*[:=]\s*[^\s,;'\"\\]+"),
    ),
    (
        "custom headers",
        re.compile(r"(?i)\b[A-Z0-9_]*CUSTOM_HEADERS\s*[=:]\s*\S+"),
    ),
    (
        "credential assignment",
        re.compile(
            r"(?i)(?<![A-Z0-9_./-])"
            r"(?:[A-Z0-9_]*(?:API_?KEY|SECRET|PASSWORD|CREDENTIAL|HEADERS)[A-Z0-9_]*|"
            r"AUTH(?:_[A-Z0-9_]+)?|"
            r"[A-Z0-9_]+_AUTH(?:ORIZATION)?(?:_[A-Z0-9_]+)?|"
            r"(?:[A-Z0-9_]+_)?TOKEN(?:_\d+)?)"
            r"\s*[=:]\s*(?:\\?[\"'])?(?:\\(?![\"'])|[^\s,;'\"\\])+"
        ),
    ),
    (
        "credential flag",
        re.compile(
            r"(?i)(?:^|[\s\"'=])--?[^\s=\"']*"
            r"(?:api[-_]?key|token|secret|password|credential)(?:=|\s)"
        ),
    ),
)

_EMBEDDED_KEY = re.compile(r"\"([A-Za-z_][A-Za-z0-9_.-]*)\"\s*:")
_SECRET_NAME_FRAGMENTS = ("APIKEY", "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_SECRET_NAME_EXEMPTIONS = ("TOKENIZER",)

FREE_TEXT_FIELDS = frozenset(
    {
        "reasoning",
        "reflection",
        "change.summary",
        "alternatives",
    }
)


def is_secret_shaped_name(name: str) -> bool:
    """Return whether a key or variable name looks like a credential rather than a knob."""

    upper = name.strip().upper()
    for exemption in _SECRET_NAME_EXEMPTIONS:
        upper = upper.replace(exemption, "")
    return any(fragment in upper for fragment in _SECRET_NAME_FRAGMENTS)


def _text_finding(text: str, *, free_text: bool) -> str | None:
    for kind, pattern in _TOKEN_FORMATS:
        if pattern.search(text):
            return kind
    if free_text:
        return None
    for kind, pattern in _CONFIGURATION_RULES:
        if pattern.search(text):
            return kind
    for match in _EMBEDDED_KEY.finditer(text):
        if is_secret_shaped_name(match.group(1)):
            return "credential-shaped key"
    return None


def _is_free_text(path: str) -> bool:
    return any(path == name or path.startswith(f"{name}[") for name in FREE_TEXT_FIELDS)


def find_sensitive(value: JsonValue, path: str = "") -> str | None:
    """Return ``"<field path>: <finding>"`` for the first credential-shaped content."""

    if isinstance(value, str):
        finding = _text_finding(value, free_text=_is_free_text(path))
        return f"{path or '<root>'}: {finding}" if finding else None
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if is_secret_shaped_name(str(key)):
                return f"{child}: credential-shaped key"
            finding = find_sensitive(item, child)
            if finding:
                return finding
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            finding = find_sensitive(item, f"{path}[{index}]")
            if finding:
                return finding
    return None


__all__ = ["FREE_TEXT_FIELDS", "find_sensitive", "is_secret_shaped_name"]
