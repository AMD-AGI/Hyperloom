"""Credential detection enforced on every collected Experience and the files it names.

A mapping cannot switch these checks off: an Experience that carries a
credential-shaped value is skipped rather than published. Free-text fields get
only the high-confidence token formats, because prose such as "per token: ..."
would otherwise trip the assignment rules written for configuration text.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path

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

#: The fields that hold prose by definition: the reasoning and the alternatives weighed, how the outcome reads, and
#: labelled notes. Preconditions are not prose: they may restate the configuration measured against.
FREE_TEXT_FIELDS = frozenset({"rationale.reasoning", "rationale.alternatives", "reflection", "notes"})


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


def _is_free_text(path: str, free_text: Collection[str]) -> bool:
    return any(path == name or path.startswith((f"{name}[", f"{name}.")) for name in free_text)


def find_sensitive(value: JsonValue, path: str = "", *, free_text: Collection[str] = FREE_TEXT_FIELDS) -> str | None:
    """Return ``"<field path>: <finding>"`` for the first credential-shaped content.

    A string at or below a path in ``free_text`` is screened as prose.
    """

    if isinstance(value, str):
        finding = _text_finding(value, free_text=_is_free_text(path, free_text))
        return f"{path or '<root>'}: {finding}" if finding else None
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if is_secret_shaped_name(str(key)):
                return f"{child}: credential-shaped key"
            finding = find_sensitive(item, child, free_text=free_text)
            if finding:
                return finding
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            finding = find_sensitive(item, f"{path}[{index}]", free_text=free_text)
            if finding:
                return finding
    return None


def find_sensitive_in_file(path: Path, *, free_text: bool) -> str | None:
    """The first credential-shaped content of the text file at ``path``, line by line; a binary file has none."""

    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                finding = _text_finding(line, free_text=free_text)
                if finding:
                    return f"line {number}: {finding}"
    except UnicodeDecodeError:
        return None
    return None


__all__ = ["FREE_TEXT_FIELDS", "find_sensitive", "find_sensitive_in_file", "is_secret_shaped_name"]
