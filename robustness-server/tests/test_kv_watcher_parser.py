"""Tests for the BRAIN_REGISTRY KV value parser."""

from __future__ import annotations

from robustness_server.services.kv_watcher import _parse_value


def test_namespace_and_pod_split() -> None:
    assert _parse_value(b"claw/brain-1") == ("claw", "brain-1")


def test_no_namespace_keeps_pod_name() -> None:
    assert _parse_value(b"brain-only") == ("", "brain-only")


def test_empty_or_none_returns_empties() -> None:
    assert _parse_value(None) == ("", "")
    assert _parse_value(b"") == ("", "")
    assert _parse_value(b"   ") == ("", "")


def test_non_utf8_bytes_safe_default() -> None:
    assert _parse_value(b"\xff\xfe") == ("", "")
