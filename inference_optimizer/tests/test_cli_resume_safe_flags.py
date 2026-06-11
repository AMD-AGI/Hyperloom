# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the resume-safe CLI flag helpers in ``cli.py``."""

from __future__ import annotations

import argparse

from inference_optimizer.cli import (
    _resume_safe_flag,
    _resume_safe_numeric,
)


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# _resume_safe_flag
def test_resume_safe_flag_explicit_disable_wins_over_manifest():
    """Explicit current-launch ``--no-foo`` wins over an enabled manifest."""
    args = _ns(no_foo=True)
    manifest = {"foo_enabled": True}
    result = _resume_safe_flag(
        args, "no_foo", manifest, "foo_enabled", default=True, invert=True,
    )
    # invert=True means args.no_foo=True → disabled.
    assert result is False


def test_resume_safe_flag_falls_back_to_manifest_when_arg_default():
    """When ``--no-foo`` isn't re-passed on resume, honor the persisted manifest value over the argparse default."""
    args = _ns(no_foo=False)  # argparse default for store_true
    manifest = {"foo_enabled": False}
    result = _resume_safe_flag(
        args, "no_foo", manifest, "foo_enabled", default=True, invert=True,
    )
    assert result is False


def test_resume_safe_flag_uses_default_when_manifest_missing_key():
    """No manifest entry → fall through to the supplied default."""
    args = _ns(no_foo=False)
    manifest = {}
    result = _resume_safe_flag(
        args, "no_foo", manifest, "foo_enabled", default=True, invert=True,
    )
    assert result is True


def test_resume_safe_flag_handles_none_manifest():
    """``manifest=None`` → fall through to default (very-early-boot path)."""
    args = _ns(no_foo=False)
    result = _resume_safe_flag(
        args, "no_foo", None, "foo_enabled", default=True, invert=True,
    )
    assert result is True


# _resume_safe_numeric
def test_resume_safe_numeric_explicit_override_wins():
    """Explicit ``--threshold 0.5`` wins over a manifest value of 0.7."""
    args = _ns(threshold=0.5)
    manifest = {"threshold": 0.7}
    result = _resume_safe_numeric(
        args, "threshold", manifest, "threshold", default=0.8,
    )
    assert result == 0.5


def test_resume_safe_numeric_falls_back_to_manifest():
    """With no explicit value on resume, use the manifest's 0.7 over the argparse default."""
    args = _ns(threshold=0.8)  # argparse default
    manifest = {"threshold": 0.7}
    result = _resume_safe_numeric(
        args, "threshold", manifest, "threshold", default=0.8,
    )
    assert result == 0.7


def test_resume_safe_numeric_falls_back_to_default():
    """No manifest entry → use the supplied default."""
    args = _ns(threshold=0.8)
    result = _resume_safe_numeric(
        args, "threshold", {}, "threshold", default=0.8,
    )
    assert result == 0.8


def test_resume_safe_numeric_handles_malformed_manifest_value():
    """Manifest carrying a non-numeric value (corrupt JSON) → default rather than crashing the cli boot."""
    args = _ns(threshold=0.8)
    manifest = {"threshold": "garbage"}
    result = _resume_safe_numeric(
        args, "threshold", manifest, "threshold", default=0.8,
    )
    assert result == 0.8
