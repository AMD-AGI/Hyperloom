# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for hyperloom.inference_optimizer.tracelens_md shared helpers."""

from __future__ import annotations

from hyperloom.inference_optimizer.tracelens_md import strip_base64_data_urls


def test_strip_base64_data_urls_replaces_payload():
    md = "![chart](data:image/png;base64,AAAABBBB)\n\n## Summary\n"
    out = strip_base64_data_urls(md)
    assert "AAAABBBB" not in out
    assert "<<stripped: base64 image — chart>>" in out
    assert "## Summary" in out


def test_strip_base64_data_urls_passes_through_plain_text():
    md = "# Analysis\n[link](https://example.com)\n"
    assert strip_base64_data_urls(md) == md


def test_strip_base64_data_urls_handles_empty():
    assert strip_base64_data_urls("") == ""
    assert strip_base64_data_urls(None) == ""
