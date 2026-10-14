# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib

import pytest

from hyperloom.measure.runner.inventory import OutputRejected, inventory


def test_every_file_is_listed_with_its_size_and_hash(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.log").write_bytes(b"hello")
    (tmp_path / "y.json").write_bytes(b"{}")

    files = inventory(tmp_path, max_total_bytes=100)

    assert [(f.path, f.size, f.sha256) for f in files] == [
        ("a/x.log", 5, hashlib.sha256(b"hello").hexdigest()),
        ("y.json", 2, hashlib.sha256(b"{}").hexdigest()),
    ]


def test_a_symlink_is_refused_as_unsafe_output(tmp_path):
    (tmp_path / "secret").write_text("s")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "link").symlink_to(tmp_path / "secret")

    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path / "out", max_total_bytes=100)

    assert caught.value.error_class == "unsafe_output"


def test_output_over_the_limit_is_refused_as_too_large(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 60)
    (tmp_path / "b").write_bytes(b"x" * 60)

    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path, max_total_bytes=100)

    assert caught.value.error_class == "output_too_large"


def test_a_name_the_bundle_owns_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")

    with pytest.raises(OutputRejected, match="the bundle owns"):
        inventory(tmp_path, max_total_bytes=100)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path / "nope", max_total_bytes=100)

    assert caught.value.error_class == "unsafe_output"
