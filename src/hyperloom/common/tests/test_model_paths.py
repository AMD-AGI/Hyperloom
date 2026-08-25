# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for model_paths identity and matching helpers."""

from __future__ import annotations

import pytest

from hyperloom.common.model_paths import model_identities_match, model_identity_candidates


@pytest.mark.parametrize(
    ("raw", "expected_full", "expected_bare"),
    [
        ("a/Llama-8B", {"a/llama-8b"}, {"llama-8b"}),
        ("b/Llama-8B", {"b/llama-8b"}, {"llama-8b"}),
        ("Llama-8B", {"llama-8b"}, {"llama-8b"}),
        (
            "/hf/models--a--Llama-8B/snapshots/deadbeef",
            {"a/llama-8b"},
            {"llama-8b"},
        ),
        (
            "/hf/models--b--Llama-8B/snapshots/deadbeef",
            {"b/llama-8b"},
            {"llama-8b"},
        ),
    ],
)
def test_model_identity_candidates(raw, expected_full, expected_bare):
    full, bare = model_identity_candidates(raw)
    assert full == expected_full
    assert bare == expected_bare


def test_snapshot_hash_excluded_from_candidates():
    """The commit hash basename of a hub cache path is not an identity."""
    full, bare = model_identity_candidates("/cache/models--org--repo/snapshots/abc123")
    assert "abc123" not in full
    assert "abc123" not in bare


@pytest.mark.parametrize(
    ("declared", "launched", "expected"),
    [
        ("a/Llama-8B", "b/Llama-8B", False),
        ("a/Llama-8B", "a/Llama-8B", True),
        ("Llama-8B", "a/Llama-8B", True),
        ("a/Llama-8B", "/hf/models--a--Llama-8B/snapshots/x", True),
        ("a/Llama-8B", "/hf/models--b--Llama-8B/snapshots/x", False),
        ("a/Llama-8B", "Llama-8B", True),
    ],
)
def test_model_identities_match(declared, launched, expected):
    assert model_identities_match(declared, launched) == expected


def test_model_identities_match_cross_org_is_false():
    """Two different orgs sharing a repo name must NOT match."""
    assert model_identities_match("vendor-a/SmallModel", "vendor-b/SmallModel") is False


def test_model_identities_match_same_org_is_true():
    assert model_identities_match("vendor-a/SmallModel", "vendor-a/SmallModel") is True
