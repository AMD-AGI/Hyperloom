# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Smoke tests for the kernel-agent payload-aliases compat shim (duplicated copy of inference_optimizer.compat.payload_aliases; pins behaviour against drift)."""

from __future__ import annotations

import warnings

import pytest

from _payload_aliases import (  # type: ignore[import-not-found]
    CANONICAL_KEY,
    LEGACY_KEY,
    read_extra_server_args,
)


def test_shim_constants_match_canonical_names():
    assert CANONICAL_KEY == "extra_server_args"
    assert LEGACY_KEY == "extra_sglang_args"


def test_shim_canonical_key_returns_value_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({CANONICAL_KEY: "--x"})
    assert out == "--x"
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_shim_legacy_key_emits_warning():
    with pytest.warns(DeprecationWarning):
        out = read_extra_server_args({LEGACY_KEY: "--legacy"})
    assert out == "--legacy"


def test_shim_default_returned_when_empty():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert read_extra_server_args({}) == ""
        assert read_extra_server_args({}, default="z") == "z"
    assert not caught


def test_shim_canonical_wins_over_legacy_when_both_present():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({CANONICAL_KEY: "new", LEGACY_KEY: "old"})
    assert out == "new"
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
