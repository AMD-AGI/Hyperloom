# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the read-only ``extra_sglang_args`` -> ``extra_server_args`` deprecation alias in ``compat.payload_aliases``."""

from __future__ import annotations

import warnings

import pytest

from inference_optimizer.compat import payload_aliases as pa
from inference_optimizer.compat.payload_aliases import (
    CANONICAL_KEY,
    LEGACY_KEY,
    migrate_legacy_key_in_place,
    read_extra_server_args,
    read_extra_server_args_from_envs,
)


# Constant integrity
def test_canonical_and_legacy_constants_are_what_we_say_they_are():
    """Documented constants must match the runtime values."""
    assert CANONICAL_KEY == "extra_server_args"
    assert LEGACY_KEY == "extra_sglang_args"


# Canonical-key path — no warning, value returned verbatim
def test_canonical_key_returned():
    """Canonical key present → value returned, NO warning emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({CANONICAL_KEY: "--foo --bar"})
    assert out == "--foo --bar"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_canonical_empty_string_returned_without_warning():
    """Empty string is a *value*, not a missing key — no legacy-key fallback."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({CANONICAL_KEY: ""})
    assert out == ""
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_canonical_wins_when_both_present():
    """Both keys present → canonical wins, no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args(
            {CANONICAL_KEY: "new", LEGACY_KEY: "old"},
        )
    assert out == "new"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


# Legacy-key path — warning + value
def test_legacy_key_emits_deprecation_and_returns_value():
    """Legacy key only → value returned, single DeprecationWarning naming both keys."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({LEGACY_KEY: "--legacy-flag"})
    assert out == "--legacy-flag"
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    msg = str(dep_warnings[0].message)
    assert LEGACY_KEY in msg
    assert CANONICAL_KEY in msg


def test_legacy_empty_string_returned_with_warning():
    """Legacy key set to empty string still triggers the alias path (value carried + warning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({LEGACY_KEY: ""})
    assert out == ""
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1


# Default path — neither key present
def test_default_returned_when_neither_present():
    """Empty payload → default returned, no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({})
    assert out == ""
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_default_override():
    """Explicit ``default=`` is honoured when neither key is present."""
    assert read_extra_server_args({}, default="--sentinel") == "--sentinel"


# Value coercion
def test_none_value_coerced_to_empty_string():
    """``payload[CANONICAL_KEY] = None`` → returns ``""``."""
    assert read_extra_server_args({CANONICAL_KEY: None}) == ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert read_extra_server_args({LEGACY_KEY: None}) == ""


def test_non_string_value_coerced_via_str():
    """A non-string value goes through ``str()`` so the helper never raises TypeError."""
    assert read_extra_server_args({CANONICAL_KEY: 42}) == "42"


def test_list_value_space_joined_not_repr():
    """A list/tuple of flags is space-joined into shell tokens, NOT rendered as
    a Python repr (which Magpie would splice verbatim into ``vllm serve`` and
    the server would reject as ``unrecognized arguments``)."""
    out = read_extra_server_args(
        {CANONICAL_KEY: ["--max-num-batched-tokens", "32768"]}
    )
    assert out == "--max-num-batched-tokens 32768"
    assert "[" not in out and "'" not in out
    assert read_extra_server_args(
        {CANONICAL_KEY: ("--distributed-executor-backend", "mp")}
    ) == "--distributed-executor-backend mp"


# Stacklevel
def test_deprecation_stacklevel_points_at_caller():
    """``stacklevel=3`` makes the warning report the caller's caller filename."""
    def _wrapper(payload):
        return read_extra_server_args(payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _wrapper({LEGACY_KEY: "x"})

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    # Reported filename should be this test file, not payload_aliases.py.
    assert dep_warnings[0].filename.endswith("test_payload_aliases.py"), (
        f"expected stacklevel=3 to report the test caller; "
        f"got filename={dep_warnings[0].filename!r}"
    )


# Envs variant — same contract on the materializer-side dict
def test_envs_variant_canonical_returned_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args_from_envs({CANONICAL_KEY: "v"})
    assert out == "v"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_envs_variant_legacy_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args_from_envs({LEGACY_KEY: "v"})
    assert out == "v"
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1


def test_envs_variant_default():
    assert read_extra_server_args_from_envs({}, default="z") == "z"


# migrate_legacy_key_in_place
def test_migrate_legacy_only_moves_value_and_returns_true():
    """Legacy key only → copied to canonical, legacy deleted, True."""
    p = {LEGACY_KEY: "--foo"}
    changed = migrate_legacy_key_in_place(p)
    assert changed is True
    assert p == {CANONICAL_KEY: "--foo"}


def test_migrate_no_op_when_canonical_already_present():
    """Canonical present → no transform, returns False."""
    p = {CANONICAL_KEY: "new", LEGACY_KEY: "old"}
    changed = migrate_legacy_key_in_place(p)
    assert changed is False
    assert p == {CANONICAL_KEY: "new", LEGACY_KEY: "old"}


def test_migrate_no_op_when_neither_present():
    p: dict = {}
    changed = migrate_legacy_key_in_place(p)
    assert changed is False
    assert p == {}


def test_migrate_emits_no_warning():
    """Persistence-side migration is silent — the read-side warning is the audit channel."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrate_legacy_key_in_place({LEGACY_KEY: "x"})
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


# Module surface
def test_public_api_exports_match_all():
    """__all__ must enumerate every public surface of the module."""
    expected = {
        "CANONICAL_KEY",
        "LEGACY_KEY",
        "read_extra_server_args",
        "read_extra_server_args_from_envs",
        "migrate_legacy_key_in_place",
    }
    assert set(pa.__all__) == expected


def test_compat_package_importable():
    """End-to-end import smoke: the compat helpers resolve and are callable."""
    from inference_optimizer.compat import payload_aliases as imported
    assert callable(imported.read_extra_server_args)
    assert callable(imported.read_extra_server_args_from_envs)
    assert callable(imported.migrate_legacy_key_in_place)
