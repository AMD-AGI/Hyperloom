# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``inference_optimizer.compat.payload_aliases``.

Covers the read-only deprecation alias from ``extra_sglang_args`` ->
``extra_server_args``. Every reader site funnels through these
helpers, so the contract here is the source of truth for the rest of
the migration test surface.

All tests are pure-Python and hermetic — no fixtures touch the disk
or the network.
"""

from __future__ import annotations

import warnings


from inference_optimizer.compat import payload_aliases as pa
from inference_optimizer.compat.payload_aliases import (
    CANONICAL_KEY,
    LEGACY_KEY,
    migrate_legacy_key_in_place,
    read_extra_server_args,
    read_extra_server_args_from_envs,
)


# ---------------------------------------------------------------------------
# Constant integrity
# ---------------------------------------------------------------------------
def test_canonical_and_legacy_constants_are_what_we_say_they_are():
    """Constants documented in the module docstring must match the
    runtime values. Catches a future rename of the canonical name
    that forgets to update either the constants or the documentation."""
    assert CANONICAL_KEY == "extra_server_args"
    assert LEGACY_KEY == "extra_sglang_args"


# ---------------------------------------------------------------------------
# Canonical-key path — no warning, value returned verbatim
# ---------------------------------------------------------------------------
def test_canonical_key_returned():
    """Canonical key present → value returned, NO warning emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({CANONICAL_KEY: "--foo --bar"})
    assert out == "--foo --bar"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_canonical_empty_string_returned_without_warning():
    """Empty string is a *value*, not a missing key. The helper must
    distinguish in-vs-out so callers that intentionally set the key
    to empty don't get the legacy-key fallback."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({CANONICAL_KEY: ""})
    assert out == ""
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_canonical_wins_when_both_present():
    """Both keys present → canonical wins, no warning. Migration is
    considered done as soon as the canonical key arrives."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args(
            {CANONICAL_KEY: "new", LEGACY_KEY: "old"},
        )
    assert out == "new"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# Legacy-key path — warning + value
# ---------------------------------------------------------------------------
def test_legacy_key_emits_deprecation_and_returns_value():
    """Legacy key only → value returned, single DeprecationWarning
    naming both keys."""
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
    """Legacy key set to empty string still triggers the alias path
    (the value is what got carried; the warning is what flags the
    writer for migration)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read_extra_server_args({LEGACY_KEY: ""})
    assert out == ""
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1


# ---------------------------------------------------------------------------
# Default path — neither key present
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------
def test_none_value_coerced_to_empty_string():
    """``payload[CANONICAL_KEY] = None`` → returns ``""`` so callers
    that immediately ``.strip()`` get the same shape they did pre-
    rename when they wrote ``str(payload.get(...) or "")``."""
    assert read_extra_server_args({CANONICAL_KEY: None}) == ""
    # And the same on the legacy side.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert read_extra_server_args({LEGACY_KEY: None}) == ""


def test_non_string_value_coerced_via_str():
    """A non-string value (e.g. accidentally set to a list / int) goes
    through ``str()`` so the helper never raises a TypeError."""
    assert read_extra_server_args({CANONICAL_KEY: 42}) == "42"


# ---------------------------------------------------------------------------
# Stacklevel
# ---------------------------------------------------------------------------
def test_deprecation_stacklevel_points_at_caller():
    """``stacklevel=3`` means the reported filename of the warning is
    the caller's caller. Wrap the helper in a one-frame function and
    assert that THAT function's filename is what gets reported."""
    def _wrapper(payload):
        return read_extra_server_args(payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _wrapper({LEGACY_KEY: "x"})

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    # The reported filename should be this test file (the caller of
    # the wrapper), not payload_aliases.py.
    assert dep_warnings[0].filename.endswith("test_payload_aliases.py"), (
        f"expected stacklevel=3 to report the test caller; "
        f"got filename={dep_warnings[0].filename!r}"
    )


# ---------------------------------------------------------------------------
# Envs variant — same contract on the materializer-side dict
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# migrate_legacy_key_in_place
# ---------------------------------------------------------------------------
def test_migrate_legacy_only_moves_value_and_returns_true():
    """Legacy key only → copied to canonical, legacy deleted, True."""
    p = {LEGACY_KEY: "--foo"}
    changed = migrate_legacy_key_in_place(p)
    assert changed is True
    assert p == {CANONICAL_KEY: "--foo"}


def test_migrate_no_op_when_canonical_already_present():
    """Canonical present → no transform, returns False (caller decides
    whether to also drop the legacy key)."""
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
    """Persistence-side migration is silent — the read-side warning
    is the audit channel for in-flight payloads."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrate_legacy_key_in_place({LEGACY_KEY: "x"})
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------
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
    """End-to-end import smoke: ``from
    inference_optimizer.compat.payload_aliases import
    read_extra_server_args`` resolves and is callable."""
    from inference_optimizer.compat import payload_aliases as imported
    assert callable(imported.read_extra_server_args)
    assert callable(imported.read_extra_server_args_from_envs)
    assert callable(imported.migrate_legacy_key_in_place)
