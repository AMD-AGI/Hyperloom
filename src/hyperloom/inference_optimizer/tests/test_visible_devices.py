# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The shared visible-devices module (PR #1321 review).

Five layers used to carry their own copy of the mask tuple and the mask parser,
and their empty-mask semantics had already drifted apart.
``hyperloom.common.visible_devices`` is now the single definition, and it holds
two deliberately different tuples. These tests exist so the difference stays
deliberate: the counting subset is derived from the precedence chain, so a var
added to one cannot silently miss the other.
"""

from __future__ import annotations

import pytest

from hyperloom.common.visible_devices import (
    COUNTING_VISIBLE_DEVICE_VARS,
    GPU_MASK_ENV_NAMES,
    HIP_LEVEL_VARS,
    ROCR_LEVEL_VARS,
    VISIBLE_DEVICE_VARS,
    _UNCOUNTED_VISIBLE_DEVICE_VARS,
    effective_mask_tokens,
    is_rocr_level,
    mask_tokens,
    parse_device_list,
)


# --------------------------------------------------------------------------- #
# The two tuples cannot drift apart
# --------------------------------------------------------------------------- #


def test_every_chain_member_is_classified_as_counted_or_not() -> None:
    """A var added to the chain must be a deliberate decision, not an omission.

    ``COUNTING_VISIBLE_DEVICE_VARS`` is derived, so a new var is counted by
    default; leaving it out requires naming it in the exclusion set. This
    asserts the two halves partition the chain exactly.
    """
    assert set(COUNTING_VISIBLE_DEVICE_VARS) | _UNCOUNTED_VISIBLE_DEVICE_VARS == set(VISIBLE_DEVICE_VARS)
    assert set(COUNTING_VISIBLE_DEVICE_VARS) & _UNCOUNTED_VISIBLE_DEVICE_VARS == set()


def test_the_exclusion_set_names_nothing_outside_the_chain() -> None:
    """A typo'd or removed exclusion would silently widen the counting set."""
    assert _UNCOUNTED_VISIBLE_DEVICE_VARS <= set(VISIBLE_DEVICE_VARS)


def test_the_counting_subset_is_exactly_what_the_counting_layers_always_read() -> None:
    """Deriving the tuple must not have changed GPU accounting."""
    assert COUNTING_VISIBLE_DEVICE_VARS == (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    )


def test_the_counting_subset_keeps_the_chain_order() -> None:
    """Precedence is the whole point of an ordered tuple."""
    kept = [var for var in VISIBLE_DEVICE_VARS if var in COUNTING_VISIBLE_DEVICE_VARS]
    assert list(COUNTING_VISIBLE_DEVICE_VARS) == kept


def test_the_chain_is_rocr_level_before_hip_level() -> None:
    """A ROCr mask slices the set a HIP mask then indexes into, so it wins."""
    assert VISIBLE_DEVICE_VARS == ROCR_LEVEL_VARS + HIP_LEVEL_VARS
    assert all(is_rocr_level(var) for var in ROCR_LEVEL_VARS)
    assert not any(is_rocr_level(var) for var in HIP_LEVEL_VARS)


def test_the_scrub_set_covers_the_whole_chain() -> None:
    """Env scrubbing must not leave a mask behind that pin resolution honours."""
    assert GPU_MASK_ENV_NAMES == frozenset(VISIBLE_DEVICE_VARS)


# --------------------------------------------------------------------------- #
# The parsers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "tokens"),
    [
        ("4,5", ["4", "5"]),
        ("4;5", ["4", "5"]),
        (" 4 , 5 ", ["4", "5"]),
        ("4,,5", ["4", "5"]),
        ([" 4", 5], ["4", "5"]),
        ("", []),
        (None, []),
    ],
)
def test_mask_tokens_is_the_literal_split(raw: object, tokens: list[str]) -> None:
    assert mask_tokens(raw) == tokens


def test_mask_tokens_keeps_duplicates_but_effective_tokens_drops_them() -> None:
    """The literal split and the effective device set are different questions."""
    assert mask_tokens("3,3,2") == ["3", "3", "2"]
    assert effective_mask_tokens("3,3,2") == ["3", "2"]


def test_effective_tokens_drops_negative_ordinals_and_keeps_uuids() -> None:
    """A negative ordinal is not a device; a UUID is."""
    assert effective_mask_tokens("-1,2") == ["2"]
    assert effective_mask_tokens("GPU-a1b2c3,GPU-d4e5f6") == ["GPU-a1b2c3", "GPU-d4e5f6"]


def test_parse_device_list_agrees_with_effective_tokens() -> None:
    """The ids are the numeric members of the effective set, never a wider one.

    This is the invariant that keeps ``gpu_pin["count"]`` and ``gpu_pin["ids"]``
    from disagreeing: counting the literal tokens inflates the count, and
    re-serializing the parsed ints deflates the id list.
    """
    for raw in ("4,5", "3,3,2", "-1,2", "GPU-a1b2c3,4", "", " 4, 4 ,5"):
        ids = parse_device_list(raw)
        tokens = effective_mask_tokens(raw)
        assert len(ids) <= len(tokens)
        assert [str(i) for i in ids] == [t for t in tokens if t.lstrip("-").isdigit()]
