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

from hyperloom.common import amd_smi
from hyperloom.common.visible_devices import (
    COUNTING_VISIBLE_DEVICE_VARS,
    GPU_MASK_ENV_NAMES,
    HIP_LEVEL_VARS,
    ROCR_LEVEL_VARS,
    VISIBLE_DEVICE_VARS,
    _UNCOUNTED_VISIBLE_DEVICE_VARS,
    detect_gpu_count,
    effective_mask_tokens,
    is_rocr_level,
    mask_tokens,
    parse_device_list,
)


@pytest.fixture(autouse=True)
def _clear_visible_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """None of these tests may inherit a real mask from the host running the suite."""
    for var in VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)


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


def test_effective_tokens_dedup_on_the_ordinal_not_the_spelling() -> None:
    """``0`` and ``00`` name one device; counting both inflates the width."""
    assert effective_mask_tokens("0,00") == ["0"]
    assert parse_device_list("0,00") == [0]


def test_parse_device_list_agrees_with_effective_tokens() -> None:
    """The ids are the numeric members of the effective set, never a wider one.

    This is the invariant that keeps ``gpu_pin["count"]`` and ``gpu_pin["ids"]``
    from disagreeing: counting the literal tokens inflates the count, and
    re-serializing the parsed ints deflates the id list.
    """
    for raw in ("4,5", "3,3,2", "-1,2", "GPU-a1b2c3,4", "", " 4, 4 ,5", "0,00"):
        ids = parse_device_list(raw)
        tokens = effective_mask_tokens(raw)
        assert len(ids) <= len(tokens)
        assert ids == [int(t) for t in tokens if t.lstrip("-").isdigit()]


# --------------------------------------------------------------------------- #
# detect_gpu_count: env masks first, then amd-smi. This path had never been
# exercised directly before.
# --------------------------------------------------------------------------- #


def test_detect_gpu_count_prefers_rocr_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(amd_smi, "gpu_count", lambda: (_ for _ in ()).throw(AssertionError("amd-smi must not run")))
    assert detect_gpu_count() == 2


def test_detect_gpu_count_rocr_wins_over_hip_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence inside ``detect_gpu_count`` must match the chain's documented order."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert detect_gpu_count() == 3


def test_detect_gpu_count_falls_through_an_unset_var_to_the_next_in_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ROCR_VISIBLE_DEVICES`` absent (not merely empty) must defer to ``HIP_VISIBLE_DEVICES``."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    assert detect_gpu_count() == 2


def test_detect_gpu_count_empty_mask_is_zero_gpus_not_a_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ROCR_VISIBLE_DEVICES=""`` means "pinned to nothing", distinct from "unset" -- it must not defer to HIP/CUDA
    or to amd-smi, both of which could otherwise report GPUs that this process cannot actually use."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setattr(amd_smi, "gpu_count", lambda: (_ for _ in ()).throw(AssertionError("amd-smi must not run")))
    assert detect_gpu_count() == 0


def test_detect_gpu_count_falls_back_to_amd_smi_when_no_mask_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no mask anywhere in the chain, the visible count comes from ``amd-smi list``."""
    monkeypatch.setattr(amd_smi, "gpu_count", lambda: 3)
    assert detect_gpu_count() == 3


def test_detect_gpu_count_is_zero_when_amd_smi_cannot_be_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``amd_smi.gpu_count`` already degrades to 0 on a missing/failing amd-smi; this is that path end to end."""
    monkeypatch.setattr(amd_smi, "gpu_count", lambda: 0)
    assert detect_gpu_count() == 0


def test_detect_gpu_count_env_branch_does_not_dedup_like_effective_mask_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents an existing inconsistency, not a recommendation: unlike :func:`effective_mask_tokens` (used
    elsewhere in this module), the env-mask branch of ``detect_gpu_count`` counts raw comma-split tokens, so a
    duplicated or negative ordinal inflates/deflates the count instead of being normalised first. This predates the
    rocm-smi -> amd-smi migration (the env-mask loop was untouched by it) and is out of this phase's scope to fix;
    recorded here only so the behavior cannot regress further unnoticed.
    """
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3,3,2")
    assert detect_gpu_count() == 3  # effective_mask_tokens("3,3,2") would say 2
