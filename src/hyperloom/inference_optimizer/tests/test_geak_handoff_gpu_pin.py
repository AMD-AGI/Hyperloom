# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-pin forwarding in the GEAK handoff (issue #1312).

GEAK launches full servers out-of-process and writes a visible-devices mask for
each one. When the handoff carries no pin it falls back to ``0..tp-1``, so every
server lands on physical GPU 0 no matter where the run was pinned — on a shared
host that collides with a foreign tenant and the resulting OOM reads like a real
regression.

These tests guard both halves of the contract:

* ``gpu_ids`` stays in the coordinate system the consumer applies it in (HIP
  indexes into the ROCr-visible set), so existing pins keep working;
* ``gpu_pin`` carries the ABSOLUTE mask plus the variable it came from, so a
  consumer that writes ``ROCR_VISIBLE_DEVICES`` re-applies the pin instead of
  resetting the child to card 0.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.loop.coordinator_helpers import (
    _coerce_tp,
    _is_autofilled_rocr,
    _parse_device_list,
    _resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
    _resolve_handoff_tp,
)
from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS


def _autofilled(tp: int) -> dict[str, object]:
    """The ``benchmark.envs`` every materialized recipe carries.

    ``materialize_config_with_envs`` writes ``ROCR_VISIBLE_DEVICES=0..tp-1``
    unconditionally when the mask is absent or narrower than TP, so this shape
    — not an empty mapping — is what the resolver sees in production.
    """
    return {"TP": tp, "ROCR_VISIBLE_DEVICES": ",".join(str(i) for i in range(tp))}


_MASK_VARS = VISIBLE_DEVICE_VARS


@pytest.fixture(autouse=True)
def _clear_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every case against a known-unpinned environment."""
    for var in _MASK_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# _parse_device_list
# --------------------------------------------------------------------------- #


def test_parse_device_list_forms() -> None:
    assert _parse_device_list("4,5,6,7") == [4, 5, 6, 7]
    assert _parse_device_list(" 6 ") == [6]
    assert _parse_device_list("0;1") == [0, 1]
    assert _parse_device_list("3,3,2") == [3, 2]


def test_parse_device_list_tolerates_junk_and_empty() -> None:
    assert _parse_device_list("") == []
    assert _parse_device_list(None) == []
    assert _parse_device_list("a,,-1,2") == [2]


# --------------------------------------------------------------------------- #
# _resolve_gpu_pin
# --------------------------------------------------------------------------- #


def test_pin_unset_everywhere_is_empty() -> None:
    """No mask anywhere means "whole machine visible", NOT "pinned to 0"."""
    assert _resolve_gpu_pin(recipe_envs={}, environ={}) == {}


def test_pin_from_process_rocr() -> None:
    """The case issue #1312 hit: ROCm's canonical mask, previously ignored."""
    out = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "7"})
    assert out == {
        "var": "ROCR_VISIBLE_DEVICES",
        "value": "7",
        "ids": [7],
        "count": 1,
        "source": "process_env",
    }


def test_pin_prefers_rocr_over_hip_and_cuda() -> None:
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "1",
        "ROCR_VISIBLE_DEVICES": "4,5",
    }
    out = _resolve_gpu_pin(recipe_envs={}, environ=env)
    assert out["var"] == "ROCR_VISIBLE_DEVICES"
    assert out["ids"] == [4, 5]


def test_pin_prefers_process_env_over_recipe() -> None:
    """The process mask is the one the GEAK child actually inherits."""
    out = _resolve_gpu_pin(
        recipe_envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "6"},
        environ={"ROCR_VISIBLE_DEVICES": "3"},
    )
    assert out["source"] == "process_env"
    assert out["ids"] == [3]


def test_pin_uses_recipe_when_the_process_is_unmasked() -> None:
    """A hand-authored recipe mask is still a pin when nothing else says otherwise."""
    out = _resolve_gpu_pin(recipe_envs={"TP": 2, "ROCR_VISIBLE_DEVICES": "6,7"}, environ={})
    assert out["source"] == "baseline_recipe"
    assert out["ids"] == [6, 7]


def test_a_blank_value_does_not_shadow_a_real_pin_further_down_the_chain() -> None:
    """A blank ROCR must not hide a HIP pin — but it is still reported, see below.

    ``gpu_pool._visible_device_mask`` and ``gate.detect_gpu_count`` read
    ``VAR=""`` as "zero devices visible"; this resolver used to skip it
    entirely, so with a blank ROCR and a stale ``HIP=2,3`` those layers saw
    zero GPUs while the handoff advertised two. The blank is now recorded (see
    :func:`test_an_empty_mask_reports_zero_devices_rather_than_unpinned`) but
    only as the fallback, so a real pin still wins.
    """
    out = _resolve_gpu_pin(
        recipe_envs={"ROCR_VISIBLE_DEVICES": "  "},
        environ={"HIP_VISIBLE_DEVICES": "2,3"},
    )
    assert out["var"] == "HIP_VISIBLE_DEVICES"
    assert out["ids"] == [2, 3]


# --------------------------------------------------------------------------- #
# The materializer's autofilled ROCR mask (PR #1321 review)
# --------------------------------------------------------------------------- #


def test_autofilled_recipe_rocr_does_not_override_a_hip_pin() -> None:
    """Regression: recipe-first made every HIP-pinned run report cards 0..tp-1.

    ``materialize_config_with_envs`` synthesizes ``ROCR_VISIBLE_DEVICES=0,1``
    into the recipe for a ``TP=2`` run that has no ROCR anywhere. Honouring
    that as a pin overrode the real ``HIP_VISIBLE_DEVICES=4,5`` and told a
    ROCR-writing consumer to hard-pin physical cards 0 and 1 — recreating the
    card-0 collision this whole change exists to remove.
    """
    out = _resolve_gpu_pin(
        recipe_envs=_autofilled(2),
        environ={"HIP_VISIBLE_DEVICES": "4,5"},
    )
    assert out["var"] == "HIP_VISIBLE_DEVICES"
    assert out["ids"] == [4, 5]
    assert _resolve_handoff_gpu_ids(gpu_pin=out, tp=2) == "4,5"  # pre-PR value, preserved


def test_autofilled_recipe_rocr_leaves_an_unpinned_run_unpinned() -> None:
    """The documented ``{}`` contract has to be reachable in production."""
    assert _resolve_gpu_pin(recipe_envs=_autofilled(4), environ={}) == {}
    assert _resolve_handoff_gpu_ids(gpu_pin={}, tp=4) == "0,1,2,3"


def test_a_real_recipe_rocr_pin_survives_the_autofill_check() -> None:
    assert not _is_autofilled_rocr(value="4,5", recipe_envs={"TP": 2})
    assert _is_autofilled_rocr(value="0,1", recipe_envs={"TP": 2})
    # A recipe that records no TP still gets the materializer's mask written
    # into it, so fall back to the SHAPE of the autofill (0..n-1). Requiring a
    # recipe TP here left the synthetic mask posing as a pin for exactly the
    # recipes that never recorded one.
    assert _is_autofilled_rocr(value="0,1", recipe_envs={})
    assert _is_autofilled_rocr(value="0", recipe_envs={"TP": "not-a-number"})
    # A real pin is still a pin, with or without a TP to compare against.
    assert not _is_autofilled_rocr(value="4,5", recipe_envs={})
    assert not _is_autofilled_rocr(value="6", recipe_envs={})
    assert not _is_autofilled_rocr(value="1,0", recipe_envs={})


def test_variable_precedence_is_global_not_per_source() -> None:
    """A leftover recipe CUDA key must not outrank a real process ROCR pin."""
    out = _resolve_gpu_pin(
        recipe_envs={"TP": 2, "CUDA_VISIBLE_DEVICES": "0"},
        environ={"ROCR_VISIBLE_DEVICES": "6,7"},
    )
    assert out["var"] == "ROCR_VISIBLE_DEVICES"
    assert out["ids"] == [6, 7]


def test_pin_reads_process_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    assert _resolve_gpu_pin()["ids"] == [5]


# --------------------------------------------------------------------------- #
# _resolve_handoff_gpu_ids
# --------------------------------------------------------------------------- #


def test_gpu_ids_unpinned_is_range_tp() -> None:
    """Unchanged legacy behaviour for an unpinned run."""
    assert _resolve_handoff_gpu_ids(gpu_pin={}, tp=4) == "0,1,2,3"
    assert _resolve_handoff_gpu_ids(gpu_pin=None, tp=1) == "0"
    assert _resolve_handoff_gpu_ids(gpu_pin={}, tp=0) == "0"


def test_gpu_ids_rocr_pin_is_logical() -> None:
    """HIP indexes into the ROCr-visible set, so ROCR=6 is HIP index 0."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "6"})
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=1) == "0"

    pin4 = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "4,5,6,7"})
    assert _resolve_handoff_gpu_ids(gpu_pin=pin4, tp=4) == "0,1,2,3"
    # Capped at tp, as the unpinned path always was.
    assert _resolve_handoff_gpu_ids(gpu_pin=pin4, tp=2) == "0,1"
    # ...and at the mask when tp overshoots it: you cannot serve on cards you
    # cannot see.
    assert _resolve_handoff_gpu_ids(gpu_pin=pin4, tp=8) == "0,1,2,3"


def test_gpu_ids_hip_pin_is_verbatim() -> None:
    """No ROCr mask => ROCr shows every card, so HIP ids are absolute."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"HIP_VISIBLE_DEVICES": "4,5"})
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "4,5"


def test_gpu_ids_cuda_pin_is_verbatim() -> None:
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"CUDA_VISIBLE_DEVICES": "3"})
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=1) == "3"


def test_gpu_ids_forwards_a_non_numeric_hip_mask_instead_of_recentring_on_card_0() -> None:
    """A UUID HIP/CUDA mask parses to no numeric ids but is still a real pin.

    Re-serializing from ``ids`` alone yielded ``0..tp-1`` here, which moves the
    servers onto cards ``0..tp-1`` — the #1312 failure, reintroduced for anyone
    who pins by UUID. The tokens are forwarded as-is instead.
    """
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"HIP_VISIBLE_DEVICES": "GPU-a1b2c3,GPU-d4e5f6"})
    assert pin["ids"] == []
    assert pin["count"] == 2
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "GPU-a1b2c3,GPU-d4e5f6"


def test_handoff_tp_never_exceeds_the_advertised_device_count() -> None:
    """``tp`` follows ``gpu_ids`` down so the two cannot disagree.

    ``gpu_ids`` is capped at the mask width, so a stale ``$TP`` used to ship
    alongside fewer ids and GEAK would launch ``--tp N`` against fewer visible
    cards and fail to load weights.
    """
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "6"})
    ids = _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2)
    assert ids == "0"
    assert _resolve_handoff_tp(gpu_ids=ids, tp=2) == 1
    # A four-card pin with a stale TP=8 clamps to the four cards it can see.
    pin4 = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "4,5,6,7"})
    ids4 = _resolve_handoff_gpu_ids(gpu_pin=pin4, tp=8)
    assert ids4 == "0,1,2,3"
    assert _resolve_handoff_tp(gpu_ids=ids4, tp=8) == 4
    # An unpinned run is unaffected.
    assert _resolve_handoff_tp(gpu_ids="0,1", tp=2) == 2


def test_coerce_tp_never_raises_out_of_its_own_fallback() -> None:
    """A non-numeric ``$TP`` must not escape as a ValueError.

    The previous form called a bare ``int()`` inside the ``except`` that was
    handling the identical failure, so a junk ``$TP`` raised during handling.
    """
    assert _coerce_tp("2", "8") == 2
    assert _coerce_tp(None, "8") == 8
    assert _coerce_tp("", "  ") == 1
    assert _coerce_tp("not-a-number", "also-junk") == 1
    assert _coerce_tp("0", "-3", "4") == 4


def test_gpu_ids_never_empty_for_a_blank_mask() -> None:
    """A present-but-empty mask must not produce an empty device list."""
    pin = {"var": "ROCR_VISIBLE_DEVICES", "value": "", "ids": [], "count": 0, "source": "process_env"}
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "0,1"


def test_gpu_ids_counts_a_uuid_mask_instead_of_falling_back_to_card_0() -> None:
    """ROCm accepts UUID masks; they parse to zero numeric ids but N devices.

    Counting ``ids`` here would see an empty list, read the run as unpinned and
    emit ``0..tp-1`` — landing every GEAK server on card 0, the exact default
    this change exists to eliminate.
    """
    pin = _resolve_gpu_pin(
        recipe_envs={},
        environ={"ROCR_VISIBLE_DEVICES": "GPU-a1b2c3,GPU-d4e5f6"},
    )
    assert pin["ids"] == []
    assert pin["count"] == 2
    assert pin["value"] == "GPU-a1b2c3,GPU-d4e5f6"  # re-exportable as-is
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "0,1"


def test_a_yaml_sequence_mask_is_not_stringified_into_junk() -> None:
    """``ROCR_VISIBLE_DEVICES: [4, 5]`` in a recipe is a list, not a string."""
    pin = _resolve_gpu_pin(recipe_envs={"TP": 2, "ROCR_VISIBLE_DEVICES": [4, 5]}, environ={})
    assert pin["value"] == "4,5"
    assert pin["ids"] == [4, 5]


def test_gpu_ids_are_absolute_for_a_recipe_only_rocr_pin() -> None:
    """A mask the child does not inherit cannot be indexed logically.

    The phase launches GEAK with ``dict(os.environ)``, so a mask that exists
    only in the recipe never reaches the child; ROCr shows it every card and
    the absolute ids are the correct HIP indices.
    """
    pin = _resolve_gpu_pin(recipe_envs={"TP": 2, "ROCR_VISIBLE_DEVICES": "6,7"}, environ={})
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "6,7"


# --------------------------------------------------------------------------- #
# Legacy mask spellings, empty masks, nested masks, coordinate space
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("var", "expect_logical"),
    [
        ("ROCR_VISIBLE_DEVICES", True),
        ("HSA_VISIBLE_DEVICES", True),
        ("HIP_VISIBLE_DEVICES", False),
        ("CUDA_VISIBLE_DEVICES", False),
        ("GPU_DEVICE_ORDINAL", False),
    ],
)
def test_every_mask_spelling_counts_as_a_pin(monkeypatch: pytest.MonkeyPatch, var: str, expect_logical: bool) -> None:
    """A run pinned with a legacy spelling is pinned; omitting it read as unpinned."""
    monkeypatch.setenv(var, "6")
    pin = _resolve_gpu_pin(recipe_envs=_autofilled(1))
    assert pin["var"] == var
    assert pin["ids"] == [6]
    # ROCr-level masks renumber the child's devices, HIP-level ones index into them.
    assert (_resolve_handoff_gpu_ids_space(gpu_pin=pin) == "logical") is expect_logical
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=1) == ("0" if expect_logical else "6")


def test_a_recipe_autofill_is_ignored_under_the_legacy_rocr_spelling_too() -> None:
    """``HSA_VISIBLE_DEVICES`` gets the same autofill test as its modern name."""
    pin = _resolve_gpu_pin(
        recipe_envs={"TP": 2, "HSA_VISIBLE_DEVICES": "0,1"},
        environ={},
    )
    assert pin == {}


def test_an_empty_mask_reports_zero_devices_rather_than_unpinned() -> None:
    """``ROCR_VISIBLE_DEVICES=""`` is "no cards", not "whole machine"."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": ""})
    assert pin["var"] == "ROCR_VISIBLE_DEVICES"
    assert pin["count"] == 0
    assert pin["ids"] == []
    # gpu_ids still must not be blank: GEAK reads a falsy gpu_ids as "unset" and
    # falls straight back to 0..tp-1 (interface/run_e2e.py), so an empty string
    # buys nothing. The ids are declared placeholders instead — that is what
    # the third coordinate space exists for.
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "0,1"
    assert _resolve_handoff_gpu_ids_space(gpu_pin=pin) == "none"


def test_a_mask_with_no_valid_ordinal_is_also_reported_as_zero_devices() -> None:
    """``ROCR="-1"`` is non-blank but exposes nothing; it must not read as a pin."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "-1"})
    assert pin["count"] == 0
    assert _resolve_handoff_gpu_ids_space(gpu_pin=pin) == "none"


def test_a_uuid_mask_is_not_mistaken_for_zero_devices() -> None:
    """``ids`` is empty for a UUID mask, but it exposes real cards."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "GPU-a1b2c3,GPU-d4e5f6"})
    assert pin["ids"] == []
    assert pin["count"] == 2
    assert _resolve_handoff_gpu_ids_space(gpu_pin=pin) == "logical"


def test_a_real_pin_outranks_an_earlier_empty_mask() -> None:
    """An empty ROCR must not shadow a real HIP pin further down the chain."""
    pin = _resolve_gpu_pin(
        recipe_envs={},
        environ={"ROCR_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "4,5"},
    )
    assert pin["var"] == "HIP_VISIBLE_DEVICES"
    assert pin["ids"] == [4, 5]


def test_a_hip_mask_nested_in_a_rocr_pin_is_forwarded_not_overwritten() -> None:
    """ROCR=4,5,6,7 + HIP=2,3 is cards 6,7 — advertising 0,1 would move the servers."""
    pin = _resolve_gpu_pin(
        recipe_envs={},
        environ={"ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "2,3"},
    )
    assert pin["var"] == "ROCR_VISIBLE_DEVICES"
    assert pin["inner"]["var"] == "HIP_VISIBLE_DEVICES"
    assert pin["inner"]["ids"] == [2, 3]
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2) == "2,3"


def test_a_nested_hip_mask_pointing_outside_the_rocr_set_is_dropped() -> None:
    """HIP ids beyond the ROCr width name devices the child cannot see."""
    pin = _resolve_gpu_pin(
        recipe_envs={},
        environ={"ROCR_VISIBLE_DEVICES": "6", "HIP_VISIBLE_DEVICES": "3"},
    )
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=1) == "0"


def test_no_inner_mask_is_recorded_for_a_hip_level_pin() -> None:
    """``inner`` only means "nested inside a ROCr slice"; a HIP pin has no inside."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"HIP_VISIBLE_DEVICES": "4,5"})
    assert "inner" not in pin


def test_gpu_ids_space_is_absolute_when_unpinned() -> None:
    assert _resolve_handoff_gpu_ids_space(gpu_pin={}) == "absolute"
    assert _resolve_handoff_gpu_ids_space(gpu_pin=None) == "absolute"


def test_a_recipe_rocr_pin_is_not_inherited_so_its_ids_stay_absolute() -> None:
    """The child inherits the process env, not the recipe's envs."""
    pin = _resolve_gpu_pin(recipe_envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "6"}, environ={})
    assert pin["source"] == "baseline_recipe"
    assert _resolve_handoff_gpu_ids_space(gpu_pin=pin) == "absolute"
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=1) == "6"


def test_a_yaml_sequence_mask_is_joined_not_stringified() -> None:
    """``ROCR_VISIBLE_DEVICES: [4, 5]`` in YAML must not become ``"[4, 5]"``."""
    pin = _resolve_gpu_pin(recipe_envs={"TP": 2, "ROCR_VISIBLE_DEVICES": [4, 5]}, environ={})
    assert pin["value"] == "4,5"
    assert pin["ids"] == [4, 5]
    assert pin["count"] == 2


# --------------------------------------------------------------------------- #
# count / ids cardinality agreement
# --------------------------------------------------------------------------- #


def test_a_repeated_ordinal_does_not_inflate_the_device_count() -> None:
    """ROCR="3,3,2" exposes two devices; logical index 2 would abort the server."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "3,3,2"})
    assert pin["count"] == 2
    assert pin["ids"] == [3, 2]
    assert _resolve_handoff_gpu_ids(gpu_pin=pin, tp=3) == "0,1"


def test_a_negative_ordinal_is_not_counted_as_a_device() -> None:
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"ROCR_VISIBLE_DEVICES": "-1,2"})
    assert pin["count"] == 1
    assert pin["ids"] == [2]


def test_a_repeated_hip_ordinal_keeps_ids_and_count_in_agreement() -> None:
    """The mirror of the count case: gpu_ids must not deflate below ``count``."""
    pin = _resolve_gpu_pin(recipe_envs={}, environ={"HIP_VISIBLE_DEVICES": "4,4"})
    gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2)
    assert gpu_ids == "4"
    assert len(gpu_ids.split(",")) == pin["count"] == 1
    # tp follows gpu_ids down, so GEAK is never told "tp=2" alongside one device.
    assert _resolve_handoff_tp(gpu_ids=gpu_ids, tp=2) == 1


def test_a_yaml_sequence_element_is_stripped_before_it_reaches_value() -> None:
    """``[' 4', 5]`` must not produce ``" 4,5"`` — ROCm's parser rejects the token."""
    pin = _resolve_gpu_pin(recipe_envs={"TP": 2, "ROCR_VISIBLE_DEVICES": [" 4", 5]}, environ={})
    assert pin["value"] == "4,5"
