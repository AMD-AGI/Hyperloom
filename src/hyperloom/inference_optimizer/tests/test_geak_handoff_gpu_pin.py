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
    _is_autofilled_rocr,
    _parse_device_list,
    _resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
)


def _autofilled(tp: int) -> dict[str, object]:
    """The ``benchmark.envs`` every materialized recipe carries.

    ``materialize_config_with_envs`` writes ``ROCR_VISIBLE_DEVICES=0..tp-1``
    unconditionally when the mask is absent or narrower than TP, so this shape
    — not an empty mapping — is what the resolver sees in production.
    """
    return {"TP": tp, "ROCR_VISIBLE_DEVICES": ",".join(str(i) for i in range(tp))}


_MASK_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


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


def test_pin_skips_blank_values() -> None:
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
    # No resolved TP => cannot claim it was synthesized; keep the mask.
    assert not _is_autofilled_rocr(value="0,1", recipe_envs={})


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
