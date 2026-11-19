# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the operator override for the enablement accuracy floor.

The floor is shared by the baseline's eval-failure trigger and the enablement KEEP gate, so it must be one value
both read, out of range or unreadable input must fall back to the shipped default rather than disable the gate,
and the default itself must not move.
"""

from __future__ import annotations

import pytest

from hyperloom.common import gain_math


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv(gain_math.ACCURACY_FLOOR_ENV, raising=False)


def test_the_shipped_default_is_the_serving_class_floor() -> None:
    """Lowering the default for one small model would weaken the gate on the hardware it was set for."""
    assert gain_math.enablement_accuracy_floor() == 0.5


@pytest.mark.parametrize("raw", ["", "   "])
def test_an_empty_setting_is_not_an_override(monkeypatch, raw: str) -> None:
    monkeypatch.setenv(gain_math.ACCURACY_FLOOR_ENV, raw)
    assert gain_math.enablement_accuracy_floor() == 0.5


def test_a_valid_fraction_is_the_floor(monkeypatch) -> None:
    monkeypatch.setenv(gain_math.ACCURACY_FLOOR_ENV, "0.45")
    assert gain_math.enablement_accuracy_floor() == pytest.approx(0.45)


def test_the_upper_bound_is_itself_a_valid_floor(monkeypatch) -> None:
    monkeypatch.setenv(gain_math.ACCURACY_FLOOR_ENV, "1.0")
    assert gain_math.enablement_accuracy_floor() == 1.0


@pytest.mark.parametrize("raw", ["0", "0.0", "-0.1", "1.5", "2", "abc", "nan"])
def test_anything_that_is_not_a_gate_falls_back_to_the_default(monkeypatch, raw: str) -> None:
    """A floor of zero admits any model and one above 1 rejects every one; neither is a gate."""
    monkeypatch.setenv(gain_math.ACCURACY_FLOOR_ENV, raw)
    assert gain_math.enablement_accuracy_floor() == 0.5
