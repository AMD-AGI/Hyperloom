# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math

import pytest

from hyperloom.common.coerce import (
    first_float,
    first_int,
    optional_positive_int,
    to_float,
    to_int,
    to_unix,
)


class TestToFloat:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, 1.0),
            (1.5, 1.5),
            ("2.5", 2.5),
            ("  3.0  ", 3.0),
            (-4, -4.0),
        ],
    )
    def test_parses(self, value, expected):
        assert to_float(value) == expected

    @pytest.mark.parametrize("value", [None, True, False, "abc", "", object(), [1]])
    def test_rejects_to_default_none(self, value):
        assert to_float(value) is None

    def test_bool_rejected_even_though_int_subclass(self):
        assert to_float(True) is None
        assert to_float(False) is None

    def test_custom_default(self):
        assert to_float("nope", default=0.0) == 0.0
        assert to_float(None, default=-1.0) == -1.0


class TestToInt:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, 1),
            ("2", 2),
            ("  3  ", 3),
            (-4, -4),
        ],
    )
    def test_parses(self, value, expected):
        assert to_int(value) == expected

    @pytest.mark.parametrize("value", [None, True, False, "abc", "", "1.5"])
    def test_non_int_string_and_float_string_reject(self, value):
        # A float *string* like "1.5" is not a valid int(); bool/None/garbage
        # collapse to the default. A real float (2.5) truncates -- see below.
        assert to_int(value) is None

    def test_real_float_truncates(self):
        assert to_int(2.9) == 2

    def test_custom_default(self):
        assert to_int("nope", default=0) == 0


class TestFirst:
    def test_first_float_returns_first_parseable(self):
        assert first_float(None, "x", "3.5", 4) == 3.5

    def test_first_float_default_when_none(self):
        assert first_float(None, "x", default=9.0) == 9.0

    def test_first_int_returns_first_parseable(self):
        assert first_int(None, "x", "7", 8) == 7

    def test_first_int_default_when_none(self):
        assert first_int("x", None, default=-1) == -1

    def test_bool_skipped(self):
        assert first_float(True, "2.0") == 2.0


class TestOptionalPositiveInt:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, 5),
            ("10", 10),
            (" 3 ", 3),
        ],
    )
    def test_positive(self, value, expected):
        assert optional_positive_int(value) == expected

    @pytest.mark.parametrize("value", [0, -1, "0", "-5", None, True, "x", ""])
    def test_non_positive_or_invalid_to_default(self, value):
        assert optional_positive_int(value) is None

    def test_custom_default(self):
        assert optional_positive_int(0, default=1) == 1


class TestToUnix:
    def test_numeric_epoch(self):
        assert to_unix(1_700_000_000) == 1_700_000_000.0
        assert to_unix(1_700_000_000.5) == 1_700_000_000.5

    def test_iso_z_suffix(self):
        ts = to_unix("2021-01-01T00:00:00Z")
        assert ts is not None
        assert math.isclose(ts, 1_609_459_200.0)

    def test_iso_offset(self):
        ts = to_unix("2021-01-01T00:00:00+00:00")
        assert ts is not None
        assert math.isclose(ts, 1_609_459_200.0)

    def test_numeric_string_fallback(self):
        assert to_unix("1700000000") == 1_700_000_000.0

    @pytest.mark.parametrize("value", [None, True, False, "not-a-ts", object()])
    def test_reject_to_default(self, value):
        assert to_unix(value) is None

    def test_custom_default(self):
        assert to_unix("bad", default=0.0) == 0.0
