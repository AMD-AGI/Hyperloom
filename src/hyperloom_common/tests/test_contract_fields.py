# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared contract field checks raise the error a contract binds them to."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import pytest

from hyperloom_common.contract_fields import ContractFields, field_names


class _OwnError(ValueError):
    pass


class _Color(str, Enum):
    RED = "red"


FIELDS = ContractFields(_OwnError)


@pytest.mark.parametrize(
    ("check", "match"),
    [
        (lambda: FIELDS.enum(_Color, "blue", "color"), r"color must be one of \['red'\]"),
        (lambda: FIELDS.text(" padded", "name"), "name must be a non-empty string"),
        (lambda: FIELDS.matching(re.compile(r"\d+"), "x", "digits"), r"digits must match \\d\+"),
        (lambda: FIELDS.sha256("ABC", "digest"), "digest must be a lowercase hex sha256"),
        (lambda: FIELDS.integer(True, "count", minimum=0), "count must be an integer of at least 0"),
        (lambda: FIELDS.integer(0, "count", minimum=1), "count must be an integer of at least 1"),
        (lambda: FIELDS.number(float("nan"), "rate"), "rate must be a finite number"),
        (lambda: FIELDS.boolean(1, "flag"), "flag must be true or false"),
        (lambda: FIELDS.mapping([], "doc"), "doc must be an object"),
        (lambda: FIELDS.known({"a": 1, "c": 2}, ("a", "b"), "doc"), "unknown key 'c' in doc; known: a, b"),
        (lambda: FIELDS.string_map({"a": 1}, "envs"), "envs must map strings to strings"),
        (lambda: FIELDS.items("ab", "rows"), "rows must be a list"),
        (lambda: FIELDS.strings(["a", 1], "names"), "names must be a list of strings"),
        (lambda: FIELDS.string_tuple(["a"], "names"), "names must be a tuple of strings"),
        (lambda: FIELDS.unique(["a", "b", "a"], "names"), "duplicate names: a"),
    ],
)
def test_each_check_raises_the_bound_error(check, match) -> None:
    with pytest.raises(_OwnError, match=match):
        check()


def test_accepted_values_come_back() -> None:
    assert FIELDS.enum(_Color, "red", "color") is _Color.RED
    assert FIELDS.integer(3, "count", minimum=1) == 3
    assert FIELDS.number(2, "rate") == 2.0
    assert FIELDS.strings(["a", "b"], "names") == ("a", "b")
    assert FIELDS.items(["a"], "rows") == ("a",)
    assert FIELDS.optional_text(None, "name") is None
    assert FIELDS.optional_sha256(None, "digest") is None
    assert FIELDS.optional_integer(None, "count") is None
    assert FIELDS.optional_number(None, "rate") is None
    assert FIELDS.optional_boolean(None, "flag") is None


def test_field_names_are_a_dataclass_s_fields_in_declaration_order():
    @dataclass(frozen=True)
    class Pair:
        second: int
        first: int = 0

    assert field_names(Pair) == ("second", "first")
