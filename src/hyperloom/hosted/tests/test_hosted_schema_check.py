# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.hosted.schema_check import first_problem

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "kind": {"enum": ["a", "b"]},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "maybe": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "flag": {"type": "boolean"},
        "inner": {"type": "object", "properties": {"x": {"type": "number"}}, "additionalProperties": False},
    },
    "required": ["name"],
    "additionalProperties": False,
}


def test_a_fitting_value_has_no_problem() -> None:
    value = {"name": "ab", "kind": "a", "count": 5, "maybe": None, "tags": ["t"], "flag": True, "inner": {"x": 1.5}}
    assert first_problem(value, SCHEMA) is None


@pytest.mark.parametrize(
    ("value", "needle"),
    [
        ({}, "name is required"),
        ({"name": 5}, "must be of type string"),
        ({"name": "a"}, "shorter than 2"),
        ({"name": "ab", "kind": "c"}, "must be one of"),
        ({"name": "ab", "count": 0}, "below 1"),
        ({"name": "ab", "count": 6}, "above 5"),
        ({"name": "ab", "count": True}, "must be of type integer"),
        ({"name": "ab", "count": 1.5}, "must be of type integer"),
        ({"name": "ab", "maybe": 3}, "string or null"),
        ({"name": "ab", "tags": ["t", 3]}, "tags[1] must be of type string"),
        ({"name": "ab", "flag": "yes"}, "must be of type boolean"),
        ({"name": "ab", "extra": 1}, "additional properties are not allowed (extra)"),
        ({"name": "ab", "inner": {"x": "s"}}, "arguments.inner.x must be of type number"),
        ({"name": "ab", "inner": {"y": 1}}, "additional properties are not allowed (y)"),
    ],
)
def test_each_violation_is_named(value: dict[str, Any], needle: str) -> None:
    assert needle in (first_problem(value, SCHEMA) or "")


@pytest.mark.parametrize(
    ("value", "options", "fits"),
    [
        (True, [1], False),
        (False, [0], False),
        (1, [True], False),
        ([True], [[1]], False),
        ({"on": 1}, [{"on": True}], False),
        (True, [True], True),
        (1, [1], True),
        (1.0, [1], True),
        ([1, "a"], [[1, "a"]], True),
        ({"on": True}, [{"on": True}], True),
    ],
)
def test_enum_compares_as_json_so_a_boolean_is_not_a_number(value: Any, options: list[Any], fits: bool) -> None:
    assert (first_problem(value, {"enum": options}) is None) is fits
