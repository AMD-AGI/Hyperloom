# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check tool arguments against the subset of JSON Schema the MCP tools use.

The subset: ``type`` (one name or a list), ``enum``, ``properties``, ``required``, ``additionalProperties`` (a
boolean), ``items``, ``minLength``, ``minimum`` and ``maximum``. A keyword outside it is ignored, so a schema stays
a description for the client even where this check does not enforce it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _is_type(value: Any, name: str) -> bool:
    if isinstance(value, bool) and name in {"integer", "number"}:
        return False
    return isinstance(value, _TYPES.get(name, ()))


def _json_equal(left: Any, right: Any) -> bool:
    """Equality as JSON Schema defines it: a boolean never equals a number, at any depth."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(_json_equal, left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    return left == right


def _bounds(value: Any, schema: Mapping[str, Any], where: str) -> str | None:
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return f"{where} is shorter than {schema['minLength']}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{where} is below {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{where} is above {schema['maximum']}"
    return None


def _members(value: dict[str, Any], schema: Mapping[str, Any], where: str) -> str | None:
    properties = schema.get("properties", {})
    for name in schema.get("required", ()):
        if name not in value:
            return f"{where}: {name} is required"
    for name, item in value.items():
        if name in properties:
            problem = first_problem(item, properties[name], f"{where}.{name}")
            if problem:
                return problem
        elif schema.get("additionalProperties", True) is False:
            return f"{where}: additional properties are not allowed ({name})"
    return None


def first_problem(value: Any, schema: Mapping[str, Any], where: str = "arguments") -> str | None:
    """The first way ``value`` breaks ``schema``, or ``None`` when it fits."""
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(value, name) for name in names):
            return f"{where} must be of type {' or '.join(names)}"
    if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
        return f"{where} must be one of {schema['enum']}"
    if isinstance(value, dict):
        return _members(value, schema, where)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            problem = first_problem(item, schema["items"], f"{where}[{index}]")
            if problem:
                return problem
    return _bounds(value, schema, where)
