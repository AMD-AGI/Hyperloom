# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The JavaScript literal reader: literals come back as data, code as its source text, never evaluated."""

from __future__ import annotations

import pytest

from inference_suite.recipe.jslit import JsExpression, JsLiteralError, exported_value, has_expression


def test_literals_read_with_js_syntax():
    text = """
    // a config
    export const config = {
      name: 'glm', "quoted-key": "a\\"b", 7: 0x1f, ratio: -0.85, big: 1e3,
      list: [1, 2, /* inline */ 3,], flag: true, off: false, none: null,
      multi: `line one
    line two`, unicode: "\\u00e9\\x41",
    };
    """
    assert exported_value(text, "config") == {
        "name": "glm",
        "quoted-key": 'a"b',
        "7": 31,
        "ratio": -0.85,
        "big": 1000.0,
        "list": [1, 2, 3],
        "flag": True,
        "off": False,
        "none": None,
        "multi": "line one\n    line two",
        "unicode": "éA",
    }


@pytest.mark.parametrize(
    ("source", "kept"),
    [
        ("(s) => s.hw === 'a5' ? ['x'] : []", "(s) => s.hw === 'a5' ? ['x'] : []"),
        ("BLACKWELL_NOTE", "BLACKWELL_NOTE"),
        ("'a' + suffix", "'a' + suffix"),
        ("`tp ${n}`", "`tp ${n}`"),
        ("[1, 2].flatMap((x) => [x, x])", "[1, 2].flatMap((x) => [x, x])"),
        ("{ ...base, tp: 8 }", "{ ...base, tp: 8 }"),
        ("[...cells]", "[...cells]"),
        ("(() => { return [1]; })()", "(() => { return [1]; })()"),
    ],
)
def test_code_is_kept_as_its_source(source, kept):
    value = exported_value(f"export const config = {{ field: {source}, after: 1 }};", "config")

    assert value == {"field": JsExpression(kept), "after": 1}


def test_a_method_is_code_and_the_rest_of_the_object_stays_literal():
    value = exported_value("export const config = { cellFor(s) { return s, 1; }, cells: [1] };", "config")

    assert value == {"cellFor": JsExpression("(s) { return s, 1; }"), "cells": [1]}


def test_has_expression_looks_inside_containers():
    assert has_expression({"flags": ["--tp 8", JsExpression("x")]})
    assert not has_expression({"flags": ["--tp 8"], "env": []})


def test_a_missing_export_or_unreadable_source_is_an_error():
    with pytest.raises(JsLiteralError, match="exports no const config"):
        exported_value("export const other = {};", "config")
    with pytest.raises(JsLiteralError, match="unterminated string"):
        exported_value('export const config = { name: "glm };', "config")
