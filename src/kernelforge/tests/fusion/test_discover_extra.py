# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover discovery trace, parsing, and source-read edge cases."""

from __future__ import annotations

import inspect
import json

from kernelforge.fusion import discover as discover_module
from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.discover import (
    _extract_json_array,
    _salvage_objects,
    discover_recipes,
    hot_kernels_from_trace,
    parse_discovered_recipes,
)


def _candidate_diag():
    return diagnose_from_shares(
        {"gemm": 0.4, "add": 0.14, "elementwise": 0.14, "cast": 0.13, "mul": 0.08},
        busy_fraction_of_wall=0.21,
    )


def test_hot_kernels_events_not_list(tmp_path):
    path = tmp_path / "d.trace.json"
    path.write_text(json.dumps({"traceEvents": "oops"}))
    assert hot_kernels_from_trace(path) == []


def test_hot_kernels_skips_bad_dur_and_nonpositive(tmp_path):
    path = tmp_path / "d.trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "kernel", "name": "a", "dur": "notnum"},
                    {"cat": "kernel", "name": "b", "dur": 0},
                    {"cat": "kernel", "name": "c", "dur": -5},
                    {"cat": "cpu_op", "name": "skip", "dur": 100},
                    {"cat": "kernel", "name": "mul_kernel", "dur": 20},
                ]
            }
        )
    )
    hot = hot_kernels_from_trace(path)
    assert [row["name"] for row in hot] == ["mul_kernel"]


def test_hot_kernels_total_zero_returns_empty(tmp_path):
    path = tmp_path / "d.trace.json"
    path.write_text(json.dumps({"traceEvents": [{"cat": "kernel", "name": "a", "dur": 0}]}))
    assert hot_kernels_from_trace(path) == []


def test_hot_kernels_gz(tmp_path):
    import gzip

    path = tmp_path / "d.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": [{"cat": "kernel", "name": "mul_k", "dur": 10}]}, handle)
    hot = hot_kernels_from_trace(path)
    assert hot and hot[0]["name"] == "mul_k"


def test_salvage_ignores_braces_in_strings():
    text = '[{"name": "a {nested}", "v": 1}, {"name": "b", "esc": "x\\"y"}]'
    objects = _salvage_objects(text)
    assert len(objects) == 2
    assert objects[0]["name"] == "a {nested}"


def test_extract_empty_text():
    assert _extract_json_array("") == []


def test_extract_skips_invalid_array_then_salvages():
    assert _extract_json_array('[{"name": "a"}, {broken') == [{"name": "a"}]


def test_parse_anchor_as_string_coerced():
    text = '[{"name":"x","env_flag":"F","source_anchors":"single_anchor"}]'
    recipes = parse_discovered_recipes(text, model_type="m", framework="f", source_file="/x.py", shapes={})
    assert recipes[0].source_hints == ["single_anchor"]


def test_discover_recipes_empty_source_file():
    recipes = discover_recipes(
        _candidate_diag(),
        model_type="m",
        framework="f",
        source_file="",
        shapes={},
        trace_path="/x",
        llm_fn=lambda _prompt: "[]",
    )
    assert recipes == []


def test_discover_recipes_unreadable_source(tmp_path):
    recipes = discover_recipes(
        _candidate_diag(),
        model_type="m",
        framework="f",
        source_file=str(tmp_path / "nope.py"),
        shapes={},
        trace_path="/x",
        llm_fn=lambda _prompt: "[]",
    )
    assert recipes == []


def test_discovery_module_has_no_direct_sdk_fallback() -> None:
    source = inspect.getsource(discover_module)
    assert "default_llm_fn" not in source
    assert "from openai import" not in source
    assert "from anthropic import" not in source
