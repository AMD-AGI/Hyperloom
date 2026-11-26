# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A kernel knob of the pack: how it forces a value per key, and what the loader refuses."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from hyperloom.common.outside_facts.kernel_knob import KnobSpecError, parse_knob
from hyperloom.common.outside_facts.rule_pack import DEFAULT_PACK_PATH, RulePackError, load_pack

_ANCHOR = "    if (table_id == RDNA3) {"
_SOURCE = "int f(int type, int n) {\n" + _ANCHOR + "\n        return 1;\n    }\n    return 2;\n}\n"
_BODY = {
    "component": "llamacpp",
    "file": "ggml/src/k.cu",
    "anchor": _ANCHOR,
    "insert": "        if (type == ${type} && n == ${ncols}) { return ${value}; }",
    "axes": {"type": ["Q8_0", "Q4_0"], "ncols": ["1", "2"]},
    "values": ["1", "2", "4"],
    "archs": ["gfx1100"],
    "bench": {"type_a": "${type}", "n": "${ncols}"},
}


def _knob(**changes):
    return parse_knob("entry 'k'", {**_BODY, **changes})


@pytest.mark.parametrize("source", [_SOURCE.replace(_ANCHOR, "    if (x) {"), _SOURCE + _ANCHOR + "\n"])
def test_apply_refuses_a_source_without_the_anchor_exactly_once(source: str) -> None:
    with pytest.raises(KnobSpecError, match="not exactly one"):
        _knob().apply(source, {("Q8_0", "1"): "2"})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"insert": "if (type == ${type}) { return ${value}; }"}, "insert must name each axis"),
        ({"bench": {"type_a": "${type}", "n": "1"}}, "does not tell the keys apart"),
        ({"file": "../outside.cu"}, "path in the upstream tree"),
    ],
)
def test_parse_refuses_a_knob_that_cannot_be_used(changes, message: str) -> None:
    with pytest.raises(KnobSpecError, match=message):
        _knob(**changes)


def test_the_loader_refuses_a_knob_whose_example_does_not_give_its_expected_text(tmp_path: Path) -> None:
    document = yaml.safe_load(DEFAULT_PACK_PATH.read_text(encoding="utf-8"))
    entry = copy.deepcopy(document["facts"]["kernel_knob_llamacpp_mmvq_nwarps_rdna3"])
    entry["examples"][0]["expected"] = entry["examples"][0]["source"]
    document["facts"]["kernel_knob_llamacpp_mmvq_nwarps_rdna3"] = entry
    path = tmp_path / "pack.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(RulePackError, match="(?s)kernel_knob_llamacpp_mmvq_nwarps_rdna3.*not its expected text"):
        load_pack(path)
