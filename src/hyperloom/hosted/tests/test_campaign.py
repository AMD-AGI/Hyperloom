# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import json
from importlib import resources

import jsonschema
import pytest

from hyperloom.hosted import campaign
from hyperloom.hosted.campaign import CampaignError, Kind, optimize_args, parse_campaign
from hyperloom.inference_optimizer.cli.parser import _build_parser

FULL = {
    "schema": "hyperloom.campaign-spec/v1",
    "name": "qwen3-8b-mi355x",
    "flow": "single",
    "multibranch": {"anything": ["goes", 1, None]},
    "model": {"path": "/shared/models/Qwen3-8B"},
    "framework": {"name": "sglang", "image": "registry.example/sglang:v0.5.4-rocm"},
    "hardware": {"gpu_type": "MI355X", "tp": 2, "ep": 1, "nodes": 1},
    "workload": {"conc": 64, "isl": 1024, "osl": 512, "precision": "bf16"},
    "objective": {"target_gain_pct": 10},
    "budget": {"max_hours": 8, "max_gpu_hours": 0, "max_llm_usd": 12.5},
    "launch": {"server_args": "--mem-fraction-static 0.8", "extra_env": {"A": "1", "B": "x=y"}},
    "options": {"kernel": False, "framework_agent": True, "eval": True, "enablement": "launch"},
    "placement": {
        "profile": "hosted-gpu",
        "measurement": "local",
        "max_parallel_tests": 4,
        "specialist_capacity": 3,
    },
    "source": {"repo": "AMD-AGI/Hyperloom", "sha": "abc123", "kind": "ci", "task_id": "run-123"},
}
MINIMAL = {
    "schema": "hyperloom.campaign-spec/v1",
    "model": {"path": "/m"},
    "framework": {"name": "vllm", "image": "img"},
    "hardware": {"gpu_type": "MI300X", "tp": 1},
    "budget": {"max_hours": 1},
    "placement": {"profile": "hosted-gpu"},
}


def _with(base, path, value):
    doc = copy.deepcopy(base)
    node = doc
    *parents, leaf = path.split(".")
    for key in parents:
        node = node[key]
    if value is _DELETE:
        node.pop(leaf)
    else:
        node[leaf] = value
    return doc


_DELETE = object()

# (document, the message the refusal must carry)
REFUSALS = [
    ("not an object", "the spec must be a JSON object"),
    (_with(MINIMAL, "schema", "hyperloom.campaign-spec/v2"), "schema must be 'hyperloom.campaign-spec/v1'"),
    (_with(MINIMAL, "schema", _DELETE), "schema must be 'hyperloom.campaign-spec/v1', got None"),
    (_with(MINIMAL, "model", _DELETE), "model is required"),
    (_with(MINIMAL, "model", "/m"), "model must be an object"),
    (_with(MINIMAL, "model.path", _DELETE), "model.path is required"),
    (_with(MINIMAL, "model.path", "  "), "model.path must be a non-empty string"),
    (_with(MINIMAL, "framework.image", _DELETE), "framework.image is required"),
    (_with(MINIMAL, "framework.name", 3), "framework.name must be a non-empty string"),
    (_with(MINIMAL, "hardware.gpu_type", _DELETE), "hardware.gpu_type is required"),
    (_with(MINIMAL, "hardware.tp", _DELETE), "hardware.tp is required"),
    (_with(MINIMAL, "hardware.tp", 0), "hardware.tp must be an integer >= 1"),
    (_with(MINIMAL, "hardware.tp", True), "hardware.tp must be an integer >= 1, got True"),
    (_with(MINIMAL, "hardware.tp", 1.5), "hardware.tp must be an integer >= 1, got 1.5"),
    (_with(MINIMAL, "hardware.nodes", 2), "hardware.nodes must be one of 1"),
    (_with(MINIMAL, "workload", {"conc": -1}), "workload.conc must be an integer >= 1"),
    (_with(MINIMAL, "workload", {"precision": ""}), "workload.precision must be a non-empty string"),
    (_with(MINIMAL, "objective", {"target_gain_pct": 0}), "objective.target_gain_pct must be a number > 0"),
    (_with(MINIMAL, "budget.max_hours", _DELETE), "budget.max_hours is required"),
    (_with(MINIMAL, "budget.max_hours", 0), "budget.max_hours must be a number > 0, got 0"),
    (_with(MINIMAL, "budget.max_hours", "8"), "budget.max_hours must be a number > 0, got '8'"),
    (_with(MINIMAL, "budget.max_gpu_hours", -1), "budget.max_gpu_hours must be a number >= 0"),
    (_with(MINIMAL, "budget.max_llm_usd", -0.1), "budget.max_llm_usd must be a number >= 0"),
    (_with(MINIMAL, "launch", {"extra_env": {"A": 1}}), "launch.extra_env must be an object of string values"),
    (_with(MINIMAL, "launch", {"extra_env": {"A=B": "1"}}), "launch.extra_env has the invalid name 'A=B'"),
    (_with(MINIMAL, "launch", {"server_args": ["x"]}), "launch.server_args must be a string"),
    (_with(MINIMAL, "options", {"kernel": "yes"}), "options.kernel must be true or false"),
    (_with(MINIMAL, "options", {"eval": False}), "options.eval must be one of True, got False"),
    (_with(MINIMAL, "options", {"enablement": "some"}), "options.enablement must be one of 'all'"),
    (_with(MINIMAL, "placement.profile", _DELETE), "placement.profile is required"),
    (_with(MINIMAL, "placement.profile", "local"), "placement.profile must be one of 'hosted-gpu', 'hosted-cpu'"),
    (_with(MINIMAL, "placement.measurement", "hosted"), "placement.measurement must be 'local' for flow 'single'"),
    (_with(MINIMAL, "placement.max_parallel_tests", 0), "placement.max_parallel_tests must be an integer >= 1"),
    (_with(MINIMAL, "flow", "mesh"), "flow must be one of 'single', 'multibranch'"),
    (
        _with(_with(MINIMAL, "flow", "multibranch"), "placement.measurement", "local"),
        "must be 'hosted' for flow 'multibranch'",
    ),
    (_with(MINIMAL, "source", {"kind": "bot"}), "source.kind must be one of 'user', 'ci'"),
    (
        _with(MINIMAL, "placement", {"profile": "hosted-gpu", "specialist_capacity": -1}),
        "specialist_capacity must be an integer >= 0",
    ),
]


@pytest.mark.parametrize(("doc", "message"), REFUSALS)
def test_each_refusal_carries_its_own_message(doc, message):
    with pytest.raises(CampaignError) as raised:
        parse_campaign(doc)
    assert message in str(raised.value)


def test_every_problem_is_reported_at_once():
    doc = _with(_with(MINIMAL, "hardware.tp", 0), "budget.max_hours", _DELETE)
    with pytest.raises(CampaignError) as raised:
        parse_campaign(doc)
    assert len(raised.value.problems) == 2


@pytest.mark.parametrize("multibranch", [None, 42, "text", [], {}, {"specialists": "many", "sandboxes": -1}])
def test_the_multibranch_section_is_not_validated_and_is_kept_as_given(multibranch):
    doc = {**MINIMAL, "multibranch": multibranch}
    assert parse_campaign(doc).multibranch == multibranch


def test_unknown_fields_are_ignored():
    doc = {**MINIMAL, "future": {"x": 1}, "model": {"path": "/m", "revision": "r"}}
    assert parse_campaign(doc).model_path == "/m"


def test_the_platform_only_placement_fields_are_not_part_of_the_spec():
    names = {field.path for field in campaign.FIELDS}
    gone = {
        "container_gpus",
        "container_cpu",
        "container_memory_gi",
        "gpu_workspace",
        "app_workspace",
        "subrun_gpu_workspace",
    }
    assert not {name for name in names if name.split(".")[-1] in gone}


def test_defaults_follow_the_flow():
    assert (parse_campaign(MINIMAL).flow, parse_campaign(MINIMAL).measurement) == ("single", "local")
    multibranch = parse_campaign({**MINIMAL, "flow": "multibranch"})
    assert (multibranch.measurement, multibranch.max_parallel_tests) == ("hosted", 8)


def test_an_integer_written_as_a_float_is_an_integer():
    assert parse_campaign(_with(MINIMAL, "hardware.tp", 4.0)).tp == 4


def test_the_budget_limits_that_optimize_does_not_enforce_are_kept_not_mapped():
    spec = parse_campaign(FULL)
    assert (spec.max_gpu_hours, spec.max_llm_usd) == (0, 12.5)
    assert not [arg for arg in optimize_args(spec) if "gpu-hours" in arg or "llm" in arg]


# -- the mapping to `optimize` ------------------------------------------------------------------------------------
def test_the_mapping_reaches_the_real_parser():
    argv = optimize_args(parse_campaign(FULL), session_dir=None)
    parsed = _build_parser().parse_args(argv)
    assert (str(parsed.model), parsed.framework, parsed.gpu_type.lower(), parsed.tp, parsed.ep) == (
        "/shared/models/Qwen3-8B",
        "sglang",
        "mi355x",
        2,
        1,
    )
    assert (parsed.conc, parsed.isl, parsed.osl, parsed.precision) == (64, 1024, 512, "bf16")
    assert (parsed.target_gain, parsed.max_hours, parsed.enablement, parsed.research_lane_capacity) == (
        10,
        8,
        "launch",
        3,
    )
    assert parsed.server_args == "--mem-fraction-static 0.8"
    assert parsed.extra_env == ["A=1", "B=x=y"]
    assert (parsed.no_kernel, parsed.no_framework_agent) == (True, False)


def test_a_value_that_looks_like_a_flag_is_bound_to_its_flag():
    spec = parse_campaign(_with(MINIMAL, "launch", {"server_args": "--tp"}))
    assert _build_parser().parse_args(optimize_args(spec)).server_args == "--tp"


def test_the_minimal_spec_leaves_everything_else_to_optimize():
    assert optimize_args(parse_campaign(MINIMAL)) == [
        "optimize",
        "--model=/m",
        "--framework=vllm",
        "--gpu-type=MI300X",
        "--tp=1",
        "--max-hours=1",
    ]


def test_the_plug_profile_and_session_dir_are_appended_when_given():
    argv = optimize_args(parse_campaign(MINIMAL), plug_profile="hosted-gpu", session_dir="/w/m/s")
    assert argv[-2:] == ["--plug-profile=hosted-gpu", "--session-dir=/w/m/s"]


# -- the schema file and the validator say the same thing ---------------------------------------------------------
_JSON_TYPE = {
    Kind.STR: "string",
    Kind.TEXT: "string",
    Kind.INT: "integer",
    Kind.NUM: "number",
    Kind.BOOL: "boolean",
    Kind.STR_MAP: "object",
    Kind.ENUM: "enum",  # the schema spells these with enum or const and no type
}


def _schema():
    text = resources.files("hyperloom.contracts").joinpath("campaign_spec.v1.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def _schema_field(schema, path):
    node = schema
    for key in path.split("."):
        node = node["properties"][key]
    return node


def test_the_schema_is_a_valid_schema():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_the_table_and_the_schema_name_the_same_fields_with_the_same_bounds():
    schema = _schema()
    for field in campaign.FIELDS:
        node = _schema_field(schema, field.path)
        section, _, leaf = field.path.rpartition(".")
        parent = _schema_field(schema, section) if section else schema
        assert (leaf in parent.get("required", [])) == field.required, field.path
        assert node.get("minimum") == field.minimum, field.path
        assert node.get("exclusiveMinimum") == field.exclusive_minimum, field.path
        choices = tuple(node["enum"]) if "enum" in node else ((node["const"],) if "const" in node else ())
        assert choices == field.choices, field.path
        assert node.get("type", "enum") == _JSON_TYPE[field.kind], field.path
    in_table = {f.path for f in campaign.FIELDS}
    in_schema = {
        f"{section}.{leaf}"
        for section, node in schema["properties"].items()
        if node.get("type") == "object"
        for leaf in node.get("properties", {})
    } | {key for key, node in schema["properties"].items() if node.get("type") != "object"}
    assert in_table == in_schema - {"schema", "multibranch"}
    assert set(schema["required"]) == {"schema", *campaign.REQUIRED_SECTIONS}


@pytest.mark.parametrize(
    "doc",
    [
        FULL,
        MINIMAL,
        *(doc for doc, _ in REFUSALS[1:]),
        {**MINIMAL, "flow": "multibranch", "placement": {"profile": "hosted-gpu", "measurement": "hosted"}},
    ],
)
def test_the_schema_and_the_validator_agree_on_accepting_and_refusing(doc):
    schema_says = jsonschema.Draft202012Validator(_schema()).is_valid(doc)
    try:
        parse_campaign(doc)
        validator_says = True
    except CampaignError:
        validator_says = False
    # The cross-field rule (flow against measurement) is not expressible in the schema; it is the one allowed gap.
    cross_field = (
        isinstance(doc, dict)
        and doc.get("flow") in ("single", "multibranch", None)
        and not validator_says
        and schema_says
    )
    assert schema_says == validator_says or cross_field
