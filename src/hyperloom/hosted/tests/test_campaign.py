# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import json
from importlib import resources

import jsonschema
import pytest

from hyperloom.hosted import campaign
from hyperloom.hosted.campaign import CampaignError, Kind, parse_campaign

FULL = {
    "schema": "hyperloom.campaign-spec/v1",
    "name": "qwen3-8b-mi355x",
    "flow": "multibranch",
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
    "multibranch": {"workload": "inferencex"},
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
#: A multibranch spec measured on the application container's own cards.
LOCAL_MULTIBRANCH = _with(MINIMAL, "placement.measurement", "local")

# (document, the message the refusal must carry)
REFUSALS = [
    ("not an object", "the spec must be a JSON object"),
    (_with(MINIMAL, "schema", "hyperloom.campaign-spec/v2"), "schema must be 'hyperloom.campaign-spec/v1'"),
    (_with(MINIMAL, "schema", _DELETE), "schema must be 'hyperloom.campaign-spec/v1', got None"),
    (_with(MINIMAL, "model", _DELETE), "model is required"),
    (_with(MINIMAL, "model", "/m"), "model must be an object"),
    (_with(MINIMAL, "model.path", _DELETE), "model.path is required"),
    (_with(MINIMAL, "model.path", "  "), "model.path must be a non-empty string"),
    (_with(MINIMAL, "hardware.gpu_type", "\t"), "hardware.gpu_type must be a non-empty string"),
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
    (_with(MINIMAL, "placement.max_parallel_tests", 0), "placement.max_parallel_tests must be an integer >= 1"),
    (_with(MINIMAL, "flow", "mesh"), "flow must be one of 'multibranch', got 'mesh'"),
    (_with(MINIMAL, "flow", "single"), "flow 'single' is retired; leave flow out or set it to 'multibranch'"),
    (_with(MINIMAL, "multibranch", _DELETE), campaign.RETIRED_DEFAULT),
    (
        _with(LOCAL_MULTIBRANCH, "placement.profile", "hosted-cpu"),
        "placement.measurement 'local' measures on the application container's own cards:"
        " placement.profile must be 'hosted-gpu', got 'hosted-cpu'",
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


def test_a_string_a_submitter_wrote_is_not_echoed_in_the_refusal():
    doc = _with(MINIMAL, "launch", {"extra_env": {"HF_TOKEN": 7, "OTHER": "hf_secret_value"}})
    with pytest.raises(CampaignError) as raised:
        parse_campaign(doc)
    assert "launch.extra_env must be an object of string values" in str(raised.value)
    assert "hf_secret_value" not in str(raised.value) and "7" not in str(raised.value).split("values")[1]


def test_an_integer_too_large_for_a_float_is_judged_not_a_crash():
    assert parse_campaign(_with(MINIMAL, "hardware.tp", 10**400)).tp == 10**400
    with pytest.raises(CampaignError, match="hardware.tp must be an integer >= 1"):
        parse_campaign(_with(MINIMAL, "hardware.tp", -(10**400)))


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


def test_a_spec_without_a_flow_is_a_multibranch_one_measured_hosted():
    spec = parse_campaign(MINIMAL)
    assert (spec.measurement, spec.max_parallel_tests) == ("hosted", 8)
    assert parse_campaign({**MINIMAL, "flow": "multibranch"}) == spec


def test_the_retired_single_flow_is_refused_once_naming_why():
    with pytest.raises(CampaignError) as refused:
        parse_campaign({**MINIMAL, "flow": "single"})

    assert refused.value.problems == ("flow 'single' is retired; leave flow out or set it to 'multibranch'",)


def test_an_unchanged_single_spec_without_flow_or_multibranch_is_refused_once_naming_the_retired_flow():
    doc = _with(MINIMAL, "multibranch", _DELETE)
    with pytest.raises(CampaignError) as refused:
        parse_campaign(doc)

    assert refused.value.problems == (
        "a spec with neither flow nor multibranch ran the single flow, which is retired;"
        " set flow to 'multibranch' and add a multibranch section with workload, and give workload.conc and"
        " workload.precision",
    )
    assert parse_campaign({**doc, "flow": "multibranch"}).multibranch is None


def test_a_multibranch_spec_measures_on_the_application_containers_own_cards_when_it_has_them():
    spec = parse_campaign(LOCAL_MULTIBRANCH)

    assert (spec.measurement, spec.profile) == ("local", "hosted-gpu")


def test_a_multibranch_spec_without_a_profile_is_refused_once_for_the_profile():
    with pytest.raises(CampaignError) as refused:
        parse_campaign(_with(LOCAL_MULTIBRANCH, "placement.profile", _DELETE))

    assert refused.value.problems == ("placement.profile is required",)


def test_an_integer_written_as_a_float_is_an_integer():
    assert parse_campaign(_with(MINIMAL, "hardware.tp", 4.0)).tp == 4


def test_the_budget_limits_are_kept():
    spec = parse_campaign(FULL)
    assert (spec.max_gpu_hours, spec.max_llm_usd) == (0, 12.5)


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
        {**MINIMAL, "flow": "multibranch", "placement": {"profile": "hosted-gpu", "measurement": "local"}},
        {**MINIMAL, "flow": "multibranch", "placement": {"profile": "hosted-cpu", "measurement": "local"}},
    ],
)
def test_the_schema_and_the_validator_agree_on_accepting_and_refusing(doc):
    schema_says = jsonschema.Draft202012Validator(_schema()).is_valid(doc)
    try:
        parse_campaign(doc)
        validator_says = True
    except CampaignError:
        validator_says = False
    # The cross-field rule (measurement against profile) is not expressed in the schema; the one allowed gap.
    cross_field = (
        isinstance(doc, dict) and doc.get("flow") in ("multibranch", None) and not validator_says and schema_says
    )
    assert schema_says == validator_says or cross_field
