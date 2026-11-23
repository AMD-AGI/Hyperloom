# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign spec a hosted run is given, validated field by field.

The contract is ``contracts/campaign_spec.v1.schema.json``.  This module does not use a JSON Schema library at run
time; :data:`FIELDS` is the same contract as a table, and a test holds the two together.  Unknown fields are ignored
(a reader of one major version accepts what a later minor version adds).  ``multibranch`` is not looked at: it is kept
as given for the multibranch entry.  ``flow`` may be left out; the only flow is ``multibranch``, and the retired
``single`` is refused, naming why.  A spec with neither ``flow`` nor ``multibranch`` is one written for the retired
default, the single flow, and is refused the same way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA = "hyperloom.campaign-spec/v1"
FLOW = "flow"
FLOW_MULTIBRANCH = "multibranch"
MULTIBRANCH_SECTION = "multibranch"
#: The legacy single-session flow, retired: a spec that still names it is refused with this reason.
RETIRED_FLOWS = {"single": "flow 'single' is retired; leave flow out or set it to 'multibranch'"}
#: A spec with neither ``flow`` nor ``multibranch`` relied on the retired default, the single flow: refused, naming it.
RETIRED_DEFAULT = (
    "a spec with neither flow nor multibranch ran the single flow, which is retired;"
    " set flow to 'multibranch' and add a multibranch section with workload, and give workload.conc and"
    " workload.precision"
)
MEASUREMENT_LOCAL = "local"
MEASUREMENT_HOSTED = "hosted"
#: The profile whose application container has cards of its own, the only one a multibranch run measures locally on.
PROFILE_WITH_CARDS = "hosted-gpu"
DEFAULT_MAX_PARALLEL_TESTS = 8


class CampaignError(ValueError):
    """The spec is refused.  ``problems`` holds every reason found, each naming its field."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("campaign spec refused: " + "; ".join(problems))


class Kind(str, Enum):
    """What a field's value must be."""

    STR = "non-empty string"
    TEXT = "string"
    INT = "integer"
    NUM = "number"
    BOOL = "boolean"
    STR_MAP = "object of strings"
    ENUM = "one of a fixed set"


@dataclass(frozen=True)
class Field:
    """One scalar field of the spec: where it is, what it must be, and the attribute it lands on."""

    path: str  # dotted: "hardware.tp"
    attr: str
    kind: Kind
    required: bool = False
    minimum: float | None = None
    exclusive_minimum: float | None = None
    choices: tuple[Any, ...] = ()


#: The contract as a table.  Section-level requirements follow from the ``required`` flags below.
FIELDS: tuple[Field, ...] = (
    Field("name", "name", Kind.TEXT),
    Field(FLOW, FLOW, Kind.ENUM, choices=(FLOW_MULTIBRANCH,)),
    Field("model.path", "model_path", Kind.STR, required=True),
    Field("framework.name", "framework_name", Kind.STR, required=True),
    Field("framework.image", "framework_image", Kind.STR, required=True),
    Field("hardware.gpu_type", "gpu_type", Kind.STR, required=True),
    Field("hardware.tp", "tp", Kind.INT, required=True, minimum=1),
    Field("hardware.ep", "ep", Kind.INT, minimum=1),
    Field("hardware.nodes", "nodes", Kind.ENUM, choices=(1,)),
    Field("workload.conc", "conc", Kind.INT, minimum=1),
    Field("workload.isl", "isl", Kind.INT, minimum=1),
    Field("workload.osl", "osl", Kind.INT, minimum=1),
    Field("workload.precision", "precision", Kind.STR),
    Field("objective.target_gain_pct", "target_gain_pct", Kind.NUM, exclusive_minimum=0),
    Field("budget.max_hours", "max_hours", Kind.NUM, required=True, exclusive_minimum=0),
    Field("budget.max_gpu_hours", "max_gpu_hours", Kind.NUM, minimum=0),
    Field("budget.max_llm_usd", "max_llm_usd", Kind.NUM, minimum=0),
    Field("launch.server_args", "server_args", Kind.TEXT),
    Field("launch.extra_env", "extra_env", Kind.STR_MAP),
    Field("options.kernel", "kernel", Kind.BOOL),
    Field("options.framework_agent", "framework_agent", Kind.BOOL),
    Field("options.eval", "eval", Kind.ENUM, choices=(True,)),
    Field("options.enablement", "enablement", Kind.ENUM, choices=("all", "launch", "eval", "off")),
    Field("placement.profile", "profile", Kind.ENUM, required=True, choices=(PROFILE_WITH_CARDS, "hosted-cpu")),
    Field("placement.measurement", "measurement", Kind.ENUM, choices=(MEASUREMENT_LOCAL, MEASUREMENT_HOSTED)),
    Field("placement.max_parallel_tests", "max_parallel_tests", Kind.INT, minimum=1),
    Field("placement.specialist_capacity", "specialist_capacity", Kind.INT, minimum=0),
    Field("source.repo", "source_repo", Kind.STR),
    Field("source.sha", "source_sha", Kind.STR),
    Field("source.kind", "source_kind", Kind.ENUM, choices=("user", "ci")),
    Field("source.task_id", "source_task_id", Kind.TEXT),
)

#: Sections that must be present because one of their fields is required.
REQUIRED_SECTIONS: tuple[str, ...] = tuple(dict.fromkeys(f.path.split(".")[0] for f in FIELDS if f.required))


@dataclass(frozen=True)
class CampaignSpec:
    """A validated spec.  ``None`` means the field was not given."""

    measurement: str
    max_parallel_tests: int
    multibranch: Any
    name: str | None
    model_path: str
    framework_name: str
    framework_image: str
    gpu_type: str
    tp: int
    max_hours: float
    profile: str
    ep: int | None = None
    conc: int | None = None
    isl: int | None = None
    osl: int | None = None
    precision: str | None = None
    target_gain_pct: float | None = None
    max_gpu_hours: float | None = None
    max_llm_usd: float | None = None
    server_args: str | None = None
    extra_env: Mapping[str, str] | None = None
    kernel: bool | None = None
    framework_agent: bool | None = None
    eval: bool | None = None
    enablement: str | None = None
    specialist_capacity: int | None = None
    source_repo: str | None = None
    source_sha: str | None = None
    source_kind: str | None = None
    source_task_id: str | None = None


def _is_num(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))


def _is_int(value: object) -> bool:
    """JSON Schema's integer: ``8`` and ``8.0`` both are."""
    if isinstance(value, int):
        return not isinstance(value, bool)
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


_DESCRIPTIONS = {
    Kind.STR: "a non-empty string",
    Kind.TEXT: "a string",
    Kind.BOOL: "true or false",
    Kind.STR_MAP: "an object of string values",
}


def _describe(field: Field) -> str:
    if field.kind == Kind.ENUM:
        return "one of " + ", ".join(repr(c) for c in field.choices)
    if field.kind in (Kind.INT, Kind.NUM):
        noun = "an integer" if field.kind == Kind.INT else "a number"
        if field.exclusive_minimum is not None:
            return f"{noun} > {field.exclusive_minimum:g}"
        return noun if field.minimum is None else f"{noun} >= {field.minimum:g}"
    return _DESCRIPTIONS[field.kind]


def _well_typed(field: Field, value: object) -> bool:
    kind = field.kind
    if kind == Kind.STR:
        return isinstance(value, str) and bool(value.strip())
    if kind == Kind.TEXT:
        return isinstance(value, str)
    if kind == Kind.BOOL:
        return isinstance(value, bool)
    if kind == Kind.ENUM:
        return any(value == c and type(value) is type(c) for c in field.choices)
    if kind == Kind.STR_MAP:
        return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
    if not (_is_int(value) if kind == Kind.INT else _is_num(value)):
        return False
    if field.minimum is not None and value < field.minimum:
        return False
    return field.exclusive_minimum is None or value > field.exclusive_minimum


_ECHOED = (Kind.INT, Kind.NUM, Kind.BOOL, Kind.ENUM)


def _got(field: Field, value: object) -> str:
    """What was found, for a field whose values are not free text: a string a submitter put in is never echoed."""
    return f", got {value!r:.40}" if field.kind in _ECHOED else ""


def _lookup(doc: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """Whether the dotted path is present, and its value."""
    section, _, key = path.rpartition(".")
    parent: Any = doc.get(section) if section else doc
    return (key in parent, parent.get(key))


def _check_sections(doc: Mapping[str, Any], problems: list[str]) -> None:
    for section in dict.fromkeys(f.path.split(".")[0] for f in FIELDS if "." in f.path):
        if section in doc and not isinstance(doc[section], dict):
            problems.append(f"{section} must be an object")
        elif section in REQUIRED_SECTIONS and section not in doc:
            problems.append(f"{section} is required")


def _read_fields(doc: Mapping[str, Any], problems: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in FIELDS:
        section = field.path.split(".")[0]
        if "." in field.path and not isinstance(doc.get(section), dict):
            continue  # reported once by _check_sections
        present, value = _lookup(doc, field.path)
        if not present:
            if field.required:
                problems.append(f"{field.path} is required")
        elif field.attr == FLOW and isinstance(value, str) and value in RETIRED_FLOWS:
            problems.append(RETIRED_FLOWS[value])
        elif not _well_typed(field, value):
            problems.append(f"{field.path} must be {_describe(field)}{_got(field, value)}")
        else:
            values[field.attr] = int(value) if field.kind == Kind.INT else value
    return values


def _check_rules(values: Mapping[str, Any], problems: list[str]) -> None:
    measurement = values.get("measurement", MEASUREMENT_HOSTED)
    profile = values.get("profile")
    if measurement == MEASUREMENT_LOCAL and profile not in (None, PROFILE_WITH_CARDS):
        problems.append(
            f"placement.measurement 'local' measures on the application container's own cards:"
            f" placement.profile must be {PROFILE_WITH_CARDS!r}, got {profile!r}"
        )
    for key in values.get("extra_env") or {}:
        if not key or "=" in key:
            problems.append(f"launch.extra_env has the invalid name {key!r}")


def parse_campaign(doc: object) -> CampaignSpec:
    """Check *doc* (a parsed JSON document) and return the spec; every problem found is reported in one error."""
    if not isinstance(doc, dict):
        raise CampaignError(["the spec must be a JSON object"])
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    if FLOW not in doc and MULTIBRANCH_SECTION not in doc:
        problems.append(RETIRED_DEFAULT)
    _check_sections(doc, problems)
    values = _read_fields(doc, problems)
    _check_rules(values, problems)
    if problems:
        raise CampaignError(problems)
    values.pop("nodes", None)
    values.pop(FLOW, None)  # the only flow
    return CampaignSpec(
        measurement=values.pop("measurement", MEASUREMENT_HOSTED),
        max_parallel_tests=values.pop("max_parallel_tests", DEFAULT_MAX_PARALLEL_TESTS),
        multibranch=doc.get(MULTIBRANCH_SECTION),
        name=values.pop("name", None),
        **values,
    )
