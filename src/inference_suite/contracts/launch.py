# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``launch.json`` (``inference-suite-launch/1``): the exact launch a round ran, with the base it was composed on.

Every flag group and environment variable names the layer that set it and the layers it overrode; ``removed`` names
what a layer took out. The final ``argv`` and ``env`` are derived from those records, so a document cannot disagree
with itself. :meth:`Launch.digest` covers the argv, env, client, eval and base digest; argv keeps its order.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom_common.contract_fields import ContractFields, field_names
from inference_suite.contracts.base_launch import (
    _ENV_NAME,
    DIGEST_PREFIX,
    BaseLaunch,
    EnvValue,
    GoldenAcceptance,
    Redacted,
)
from inference_suite.contracts.config import GoldenUse
from inference_suite.frameworks import Framework

LAUNCH_SCHEMA = "inference-suite-launch/1"
STAGING_MANIFEST_KEY = "staging_manifest_sha256"

_FLAG = re.compile(r"--?[A-Za-z0-9][\w.-]*")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class LaunchError(ValueError):
    """A launch or its document breaks the contract."""


_FIELDS = ContractFields(LaunchError)


class LayerSource(str, Enum):
    """The layers a launch is composed from, in the order they apply."""

    BASE = "base"
    USER = "user"
    CANDIDATE = "candidate"
    GOLDEN = "golden"
    #: The round's runtime facts: model path, address and GPU mask.
    GUARD = "guard"
    #: The runner's own layer (D12).
    RUNNER = "runner"
    #: The torch profiler of a profile round, last so the trace dir it names is the one the server writes to.
    PROFILE = "profile"


class VersionCheck(str, Enum):
    """How the installed framework was held to the recipe's version (D6); a failed check composes no launch."""

    PASSED = "passed"
    #: The base names no recipe image, so there is no version to reach.
    NOT_APPLICABLE = "not_applicable"


def _canonical(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sources(value: object, what: str) -> tuple[LayerSource, ...]:
    return tuple(_FIELDS.enum(LayerSource, item, what) for item in _FIELDS.items(value, what))


def _env_value_document(value: EnvValue) -> object:
    return value if isinstance(value, str) else value.to_document()


def _env_value(value: object, what: str) -> EnvValue:
    if isinstance(value, str):
        return value
    return Redacted(_FIELDS.known(value, ("sha256",), what).get("sha256"))


@dataclass(frozen=True)
class LaunchArg:
    """One flag group of the final argv: the flag as spelled, its values, the layer that set it and those it beat."""

    flag: str
    values: tuple[str, ...]
    source: LayerSource
    overrode: tuple[LayerSource, ...] = ()

    def __post_init__(self) -> None:
        _FIELDS.matching(_FLAG, self.flag, "a launch flag")
        _FIELDS.string_tuple(self.values, f"the values of {self.flag}")
        object.__setattr__(self, "source", _FIELDS.enum(LayerSource, self.source, "a flag's source"))
        object.__setattr__(self, "overrode", tuple(LayerSource(item) for item in self.overrode))

    def tokens(self) -> tuple[str, ...]:
        return (self.flag, *self.values)

    def to_document(self) -> dict[str, object]:
        return {
            "flag": self.flag,
            "values": list(self.values),
            "source": self.source.value,
            "overrode": [item.value for item in self.overrode],
        }

    @classmethod
    def from_document(cls, document: object) -> LaunchArg:
        fields = _FIELDS.known(document, field_names(cls), "a launch arg")
        return cls(
            flag=fields.get("flag"),
            values=_FIELDS.strings(fields.get("values", ()), "a launch arg's values"),
            source=fields.get("source"),
            overrode=_sources(fields.get("overrode", ()), "a launch arg's overridden sources"),
        )


@dataclass(frozen=True)
class LaunchEnv:
    """One variable of the final environment, with the layer that set it and those it beat."""

    name: str
    value: EnvValue
    source: LayerSource
    overrode: tuple[LayerSource, ...] = ()

    def __post_init__(self) -> None:
        _FIELDS.matching(_ENV_NAME, self.name, "a launch env name")
        if not isinstance(self.value, (str, Redacted)):
            raise LaunchError(f"the launch env {self.name} must be a string or redacted, got {self.value!r}")
        object.__setattr__(self, "source", _FIELDS.enum(LayerSource, self.source, "an env's source"))
        object.__setattr__(self, "overrode", tuple(LayerSource(item) for item in self.overrode))

    def to_document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": _env_value_document(self.value),
            "source": self.source.value,
            "overrode": [item.value for item in self.overrode],
        }

    @classmethod
    def from_document(cls, document: object) -> LaunchEnv:
        fields = _FIELDS.known(document, field_names(cls), "a launch env")
        return cls(
            name=fields.get("name"),
            value=_env_value(fields.get("value"), f"the launch env {fields.get('name')}"),
            source=fields.get("source"),
            overrode=_sources(fields.get("overrode", ()), "a launch env's overridden sources"),
        )


@dataclass(frozen=True)
class Removal:
    """A flag (``--name``) or environment variable a layer took out of the launch."""

    name: str
    by: LayerSource

    def __post_init__(self) -> None:
        if not (_FLAG.fullmatch(str(self.name)) or _ENV_NAME.fullmatch(str(self.name))):
            raise LaunchError(f"a removal names a flag or an environment variable, got {self.name!r}")
        object.__setattr__(self, "by", _FIELDS.enum(LayerSource, self.by, "a removal's layer"))

    def to_document(self) -> dict[str, object]:
        return {"name": self.name, "by": self.by.value}

    @classmethod
    def from_document(cls, document: object) -> Removal:
        fields = _FIELDS.known(document, field_names(cls), "a removal")
        return cls(fields.get("name"), fields.get("by"))


@dataclass(frozen=True)
class FrameworkCheck:
    """The framework the round ran: the interpreter, its installed version and the recipe version it reached (D6)."""

    name: Framework
    installed_version: str
    python: str
    recipe_min_version: str | None
    version_check: VersionCheck

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _FIELDS.enum(Framework, self.name, "the framework"))
        _FIELDS.text(self.installed_version, "the installed framework version")
        _FIELDS.text(self.python, "the framework python")
        _FIELDS.optional_text(self.recipe_min_version, "the recipe framework version")
        check = _FIELDS.enum(VersionCheck, self.version_check, "the version check")
        object.__setattr__(self, "version_check", check)
        if (check is VersionCheck.NOT_APPLICABLE) != (self.recipe_min_version is None):
            raise LaunchError("the version check is not_applicable exactly when there is no recipe version")

    def to_document(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "installed_version": self.installed_version,
            "python": self.python,
            "recipe_min_version": self.recipe_min_version,
            "version_check": self.version_check.value,
        }

    @classmethod
    def from_document(cls, document: object) -> FrameworkCheck:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "the framework check")
        return cls(*(fields.get(key) for key in keys))


@dataclass(frozen=True)
class ClientInvocation:
    """The load client as it ran, with both seeds explicit."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    seed: int
    warmup_seed: int

    def __post_init__(self) -> None:
        if not _FIELDS.string_tuple(self.argv, "the client argv"):
            raise LaunchError("the client argv must not be empty")
        for name in _FIELDS.string_map(self.env, "the client env"):
            _FIELDS.matching(_ENV_NAME, name, "a client env name")
        _FIELDS.integer(self.seed, "the client seed", minimum=0)
        _FIELDS.integer(self.warmup_seed, "the client warmup seed", minimum=0)

    def to_document(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "env": dict(sorted(self.env.items())),
            "seed": self.seed,
            "warmup_seed": self.warmup_seed,
        }

    @classmethod
    def from_document(cls, document: object) -> ClientInvocation:
        fields = _FIELDS.known(document, field_names(cls), "the client invocation")
        return cls(
            argv=_FIELDS.strings(fields.get("argv"), "the client argv"),
            env=fields.get("env", {}),
            seed=fields.get("seed"),
            warmup_seed=fields.get("warmup_seed"),
        )


@dataclass(frozen=True)
class EvalInvocation:
    """The eval process as it ran, and the digest of the definition its score belongs to."""

    argv: tuple[str, ...]
    definition_digest: str

    def __post_init__(self) -> None:
        if not _FIELDS.string_tuple(self.argv, "the eval argv"):
            raise LaunchError("the eval argv must not be empty")
        _FIELDS.matching(_DIGEST, self.definition_digest, "the eval definition digest")

    def to_document(self) -> dict[str, object]:
        return {"argv": list(self.argv), "definition_digest": self.definition_digest}

    @classmethod
    def from_document(cls, document: object) -> EvalInvocation:
        fields = _FIELDS.known(document, field_names(cls), "the eval invocation")
        return cls(_FIELDS.strings(fields.get("argv"), "the eval argv"), fields.get("definition_digest"))


@dataclass(frozen=True)
class ToolPins:
    """The staged tools the round ran (D11): each tool's pin, and the staging manifest they were verified against."""

    staging_manifest_sha256: str
    pins: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _FIELDS.sha256(self.staging_manifest_sha256, "the staging manifest sha256")
        for name in _FIELDS.string_map(self.pins, "the tool pins"):
            _FIELDS.text(name, "a tool name")
        if STAGING_MANIFEST_KEY in self.pins:
            raise LaunchError(f"{STAGING_MANIFEST_KEY} is not a tool name")

    def to_document(self) -> dict[str, object]:
        return {**dict(sorted(self.pins.items())), STAGING_MANIFEST_KEY: self.staging_manifest_sha256}

    @classmethod
    def from_document(cls, document: object) -> ToolPins:
        pins = dict(_FIELDS.mapping(document, "the tools"))
        return cls(pins.pop(STAGING_MANIFEST_KEY, None), pins)


#: Written for readers, and checked against what the records give on the way in.
_DERIVED_KEYS = ("base_digest", "argv", "env", "golden", "forced_acceptance", "digest")


@dataclass(frozen=True)
class Launch:
    """The launch one round ran.

    ``golden`` is the golden layer when the round applied it; such a launch forces acceptance, so its output is not the
    target model's own text.
    """

    base: BaseLaunch
    config_sha256: str
    golden_use: GoldenUse
    args: tuple[LaunchArg, ...]
    env_sources: tuple[LaunchEnv, ...]
    framework: FrameworkCheck
    tools: ToolPins
    removed: tuple[Removal, ...] = ()
    client: ClientInvocation | None = None
    eval: EvalInvocation | None = None
    source_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _FIELDS.sha256(self.config_sha256, "the config sha256")
        object.__setattr__(self, "golden_use", _FIELDS.enum(GoldenUse, self.golden_use, "the golden use"))
        if self.golden_use is GoldenUse.EXCLUDE and self.base.golden_al is None:
            raise LaunchError("golden exclude needs a base with a golden layer")
        _FIELDS.unique([entry.name for entry in self.env_sources], "launch env names")
        if self.framework.name.value != self.base.framework:
            raise LaunchError(f"the launch runs {self.framework.name.value}, its base {self.base.framework}")
        if self.framework.recipe_min_version != self.base.image.framework_version:
            raise LaunchError(
                f"the recipe version checked, {self.framework.recipe_min_version!r}, is not the base's "
                f"{self.base.image.framework_version!r}"
            )
        for path, sha256 in _FIELDS.string_map(self.source_files, "the source files").items():
            _FIELDS.sha256(sha256, f"the sha256 of source file {path}")

    @property
    def argv(self) -> tuple[str, ...]:
        return (*self.base.entrypoint, *(token for arg in self.args for token in arg.tokens()))

    @property
    def env(self) -> dict[str, EnvValue]:
        return {entry.name: entry.value for entry in self.env_sources}

    @property
    def golden(self) -> GoldenAcceptance | None:
        return self.base.golden_al if self.golden_use is GoldenUse.APPLY else None

    @property
    def forced_acceptance(self) -> bool:
        return self.golden is not None

    def digest(self) -> str:
        identity = {
            "argv": list(self.argv),
            "env": {name: _env_value_document(value) for name, value in sorted(self.env.items())},
            "client": None if self.client is None else self.client.to_document(),
            "eval": None if self.eval is None else self.eval.to_document(),
            "base_digest": self.base.digest(),
        }
        return DIGEST_PREFIX + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def to_document(self) -> dict[str, object]:
        golden = self.golden
        return {
            "schema": LAUNCH_SCHEMA,
            "base": self.base.to_document(),
            "base_digest": self.base.digest(),
            "config_sha256": self.config_sha256,
            "argv": list(self.argv),
            "env": {name: _env_value_document(value) for name, value in sorted(self.env.items())},
            "args": [arg.to_document() for arg in self.args],
            "env_sources": [entry.to_document() for entry in self.env_sources],
            "removed": [removal.to_document() for removal in self.removed],
            "golden_use": self.golden_use.value,
            "golden": None if golden is None else golden.to_document(),
            "forced_acceptance": self.forced_acceptance,
            "framework": self.framework.to_document(),
            "client": None if self.client is None else self.client.to_document(),
            "eval": None if self.eval is None else self.eval.to_document(),
            "tools": self.tools.to_document(),
            "source_files": dict(sorted(self.source_files.items())),
            "digest": self.digest(),
        }

    @classmethod
    def from_document(cls, document: object) -> Launch:
        """The launch a document records; one whose derived fields disagree with its records is refused."""

        fields = _FIELDS.known(document, ("schema", *field_names(cls), *_DERIVED_KEYS), "a launch")
        if fields.get("schema") != LAUNCH_SCHEMA:
            raise LaunchError(f"launch schema {fields.get('schema')!r} is not {LAUNCH_SCHEMA}")
        client, eval_ = fields.get("client"), fields.get("eval")
        launch = cls(
            base=BaseLaunch.from_document(fields.get("base")),
            config_sha256=fields.get("config_sha256"),
            golden_use=fields.get("golden_use"),
            args=tuple(LaunchArg.from_document(item) for item in _FIELDS.items(fields.get("args"), "the args")),
            env_sources=tuple(
                LaunchEnv.from_document(item) for item in _FIELDS.items(fields.get("env_sources"), "the env sources")
            ),
            removed=tuple(Removal.from_document(item) for item in _FIELDS.items(fields.get("removed", ()), "removed")),
            framework=FrameworkCheck.from_document(fields.get("framework")),
            client=None if client is None else ClientInvocation.from_document(client),
            eval=None if eval_ is None else EvalInvocation.from_document(eval_),
            tools=ToolPins.from_document(fields.get("tools")),
            source_files=fields.get("source_files", {}),
        )
        written = launch.to_document()
        mismatched = [key for key in _DERIVED_KEYS if fields.get(key) != written[key]]
        if mismatched:
            raise LaunchError(f"the launch document's {', '.join(mismatched)} disagree with its records")
        return launch


__all__ = [
    "LAUNCH_SCHEMA",
    "STAGING_MANIFEST_KEY",
    "ClientInvocation",
    "EvalInvocation",
    "FrameworkCheck",
    "Launch",
    "LaunchArg",
    "LaunchEnv",
    "LaunchError",
    "LayerSource",
    "Removal",
    "ToolPins",
    "VersionCheck",
]
