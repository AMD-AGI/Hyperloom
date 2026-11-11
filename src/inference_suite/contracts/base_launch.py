# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``base_launch.json`` (``hyperloom-base-launch/1``): the launch a session resolved once and every round builds on.

:meth:`BaseLaunch.digest` is the base's identity. It covers what decides how the server and the client run --
framework, entrypoint, the model's id, topology, argv, environment, the golden layer and the client -- and leaves out
where the launch came from, which image it was read from, and what the runner places on each host: the model's local
path and the runner-owned flags and environment. A scriptable base runs a script instead of a server and a client, so
its identity is the script's: the entrypoint that names it and the sha256 of its content. Argv keeps its order: of two aliases of one flag, or a switch
and its negation, the engine applies the last, so reordering may change the launch.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from hyperloom_common.contract_fields import ContractFields, field_names
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, normalize_argv
from inference_suite.frameworks import Framework, framework_facts
from inference_suite.golden import golden_owned_settings

BASE_LAUNCH_SCHEMA = "hyperloom-base-launch/1"
DIGEST_PREFIX = "sha256:"
DEFAULT_THINKING_MODE = "thinking_on"
#: The frameworks InferenceX and the cookbooks publish AgentX launches for.
AGENTX_FRAMEWORKS = frozenset({"sglang", "vllm"})

#: The field every launch layer keeps its argv in.
_ARGS = "args"
_RUNNER_OWNED_ENV = "the runner-owned env"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OPTION = re.compile(r"--[A-Za-z0-9][\w.-]*")
_VERSION = re.compile(r"\d+(?:\.\d+)+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?")


class BaseLaunchError(ValueError):
    """A base launch or its document breaks the contract."""


_FIELDS = ContractFields(BaseLaunchError)


class Workload(str, Enum):
    FIXED = "fixed"
    AGENTX = "agentx"
    #: A server-less script (xDiT, or an operator's own), which the base names as its entrypoint.
    SCRIPTABLE = "scriptable"


class SourceKind(str, Enum):
    USER_FILE = "user_file"
    INFERENCEX = "inferencex"
    SGLANG_COOKBOOK = "sglang_cookbook"
    VLLM_RECIPES = "vllm_recipes"
    ATOM_CATALOG = "atom_catalog"
    DEFAULT_TABLE = "default_table"


#: Sources that name no image, so there is no recipe framework version to hold the installed one to.
_IMAGELESS_SOURCES = frozenset({SourceKind.USER_FILE, SourceKind.DEFAULT_TABLE})


class VersionBasis(str, Enum):
    """Where the recipe's framework version came from."""

    TAG = "tag"
    #: The recipe's image tag names no version (a nightly, ``latest``, ``dev-*``), so the framework's latest
    #: release at resolve time stands in for it; the installed framework is not held to it.
    ASSUMED_LATEST = "assumed_latest"
    NOT_APPLICABLE = "not_applicable"


def _canonical(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _env_names(names: Sequence[str], what: str) -> None:
    for name in names:
        _FIELDS.matching(_ENV_NAME, name, f"an environment variable name in {what}")


def _argv(value: object, what: str) -> tuple[str, ...]:
    return normalize_argv(_FIELDS.string_tuple(value, what))


def _sorted_strings(value: Mapping[str, str]) -> dict[str, str]:
    return dict(sorted(value.items()))


@dataclass(frozen=True)
class LaunchQuery:
    """What a session asked for; ``tp`` and ``ep`` are set only when the operator gave them.

    A fixed workload names its sequence shape (``isl``, ``osl``); an AgentX workload replays a trace and a scriptable
    one runs a script, so neither names one.
    """

    model_id: str
    framework: str
    board: str
    workload: Workload
    concurrencies: tuple[int, ...]
    tp: int | None = None
    ep: int | None = None
    isl: int | None = None
    osl: int | None = None

    def __post_init__(self) -> None:
        _FIELDS.text(self.model_id, "the query's model id")
        _FIELDS.text(self.framework, "the query's framework")
        _FIELDS.text(self.board, "the query's board")
        object.__setattr__(self, "workload", _FIELDS.enum(Workload, self.workload, "the query's workload"))
        if not isinstance(self.concurrencies, tuple) or not self.concurrencies:
            raise BaseLaunchError(f"the query's concurrencies must be a non-empty tuple, got {self.concurrencies!r}")
        for conc in self.concurrencies:
            _FIELDS.integer(conc, "a query concurrency", minimum=1)
        _FIELDS.optional_integer(self.tp, "the query's tp", minimum=1)
        _FIELDS.optional_integer(self.ep, "the query's ep", minimum=1)
        _FIELDS.optional_integer(self.isl, "the query's isl", minimum=1)
        _FIELDS.optional_integer(self.osl, "the query's osl", minimum=1)
        fixed = self.workload is Workload.FIXED
        if (self.isl is not None, self.osl is not None) != (fixed, fixed):
            raise BaseLaunchError(
                f"a fixed query names its isl and osl and any other query neither, got {self.workload.value} "
                f"with isl {self.isl!r} and osl {self.osl!r}"
            )

    def to_document(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "framework": self.framework,
            "board": self.board,
            "workload": self.workload.value,
            "concurrencies": list(self.concurrencies),
            "tp": self.tp,
            "ep": self.ep,
            "isl": self.isl,
            "osl": self.osl,
        }

    @classmethod
    def from_document(cls, document: object) -> LaunchQuery:
        fields = _FIELDS.known(document, field_names(cls), "a launch query")
        return cls(
            model_id=fields.get("model_id"),
            framework=fields.get("framework"),
            board=fields.get("board"),
            workload=fields.get("workload"),
            concurrencies=_FIELDS.items(fields.get("concurrencies"), "the query's concurrencies"),
            tp=fields.get("tp"),
            ep=fields.get("ep"),
            isl=fields.get("isl"),
            osl=fields.get("osl"),
        )


@dataclass(frozen=True)
class SourceApi:
    """An HTTP response a source was read from, so the launch can be traced back to the bytes that gave it."""

    url: str
    response_sha256: str
    fetched_unix: int

    def __post_init__(self) -> None:
        _FIELDS.text(self.url, "a source API url")
        _FIELDS.sha256(self.response_sha256, "a source API response sha256")
        _FIELDS.integer(self.fetched_unix, "a source API fetch time", minimum=0)

    def to_document(self) -> dict[str, object]:
        return {"url": self.url, "response_sha256": self.response_sha256, "fetched_unix": self.fetched_unix}

    @classmethod
    def from_document(cls, document: object) -> SourceApi:
        fields = _FIELDS.known(document, field_names(cls), "a source API")
        return cls(fields.get("url"), fields.get("response_sha256"), fields.get("fetched_unix"))


@dataclass(frozen=True)
class LaunchSource:
    """Where the launch was resolved from: a repository at a commit, an API response, or a user's file.

    ``files`` maps each file the launch was read from to its sha256; ``tools`` names the pinned tools that turned the
    source into argv (InferenceX's srtctl, ...) by their commit.
    """

    kind: SourceKind
    repo: str | None = None
    commit: str | None = None
    api: SourceApi | None = None
    path: str | None = None
    entry: str | None = None
    locator: str | None = None
    strategy: str | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    tools: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _FIELDS.enum(SourceKind, self.kind, "the source kind"))
        if (self.repo is None) != (self.commit is None):
            raise BaseLaunchError("a source names its repository and commit together, or neither")
        _FIELDS.optional_text(self.repo, "the source repository")
        if self.commit is not None:
            _FIELDS.matching(_COMMIT, self.commit, "the source commit")
        for name in ("path", "entry", "locator", "strategy"):
            _FIELDS.optional_text(getattr(self, name), f"the source {name}")
        for path, sha256 in _FIELDS.string_map(self.files, "the source files").items():
            _FIELDS.sha256(sha256, f"the sha256 of source file {path}")
        for tool, commit in _FIELDS.string_map(self.tools, "the source tools").items():
            _FIELDS.matching(_COMMIT, commit, f"the commit of tool {tool}")

    def to_document(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "repo": self.repo,
            "commit": self.commit,
            "api": None if self.api is None else self.api.to_document(),
            "path": self.path,
            "entry": self.entry,
            "locator": self.locator,
            "strategy": self.strategy,
            "files": _sorted_strings(self.files),
            "tools": _sorted_strings(self.tools),
        }

    @classmethod
    def from_document(cls, document: object) -> LaunchSource:
        fields = _FIELDS.known(document, field_names(cls), "a launch source")
        api = fields.get("api")
        return cls(
            kind=fields.get("kind"),
            repo=fields.get("repo"),
            commit=fields.get("commit"),
            api=None if api is None else SourceApi.from_document(api),
            path=fields.get("path"),
            entry=fields.get("entry"),
            locator=fields.get("locator"),
            strategy=fields.get("strategy"),
            files=fields.get("files", {}),
            tools=fields.get("tools", {}),
        )


class UpstreamSource(str, Enum):
    INFERENCEX = "inferencex"
    SGLANG_COOKBOOK = "sglang_cookbook"
    VLLM_RECIPES = "vllm_recipes"
    ATOM_CATALOG = "atom_catalog"
    HF_MODEL = "hf_model"


@dataclass(frozen=True)
class UpstreamRevision:
    """The revision resolving read one upstream source at.

    A repository or a Hugging Face model is read at a commit. The vLLM recipes site publishes none and its renderings
    carry no ETag, so each response it gave is one revision: its URL and the sha256 of its body.
    """

    source: UpstreamSource
    location: str
    revision: str

    def __post_init__(self) -> None:
        source = _FIELDS.enum(UpstreamSource, self.source, "the upstream source")
        object.__setattr__(self, "source", source)
        _FIELDS.text(self.location, "the upstream location")
        if source is UpstreamSource.VLLM_RECIPES:
            _FIELDS.sha256(self.revision, "the sha256 of a vLLM recipes response")
        else:
            _FIELDS.matching(_COMMIT, self.revision, f"the {source.value} commit")

    def to_document(self) -> dict[str, object]:
        return {"source": self.source.value, "location": self.location, "revision": self.revision}

    @classmethod
    def from_document(cls, document: object) -> UpstreamRevision:
        fields = _FIELDS.known(document, field_names(cls), "an upstream revision")
        return cls(fields.get("source"), fields.get("location"), fields.get("revision"))


@dataclass(frozen=True)
class LaunchModel:
    """The checkpoint the base serves: its id, and the local path the runner serves it from on this host."""

    id: str
    path: str | None = None

    def __post_init__(self) -> None:
        _FIELDS.text(self.id, "the model id")
        _FIELDS.optional_text(self.path, "the model path")

    def to_document(self) -> dict[str, object]:
        return {"id": self.id, "path": self.path}

    @classmethod
    def from_document(cls, document: object) -> LaunchModel:
        fields = _FIELDS.known(document, field_names(cls), "the model")
        return cls(fields.get("id"), fields.get("path"))


@dataclass(frozen=True)
class Topology:
    tp: int
    ep: int = 1

    def __post_init__(self) -> None:
        _FIELDS.integer(self.tp, "the topology's tp", minimum=1)
        _FIELDS.integer(self.ep, "the topology's ep", minimum=1)

    def to_document(self) -> dict[str, object]:
        return {"tp": self.tp, "ep": self.ep}

    @classmethod
    def from_document(cls, document: object) -> Topology:
        fields = _FIELDS.known(document, field_names(cls), "a topology")
        return cls(fields.get("tp"), fields.get("ep"))


@dataclass(frozen=True)
class PointAlternative:
    """Another of the source's launches that also fits the query, at its own topology."""

    topology: Topology
    locator: str

    def __post_init__(self) -> None:
        _FIELDS.text(self.locator, "an alternative's locator")

    def to_document(self) -> dict[str, object]:
        return {"topology": self.topology.to_document(), "locator": self.locator}

    @classmethod
    def from_document(cls, document: object) -> PointAlternative:
        fields = _FIELDS.known(document, field_names(cls), "a point alternative")
        return cls(Topology.from_document(fields.get("topology")), fields.get("locator"))


@dataclass(frozen=True)
class LaunchPoint:
    """The source point the launch was taken from: its concurrency, and whether it is the one the query asked for."""

    conc: int
    exact: bool
    alternatives: tuple[PointAlternative, ...] = ()

    def __post_init__(self) -> None:
        _FIELDS.integer(self.conc, "the point's concurrency", minimum=1)
        _FIELDS.boolean(self.exact, "the point's exact")

    def to_document(self) -> dict[str, object]:
        return {
            "conc": self.conc,
            "exact": self.exact,
            "alternatives": [alternative.to_document() for alternative in self.alternatives],
        }

    @classmethod
    def from_document(cls, document: object) -> LaunchPoint:
        fields = _FIELDS.known(document, field_names(cls), "a launch point")
        alternatives = _FIELDS.items(fields.get("alternatives", ()), "the point's alternatives")
        return cls(fields.get("conc"), fields.get("exact"), tuple(map(PointAlternative.from_document, alternatives)))


@dataclass(frozen=True)
class ImageFacts:
    """The images on either side of the D6 check: ``framework_version`` is the recipe's, which the installed one
    must reach when it was read from the tag. The run image's own version is probed at launch, not recorded here."""

    run: str | None
    recipe: str | None
    framework_version: str | None
    version_basis: VersionBasis
    #: The release tag the version was read from when the recipe's tag names none.
    assumed_release: str | None = None

    def __post_init__(self) -> None:
        _FIELDS.optional_text(self.run, "the run image")
        _FIELDS.optional_text(self.recipe, "the recipe image")
        basis = _FIELDS.enum(VersionBasis, self.version_basis, "the image's version basis")
        object.__setattr__(self, "version_basis", basis)
        _FIELDS.optional_text(self.assumed_release, "the assumed release")
        if basis is VersionBasis.NOT_APPLICABLE:
            if self.framework_version is not None or self.assumed_release is not None:
                raise BaseLaunchError("a launch with no recipe image has no recipe framework version")
            return
        _FIELDS.matching(_VERSION, self.framework_version, "the recipe's framework version")
        if basis is VersionBasis.TAG and (self.recipe is None or self.assumed_release is not None):
            raise BaseLaunchError("a version read from the tag needs the recipe image and no assumed release")
        if basis is VersionBasis.ASSUMED_LATEST and self.assumed_release is None:
            raise BaseLaunchError("an assumed-latest version names the release it was read from")

    def to_document(self) -> dict[str, object]:
        return {
            "run": self.run,
            "recipe": self.recipe,
            "framework_version": self.framework_version,
            "version_basis": self.version_basis.value,
            "assumed_release": self.assumed_release,
        }

    @classmethod
    def from_document(cls, document: object) -> ImageFacts:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "the image facts")
        return cls(*(fields.get(key) for key in keys))


@dataclass(frozen=True)
class Redacted:
    """An environment value kept out of the record; its sha256 still enters the digest."""

    sha256: str

    def __post_init__(self) -> None:
        _FIELDS.sha256(self.sha256, "a redacted value's sha256")

    def to_document(self) -> dict[str, object]:
        return {"sha256": self.sha256}


EnvValue = str | Redacted


class CredentialNotHeld(ValueError):
    """A redacted value the host holds no value of, or another value; the message names it, never a value."""


def held_env(env: Mapping[str, EnvValue], host: Mapping[str, str]) -> dict[str, str]:
    """``env`` with each redacted value taken from ``host``, which must hold the value the base was resolved with.

    Raises:
        CredentialNotHeld: ``host`` holds no value of a redacted name, or another value than the resolved one.
    """
    held: dict[str, str] = {}
    for name, value in env.items():
        if isinstance(value, str):
            held[name] = value
            continue
        host_value = host.get(name)
        if host_value is None:
            raise CredentialNotHeld(f"{name} is redacted in the base and the host holds no value of it")
        if hashlib.sha256(host_value.encode()).hexdigest() != value.sha256:
            raise CredentialNotHeld(
                f"{name} is redacted in the base and the host's value is not the one the base was resolved with"
            )
        held[name] = host_value
    return held


def _env(value: object, what: str) -> dict[str, EnvValue]:
    env = dict(_FIELDS.mapping(value, what))
    _env_names(list(env), what)
    for name, item in env.items():
        if not isinstance(item, (str, Redacted)):
            raise BaseLaunchError(f"{what}: {name} must be a string or redacted, got {item!r}")
    return env


def _env_document(env: Mapping[str, EnvValue]) -> dict[str, object]:
    return {name: item if isinstance(item, str) else item.to_document() for name, item in sorted(env.items())}


def _env_from_document(value: object, what: str) -> dict[str, EnvValue]:
    env: dict[str, EnvValue] = {}
    for name, item in _FIELDS.mapping(value, what).items():
        if isinstance(item, str):
            env[name] = item
        else:
            env[name] = Redacted(_FIELDS.known(item, ("sha256",), f"{what}: {name}").get("sha256"))
    return env


@dataclass(frozen=True)
class GoldenCurve:
    """The acceptance-length table a golden value was read from."""

    repo: str
    commit: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _FIELDS.text(self.repo, "the golden curve's repository")
        _FIELDS.matching(_COMMIT, self.commit, "the golden curve's commit")
        _FIELDS.text(self.path, "the golden curve's path")
        _FIELDS.sha256(self.sha256, "the golden curve's sha256")

    def to_document(self) -> dict[str, object]:
        return {"repo": self.repo, "commit": self.commit, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_document(cls, document: object) -> GoldenCurve:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "a golden curve")
        return cls(*(fields.get(key) for key in keys))


@dataclass(frozen=True)
class GoldenAcceptance:
    """The golden layer: a simulated acceptance length and the draft it holds for, applied above every other layer.

    ``args`` and ``env`` are what the layer adds to the launch; the layers below may not set its keys.
    """

    value: float
    method: str
    token_mode: str
    spec_method: str
    draft_length: int
    curve: GoldenCurve
    draft_model: str | None = None
    thinking_mode: str = DEFAULT_THINKING_MODE
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _FIELDS.number(self.value, "the golden acceptance length") <= 0:
            raise BaseLaunchError(f"the golden acceptance length must be positive, got {self.value!r}")
        for name in ("method", "token_mode", "spec_method", "thinking_mode"):
            _FIELDS.text(getattr(self, name), f"the golden {name}")
        _FIELDS.optional_text(self.draft_model, "the golden draft model")
        _FIELDS.integer(self.draft_length, "the golden draft length", minimum=1)
        object.__setattr__(self, _ARGS, _argv(self.args, "the golden args"))
        _env_names(list(_FIELDS.string_map(self.env, "the golden env")), "the golden env")

    def to_document(self) -> dict[str, object]:
        return {
            "value": self.value,
            "method": self.method,
            "token_mode": self.token_mode,
            "thinking_mode": self.thinking_mode,
            "spec_method": self.spec_method,
            "draft_model": self.draft_model,
            "draft_length": self.draft_length,
            "curve": self.curve.to_document(),
            "args": list(self.args),
            "env": _sorted_strings(self.env),
        }

    @classmethod
    def from_document(cls, document: object) -> GoldenAcceptance:
        fields = _FIELDS.known(document, field_names(cls), "a golden acceptance")
        return cls(
            value=fields.get("value"),
            method=fields.get("method"),
            token_mode=fields.get("token_mode"),
            thinking_mode=fields.get("thinking_mode", DEFAULT_THINKING_MODE),
            spec_method=fields.get("spec_method"),
            draft_model=fields.get("draft_model"),
            draft_length=fields.get("draft_length"),
            curve=GoldenCurve.from_document(fields.get("curve")),
            args=_FIELDS.strings(fields.get("args", ()), "the golden args"),
            env=fields.get("env", {}),
        )


@dataclass(frozen=True)
class ClientLaunch:
    """The load generator the base is measured with, pinned by commit.

    :attr:`command` renders it as one shell line, the form Arbor runs a client from.
    """

    tool: str
    commit: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _FIELDS.text(self.tool, "the client tool")
        _FIELDS.matching(_COMMIT, self.commit, "the client commit")
        object.__setattr__(self, _ARGS, _argv(self.args, "the client args"))
        _env_names(list(_FIELDS.string_map(self.env, "the client env")), "the client env")

    @property
    def command(self) -> str:
        return shlex.join((self.tool, *self.args))

    def identity(self) -> dict[str, object]:
        return {"tool": self.tool, "commit": self.commit, "args": list(self.args), "env": _sorted_strings(self.env)}

    def to_document(self) -> dict[str, object]:
        return {**self.identity(), "command": self.command}

    @classmethod
    def from_document(cls, document: object) -> ClientLaunch:
        """The client a document records; a ``command`` other than the one its tool and args render is refused."""
        fields = _FIELDS.known(document, (*field_names(cls), "command"), "a client launch")
        client = cls(
            tool=fields.get("tool"),
            commit=fields.get("commit"),
            args=_FIELDS.strings(fields.get("args", ()), "the client args"),
            env=fields.get("env", {}),
        )
        command = fields.get("command")
        if command != client.command:
            raise BaseLaunchError(f"the client command {command!r} is not what its tool and args run: {client.command}")
        return client


@dataclass(frozen=True)
class RunnerOwned:
    """The flags and environment names the runner sets per round (host, port, GPU masks, ...), so no base may."""

    args: tuple[str, ...] = ()
    env: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in _FIELDS.string_tuple(self.args, "the runner-owned args"):
            _FIELDS.matching(_OPTION, name, "a runner-owned flag")
        _env_names(_FIELDS.string_tuple(self.env, _RUNNER_OWNED_ENV), _RUNNER_OWNED_ENV)

    def flags(self) -> frozenset[str]:
        """The owned flags as :func:`~inference_suite.argv.flag_occurrences` names them."""
        return frozenset(map(canonical_flag_name, self.args))

    def to_document(self) -> dict[str, object]:
        return {"args": list(self.args), "env": list(self.env)}

    @classmethod
    def from_document(cls, document: object) -> RunnerOwned:
        fields = _FIELDS.known(document, field_names(cls), "the runner-owned keys")
        return cls(
            _FIELDS.strings(fields.get("args", ()), "the runner-owned args"),
            _FIELDS.strings(fields.get("env", ()), _RUNNER_OWNED_ENV),
        )


_DIGEST_KEYS = ("schema", "framework", "entrypoint", "model", "topology", "args", "env", "golden_al", "client")


@dataclass(frozen=True)
class BaseLaunch:
    """The frozen base launch of a session.

    ``args`` and ``env`` hold neither runner-owned keys nor, when there is a golden layer, golden-owned ones.
    ``stripped`` names what was taken out of the source because the runner owns it; ``warnings`` are shown at launch.
    ``upstream`` is the revision of every upstream source resolving read, the one the launch came from or not; a
    document written before it was recorded reads with none.

    A scriptable base (``query.workload`` scriptable) runs its entrypoint's last word, :attr:`script`, and nothing
    else: no client, args, env or golden layer, and ``source.files`` records the script's sha256.
    """

    query: LaunchQuery
    source: LaunchSource
    point: LaunchPoint
    image: ImageFacts
    framework: str
    entrypoint: tuple[str, ...]
    model: LaunchModel
    topology: Topology
    args: tuple[str, ...]
    env: Mapping[str, EnvValue]
    client: ClientLaunch | None
    resolved_unix: int
    golden_al: GoldenAcceptance | None = None
    runner_owned: RunnerOwned = field(default_factory=RunnerOwned)
    stripped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    upstream: tuple[UpstreamRevision, ...] = ()

    def __post_init__(self) -> None:
        _FIELDS.text(self.framework, "the framework")
        if not _FIELDS.string_tuple(self.entrypoint, "the entrypoint") or not all(self.entrypoint):
            raise BaseLaunchError(f"the entrypoint must be non-empty words, got {self.entrypoint!r}")
        object.__setattr__(self, _ARGS, _argv(self.args, "the args"))
        object.__setattr__(self, "env", _env(self.env, "the env"))
        _FIELDS.string_tuple(self.stripped, "the stripped flags")
        _FIELDS.string_tuple(self.warnings, "the warnings")
        _FIELDS.integer(self.resolved_unix, "the resolve time", minimum=0)
        self._check_source()
        self._check_layers()
        self._check_script()

    @property
    def cards(self) -> int:
        """The GPUs its launch occupies (:meth:`FrameworkFacts.cards`)."""
        return framework_facts(Framework(self.framework)).cards(self.topology.tp, self.args)

    @property
    def script(self) -> str:
        """The script a scriptable base runs: its entrypoint's last word."""
        return self.entrypoint[-1]

    @property
    def scriptable(self) -> bool:
        return self.query.workload is Workload.SCRIPTABLE

    def _check_script(self) -> None:
        if (self.client is None) != self.scriptable:
            raise BaseLaunchError(
                f"a scriptable base runs no client and any other base runs one, got {self.query.workload.value} "
                f"with client {self.client!r}"
            )
        if not self.scriptable:
            return
        if self.args or self.env or self.golden_al is not None:
            raise BaseLaunchError("a scriptable base runs its script alone: no args, env or golden layer")
        if self.source.kind not in _IMAGELESS_SOURCES:
            raise BaseLaunchError(f"a scriptable base is a user file or a shipped script, got {self.source.kind.value}")
        if self.script not in self.source.files:
            raise BaseLaunchError(f"a scriptable base records the sha256 of its script {self.script}")

    def _check_source(self) -> None:
        if (self.source.kind in _IMAGELESS_SOURCES) != (self.image.version_basis is VersionBasis.NOT_APPLICABLE):
            raise BaseLaunchError(
                f"a {self.source.kind.value} source has version basis not_applicable exactly when it names no image, "
                f"got {self.image.version_basis.value}"
            )
        for name in ("tp", "ep"):
            asked = getattr(self.query, name)
            if asked is not None and asked != getattr(self.topology, name):
                raise BaseLaunchError(
                    f"the query asks for {name} {asked}, the launch runs {getattr(self.topology, name)}"
                )
        if self.query.workload is Workload.AGENTX:
            if self.source.kind is SourceKind.DEFAULT_TABLE:
                raise BaseLaunchError("an AgentX base needs a recipe or a user file, not the default table")
            if self.framework not in AGENTX_FRAMEWORKS:
                raise BaseLaunchError(f"an AgentX base runs {sorted(AGENTX_FRAMEWORKS)}, got {self.framework!r}")

    def _check_layers(self) -> None:
        layers = [("the args and env", self.args, self.env.keys())]
        if self.golden_al is not None:
            layers.append(("the golden layer", self.golden_al.args, self.golden_al.env.keys()))
            golden = sorted(golden_owned_settings(argv_flag_occurrences(self.args), self.env))
            if golden:
                raise BaseLaunchError(
                    f"the args and env set golden-owned {', '.join(golden)}; the golden layer owns them"
                )
        owned_flags = self.runner_owned.flags()
        for what, args, env in layers:
            clash = sorted({f"--{name}" for name, _ in argv_flag_occurrences(args) if name in owned_flags})
            clash += sorted(set(env) & set(self.runner_owned.env))
            if clash:
                raise BaseLaunchError(f"{what} set runner-owned {', '.join(clash)}")

    def _identity(self) -> dict[str, object]:
        identity = {key: value for key, value in self._body().items() if key in _DIGEST_KEYS}
        identity["model"] = {"id": self.model.id}
        if self.client is None:
            return {**identity, "script_sha256": self.source.files[self.script]}
        return {**identity, "client": self.client.identity()}

    def digest(self) -> str:
        return DIGEST_PREFIX + hashlib.sha256(_canonical(self._identity()).encode()).hexdigest()

    def _body(self) -> dict[str, object]:
        return {
            "schema": BASE_LAUNCH_SCHEMA,
            "query": self.query.to_document(),
            "source": self.source.to_document(),
            "point": self.point.to_document(),
            "image": self.image.to_document(),
            "framework": self.framework,
            "entrypoint": list(self.entrypoint),
            "model": self.model.to_document(),
            "topology": self.topology.to_document(),
            "args": list(self.args),
            "env": _env_document(self.env),
            "golden_al": None if self.golden_al is None else self.golden_al.to_document(),
            "client": None if self.client is None else self.client.to_document(),
            "runner_owned": self.runner_owned.to_document(),
            "stripped": list(self.stripped),
            "warnings": list(self.warnings),
            "resolved_unix": self.resolved_unix,
            "upstream": [revision.to_document() for revision in self.upstream],
        }

    def to_document(self) -> dict[str, object]:
        return {**self._body(), "digest": self.digest()}

    @classmethod
    def from_document(cls, document: object) -> BaseLaunch:
        """The base a document records; a document whose digest does not match its content is refused."""

        fields = _FIELDS.known(document, ("schema", *field_names(cls), "digest"), "a base launch")
        if fields.get("schema") != BASE_LAUNCH_SCHEMA:
            raise BaseLaunchError(f"base launch schema {fields.get('schema')!r} is not {BASE_LAUNCH_SCHEMA}")
        golden = fields.get("golden_al")
        client = fields.get("client")
        base = cls(
            query=LaunchQuery.from_document(fields.get("query")),
            source=LaunchSource.from_document(fields.get("source")),
            point=LaunchPoint.from_document(fields.get("point")),
            image=ImageFacts.from_document(fields.get("image")),
            framework=fields.get("framework"),
            entrypoint=_FIELDS.strings(fields.get("entrypoint"), "the entrypoint"),
            model=LaunchModel.from_document(fields.get("model")),
            topology=Topology.from_document(fields.get("topology")),
            args=_FIELDS.strings(fields.get("args"), "the args"),
            env=_env_from_document(fields.get("env"), "the env"),
            client=None if client is None else ClientLaunch.from_document(client),
            resolved_unix=fields.get("resolved_unix"),
            golden_al=None if golden is None else GoldenAcceptance.from_document(golden),
            runner_owned=RunnerOwned.from_document(fields.get("runner_owned", {})),
            stripped=_FIELDS.strings(fields.get("stripped", ()), "the stripped flags"),
            warnings=_FIELDS.strings(fields.get("warnings", ()), "the warnings"),
            upstream=tuple(
                map(UpstreamRevision.from_document, _FIELDS.items(fields.get("upstream", ()), "the upstream revisions"))
            ),
        )
        recorded = fields.get("digest")
        if recorded != base.digest():
            raise BaseLaunchError(
                f"base_digest_mismatch: the document records {recorded!r}, its content gives {base.digest()}"
            )
        return base


__all__ = [
    "AGENTX_FRAMEWORKS",
    "BASE_LAUNCH_SCHEMA",
    "DEFAULT_THINKING_MODE",
    "DIGEST_PREFIX",
    "BaseLaunch",
    "BaseLaunchError",
    "ClientLaunch",
    "CredentialNotHeld",
    "EnvValue",
    "GoldenAcceptance",
    "GoldenCurve",
    "ImageFacts",
    "LaunchPoint",
    "LaunchModel",
    "LaunchQuery",
    "LaunchSource",
    "PointAlternative",
    "Redacted",
    "RunnerOwned",
    "SourceApi",
    "SourceKind",
    "Topology",
    "UpstreamRevision",
    "UpstreamSource",
    "VersionBasis",
    "Workload",
    "held_env",
]
