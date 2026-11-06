# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve a session's base launch from a snapshot of its sources, as a pure function (D4, D5).

The sources are read in priority order -- the operator's launch file, InferenceX, the framework's cookbook (the
sglang cookbook, the vLLM recipes site or ATOM's catalog), then, for a fixed workload only, the default table -- and
the first that holds a launch for the query gives the base. An AgentX query no recipe serves is a :class:`Miss`.
Whatever needed the network or upstream code was done when the snapshot was taken; resolving only reads it. A
snapshot that names a source it could not read consults no recipe: a fixed query runs the default table, and an
AgentX query is a :class:`Miss`.

A source may hold several launches for one query, one per arm. The base runs the one with the largest TP the GPUs
fit, an arm that offloads no KV cache before one that does, and records the others as alternatives. A cookbook
AgentX launch takes its golden acceptance from InferenceX's curves here. A golden acceptance becomes the base's
golden layer, which takes the draft and simulation settings out of the launch's own args and env.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from hyperloom_common.contract_fields import ContractFields
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from inference_suite.argv import (
    SPECULATIVE_CONFIG_FLAG,
    SPECULATIVE_CONFIG_KEY_PREFIX,
    argv_flag_occurrences,
    normalize_argv,
)
from inference_suite.contracts.base_launch import (
    BaseLaunch,
    ClientLaunch,
    GoldenAcceptance,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    PointAlternative,
    RunnerOwned,
    SourceKind,
    Topology,
    Workload,
)
from inference_suite.golden import GOLDEN_OWNED_ENV_PREFIXES, GOLDEN_OWNED_SPEC_CONFIG_KEYS, is_golden_owned_flag
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT
from inference_suite.recipe.atom_catalog import atom_catalog_launches
from inference_suite.recipe.defaults import TABLE_PATH, default_table
from inference_suite.recipe.golden_curve import (
    _SGLANG_SIMULATION,
    _SIMULATION_TOKEN_MODE,
    GoldenLength,
    GoldenTable,
    _sglang_draft,
    _vllm_draft,
    with_golden,
)
from inference_suite.recipe.image_version import NO_RECIPE_IMAGE, recipe_image_facts
from inference_suite.recipe.inferencex import inferencex_launches
from inference_suite.recipe.model import (
    ATOM_ENTRYPOINT,
    ATOM_RUNNER_OWNED,
    ATOM_TOPOLOGY,
    SGLANG_ENTRYPOINT,
    SGLANG_RUNNER_OWNED,
    SGLANG_TOPOLOGY,
    VLLM_ENTRYPOINT,
    VLLM_RUNNER_OWNED,
    VLLM_TOPOLOGY,
    InferencexSnapshot,
    RecipeSourceError,
    RepoBundle,
    SourceGolden,
    SourceLaunch,
    SourceMiss,
)
from inference_suite.recipe.sglang_cookbook import sglang_cookbook_launches
from inference_suite.recipe.user_file import UserFile
from inference_suite.recipe.vllm_recipes import VllmRecipesSnapshot, vllm_recipes_launches

#: The load client every base is measured with: InferenceX's own harness, at its pinned commit.
CLIENT_TOOL = "inferencex"

_FIELDS = ContractFields(RecipeSourceError)
_T = TypeVar("_T")

_ENTRYPOINTS = {"sglang": SGLANG_ENTRYPOINT, "vllm": VLLM_ENTRYPOINT, "atom": ATOM_ENTRYPOINT}
_RUNNER_OWNED = {"sglang": SGLANG_RUNNER_OWNED, "vllm": VLLM_RUNNER_OWNED, "atom": ATOM_RUNNER_OWNED}
#: The GPU masks the runner sets for each server it starts; a source's own are stripped.
_RUNNER_OWNED_ENV = frozenset({"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"})
_TOPOLOGY = {"sglang": SGLANG_TOPOLOGY, "vllm": VLLM_TOPOLOGY, "atom": ATOM_TOPOLOGY}
_NO_KV_OFFLOADING = "none"
_SGLANG_METHOD, _SGLANG_TOKEN_MODE = _SGLANG_SIMULATION[1:]


@dataclass(frozen=True)
class UnreadSource:
    """A source the snapshot could not read, and why."""

    source: str
    reason: str

    def __post_init__(self) -> None:
        _FIELDS.text(self.source, "the unread source")
        _FIELDS.text(self.reason, "why the source could not be read")


@dataclass(frozen=True)
class SourceSnapshot:
    """Everything resolving reads, recorded once when the session starts.

    ``inferencex`` is read for every query a recipe may serve, and keys the golden curves; it is ``None`` when the
    operator's ``user_file`` is the base or InferenceX could not be read. A cookbook is carried for the query's framework only. ``golden_lengths`` are InferenceX's ``golden_length`` answers
    for the cookbook's AgentX launches (see :func:`~inference_suite.recipe.golden_curve.golden_requests`) and
    ``base_model`` the model's HF ``cardData.base_model``. ``latest_releases`` maps a framework to its latest release
    tag, which stands for a recipe image that names no version. ``user_file``, when the operator gave one, is the
    base. ``fetched_unix`` is when the snapshot was taken, and so when the base was resolved. ``unread`` names the
    source reading stopped at, when one could not be read; what it and every later source hold is missing.
    """

    inferencex: InferencexSnapshot | None
    fetched_unix: int
    user_file: UserFile | None = None
    sglang_cookbook: RepoBundle | None = None
    vllm_recipes: VllmRecipesSnapshot | None = None
    atom_catalog: RepoBundle | None = None
    golden_lengths: Mapping[str, GoldenLength] = field(default_factory=dict)
    base_model: str | None = None
    latest_releases: Mapping[str, str] = field(default_factory=dict)
    unread: UnreadSource | None = None

    def __post_init__(self) -> None:
        _FIELDS.integer(self.fetched_unix, "the snapshot's fetch time", minimum=0)
        _FIELDS.optional_text(self.base_model, "the base model")
        _FIELDS.string_map(self.latest_releases, "the latest releases")


@dataclass(frozen=True)
class Miss:
    """No source holds a launch for the query: the source that could not be read, if one could not, and why each one
    read holds none, in the order they were read."""

    misses: tuple[SourceMiss, ...]
    unread: UnreadSource | None = None

    def reasons(self) -> tuple[str, ...]:
        unread = () if self.unread is None else (f"{self.unread.source} could not be read: {self.unread.reason}",)
        return (*unread, *(f"{miss.kind.value}: {miss.reason}" for miss in self.misses))


@dataclass(frozen=True)
class _Choice:
    launch: SourceLaunch
    others: tuple[SourceLaunch, ...]


@dataclass(frozen=True)
class _Layers:
    """A launch's args and env split into what the base keeps and the golden layer that owns the rest."""

    args: tuple[str, ...]
    env: Mapping[str, str]
    golden: GoldenAcceptance | None


def _carried(value: _T | None, what: str) -> _T:
    if value is None:
        raise RecipeSourceError(f"the snapshot carries no {what}")
    return value


_Reader = Callable[[LaunchQuery, SourceSnapshot], "tuple[SourceLaunch, ...] | SourceMiss"]


def _inferencex(query: LaunchQuery, snapshot: SourceSnapshot) -> tuple[SourceLaunch, ...] | SourceMiss:
    return inferencex_launches(query, _carried(snapshot.inferencex, "InferenceX"))


def _sglang_cookbook(query: LaunchQuery, snapshot: SourceSnapshot) -> tuple[SourceLaunch, ...] | SourceMiss:
    return sglang_cookbook_launches(query, _carried(snapshot.sglang_cookbook, "sglang cookbook"))


def _vllm_recipes(query: LaunchQuery, snapshot: SourceSnapshot) -> tuple[SourceLaunch, ...] | SourceMiss:
    return vllm_recipes_launches(query, _carried(snapshot.vllm_recipes, "vLLM recipes"))


def _atom_catalog(query: LaunchQuery, snapshot: SourceSnapshot) -> tuple[SourceLaunch, ...] | SourceMiss:
    return atom_catalog_launches(query, _carried(snapshot.atom_catalog, "ATOM catalog"))


_COOKBOOKS: Mapping[str, _Reader] = {"sglang": _sglang_cookbook, "vllm": _vllm_recipes, "atom": _atom_catalog}


def _readers(framework: str) -> tuple[_Reader, ...]:
    cookbook = _COOKBOOKS.get(framework)
    return (_inferencex,) if cookbook is None else (_inferencex, cookbook)


def _where(source: LaunchSource) -> str:
    return source.locator or source.path or source.entry or source.kind.value


def _preference(launch: SourceLaunch) -> tuple[int, bool, bool]:
    return launch.topology.tp, launch.kv_offloading == _NO_KV_OFFLOADING, launch.exact


def _choose(launches: Sequence[SourceLaunch], gpus: int) -> _Choice | SourceMiss:
    """The launch to run of those one source holds, and the others that also fit ``gpus`` GPUs."""
    fitting = [launch for launch in launches if launch.topology.tp <= gpus]
    if not fitting:
        sizes = sorted({launch.topology.tp for launch in launches})
        return SourceMiss(launches[0].source.kind, f"its launches run tp {sizes}, more than the {gpus} GPUs here")
    ranked = sorted(fitting, key=_preference, reverse=True)
    return _Choice(ranked[0], tuple(ranked[1:]))


def _flag_name(token: str) -> str | None:
    named = argv_flag_occurrences((token,))
    return named[0][0] if named else None


def _golden_owned(name: str) -> bool:
    if name.startswith(SPECULATIVE_CONFIG_KEY_PREFIX):
        return name.removeprefix(SPECULATIVE_CONFIG_KEY_PREFIX).replace("-", "_") in GOLDEN_OWNED_SPEC_CONFIG_KEYS
    return name == SPECULATIVE_CONFIG_FLAG or is_golden_owned_flag(name)


def _split_args(args: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``args`` as the flags the base keeps and the golden-owned ones, each flag with its values, order kept.

    vLLM's whole ``--speculative-config`` object goes to the golden layer: it names the draft, so it holds keys the
    golden layer owns.
    """
    kept: list[str] = []
    owned: list[str] = []
    into = kept
    for token in normalize_argv(args):
        name = _flag_name(token)
        if name is not None:
            into = owned if _golden_owned(name) else kept
        into.append(token)
    return tuple(kept), tuple(owned)


def _golden_acceptance(
    launch: SourceLaunch, golden: SourceGolden, args: tuple[str, ...], env: Mapping[str, str]
) -> GoldenAcceptance:
    occurrences = argv_flag_occurrences(args)
    if launch.framework == "sglang":
        spec, _ = _sglang_draft(occurrences)
        method, token_mode = env.get(_SGLANG_METHOD), env.get(_SGLANG_TOKEN_MODE)
    elif launch.framework == "vllm":
        spec, _ = _vllm_draft(occurrences)
        # vLLM's synthetic sampler streams the drafted tokens it accepts, as sglang's real-draft-token mode does.
        method, token_mode = spec.get("rejection_sample_method"), _SIMULATION_TOKEN_MODE
    else:
        raise RecipeSourceError(f"a {launch.framework} launch simulates no golden acceptance")
    return GoldenAcceptance(
        value=golden.value,
        method=method,
        token_mode=token_mode,
        spec_method=spec.get("method"),
        draft_length=spec.get("num_speculative_tokens"),
        curve=golden.curve,
        draft_model=spec.get("model") or None,
        args=args,
        env=env,
    )


def _layers(launch: SourceLaunch) -> _Layers:
    if launch.golden is None or launch.golden.value is None:
        return _Layers(launch.args, launch.env, None)
    args, golden_args = _split_args(launch.args)
    owned = {name for name in launch.env if name.startswith(GOLDEN_OWNED_ENV_PREFIXES)}
    golden_env = {name: launch.env[name] for name in sorted(owned)}
    env = {name: value for name, value in launch.env.items() if name not in owned}
    return _Layers(args, env, _golden_acceptance(launch, launch.golden, golden_args, golden_env))


def _miss_warnings(misses: Sequence[SourceMiss]) -> tuple[str, ...]:
    return tuple(f"{miss.kind.value} holds no launch: {miss.reason}" for miss in misses)


def _alternatives_warning(choice: _Choice) -> tuple[str, ...]:
    if not choice.others:
        return ()
    others = ", ".join(
        f"tp {other.topology.tp} ep {other.topology.ep} ({_where(other.source)})" for other in choice.others
    )
    taken = choice.launch
    return (
        f"{taken.source.kind.value} holds other launches that fit: {others}; took tp {taken.topology.tp} "
        f"ep {taken.topology.ep} ({_where(taken.source)})",
    )


def _golden_warning(launch: SourceLaunch) -> tuple[str, ...]:
    if launch.golden is None or launch.golden.value is not None:
        return ()
    return (f"the launch runs with no golden acceptance: {launch.golden.reason}",)


def _runner_owned(framework: str) -> RunnerOwned:
    return RunnerOwned(
        args=tuple(f"--{name}" for name in sorted(_RUNNER_OWNED.get(framework, ()))),
        env=tuple(sorted(_RUNNER_OWNED_ENV)),
    )


def _runner_env(env: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(_RUNNER_OWNED_ENV.intersection(env)))


def _without_runner_env(env: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in env.items() if name not in _RUNNER_OWNED_ENV}


def _entrypoint(framework: str) -> tuple[str, ...]:
    return _carried(_ENTRYPOINTS.get(framework), f"{framework} serving entrypoint")


def _client(env: Mapping[str, str]) -> ClientLaunch:
    return ClientLaunch(tool=CLIENT_TOOL, commit=INFERENCEX_CLIENT_COMMIT, env=env)


def _recipe_base(
    query: LaunchQuery, snapshot: SourceSnapshot, choice: _Choice, misses: Sequence[SourceMiss]
) -> BaseLaunch:
    launch = choice.launch
    layers = _layers(launch)
    alternatives = tuple(PointAlternative(other.topology, _where(other.source)) for other in choice.others)
    return BaseLaunch(
        query=query,
        source=launch.source,
        point=LaunchPoint(launch.conc, launch.exact, alternatives),
        image=recipe_image_facts(launch.image, snapshot.latest_releases.get(launch.framework)),
        framework=launch.framework,
        entrypoint=launch.entrypoint,
        model=LaunchModel(query.model_id),
        topology=launch.topology,
        args=layers.args,
        env=_without_runner_env(layers.env),
        client=_client(launch.client_env),
        resolved_unix=snapshot.fetched_unix,
        golden_al=layers.golden,
        runner_owned=_runner_owned(launch.framework),
        stripped=(*launch.stripped, *_runner_env(layers.env)),
        warnings=(
            *_miss_warnings(misses),
            *launch.warnings,
            *_alternatives_warning(choice),
            *_golden_warning(launch),
        ),
    )


def _user_file_base(query: LaunchQuery, snapshot: SourceSnapshot, user_file: UserFile) -> BaseLaunch:
    """The operator's file as the base: a frozen base as it was resolved, a shell launch as it is written."""
    if isinstance(user_file.launch, BaseLaunch):
        return dataclasses.replace(user_file.launch, query=query)
    shell = user_file.launch
    spelling = _carried(_TOPOLOGY.get(query.framework), f"{query.framework} topology spellings")
    unresolved = ()
    if shell.unresolved_env:
        unresolved = (
            f"{user_file.location} exports {', '.join(shell.unresolved_env)} from shell variables; "
            "the base carries none of them",
        )
    return BaseLaunch(
        query=query,
        source=LaunchSource(
            kind=SourceKind.USER_FILE, path=user_file.location, files={user_file.location: user_file.sha256}
        ),
        point=LaunchPoint(max(query.concurrencies), exact=True),
        image=NO_RECIPE_IMAGE,
        framework=query.framework,
        entrypoint=_entrypoint(query.framework),
        model=LaunchModel(query.model_id),
        topology=spelling.read(shell.args),
        args=shell.args,
        env=_without_runner_env(shell.env),
        client=_client({}),
        resolved_unix=snapshot.fetched_unix,
        runner_owned=_runner_owned(query.framework),
        stripped=(*shell.stripped, *_runner_env(shell.env)),
        warnings=unresolved,
    )


def _default_base(query: LaunchQuery, snapshot: SourceSnapshot, misses: tuple[SourceMiss, ...]) -> BaseLaunch | Miss:
    """The default table's launch for a fixed ``query`` no recipe serves, or why the table has none."""
    board = AMD_GPU_BOARDS.get(query.board)
    row = None if board is None else default_table().row(query.framework, board.gfx)
    if board is None or row is None:
        named = query.board if board is None else f"{query.board} ({board.gfx})"
        missing = SourceMiss(SourceKind.DEFAULT_TABLE, f"it has no {query.framework} row for {named}")
        return Miss((*misses, missing), snapshot.unread)
    topology = Topology(tp=query.tp or 1, ep=query.ep or 1)
    topology_args = _TOPOLOGY[query.framework].argv(topology)
    if isinstance(topology_args, str):
        return Miss(
            (*misses, SourceMiss(SourceKind.DEFAULT_TABLE, f"{query.framework} {topology_args}")), snapshot.unread
        )
    return BaseLaunch(
        query=query,
        source=LaunchSource(kind=SourceKind.DEFAULT_TABLE, path=TABLE_PATH, locator=row.origin),
        point=LaunchPoint(max(query.concurrencies), exact=True),
        image=NO_RECIPE_IMAGE,
        framework=query.framework,
        entrypoint=_entrypoint(query.framework),
        model=LaunchModel(query.model_id),
        topology=topology,
        args=(*row.args, *topology_args),
        env=row.env,
        client=_client({}),
        resolved_unix=snapshot.fetched_unix,
        runner_owned=_runner_owned(query.framework),
        warnings=(
            *_miss_warnings(misses),
            f"{_why_default(query, snapshot)}; the default {query.framework} launch for {board.gfx} runs it",
        ),
    )


def _why_default(query: LaunchQuery, snapshot: SourceSnapshot) -> str:
    if snapshot.unread is None:
        return f"no recipe serves {query.model_id}"
    return f"{snapshot.unread.source} could not be read, so no recipe was consulted ({snapshot.unread.reason})"


def _golden_table(snapshot: SourceSnapshot) -> GoldenTable:
    return GoldenTable(_carried(snapshot.inferencex, "InferenceX").bundle, snapshot.golden_lengths)


def resolve_base(query: LaunchQuery, snapshot: SourceSnapshot, *, gpus: int) -> BaseLaunch | Miss:
    """The base launch of a session asking ``query`` on ``gpus`` visible GPUs, read from ``snapshot`` alone.

    A :class:`Miss` names why each source holds no launch: always for an AgentX query no recipe serves or whose
    snapshot could not read a source, and for a fixed one the default table has no row for either.
    """
    _FIELDS.integer(gpus, "the visible GPU count", minimum=1)
    if snapshot.user_file is not None:
        return _user_file_base(query, snapshot, snapshot.user_file)
    if snapshot.unread is not None:
        if query.workload is Workload.AGENTX:
            return Miss((), snapshot.unread)
        return _default_base(query, snapshot, ())
    misses: list[SourceMiss] = []
    for read in _readers(query.framework):
        found = read(query, snapshot)
        choice = found if isinstance(found, SourceMiss) else _choose(found, gpus)
        if isinstance(choice, SourceMiss):
            misses.append(choice)
            continue
        if query.workload is Workload.AGENTX and choice.launch.golden is None:
            launch = with_golden(choice.launch, query, _golden_table(snapshot), snapshot.base_model)
            choice = dataclasses.replace(choice, launch=launch)
        return _recipe_base(query, snapshot, choice, misses)
    if query.workload is Workload.AGENTX:
        return Miss(tuple(misses))
    return _default_base(query, snapshot, tuple(misses))


__all__ = ["CLIENT_TOOL", "Miss", "SourceSnapshot", "UnreadSource", "resolve_base"]
