# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ATOM as a recipe source for fixed workloads: its benchmark catalog, then the literal launches on its recipe pages.

The catalog (``models.json``) is read the way ATOM's ``catalog.py`` composes a server command, in its fixed order
``--kv_cache_dtype <dtype> [-tp <n>] [--trust-remote-code] [config.extra_args] [variant.extra_args]``; a variant is
an arm whose points are its scenarios' concurrencies inside its band. The recipe pages are read only when the catalog
gives no launch, and only through shell blocks whose text alone fixes the launch: a block that expands a variable,
branches or runs anything besides exports and the server is refused, never executed.

ATOM's agentic catalogs are not read: an AgentX session does not run ATOM. Environment the catalog's harness reads
for the GPUs and the load is not the server's, so it is taken out and named in ``stripped``.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from hyperloom_common.contract_fields import ContractFields
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value
from inference_suite.contracts.base_launch import LaunchQuery, LaunchSource, SourceKind, Topology, Workload
from inference_suite.recipe.model import (
    ATOM_ENTRYPOINT,
    ATOM_RUNNER_OWNED,
    ATOM_TOPOLOGY,
    RecipeSourceError,
    RepoBundle,
    SourceLaunch,
    SourceMiss,
    boards_interchangeable,
)
from inference_suite.recipe.shell_launch import _logical_lines

ATOM_REPO = "https://github.com/ROCm/ATOM"
CATALOG = ".github/benchmark/models.json"
RECIPES_DIR = "recipes/"

_FIELDS = ContractFields(RecipeSourceError)
_FRAMEWORK = "atom"
# catalog.py's band when a variant sets none, at the pinned commit.
_DEFAULT_CONC_MIN = 0
_DEFAULT_CONC_MAX = 256
_BASE_VARIANT: Mapping[str, Any] = {"label": "", "suffix": ""}
_MODEL = frozenset({"model"})
_SPECULATIVE = frozenset({"method", "num-speculative-tokens", "draft-model"})
# Ranks beside tensor parallelism, under every alias atom/model_engine/arg_utils.py takes; a topology of tp and ep
# cannot state them.
_STAGES = frozenset(
    {"pipeline-parallel-size", "pp", "prefill-context-parallel-size", "pcp", "data-parallel-size", "dp"}
)
# Environment ATOM's benchmark harness reads for the GPUs it hands out and the load it drives, not the server.
_HARNESS_ENV = frozenset({"HIP_VISIBLE_DEVICES", "PRECISION"})
_HARNESS_ENV_PREFIXES = ("AIPERF_", "BENCHMARK_", "ATOM_BENCHMARK_")
_SHELL_OPERATORS = ";&|<>()"
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_FENCE = re.compile(r"^\s*```(\w*)")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$")
_SHELL_LANGUAGES = frozenset({"bash", "sh", "shell"})
_BOARD_NAME = re.compile(r"MI(\d{3})X?(?![0-9])")
_DOCKER_PULL = re.compile(r"^\s*docker pull (\S+)\s*$", re.MULTILINE)


def _miss(reason: str) -> SourceMiss:
    return SourceMiss(SourceKind.ATOM_CATALOG, reason)


def _words(line: str) -> list[str] | str:
    """The words of one literal shell line, or why the line is not literal."""
    if "$" in line or "`" in line:
        return "it expands shell variables or commands"
    lexer = shlex.shlex(line, posix=True, punctuation_chars=_SHELL_OPERATORS)
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError as error:
        return f"it does not parse as shell words: {error}"
    operator = next((word for word in words if set(word) <= set(_SHELL_OPERATORS)), None)
    return f"it uses shell operator {operator!r}" if operator else words


@dataclass(frozen=True)
class _Command:
    """A server launch a block's text fixes: the flags after the entrypoint and the environment set for it."""

    argv: tuple[str, ...]
    env: Mapping[str, str]


def _assignments(words: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """The leading ``NAME=value`` words as environment, and the words after them."""
    env: dict[str, str] = {}
    for index, word in enumerate(words):
        match = _ASSIGNMENT.match(word)
        if match is None:
            return env, list(words[index:])
        env[match.group(1)] = match.group(2)
    return env, []


def _literal_command(text: str) -> _Command | str:
    """The one ATOM server launch ``text`` fixes, or why it fixes none: only exports and the server may run."""
    env: dict[str, str] = {}
    argv: tuple[str, ...] | None = None
    for line in _logical_lines(text):
        words = _words(line)
        if isinstance(words, str):
            return words
        exporting = words[:1] == ["export"]
        assigned, rest = _assignments(words[1:] if exporting else words)
        env.update(assigned)
        if not rest:
            continue
        if exporting or tuple(rest[1:3]) != ATOM_ENTRYPOINT[1:] or rest[0] not in ("python", "python3"):
            return f"it runs {shlex.join(rest[:3])!r}"
        if argv is not None:
            return "it starts more than one server"
        argv = tuple(rest[3:])
    if argv is None:
        return "it starts no ATOM server"
    return _Command(argv, env)


def _server_args(argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The flags with the runner-owned ones taken out, and the owned flags taken out."""
    args: list[str] = []
    stripped: list[str] = []
    owned = False
    for token in argv:
        if token.startswith("-") and not token[1:2].isdigit():
            flag = token.partition("=")[0]
            owned = canonical_flag_name(flag) in ATOM_RUNNER_OWNED
            stripped += [flag] if owned else []
        if not owned:
            args.append(token)
    return tuple(args), tuple(stripped)


def _server_env(env: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """The server's environment, and the names of the harness's taken out of it."""
    harness = tuple(name for name in env if name in _HARNESS_ENV or name.startswith(_HARNESS_ENV_PREFIXES))
    return {name: value for name, value in env.items() if name not in harness}, harness


@dataclass(frozen=True)
class _Candidate:
    """A launch ATOM states for the query's model, before it is held to the rest of the query."""

    source: LaunchSource
    board: str
    conc: int
    exact: bool
    image: str | None
    command: _Command
    warnings: tuple[str, ...] = ()


def _misfit(args: Sequence[str], topology: Topology, query: LaunchQuery) -> str | None:
    occurrences = argv_flag_occurrences(args)
    if last_alias_value(occurrences, _SPECULATIVE) is not None:
        return "it drafts tokens, and a fixed workload runs no speculative decoding"
    stages = sorted(name for name, _ in occurrences if name in _STAGES)
    if stages:
        return f"it runs {', '.join(stages)} beside tensor parallelism"
    for name, asked in (("tp", query.tp), ("ep", query.ep)):
        if asked is not None and asked != getattr(topology, name):
            return f"it runs {name} {getattr(topology, name)}, not {asked}"
    return None


def _launch(candidate: _Candidate, query: LaunchQuery) -> SourceLaunch | str:
    """The launch ``candidate`` states for ``query``, or why it does not serve it."""
    where = f"{candidate.source.path} {candidate.source.locator}"
    if not boards_interchangeable(query.board, candidate.board):
        return f"{where}: it is tuned on {candidate.board}, not {query.board}"
    args, stripped = _server_args(candidate.command.argv)
    topology = ATOM_TOPOLOGY.read(args)
    misfit = _misfit(args, topology, query)
    if misfit is not None:
        return f"{where}: {misfit}"
    env, harness = _server_env(candidate.command.env)
    warnings = candidate.warnings
    if candidate.board != query.board:
        warnings += (f"ATOM tuned {candidate.source.entry} on {candidate.board}, not {query.board}",)
    return SourceLaunch(
        source=candidate.source,
        framework=_FRAMEWORK,
        entrypoint=ATOM_ENTRYPOINT,
        topology=topology,
        conc=candidate.conc,
        exact=candidate.exact,
        image=candidate.image,
        args=args,
        env=env,
        stripped=stripped + harness,
        warnings=warnings,
    )


def _runner_board(runner: str) -> str | None:
    """The board a catalog runner label names (``linux-atom-pitt-mi350x-8`` names mi350x), if any."""
    return next((word for word in runner.split("-") if word in AMD_GPU_BOARDS), None)


def _catalog_args(config: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    """The server flags of a catalog variant, composed in ``catalog.py``'s fixed order."""
    parts = [f"--kv_cache_dtype {config.get('kv_cache_dtype', 'fp8')}"]
    if config.get("tp") is not None:
        parts.append(f"-tp {config['tp']}")
    if config.get("trust_remote_code"):
        parts.append("--trust-remote-code")
    parts += [str(extra) for extra in (config.get("extra_args"), variant.get("extra_args")) if extra]
    return " ".join(parts)


def _catalog_env(model: Mapping[str, Any], variant: Mapping[str, Any], where: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for owner in (model, variant):
        for line in filter(None, str(owner.get("env_vars") or "").splitlines()):
            match = _ASSIGNMENT.match(line.strip())
            if match is None:
                raise RecipeSourceError(f"{CATALOG} {where} sets env line {line!r}, not NAME=value")
            env[match.group(1)] = match.group(2)
    return env


def _variant_points(
    catalog: Mapping[str, Any], model: Mapping[str, Any], variant: Mapping[str, Any]
) -> tuple[int, ...]:
    """The concurrencies the variant is benchmarked at: its scenarios' (else its model's, else the catalog's) inside
    its band."""
    scenarios = variant.get("scenarios") or model.get("scenarios") or catalog.get("default_scenarios") or ()
    low = variant.get("conc_min", _DEFAULT_CONC_MIN)
    high = variant.get("conc_max", _DEFAULT_CONC_MAX)
    concs = {conc for scenario in scenarios for conc in scenario.get("concurrency", ()) if low <= conc <= high}
    return tuple(sorted(concs))


@dataclass(frozen=True)
class _Variant:
    """One catalog variant: its model entry, the variant itself and where the catalog holds it."""

    model: Mapping[str, Any]
    variant: Mapping[str, Any]
    where: str


def _catalog_candidate(bundle: RepoBundle, catalog: Mapping[str, Any], held: _Variant, target: int) -> _Candidate | str:
    model, variant, where = held.model, held.variant, held.where
    entry = " ".join(filter(None, (str(model.get("display")), variant.get("label"))))
    fitting = [conc for conc in _variant_points(catalog, model, variant) if conc >= target]
    if not fitting:
        return f"{CATALOG} {where}: {entry} is benchmarked at no concurrency of {target} or more"
    board = _runner_board(str(model.get("runner")))
    if board is None:
        return f"{CATALOG} {where}: its runner {model.get('runner')!r} names no board"
    words = _words(_catalog_args(model.get("config") or {}, variant))
    if isinstance(words, str):
        return f"{CATALOG} {where}: its args are not literal: {words}"
    conc = min(fitting)
    source = LaunchSource(
        kind=SourceKind.ATOM_CATALOG,
        repo=bundle.repo,
        commit=bundle.commit,
        path=CATALOG,
        entry=entry,
        locator=where,
        files={CATALOG: bundle.sha256(CATALOG)},
    )
    warnings = () if conc == target else (f"{entry} has no point at concurrency {target}; took {conc}",)
    command = _Command(tuple(words), _catalog_env(model, variant, where))
    return _Candidate(source, board, conc, conc == target, None, command, warnings)


def _catalog_candidates(query: LaunchQuery, bundle: RepoBundle) -> Iterator[_Candidate | str]:
    """Each catalog variant serving the query's model, as a candidate or why it is none."""
    try:
        catalog = _FIELDS.mapping(json.loads(bundle.text(CATALOG)), f"the ATOM catalog {CATALOG}")
    except ValueError as error:
        raise RecipeSourceError(f"the ATOM catalog {CATALOG} is no JSON: {error}") from error
    for model_index, model in enumerate(_FIELDS.items(catalog.get("models"), "the ATOM catalog models")):
        model = _FIELDS.mapping(model, f"ATOM catalog model {model_index}")
        for variant_index, variant in enumerate(model.get("variants") or [_BASE_VARIANT]):
            where = f"models[{model_index}].variants[{variant_index}]"
            variant = _FIELDS.mapping(variant, f"ATOM catalog variant {where}")
            if variant.get("path", model.get("path")) == query.model_id:
                held = _Variant(model, variant, where)
                yield _catalog_candidate(bundle, catalog, held, max(query.concurrencies))


@dataclass(frozen=True)
class _PageBlock:
    """A shell block naming the ATOM server on a recipe page (``index`` counts those blocks), with the headings it
    sits under, nearest first."""

    path: str
    index: int
    headings: tuple[str, ...]
    text: str


def _fenced_body(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        if line.strip() == "```":
            return
        yield line


def _page_blocks(path: str, text: str) -> Iterator[_PageBlock]:
    headings: dict[int, str] = {}
    lines = iter(text.splitlines())
    index = 0
    for line in lines:
        heading = _HEADING.match(line)
        if heading is not None:
            level = len(line) - len(line.lstrip("#"))
            headings = {depth: title for depth, title in headings.items() if depth < level}
            headings[level] = heading.group(1)
            continue
        fence = _FENCE.match(line)
        if fence is None:
            continue
        body = "\n".join(_fenced_body(lines))
        if fence.group(1) in _SHELL_LANGUAGES and ATOM_ENTRYPOINT[-1] in body:
            nearest_first = tuple(headings[depth] for depth in sorted(headings, reverse=True))
            yield _PageBlock(path, index, nearest_first, body)
            index += 1


def _heading_board(block: _PageBlock, query: LaunchQuery) -> str | None:
    """The board the nearest heading naming any tunes the block for: the query's, else one interchangeable with it,
    else the first it names; ``None`` when no heading names one."""
    for heading in block.headings:
        named = [f"mi{digits}x" for digits in _BOARD_NAME.findall(heading)]
        if named:
            fits = [board for board in named if boards_interchangeable(query.board, board)]
            return query.board if query.board in named else (fits or named)[0]
    return None


def _page_candidate(block: _PageBlock, bundle: RepoBundle, query: LaunchQuery) -> _Candidate | str | None:
    """The block as a candidate, why it cannot be one, or ``None`` when it launches another model."""
    where = f"{block.path} block {block.index}"
    command = _literal_command(block.text)
    if isinstance(command, str):
        return f"{where}: it is not a literal launch: {command}" if query.model_id in block.text else None
    if last_alias_value(argv_flag_occurrences(command.argv), _MODEL) != query.model_id:
        return None
    board = _heading_board(block, query)
    if board is None:
        return f"{where}: no heading above it names a board"
    image = _DOCKER_PULL.search(bundle.text(block.path))
    source = LaunchSource(
        kind=SourceKind.ATOM_CATALOG,
        repo=bundle.repo,
        commit=bundle.commit,
        path=block.path,
        entry=block.headings[0] if block.headings else block.path,
        locator=f"block {block.index}",
        files={block.path: bundle.sha256(block.path)},
    )
    target = max(query.concurrencies)
    warning = f"{where} names no concurrency; it is taken as tuned for {target}"
    return _Candidate(source, board, target, True, image and image.group(1), command, (warning,))


def _page_candidates(query: LaunchQuery, bundle: RepoBundle) -> Iterator[_Candidate | str]:
    pages = sorted(
        path
        for path in bundle.files
        if path.startswith(RECIPES_DIR) and path.endswith(".md") and "/" not in path[len(RECIPES_DIR) :]
    )
    for path in pages:
        for block in _page_blocks(path, bundle.text(path)):
            candidate = _page_candidate(block, bundle, query)
            if candidate is not None:
                yield candidate


def _held(candidates: Iterator[_Candidate | str], query: LaunchQuery) -> list[SourceLaunch | str]:
    return [item if isinstance(item, str) else _launch(item, query) for item in candidates]


def _outcome(found: Sequence[SourceLaunch | str], query: LaunchQuery) -> tuple[SourceLaunch, ...] | SourceMiss:
    launches = [item for item in found if isinstance(item, SourceLaunch)]
    if not launches:
        reasons = [item for item in found if isinstance(item, str)]
        return _miss("; ".join(reasons) or f"ATOM's catalog and recipe pages hold no launch of {query.model_id}")
    if len(launches) == 1:
        return (launches[0],)
    named = ", ".join(f"{launch.source.path} {launch.source.locator}" for launch in launches)
    warning = f"ATOM holds {len(launches)} launches of {query.model_id} that fit: {named}"
    return tuple(replace(launch, warnings=(*launch.warnings, warning)) for launch in launches)


def atom_catalog_launches(query: LaunchQuery, bundle: RepoBundle) -> tuple[SourceLaunch, ...] | SourceMiss:
    """The launches ATOM states for a fixed ``query`` at ``max(query.concurrencies)``: the catalog's variants that
    fit, else the literal page blocks that do."""
    if query.framework != _FRAMEWORK:
        return _miss(f"ATOM recipes launch atom, not {query.framework}")
    if query.workload is not Workload.FIXED:
        return _miss("ATOM recipes serve fixed workloads only; its agentic catalogs are not read")
    found = _held(_catalog_candidates(query, bundle), query)
    if not any(isinstance(item, SourceLaunch) for item in found):
        found += _held(_page_candidates(query, bundle), query)
    return _outcome(found, query)


__all__ = [
    "ATOM_REPO",
    "CATALOG",
    "RECIPES_DIR",
    "atom_catalog_launches",
]
