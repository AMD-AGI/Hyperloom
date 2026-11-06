# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sglang cookbook as a recipe source: deployment cells, AgentX launches and the launch blocks of its pages.

The cookbook states a launch in three forms, read in this order. A deployment config's ``cells`` (fixed workloads
only) are JavaScript, of which only literals are taken: a cell whose match, flags or env is code is refused, never
evaluated, and the page's overlay options are not composed on top. An AgentX file (AgentX only) is JSON whose
verified single-node aggregated launches are read with the point's patch applied. A page's ``bash`` launch block
serves the workload its section heads: AgentX under an Agentic or AgentX heading, fixed elsewhere; its board comes
from the nearest heading naming one, its concurrency range from its title and its image from the section's text.

Every form drops the flags the runner owns and keeps the rest as written. The cookbook pairs no golden curve with a
launch, so its launches carry no golden acceptance.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value
from inference_suite.contracts.base_launch import LaunchQuery, LaunchSource, SourceKind, Topology, Workload
from inference_suite.recipe.jslit import JsLiteralError, exported_value, has_expression
from inference_suite.recipe.model import (
    SGLANG_ENTRYPOINT,
    SGLANG_RUNNER_OWNED,
    SGLANG_TOPOLOGY,
    RecipeSourceError,
    RepoBundle,
    SourceLaunch,
    SourceMiss,
    boards_interchangeable,
)

SGLANG_REPO = "https://github.com/sgl-project/sglang"
CONFIG_DIR = "docs/src/snippets/configs/"
AGENTX_DIR = "docs/src/snippets/agentx/"
COOKBOOK_DIR = "docs/cookbook/"

_FRAMEWORK = "sglang"
#: The command lines that start an sglang server, as the cookbook spells them.
_ENTRYPOINTS = (("python", "-m", "sglang.launch_server"), SGLANG_ENTRYPOINT, ("sglang", "serve"))
_MODEL_FLAGS = frozenset({"model-path", "model"})
_MAX_RUNNING = frozenset({"max-running-requests"})
_DEFAULT_STRATEGY = "balanced"
_SPECULATIVE = frozenset({"speculative-algorithm"})
_HICACHE = frozenset({"enable-hierarchical-cache"})
_PLACEHOLDER = re.compile(r"{{(\w+)}}")
_PLACEHOLDER_IMAGE = re.compile(r"^<[^<>]*>$")
_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_EXPORT = re.compile(r"^export\s+const\s+\w+\s*=", re.MULTILINE)
_FENCE = re.compile(r"^```(\w*)[ \t]*(.*?)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_SHELL_LANGUAGES = frozenset({"bash", "sh", "shell"})
_AGENTX_HEADING = re.compile(r"agentic|agentx", re.IGNORECASE)
_CONCURRENCY = re.compile(r"concurrency\s+(\d+)(?:\s*[-\u2013]\s*(\d+))?", re.IGNORECASE)
_SHELL_SYNTAX = re.compile(r"[$`|;<>]|&&|\bexport\b")
_PROSE_FLAG = re.compile(r"`(--[\w-]+)")
_PROSE_IMAGE = re.compile(r"lmsysorg/sglang(?:-rocm)?:[\w.-]+")
#: The flags a cookbook AgentX worker lists first, in this order; the rest follow sorted (``_agentx.jsx``).
_LEAD_FLAGS = (
    *("--model-path", "--served-model-name", "--host", "--port", "--disaggregation-mode"),
    *("--disaggregation-bootstrap-port", "--dist-init-addr", "--nnodes", "--node-rank", "--request-plane"),
    "--kv-events-config",
)


@dataclass(frozen=True)
class _Command:
    """A server's flags and environment as the cookbook states them, the model they serve and the runner-owned
    flags taken out."""

    args: tuple[str, ...]
    env: Mapping[str, str]
    model: str | None
    stripped: tuple[str, ...]


def _flag_groups(words: Sequence[str]) -> list[list[str]]:
    """``words`` as flags, each with the values that follow it up to the next flag."""
    groups: list[list[str]] = []
    for word in words:
        if word.startswith("--"):
            flag, assigned, value = word.partition("=")
            groups.append([flag, value] if assigned else [flag])
        elif groups:
            groups[-1].append(word)
        else:
            raise RecipeSourceError(f"it passes {word!r} before any flag")
    return groups


def _command(groups: Sequence[Sequence[str]], env: Mapping[str, str]) -> _Command:
    """The server command of ``groups``, each a flag followed by its values, run with ``env``."""
    args: list[str] = []
    stripped: list[str] = []
    model = None
    for flag, *values in groups:
        if any("$" in word for word in values):
            raise RecipeSourceError(f"it sets {flag} from a shell variable")
        name = canonical_flag_name(flag)
        if name in _MODEL_FLAGS or name in SGLANG_RUNNER_OWNED:
            stripped.append(flag)
            model = values[-1] if name in _MODEL_FLAGS and values else model
        else:
            args.extend((flag, *values))
    return _Command(tuple(args), dict(env), model, tuple(stripped))


def _shell_words(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError as error:
        raise RecipeSourceError(f"it does not split as shell words: {error}") from error


@dataclass(frozen=True)
class _Fit:
    """The concurrency a recipe is taken at, and whether it is the one asked for."""

    conc: int
    exact: bool
    warning: str | None = None


@dataclass(frozen=True)
class _Recipe:
    """A launch as one cookbook form states it, before it is held to a query.

    ``board`` is the board it was tuned on and ``fit`` the concurrency it serves the query at. ``group`` names the cells it is one strategy of, and ``preference`` ranks
    its strategy among theirs: the default strategy first, then the page's order.
    """

    path: str
    locator: str
    board: str
    command: _Command
    fit: _Fit
    image: str | None
    strategy: str | None = None
    entry: str | None = None
    kv_offloading: str | None = None
    group: str = ""
    preference: int = 0
    warnings: tuple[str, ...] = ()


def _paths(bundle: RepoBundle, directory: str, suffix: str) -> list[str]:
    return sorted(path for path in bundle.files if path.startswith(directory) and path.endswith(suffix))


def _misfit(recipe: _Recipe, topology: Topology, query: LaunchQuery) -> str | None:
    """Why ``recipe``, which serves the query's model on a like board, cannot serve ``query``."""
    occurrences = argv_flag_occurrences(recipe.command.args)
    if query.workload is Workload.FIXED and last_alias_value(occurrences, _SPECULATIVE) is not None:
        return "it drafts tokens, and a fixed workload runs no speculative decoding"
    for name, asked in (("tp", query.tp), ("ep", query.ep)):
        if asked is not None and asked != getattr(topology, name):
            return f"it runs {name} {getattr(topology, name)}, not {asked}"
    return None


def _launch(recipe: _Recipe, query: LaunchQuery, bundle: RepoBundle) -> SourceLaunch | str:
    """The launch ``recipe`` gives ``query``, or why it gives none."""
    try:
        topology = SGLANG_TOPOLOGY.read(recipe.command.args)
    except RecipeSourceError as error:
        return f"{recipe.path} {recipe.locator}: {error}"
    misfit = _misfit(recipe, topology, query)
    if misfit is not None:
        return f"{recipe.path} {recipe.locator}: {misfit}"
    args = recipe.command.args
    kv_offloading = recipe.kv_offloading
    if kv_offloading is None:
        hicache = last_alias_value(argv_flag_occurrences(args), _HICACHE) is not None
        kv_offloading = "hicache" if hicache else "none"
    warnings = list(recipe.warnings)
    if recipe.board != query.board:
        warnings.append(f"the cookbook tuned {recipe.path} {recipe.locator} on {recipe.board}, not {query.board}")
    if recipe.fit.warning is not None:
        warnings.append(recipe.fit.warning)
    source = LaunchSource(
        kind=SourceKind.SGLANG_COOKBOOK,
        repo=bundle.repo,
        commit=bundle.commit,
        path=recipe.path,
        entry=recipe.entry,
        locator=recipe.locator,
        strategy=recipe.strategy,
        files={recipe.path: bundle.sha256(recipe.path)},
    )
    return SourceLaunch(
        source=source,
        framework=_FRAMEWORK,
        entrypoint=SGLANG_ENTRYPOINT,
        topology=topology,
        conc=recipe.fit.conc,
        exact=recipe.fit.exact,
        image=recipe.image,
        args=args,
        env=dict(recipe.command.env),
        kv_offloading=kv_offloading,
        stripped=recipe.command.stripped,
        warnings=tuple(warnings),
    )


def _outcome(found: Sequence[SourceLaunch | str], nothing: str) -> tuple[SourceLaunch, ...] | SourceMiss:
    """The launches in ``found``, else a miss naming why each candidate was refused (``nothing`` when none was)."""
    launches = tuple(item for item in found if isinstance(item, SourceLaunch))
    if launches:
        return launches
    return SourceMiss(SourceKind.SGLANG_COOKBOOK, "; ".join(item for item in found if isinstance(item, str)) or nothing)


def _image(named: object) -> str | None:
    """The image a recipe names, ``None`` for none or a placeholder."""
    return named if isinstance(named, str) and named and not _PLACEHOLDER_IMAGE.match(named) else None


def _image_warnings(named: object, where: str) -> tuple[str, ...]:
    if _image(named) is not None:
        return ()
    return (f"{where} names no image here" if not named else f"{where} names the placeholder image {named}",)


# Deployment cells ---------------------------------------------------------------------------------------------------


def _config(text: str) -> Mapping[str, Any]:
    try:
        config = exported_value(text, "config")
    except JsLiteralError as error:
        raise RecipeSourceError(f"its config does not read as JavaScript: {error}") from error
    if not isinstance(config, Mapping):
        raise RecipeSourceError("its config is computed by code")
    if not isinstance(config.get("cells"), list):
        raise RecipeSourceError("its cells are computed by code")
    if not isinstance(config.get("modelNames"), Mapping):
        raise RecipeSourceError("its model names are computed by code")
    return config


def _lookup(table: object, keys: Sequence[str | None]) -> Any:
    """The first of ``keys`` ``table`` maps, the cookbook's most-specific-first lookup."""
    if not isinstance(table, Mapping):
        return None
    return next((table[key] for key in keys if key is not None and table.get(key)), None)


def _keys(match: Mapping[str, Any], *dims: str) -> str | None:
    """``match``'s values for ``dims`` joined as the cookbook keys its tables, or None when a dim is absent."""
    values = [match.get(dim) for dim in dims]
    return "|".join(values) if all(isinstance(value, str) for value in values) else None


def _cell_model(config: Mapping[str, Any], match: Mapping[str, Any]) -> Any:
    keys = [
        _keys(match, "hw", "variant", "quant"),
        _keys(match, "variant", "quant"),
        _keys(match, "hw", "quant"),
        _keys(match, "quant"),
        _keys(match, "hw"),
        "default",
    ]
    return _lookup(config["modelNames"], keys)


def _cell_image(config: Mapping[str, Any], match: Mapping[str, Any]) -> Any:
    keys = [
        _keys(match, "hw", "variant", "quant"),
        _keys(match, "variant", "quant"),
        _keys(match, "hw", "quant", "strategy"),
        _keys(match, "hw", "quant"),
        _keys(match, "hw"),
    ]
    return _lookup(config.get("dockerImages"), keys)


def _single_node(cell: Mapping[str, Any]) -> bool:
    match = cell["match"]
    return (
        match.get("nodes", "single") == "single"
        and match.get("pdMode", "unified") == "unified"
        and cell.get("nnodes", 1) == 1
        and not cell.get("pd")
        and not cell.get("redirect")
    )


def _cell_env(entries: object) -> dict[str, str]:
    if not isinstance(entries, list):
        raise RecipeSourceError("its env is not a list")
    env: dict[str, str] = {}
    for entry in entries:
        words = _shell_words(entry) if isinstance(entry, str) else []
        assignment = _ENV_ASSIGNMENT.match(words[0]) if len(words) == 1 else None
        if assignment is None:
            raise RecipeSourceError(f"its env entry {entry!r} is not one NAME=value")
        env[assignment.group(1)] = assignment.group(2)
    return env


def _cell_command(config: Mapping[str, Any], cell: Mapping[str, Any], model: str) -> _Command:
    flags = cell.get("flags")
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise RecipeSourceError("its flags are not a list of strings")
    placeholders = config.get("placeholders")
    defaults = {
        name: str(spec["default"])
        for name, spec in (placeholders.items() if isinstance(placeholders, Mapping) else ())
        if isinstance(spec, Mapping) and isinstance(spec.get("default"), str | int)
    }
    defaults["MODEL_NAME"] = model
    text = " ".join(flags)
    missing = sorted(set(_PLACEHOLDER.findall(text)) - set(defaults))
    if missing:
        raise RecipeSourceError(f"its flags name placeholders {', '.join(missing)} with no default")
    words = _shell_words(_PLACEHOLDER.sub(lambda found: defaults[found.group(1)], text))
    return _command(_flag_groups(words), _cell_env(cell.get("env", [])))


def _strategy_preference(config: Mapping[str, Any], strategy: object) -> int:
    if strategy == _DEFAULT_STRATEGY:
        return 0
    items = config.get("strategies")
    listed = [item.get("id") for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []
    return 1 + (listed.index(strategy) if strategy in listed else len(listed))


def _running_cap(command: _Command) -> int | None:
    """The most requests ``command`` runs at once, ``None`` when it sets no cap."""
    cap = last_alias_value(argv_flag_occurrences(command.args), _MAX_RUNNING)
    try:
        return int(cap) if cap is not None else None
    except ValueError as error:
        raise RecipeSourceError(f"its --max-running-requests {cap!r} is not a count") from error


def _cell_fit(where: str, command: _Command, target: int) -> _Fit:
    """``target`` as a cell serves it: inexact when the cell caps its running requests below it."""
    running = _running_cap(command)
    if running is None or running >= target:
        return _Fit(target, exact=True)
    return _Fit(target, exact=False, warning=f"{where} runs at most {running} requests at once, below {target}")


def _cell_recipe(path: str, index: int, config: Mapping[str, Any], model: str, target: int) -> _Recipe:
    cell = config["cells"][index]
    match = cell["match"]
    if has_expression({"match": match, "flags": cell.get("flags"), "env": cell.get("env", [])}):
        raise RecipeSourceError("its match, flags or env is computed by code")
    image = _cell_image(config, match)
    warnings = _image_warnings(image, f"{path} for {match['hw']}")
    overlays = [dim.get("id") for dim in config.get("overlayDims") or [] if isinstance(dim, Mapping)]
    if overlays:
        warnings += (f"{path} composes overlay options {overlays} over its cells; the cell is taken as written",)
    command = _cell_command(config, cell, model)
    return _Recipe(
        path=path,
        locator=f"cells[{index}]",
        board=match["hw"],
        command=command,
        fit=_cell_fit(f"{path} cells[{index}]", command, target),
        image=_image(image),
        strategy=match.get("strategy"),
        group=json.dumps({key: value for key, value in match.items() if key != "strategy"}, sort_keys=True),
        preference=_strategy_preference(config, match.get("strategy")),
        warnings=warnings,
    )


def _config_recipes(path: str, text: str, query: LaunchQuery) -> list[_Recipe | str]:
    """The single-node cells of the config ``text`` that serve the query's model on a like board, a refused cell
    as the reason it is refused."""
    config = _config(text)
    recipes: list[_Recipe | str] = []
    for index, cell in enumerate(config["cells"]):
        if not isinstance(cell, Mapping) or not isinstance(cell.get("match"), Mapping):
            continue
        hw = cell["match"].get("hw")
        model = _cell_model(config, cell["match"])
        if model != query.model_id or not isinstance(hw, str) or not boards_interchangeable(query.board, hw):
            continue
        if _single_node(cell):
            try:
                recipes.append(_cell_recipe(path, index, config, model, max(query.concurrencies)))
            except RecipeSourceError as error:
                recipes.append(f"{path} cells[{index}]: {error}")
    return recipes


def _cell_rank(recipe: _Recipe) -> tuple[bool, int, int]:
    cap = _running_cap(recipe.command)
    shortfall = 0 if recipe.fit.exact else recipe.fit.conc - (cap or 0)
    return recipe.preference > 0 and cap is None, shortfall, recipe.preference


def _one_strategy(launches: list[tuple[_Recipe, SourceLaunch]]) -> list[SourceLaunch]:
    """Per group of cells, the launch that serves the concurrency asked for, else the one that caps its running
    requests nearest below it; among equals the default strategy, then the page's order. A cell of another strategy
    that sets no cap states no concurrency it is tuned for, so it serves only a group where no cell sets one."""
    groups: dict[str, list[tuple[_Recipe, SourceLaunch]]] = {}
    for recipe, launch in launches:
        groups.setdefault(recipe.group, []).append((recipe, launch))
    chosen = []
    for members in groups.values():
        recipe, launch = min(members, key=lambda member: _cell_rank(member[0]))
        if recipe.strategy != _DEFAULT_STRATEGY:
            note = f"{recipe.path} has no {_DEFAULT_STRATEGY} cell that fits; took strategy {recipe.strategy}"
            launch = dataclasses.replace(launch, warnings=(*launch.warnings, note))
        chosen.append(launch)
    return chosen


def _cell_launches(query: LaunchQuery, bundle: RepoBundle) -> tuple[SourceLaunch, ...] | SourceMiss:
    found: list[tuple[_Recipe, SourceLaunch]] = []
    reasons = []
    for path in _paths(bundle, CONFIG_DIR, ".jsx"):
        text = bundle.text(path)
        if query.model_id not in text:
            continue
        try:
            recipes = _config_recipes(path, text, query)
        except RecipeSourceError as error:
            reasons.append(f"{path}: {error}")
            continue
        for recipe in recipes:
            launch = recipe if isinstance(recipe, str) else _launch(recipe, query, bundle)
            if isinstance(launch, str):
                reasons.append(launch)
            else:
                found.append((recipe, launch))
    on_board = [(recipe, launch) for recipe, launch in found if recipe.board == query.board]
    return _outcome([*_one_strategy(on_board or found), *reasons], "no deployment cell serves it here")


# AgentX launches -----------------------------------------------------------------------------------------------------


def _agentx_data(text: str) -> Mapping[str, Any]:
    export = _EXPORT.search(text)
    if export is None:
        raise RecipeSourceError("it exports no AgentX data")
    try:
        data = json.loads(text[export.end() :].strip().removesuffix(";"))
    except ValueError as error:
        raise RecipeSourceError(f"its AgentX data is not JSON: {error}") from error
    if not isinstance(data, Mapping) or not isinstance(data.get("cells"), list):
        raise RecipeSourceError("its AgentX data holds no cells")
    return data


def _arg_rank(arg: Sequence[str]) -> tuple[int, str]:
    return (_LEAD_FLAGS.index(arg[0]), "") if arg[0] in _LEAD_FLAGS else (len(_LEAD_FLAGS), arg[0])


def _apply_diff(block: Mapping[str, Any], diff: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """``block`` with a point's per-block patch applied: ``None`` drops a flag or variable, a value sets it."""
    if not diff:
        return block
    if "block" in diff:
        return diff["block"]
    args = _patched_args(block["args"], diff.get("args") or {})
    if block.get("kind") == "worker":
        args.sort(key=_arg_rank)
    env_changes = diff.get("env") or {}
    env = {name: env_changes.get(name, value) for name, value in (block.get("env") or {}).items()}
    env.update({name: value for name, value in env_changes.items() if name not in env})
    return {**block, "args": args, "env": {name: env[name] for name in sorted(env) if env[name] is not None}}


def _patched_args(args: Sequence[Sequence[str]], changes: Mapping[str, Any]) -> list[list[str]]:
    """``args`` with each changed flag's values replaced in place and new flags last; ``None`` removes a flag."""
    patched = [
        [arg[0], *changes[arg[0]]] if arg[0] in changes else list(arg)
        for arg in args
        if changes.get(arg[0], arg) is not None
    ]
    named = {arg[0] for arg in patched}
    return patched + [[flag, *values] for flag, values in changes.items() if values is not None and flag not in named]


def _point_blocks(data: Mapping[str, Any], cell: Mapping[str, Any], point: int) -> list[Mapping[str, Any]]:
    """The sglang router's blocks at the cell's ``point``, as the cookbook's AgentX explorer decodes them."""
    blocks = []
    for block in cell["routers"]["sglang"]["blocks"]:
        if "ref" in block or "from" in block:
            raise RecipeSourceError("its launch is written relative to another launch")
        if "envRef" in block:
            block = {**block, "env": data["envs"][block["envRef"]]}
        blocks.append(block)
    patch = (cell["points"][point].get("patch") or {}).get("sglang") if point else None
    if isinstance(patch, Mapping):
        return list(patch["replace"])
    return [_apply_diff(block, patch[index] if patch else None) for index, block in enumerate(blocks)]


def _agentx_point(cell: Mapping[str, Any], target: int) -> int:
    """The index of the cell's point at ``target``, else of its nearest larger one."""
    fitting = [(conc, index) for index, point in enumerate(cell["points"]) for conc in point["concs"] if conc >= target]
    if not fitting:
        raise RecipeSourceError(f"it has no point at concurrency {target} or above")
    return min(fitting)[1]


def _agentx_fit(cell: Mapping[str, Any], index: int, target: int) -> _Fit:
    conc = min(conc for conc in cell["points"][index]["concs"] if conc >= target)
    warning = None if conc == target else f"{cell['id']} has no point at concurrency {target}; took {conc}"
    return _Fit(conc, exact=conc == target, warning=warning)


def _agentx_worker(cell: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    router = cell["routers"]["sglang"]
    if cell.get("setup") or router.get("install") or router.get("files"):
        raise RecipeSourceError("it needs setup, installs or files beside its launch")
    shape = [
        (block.get("kind"), block.get("role"), block.get("workers"), block.get("nodes_per_worker")) for block in blocks
    ]
    if shape != [("worker", "agg", 1, 1)]:
        raise RecipeSourceError(f"it runs {shape}, not one aggregated single-node worker")
    return blocks[0]


def _agentx_recipe(path: str, data: Mapping[str, Any], cell: Mapping[str, Any], target: int) -> _Recipe:
    if (cell.get("prov") or {}).get("status") != "verified":
        raise RecipeSourceError(f"its provenance is {(cell.get('prov') or {}).get('status')!r}, not verified")
    if "sglang" not in (cell.get("routers") or {}):
        raise RecipeSourceError(f"it runs behind {sorted(cell.get('routers') or {})}, not sglang alone")
    index = _agentx_point(cell, target)
    worker = _agentx_worker(cell, _point_blocks(data, cell, index))
    if tuple(_shell_words(worker["cmd"])) not in _ENTRYPOINTS:
        raise RecipeSourceError(f"it runs {worker['cmd']!r}, not an sglang server")
    env = {name: str(value) for name, value in (worker.get("env") or {}).items()}
    return _Recipe(
        path=path,
        locator=f"{cell['id']} points[{index}]",
        board=cell["hw"],
        command=_command([[str(word) for word in arg] for arg in worker["args"]], env),
        fit=_agentx_fit(cell, index, target),
        image=_image(cell.get("image")),
        entry=cell.get("configKey"),
        kv_offloading=cell["points"][index].get("kv", "none"),
        warnings=_image_warnings(cell.get("image"), f"{path} {cell['id']}"),
    )


def _agentx_cell_launch(
    path: str, data: Mapping[str, Any], cell: Mapping[str, Any], query: LaunchQuery, bundle: RepoBundle
) -> SourceLaunch | str:
    try:
        recipe = _agentx_recipe(path, data, cell, max(query.concurrencies))
    except RecipeSourceError as error:
        return f"{path} {cell.get('id')}: {error}"
    return _launch(recipe, query, bundle)


def _agentx_launches(query: LaunchQuery, bundle: RepoBundle) -> tuple[SourceLaunch, ...] | SourceMiss:
    found: list[SourceLaunch | str] = []
    for path in _paths(bundle, AGENTX_DIR, ".jsx"):
        text = bundle.text(path)
        if query.model_id not in text:
            continue
        try:
            data = _agentx_data(text)
        except RecipeSourceError as error:
            found.append(f"{path}: {error}")
            continue
        found += [
            _agentx_cell_launch(path, data, cell, query, bundle)
            for cell in data["cells"]
            if cell.get("ckpt") == query.model_id and boards_interchangeable(query.board, cell.get("hw", ""))
        ]
    return _outcome(found, "no AgentX launch serves it here")


# Page launch blocks --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _PageBlock:
    """A shell code block of a cookbook page, with the headings above it and the prose of its section."""

    path: str
    line: int
    title: str
    text: str
    headings: tuple[str, ...]
    prose: str


def _section_blocks(
    path: str, headings: Sequence[tuple[int, str]], prose: Sequence[str], blocks: Sequence[tuple[int, str, str]]
) -> Iterator[_PageBlock]:
    titles = tuple(title for _, title in headings)
    for line, title, text in blocks:
        yield _PageBlock(path, line, title, text, titles, "\n".join(prose))


def _page_blocks(path: str, text: str) -> Iterator[_PageBlock]:
    """The shell code blocks of the page ``text``, each with its section's headings and prose."""
    headings: list[tuple[int, str]] = []
    prose: list[str] = []
    blocks: list[tuple[int, str, str]] = []
    fence: tuple[int, str, bool] | None = None
    fenced: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if fence is not None and not line.strip().startswith("```"):
            fenced.append(line)
            continue
        if fence is not None:
            if fence[2]:
                blocks.append((fence[0], fence[1], "\n".join(fenced)))
            fence = None
            continue
        opening, heading = _FENCE.match(line.strip()), _HEADING.match(line)
        if opening is not None:
            fence, fenced = (number, opening.group(2), opening.group(1) in _SHELL_LANGUAGES), []
        elif heading is not None:
            yield from _section_blocks(path, headings, prose, blocks)
            level = len(heading.group(1))
            headings = [*(item for item in headings if item[0] < level), (level, heading.group(2))]
            prose, blocks = [], []
        else:
            prose.append(line)
    yield from _section_blocks(path, headings, prose, blocks)


def _block_board(block: _PageBlock) -> str:
    """The board named by the heading nearest ``block``."""
    for heading in reversed(block.headings):
        named = [product for product, board in AMD_GPU_BOARDS.items() if board.named_in(heading)]
        if len(named) == 1:
            return named[0]
    raise RecipeSourceError("no heading above it names one AMD board")


def _block_command(block: _PageBlock) -> _Command:
    """The launch of ``block``'s one command line."""
    if _SHELL_SYNTAX.search(block.text):
        raise RecipeSourceError("it uses shell syntax beyond one literal command")
    lines = [line for line in block.text.replace("\\\n", " ").splitlines() if line.strip()]
    lines = [line for line in lines if not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise RecipeSourceError(f"it holds {len(lines)} commands, not one")
    words = _shell_words(lines[0])
    env = {}
    while words and _ENV_ASSIGNMENT.match(words[0]):
        name, _, value = words.pop(0).partition("=")
        env[name] = value
    entrypoint = next((start for start in _ENTRYPOINTS if tuple(words[: len(start)]) == start), None)
    if entrypoint is None:
        raise RecipeSourceError("it does not start an sglang server")
    return _command(_flag_groups(words[len(entrypoint) :]), env)


def _block_fit(block: _PageBlock, target: int) -> _Fit:
    """The concurrency the range in ``block``'s title gives ``target``; a block naming none serves any."""
    found = _CONCURRENCY.search(block.title)
    if found is None:
        return _Fit(target, exact=True)
    low, high = int(found.group(1)), int(found.group(2) or found.group(1))
    if target > high:
        raise RecipeSourceError(f"it is tuned for concurrency {low}-{high}, below {target}")
    if target < low:
        return _Fit(low, exact=False, warning=f"{block.title!r} starts at concurrency {low}; took it for {target}")
    return _Fit(target, exact=True)


def _block_recipe(block: _PageBlock, command: _Command, target: int) -> _Recipe:
    where = f"{block.path}:{block.line}"
    image = next(iter(_PROSE_IMAGE.findall(block.prose)), None)
    warnings = _image_warnings(image, f"the section of {where}")
    named = {flag for flag in _PROSE_FLAG.findall(block.prose) if flag in command.args}
    if named:
        note = f"the section of {where} adjusts {', '.join(sorted(named))} in its text; the command is taken as written"
        warnings += (note,)
    return _Recipe(
        path=block.path,
        locator=f"line {block.line} {block.title!r}",
        board=_block_board(block),
        command=command,
        fit=_block_fit(block, target),
        image=image,
        warnings=warnings,
    )


def _block_launch(block: _PageBlock, query: LaunchQuery, bundle: RepoBundle) -> SourceLaunch | str | None:
    """The launch ``block`` gives ``query``, why it gives none, or ``None`` when it serves another model or board."""
    try:
        command = _block_command(block)
        if command.model != query.model_id or not boards_interchangeable(query.board, _block_board(block)):
            return None
        return _launch(_block_recipe(block, command, max(query.concurrencies)), query, bundle)
    except RecipeSourceError as error:
        return f"{block.path}:{block.line}: {error}"


def _page_launches(query: LaunchQuery, bundle: RepoBundle) -> tuple[SourceLaunch, ...] | SourceMiss:
    agentx = query.workload is Workload.AGENTX
    markers = [" ".join(entrypoint) for entrypoint in _ENTRYPOINTS]
    found: list[SourceLaunch | str] = []
    for path in _paths(bundle, COOKBOOK_DIR, ".mdx"):
        for block in _page_blocks(path, bundle.text(path)):
            if not any(marker in block.text for marker in markers) or query.model_id not in block.text:
                continue
            if any(_AGENTX_HEADING.search(heading) for heading in block.headings) == agentx:
                launch = _block_launch(block, query, bundle)
                found += [] if launch is None else [launch]
    return _outcome(found, "no page launch block serves it here")


def sglang_cookbook_launches(query: LaunchQuery, bundle: RepoBundle) -> tuple[SourceLaunch, ...] | SourceMiss:
    """The launches the first cookbook form that serves ``query`` holds for it at ``max(query.concurrencies)``.

    A fixed workload reads the deployment cells, then the page blocks; an AgentX workload the AgentX launches, then
    the page blocks.
    """
    if query.framework != _FRAMEWORK:
        return SourceMiss(SourceKind.SGLANG_COOKBOOK, f"the sglang cookbook launches sglang, not {query.framework}")
    forms = (_cell_launches, _page_launches) if query.workload is Workload.FIXED else (_agentx_launches, _page_launches)
    reasons = []
    for form in forms:
        found = form(query, bundle)
        if not isinstance(found, SourceMiss):
            return found
        reasons.append(found.reason)
    return SourceMiss(SourceKind.SGLANG_COOKBOOK, "; ".join(reasons))


__all__ = ["AGENTX_DIR", "COOKBOOK_DIR", "CONFIG_DIR", "SGLANG_REPO", "sglang_cookbook_launches"]
