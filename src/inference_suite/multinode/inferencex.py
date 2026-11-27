# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD launches from InferenceX main: its master multi-node disaggregated entries.

A master entry (``multinode: true, disagg: true``) names a srt-slurm recipe per ``search-space`` row as
``srt-recipe: <file>:<variant>``; the row states the topology and concurrencies InferenceX benchmarks it at. Variants
expand as srtctl expands them: ``base`` deep-merged with an ``override_*`` block, or with one slice of a
``zip_override_*`` block.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from inference_suite.contracts.base_launch import Workload
from inference_suite.argv import canonical_flag_name
from inference_suite.frameworks import Framework
from inference_suite.multinode.query import PdQuery, RoleAsk
from inference_suite.multinode.render import DP_ATTENTION, TOPOLOGY_SPELLING
from inference_suite.multinode.record import PD_ROLES, Role
from inference_suite.recipe.inferencex import (
    MASTER,
    PROJECT,
    RUNNERS,
    SCENARIO_OF,
    UNSET,
    concurrencies,
    engine_type,
    org_models_named,
    runner_board,
)
from inference_suite.recipe.model import RecipeSourceError
from inference_suite.recipe.sources import Checkout

MULTI_NODE_RECIPES = f"{PROJECT}/benchmarks/multi_node/srt-slurm-recipes"
#: What a PD resolution checks out of InferenceX.
CHECKOUT_DIRECTORIES = (f"{PROJECT}/configs", MULTI_NODE_RECIPES)
_ZIP_SELECTOR = re.compile(r"(zip_override_[\w-]+)\[(\d+)\]")


@dataclass(frozen=True)
class RoleFit:
    """The topology a candidate states for one role: workers, each worker's TP, EP, DP attention and GPUs."""

    workers: int
    tp: int
    ep: int
    dp_attention: bool
    gpus: int


@dataclass(frozen=True)
class Candidate:
    """One InferenceX recipe variant that serves a query: where it is, the merged recipe, both roles' topology, the
    entry's KV transfer backend and benchmarked concurrency and the warnings it carries."""

    entry: str
    path: str
    locator: str
    recipe: Mapping[str, Any]
    fits: Mapping[Role, RoleFit]
    conc: int
    transfer_backend: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidates:
    """The candidates a source holds for a query, in its order, and why every other one it read misses."""

    found: tuple[Candidate, ...]
    misses: tuple[str, ...]


# --- srtctl variants ----------------------------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """srtctl's merge: ``override`` wins, mappings merge, lists replace, null deletes."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _list_lengths(group: Mapping[str, Any]) -> list[int]:
    lengths: list[int] = []
    for value in group.values():
        if isinstance(value, list):
            lengths.append(len(value))
        elif isinstance(value, Mapping):
            lengths += _list_lengths(value)
    return lengths


def _zip_slice(group: Mapping[str, Any], index: int) -> dict[str, Any]:
    """``group`` with each list replaced by its ``index`` item, or its only item."""
    sliced: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, list):
            sliced[key] = value[0 if len(value) == 1 else index]
        elif isinstance(value, Mapping):
            sliced[key] = _zip_slice(value, index)
        else:
            sliced[key] = value
    return sliced


def _without_nulls(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    return value


def _zip_variants(raw: Mapping[str, Any], key: str) -> list[tuple[str, dict[str, Any]]]:
    sizes = set(_list_lengths(raw[key])) - {1}
    if len(sizes) > 1 or 0 in sizes:
        raise RecipeSourceError(f"{key} has list values of lengths {sorted(sizes)}")
    count = sizes.pop() if sizes else 1
    return [
        (f"{key}[{index}]", _without_nulls(deep_merge(raw["base"], _zip_slice(raw[key], index))))
        for index in range(count)
    ]


def expand_variants(raw: Mapping[str, Any], selector: str = "") -> list[tuple[str, dict[str, Any]]]:
    """The variants ``selector`` picks from a recipe file, each with its selector: a plain recipe as itself, else
    every ``override_*`` then every ``zip_override_*`` variant in name order, or the one ``selector`` names.

    Raises:
        RecipeSourceError: ``selector`` names no variant of the file.
    """
    if "base" not in raw:
        if selector:
            raise RecipeSourceError(f"selector {selector} needs a recipe with a base")
        return [("", dict(raw))]
    if selector == "base":
        return [("base", copy.deepcopy(dict(raw["base"])))]
    zipped = _ZIP_SELECTOR.fullmatch(selector)
    if zipped and zipped[1] in raw:
        variants = _zip_variants(raw, zipped[1])
        if int(zipped[2]) >= len(variants):
            raise RecipeSourceError(f"{selector}: {zipped[1]} has {len(variants)} variants")
        return [variants[int(zipped[2])]]
    if selector:
        if not selector.startswith("override_") or selector not in raw:
            raise RecipeSourceError(f"the recipe has no variant {selector}")
        return [(selector, deep_merge(raw["base"], raw[selector]))]
    variants = [(key, deep_merge(raw["base"], raw[key])) for key in sorted(k for k in raw if k.startswith("override_"))]
    for key in sorted(k for k in raw if k.startswith("zip_override_")):
        variants += _zip_variants(raw, key)
    return variants


# --- matching -----------------------------------------------------------------------------------------------------


def _master_model(model: str, masters: Sequence[Mapping[str, Any]]) -> tuple[str, str | None]:
    """The id ``masters`` name ``model`` by, and a note when that is not ``model`` itself.

    A bare repo name (no ``org/``) the masters do not name is taken for the one ``org/<name>`` they name, matched
    case-insensitively; it stays itself when it names an org, the masters name it, or several orgs share the name.
    """
    if "/" in model or any(entry.get("model") == model for master in masters for entry in master.values()):
        return model, None
    hits = org_models_named(model, masters)
    if len(hits) == 1:
        return hits[0], f"InferenceX names {model} {hits[0]}"
    if hits:
        return model, f"InferenceX names several models {model}: {', '.join(hits)}; none is taken"
    return model, None


def _yaml(checkout: Checkout, path: str) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load((checkout.root / path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RecipeSourceError(f"InferenceX {path} cannot be read: {error}") from error
    if not isinstance(document, Mapping):
        raise RecipeSourceError(f"InferenceX {path} is not a mapping")
    return document


def _misfit(ask: RoleAsk, fit: RoleFit, gpus_per_node: int) -> str | None:
    """Why ``fit`` cannot run on the role's hosts, each worker on whole nodes of its own, or a topology the operator
    fixed differs."""
    nodes_per_worker = -(-fit.gpus // gpus_per_node)
    if fit.workers * nodes_per_worker != len(ask.hosts):
        return f"{fit.workers} workers of {fit.gpus} GPUs need {fit.workers * nodes_per_worker} nodes, not {len(ask.hosts)}"
    for name, asked in (("tp", ask.tp), ("ep", ask.ep), ("dp_attention", ask.dp_attention)):
        if asked is not None and asked != getattr(fit, name):
            return f"it runs {name} {getattr(fit, name)}, not {asked}"
    return None


def _roles_misfit(query: PdQuery, fits: Mapping[Role, RoleFit]) -> str | None:
    for role in PD_ROLES:
        misfit = _misfit(query.roles[role], fits[role], query.gpus_per_node)
        if misfit is not None:
            return f"{role.value}: {misfit}"
    return None


def _unported(recipe: Mapping[str, Any], query: PdQuery) -> str | None:
    """What of a recipe the PD launcher cannot reproduce on the query's cluster, or ``None``."""
    framework, served = query.framework, query.served_frontend.value
    roles = recipe.get("roles") or {}
    if set(roles) != {Role.PREFILL.value, Role.DECODE.value}:
        return f"it runs roles {sorted(roles)}, not prefill and decode"
    engines = {engine_type(recipe), *(str(role.get("engine")) for role in roles.values() if role.get("engine"))}
    if engines != {framework.value}:
        return f"it runs engine {sorted(engines)}, not {framework.value}"
    frontend = recipe.get("frontend") or {}
    if frontend.get("type") != served:
        return f"it serves through the {frontend.get('type')} frontend, not {served}"
    if frontend.get("enable_multiple_frontends"):
        return "it runs several frontends"
    for extra in ("setup_script", "services"):
        if recipe.get(extra):
            return f"it needs {extra}"
    return None


def _spellings(flags: Sequence[str]) -> tuple[str, ...]:
    """The canonical names of ``flags``, the form :func:`_arg` matches."""
    return tuple(canonical_flag_name(flag) for flag in flags)


def _arg(args: Mapping[str, Any], keys: Sequence[str]) -> Any:
    return next((args[key] for key in args if key.replace("_", "-") in keys), None)


def recipe_fit(role: Mapping[str, Any], framework: Framework) -> RoleFit:
    """The topology a recipe role states: its workers and GPUs, and TP, EP and DP attention from its args."""
    args = role.get("args") or {}
    gpus = int(role.get("gpus") or 0)
    spelling = TOPOLOGY_SPELLING[framework]
    tp = int(_arg(args, _spellings(spelling.tp)) or gpus)
    if spelling.ep_switch:
        ep = tp if _arg(args, _spellings(spelling.ep_switch)) else 1
    else:
        ep = int(_arg(args, _spellings(spelling.ep)) or 1)
    return RoleFit(int(role.get("workers") or 1), tp, ep, bool(_arg(args, (DP_ATTENTION,))), gpus or tp)


def _row_fits(row: Mapping[str, Any]) -> dict[Role, RoleFit]:
    fits = {}
    for role in PD_ROLES:
        side = row[role.value]
        tp = int(side["tp"])
        fits[role] = RoleFit(int(side["num-worker"]), tp, int(side.get("ep") or 1), bool(side.get("dp-attn")), tp)
    return fits


def _point(row: Mapping[str, Any], target: int) -> tuple[int | None, tuple[str, ...]]:
    """The row's concurrency at ``target``, else its nearest larger one with a warning; ``None`` without one."""
    larger = [conc for conc in concurrencies(row) if conc >= target]
    if not larger:
        return None, ()
    conc = min(larger)
    return conc, (() if conc == target else (f"InferenceX has no point at concurrency {target}; took {conc}",))


@dataclass(frozen=True)
class _MasterRow:
    """One ``search-space`` row of a master entry: ``<entry>[<index>]``, the entry, its scenario block and the row."""

    arm: str
    entry: Mapping[str, Any]
    block: Mapping[str, Any]
    row: Mapping[str, Any]


def _master_rows(query: PdQuery, master: Mapping[str, Any]) -> Iterator[_MasterRow]:
    """Every row of the query's scenario in the multi-node disaggregated entries of its model and framework."""
    frameworks = {query.framework.value, f"{query.framework.value}-disagg"}
    scenario = SCENARIO_OF[query.workload].value
    model, _ = _master_model(query.model_id, [master])
    for name, entry in master.items():
        if not (entry.get("multinode") and entry.get("disagg")):
            continue
        if entry.get("model") != model or entry.get("framework") not in frameworks:
            continue
        for block in (entry.get("scenarios") or {}).get(scenario) or ():
            for index, row in enumerate(block.get("search-space") or ()):
                yield _MasterRow(f"{name}[{index}]", entry, block, row)


def _row_candidate(query: PdQuery, checkout: Checkout, master_row: _MasterRow) -> Candidate | str:
    """The candidate a master row binds for ``query``, or why it cannot."""
    entry, block, row = master_row.entry, master_row.block, master_row.row
    if query.workload is Workload.FIXED:
        if (block.get("isl"), block.get("osl")) != (query.isl, query.osl):
            return f"it runs {block.get('isl')}/{block.get('osl')}, not {query.isl}/{query.osl}"
        if row.get("spec-decoding", UNSET) != UNSET:
            return f"it drafts with {row['spec-decoding']}, and a fixed workload runs no speculative decoding"
    fits = _row_fits(row)
    misfit = _roles_misfit(query, fits)
    if misfit is not None:
        return misfit
    conc, warnings = _point(row, query.target_concurrency)
    if conc is None:
        return f"it has no point at concurrency {query.target_concurrency} or above"
    file, _, selector = str(row["srt-recipe"]).partition(":")
    path = f"{MULTI_NODE_RECIPES}/{entry['srt-recipe-dir']}/{file}"
    recipe = expand_variants(_yaml(checkout, path), selector)[0][1]
    unported = _unported(recipe, query)
    if unported is not None:
        return unported
    return Candidate(
        entry=master_row.arm,
        path=path,
        locator=selector,
        recipe=recipe,
        fits=fits,
        transfer_backend=str(entry.get("kv-p2p-transfer") or ""),
        conc=conc,
        warnings=warnings,
    )


def master_candidates(query: PdQuery, checkout: Checkout) -> Candidates:
    """The master rows that serve ``query``, in entry then row order, on its board."""
    master = _yaml(checkout, MASTER)
    labels = _yaml(checkout, RUNNERS).get("labels") or {}
    _, note = _master_model(query.model_id, [master])
    found: list[Candidate] = []
    misses: list[str] = [note] if note else []
    rows = list(_master_rows(query, master))
    if not rows:
        scenario = SCENARIO_OF[query.workload].value
        misses.append(
            f"{MASTER}: no multi-node disaggregated entry serves {query.model_id} with {query.framework.value} "
            f"in its {scenario} scenario"
        )
    for master_row in rows:
        arm = master_row.arm
        try:
            board = runner_board(str(master_row.entry.get("runner")), labels)
        except RecipeSourceError as error:
            misses.append(f"{arm}: {error}")
            continue
        if board != query.board:
            misses.append(f"{arm}: it runs on {board}, not {query.board}")
            continue
        try:
            candidate = _row_candidate(query, checkout, master_row)
        except RecipeSourceError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as error:
            misses.append(f"{arm}: malformed row: {error!r}")
            continue
        if isinstance(candidate, str):
            misses.append(f"{arm}: {candidate}")
            continue
        found.append(_with_warning(candidate, note) if note else candidate)
    return Candidates(tuple(found), tuple(misses))


def _with_warning(candidate: Candidate, warning: str) -> Candidate:
    return dataclasses.replace(candidate, warnings=(*candidate.warnings, warning))


__all__ = [
    "CHECKOUT_DIRECTORIES",
    "MULTI_NODE_RECIPES",
    "Candidate",
    "Candidates",
    "RoleFit",
    "deep_merge",
    "expand_variants",
    "master_candidates",
    "recipe_fit",
]
