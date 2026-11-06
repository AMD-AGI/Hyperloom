# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The vLLM recipes site as a recipe source: the single-node TP launches it renders for a model on a board.

The site has no commit, so a launch is traced to the response it was read from (URL, body sha256, fetch time). A
fixed workload reads the rendering with tuned presets dropped (``preset=baseline``). An AgentX workload reads the
model's recipe document for its Agentic presets on the board and the rendering of each at the point whose
concurrency serves the query: the one asked for, else the nearest larger.

The site falls back silently, so two checks are hard: a rendering must be for the board asked for, with nothing
adjusted, and must serve the model asked for, not another checkpoint of the same recipe. The site pairs no golden
curve with a launch, so its launches carry no golden acceptance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from hyperloom_common.contract_fields import ContractFields
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value, speculative_config
from inference_suite.contracts.base_launch import LaunchQuery, LaunchSource, SourceKind, Topology, Workload
from inference_suite.recipe.model import (
    VLLM_ENTRYPOINT,
    VLLM_RUNNER_OWNED,
    ApiResponse,
    RecipeSourceError,
    SourceLaunch,
    SourceMiss,
)

VLLM_RECIPES_SITE = "https://recipes.vllm.ai"
STRATEGY = "single_node_tp"
BASELINE_PRESET = "baseline"

_FIELDS = ContractFields(RecipeSourceError)
_FRAMEWORK = "vllm"
_OK = 200
_AGENTIC = re.compile(r"^agentic\b", re.IGNORECASE)
_TP = frozenset({"tensor-parallel-size", "tp"})
_DP = frozenset({"data-parallel-size", "dp"})
_EXPERT_PARALLEL = frozenset({"enable-expert-parallel"})
_NO_OFFLOAD = frozenset({"off", "none"})


def recipe_url(model_id: str) -> str:
    """The URL of the recipe document serving ``model_id``: its variants, presets and points."""
    return f"{VLLM_RECIPES_SITE}/{quote(model_id, safe='/')}.json"


def rendering_url(model_id: str, board: str, preset: str, point: str | None = None) -> str:
    """The URL of the launch the site renders for ``model_id`` on ``board`` under ``preset`` at ``point``."""
    query = {"hardware": board, "strategy": STRATEGY, "preset": preset}
    if point is not None:
        query["point"] = point
    return f"{recipe_url(model_id)}?{urlencode(query)}"


@dataclass(frozen=True)
class VllmRecipesSnapshot:
    """The site's responses recorded for a query, keyed by the URL each was fetched from."""

    responses: Mapping[str, ApiResponse]

    def __post_init__(self) -> None:
        for url, response in _FIELDS.mapping(self.responses, "the vLLM recipes responses").items():
            if not isinstance(response, ApiResponse) or response.url != url:
                raise RecipeSourceError(f"the response recorded under {url} is not the one fetched from it")

    def response(self, url: str) -> ApiResponse:
        if url not in self.responses:
            raise RecipeSourceError(f"the vLLM recipes snapshot does not carry {url}")
        return self.responses[url]

    def to_document(self) -> dict[str, object]:
        return {"responses": [self.responses[url].to_document() for url in sorted(self.responses)]}

    @classmethod
    def from_document(cls, document: object) -> VllmRecipesSnapshot:
        fields = _FIELDS.known(document, ("responses",), "a vLLM recipes snapshot")
        responses = map(ApiResponse.from_document, _FIELDS.items(fields.get("responses", ()), "the responses"))
        return cls({response.url: response for response in responses})


@dataclass(frozen=True)
class _Rendering:
    """A rendering the query reads: its preset and point, and the concurrency that point is tuned for."""

    url: str
    preset: str
    point: str | None
    conc: int
    exact: bool


def _miss(reason: str) -> SourceMiss:
    return SourceMiss(SourceKind.VLLM_RECIPES, reason)


def _refusal(response: ApiResponse) -> str | None:
    """Why the site answered ``response`` with no recipe, or ``None`` when it answered with one."""
    if response.status == _OK:
        return None
    return f"{response.url} answered {response.status}: {response.body.strip()[:200]}"


def _point_renderings(query: LaunchQuery, name: str, points: Mapping[str, Any], target: int) -> list[_Rendering]:
    """The renderings of preset ``name`` at its points tuned for ``target``, else for the nearest larger concurrency."""
    tuned = {
        key: _FIELDS.integer(
            _FIELDS.mapping(point, f"point {key} of {name}").get("concurrency"),
            f"the concurrency of point {key} of {name}",
            minimum=1,
        )
        for key, point in points.items()
    }
    fitting = [conc for conc in tuned.values() if conc >= target]
    if not fitting:
        return []
    conc = min(fitting)
    return [
        _Rendering(rendering_url(query.model_id, query.board, name, key), name, key, conc, exact=conc == target)
        for key in sorted(key for key, tuned_conc in tuned.items() if tuned_conc == conc)
    ]


def _agentx_renderings(query: LaunchQuery, recipe: Mapping[str, Any]) -> list[_Rendering] | SourceMiss:
    """The renderings of the recipe's Agentic single-node TP presets on the query's board."""
    target = max(query.concurrencies)
    presets = _FIELDS.mapping(recipe.get("presets") or {}, "the recipe's presets")
    agentic = {
        name: preset
        for name, preset in sorted(presets.items())
        if _AGENTIC.match(str(preset.get("workload", "")))
        and (preset.get("hardware"), preset.get("strategy")) == (query.board, STRATEGY)
    }
    if not agentic:
        return _miss(f"the recipe has no Agentic {STRATEGY} preset for {query.board}")
    renderings: list[_Rendering] = []
    for name, preset in agentic.items():
        points = preset.get("points")
        if not points:
            renderings.append(_Rendering(rendering_url(query.model_id, query.board, name), name, None, target, True))
        else:
            renderings += _point_renderings(query, name, _FIELDS.mapping(points, f"the points of {name}"), target)
    if not renderings:
        return _miss(f"no Agentic preset for {query.board} has a point at concurrency {target} or above")
    return renderings


def _renderings(query: LaunchQuery, snapshot: VllmRecipesSnapshot) -> list[_Rendering] | SourceMiss:
    if query.workload is Workload.FIXED:
        url = rendering_url(query.model_id, query.board, BASELINE_PRESET)
        return [_Rendering(url, BASELINE_PRESET, None, max(query.concurrencies), exact=True)]
    response = snapshot.response(recipe_url(query.model_id))
    refusal = _refusal(response)
    return _miss(refusal) if refusal is not None else _agentx_renderings(query, response.json())


def vllm_recipes_requests(query: LaunchQuery, snapshot: VllmRecipesSnapshot) -> tuple[str, ...]:
    """The URLs ``query`` reads that ``snapshot`` does not yet carry; fetch them and ask again until none are left.

    An AgentX query reads the recipe document first, since which renderings it reads depends on its presets.
    """
    if query.framework != _FRAMEWORK:
        return ()
    if query.workload is Workload.AGENTX and recipe_url(query.model_id) not in snapshot.responses:
        return (recipe_url(query.model_id),)
    renderings = _renderings(query, snapshot)
    if isinstance(renderings, SourceMiss):
        return ()
    return tuple(rendering.url for rendering in renderings if rendering.url not in snapshot.responses)


def _adjusted(selection: Mapping[str, Any]) -> str | None:
    adjusted = selection.get("adjusted") or []
    if not adjusted:
        return None
    changes = [f"{change.get('param')} {change.get('requested')} -> {change.get('used')}" for change in adjusted]
    return f"the site adjusted {', '.join(changes)}"


def _served_model(rendered: Mapping[str, Any]) -> str:
    argv = _FIELDS.strings(rendered.get("argv"), "the rendering's argv")
    if tuple(argv[: len(VLLM_ENTRYPOINT)]) != VLLM_ENTRYPOINT or len(argv) <= len(VLLM_ENTRYPOINT):
        raise RecipeSourceError(f"its argv {list(argv)} does not start a vLLM server for a model")
    return argv[len(VLLM_ENTRYPOINT)]


def _mismatch(rendered: Mapping[str, Any], query: LaunchQuery) -> str | None:
    """Why the rendering is not the launch ``query`` asked for: another board, another checkpoint or more nodes."""
    selection = _FIELDS.mapping(rendered.get("selection") or {}, "the rendering's selection")
    if rendered.get("hardware") != query.board:
        return f"the site serves it on {rendered.get('hardware')}, not {query.board}"
    adjusted = _adjusted(selection)
    if adjusted is not None:
        return adjusted
    if (rendered.get("deploy_type"), rendered.get("node_count")) != ("single_node", 1):
        return f"it deploys {rendered.get('deploy_type')} on {rendered.get('node_count')} nodes"
    served = _served_model(rendered)
    if served != query.model_id:
        return f"the recipe serves checkpoint {served}, not {query.model_id}"
    return None


def _server_args(argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The flags after the served model with the runner-owned ones taken out, and the owned flags taken out."""
    args: list[str] = []
    stripped: list[str] = []
    owned = False
    for token in argv[len(VLLM_ENTRYPOINT) + 1 :]:
        if token.startswith("-") and not token[1:2].isdigit():
            flag = token.partition("=")[0]
            owned = canonical_flag_name(flag) in VLLM_RUNNER_OWNED
            stripped += [flag] if owned else []
        if not owned:
            args.append(token)
    return tuple(args), tuple(stripped)


def _topology(args: Sequence[str]) -> Topology:
    occurrences = argv_flag_occurrences(args)
    tp, dp = last_alias_value(occurrences, _TP), last_alias_value(occurrences, _DP)
    try:
        tp_size, dp_size = int(tp or 1), int(dp or 1)
    except ValueError as error:
        raise RecipeSourceError(f"its tp {tp!r} or dp {dp!r} is not a size") from error
    expert_parallel = last_alias_value(occurrences, _EXPERT_PARALLEL) is not None
    return Topology(tp=tp_size, ep=tp_size * dp_size if expert_parallel else 1)


def _misfit(args: Sequence[str], topology: Topology, query: LaunchQuery) -> str | None:
    if query.workload is Workload.FIXED and speculative_config(argv_flag_occurrences(args)):
        return "it drafts tokens, and a fixed workload runs no speculative decoding"
    for name, asked in (("tp", query.tp), ("ep", query.ep)):
        if asked is not None and asked != getattr(topology, name):
            return f"it runs {name} {getattr(topology, name)}, not {asked}"
    return None


def _launch(rendering: _Rendering, response: ApiResponse, query: LaunchQuery) -> SourceLaunch | str:
    """The launch ``response`` renders for ``query``, or why it gives none."""
    refusal = _refusal(response)
    if refusal is not None:
        return refusal
    rendered = response.json()
    mismatch = _mismatch(rendered, query)
    if mismatch is not None:
        return f"{response.url}: {mismatch}"
    args, stripped = _server_args(_FIELDS.strings(rendered.get("argv"), "the rendering's argv"))
    topology = _topology(args)
    misfit = _misfit(args, topology, query)
    if misfit is not None:
        return f"{response.url}: {misfit}"
    selection = _FIELDS.mapping(rendered.get("selection") or {}, "the rendering's selection")
    kv_offload = selection.get("kv_offload")
    locator = f"variant {rendered.get('variant')}, preset {rendering.preset}"
    warnings: tuple[str, ...] = ()
    if not rendering.exact:
        target = max(query.concurrencies)
        warnings = (f"preset {rendering.preset} has no point at concurrency {target}; took {rendering.conc}",)
    return SourceLaunch(
        source=LaunchSource(
            kind=SourceKind.VLLM_RECIPES,
            api=response.provenance(),
            entry=_FIELDS.text(rendered.get("hf_id"), "the rendering's recipe"),
            locator=locator if rendering.point is None else f"{locator}, point {rendering.point}",
            strategy=STRATEGY,
        ),
        framework=_FRAMEWORK,
        entrypoint=VLLM_ENTRYPOINT,
        topology=topology,
        conc=rendering.conc,
        exact=rendering.exact,
        image=_FIELDS.optional_text(rendered.get("docker_image"), "the rendering's image"),
        args=args,
        env=dict(_FIELDS.string_map(rendered.get("env") or {}, "the rendering's env")),
        kv_offloading="none" if kv_offload is None or kv_offload in _NO_OFFLOAD else str(kv_offload),
        stripped=stripped,
        warnings=warnings,
    )


def vllm_recipes_launches(query: LaunchQuery, snapshot: VllmRecipesSnapshot) -> tuple[SourceLaunch, ...] | SourceMiss:
    """The launches the site renders for ``query`` at ``max(query.concurrencies)``, one per preset point read."""
    if query.framework != _FRAMEWORK:
        return _miss(f"the vLLM recipes launch vllm, not {query.framework}")
    renderings = _renderings(query, snapshot)
    if isinstance(renderings, SourceMiss):
        return renderings
    found = [_launch(rendering, snapshot.response(rendering.url), query) for rendering in renderings]
    launches = tuple(item for item in found if isinstance(item, SourceLaunch))
    return launches or _miss("; ".join(item for item in found if isinstance(item, str)))


__all__ = [
    "BASELINE_PRESET",
    "STRATEGY",
    "VLLM_RECIPES_SITE",
    "VllmRecipesSnapshot",
    "recipe_url",
    "rendering_url",
    "vllm_recipes_launches",
    "vllm_recipes_requests",
]
