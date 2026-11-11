# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The vLLM recipes site as a recipe source, against responses recorded from it.

``20261106.json.gz`` holds the site's MI355X single-node TP renderings with presets dropped for Qwen3-8B (which the
site falls back to a CPU for), DeepSeek-R1-0528, DeepSeek-V4.1-Flash (whose recipe serves another checkpoint) and
Kimi-K3; the recipe documents of Kimi-K3 and DeepSeek-R1-0528; and Kimi-K3's AgentX renderings at concurrency 4, 50
and 64.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from functools import cache
from pathlib import Path

import pytest

from inference_suite.contracts import LaunchQuery, SourceKind, Topology, Workload
from inference_suite.recipe.model import ApiResponse, RecipeSourceError, SourceLaunch, SourceMiss
from inference_suite.recipe.vllm_recipes import (
    VllmRecipesSnapshot,
    recipe_url,
    rendering_url,
    vllm_recipes_launches,
    vllm_recipes_requests,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "vllm_recipes" / "20261106.json.gz"
_R1 = "deepseek-ai/DeepSeek-R1-0528"
_KIMI = "moonshotai/Kimi-K3"


@cache
def _recorded() -> VllmRecipesSnapshot:
    return VllmRecipesSnapshot.from_document(json.loads(gzip.decompress(_FIXTURE.read_bytes())))


def _fixed(model: str, **topology) -> LaunchQuery:
    return LaunchQuery(model, "vllm", "mi355x", Workload.FIXED, (64,), isl=1024, osl=1024, **topology)


def _agentx(model: str, conc: int) -> LaunchQuery:
    return LaunchQuery(model, "vllm", "mi355x", Workload.AGENTX, (conc,))


def _launches(query: LaunchQuery, snapshot: VllmRecipesSnapshot | None = None) -> tuple[SourceLaunch, ...]:
    found = vllm_recipes_launches(query, snapshot or _recorded())
    assert not isinstance(found, SourceMiss), found
    return found


def _miss(query: LaunchQuery, snapshot: VllmRecipesSnapshot | None = None) -> str:
    found = vllm_recipes_launches(query, snapshot or _recorded())
    assert isinstance(found, SourceMiss), found
    assert found.kind is SourceKind.VLLM_RECIPES
    return found.reason


def _edited(url: str, edit) -> VllmRecipesSnapshot:
    """The recorded snapshot with the rendering at ``url`` changed by ``edit``."""
    responses = dict(_recorded().responses)
    rendered = json.loads(responses[url].body)
    edit(rendered)
    responses[url] = dataclasses.replace(responses[url], body=json.dumps(rendered))
    return VllmRecipesSnapshot(responses)


def test_fixed_r1_0528_takes_the_baseline_rendering_traced_to_its_response():
    (launch,) = _launches(_fixed(_R1))

    url = rendering_url(_R1, "mi355x", "baseline")
    response = _recorded().response(url)
    assert url == f"https://recipes.vllm.ai/{_R1}.json?hardware=mi355x&strategy=single_node_tp&preset=baseline"
    assert launch.source.kind is SourceKind.VLLM_RECIPES
    assert (launch.source.repo, launch.source.commit) == (None, None)
    assert launch.source.api.url == url
    assert launch.source.api.response_sha256 == hashlib.sha256(response.body.encode()).hexdigest()
    assert launch.source.api.fetched_unix == response.fetched_unix
    assert (launch.source.entry, launch.source.locator) == (
        "deepseek-ai/DeepSeek-R1",
        "variant r1_0528, preset baseline",
    )
    assert (launch.framework, launch.entrypoint) == ("vllm", ("vllm", "serve"))
    assert launch.args == (
        *("--trust-remote-code", "--tensor-parallel-size", "8", "--enable-auto-tool-choice"),
        *("--tool-call-parser", "deepseek_v3", "--chat-template", "examples/tool_chat_template_deepseekr1.jinja"),
        *("--reasoning-parser", "deepseek_r1"),
    )
    assert launch.topology == Topology(tp=8, ep=1)
    assert (launch.conc, launch.exact, launch.kv_offloading) == (64, True, "none")
    assert launch.image == "vllm/vllm-openai-rocm:latest"
    assert launch.env["VLLM_ROCM_USE_AITER"] == "1"
    assert launch.golden is None


def test_a_rendering_for_another_board_is_a_miss():
    reason = _miss(_fixed("Qwen/Qwen3-8B"))

    assert "the site serves it on xeon6, not mi355x" in reason


def test_a_rendering_the_site_adjusted_is_a_miss_even_on_the_board_asked_for():
    url = rendering_url(_R1, "mi355x", "baseline")
    adjusted = {"param": "variant", "requested": "default", "used": "r1_0528", "reason": "not available"}
    snapshot = _edited(url, lambda rendered: rendered["selection"].update(adjusted=[adjusted]))

    assert "the site adjusted variant default -> r1_0528" in _miss(_fixed(_R1), snapshot)


def test_a_rendering_of_another_checkpoint_is_a_miss():
    reason = _miss(_fixed("deepseek-ai/DeepSeek-V4.1-Flash"))

    assert "serves checkpoint deepseek-ai/DeepSeek-V4.1-Flash-1017, not deepseek-ai/DeepSeek-V4.1-Flash" in reason


def test_a_fixed_workload_refuses_a_rendering_that_drafts_tokens():
    url = rendering_url(_R1, "mi355x", "baseline")
    draft = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']
    snapshot = _edited(url, lambda rendered: rendered["argv"].extend(draft))

    assert "a fixed workload runs no speculative decoding" in _miss(_fixed(_R1), snapshot)


def test_runner_owned_flags_are_stripped_with_their_values():
    url = rendering_url(_R1, "mi355x", "baseline")
    owned = ["--port", "8000", "--host=0.0.0.0", "--served-model-name", "r1"]
    snapshot = _edited(
        url, lambda rendered: rendered.update(argv=[*rendered["argv"][:3], *owned, *rendered["argv"][3:]])
    )

    (launch,) = _launches(_fixed(_R1), snapshot)

    assert launch.stripped == ("--port", "--host", "--served-model-name")
    assert launch.args[:3] == ("--trust-remote-code", "--tensor-parallel-size", "8")


def test_an_operator_topology_the_rendering_does_not_run_is_a_miss():
    assert "it runs tp 8, not 4" in _miss(_fixed(_R1, tp=4))


def test_agentx_reads_each_agentic_preset_at_the_point_for_the_concurrency():
    launches = _launches(_agentx(_KIMI, 4))

    assert [launch.source.locator for launch in launches] == [
        "variant default, preset mi355x_agentx",
        "variant default, preset mi355x_agentx_cells, point o4",
        "variant default, preset mi355x_agentx_cells, point r4",
    ]
    assert [launch.kv_offloading for launch in launches] == ["none", "simple", "none"]
    assert all((launch.conc, launch.exact, launch.warnings) == (4, True, ()) for launch in launches)
    assert launches[2].source.api.url == rendering_url(_KIMI, "mi355x", "mi355x_agentx_cells", "r4")
    assert launches[2].args[launches[2].args.index("--max-num-seqs") :][:2] == ("--max-num-seqs", "2")


def test_agentx_takes_the_nearest_larger_point_and_warns():
    _, cells = _launches(_agentx(_KIMI, 50))

    assert cells.source.locator == "variant default, preset mi355x_agentx_cells, point o52"
    assert (cells.conc, cells.exact) == (52, False)
    assert cells.warnings == ("preset mi355x_agentx_cells has no point at concurrency 50; took 52",)


def test_agentx_skips_a_preset_with_no_point_at_or_above_the_concurrency():
    (launch,) = _launches(_agentx(_KIMI, 64))

    assert launch.source.locator == "variant default, preset mi355x_agentx"


def test_agentx_without_an_agentic_preset_for_the_board_is_a_miss():
    assert _miss(_agentx(_R1, 8)) == "the recipe has no Agentic single_node_tp preset for mi355x"


def test_a_model_the_site_has_no_recipe_for_is_a_miss_naming_the_answer():
    url = recipe_url("foo/bar")
    snapshot = VllmRecipesSnapshot({url: ApiResponse(url, 404, "No recipe for foo/bar\n", 1_762_400_000)})

    assert _miss(_agentx("foo/bar", 8), snapshot) == f"{url} answered 404: No recipe for foo/bar"


def test_requests_read_the_recipe_document_before_the_renderings_it_names():
    query = _agentx(_KIMI, 4)
    renderings = (
        rendering_url(_KIMI, "mi355x", "mi355x_agentx"),
        rendering_url(_KIMI, "mi355x", "mi355x_agentx_cells", "o4"),
        rendering_url(_KIMI, "mi355x", "mi355x_agentx_cells", "r4"),
    )
    document = recipe_url(_KIMI)

    assert vllm_recipes_requests(query, VllmRecipesSnapshot({})) == (document,)
    assert vllm_recipes_requests(query, VllmRecipesSnapshot({document: _recorded().response(document)})) == renderings
    assert vllm_recipes_requests(query, _recorded()) == ()
    assert vllm_recipes_requests(_fixed(_R1), VllmRecipesSnapshot({})) == (rendering_url(_R1, "mi355x", "baseline"),)
    assert vllm_recipes_requests(dataclasses.replace(_fixed(_R1), framework="sglang"), VllmRecipesSnapshot({})) == ()


def test_a_snapshot_missing_a_url_the_query_reads_is_an_error():
    with pytest.raises(RecipeSourceError, match="does not carry"):
        vllm_recipes_launches(_fixed("Qwen/Qwen3-0.6B"), _recorded())


def test_another_framework_is_a_miss():
    query = dataclasses.replace(_fixed(_R1), framework="sglang")

    assert _miss(query) == "the vLLM recipes launch vllm, not sglang"


def test_the_snapshot_round_trips_and_keys_each_response_by_its_url():
    assert VllmRecipesSnapshot.from_document(_recorded().to_document()) == _recorded()

    response = ApiResponse("https://recipes.vllm.ai/a/b.json", 200, "{}", 0)
    with pytest.raises(RecipeSourceError, match="is not the one fetched from it"):
        VllmRecipesSnapshot({"https://recipes.vllm.ai/c/d.json": response})
