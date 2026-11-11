# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ATOM as a recipe source, against a snapshot of ATOM main.

``b90ae9990.json.gz`` holds, at ATOM b90ae9990, the benchmark catalog ``models.json``, the agentic catalog
``models_agentic.json`` and the recipe pages of GPT-OSS, Qwen3-Next, DeepSeek-R1, Kimi-K2-Thinking, DeepSeek-V4 and
GLM-5.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import cache
from pathlib import Path

import pytest

from inference_suite.contracts import LaunchQuery, SourceKind, Topology, Workload
from inference_suite.recipe.atom_catalog import ATOM_REPO, CATALOG, atom_catalog_launches
from inference_suite.recipe.model import ATOM_ENTRYPOINT, RecipeSourceError, RepoBundle, SourceLaunch, SourceMiss

_FIXTURE = Path(__file__).parent / "fixtures" / "atom" / "b90ae9990.json.gz"
_COMMIT = "b90ae9990abd42ed0424c0255a34a74ef464a060"
_AGENTIC_CATALOG = ".github/benchmark/models_agentic.json"
_DSV4_PRO = "deepseek-ai/DeepSeek-V4-Pro"
_DSV4_PRO_BASE = ("--kv_cache_dtype", "fp8", "-tp", "8", "--index_cache_dtype", "fp4", "--no-enable_prefix_caching")


@cache
def _main() -> RepoBundle:
    return RepoBundle.from_document(json.loads(gzip.decompress(_FIXTURE.read_bytes())))


def _with(files: dict[str, str | None]) -> RepoBundle:
    merged = {**_main().files, **files}
    return RepoBundle(ATOM_REPO, _COMMIT, {path: text for path, text in merged.items() if text is not None})


def _fixed(model: str, *, board: str = "mi355x", concurrencies=(64,), framework: str = "atom", **topology):
    return LaunchQuery(model, framework, board, Workload.FIXED, tuple(concurrencies), isl=8192, osl=1024, **topology)


def _launches(query: LaunchQuery, bundle: RepoBundle | None = None) -> tuple[SourceLaunch, ...]:
    found = atom_catalog_launches(query, bundle or _main())
    assert not isinstance(found, SourceMiss), found
    return found


def _miss(query: LaunchQuery, bundle: RepoBundle | None = None) -> str:
    found = atom_catalog_launches(query, bundle or _main())
    assert isinstance(found, SourceMiss), found
    assert found.kind is SourceKind.ATOM_CATALOG
    return found.reason


def _by_locator(launches: tuple[SourceLaunch, ...]) -> dict[str, SourceLaunch]:
    return {launch.source.locator: launch for launch in launches}


def test_catalog_args_compose_in_catalog_order():
    (kimi,) = _launches(_fixed("moonshotai/Kimi-K3"))
    (gpt_oss,) = _launches(_fixed("openai/gpt-oss-120b"))
    dpa_tbo = _by_locator(_launches(_fixed(_DSV4_PRO, concurrencies=(1024,))))["models[0].variants[3]"]

    assert kimi.args == (
        *("--kv_cache_dtype", "fp8", "-tp", "8", "--trust-remote-code"),
        *("--gpu-memory-utilization", "0.93", "--block-size", "128", "--no-enable_prefix_caching"),
    )
    assert gpt_oss.args == ("--kv_cache_dtype", "fp8", "--gpu-memory-utilization", "0.9")
    assert gpt_oss.topology == Topology(tp=1)
    assert dpa_tbo.args == (*_DSV4_PRO_BASE, "--enable-dp-attention", "--enable-tbo")
    assert dpa_tbo.env == {
        "AITER_BF16_FP8_MOE_BOUND": "0",
        "ATOM_MOE_GU_ITLV": "1",
        "GPU_MAX_HW_QUEUES": "5",
        "ATOM_NUMA_BIND": "1",
    }


def test_catalog_launch_names_its_source_and_tuned_board():
    (kimi,) = _launches(_fixed("moonshotai/Kimi-K3"))

    assert kimi.source.kind is SourceKind.ATOM_CATALOG
    assert (kimi.source.repo, kimi.source.commit, kimi.source.path) == (ATOM_REPO, _COMMIT, CATALOG)
    assert (kimi.source.entry, kimi.source.locator) == ("Kimi-K3", "models[5].variants[0]")
    assert kimi.source.files == {CATALOG: hashlib.sha256(_main().text(CATALOG).encode()).hexdigest()}
    assert (kimi.framework, kimi.entrypoint, kimi.image) == ("atom", ATOM_ENTRYPOINT, None)
    assert (kimi.conc, kimi.exact) == (64, True)
    assert kimi.warnings == ("ATOM tuned Kimi-K3 on mi350x, not mi355x",)
    assert _launches(_fixed("moonshotai/Kimi-K3", board="mi350x"))[0].warnings == ()


def test_catalog_refuses_a_board_of_another_architecture():
    reason = _miss(_fixed("moonshotai/Kimi-K3", board="mi300x"))

    assert "it is tuned on mi350x, not mi300x" in reason


def test_several_fitting_variants_warn_and_name_each_other():
    launches = _launches(_fixed(_DSV4_PRO, concurrencies=(256,)))

    locators = [launch.source.locator for launch in launches]
    assert locators == [
        "models[0].variants[0]",
        "models[0].variants[1]",
        "models[0].variants[2]",
        "models[0].variants[3]",
        "models[1].variants[0]",
        "models[1].variants[1]",
    ]
    for launch in launches:
        assert launch.warnings[-1].startswith(f"ATOM holds 6 launches of {_DSV4_PRO} that fit: ")
        assert all(f"{CATALOG} {locator}" in launch.warnings[-1] for locator in locators)
    eplb = _by_locator(launches)["models[1].variants[0]"]
    assert (eplb.conc, eplb.exact) == (512, False)
    assert "DeepSeek-V4-Pro EPLB r0 MegaMoE MegaMoE has no point at concurrency 256; took 512" in eplb.warnings
    assert eplb.topology == Topology(tp=8, ep=8)


def test_harness_env_is_taken_out_of_the_server_env():
    (eplb,) = _launches(_fixed(_DSV4_PRO, concurrencies=(4096,)))

    assert eplb.env == {
        "AITER_BF16_FP8_MOE_BOUND": "0",
        "ATOM_MOE_GU_ITLV": "1",
        "AITER_LOG_LEVEL": "WARNING",
        "MORI_SHMEM_HEAP_SIZE": "16G",
    }
    assert eplb.stripped == ("ATOM_BENCHMARK_STUCK_POLLS",)
    assert eplb.args[-2:] == ("--gpu-memory-utilization", "0.85")


def test_harness_env_names_are_all_filtered():
    catalog = json.loads(_main().text(CATALOG))
    catalog["models"][5]["env_vars"] = "HIP_VISIBLE_DEVICES=0,1\nPRECISION=fp8\nAIPERF_X=1\nBENCHMARK_Y=2\nKEEP=1"

    (kimi,) = _launches(_fixed("moonshotai/Kimi-K3"), _with({CATALOG: json.dumps(catalog)}))

    assert kimi.env == {"KEEP": "1"}
    assert kimi.stripped == ("HIP_VISIBLE_DEVICES", "PRECISION", "AIPERF_X", "BENCHMARK_Y")


def test_fixed_skips_variants_that_draft_tokens():
    reason = _miss(_fixed("deepseek-ai/DeepSeek-V4.1-Flash", concurrencies=(8,)))

    assert (
        reason
        == f"{CATALOG} models[2].variants[0]: it drafts tokens, and a fixed workload runs no speculative decoding"
    )


def test_a_concurrency_past_every_band_misses():
    reason = _miss(_fixed("moonshotai/Kimi-K3", concurrencies=(512,)))

    assert "Kimi-K3 is benchmarked at no concurrency of 512 or more" in reason


def test_operator_topology_must_match():
    assert "it runs tp 8, not 4" in _miss(_fixed("moonshotai/Kimi-K3", tp=4))


def test_agentx_is_refused_without_reading_the_agentic_catalog():
    query = LaunchQuery("deepseek-ai/DeepSeek-V4.1-Flash", "atom", "mi355x", Workload.AGENTX, (8,))

    assert "agentic catalogs are not read" in _miss(query)
    assert "agentic catalogs are not read" in _miss(query, _with({_AGENTIC_CATALOG: "not json"}))


def test_the_agentic_catalog_never_feeds_a_fixed_launch():
    query = _fixed(_DSV4_PRO, concurrencies=(256,))

    assert _launches(query, _with({_AGENTIC_CATALOG: "not json"})) == _launches(query)
    assert _launches(query, _with({_AGENTIC_CATALOG: None})) == _launches(query)


def test_other_frameworks_miss():
    assert "not sglang" in _miss(_fixed(_DSV4_PRO, framework="sglang"))


def test_page_literal_block_when_the_catalog_has_no_model():
    model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    (launch,) = _launches(_fixed(model, concurrencies=(8,)))

    page = "recipes/Qwen3-Next.md"
    assert (launch.source.path, launch.source.locator, launch.source.entry) == (
        page,
        "block 0",
        "Serving on 8xMI355 GPUs",
    )
    assert launch.source.files == {page: hashlib.sha256(_main().text(page).encode()).hexdigest()}
    assert launch.args == ("-tp", "8")
    assert launch.stripped == ("--model",)
    assert launch.image == "rocm/atom:latest"
    assert (launch.conc, launch.exact, launch.topology) == (8, True, Topology(tp=8))
    assert launch.warnings == (f"{page} block 0 names no concurrency; it is taken as tuned for 8",)


def test_page_blocks_take_exports_without_harness_env_and_the_heading_board():
    launches = _launches(_fixed("amd/Kimi-K2-Thinking-MXFP4", concurrencies=(8,)))

    assert [launch.topology.tp for launch in launches] == [4, 8]
    assert all(launch.stripped == ("--model", "HIP_VISIBLE_DEVICES") for launch in launches)
    assert all("ATOM holds 2 launches" in launch.warnings[-1] for launch in launches)
    (tp4,) = _launches(_fixed("amd/Kimi-K2-Thinking-MXFP4", concurrencies=(8,), tp=4))
    assert tp4.source.entry == "Serving on 4xMI355 GPUs"
    assert tp4.args == ("--trust-remote-code", "-tp", "4", "--kv_cache_dtype", "fp8")


def test_page_inline_assignments_become_env():
    launches = _launches(_fixed(_DSV4_PRO, concurrencies=(64,)), _with({CATALOG: '{"models": []}'}))

    first = launches[0]
    assert first.source.path == "recipes/DeepSeek-V4.md"
    assert first.env == {"AITER_BF16_FP8_MOE_BOUND": "0", "ATOM_MOE_GU_ITLV": "1", "AITER_LOG_LEVEL": "WARNING"}


def test_page_board_from_a_heading_naming_several():
    (launch,) = _launches(_fixed("deepseek-ai/DeepSeek-R1-0528", board="mi300x", concurrencies=(8,)))

    assert launch.source.entry == "FP8 on 8xMI300X/MI355X GPUs (TP8 + FP8 KV Cache)"
    assert not any(warning.startswith("ATOM tuned") for warning in launch.warnings)


def test_dynamic_page_blocks_are_refused():
    reason = _miss(_fixed("amd/GLM-5.2-MXFP4"))

    assert "recipes/GLM-5.md block 0: it is not a literal launch: it expands shell variables or commands" in reason
    assert "lm_eval" not in reason


@pytest.mark.parametrize(
    ("block", "why"),
    [
        ("python -m atom.entrypoints.openai_server --model m/x -tp $((TP))", "expands shell variables"),
        ("case 1 in 1) python -m atom.entrypoints.openai_server --model m/x;; esac", "shell operator"),
        ("python -m atom.entrypoints.openai_server --model m/x > log", "shell operator '>'"),
        ("rm -rf /tmp/x\npython -m atom.entrypoints.openai_server --model m/x", "it runs 'rm -rf /tmp/x'"),
        ("if true\nthen python -m atom.entrypoints.openai_server --model m/x\nfi", "it runs 'if true'"),
    ],
)
def test_page_blocks_beyond_exports_and_the_server_are_refused(block, why):
    page = f"# M\n\n## Serving on 8xMI355X\n\n```bash\n{block}\n```\n"

    reason = _miss(_fixed("m/x"), _with({"recipes/M.md": page}))

    assert reason.startswith("recipes/M.md block 0: it is not a literal launch: ")
    assert why in reason


@pytest.mark.parametrize(
    ("flags", "stage"),
    [
        ("-pp 2", "pp"),
        ("--pipeline-parallel-size 2", "pipeline-parallel-size"),
        ("-pcp 2", "pcp"),
        ("--prefill-context-parallel-size 2", "prefill-context-parallel-size"),
        ("-dp 2", "dp"),
        ("--data-parallel-size 2", "data-parallel-size"),
    ],
)
def test_ranks_beside_tensor_parallelism_are_refused(flags, stage):
    block = f"python -m atom.entrypoints.openai_server --model m/x -tp 4 {flags}"
    page = f"# M\n\n## Serving on 8xMI355X\n\n```bash\n{block}\n```\n"

    reason = _miss(_fixed("m/x"), _with({"recipes/M.md": page}))

    assert reason == f"recipes/M.md block 0: it runs {stage} beside tensor parallelism"


def test_pages_are_read_only_when_the_catalog_gives_nothing():
    page = "## On 8xMI355X\n\n```bash\npython -m atom.entrypoints.openai_server --model moonshotai/Kimi-K3 -tp 8\n```\n"
    bundle = _with({"recipes/Kimi-K3.md": page, "recipes/sub/Kimi-K3.md": page})

    assert [launch.source.path for launch in _launches(_fixed("moonshotai/Kimi-K3"), bundle)] == [CATALOG]
    bundle = _with({CATALOG: '{"models": []}', "recipes/sub/Kimi-K3.md": page})
    assert "hold no launch of moonshotai/Kimi-K3" in _miss(_fixed("moonshotai/Kimi-K3"), bundle)


def test_a_page_block_under_no_board_heading_misses():
    page = "## Serving\n\n```bash\npython -m atom.entrypoints.openai_server --model m/x -tp 8\n```\n"

    assert (
        _miss(_fixed("m/x"), _with({"recipes/M.md": page})) == "recipes/M.md block 0: no heading above it names a board"
    )


def test_a_malformed_catalog_is_a_snapshot_error():
    with pytest.raises(RecipeSourceError, match="no JSON"):
        atom_catalog_launches(_fixed("m/x"), _with({CATALOG: "{"}))
