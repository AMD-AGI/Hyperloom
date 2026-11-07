# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolving a base launch from recorded source snapshots: priority, point and topology choice, the golden layer,
the default table and the recipe image's framework version.

Every test runs with subprocesses and sockets refused, so resolving is shown to read the snapshot alone. The
snapshots are the source fixtures the other recipe tests read. The GLM-5.3 cookbook answer names a curve the main
InferenceX fixture carries; it stands for whatever ``golden_length`` records there.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import socket
import subprocess
from functools import cache
from pathlib import Path

import pytest

from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from inference_suite.argv import argv_flag_occurrences
from inference_suite.contracts import BaseLaunch, LaunchQuery, SourceKind, Topology, VersionBasis, Workload
from inference_suite.golden import golden_owned_settings
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT, INFERENCEX_TOOL
from inference_suite.recipe.defaults import TABLE_PATH, default_table
from inference_suite.recipe.golden_curve import GoldenLength, golden_requests
from inference_suite.recipe.image_version import recipe_image_facts, tag_version
from inference_suite.recipe.model import (
    ATOM_TOPOLOGY,
    SGLANG_TOPOLOGY,
    VLLM_TOPOLOGY,
    InferencexSnapshot,
    RecipeSourceError,
    RepoBundle,
)
from inference_suite.recipe.resolve import Miss, SourceSnapshot, UnreadSource, resolve_base
from inference_suite.recipe.sglang_cookbook import sglang_cookbook_launches
from inference_suite.recipe.user_file import parse_user_file
from inference_suite.recipe.vllm_recipes import VllmRecipesSnapshot, rendering_url

_FIXTURES = Path(__file__).parent / "fixtures"
_FETCHED = 1_793_000_000
_RELEASES = {"sglang": "v0.5.23", "vllm": "v0.32.1", "atom": "v0.1.3"}
_DSR1 = "deepseek-ai/DeepSeek-R1-0528"
_GLM53_MXFP4 = "amd/GLM-5.3-Quark-MXFP4-AttnFP8"
_GLM53_FLASH_CURVE = "glm-5.3-flash__mtp__native__default-default"


@pytest.fixture(autouse=True)
def _no_io(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("resolving ran a subprocess or opened a connection")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _load(*parts: str) -> dict:
    return json.loads(gzip.decompress(_FIXTURES.joinpath(*parts).read_bytes()))


@cache
def _inferencex(commit: str) -> InferencexSnapshot:
    return InferencexSnapshot.from_document(_load("inferencex", f"{commit}.json.gz")["snapshot"])


@cache
def _snapshot(inferencex: str = "ce35bc935") -> SourceSnapshot:
    return SourceSnapshot(
        _inferencex(inferencex),
        _FETCHED,
        sglang_cookbook=RepoBundle.from_document(_load("sglang", "c843608d39.json.gz")),
        vllm_recipes=VllmRecipesSnapshot.from_document(_load("vllm_recipes", "20261106.json.gz")),
        atom_catalog=RepoBundle.from_document(_load("atom", "b90ae9990.json.gz")),
        latest_releases=_RELEASES,
    )


def _fixed(model: str = _DSR1, *, framework="sglang", board="mi355x", concurrencies=(64,), isl=8192, osl=1024, **kw):
    return LaunchQuery(model, framework, board, Workload.FIXED, tuple(concurrencies), isl=isl, osl=osl, **kw)


def _agentx(model: str, *, framework: str = "sglang", board: str = "mi355x", concurrencies=(8,), **kw) -> LaunchQuery:
    return LaunchQuery(model, framework, board, Workload.AGENTX, tuple(concurrencies), **kw)


def _base(query: LaunchQuery, snapshot: SourceSnapshot | None = None, gpus: int = 8) -> BaseLaunch:
    base = resolve_base(query, snapshot or _snapshot(), gpus=gpus)
    assert isinstance(base, BaseLaunch), base
    return base


def _miss(query: LaunchQuery, snapshot: SourceSnapshot | None = None, gpus: int = 8) -> Miss:
    miss = resolve_base(query, snapshot or _snapshot(), gpus=gpus)
    assert isinstance(miss, Miss), miss
    return miss


def _drafts(base: BaseLaunch) -> bool:
    return bool(golden_owned_settings(argv_flag_occurrences(base.args), base.env))


def test_inferencex_wins_and_its_base_round_trips_through_its_document():
    base = _base(_fixed())

    assert base.source.kind is SourceKind.INFERENCEX
    assert base.source.entry == "dsr1-fp8-mi355x-sglang"
    assert (base.topology, base.point.conc, base.point.exact) == (Topology(8), 64, True)
    assert (base.image.framework_version, base.image.version_basis) == ("0.5.22", VersionBasis.TAG)
    assert (base.client.tool, base.client.commit) == (INFERENCEX_TOOL, INFERENCEX_CLIENT_COMMIT)
    assert base.resolved_unix == _FETCHED
    assert "--served-model-name" in base.runner_owned.args
    assert "HIP_VISIBLE_DEVICES" in base.runner_owned.env
    assert (base.model.id, base.model.path) == (_fixed().model_id, None)
    assert BaseLaunch.from_document(json.loads(json.dumps(base.to_document()))) == base


def test_the_cookbook_serves_what_inferencex_does_not_and_names_the_miss():
    base = _base(_fixed(framework="vllm", isl=1024, osl=1024))

    assert base.source.kind is SourceKind.VLLM_RECIPES
    assert base.warnings[0].startswith("inferencex holds no launch: no entry serves deepseek-ai/DeepSeek-R1-0528")


def test_the_client_commit_is_the_pin_whatever_inferencex_main_is_at():
    query = _agentx(_GLM53_MXFP4)
    commits = {_inferencex(commit).bundle.commit for commit in ("ce35bc935", "a1e8f975")}
    clients = {_base(query, _snapshot(commit)).client.commit for commit in ("ce35bc935", "a1e8f975")}

    assert len(commits) == 2
    assert clients == {INFERENCEX_CLIENT_COMMIT}


def test_a_shell_user_file_wins_over_every_recipe():
    script = (
        b"export SGLANG_USE_AITER=1\nexport HF_TOKEN=$TOKEN\nexport HIP_VISIBLE_DEVICES=0,1,2,3\n"
        b"python3 -m sglang.launch_server --model-path /models/r1 --tp 4 --mem-fraction-static 0.7 --port 30000\n"
    )
    user_file = parse_user_file("launch.sh", script, framework="sglang")
    base = _base(_fixed(), dataclasses.replace(_snapshot(), user_file=user_file))

    assert base.source.kind is SourceKind.USER_FILE
    assert base.source.files == {"launch.sh": user_file.sha256}
    assert (base.args, base.env, base.topology) == (
        ("--tp", "4", "--mem-fraction-static", "0.7"),
        {"SGLANG_USE_AITER": "1"},
        Topology(4),
    )
    assert base.image.version_basis is VersionBasis.NOT_APPLICABLE
    assert {"--port", "HIP_VISIBLE_DEVICES"} <= set(base.stripped)
    assert base.warnings == ("launch.sh exports HF_TOKEN from shell variables; the base carries none of them",)


def test_a_frozen_base_as_user_file_keeps_its_digest_under_the_new_query():
    frozen = _base(_fixed())
    content = json.dumps(frozen.to_document()).encode()
    user_file = parse_user_file("base_launch.json", content, framework="sglang")
    query = _fixed(concurrencies=(4, 64))

    base = _base(query, dataclasses.replace(_snapshot(), user_file=user_file))

    assert base.digest() == frozen.digest()
    assert base.query == query
    assert base.source == frozen.source


def test_fixed_never_runs_speculative_decoding():
    mtp_only = _base(_fixed("amd/DeepSeek-R1-0528-MXFP4", concurrencies=(4,)))
    assert mtp_only.source.kind is SourceKind.DEFAULT_TABLE
    assert "a fixed workload runs no speculative decoding" in mtp_only.warnings[0]

    for query in (_fixed(concurrencies=(4,)), _fixed(framework="atom", isl=1024, osl=1024)):
        base = _base(query)
        assert base.golden_al is None and not _drafts(base), base.source


def test_agentx_with_no_recipe_is_a_miss_naming_every_source():
    miss = _miss(_agentx("Qwen/Qwen3-8B"))

    assert [item.kind for item in miss.misses] == [SourceKind.INFERENCEX, SourceKind.SGLANG_COOKBOOK]
    assert miss.reasons()[0] == "inferencex: no entry serves Qwen/Qwen3-8B with sglang on a mi355x"


def test_agentx_on_atom_is_a_miss_not_a_default():
    miss = _miss(_agentx(_DSR1, framework="atom"))

    assert [item.kind for item in miss.misses] == [SourceKind.INFERENCEX, SourceKind.ATOM_CATALOG]


def test_one_point_serves_the_whole_sweep():
    base = _base(_fixed(concurrencies=(8, 48)))

    assert (base.point.conc, base.point.exact) == (64, False)
    assert any("no point at concurrency 48" in warning for warning in base.warnings)


def test_tp_ambiguity_takes_the_largest_tp_the_gpus_fit_and_records_the_rest():
    eight = _base(_fixed(), gpus=8)
    assert eight.topology == Topology(8)
    assert [alternative.topology for alternative in eight.point.alternatives] == [Topology(4)]
    assert eight.point.alternatives[0].locator.endswith("8k1k.yaml:zip_override_tp4_ep1[1]")
    assert any("holds other launches that fit: tp 4 ep 1" in warning for warning in eight.warnings)

    four = _base(_fixed(), gpus=4)
    assert (four.topology, four.point.alternatives) == (Topology(4), ())

    two = _base(_fixed(), gpus=2)
    assert two.source.kind is SourceKind.DEFAULT_TABLE
    assert "inferencex holds no launch: its launches run tp [4, 8], more than the 2 GPUs here" in two.warnings


def test_an_operator_tp_keeps_only_its_arm():
    base = _base(_fixed(tp=4))

    assert (base.topology, base.point.alternatives) == (Topology(4), ())


def test_an_arm_offloading_no_kv_cache_wins_at_the_same_tp():
    base = _base(_agentx("zai-org/GLM-5.2-FP8"))

    assert base.source.locator.endswith("agentic.yaml:override_tp8_c10")
    assert [alternative.locator.rpartition(":")[2] for alternative in base.point.alternatives] == [
        "override_tp8_c8_hicache"
    ]


def test_boards_of_one_architecture_and_capacity_share_launches():
    twin = _base(_agentx("zai-org/GLM-5.2-FP8", board="mi350x"))
    assert twin.source.kind is SourceKind.INFERENCEX
    assert any("on mi355x, not mi350x" in warning for warning in twin.warnings)

    miss = _miss(_agentx("zai-org/GLM-5.2-FP8", board="mi300x"))
    assert miss.reasons()[0] == "inferencex: no entry serves zai-org/GLM-5.2-FP8 with sglang on a mi300x"


def test_an_inferencex_golden_acceptance_becomes_the_golden_layer():
    base = _base(_agentx("amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8", concurrencies=(1,)))
    golden = base.golden_al

    assert (golden.value, golden.method, golden.token_mode) == (2.32, "match-expected", "real-draft-token")
    assert (golden.spec_method, golden.draft_length, golden.draft_model) == ("nextn", 3, None)
    assert "--speculative-algorithm" in golden.args and "--speculative-algorithm" not in base.args
    assert golden.env["SGLANG_SIMULATE_ACC_LEN"] == "2.32"
    assert not _drafts(base)


def test_a_vllm_golden_layer_takes_the_whole_speculative_config():
    base = _base(_agentx("deepseek-ai/DeepSeek-V4.1-Flash", framework="vllm", board="mi300x", concurrencies=(1,)))
    golden = base.golden_al

    assert golden.args[0] == "--speculative-config"
    assert json.loads(golden.args[1])["synthetic_acceptance_length"] == 3.51
    assert (golden.method, golden.spec_method, golden.draft_length) == ("synthetic", "dspark", 5)
    assert not _drafts(base)


def _glm53_snapshot(answer: GoldenLength) -> SourceSnapshot:
    snapshot = _snapshot()
    query = _agentx(_GLM53_MXFP4)
    launches = sglang_cookbook_launches(query, snapshot.sglang_cookbook)
    (request,) = golden_requests(query, launches, snapshot.inferencex.bundle, "zai-org/GLM-5.3")
    return dataclasses.replace(snapshot, base_model="zai-org/GLM-5.3", golden_lengths={request.key: answer})


def test_a_cookbook_agentx_launch_takes_its_golden_from_the_recorded_answer():
    base = _base(_agentx(_GLM53_MXFP4), _glm53_snapshot(GoldenLength(3.48, curve=_GLM53_FLASH_CURVE)))

    assert base.source.kind is SourceKind.SGLANG_COOKBOOK
    assert (base.golden_al.value, base.golden_al.spec_method, base.golden_al.draft_length) == (3.48, "eagle", 5)
    assert base.golden_al.curve.path.endswith(f"{_GLM53_FLASH_CURVE}.yaml")
    assert base.golden_al.env["SGLANG_SIMULATE_ACC_LEN"] == "3.48"
    assert not _drafts(base)


def test_a_cookbook_agentx_launch_with_no_curve_runs_with_real_acceptance_and_says_why():
    base = _base(_agentx(_GLM53_MXFP4), _glm53_snapshot(GoldenLength(None, error="no committed curve")))

    assert base.golden_al is None
    assert _drafts(base)
    assert "the launch runs with no golden acceptance: InferenceX has no golden curve for it" in base.warnings[-1]


def test_inferencex_at_the_unmerged_entry_wins_over_the_cookbook():
    base = _base(_agentx(_GLM53_MXFP4), _snapshot("a1e8f975"))

    assert (base.source.kind, base.source.entry) == (SourceKind.INFERENCEX, "glm5.3-fp4-mi355x-sglang-agentic-mtp")
    assert base.golden_al.value == 3.48


@pytest.mark.parametrize(
    ("query", "snapshot", "corpus"),
    [
        (_agentx("amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8", concurrencies=(1,)), None, "_256k"),
        (_agentx(_GLM53_MXFP4), _snapshot("a1e8f975"), ""),
        (_agentx("deepseek-ai/DeepSeek-V4.1-Flash", framework="vllm", board="mi300x", concurrencies=(1,)), None, ""),
    ],
)
def test_an_agentx_base_names_the_corpus_inferencex_replays_for_its_model_family(query, snapshot, corpus):
    base = _base(query, snapshot)

    assert base.client.env["WEKA_LOADER_OVERRIDE"] == f"semianalysis_cc_traces_weka_062126{corpus}"


def test_a_cookbook_agentx_base_names_the_corpus_of_its_base_models_family():
    base = _base(_agentx(_GLM53_MXFP4), _glm53_snapshot(GoldenLength(3.48, curve=_GLM53_FLASH_CURVE)))

    assert base.source.kind is SourceKind.SGLANG_COOKBOOK
    assert base.client.env["WEKA_LOADER_OVERRIDE"] == "semianalysis_cc_traces_weka_062126"


def test_a_fixed_base_names_no_corpus():
    assert "WEKA_LOADER_OVERRIDE" not in _base(_fixed()).client.env


def test_a_recipe_image_without_a_version_assumes_the_latest_release():
    kimi = _base(_agentx("moonshotai/Kimi-K3.1-Pro", concurrencies=(32,)))
    assert (kimi.image.version_basis, kimi.image.framework_version) == (VersionBasis.ASSUMED_LATEST, "0.5.23")
    assert kimi.image.assumed_release == "v0.5.23"

    without_release = dataclasses.replace(_snapshot(), latest_releases={})
    with pytest.raises(RecipeSourceError, match="records no release version"):
        resolve_base(_agentx("moonshotai/Kimi-K3.1-Pro", concurrencies=(32,)), without_release, gpus=8)


@pytest.mark.parametrize(
    ("image", "version"),
    [
        ("lmsysorg/sglang:v0.5.22-rocm720-mi35x@sha256:" + "0" * 64, "0.5.22"),
        ("lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260929", "0.5.20"),
        ("vllm/vllm-openai-rocm:v0.32.0", "0.32.0"),
        ("rocm/sgl-dev:v0.5.9.post1-rocm700", "0.5.9.post1"),
        ("registry:5000/vllm:v1.2.3.4", None),
        ("vllm/vllm-openai-rocm:nightly-" + "a" * 40, None),
        ("lmsysorg/sglang:dev-kimi-k31-pro-mi35x", None),
        ("rocm/atom:rocm7.2_atom20260511", None),
        ("vllm/vllm-openai:latest", None),
        (None, None),
    ],
)
def test_the_tag_names_the_recipe_framework_version(image, version):
    assert tag_version(image) == version


def test_an_unversioned_image_needs_a_release_that_names_a_version():
    assert recipe_image_facts("x/y:latest", "v0.32.1").framework_version == "0.32.1"
    with pytest.raises(RecipeSourceError):
        recipe_image_facts("x/y:latest", "nightly")


_ROWS = [
    ("sglang", {"gfx942"}, "--trust-remote-code --mem-fraction-static 0.8 --disable-radix-cache",
     {"SGLANG_USE_AITER": "1", "SGLANG_AITER_MLA_PERSIST": "1"}, "sglang_mi300x.sh"),
    ("sglang", {"gfx950"}, "--trust-remote-code --mem-fraction-static 0.8 --disable-radix-cache",
     {"SGLANG_USE_AITER": "1"}, "sglang_mi355x.sh"),
    ("sglang", {"gfx1151"},
     "--trust-remote-code --mem-fraction-static 0.5 --disable-radix-cache --attention-backend triton --disable-cuda-graph",
     {"SGLANG_USE_AITER": "0"}, "sglang_radeon8060s.sh"),
    ("vllm", {"gfx942", "gfx950"}, "--trust-remote-code --gpu-memory-utilization 0.95",
     {"VLLM_ROCM_USE_AITER": "1"}, "vllm_mi355x.sh"),
    ("vllm", {"gfx1201"}, "--trust-remote-code --gpu-memory-utilization 0.9",
     {"VLLM_ROCM_USE_AITER": "0", "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}, "vllm_gfx12.sh"),
    ("vllm", {"gfx1151"}, "--trust-remote-code --gpu-memory-utilization 0.95",
     {"VLLM_ROCM_USE_AITER": "0", "VLLM_ROCM_USE_AITER_MHA": "0", "VLLM_ROCM_USE_AITER_RMSNORM": "0"},
     "vllm_radeon8060s.sh"),
    ("atom", {"gfx942", "gfx950"}, "--trust-remote-code", {}, "baseline_atom.yaml"),
]  # fmt: skip


def test_each_default_row_is_the_launcher_it_names():
    rows = default_table().rows

    assert [(row.framework, set(row.gfx), " ".join(row.args), dict(row.env)) for row in rows] == [
        row[:4] for row in _ROWS
    ]
    assert all(origin in row.origin for row, (*_, origin) in zip(rows, _ROWS, strict=True))
    assert {gfx for row in rows for gfx in row.gfx} <= {board.gfx for board in AMD_GPU_BOARDS.values()}


def test_the_default_table_leaves_dual_chunk_attention_and_the_moe_runner_to_the_host():
    for row in default_table().rows:
        assert not any("dual" in token or "moe-runner" in token for token in row.args), row.origin
        assert not any("DUAL_CHUNK" in name or "MOE" in name for name in row.env), row.origin


def test_a_fixed_query_no_recipe_serves_runs_the_default_row_with_its_topology():
    base = _base(_fixed("Qwen/Qwen3-8B", framework="vllm", isl=1024, osl=1024, tp=2, ep=2))

    assert (base.source.kind, base.source.path) == (SourceKind.DEFAULT_TABLE, TABLE_PATH)
    assert base.args == (
        *("--trust-remote-code", "--gpu-memory-utilization", "0.95"),
        *("--tensor-parallel-size", "2", "--enable-expert-parallel"),
    )
    assert base.env == {"VLLM_ROCM_USE_AITER": "1"}
    assert base.image.version_basis is VersionBasis.NOT_APPLICABLE
    assert base.warnings[-1] == "no recipe serves Qwen/Qwen3-8B; the default vllm launch for gfx950 runs it"


def test_no_default_row_or_no_topology_is_a_miss():
    no_row = _miss(_fixed("Qwen/Qwen3-8B", board="r9700", isl=1024, osl=1024))
    assert no_row.reasons()[-1] == "default_table: it has no sglang row for r9700 (gfx1201)"

    split_experts = _miss(_fixed("Qwen/Qwen3-8B", framework="vllm", isl=1024, osl=1024, tp=4, ep=2))
    assert "its experts span all 4 TP ranks, so it runs no ep 2" in split_experts.reasons()[-1]


def _unread_snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        _inferencex("ce35bc935"),
        _FETCHED,
        unread=UnreadSource("the sglang cookbook", "git clone exited 128: could not resolve host"),
    )


def test_a_fixed_query_whose_source_could_not_be_read_runs_the_default_row_and_records_why():
    base = _base(_fixed(tp=8), _unread_snapshot())

    assert (base.source.kind, base.source.path) == (SourceKind.DEFAULT_TABLE, TABLE_PATH)
    assert base.client.commit == INFERENCEX_CLIENT_COMMIT
    assert base.warnings == (
        "the sglang cookbook could not be read, so no recipe was consulted (git clone exited 128: could not resolve "
        "host); the default sglang launch for gfx950 runs it",
    )
    assert BaseLaunch.from_document(json.loads(json.dumps(base.to_document()))).warnings == base.warnings


def test_an_agentx_query_whose_source_could_not_be_read_is_a_miss_naming_it():
    miss = _miss(_agentx("zai-org/GLM-5.3-FP8"), _unread_snapshot())

    assert miss.reasons() == ("the sglang cookbook could not be read: git clone exited 128: could not resolve host",)


def _no_inferencex_snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        None, _FETCHED, unread=UnreadSource("InferenceX", "git clone exited 128: could not resolve host")
    )


def test_a_fixed_query_whose_inferencex_could_not_be_read_runs_the_default_row_with_the_pinned_client():
    base = _base(_fixed(tp=8), _no_inferencex_snapshot())

    assert base.source.kind is SourceKind.DEFAULT_TABLE
    assert (base.client.tool, base.client.commit) == (INFERENCEX_TOOL, INFERENCEX_CLIENT_COMMIT)
    assert base.warnings[-1].startswith("InferenceX could not be read, so no recipe was consulted")
    assert BaseLaunch.from_document(json.loads(json.dumps(base.to_document()))) == base


def test_an_agentx_query_whose_inferencex_could_not_be_read_is_a_miss_naming_it():
    miss = _miss(_agentx("zai-org/GLM-5.3-FP8"), _no_inferencex_snapshot())

    assert miss.reasons() == ("InferenceX could not be read: git clone exited 128: could not resolve host",)


def test_a_user_file_is_the_base_with_the_pinned_client_and_no_source_read():
    user_file = parse_user_file("launch.sh", b"python3 -m sglang.launch_server --tp-size 8\n", framework="sglang")
    base = _base(_fixed(), SourceSnapshot(None, _FETCHED, user_file=user_file))

    assert (base.source.kind, base.client.commit) == (SourceKind.USER_FILE, INFERENCEX_CLIENT_COMMIT)


def test_a_user_file_is_the_base_whatever_source_could_not_be_read():
    user_file = parse_user_file("launch.sh", b"python3 -m sglang.launch_server --tp-size 8\n", framework="sglang")
    snapshot = dataclasses.replace(_unread_snapshot(), user_file=user_file)

    assert _base(_fixed(), snapshot).source.kind is SourceKind.USER_FILE


@pytest.mark.parametrize(
    ("spelling", "args", "topology"),
    [
        (SGLANG_TOPOLOGY, ("--tp-size", "4", "--ep-size", "2"), Topology(4, 2)),
        (VLLM_TOPOLOGY, ("-tp", "4", "--data-parallel-size", "2", "--enable-expert-parallel"), Topology(4, 8)),
        (VLLM_TOPOLOGY, ("-tp", "4", "-ep"), Topology(4, 4)),
        (ATOM_TOPOLOGY, ("-tp", "8", "--enable-expert-parallel"), Topology(8, 8)),
        (VLLM_TOPOLOGY, ("--max-num-seqs", "8"), Topology(1, 1)),
    ],
)
def test_each_framework_reads_its_topology_spellings(spelling, args, topology):
    assert spelling.read(args) == topology


def test_a_snapshot_missing_the_framework_cookbook_is_refused():
    with pytest.raises(RecipeSourceError, match="carries no vLLM recipes"):
        resolve_base(
            _fixed(framework="vllm", isl=1024, osl=1024),
            dataclasses.replace(_snapshot(), vllm_recipes=None),
            gpus=8,
        )


def test_vllm_short_expert_parallel_reads_the_same_topology_from_a_recipe_and_the_default_table():
    url = rendering_url(_DSR1, "mi355x", "baseline")
    responses = dict(_snapshot().vllm_recipes.responses)
    rendered = json.loads(responses[url].body)
    rendered["argv"] = [*rendered["argv"][:3], "-tp", "4", "-ep"]
    responses[url] = dataclasses.replace(responses[url], body=json.dumps(rendered))
    recipe = _base(
        _fixed(framework="vllm", isl=1024, osl=1024),
        dataclasses.replace(_snapshot(), vllm_recipes=VllmRecipesSnapshot(responses)),
    )
    default = _base(_fixed("Qwen/Qwen3-8B", framework="vllm", isl=1024, osl=1024, tp=4, ep=4))

    assert (recipe.source.kind, default.source.kind) == (SourceKind.VLLM_RECIPES, SourceKind.DEFAULT_TABLE)
    for base in (recipe, default):
        assert base.topology == VLLM_TOPOLOGY.read(base.args) == Topology(4, 4)
