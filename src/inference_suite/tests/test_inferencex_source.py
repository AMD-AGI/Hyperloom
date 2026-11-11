# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX as a recipe source, against snapshots recorded from InferenceX itself.

``ce35bc935.json.gz`` records InferenceX main at ce35bc935: its master and runner configs, ``select_recipe``,
``acceptance_length`` and ``build_overrides`` for every AMD single-node point, srtctl's (acfb4dc)
``_config_to_cli_args`` on each selected worker's args, and a digest of each point InferenceX's own matrix generator
and launcher produce. ``a1e8f975.json.gz`` records the head of InferenceX PR #4593 for the GLM-5.3 MXFP4 AgentX
entry it adds.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import subprocess
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest
import yaml

from inference_suite.contracts import LaunchQuery, SourceKind, Workload
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT, INFERENCEX_REPO_URL
from inference_suite.recipe import inferencex
from inference_suite.recipe.inferencex import (
    MASTER,
    PROJECT,
    agentx_trace,
    inferencex_launches,
    inferencex_points,
    srtctl_cli_args,
)
from inference_suite.recipe.model import (
    InferencexSnapshot,
    RepoBundle,
    RecipeSourceError,
    SourceLaunch,
    SourceMiss,
    boards_interchangeable,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "inferencex"
_DSR1 = "deepseek-ai/DeepSeek-R1-0528"
_GLM53 = "amd/GLM-5.3-Quark-MXFP4-AttnFP8"


@cache
def _recording(name: str) -> dict:
    return json.loads(gzip.decompress((_FIXTURES / name).read_bytes()))


def _main() -> InferencexSnapshot:
    return InferencexSnapshot.from_document(_recording("ce35bc935.json.gz")["snapshot"])


def _pr4593() -> InferencexSnapshot:
    return InferencexSnapshot.from_document(_recording("a1e8f975.json.gz")["snapshot"])


def _fixed(model: str = _DSR1, *, concurrencies=(64,), board: str = "mi355x", **changes) -> LaunchQuery:
    query = {"framework": "sglang", "isl": 8192, "osl": 1024, **changes}
    return LaunchQuery(
        model_id=model, board=board, workload=Workload.FIXED, concurrencies=tuple(concurrencies), **query
    )


def _agentx(model: str, *, concurrencies, framework: str = "sglang", board: str = "mi355x") -> LaunchQuery:
    return LaunchQuery(
        model_id=model, framework=framework, board=board, workload=Workload.AGENTX, concurrencies=concurrencies
    )


def _launches(query: LaunchQuery, snapshot: InferencexSnapshot | None = None) -> tuple[SourceLaunch, ...]:
    found = inferencex_launches(query, snapshot or _main())
    assert not isinstance(found, SourceMiss), found
    return found


def _point_digest(recipe: str, environment: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps({"recipe": recipe, "environment": environment}, sort_keys=True).encode()
    ).hexdigest()


def test_points_and_their_launcher_environment_match_inferencex_own_matrix():
    points = inferencex_points(_main().bundle)

    assert len(points) == 321
    assert sorted(_point_digest(point.recipe, point.environment()) for point in points) == sorted(
        _recording("ce35bc935.json.gz")["upstream_points"]
    )


def test_every_point_binds_exactly_one_variant_as_recorded():
    snapshot = _main()
    points = inferencex_points(snapshot.bundle)

    assert {point.key for point in points} == set(snapshot.selections)
    assert not snapshot.rejections
    for point in points:
        assert snapshot.selections[point.key].variant.startswith(point.recipe), point.key


def test_argv_renders_role_args_as_srtctl_does():
    recorded = _recording("ce35bc935.json.gz")["srtctl_args"]
    for key, selection in _main().selections.items():
        engine = selection.recipe["engine"]
        engine = engine["type"] if isinstance(engine, dict) else engine

        assert list(srtctl_cli_args(engine, selection.recipe["roles"]["agg"]["args"])) == recorded[key], key


def test_dsr1_fixed_launch_is_the_selected_variant_without_runner_owned_flags():
    [tp8] = _launches(_fixed(tp=8))

    assert tp8.source.kind is SourceKind.INFERENCEX
    assert tp8.source.commit == "ce35bc935c592818bb7afd15aeefdf3a45aae799"
    assert tp8.source.entry == "dsr1-fp8-mi355x-sglang"
    assert (
        tp8.source.locator
        == "benchmarks/single_node/srt-slurm-recipes/dsr1/sglang/mi355x-fp8/8k1k.yaml:zip_override_tp8_ep1[4]"
    )
    assert tp8.source.tools == {"srt-slurm": "acfb4dcc3bd48fd6a1e7b615e849e7954c0c3852"}
    assert f"{PROJECT}/benchmarks/single_node/srt-slurm-recipes/dsr1/sglang/mi355x-fp8/8k1k.yaml" in tp8.source.files
    assert tp8.entrypoint == ("python3", "-m", "sglang.launch_server")
    assert tp8.image.startswith("lmsysorg/sglang:v0.5.22-rocm720-mi35x@sha256:")
    assert (tp8.topology.tp, tp8.topology.ep, tp8.conc, tp8.exact) == (8, 1, 64, True)
    assert tp8.args == (
        *("--attention-backend", "aiter", "--chunked-prefill-size", "196608", "--cuda-graph-max-bs-decode", "64"),
        *("--data-parallel-size", "1", "--disable-radix-cache", "--expert-parallel-size", "1"),
        *("--kv-cache-dtype", "fp8_e4m3", "--max-prefill-tokens", "196608", "--mem-fraction-static", "0.8"),
        *("--num-continuous-decode-steps", "8", "--tensor-parallel-size", "8", "--trust-remote-code"),
    )
    assert tp8.stripped == ("--enable-metrics", "--served-model-name")
    assert tp8.env == {"RCCL_MSCCL_ENABLE": "0", "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4", "SGLANG_USE_AITER": "1"}
    assert tp8.golden is None
    assert tp8.warnings == ()


def test_dsr1_c64_is_served_by_both_tp_arms():
    launches = _launches(_fixed())

    assert sorted((launch.topology.tp, launch.conc) for launch in launches) == [(4, 64), (8, 64)]
    assert {launch.source.entry for launch in launches} == {"dsr1-fp8-mi355x-sglang"}


def test_a_concurrency_between_points_takes_the_nearest_larger_point_of_each_arm():
    launches = _launches(_fixed(concurrencies=(8, 48)))

    assert sorted((launch.topology.tp, launch.conc, launch.exact) for launch in launches) == [
        (4, 64, False),
        (8, 64, False),
    ]
    assert all("no point at concurrency 48" in launch.warnings[0] for launch in launches)


def test_fixed_workload_skips_speculative_entries():
    launches = _launches(_fixed(concurrencies=(4,)))
    assert {launch.source.entry for launch in launches} == {"dsr1-fp8-mi355x-sglang"}

    miss = inferencex_launches(_fixed("amd/DeepSeek-R1-0528-MXFP4", concurrencies=(4,)), _main())
    assert isinstance(miss, SourceMiss)
    assert "a fixed workload runs no speculative decoding" in miss.reason


def test_fixed_workload_needs_the_same_sequence_shape():
    miss = inferencex_launches(_fixed(isl=1024, osl=1024), _main())

    assert isinstance(miss, SourceMiss)
    assert "runs 8192/1024, not 1024/1024" in miss.reason


def test_an_operator_topology_keeps_only_its_arm():
    miss = inferencex_launches(_fixed(tp=2), _main())

    assert isinstance(miss, SourceMiss)
    assert "it runs tp 4, not 2" in miss.reason


def test_null_in_an_override_deletes_the_base_setting():
    model = "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
    [c1] = [launch for launch in _launches(_agentx(model, concurrencies=(1,))) if launch.exact]

    assert c1.source.locator.endswith("agentic.yaml:override_c1")
    assert "--enable-amd-monokernel" in c1.args
    assert "--enable-linear-replayssm-spec" not in c1.args
    assert "--ple-offload-embedding" not in c1.args


def test_agentx_launch_carries_the_recorded_golden_acceptance():
    model = "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
    [c1] = [launch for launch in _launches(_agentx(model, concurrencies=(1,))) if launch.exact]

    assert c1.golden.value == 2.32
    assert (
        c1.golden.curve.path
        == f"{PROJECT}/infx/golden_al_distribution/qwen3.8-flash-next-fp8__mtp__native__default-default.yaml"
    )
    assert c1.env["SGLANG_SIMULATE_ACC_LEN"] == "2.32"
    assert c1.env["SGLANG_SIMULATE_ACC_METHOD"] == "match-expected"
    assert c1.client_env["AIPERF_REQUIRED_SERVER_METRIC_PREFIX"] == "sglang:"
    assert "CONC" not in c1.client_env


def test_agentx_skips_atom_entries():
    points = [point for point in inferencex_points(_main().bundle) if point.framework == "atom"]
    agentic = next(point for point in points if point.scenario.value == "agentic-coding")
    query = _agentx(agentic.model, concurrencies=(agentic.conc,), framework="atom", board=agentic.board)

    miss = inferencex_launches(query, _main())

    assert isinstance(miss, SourceMiss)
    assert "an AgentX launch runs ['sglang', 'vllm'], not atom" in miss.reason


def test_main_snapshot_does_not_see_an_unmerged_entry():
    miss = inferencex_launches(_agentx(_GLM53, concurrencies=(8,)), _main())

    assert isinstance(miss, SourceMiss)
    assert miss.kind is SourceKind.INFERENCEX
    assert f"no entry serves {_GLM53}" in miss.reason


def test_pr4593_snapshot_serves_glm53_agentx_c8_with_golden_3_48():
    [c8] = _launches(_agentx(_GLM53, concurrencies=(8,)), _pr4593())

    assert c8.source.commit == "a1e8f9751644d3f3788a561c2f36b3bd8dccd4a8"
    assert c8.source.entry == "glm5.3-fp4-mi355x-sglang-agentic-mtp"
    assert (c8.topology.tp, c8.topology.ep, c8.conc, c8.exact) == (4, 4, 8, True)
    assert c8.kv_offloading == "dram"
    assert c8.golden.value == 3.48
    assert c8.golden.curve.path.endswith("glm-5.3-nvfp4__mtp__native__default-default.yaml")
    assert c8.env["SGLANG_SIMULATE_ACC_LEN"] == "3.48"


def test_a_cluster_runner_stands_for_its_board_and_like_boards_share_launches():
    points = {point.runner: point.board for point in inferencex_points(_main().bundle)}
    assert points["cluster:mi355x-amds"] == "mi355x"
    assert points["cluster:mi325x-amd"] == "mi325x"
    assert points["cluster:mi300x-amd"] == "mi300x"

    [mi350x] = _launches(_fixed(board="mi350x", tp=8))
    assert mi350x.warnings == ("InferenceX tuned dsr1-fp8-mi355x-sglang on mi355x, not mi350x",)
    assert boards_interchangeable("mi350x", "mi355x")
    assert not boards_interchangeable("mi325x", "mi300x")


def test_a_snapshot_without_the_point_selection_is_refused():
    snapshot = _pr4593()

    with pytest.raises(RecipeSourceError, match="records no selection"):
        inferencex_launches(_fixed(), snapshot)


def test_snapshot_round_trips_through_its_document():
    snapshot = _pr4593()

    assert InferencexSnapshot.from_document(json.loads(json.dumps(snapshot.to_document()))) == snapshot


def test_vllm_agentx_launch_serves_its_model_and_simulates_the_golden_acceptance():
    launches = _launches(
        _agentx("deepseek-ai/DeepSeek-V4.1-Flash", concurrencies=(1,), framework="vllm", board="mi300x")
    )

    for launch in launches:
        spec = json.loads(launch.args[launch.args.index("--speculative-config") + 1])
        assert launch.entrypoint == ("vllm", "serve")
        assert launch.stripped == ("--served-model-name",)
        assert (spec["rejection_sample_method"], spec["synthetic_acceptance_length"]) == ("synthetic", 3.51)
        assert launch.golden.value == 3.51


def test_a_launch_needing_a_router_frontend_is_not_reproduced():
    miss = inferencex_launches(_agentx("deepseek-ai/DeepSeek-V4.1-Flash", concurrencies=(64,)), _main())

    assert isinstance(miss, SourceMiss)
    assert "search-space[2]: it serves through the sglang-router frontend" in miss.reason
    assert "search-space[0]: it has no point at concurrency 64 or above" in miss.reason


_FULL_CORPUS = "semianalysis_cc_traces_weka_062126"


@pytest.mark.parametrize(
    ("model", "corpus"),
    [
        ("amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4", _FULL_CORPUS),
        ("zai-org/GLM-5.3", _FULL_CORPUS),
        ("org/a-model-inferencex-does-not-name", _FULL_CORPUS),
        (_DSR1, f"{_FULL_CORPUS}_256k"),
        ("Qwen/Qwen3.5-397B-A17B-FP8", f"{_FULL_CORPUS}_256k"),
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8", f"{_FULL_CORPUS}_256k"),
    ],
)
def test_only_the_families_inferencex_caps_replay_the_256k_corpus(model, corpus):
    assert agentx_trace(_main().bundle, (model,)) == corpus


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.mark.upstream_sources
def test_the_capped_families_are_inferencex_clients_at_its_pin(tmp_path: Path) -> None:
    if subprocess.run(("git", "ls-remote", "--exit-code", INFERENCEX_REPO_URL, "HEAD"), capture_output=True).returncode:
        pytest.skip(f"{INFERENCEX_REPO_URL} is not reachable")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "fetch", "-q", "--depth=1", "--filter=blob:none", INFERENCEX_REPO_URL, INFERENCEX_CLIENT_COMMIT)
    source = _git(tmp_path, "show", f"FETCH_HEAD:{PROJECT}/infx/bench/agentic/traces.py")
    constants = {
        target.id: ast.literal_eval(statement.value)
        for statement in ast.parse(source).body
        if isinstance(statement, ast.Assign) and isinstance(statement.value, (ast.Constant, ast.Tuple))
        for target in statement.targets
        if isinstance(target, ast.Name)
    }

    assert constants["CAPPED_FAMILIES"] == inferencex._CAPPED_FAMILIES
    assert (constants["UNCAPPED"], constants["CAPPED"]) == (inferencex._UNCAPPED_TRACE, inferencex._CAPPED_TRACE)


def _with_entries(snapshot: InferencexSnapshot, **entries: dict) -> InferencexSnapshot:
    """``snapshot`` with ``entries`` added to its master config, each a copy of the first DeepSeek-R1 entry."""
    bundle = snapshot.bundle
    master = yaml.safe_load(bundle.text(MASTER))
    template = next(entry for entry in master.values() if entry["model"] == _DSR1)
    master.update({name: {**template, **changes} for name, changes in entries.items()})
    files = {**bundle.files, MASTER: yaml.safe_dump(master)}
    return replace(snapshot, bundle=RepoBundle(bundle.repo, bundle.commit, files))


def test_an_entry_of_another_model_that_cannot_be_placed_does_not_stop_this_one():
    snapshot = _with_entries(_main(), drifted={"model": "org/drifted", "runner": "cluster:nowhere"})

    assert _launches(_fixed(), snapshot)


def test_an_entry_of_the_model_that_cannot_be_placed_is_its_miss():
    snapshot = _with_entries(
        _main(),
        drifted={"model": "org/drifted", "runner": "cluster:nowhere"},
        shapeless={"model": "org/drifted", "scenarios": {"fixed-seq-len": [{"isl": 8192, "osl": 1024}]}},
    )

    miss = inferencex_launches(_fixed("org/drifted"), snapshot)

    assert isinstance(miss, SourceMiss)
    assert "drifted: InferenceX's config does not place it: runner cluster:nowhere is on 0 board labels" in miss.reason
    assert "shapeless: InferenceX's config does not place it" in miss.reason
