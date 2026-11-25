# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import textwrap
import types
from dataclasses import replace

import pytest

from inference_suite.contracts.base_launch import LaunchSource, Redacted, SourceKind, Topology, Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.inferencex import (
    MULTI_NODE_RECIPES,
    RoleFit,
    deep_merge,
    expand_variants,
    master_candidates,
    recipe_fit,
)
from inference_suite.multinode.query import PdQuery, RoleAsk
from inference_suite.multinode.record import FrontendType, PdMode, PdSourceKind, Role
from inference_suite.multinode.recipe import OperatorArgs, PdResolveError, ResolveInputs, resolve
from inference_suite.recipe.inferencex import MASTER, RUNNERS
from inference_suite.recipe.resolve import Miss
from inference_suite.recipe.sources import Checkout, SourceFetchError

_MASTER = """
m-fp8-mi355x-sglang-disagg:
  model: org/M
  model-prefix: m
  runner: cluster:mi355x-amds
  precision: fp8
  framework: sglang-disagg
  multinode: true
  disagg: true
  kv-p2p-transfer: mori
  srt-recipe-dir: m/sglang/mi355x-fp8
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - srt-recipe: 1k1k/disagg.yaml:override_1p1d
        prefill: {num-worker: 1, tp: 8}
        decode: {num-worker: 1, tp: 8}
        conc-list: [32, 128]
      - srt-recipe: 1k1k/disagg.yaml:override_1p1d_mtp
        prefill: {num-worker: 1, tp: 8}
        decode: {num-worker: 1, tp: 8}
        spec-decoding: mtp
        conc-list: [64]
"""
_RUNNERS = """
labels:
  mi355x: [n1, n2]
  cluster:mi355x-amds: [n1]
"""
_RECIPE = """
schema: 2
base:
  resources: {gpu_type: mi355x, gpus_per_node: 8}
  frontend:
    type: sglang-router
    args: {policy: round_robin, port: 8000}
  engine: sglang
  roles:
    prefill:
      workers: 1
      gpus: 8
      env: {HF_HOME: /hf, MORI_SHMEM_MODE: ISOLATION}
      args: {tp-size: 8, model-path: /m, mem-fraction-static: 0.8}
    decode:
      workers: 1
      gpus: 8
      args: {tp-size: 8, mem-fraction-static: 0.85}
override_1p1d:
  roles: {decode: {args: {max-running-requests: 128}}}
override_1p1d_mtp:
  roles: {decode: {args: {speculative-algorithm: EAGLE}}}
zip_override_1p2d:
  roles:
    decode:
      workers: [2]
      args: {max-running-requests: [64, 256], mem-fraction-static: [null, 0.9]}
"""


def _write_checkout(root, files):
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(textwrap.dedent(text))
    return Checkout("https://example/InferenceX", "c0ffee", root)


@pytest.fixture
def checkout(tmp_path):
    recipe = f"{MULTI_NODE_RECIPES}/m/sglang/mi355x-fp8/1k1k/disagg.yaml"
    return _write_checkout(tmp_path, {MASTER: _MASTER, RUNNERS: _RUNNERS, recipe: _RECIPE})


def _query(prefill=1, decode=1, **fields):
    query = PdQuery(
        model_id="org/M",
        model_path="/models/M",
        framework=Framework.SGLANG,
        board="mi355x",
        gpus_per_node=8,
        workload=Workload.FIXED,
        concurrencies=(64,),
        roles={
            Role.PREFILL: RoleAsk(tuple(f"p{i}" for i in range(prefill))),
            Role.DECODE: RoleAsk(tuple(f"d{i}" for i in range(decode))),
        },
        router_host="r0",
        isl=1024,
        osl=1024,
    )
    return replace(query, **fields)


def test_deep_merge_follows_srtctl():
    merged = deep_merge({"a": {"b": 1, "c": 2}, "l": [1, 2], "d": 1}, {"a": {"b": 3}, "l": [9], "d": None})
    assert merged == {"a": {"b": 3, "c": 2}, "l": [9]}


def test_zip_variants_broadcast_and_drop_nulls(checkout):
    raw = {"base": {"x": 1, "y": 2}, "zip_override_z": {"x": [5], "y": [None, 7]}}
    assert expand_variants(raw) == [("zip_override_z[0]", {"x": 5}), ("zip_override_z[1]", {"x": 5, "y": 7})]
    assert expand_variants(raw, "zip_override_z[1]") == [("zip_override_z[1]", {"x": 5, "y": 7})]


def test_master_row_matches_hosts_and_takes_nearest_larger_conc(checkout):
    found = master_candidates(_query(), checkout)
    assert [(c.entry, c.locator, c.conc) for c in found.found] == [
        ("m-fp8-mi355x-sglang-disagg[0]", "override_1p1d", 128)
    ]
    assert any("speculative" in miss for miss in found.misses)


def test_master_row_refuses_other_host_counts(checkout):
    assert master_candidates(_query(decode=2), checkout).found == ()


@pytest.mark.parametrize(
    ("candidates", "decode", "model_id", "locators"),
    [(master_candidates, 1, "M", ["override_1p1d"])],
    ids=["master"],
)
def test_a_bare_repo_name_finds_what_inferencex_names_it_by(checkout, candidates, decode, model_id, locators):
    found = candidates(_query(decode=decode, model_id=model_id), checkout).found
    assert [c.locator for c in found] == locators
    assert all(f"InferenceX names {model_id} org/M" in c.warnings for c in found)


def test_a_bare_repo_name_several_orgs_name_is_not_taken():
    from inference_suite.multinode import inferencex

    model, note = inferencex._master_model("M", [{"a": {"model": "x/M"}, "b": {"model": "y/m"}}])

    assert model == "M"
    assert note is not None and "x/M" in note and "y/m" in note


def test_resolve_prefers_master_and_strips_runner_owned(checkout):
    launch = resolve(_query(), ResolveInputs(inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.INFERENCEX_MASTER
    assert launch.transfer_backend == "mori"
    assert launch.role(Role.PREFILL).args == ("--mem-fraction-static", "0.8")
    assert launch.role(Role.PREFILL).env == {"MORI_SHMEM_MODE": "ISOLATION"}
    assert "--model-path" in launch.role(Role.PREFILL).stripped and "env HF_HOME" in launch.role(Role.PREFILL).stripped
    assert launch.frontend.args == ("--policy", "round_robin")
    assert set(launch.source.files) == {MASTER, RUNNERS, launch.source.path}


def test_operator_args_win(checkout):
    operator = OperatorArgs(prefill=("--x",), decode=("--tp-size", "8", "--y"))
    launch = resolve(_query(), ResolveInputs(operator=operator, inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.OPERATOR
    assert (launch.role(Role.PREFILL).args, launch.role(Role.DECODE).args) == (("--x",), ("--y",))


def test_launch_names_the_model_and_its_context(checkout):
    launch = resolve(_query(), ResolveInputs(inferencex=lambda: checkout))
    assert (launch.model, launch.served_model_name, launch.context_length) == ("/models/M", "org/M", 2048)


def test_replay_keeps_the_launch_on_new_hosts(checkout):
    first = resolve(_query(), ResolveInputs(inferencex=lambda: checkout))
    replayed = resolve(_query(router_host="r1"), ResolveInputs(replay=first))
    assert replayed.digest() == first.digest() and replayed.frontend.host == "r1"
    with pytest.raises(PdResolveError, match="hosts"):
        resolve(_query(decode=2), ResolveInputs(replay=first))
    with pytest.raises(PdResolveError, match="serves"):
        resolve(_query(model_path="/models/N"), ResolveInputs(replay=first))


def _unreadable():
    raise SourceFetchError("offline")


def test_unreadable_inferencex_falls_to_the_table():
    query = _query(model_id="Qwen/Qwen3.5-397B-A17B-FP8", model_path="/m/Qwen3.5-397B-A17B-FP8")
    launch = resolve(query, ResolveInputs(inferencex=_unreadable))
    assert launch.source.kind is PdSourceKind.PD_TABLE
    assert launch.role(Role.PREFILL).args[0] == "--trust-remote-code"
    assert any("offline" in warning for warning in launch.source.warnings)


def test_the_table_launch_names_why_each_inferencex_source_passed(checkout):
    query = _query(model_id="Qwen/Qwen3.5-397B-A17B-FP8", model_path="/m/Qwen3.5-397B-A17B-FP8", board="mi300x")
    launch = resolve(query, ResolveInputs(inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.PD_TABLE
    warnings = "\n".join(launch.source.warnings)
    assert "no multi-node disaggregated entry serves Qwen/Qwen3.5-397B-A17B-FP8 with sglang" in warnings


_MALFORMED_MASTER = """
q-fp8-mi355x-sglang-disagg:
  model: Qwen/Qwen3.5-397B-A17B-FP8
  model-prefix: q
  runner: cluster:mi355x-amds
  framework: sglang-disagg
  multinode: true
  disagg: true
  srt-recipe-dir: q/sglang/mi355x-fp8
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - prefill: {num-worker: 1, tp: 8}
        decode: {num-worker: 1, tp: 8}
        conc-list: [64]
      - srt-recipe: 1k1k/base.yaml
        prefill: {num-worker: 1, tp: 8}
        decode: {num-worker: 1, tp: 8}
        conc-list: [64]
"""


def test_malformed_master_rows_miss_and_resolve_falls_to_the_table(tmp_path):
    files = {
        MASTER: _MALFORMED_MASTER,
        RUNNERS: _RUNNERS,
        f"{MULTI_NODE_RECIPES}/q/sglang/mi355x-fp8/1k1k/base.yaml": "base: {engine: sglang}\n",
    }
    malformed = _write_checkout(tmp_path, files)
    query = _query(model_id="Qwen/Qwen3.5-397B-A17B-FP8", model_path="/m/Qwen3.5-397B-A17B-FP8")

    found = master_candidates(query, malformed)
    assert found.found == ()
    assert sum("malformed row" in miss for miss in found.misses) == 2
    assert resolve(query, ResolveInputs(inferencex=lambda: malformed)).source.kind is PdSourceKind.PD_TABLE


@pytest.mark.parametrize(("tuned", "asked"), [("mi325x", "mi300x"), ("mi300x", "mi325x")])
def test_a_pd_session_refuses_a_launch_tuned_on_the_other_gfx942_board(tmp_path, tuned, asked):
    files = {
        MASTER: _MASTER.replace("mi355x", tuned),
        RUNNERS: _RUNNERS.replace("mi355x", tuned),
        f"{MULTI_NODE_RECIPES}/m/sglang/{tuned}-fp8/1k1k/disagg.yaml": _RECIPE.replace("mi355x", tuned),
    }
    board_checkout = _write_checkout(tmp_path, files)

    master = master_candidates(_query(board=asked), board_checkout)

    assert master.found == ()
    assert any(f"it runs on {tuned}, not {asked}" in miss for miss in master.misses)


def test_recipe_fit_reads_each_frameworks_topology_spelling():
    sglang = {"workers": 2, "gpus": 8, "args": {"tp-size": 8, "ep-size": 4}}
    assert recipe_fit(sglang, Framework.SGLANG) == RoleFit(2, 8, 4, False, 8)
    atom = {"workers": 1, "gpus": 8, "args": {"tp": 4, "enable-expert-parallel": True}}
    assert recipe_fit(atom, Framework.ATOM) == RoleFit(1, 4, 4, False, 8)


def test_pinned_unreadable_inferencex_is_refused():
    with pytest.raises(PdResolveError, match="pinned"):
        resolve(_query(), ResolveInputs(inferencex=_unreadable, pinned=True))


def test_no_source_is_refused(checkout):
    with pytest.raises(PdResolveError, match="no source serves"):
        resolve(_query(decode=3), ResolveInputs(inferencex=lambda: checkout))


def test_inferencex_recipes_for_another_router_miss(checkout):
    with pytest.raises(PdResolveError, match="sglang-router frontend, not atomesh"):
        resolve(_query(frontend=FrontendType.ATOMESH), ResolveInputs(inferencex=lambda: checkout))


def test_table_runs_the_asked_router_without_the_other_routers_args():
    query = _query(model_id="Qwen/Qwen3.5-397B-A17B-FP8", model_path="/m/Qwen3.5-397B-A17B-FP8")
    default = resolve(query, ResolveInputs(inferencex=_unreadable))
    asked = resolve(replace(query, frontend=FrontendType.ATOMESH), ResolveInputs(inferencex=_unreadable))
    assert default.frontend.type is FrontendType.SGLANG_ROUTER and default.frontend.args
    assert (asked.frontend.type, asked.frontend.args) == (FrontendType.ATOMESH, ())
    assert asked.role(Role.PREFILL).args == default.role(Role.PREFILL).args


def test_infera_serves_sglang_through_the_platform_frontend():
    assert _query(frontend=FrontendType.INFERA).served_frontend is FrontendType.INFERA


def _aggregated(workers=2, **fields):
    query = _query(roles={Role.WORKER: RoleAsk(tuple(f"w{i}" for i in range(workers)))}, mode=PdMode.AGGREGATED)
    return replace(query, frontend=FrontendType.INFERA, **fields)


def _base_launch(tp=8):
    return types.SimpleNamespace(
        topology=Topology(tp=tp, ep=1),
        args=("--tp-size", str(tp), "--enable-dp-attention", "--mem-fraction-static", "0.8"),
        env={"SGLANG_X": "1", "HF_TOKEN": Redacted("ab" * 32)},
        stripped=("--port",),
        warnings=("tuned elsewhere",),
        source=LaunchSource(SourceKind.DEFAULT_TABLE, path="models.yaml", entry="M"),
    )


def test_aggregated_workers_run_the_single_node_base(checkout):
    launch = resolve(_aggregated(), ResolveInputs(inferencex=lambda: checkout, base=_base_launch))
    worker = launch.role(Role.WORKER)
    assert launch.mode is PdMode.AGGREGATED and launch.source.kind is PdSourceKind.BASE_LAUNCH
    assert (worker.hosts, worker.workers, worker.tp, worker.dp_attention) == (("w0", "w1"), 2, 8, True)
    assert worker.args == ("--enable-dp-attention", "--mem-fraction-static", "0.8")
    assert worker.env == {"SGLANG_X": "1"} and "env HF_TOKEN" in worker.stripped and "--tp-size" in worker.stripped
    assert launch.transfer_backend == "" and launch.frontend.type is FrontendType.INFERA
    assert "tuned elsewhere" in launch.source.warnings and launch.source.entry == "M"


def test_a_base_spanning_hosts_is_one_worker():
    launch = resolve(_aggregated(), ResolveInputs(base=lambda: _base_launch(tp=16)))
    assert (launch.role(Role.WORKER).workers, launch.role(Role.WORKER).tp) == (1, 16)


def test_aggregated_operator_args_win_over_the_base():
    operator = OperatorArgs(worker=("--tp-size", "16", "--x"))
    launch = resolve(_aggregated(), ResolveInputs(operator=operator, base=lambda: pytest.fail("resolved a base")))
    assert launch.source.kind is PdSourceKind.OPERATOR and launch.role(Role.WORKER).args == ("--x",)


def test_aggregated_session_without_a_base_is_refused():
    with pytest.raises(PdResolveError, match="no source serves the workers"):
        resolve(_aggregated(), ResolveInputs(base=lambda: Miss(())))
    with pytest.raises(PdResolveError, match="no resolver"):
        resolve(_aggregated(), ResolveInputs())


def test_replay_keeps_the_mode():
    first = resolve(_aggregated(), ResolveInputs(base=_base_launch))
    assert resolve(_aggregated(router_host="r9"), ResolveInputs(replay=first)).digest() == first.digest()
    with pytest.raises(PdResolveError, match="aggregated, not disaggregated"):
        resolve(_query(), ResolveInputs(replay=first))


def test_operator_topology_is_the_launch_topology():
    operator = OperatorArgs(decode=("--tp-size", "16", "--ep-size", "16", "--enable-dp-attention", "--dp-size", "16"))
    decode = resolve(_query(decode=2), ResolveInputs(operator=operator)).role(Role.DECODE)
    assert (decode.tp, decode.ep, decode.dp_attention, decode.workers) == (16, 16, True, 1)


def test_operator_topology_conflicting_with_the_node_list_is_refused():
    query = replace(_query(), roles={Role.PREFILL: RoleAsk(("p0",), tp=8), Role.DECODE: RoleAsk(("d0",))})
    with pytest.raises(PdResolveError, match="prefill tp"):
        resolve(query, ResolveInputs(operator=OperatorArgs(prefill=("--tp-size", "4"), decode=("--x",))))


def test_router_args_alone_replace_the_source_router(checkout):
    operator = OperatorArgs(router=("--policy", "cache_aware"))
    launch = resolve(_query(), ResolveInputs(operator=operator, inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.INFERENCEX_MASTER
    assert launch.frontend.args == ("--policy", "cache_aware")
