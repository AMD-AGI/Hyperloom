# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import textwrap
from dataclasses import replace

import pytest

from inference_suite.contracts.base_launch import Workload
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
from inference_suite.multinode.record import Role
from inference_suite.recipe.inferencex import MASTER, RUNNERS
from inference_suite.recipe.sources import Checkout

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


def test_a_pd_session_refuses_a_launch_tuned_on_another_board(tmp_path):
    runners = "labels:\n  mi325x: [n1, n2]\n  cluster:mi325x-amds: [n1]\n"
    master = _MASTER.replace("cluster:mi355x-amds", "cluster:mi325x-amds")
    recipe = f"{MULTI_NODE_RECIPES}/m/sglang/mi355x-fp8/1k1k/disagg.yaml"
    checkout = _write_checkout(tmp_path, {MASTER: master, RUNNERS: runners, recipe: _RECIPE})
    found = master_candidates(_query(board="mi300x"), checkout)
    assert found.found == ()
    assert any("it runs on mi325x, not mi300x" in miss for miss in found.misses)


def test_recipe_fit_reads_each_frameworks_topology_spelling():
    sglang = {"workers": 2, "gpus": 8, "args": {"tp-size": 8, "ep-size": 4}}
    assert recipe_fit(sglang, Framework.SGLANG) == RoleFit(2, 8, 4, False, 8)
    atom = {"workers": 1, "gpus": 8, "args": {"tp": 4, "enable-expert-parallel": True}}
    assert recipe_fit(atom, Framework.ATOM) == RoleFit(1, 4, 4, False, 8)
