# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import textwrap
from dataclasses import replace

import pytest

from inference_suite.contracts.base_launch import Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.inferencex import (
    MULTI_NODE_RECIPES,
    deep_merge,
    expand_variants,
    lookup_candidates,
    master_candidates,
)
from inference_suite.multinode.query import PdQuery, RoleAsk
from inference_suite.multinode.record import PdSourceKind
from inference_suite.multinode.recipe import OperatorArgs, PdResolveError, ResolveInputs, resolve
from inference_suite.recipe.inferencex import MASTER, RUNNERS
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


@pytest.fixture
def checkout(tmp_path):
    files = {
        MASTER: _MASTER,
        RUNNERS: _RUNNERS,
        f"{MULTI_NODE_RECIPES}/m/sglang/mi355x-fp8/1k1k/disagg.yaml": _RECIPE,
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(textwrap.dedent(text))
    return Checkout("https://example/InferenceX", "c0ffee", tmp_path)


def _query(prefill=1, decode=1, **fields):
    query = PdQuery(
        model_id="org/M",
        model_path="/models/M",
        framework=Framework.SGLANG,
        board="mi355x",
        gpus_per_node=8,
        workload=Workload.FIXED,
        concurrencies=(64,),
        prefill=RoleAsk(tuple(f"p{i}" for i in range(prefill))),
        decode=RoleAsk(tuple(f"d{i}" for i in range(decode))),
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


def test_lookup_finds_variants_no_master_entry_names(checkout):
    found = lookup_candidates(_query(decode=2), checkout)
    assert [c.locator for c in found.found] == ["zip_override_1p2d[0]", "zip_override_1p2d[1]"]
    assert all(c.kind is PdSourceKind.INFERENCEX_LOOKUP and c.warnings for c in found.found)


def test_resolve_prefers_master_and_strips_runner_owned(checkout):
    launch = resolve(_query(), ResolveInputs(inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.INFERENCEX_MASTER
    assert launch.transfer_backend == "mori"
    assert launch.prefill.args == ("--mem-fraction-static", "0.8")
    assert launch.prefill.env == {"MORI_SHMEM_MODE": "ISOLATION"}
    assert "--model-path" in launch.prefill.stripped and "env HF_HOME" in launch.prefill.stripped
    assert launch.frontend.args == ("--policy", "round_robin")
    assert set(launch.source.files) == {MASTER, RUNNERS, launch.source.path}


def test_resolve_falls_to_lookup(checkout):
    launch = resolve(_query(decode=2), ResolveInputs(inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.INFERENCEX_LOOKUP
    assert launch.decode.workers == 2
    assert launch.source.alternatives == (f"{launch.source.path}:zip_override_1p2d[1]",)


def test_operator_args_win(checkout):
    operator = OperatorArgs(shared=("--x",), decode=("--tp-size", "8", "--y"))
    launch = resolve(_query(), ResolveInputs(operator=operator, inferencex=lambda: checkout))
    assert launch.source.kind is PdSourceKind.OPERATOR
    assert (launch.prefill.args, launch.decode.args) == (("--x",), ("--x", "--y"))


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
    assert launch.prefill.args[0] == "--trust-remote-code"
    assert any("offline" in warning for warning in launch.source.warnings)


def test_pinned_unreadable_inferencex_is_refused():
    with pytest.raises(PdResolveError, match="pinned"):
        resolve(_query(), ResolveInputs(inferencex=_unreadable, pinned=True))


def test_no_source_is_refused(checkout):
    with pytest.raises(PdResolveError, match="no source serves"):
        resolve(_query(decode=3), ResolveInputs(inferencex=lambda: checkout))
