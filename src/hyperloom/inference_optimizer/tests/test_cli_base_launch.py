# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fresh session resolves its base launch once and freezes it as ``base_launch.json``; a resume reads that file.

The snapshot every test resolves from is the recorded one ``inference_suite``'s own recipe tests read, and subprocesses
and sockets are refused throughout.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import socket
import subprocess
from functools import cache
from pathlib import Path

import pytest

import inference_suite
from hyperloom.inference_optimizer.cli import base_launch as bl
from hyperloom.inference_optimizer.session.base_launch import BASE_LAUNCH_FILE, read_base_launch
from inference_suite import SourceSnapshot
from inference_suite.contracts import BaseLaunch, BaseLaunchError, Redacted, SourceKind, Topology
from inference_suite.recipe.model import InferencexSnapshot, RepoBundle
from inference_suite.recipe.vllm_recipes import VllmRecipesSnapshot

_FIXTURES = Path(inference_suite.__file__).parent / "tests" / "fixtures"
_DSR1 = "deepseek-ai/DeepSeek-R1-0528"


def _load(*parts: str) -> dict:
    return json.loads(gzip.decompress(_FIXTURES.joinpath(*parts).read_bytes()))


@cache
def _snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        InferencexSnapshot.from_document(_load("inferencex", "ce35bc935.json.gz")["snapshot"]),
        1_793_000_000,
        sglang_cookbook=RepoBundle.from_document(_load("sglang", "c843608d39.json.gz")),
        vllm_recipes=VllmRecipesSnapshot.from_document(_load("vllm_recipes", "20261106.json.gz")),
        atom_catalog=RepoBundle.from_document(_load("atom", "b90ae9990.json.gz")),
        latest_releases={"sglang": "v0.5.23", "vllm": "v0.32.1", "atom": "v0.1.3"},
    )


@pytest.fixture(autouse=True)
def _no_io(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the test ran a subprocess or opened a connection")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(autouse=True)
def visible_gpus(monkeypatch) -> list[int]:
    """The GPU count this host reports; a test sets ``[0]`` to change it."""
    count = [8]
    monkeypatch.setattr(bl, "detect_gpu_count", lambda: count[0])
    return count


@pytest.fixture(autouse=True)
def _session_topology_env(monkeypatch) -> None:
    monkeypatch.setenv("TP", "1")
    monkeypatch.setenv("EP", "1")


@pytest.fixture
def fetched(monkeypatch) -> list:
    """Every ``(query, user_file)`` the sources were read for."""
    calls: list = []

    def fetch(query, *, user_file=None):
        calls.append((query, user_file))
        return dataclasses.replace(_snapshot(), user_file=user_file)

    monkeypatch.setattr(bl, "fetch_sources", fetch)
    return calls


def _args(**overrides) -> argparse.Namespace:
    values = dict(model=_DSR1, gpu_type="mi355x", operator_tp=8, operator_ep=1, isl=8192, osl=1024, reference_script="")
    values.update(overrides)
    return argparse.Namespace(**values)


def _resolve(session_dir: Path, args: argparse.Namespace, **overrides) -> BaseLaunch | None:
    kwargs = dict(framework="sglang", benchmark_mode="synthetic", concurrencies=(64, 4, 64))
    kwargs.update(overrides)
    return bl.resolve_base_launch(args, session_dir, **kwargs)


def _refused(session_dir: Path, args: argparse.Namespace, capsys, **overrides) -> str:
    with pytest.raises(SystemExit) as exc:
        _resolve(session_dir, args, **overrides)
    assert exc.value.code == 2
    assert not (session_dir / BASE_LAUNCH_FILE).exists()
    return capsys.readouterr().err


def test_a_fresh_session_freezes_the_recipe_base_it_resolves(tmp_path, fetched, capsys):
    base = _resolve(tmp_path, _args())

    assert base is not None and base.source.kind is SourceKind.INFERENCEX
    query = fetched[0][0]
    assert (query.board, query.tp, query.ep, query.concurrencies, fetched[0][1]) == ("mi355x", 8, 1, (4, 64), None)
    assert read_base_launch(tmp_path) == base
    assert "Base launch     : inferencex" in capsys.readouterr().out
    assert bl.reference_launch_fields(base) == {
        "reference_server_args": " ".join(base.args),
        "reference_envs": dict(base.env),
    }


def test_a_model_no_recipe_serves_runs_the_default_table_and_says_so(tmp_path, fetched, capsys):
    base = _resolve(tmp_path, _args(model="Kimi-K2.6", gpu_type="mi300x", operator_tp=1))

    assert base is not None and base.source.kind is SourceKind.DEFAULT_TABLE
    assert base.topology == Topology(1)
    assert "no recipe serves Kimi-K2.6; the default sglang launch for gfx942 runs it" in capsys.readouterr().err


def test_the_operator_file_is_the_base_and_its_credentials_stay_out_of_the_frozen_file(tmp_path, fetched):
    script = tmp_path / "launch.sh"
    script.write_text(
        "export SGLANG_USE_AITER=1\nexport HF_TOKEN=hunter2\n"
        "python3 -m sglang.launch_server --model-path /models/r1 --tp 4 --mem-fraction-static 0.7 --port 30000\n"
    )
    session = tmp_path / "session"
    session.mkdir()

    base = _resolve(session, _args(operator_tp=4, reference_script=str(script)))

    assert base is not None and base.source.kind is SourceKind.USER_FILE
    assert fetched[0][1].location == str(script)
    assert base.args == ("--tp", "4", "--mem-fraction-static", "0.7")
    assert isinstance(base.env["HF_TOKEN"], Redacted)
    assert "hunter2" not in (session / BASE_LAUNCH_FILE).read_text()
    assert bl.reference_launch_fields(base)["reference_envs"] == {"SGLANG_USE_AITER": "1"}


@pytest.mark.parametrize(
    ("script", "tp", "reason"),
    [
        ("python3 -m sglang.launch_server --tp $TP\n", 8, "from shell variables"),
        ('# hyperloom-launch-controls: {"args_mode": "replace"}\npython3 -m sglang.launch_server\n', 8, "controls"),
        ("echo hello\n", 8, "no sglang serving entrypoint"),
        ("python3 -m sglang.launch_server --tp 4\n", 8, "the query asks for tp 8, the launch runs 4"),
    ],
)
def test_an_operator_file_that_cannot_be_the_base_exits_2(tmp_path, fetched, capsys, script, tp, reason):
    path = tmp_path / "launch.sh"
    path.write_text(script)
    session = tmp_path / "session"
    session.mkdir()

    assert reason in _refused(session, _args(operator_tp=tp, reference_script=str(path)), capsys)


def test_an_unreadable_operator_file_exits_2_before_any_source_is_read(tmp_path, fetched, capsys):
    assert "could not be resolved" in _refused(tmp_path, _args(reference_script=str(tmp_path / "missing.sh")), capsys)
    assert fetched == []


def test_a_session_no_source_serves_exits_2_naming_every_miss(tmp_path, fetched, capsys):
    err = _refused(tmp_path, _args(model="Kimi-K2.6", gpu_type="r9700", operator_tp=1), capsys)

    assert "no source serves" in err
    assert "default_table: it has no sglang row for r9700 (gfx1201)" in err


@pytest.mark.parametrize(
    ("layout", "config", "expected"),
    [
        ("hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/abc", {}, _DSR1),
        ("models/DeepSeek-R1-0528", {"_name_or_path": _DSR1}, _DSR1),
        ("models/DeepSeek-R1-0528", {"_name_or_path": "/scratch/r1"}, "DeepSeek-R1-0528"),
        ("models/DeepSeek-R1-0528", None, "DeepSeek-R1-0528"),
    ],
)
def test_a_local_checkpoint_is_queried_by_the_hub_id_it_records(tmp_path, fetched, layout, config, expected):
    model = tmp_path / layout
    model.mkdir(parents=True)
    if config is not None:
        (model / "config.json").write_text(json.dumps(config))

    _resolve(tmp_path, _args(model=str(model), operator_tp=1))

    assert fetched[0][0].model_id == expected


def test_an_open_topology_is_chosen_by_the_recipe_for_the_visible_gpus_and_the_session_runs_it(tmp_path, fetched):
    args = _args(tp=1, ep=1, operator_tp=None, operator_ep=None)
    base = _resolve(tmp_path, args)

    query = fetched[0][0]
    assert (query.tp, query.ep) == (None, None)
    assert base is not None and base.source.kind is SourceKind.INFERENCEX
    assert base.topology.tp == 8
    assert (args.tp, args.ep) == (base.topology.tp, base.topology.ep)
    assert (os.environ["TP"], os.environ["EP"]) == (str(base.topology.tp), str(base.topology.ep))


def test_an_open_topology_takes_the_largest_recipe_that_fits_the_visible_gpus(tmp_path, fetched, visible_gpus):
    (tmp_path / "wide").mkdir()
    (tmp_path / "full").mkdir()
    visible_gpus[0] = 4
    wide = _resolve(tmp_path / "wide", _args(operator_tp=None, operator_ep=None))
    visible_gpus[0] = 8
    full = _resolve(tmp_path / "full", _args(operator_tp=None, operator_ep=None))

    assert wide is not None and full is not None
    assert wide.topology.tp <= 4 < full.topology.tp


def test_a_host_with_no_visible_gpu_exits_2(tmp_path, fetched, visible_gpus, capsys):
    visible_gpus[0] = 0

    assert "none is visible" in _refused(tmp_path, _args(), capsys)
    assert fetched == []


def test_an_operator_file_sets_the_topology_the_command_line_left_open(tmp_path, fetched):
    script = tmp_path / "launch.sh"
    script.write_text("python3 -m sglang.launch_server --model-path /models/r1 --tp 8\n")
    session = tmp_path / "session"
    session.mkdir()

    args = _args(tp=1, ep=1, operator_tp=None, operator_ep=None, reference_script=str(script))
    base = _resolve(session, args)

    assert base is not None and base.source.kind is SourceKind.USER_FILE
    assert base.topology.tp == 8 and base.query.tp is None
    assert args.tp == 8 and os.environ["TP"] == "8"


def test_a_multi_node_session_fits_the_base_to_every_node_and_needs_no_gpu_on_the_head(
    tmp_path, fetched, visible_gpus, monkeypatch
):
    visible_gpus[0] = 0
    spans: list[int] = []
    resolve = bl.resolve_base

    def spy(query, snapshot, *, gpus):
        spans.append(gpus)
        return resolve(query, snapshot, gpus=gpus)

    monkeypatch.setattr(bl, "resolve_base", spy)
    args = _args(nodes=2, gpus_per_node=8, tp=16, operator_tp=16, operator_ep=None)

    base = _resolve(tmp_path, args)

    assert spans == [16]
    assert base is not None and base.topology.tp == 16
    assert args.tp == 16 and os.environ["TP"] == "16"


def test_a_hub_id_is_queried_with_its_org_and_case(tmp_path, fetched):
    _resolve(tmp_path, _args(model="deepseek-ai/DeepSeek-R1-0528/"))

    assert fetched[0][0].model_id == _DSR1


def test_a_display_name_is_the_query_model(tmp_path, fetched):
    _resolve(tmp_path, _args(model_display_name="DeepSeek-R1-0528-quantized", operator_tp=1))

    assert fetched[0][0].model_id == "DeepSeek-R1-0528-quantized"


def test_a_malformed_query_exits_2(tmp_path, fetched, capsys):
    assert "a query concurrency" in _refused(tmp_path, _args(), capsys, concurrencies=(0, 4))
    assert fetched == []


def test_a_session_without_a_board_exits_2(tmp_path, fetched, capsys):
    assert "--gpu-type" in _refused(tmp_path, _args(gpu_type=None), capsys)
    assert fetched == []


@pytest.mark.parametrize(("framework", "mode"), [("xdit", "synthetic"), ("sglang", "agentx")])
def test_scriptable_and_agentx_sessions_resolve_no_base(tmp_path, fetched, framework, mode):
    assert _resolve(tmp_path, _args(), framework=framework, benchmark_mode=mode) is None
    assert fetched == []
    assert read_base_launch(tmp_path) is None


def test_a_launch_file_for_a_scriptable_workload_exits_2(tmp_path, fetched, capsys):
    err = _refused(tmp_path, _args(reference_script="launch.sh"), capsys, framework="custom")
    assert "launches no server" in err


def test_a_resume_reads_the_frozen_base_without_reading_a_source(tmp_path, fetched, monkeypatch):
    base = _resolve(tmp_path, _args())

    def unreachable(*args, **kwargs):
        raise AssertionError("a resume read a source")

    monkeypatch.setattr(bl, "fetch_sources", unreachable)
    assert read_base_launch(tmp_path) == base

    document = json.loads((tmp_path / BASE_LAUNCH_FILE).read_text())
    document["args"].append("--disable-cuda-graph")
    (tmp_path / BASE_LAUNCH_FILE).write_text(json.dumps(document))
    with pytest.raises(BaseLaunchError, match="base_digest_mismatch"):
        read_base_launch(tmp_path)
