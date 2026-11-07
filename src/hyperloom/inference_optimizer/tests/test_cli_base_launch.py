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
from hyperloom.inference_optimizer.cli.lineup_banners import print_lineup_banners
from hyperloom.inference_optimizer.session.base_launch import BASE_LAUNCH_FILE, read_base_launch, shared_state_fields
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite import SourceFetchError, SourceSnapshot
from inference_suite.contracts import BaseLaunch, BaseLaunchError, Redacted, SourceKind, Topology, Workload
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT
from inference_suite.recipe import sources
from inference_suite.recipe.model import InferencexSnapshot, RepoBundle
from inference_suite.recipe.resolve import UnreadSource
from inference_suite.recipe.vllm_recipes import VllmRecipesSnapshot

_FIXTURES = Path(inference_suite.__file__).parent / "tests" / "fixtures"
_DSR1 = "deepseek-ai/DeepSeek-R1-0528"
_GLM52 = "zai-org/GLM-5.2-FP8"
_REFERENCE_FIELDS = ("reference_server_args", "reference_envs", "reference_launch_controls", "reference_model")


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
def _session_topology_env(monkeypatch) -> None:
    monkeypatch.setenv("TP", "1")
    monkeypatch.setenv("EP", "1")


@pytest.fixture
def spans(monkeypatch) -> list[int]:
    """The GPU count every resolve was asked to fit."""
    seen: list[int] = []
    resolve = bl.resolve_base

    def spy(query, snapshot, *, gpus):
        seen.append(gpus)
        return resolve(query, snapshot, gpus=gpus)

    monkeypatch.setattr(bl, "resolve_base", spy)
    return seen


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
    values = dict(
        model=_DSR1,
        gpu_type="mi355x",
        operator_tp=8,
        operator_ep=1,
        isl=8192,
        osl=1024,
        reference_script="",
        nodes=1,
        gpus_per_node=None,
    )
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
    assert (base.model.id, base.model.path) == (_DSR1, _DSR1)
    assert "Base launch     : inferencex" in capsys.readouterr().out
    assert shared_state_fields(base) == {"base_launch_digest": base.digest()}


def test_a_new_session_state_carries_no_reference_launch(tmp_path, fetched):
    base = _resolve(tmp_path, _args())

    state = SharedState(session_id="s", **shared_state_fields(base))
    state.save(tmp_path)
    saved = json.loads((tmp_path / "state.json").read_text())

    assert saved["base_launch_digest"] == base.digest()
    assert not set(_REFERENCE_FIELDS) & saved.keys()


def test_the_seeded_state_names_the_frozen_file_by_its_digest_across_a_resume(tmp_path, fetched):
    base = _resolve(tmp_path, _args())

    state = SharedState(session_id="s", **shared_state_fields(base))
    state.save(tmp_path)
    resumed = SharedState.from_dict(json.loads((tmp_path / "state.json").read_text()))

    frozen = read_base_launch(tmp_path)
    assert frozen is not None
    assert resumed.base_launch_digest == frozen.digest() == base.digest()
    assert resumed.base_launch_digest == json.loads((tmp_path / BASE_LAUNCH_FILE).read_text())["digest"]


def test_a_session_that_froze_no_base_seeds_no_base_fields():
    assert shared_state_fields(None) == {}


def _banner_err(session_dir: Path, monkeypatch, capsys) -> str:
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session_dir))
    capsys.readouterr()
    print_lineup_banners(argparse.Namespace(), no_kernel=False, no_framework_agent=False)
    return capsys.readouterr().err


def test_a_model_no_recipe_serves_runs_the_default_table_and_the_banner_says_so(tmp_path, fetched, monkeypatch, capsys):
    base = _resolve(tmp_path, _args(model="Kimi-K2.6", gpu_type="mi300x", operator_tp=1))

    assert base is not None and base.source.kind is SourceKind.DEFAULT_TABLE
    assert base.topology == Topology(1)
    assert "WARNING: base launch" not in capsys.readouterr().err
    assert "no recipe serves Kimi-K2.6; the default sglang launch for gfx942 runs it" in _banner_err(
        tmp_path, monkeypatch, capsys
    )


def test_a_session_that_froze_no_base_has_no_base_launch_warnings_in_its_banner(tmp_path, monkeypatch, capsys):
    assert "base launch" not in _banner_err(tmp_path, monkeypatch, capsys)


def test_a_fixed_session_whose_recipe_source_cannot_be_read_freezes_the_default_table_and_says_why(
    tmp_path, monkeypatch, capsys
):
    unread = UnreadSource("the sglang cookbook", "git clone exited 128: could not resolve host")
    monkeypatch.setattr(
        bl, "fetch_sources", lambda query, *, user_file=None: dataclasses.replace(_snapshot(), unread=unread)
    )

    base = _resolve(tmp_path, _args())

    assert base is not None and base.source.kind is SourceKind.DEFAULT_TABLE
    frozen = json.loads((tmp_path / BASE_LAUNCH_FILE).read_text())
    assert frozen["source"]["kind"] == "default_table"
    assert any(
        "the sglang cookbook could not be read" in warning and unread.reason in warning
        for warning in frozen["warnings"]
    )
    assert "WARNING: base launch: the sglang cookbook could not be read" in _banner_err(tmp_path, monkeypatch, capsys)


def test_a_fixed_session_whose_inferencex_cannot_be_read_freezes_the_default_table_with_the_pinned_client(
    tmp_path, monkeypatch
):
    def unreachable(into):
        raise SourceFetchError("git clone exited 128: could not resolve host")

    monkeypatch.setattr(sources, "_inferencex", unreachable)

    base = _resolve(tmp_path, _args())

    frozen = json.loads((tmp_path / BASE_LAUNCH_FILE).read_text())
    assert base is not None and frozen["source"]["kind"] == "default_table"
    assert frozen["client"]["commit"] == INFERENCEX_CLIENT_COMMIT
    assert any(warning.startswith("InferenceX could not be read") for warning in frozen["warnings"])


def test_the_operator_file_is_the_base_and_its_credentials_stay_out_of_the_frozen_file(tmp_path, fetched):
    script = tmp_path / "launch.sh"
    script.write_text(
        "export SGLANG_USE_AITER=1\nexport HF_TOKEN=hunter2\nexport DEEPEP_COMBINE_TOKEN_BALANCED=1\n"
        "python3 -m sglang.launch_server --model-path /models/r1 --tp 4 --mem-fraction-static 0.7 --port 30000"
        " --prefill-delayer-token-usage-low-watermark 0.7\n"
    )
    session = tmp_path / "session"
    session.mkdir()

    base = _resolve(session, _args(operator_tp=4, reference_script=str(script)))

    assert base is not None and base.source.kind is SourceKind.USER_FILE
    assert fetched[0][1].location == str(script)
    assert base.args == (
        "--tp",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--prefill-delayer-token-usage-low-watermark",
        "0.7",
    )
    assert isinstance(base.env["HF_TOKEN"], Redacted)
    assert "hunter2" not in (session / BASE_LAUNCH_FILE).read_text()
    assert base.model.path == _DSR1
    assert base.env["SGLANG_USE_AITER"] == "1"
    assert base.env["DEEPEP_COMBINE_TOKEN_BALANCED"] == "1"


_PROBE_ENV = (
    "AWS_ACCESS_KEY_ID",
    "LLM_GATEWAY_KEY",
    "HYPERLOOM_SANDBOX_KEY_MEASUREMENT",
    "DB_PASSWD",
    "SSH_PRIVATE_KEY",
    "GPG_PASSPHRASE",
    "HF_AUTH",
    "MODELSCOPE_ACCESS_KEY",
)


def test_every_credential_named_env_of_the_operator_file_is_frozen_as_its_sha256(tmp_path, fetched):
    script = tmp_path / "launch.sh"
    exports = "".join(f"export {name}=leaked{index}\n" for index, name in enumerate(_PROBE_ENV))
    script.write_text(exports + "python3 -m sglang.launch_server --model-path /models/r1 --tp 4\n")
    session = tmp_path / "session"
    session.mkdir()

    base = _resolve(session, _args(operator_tp=4, reference_script=str(script)))

    assert base is not None and all(isinstance(base.env[name], Redacted) for name in _PROBE_ENV)
    assert "leaked" not in (session / BASE_LAUNCH_FILE).read_text()


@pytest.mark.parametrize(
    "credential",
    ["--api-key X", "--api-key=X", "--hf-token=hf_" + "q" * 34, "--hf-token hf_" + "q" * 34],
)
def test_an_operator_file_passing_a_credential_as_an_arg_exits_2_naming_where_never_the_value(
    tmp_path, fetched, capsys, credential
):
    script = tmp_path / "launch.sh"
    script.write_text(f"python3 -m sglang.launch_server --model-path /models/r1 --tp 4 {credential}\n")
    session = tmp_path / "session"
    session.mkdir()

    err = _refused(session, _args(operator_tp=4, reference_script=str(script)), capsys)

    assert "holds a credential in arg" in err and "through the environment" in err
    assert " X" not in err and "qqqq" not in err


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


def test_an_open_topology_is_chosen_by_the_recipe_for_eight_gpus_and_the_session_runs_it(tmp_path, fetched):
    args = _args(tp=1, ep=1, operator_tp=None, operator_ep=None)
    base = _resolve(tmp_path, args)

    query = fetched[0][0]
    assert (query.tp, query.ep) == (None, None)
    assert base is not None and base.source.kind is SourceKind.INFERENCEX
    assert base.topology.tp == 8
    assert (args.tp, args.ep) == (base.topology.tp, base.topology.ep)
    assert (os.environ["TP"], os.environ["EP"]) == (str(base.topology.tp), str(base.topology.ep))


def test_an_open_topology_takes_the_largest_recipe_that_fits_the_gpus_per_node(tmp_path, fetched):
    (tmp_path / "wide").mkdir()
    (tmp_path / "full").mkdir()
    wide = _resolve(tmp_path / "wide", _args(operator_tp=None, operator_ep=None, gpus_per_node=4))
    full = _resolve(tmp_path / "full", _args(operator_tp=None, operator_ep=None))

    assert wide is not None and full is not None
    assert wide.topology.tp <= 4 < full.topology.tp


def test_a_single_node_session_is_resolved_for_eight_gpus_without_probing_the_host(
    tmp_path, fetched, spans, monkeypatch
):
    from hyperloom.common import visible_devices

    def probe():
        raise AssertionError("the base launch probed this host's GPUs")

    monkeypatch.setattr(visible_devices, "detect_gpu_count", probe)

    base = _resolve(tmp_path, _args(operator_tp=None, operator_ep=None))

    assert spans == [8]
    assert base is not None and base.topology.tp == 8


def test_an_explicit_gpus_per_node_sizes_the_span(tmp_path, fetched, spans):
    _resolve(tmp_path, _args(operator_tp=None, operator_ep=None, gpus_per_node=4))

    assert spans == [4]


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


def test_a_multi_node_session_fits_the_base_to_eight_gpus_on_every_node(tmp_path, fetched, spans, monkeypatch):
    monkeypatch.delenv("TOPOLOGY_HANDOFF_FILE", raising=False)
    args = _args(nodes=2, tp=16, operator_tp=16, operator_ep=None)

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


def test_a_scriptable_session_resolves_no_base(tmp_path, fetched):
    assert _resolve(tmp_path, _args(), framework="xdit") is None
    assert fetched == []
    assert read_base_launch(tmp_path) is None


def test_an_mlperf_agentx_session_freezes_a_fixed_base_for_the_server_it_boots(tmp_path, fetched, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")

    base = _resolve(tmp_path, _args(), benchmark_mode="agentx", concurrencies=(8,))

    query = fetched[0][0]
    assert (query.workload, query.isl, query.osl, query.concurrencies) == (Workload.FIXED, 8192, 1024, (8,))
    assert base is not None and read_base_launch(tmp_path) == base


def _agentx(session_dir: Path, args: argparse.Namespace, **overrides) -> BaseLaunch | None:
    return _resolve(session_dir, args, benchmark_mode="agentx", concurrencies=(8,), **overrides)


def test_an_agentx_session_freezes_the_recipe_base_for_its_trace_replay(tmp_path, fetched):
    args = _args(model=_GLM52, tp=1, ep=1, operator_tp=None, operator_ep=None)

    base = _agentx(tmp_path, args)

    query = fetched[0][0]
    assert (query.workload, query.isl, query.osl, query.concurrencies) == (Workload.AGENTX, None, None, (8,))
    assert base is not None and base.source.kind is SourceKind.INFERENCEX
    assert read_base_launch(tmp_path) == base
    assert args.tp == base.topology.tp


def test_an_operator_file_is_the_agentx_base_ahead_of_a_recipe_that_serves_it(tmp_path, fetched):
    script = tmp_path / "launch.sh"
    script.write_text("python3 -m sglang.launch_server --model-path /models/glm --tp 8 --mem-fraction-static 0.7\n")
    session = tmp_path / "session"
    session.mkdir()

    base = _agentx(session, _args(model=_GLM52, operator_tp=8, reference_script=str(script)))

    assert base is not None and base.source.kind is SourceKind.USER_FILE
    assert base.query.workload is Workload.AGENTX
    assert base.args == ("--tp", "8", "--mem-fraction-static", "0.7")


def test_an_agentx_session_no_recipe_serves_exits_2_without_a_default(tmp_path, fetched, capsys):
    err = _refused(tmp_path, _args(model="Qwen/Qwen3-8B", operator_tp=1), capsys, benchmark_mode="agentx")

    assert "no source serves" in err
    assert "inferencex: no entry serves Qwen/Qwen3-8B with sglang on a mi355x" in err
    assert "default_table" not in err


def test_an_agentx_session_whose_sources_cannot_be_read_exits_2(tmp_path, monkeypatch, capsys):
    unread = UnreadSource("InferenceX main", "connection refused")
    monkeypatch.setattr(
        bl, "fetch_sources", lambda query, *, user_file=None: dataclasses.replace(_snapshot(), unread=unread)
    )

    err = _refused(tmp_path, _args(model=_GLM52, operator_tp=None, operator_ep=None), capsys, benchmark_mode="agentx")

    assert "InferenceX main could not be read: connection refused" in err


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
