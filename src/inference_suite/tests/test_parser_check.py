# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""D7: the installed parser, run in the serving interpreter over a composed launch, refuses a flag it does not have
and names the layer that set it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from inference_suite.contracts import FailedStep, LaunchDelta
from inference_suite.contracts.launch import LayerSource
from inference_suite.framework_probe import FrameworkProbeError
from inference_suite.frameworks import Framework
from inference_suite.parser_check import (
    ParserCheckError,
    ParseStatus,
    RefusedFlag,
    check,
    parser_probe,
    read_answer,
)
from inference_suite.tests.test_compose import _launch

#: A directory that holds no package, for the interpreter to start in.
_NEUTRAL = Path("/")

#: The installed sglang's argument surface: every flag the composed test launch carries, plus a choice-checked one.
_SERVER_ARGS = """
class ServerArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--tp", "--tensor-parallel-size", type=int)
        for flag in ("--host", "--port", "--served-model-name", "--context-length", "--json-model-override-args",
                     "--chunked-prefill-size", "--speculative-algorithm", "--speculative-num-steps",
                     "--watchdog-timeout"):
            parser.add_argument(flag)
        for switch in ("--disable-radix-cache", "--enable-metrics"):
            parser.add_argument(switch, action="store_true")
        parser.add_argument("--attention-backend", choices=("triton", "aiter"))
        return parser
"""


_RAISING_SERVER_ARGS = """
import argparse

class _Expand(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        raise RuntimeError("config expansion failed")

class ServerArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--config", action=_Expand)
        return parser
"""


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    srt = tmp_path / "sglang" / "srt"
    srt.mkdir(parents=True)
    (tmp_path / "sglang" / "__init__.py").write_text("")
    (srt / "__init__.py").write_text("")
    (srt / "server_args.py").write_text(_SERVER_ARGS)
    return {"PYTHONPATH": str(tmp_path), "PATH": os.environ.get("PATH", "")}


def test_a_launch_the_parser_accepts_passes(env):
    assert (
        check(
            _launch(candidate=LaunchDelta(args_add=("--chunked-prefill-size", "8192"))),
            sys.executable,
            env,
            cwd=_NEUTRAL,
        )
        is None
    )


def test_an_unknown_candidate_flag_is_flag_unknown_from_the_candidate(env):
    launch = _launch(candidate=LaunchDelta(args_add=("--moe-backend", "triton")))
    refusal = check(launch, sys.executable, env, cwd=_NEUTRAL)

    assert refusal is not None
    assert refusal.step is FailedStep.FLAG_UNKNOWN
    assert refusal.flags == (RefusedFlag("--moe-backend", LayerSource.CANDIDATE),)
    assert refusal.source is LayerSource.CANDIDATE
    assert "--moe-backend" in refusal.detail
    assert "--moe-backend" in launch.argv


def test_an_unknown_user_flag_beside_a_candidate_one_is_charged_to_the_user(env):
    launch = _launch(
        user=LaunchDelta(args_add=("--max-model-len", "8192")),
        candidate=LaunchDelta(args_add=("--moe-backend", "triton")),
    )
    refusal = check(launch, sys.executable, env, cwd=_NEUTRAL)

    assert refusal is not None
    assert set(refusal.flags) == {
        RefusedFlag("--max-model-len", LayerSource.USER),
        RefusedFlag("--moe-backend", LayerSource.CANDIDATE),
    }
    assert refusal.source is LayerSource.USER


def test_a_rejected_value_is_refused_and_names_the_layer_that_set_it(env):
    refusal = check(
        _launch(candidate=LaunchDelta(args_add=("--attention-backend", "fa4"))), sys.executable, env, cwd=_NEUTRAL
    )

    assert refusal is not None
    assert refusal.step is FailedStep.FLAG_UNKNOWN
    assert refusal.flags == (RefusedFlag("--attention-backend", LayerSource.CANDIDATE),)
    assert "invalid choice" in refusal.detail


def test_an_interpreter_without_the_framework_is_a_probe_error(tmp_path):
    with pytest.raises(FrameworkProbeError, match="no answer"):
        check(_launch(), sys.executable, {"PYTHONPATH": str(tmp_path)}, cwd=_NEUTRAL)


def test_a_parser_that_raises_is_a_probe_error_carrying_where_it_raised(env):
    server_args = Path(env["PYTHONPATH"]) / "sglang" / "srt" / "server_args.py"
    server_args.write_text(_RAISING_SERVER_ARGS)

    with pytest.raises(FrameworkProbeError, match="RuntimeError: config expansion failed") as raised:
        check(_launch(candidate=LaunchDelta(args_add=("--config", "x.yaml"))), sys.executable, env, cwd=_NEUTRAL)
    assert f'File "{server_args}", line 6, in __call__' in str(raised.value)


def test_a_framework_with_no_server_parser_has_no_probe():
    assert parser_probe(Framework.XDIT, ("--x",)) is None
    assert parser_probe(Framework.CUSTOM, ()) is None
    assert all(parser_probe(item, ()) is not None for item in (Framework.SGLANG, Framework.VLLM, Framework.ATOM))


def test_an_answer_reads_unknown_flags_without_values_and_once():
    answer = read_answer({"status": "invalid", "message": "unrecognized", "unknown": ["--a=1", "x", "--a", "--b"]})
    assert answer.status is ParseStatus.REJECTED
    assert answer.unknown_flags == ("--a", "--b")


def test_an_answer_names_every_spelling_of_the_rejected_argument():
    answer = read_answer({"status": "invalid", "message": "argument --tp/--tensor-parallel-size: invalid int"})
    assert answer.unknown_flags == ()
    assert answer.rejected_option == ("--tp", "--tensor-parallel-size")


def test_an_answer_with_a_status_the_program_never_prints_is_an_error():
    with pytest.raises(ParserCheckError, match="'maybe'"):
        read_answer({"status": "maybe"})
