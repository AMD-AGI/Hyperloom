# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""D7 over ATOM's two parser generations: the native ``FlexibleArgumentParser`` and the plain argparse one before it,
and over a parser that raises: an install that cannot answer is a probe error, never an accepted argv."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from inference_suite.framework_probe import FrameworkProbeError, run_probe
from inference_suite.frameworks import Framework
from inference_suite.parser_check import ParserAnswer, ParseStatus, parser_probe, read_answer

_ENGINE_ARGS = """
class EngineArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--kv_cache_dtype", choices=("bf16", "fp8"), default="bf16")
        return parser
"""

_NATIVE_PARSER = """
import argparse

class FlexibleArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *names, **kwargs):
        aliases = [name.replace("_", "-") for name in names if name.startswith("--") and "_" in name]
        return super().add_argument(*names, *aliases, **kwargs)
"""

#: ``None`` is an install that predates the native parser module.
_KV_CACHE_FLAGS = (
    pytest.param(_NATIVE_PARSER, "--kv_cache_dtype", id="native-snake"),
    pytest.param(_NATIVE_PARSER, "--kv-cache-dtype", id="native-kebab"),
    pytest.param(None, "--kv_cache_dtype", id="legacy-module-absent"),
)
_SPELLINGS = pytest.mark.parametrize("equals", (False, True), ids=("separate-value", "equals-value"))


@pytest.fixture
def atom(tmp_path: Path, request: pytest.FixtureRequest) -> dict[str, str]:
    site = tmp_path / "serve"
    package = site / "atom"
    for module in (package, package / "model_engine", package / "utils"):
        module.mkdir(parents=True, exist_ok=True)
        (module / "__init__.py").write_text("", encoding="utf-8")
    (package / "model_engine" / "arg_utils.py").write_text(_ENGINE_ARGS, encoding="utf-8")
    parser_source = getattr(request, "param", _NATIVE_PARSER)
    if parser_source is not None:
        (package / "utils" / "arg_parser.py").write_text(parser_source, encoding="utf-8")
    return {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(site)}


def _judge(argv: Sequence[str], env: dict[str, str]) -> ParserAnswer:
    probe = parser_probe(Framework.ATOM, argv)
    assert probe is not None
    return read_answer(run_probe(sys.executable, probe.program, probe.args, env, cwd=Path("/")))


def _kv(flag: str, value: str, equals: bool) -> tuple[str, ...]:
    return (f"{flag}={value}",) if equals else (flag, value)


@pytest.mark.parametrize(("atom", "flag"), _KV_CACHE_FLAGS, indirect=("atom",))
@_SPELLINGS
def test_every_kv_cache_dtype_spelling_the_install_registers_is_accepted(atom, flag, equals):
    answer = _judge(_kv(flag, "fp8", equals), atom)

    assert answer.status is ParseStatus.ACCEPTED
    assert answer.unknown_flags == ()


@pytest.mark.parametrize(("atom", "flag"), _KV_CACHE_FLAGS, indirect=("atom",))
@_SPELLINGS
def test_an_alias_does_not_turn_a_rejected_value_into_an_unknown_flag(atom, flag, equals):
    answer = _judge(_kv(flag, "invalid", equals), atom)

    assert answer.status is ParseStatus.REJECTED
    assert answer.unknown_flags == ()
    assert "invalid choice" in answer.detail


@pytest.mark.parametrize(("atom", "flag"), _KV_CACHE_FLAGS, indirect=("atom",))
@_SPELLINGS
def test_an_unknown_flag_beside_a_known_one_is_named_alone(atom, flag, equals):
    answer = _judge((*_kv(flag, "fp8", equals), "--unknown-option", "value"), atom)

    assert answer.status is ParseStatus.REJECTED
    assert answer.unknown_flags == ("--unknown-option",)


@pytest.mark.parametrize("atom", (None,), ids=("module-absent",), indirect=True)
@_SPELLINGS
def test_the_legacy_parser_has_no_kebab_alias(atom, equals):
    answer = _judge(_kv("--kv-cache-dtype", "fp8", equals), atom)

    assert answer.status is ParseStatus.REJECTED
    assert answer.unknown_flags == ("--kv-cache-dtype",)
    assert "unrecognized arguments" in answer.detail


@pytest.mark.parametrize(
    ("atom", "detail"),
    (
        pytest.param("import atom_parser_missing_dependency\n", "atom_parser_missing_dependency", id="module-absent"),
        pytest.param(
            "from argparse import atom_parser_missing_dependency\n",
            "atom_parser_missing_dependency",
            id="class-absent",
        ),
        pytest.param(
            "from atom.utils import atom_parser_missing_dependency\n",
            "atom_parser_missing_dependency",
            id="same-package-dependency-absent",
        ),
        pytest.param(
            _NATIVE_PARSER
            + "\n    def __init__(self):\n        raise ImportError('native parser construction failed')\n",
            "native parser construction failed",
            id="construction-failed",
        ),
    ),
    indirect=("atom",),
)
def test_a_broken_native_parser_is_a_probe_error_not_the_legacy_one(atom, detail):
    with pytest.raises(FrameworkProbeError, match=detail):
        _judge(("--kv_cache_dtype", "fp8"), atom)


@pytest.mark.parametrize("atom", (_NATIVE_PARSER, None), ids=("native", "legacy"), indirect=True)
@pytest.mark.parametrize(
    ("engine_source", "detail"),
    (
        pytest.param("raise ImportError('engine import failed')\n", "engine import failed", id="import-failed"),
        pytest.param(
            "class EngineArgs:\n"
            "    @staticmethod\n"
            "    def add_cli_args(parser):\n"
            "        raise ImportError('engine registration failed')\n",
            "engine registration failed",
            id="registration-import-failed",
        ),
        pytest.param(
            "class EngineArgs:\n"
            "    @staticmethod\n"
            "    def add_cli_args(parser):\n"
            "        raise RuntimeError('engine registration failed')\n",
            "engine registration failed",
            id="registration-runtime-failed",
        ),
    ),
)
def test_neither_parser_generation_hides_an_engine_args_error(atom, engine_source, detail):
    engine_module = Path(atom["PYTHONPATH"]) / "atom" / "model_engine" / "arg_utils.py"
    engine_module.write_text(engine_source, encoding="utf-8")

    with pytest.raises(FrameworkProbeError, match=detail):
        _judge(("--kv_cache_dtype", "fp8"), atom)
