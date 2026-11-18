# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""D7 over vLLM's three serve-parser modules: ``entrypoints.common``, ``entrypoints.launchers`` and
``entrypoints.openai``. The newest one an install holds judges the argv, and a parser module that is present but
cannot be imported is a probe error, never a fall back to an older one."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from inference_suite.framework_probe import FrameworkProbeError, run_probe
from inference_suite.frameworks import Framework
from inference_suite.parser_check import ParserAnswer, ParseStatus, parser_probe, read_answer

_FLEXIBLE_PARSER = """
import argparse

class FlexibleArgumentParser(argparse.ArgumentParser):
    pass
"""


def _cli_args(*flags: str) -> str:
    adds = "".join(f"    parser.add_argument({flag!r})\n" for flag in ("--model", *flags))
    return f"def make_arg_parser(parser):\n{adds}    return parser\n"


@pytest.fixture
def vllm(tmp_path: Path):
    site = tmp_path / "serve"

    def install(**generations: str) -> dict[str, str]:
        package = site / "vllm"
        modules = (
            package,
            package / "utils",
            package / "entrypoints",
            *(package / "entrypoints" / g for g in generations),
        )
        for module in modules:
            module.mkdir(parents=True, exist_ok=True)
            (module / "__init__.py").write_text("", encoding="utf-8")
        (package / "utils" / "argparse_utils.py").write_text(_FLEXIBLE_PARSER, encoding="utf-8")
        for generation, source in generations.items():
            (package / "entrypoints" / generation / "cli_args.py").write_text(source, encoding="utf-8")
        return {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(site)}

    return install


def _judge(argv: Sequence[str], env: dict[str, str]) -> ParserAnswer:
    probe = parser_probe(Framework.VLLM, argv)
    assert probe is not None
    return read_answer(run_probe(sys.executable, probe.program, probe.args, env, cwd=Path("/")))


@pytest.mark.parametrize("generation", ("common", "launchers", "openai"))
def test_each_generation_of_the_serve_parser_judges_the_argv(vllm, generation):
    env = vllm(**{generation: _cli_args("--monokernel")})

    accepted = _judge(("--model", "m", "--monokernel", "on"), env)
    refused = _judge(("--model", "m", "--unknown-option", "on"), env)

    assert accepted.status is ParseStatus.ACCEPTED
    assert refused.status is ParseStatus.REJECTED
    assert refused.unknown_flags == ("--unknown-option",)


def test_the_newest_parser_an_install_holds_judges_it_over_an_older_one_beside_it(vllm):
    env = vllm(common=_cli_args("--monokernel"), launchers=_cli_args())

    assert _judge(("--model", "m", "--monokernel", "on"), env).status is ParseStatus.ACCEPTED


def test_a_parser_module_that_cannot_be_imported_is_a_probe_error_not_an_older_parser(vllm):
    env = vllm(common="import vllm_parser_missing_dependency\n", launchers=_cli_args("--monokernel"))

    with pytest.raises(FrameworkProbeError, match="vllm_parser_missing_dependency"):
        _judge(("--model", "m", "--monokernel", "on"), env)


def test_an_install_with_no_serve_parser_is_a_probe_error(vllm):
    env = vllm()

    with pytest.raises(FrameworkProbeError, match="vllm.entrypoints.openai"):
        _judge(("--model", "m"), env)
