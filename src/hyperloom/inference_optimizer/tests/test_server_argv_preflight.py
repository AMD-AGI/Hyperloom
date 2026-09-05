# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The installed parser, not a rule table, decides whether an argv will launch.

Both failures reproduced here are ones that cost real sessions a round each: a
flag whose accepted spelling changed between framework versions, and a flag
that belongs to a different framework. Neither is visible in the argv, and
neither would be caught by a list of known-good flags kept in this repository,
because such a list is a copy of a parser that keeps moving.

So the framework here is a real installed package with a real ``argparse``
parser, reached through the production sglang adapter, and the verdicts come
from running it. No GPU, no server, and no rule added anywhere for either
failure.
"""

from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from hyperloom.orchestrator.bringup import argv_preflight as pf

#: The argument surface of the "installed" sglang. Written as a real parser
#: because the thing under test is what a real parser does with an argv --
#: abbreviation matching, choice validation and all.
_LAUNCH_SERVER = """
import argparse

parser = argparse.ArgumentParser(prog="sglang.launch_server")
parser.add_argument("--model-path", required=True)
parser.add_argument("--tp", "--tensor-parallel-size", type=int, default=1)
parser.add_argument("--context-length", type=int)
parser.add_argument("--moe-runner-backend", default="auto")
parser.add_argument("--attention-backend", choices=("triton", "aiter"))
"""


def _install_sglang(root: Path, version: str) -> Path:
    """Write an importable ``sglang`` with a metadata version under ``root``."""
    package = root / "sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "launch_server.py").write_text(_LAUNCH_SERVER, encoding="utf-8")
    dist = root / f"sglang-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: sglang\nVersion: {version}\n", encoding="utf-8")
    return root


@pytest.fixture
def serving(tmp_path, monkeypatch):
    """A launch env whose sglang is the one the probe interpreter also sees.

    The probe interpreter is pinned to this process's own, which is the case
    the check is meant to accept: one interpreter, one install, one verdict.
    """
    site = _install_sglang(tmp_path / "serve", "0.5.1")
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: sys.executable)
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(site),
        pf.SERVING_PYTHON_ENV: sys.executable,
    }


def _check(argv, env, **kwargs):
    """Run the preflight over ``argv`` written as an argument string."""
    return pf.check_server_argv(framework="sglang", argv=argv, text=" ".join(argv), launch_env=env, **kwargs)


def test_an_argv_the_installed_parser_accepts_is_ok(serving):
    """The accepting case has to be cheap and quiet, or the check gets turned off."""
    verdict = _check(["--tp", "8", "--moe-runner-backend", "triton"], serving)
    assert verdict.status == pf.OK
    assert verdict.dropped == ()


@pytest.mark.parametrize(
    ("argv", "flag"),
    (
        # The accepted spelling moved between framework versions.
        (["--tp", "8", "--moe-backend", "triton"], "--moe-backend"),
        # A vLLM flag on an sglang server; sglang spells it ``--context-length``.
        (["--max-model-len", "8192"], "--max-model-len"),
    ),
)
def test_the_parser_refuses_an_argument_it_does_not_have(serving, argv, flag):
    """Both failures are named by the installed parser, with no rule added here.

    The repair is withheld so what is asserted is the refusal itself rather
    than what the round does about it.
    """
    verdict = _check(argv, serving, digest="")
    assert verdict.status == pf.INVALID
    assert verdict.dropped == (flag,)
    assert flag in verdict.detail


def test_one_unrecognised_flag_is_dropped_and_the_argv_revalidated(serving):
    """The single allowed repair removes the flag and asks the parser again."""
    verdict = _check(["--tp", "8", "--moe-backend", "triton"], serving, digest="argv-1")
    assert verdict.status == pf.OK
    assert verdict.reason == pf.PARSED_AFTER_DROP
    assert verdict.dropped == ("--moe-backend",)
    assert verdict.argv == ("--tp", "8")
    # The caller is handed the key it must record, or the argv gets a second repair.
    assert verdict.repaired_digest == "argv-1"


def test_the_same_argv_does_not_get_a_second_repair(serving):
    """A recomposed argv that failed once has spent its repair; it is terminal."""
    verdict = _check(["--tp", "8", "--moe-backend", "triton"], serving, digest="argv-1", repaired=("argv-1",))
    assert verdict.status == pf.INVALID
    assert verdict.reason == pf.REPAIR_SPENT
    assert verdict.repaired_digest == ""


def test_a_rejected_value_is_terminal_and_never_rewritten(serving):
    """The harness does not know what value the framework meant, so it does not guess."""
    verdict = _check(["--attention-backend", "fa4"], serving, digest="argv-2")
    assert verdict.status == pf.INVALID
    assert verdict.reason == pf.VALUE_REJECTED
    assert verdict.dropped == ()
    assert "invalid choice" in verdict.detail


def test_a_drop_that_does_not_fix_the_argv_is_terminal(serving):
    """Two walls behind one flag is not a repair; it is a second guess."""
    verdict = _check(["--moe-backend", "triton", "--attention-backend", "fa4"], serving, digest="argv-3")
    assert verdict.status == pf.INVALID
    assert verdict.reason in (pf.REPAIR_FAILED, pf.VALUE_REJECTED)


def test_a_probe_interpreter_holding_another_version_yields_unavailable(tmp_path, monkeypatch):
    """A verdict from the wrong install is worse than no verdict: it is a silent false accept."""
    serve_site = _install_sglang(tmp_path / "serve", "0.5.1")
    probe_site = _install_sglang(tmp_path / "probe", "0.4.2")
    shim = tmp_path / "probe-python"
    shim.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            PYTHONPATH="{probe_site}:$PYTHONPATH" exec "{sys.executable}" "$@"
            """
        ),
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: str(shim))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(serve_site),
        pf.SERVING_PYTHON_ENV: sys.executable,
    }
    # An argv this install accepts, so nothing but the mismatch can decide it.
    verdict = _check(["--tp", "8"], env)
    assert verdict.status == pf.UNAVAILABLE
    assert verdict.reason == pf.INTERPRETER_MISMATCH


def test_an_unreachable_framework_is_unavailable_not_invalid(tmp_path, monkeypatch):
    """Nothing importable means nothing was checked; the launch stays the only verdict."""
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: sys.executable)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path), pf.SERVING_PYTHON_ENV: sys.executable}
    verdict = _check(["--tp", "8"], env)
    assert verdict.status == pf.UNAVAILABLE
    assert verdict.reason == pf.INTERPRETER_UNPROVEN


def test_a_framework_with_no_adapter_parser_is_unavailable(serving):
    """xDiT exposes no parser here; that is an absent check, not a passing one."""
    verdict = pf.check_server_argv(framework="xdit", argv=("--x",), text="--x", launch_env=serving)
    assert verdict.status == pf.UNAVAILABLE
    assert verdict.reason == pf.NO_PARSER
