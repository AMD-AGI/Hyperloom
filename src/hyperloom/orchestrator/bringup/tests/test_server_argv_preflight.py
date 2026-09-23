# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The installed parser, not a rule table, decides whether an argv will launch."""

from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from hyperloom.orchestrator.bringup import argv_preflight as pf

#: The argument surface of the "installed" sglang, in the shape the adapter
#: reaches for: a registrar that fills a parser the caller owns. Written as a
#: real parser because the thing under test is what a real parser does with an
#: argv -- abbreviation matching, choice validation and all.
_SERVER_ARGS = """
class ServerArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--tp", "--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--context-length", type=int)
        parser.add_argument("--moe-runner-backend", default="auto")
        parser.add_argument("--attention-backend", choices=("triton", "aiter"))
        return parser
"""


def _install_sglang(root: Path, version: str) -> Path:
    """Write an importable ``sglang`` with a metadata version under ``root``."""
    srt = root / "sglang" / "srt"
    srt.mkdir(parents=True)
    (root / "sglang" / "__init__.py").write_text("", encoding="utf-8")
    (srt / "__init__.py").write_text("", encoding="utf-8")
    (srt / "server_args.py").write_text(_SERVER_ARGS, encoding="utf-8")
    dist = root / f"sglang-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: sglang\nVersion: {version}\n", encoding="utf-8")
    return root


@pytest.fixture
def serving(tmp_path, monkeypatch):
    """A launch env whose sglang is the one the probe interpreter also sees."""
    site = _install_sglang(tmp_path / "serve", "0.5.1")
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: sys.executable)
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(site),
        pf.SERVING_PYTHON_ENV: sys.executable,
    }


_ATOM_ENGINE_ARGS = """
class EngineArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--kv_cache_dtype", choices=("bf16", "fp8"), default="bf16")
        return parser
"""

_ATOM_ARG_PARSER = """
import argparse

class FlexibleArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *names, **kwargs):
        aliases = [name.replace("_", "-") for name in names if name.startswith("--") and "_" in name]
        return super().add_argument(*names, *aliases, **kwargs)
"""


_ATOM_KV_CACHE_FLAGS = (
    pytest.param(_ATOM_ARG_PARSER, "--kv_cache_dtype", id="native-snake"),
    pytest.param(_ATOM_ARG_PARSER, "--kv-cache-dtype", id="native-kebab"),
    pytest.param(None, "--kv_cache_dtype", id="legacy-module-absent"),
    pytest.param("", "--kv_cache_dtype", id="legacy-class-absent"),
)


@pytest.fixture
def atom_serving(tmp_path, monkeypatch, request):
    """Expose the installed ATOM parser contract to the real probe subprocess."""
    site = tmp_path / "serve"
    atom = site / "atom"
    for package in (atom, atom / "model_engine", atom / "utils"):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    (atom / "model_engine" / "arg_utils.py").write_text(_ATOM_ENGINE_ARGS, encoding="utf-8")
    parser_source = getattr(request, "param", _ATOM_ARG_PARSER)
    if parser_source is not None:
        (atom / "utils" / "arg_parser.py").write_text(parser_source, encoding="utf-8")
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: sys.executable)
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(site),
        pf.SERVING_PYTHON_ENV: sys.executable,
    }


@pytest.mark.parametrize(("atom_serving", "flag"), _ATOM_KV_CACHE_FLAGS, indirect=("atom_serving",))
@pytest.mark.parametrize("equals", (False, True), ids=("separate-value", "equals-value"))
@pytest.mark.parametrize("digest", ("", "atom-fp8"), ids=("no-repair", "repair-available"))
def test_atom_kv_cache_dtype_spellings_are_preserved(atom_serving, flag, equals, digest):
    """A valid FP8 request must not be repaired into the BF16 default."""
    argv = (f"{flag}=fp8",) if equals else (flag, "fp8")
    text = " ".join(argv)
    verdict = pf.check_server_argv(framework="atom", argv=argv, text=text, launch_env=atom_serving, digest=digest)
    assert verdict.status == pf.OK
    assert verdict.argv == argv
    assert verdict.text == text
    assert verdict.reason == pf.PARSED
    assert verdict.dropped == ()
    assert verdict.repaired_digest == ""


@pytest.mark.parametrize(("atom_serving", "flag"), _ATOM_KV_CACHE_FLAGS, indirect=("atom_serving",))
@pytest.mark.parametrize("equals", (False, True), ids=("separate-value", "equals-value"))
def test_atom_invalid_kv_cache_dtype_is_never_repaired(atom_serving, flag, equals):
    """Alias support does not turn rejected values into droppable unknown flags."""
    argv = (f"{flag}=invalid",) if equals else (flag, "invalid")
    text = " ".join(argv)
    verdict = pf.check_server_argv(
        framework="atom", argv=argv, text=text, launch_env=atom_serving, digest="atom-invalid"
    )
    assert verdict.status == pf.INVALID
    assert verdict.reason == pf.VALUE_REJECTED
    assert verdict.argv == argv
    assert verdict.text == text
    assert verdict.dropped == ()
    assert verdict.repaired_digest == ""
    assert "invalid choice" in verdict.detail


@pytest.mark.parametrize(("atom_serving", "flag"), _ATOM_KV_CACHE_FLAGS, indirect=("atom_serving",))
@pytest.mark.parametrize("equals", (False, True), ids=("separate-value", "equals-value"))
def test_atom_unknown_flag_is_dropped_once_without_losing_kv_cache_dtype(atom_serving, flag, equals):
    """Only a genuinely unknown flag may consume the one drop-only repair."""
    kv_argv = (f"{flag}=fp8",) if equals else (flag, "fp8")
    argv = (*kv_argv, "--unknown-option", "value")
    text = " ".join(argv)
    verdict = pf.check_server_argv(
        framework="atom", argv=argv, text=text, launch_env=atom_serving, digest="atom-unknown"
    )
    assert verdict.status == pf.OK
    assert verdict.reason == pf.PARSED_AFTER_DROP
    assert verdict.argv == kv_argv
    assert verdict.text == " ".join(kv_argv)
    assert verdict.dropped == ("--unknown-option",)
    assert verdict.repaired_digest == "atom-unknown"

    repeated = pf.check_server_argv(
        framework="atom",
        argv=argv,
        text=text,
        launch_env=atom_serving,
        digest="atom-unknown",
        repaired=(verdict.repaired_digest,),
    )
    assert repeated.status == pf.INVALID
    assert repeated.reason == pf.REPAIR_SPENT
    assert repeated.argv == argv
    assert repeated.dropped == ("--unknown-option",)
    assert repeated.repaired_digest == ""


@pytest.mark.parametrize("atom_serving", (None, ""), ids=("module-absent", "class-absent"), indirect=True)
@pytest.mark.parametrize("equals", (False, True), ids=("separate-value", "equals-value"))
def test_atom_legacy_parser_does_not_invent_kebab_aliases(atom_serving, equals):
    """A plain argparse server only accepts the flags its EngineArgs registers."""
    argv = ("--kv-cache-dtype=fp8",) if equals else ("--kv-cache-dtype", "fp8")
    text = " ".join(argv)
    verdict = pf.check_server_argv(framework="atom", argv=argv, text=text, launch_env=atom_serving)
    assert verdict.status == pf.INVALID
    assert verdict.reason == pf.REPAIR_SPENT
    assert verdict.argv == argv
    assert verdict.text == text
    assert verdict.dropped == ("--kv-cache-dtype",)
    assert verdict.repaired_digest == ""
    assert "unrecognized arguments" in verdict.detail


@pytest.mark.parametrize(
    ("atom_serving", "detail"),
    (
        pytest.param(
            "import atom_parser_missing_dependency\n",
            "atom_parser_missing_dependency",
            id="dependency-module-absent",
        ),
        pytest.param(
            "from argparse import atom_parser_missing_dependency\n",
            "atom_parser_missing_dependency",
            id="dependency-class-absent",
        ),
        pytest.param(
            "from atom.utils import atom_parser_missing_dependency\n",
            "atom_parser_missing_dependency",
            id="same-package-dependency-absent",
        ),
        pytest.param(
            _ATOM_ARG_PARSER
            + "\n    def __init__(self):\n        raise ImportError('native parser construction failed')\n",
            "native parser construction failed",
            id="parser-construction-failed",
        ),
    ),
    indirect=("atom_serving",),
)
def test_atom_broken_native_parser_is_unavailable_not_legacy(atom_serving, detail):
    """A broken modern parser must not be mistaken for a legacy installation."""
    argv = ("--kv_cache_dtype", "fp8")
    text = " ".join(argv)
    verdict = pf.check_server_argv(
        framework="atom", argv=argv, text=text, launch_env=atom_serving, digest="atom-parser-error"
    )
    assert verdict.status == pf.UNAVAILABLE
    assert verdict.reason == pf.PROBE_FAILED
    assert detail in verdict.detail
    assert verdict.argv == argv
    assert verdict.text == text
    assert verdict.dropped == ()
    assert verdict.repaired_digest == ""


@pytest.mark.parametrize("atom_serving", (_ATOM_ARG_PARSER, None), ids=("native", "legacy"), indirect=True)
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
def test_atom_engine_args_errors_are_unavailable(atom_serving, engine_source, detail):
    """Neither parser generation may hide EngineArgs import or registration errors."""
    engine_module = Path(atom_serving["PYTHONPATH"]) / "atom" / "model_engine" / "arg_utils.py"
    engine_module.write_text(engine_source, encoding="utf-8")
    argv = ("--kv_cache_dtype", "fp8")
    text = " ".join(argv)
    verdict = pf.check_server_argv(
        framework="atom", argv=argv, text=text, launch_env=atom_serving, digest="atom-engine-error"
    )
    assert verdict.status == pf.UNAVAILABLE
    assert verdict.reason == pf.PROBE_FAILED
    assert detail in verdict.detail
    assert verdict.argv == argv
    assert verdict.text == text
    assert verdict.dropped == ()
    assert verdict.repaired_digest == ""


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
    """Both failures are named by the installed parser, with no rule added here."""
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
