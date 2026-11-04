# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The image check asks the environment about the facts of the rule pack and says pass, fail or nothing.

What is pinned, against a fixture tree of the files a real image holds (and, for a server's flags, a real subprocess):
each kind of check passes on what confirms the fact, fails on what contradicts it in the image the fact is about, and is
unverifiable everywhere else (a package that is not installed, a variable that is not set, a command that did not run);
a fact is a pass only when all its checks are; a session warns by default, stops under ``strict`` and does nothing under
``off``; the report names what failed, and the check runs on its own from the command line.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.common.outside_facts import image_check
from hyperloom.common.outside_facts.image_check import (
    MODE_ENV,
    ImageCheckFailed,
    ImageCheckMode,
    build_report,
    evaluate_pack,
    main,
    mode_from_environment,
    run_image_check,
)
from hyperloom.common.outside_facts.image_check_eval import Status, evaluate
from hyperloom.common.outside_facts.image_check_spec import (
    CheckExamples,
    CheckKind,
    CheckSpec,
    FileType,
    MatchMode,
    PathSpec,
)
from hyperloom.common.outside_facts.image_env import (
    CommandResult,
    HostImage,
    ImageReadError,
    run_command,
)
from hyperloom.common.outside_facts.rule_pack import (
    DEFAULT_PACK_PATH,
    PACK_ENV,
    Entry,
    Kind,
    RulePack,
    load_pack,
)

_LISTING = """\
usage: serve [-h] --model-path MODEL_PATH
  --model-path MODEL_PATH
  --tokenizer-path TOKENIZER_PATH
  --context-length CONTEXT_LENGTH
  --port PORT
"""
_SUMMARY = "usage: serve [options]\nConfig groups:\n  ModelConfig  Configuration for the model\n"
_OLD_AITER = 'config_path = Path("/tmp/aiter_configs/")'
_NEW_AITER = 'config_path = Path(tempfile.gettempdir()) / "aiter_configs"'
_AITER_META = (
    'develop = open(f"{this_dir}/../install_mode").read().strip() == "develop"\n'
    'AITER_META_DIR = AITER_ROOT_DIR if develop else f"{AITER_ROOT_DIR}/aiter_meta/"\n'
    'AITER_META_DIR = os.path.abspath(os.environ.get("AITER_META_DIR") or AITER_META_DIR)\n'
    'if not os.path.exists(os.path.join(AITER_META_DIR, "csrc")):\n'
    "    AITER_META_DIR = AITER_ROOT_DIR"
)
_LOGGER = "hyperloom.common.outside_facts.image_check"


def _spec(kind: CheckKind, **fields: Any) -> CheckSpec:
    return CheckSpec(kind=kind, examples=CheckExamples((), ()), **fields)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _package(tmp_path: Path, name: str, files: Mapping[str, str] | None = None) -> None:
    _write(tmp_path / "site" / name / "__init__.py", "")
    for relative, text in (files or {}).items():
        _write(tmp_path / "site" / name / relative, text)


class _Helps:
    """The runner of an image whose servers print the help it is given; the probe of packages is the real one."""

    def __init__(self, answers: Mapping[tuple[str, str], CommandResult | ImageReadError] | None = None) -> None:
        self.answers = dict(answers or {})
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], env: Mapping[str, str], timeout: float) -> CommandResult:
        self.calls.append(tuple(argv[1:3]))
        if argv[1] == "-c":
            return run_command(argv, env, timeout)
        answer = self.answers[(argv[2], argv[3])]
        if isinstance(answer, ImageReadError):
            raise answer
        return answer


def _image(tmp_path: Path, helps: _Helps | None = None, **environ: str) -> HostImage:
    variables = {"PYTHONPATH": str(tmp_path / "site"), **environ}
    return HostImage(root=tmp_path, environ=variables, run=helps or _Helps())


def _ok(text: str) -> CommandResult:
    return CommandResult(0, text, "")


def _status(spec: CheckSpec, image: HostImage) -> tuple[Status, str]:
    outcome = evaluate(spec, image)
    return outcome.status, outcome.detail


def test_a_path_check_passes_fails_on_a_wrong_type_and_names_the_path_it_looked_at(tmp_path: Path) -> None:
    _package(tmp_path, "sglang")
    _write(tmp_path / "sgl-workspace" / "aiter" / "configs" / "table.csv", "M,N\n")
    path = PathSpec(None, ("/sgl-workspace", "aiter/configs"))
    image = _image(tmp_path)

    found = _spec(CheckKind.PATH_EXISTS, path=path, installed="sglang", file_type=FileType.DIR)
    assert _status(found, image) == (Status.PASS, "found at /sgl-workspace/aiter/configs")
    moved = _spec(CheckKind.PATH_EXISTS, path=PathSpec(None, ("/sgl-workspace", "aiter/tables")), installed="sglang")
    assert _status(moved, image) == (Status.FAIL, "not found at /sgl-workspace/aiter/tables")
    wrong = _spec(CheckKind.PATH_EXISTS, path=path, installed="sglang", file_type=FileType.FILE)
    assert _status(wrong, image) == (Status.FAIL, "/sgl-workspace/aiter/configs is a dir, the pack says a file")


def test_a_path_check_says_nothing_where_the_image_or_the_checkout_is_not_the_one_it_is_about(tmp_path: Path) -> None:
    _write(tmp_path / "ix" / ".gitmodules", "")
    image = _image(tmp_path, INFERENCEX_PATH="/ix", EMPTY_PATH="/nowhere", RELATIVE="ix")
    absent = _spec(CheckKind.PATH_EXISTS, path=PathSpec(None, ("/sgl-workspace",)), installed="sglang")
    assert _status(absent, image) == (Status.UNVERIFIABLE, "the Python package sglang is not installed here")

    def based(variable: str) -> CheckSpec:
        return _spec(CheckKind.PATH_EXISTS, path=PathSpec(variable, (".gitmodules",)))

    assert _status(based("INFERENCEX_PATH"), image) == (Status.PASS, "found at /ix/.gitmodules")
    assert _status(based("UNSET"), image) == (Status.UNVERIFIABLE, "UNSET is not set")
    assert _status(based("EMPTY_PATH"), image)[1] == "EMPTY_PATH does not name a directory here"
    assert _status(based("RELATIVE"), image)[1] == "RELATIVE does not name a directory here"


def test_a_path_the_process_may_not_examine_is_unverifiable_and_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _package(tmp_path, "sglang")
    real = Path.is_dir

    def denied(self: Path) -> bool:
        if self.name == "locked":
            raise PermissionError(13, "denied")
        return real(self)

    monkeypatch.setattr(Path, "is_dir", denied)
    spec = _spec(CheckKind.PATH_EXISTS, path=PathSpec(None, ("/sgl-workspace", "locked")), installed="sglang")
    assert _status(spec, _image(tmp_path)) == (
        Status.UNVERIFIABLE,
        "could not be read: locked cannot be examined: PermissionError",
    )


def test_a_csv_check_lists_the_columns_that_are_missing_and_the_header_it_found(tmp_path: Path) -> None:
    _package(tmp_path, "aiter")
    _write(tmp_path / "configs" / "t.csv", "gfx,cu_num,M,N,K\n256,1,2,3,4\n")
    _write(tmp_path / "configs" / "empty.csv", "")
    image = _image(tmp_path)

    def spec(name: str, *columns: str) -> CheckSpec:
        return _spec(
            CheckKind.CSV_HAS_COLUMNS, path=PathSpec(None, ("/configs", name)), installed="aiter", columns=columns
        )

    assert _status(spec("t.csv", "cu_num", "M"), image) == (Status.PASS, "all 2 columns are in /configs/t.csv")
    status, detail = _status(spec("t.csv", "cu_num", "kernelName", "libtype"), image)
    assert status is Status.FAIL
    assert (
        detail
        == "/configs/t.csv lacks the columns ['kernelName', 'libtype']; its header is ['gfx', 'cu_num', 'M', 'N', 'K']"
    )
    assert _status(spec("absent.csv", "M"), image) == (Status.FAIL, "no such file: /configs/absent.csv")
    assert _status(spec("empty.csv", "M"), image)[0] is Status.FAIL


def test_a_source_check_reads_the_module_without_importing_it(tmp_path: Path) -> None:
    _package(tmp_path, "aiter", {"jit/core.py": f"raise SystemExit('imported')\n{_OLD_AITER}\n"})
    image = _image(tmp_path)

    present = _spec(CheckKind.SOURCE_CONTAINS, module="aiter.jit.core", text=("/tmp/aiter_configs",))
    assert _status(present, image) == (Status.PASS, "all 1 texts are in its source")
    reworded = _spec(CheckKind.SOURCE_CONTAINS, module="aiter.jit.core", text=("/tmp/aiter_configs", "gettempdir"))
    assert _status(reworded, image) == (Status.FAIL, "its source does not hold ['gettempdir']")
    moved = _spec(CheckKind.SOURCE_CONTAINS, module="aiter.jit.merge", text=("x",))
    assert _status(moved, image) == (Status.FAIL, "there is no such source file")
    elsewhere = _spec(CheckKind.SOURCE_CONTAINS, module="vllm.config", text=("x",))
    assert _status(elsewhere, image) == (Status.UNVERIFIABLE, "the Python package vllm is not installed here")


def test_a_source_check_on_a_file_reads_it_under_the_directory_a_variable_names(tmp_path: Path) -> None:
    _write(tmp_path / "ix" / ".gitmodules", "[submodule]\n  path = utils/aiperf\n")
    image = _image(tmp_path, INFERENCEX_PATH="/ix")
    path = PathSpec("INFERENCEX_PATH", (".gitmodules",))

    assert _status(_spec(CheckKind.SOURCE_CONTAINS, path=path, text=("utils/aiperf",)), image)[0] is Status.PASS
    status, detail = _status(_spec(CheckKind.SOURCE_CONTAINS, path=path, text=("utils/magpie",)), image)
    assert (status, detail) == (Status.FAIL, "/ix/.gitmodules does not hold ['utils/magpie']")
    missing = _spec(CheckKind.SOURCE_CONTAINS, path=PathSpec("INFERENCEX_PATH", ("absent",)), text=("x",))
    assert _status(missing, image) == (Status.FAIL, "there is no such source file")


def test_a_word_is_found_only_as_a_whole_word_and_a_substring_anywhere(tmp_path: Path) -> None:
    _package(tmp_path, "lib", {"has_tp.py": "HTTP_PORT = 1\nTP = 2\n", "no_tp.py": "HTTP_PORT = 1\n"})
    image = _image(tmp_path)

    def found(module: str, match: MatchMode) -> Status:
        return evaluate(_spec(CheckKind.SOURCE_CONTAINS, module=module, text=("TP",), match=match), image).status

    assert found("lib.has_tp", MatchMode.WORD) is Status.PASS
    assert found("lib.no_tp", MatchMode.WORD) is Status.FAIL
    assert found("lib.no_tp", MatchMode.SUBSTRING) is Status.PASS


@pytest.mark.parametrize(
    ("source", "attr", "status"),
    [
        ("def run(): pass\n", "run", Status.PASS),
        ("class Config:\n    def update(self): pass\n", "Config.update", Status.PASS),
        ("class Config:\n    mode = 1\n", "Config.mode", Status.PASS),
        ("import numpy as np\n", "np", Status.PASS),
        ("try:\n    import fast\nexcept ImportError:\n    fast = None\nVALUE: int = 3\n", "VALUE", Status.PASS),
        ("if True:\n    def late(): pass\n", "late", Status.PASS),
        ("def run(): pass\n", "start", Status.FAIL),
        ("class Config:\n    def update(self): pass\n", "Config.merge", Status.FAIL),
        ("def run():\n    inner = 1\n", "run.inner", Status.FAIL),
        ("def broken(:\n", "broken", Status.UNVERIFIABLE),
    ],
    ids=[
        "function",
        "method",
        "class-attribute",
        "import-alias",
        "guarded-import",
        "nested-block",
        "missing",
        "missing-method",
        "local-name",
        "syntax-error",
    ],
)
def test_an_attribute_check_reads_what_the_source_defines(
    tmp_path: Path, source: str, attr: str, status: Status
) -> None:
    _package(tmp_path, "lib", {"mod.py": source})
    spec = _spec(CheckKind.MODULE_ATTR, module="lib.mod", attr=attr)
    assert evaluate(spec, _image(tmp_path)).status is status


def test_a_flag_check_reads_the_help_and_asks_for_the_full_listing_when_the_first_does_not_list_the_flag(
    tmp_path: Path,
) -> None:
    _package(tmp_path, "vllm")
    spec = _spec(CheckKind.CLI_ACCEPTS_FLAG, module="vllm.serve", flag="--context-length")

    helps = _Helps({("vllm.serve", "--help"): _ok(_LISTING)})
    assert _status(spec, _image(tmp_path, helps)) == (Status.PASS, "the help lists --context-length")
    assert ("-m", "vllm.serve") in helps.calls and len(helps.calls) == 2

    helps = _Helps({("vllm.serve", "--help"): _ok(_SUMMARY), ("vllm.serve", "--help=all"): _ok(_LISTING)})
    assert _status(spec, _image(tmp_path, helps))[0] is Status.PASS

    helps = _Helps(
        {
            ("vllm.serve", "--help"): _ok(_LISTING.replace("--context-length", "--ctx-length")),
            ("vllm.serve", "--help=all"): _ok(_LISTING),
        }
    )
    assert _status(spec, _image(tmp_path, helps))[0] is Status.PASS


def test_a_flag_that_the_listing_lacks_fails_and_a_longer_flag_does_not_stand_in_for_it(tmp_path: Path) -> None:
    _package(tmp_path, "sglang")
    spec = _spec(CheckKind.CLI_ACCEPTS_FLAG, module="sglang.serve", flag="--context-len")
    helps = _Helps(
        {("sglang.serve", "--help"): _ok(_LISTING), ("sglang.serve", "--help=all"): ImageReadError("rejected")}
    )
    assert _status(spec, _image(tmp_path, helps)) == (Status.FAIL, "the help does not list --context-len")


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        (
            {
                ("sglang.serve", "--help"): CommandResult(
                    1, "", "Traceback\nModuleNotFoundError: No module named 'sgl_kernel'"
                )
            },
            "ModuleNotFoundError: No module named 'sgl_kernel'",
        ),
        (
            {("sglang.serve", "--help"): ImageReadError("python did not finish within 120s")},
            "did not finish within 120s",
        ),
        (
            {("sglang.serve", "--help"): _ok("usage: serve\n"), ("sglang.serve", "--help=all"): _ok("usage: serve\n")},
            "prints no list of flags",
        ),
    ],
    ids=["the-server-does-not-start", "the-help-times-out", "the-help-lists-nothing"],
)
def test_a_flag_check_is_unverifiable_when_no_help_could_be_read(
    tmp_path: Path, answers: dict[tuple[str, str], CommandResult | ImageReadError], reason: str
) -> None:
    _package(tmp_path, "sglang")
    spec = _spec(CheckKind.CLI_ACCEPTS_FLAG, module="sglang.serve", flag="--context-length")
    status, detail = _status(spec, _image(tmp_path, _Helps(answers)))
    assert status is Status.UNVERIFIABLE
    assert reason in detail


def test_what_a_command_prints_does_not_reach_a_detail_as_control_characters(tmp_path: Path) -> None:
    _package(tmp_path, "sglang")
    noisy = CommandResult(1, "", "\x1b[31mboom\x1b[0m\nImportError: x\x00y\x07")
    spec = _spec(CheckKind.CLI_ACCEPTS_FLAG, module="sglang.serve", flag="--context-length")
    _, detail = _status(spec, _image(tmp_path, _Helps({("sglang.serve", "--help"): noisy})))
    assert detail.endswith("ImportError: x y")
    assert all(ord(character) >= 32 and ord(character) != 127 for character in detail)


def test_a_real_server_command_is_run_and_its_flags_are_read(tmp_path: Path) -> None:
    _write(
        tmp_path / "site" / "fixture_server.py",
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "for flag in ('--model-path', '--port', '--context-length'):\n"
        "    p.add_argument(flag)\n"
        "p.parse_args()\n",
    )
    _write(
        tmp_path / "site" / "grouped_server.py",
        f"import sys\nif '--help=all' in sys.argv:\n    print({_LISTING!r})\nelse:\n    print({_SUMMARY!r})\n",
    )
    image = HostImage(root=tmp_path, environ={"PYTHONPATH": str(tmp_path / "site")})
    for module in ("fixture_server", "grouped_server"):
        spec = _spec(CheckKind.CLI_ACCEPTS_FLAG, module=module, flag="--context-length")
        assert _status(spec, image) == (Status.PASS, "the help lists --context-length"), module
    absent = _spec(CheckKind.CLI_ACCEPTS_FLAG, module="fixture_server", flag="--max-model-len")
    assert _status(absent, image) == (Status.FAIL, "the help does not list --max-model-len")
    assert image.describe()["packages"]["fixture_server"] == {"installed": True, "version": None}


def test_an_interpreter_that_cannot_run_makes_every_check_unverifiable_and_not_failed(tmp_path: Path) -> None:
    image = HostImage(root=tmp_path, python=str(tmp_path / "no-such-python"), environ={})
    spec = _spec(CheckKind.SOURCE_CONTAINS, module="aiter.jit.core", text=("x",))
    status, detail = _status(spec, image)
    assert status is Status.UNVERIFIABLE
    assert detail.startswith("could not be read: ")
    assert _status(spec, image) == (status, detail)


def test_the_packages_of_every_check_are_asked_of_the_interpreter_in_one_step(tmp_path: Path) -> None:
    helps = _Helps()
    pack = _pack(
        one=(_spec(CheckKind.SOURCE_CONTAINS, module="aiter.jit.core", text=("x",)),),
        two=(_spec(CheckKind.MODULE_ATTR, module="torch.cuda.tunable", attr="a"),),
        three=(_spec(CheckKind.PATH_EXISTS, path=PathSpec(None, ("/x",)), installed="aiter"),),
    )
    evaluate_pack(pack, _image(tmp_path, helps))
    assert [call[0] for call in helps.calls] == ["-c"]


def _pack(**checks: tuple[CheckSpec, ...]) -> RulePack:
    entries = {
        name: Entry(Kind.NAMES, {"names": {"role": "value"}}, f"project of {name}", held)
        for name, held in checks.items()
    }
    return RulePack(Path("pack.yaml"), "t1", "sha256:" + "0" * 64, entries)


def test_a_fact_is_a_pass_only_when_every_check_of_it_passed(tmp_path: Path) -> None:
    _package(tmp_path, "aiter", {"core.py": "VALUE = 1\n"})
    holds = _spec(CheckKind.SOURCE_CONTAINS, module="aiter.core", text=("VALUE",))
    lacks = _spec(CheckKind.SOURCE_CONTAINS, module="aiter.core", text=("OTHER",))
    elsewhere = _spec(CheckKind.SOURCE_CONTAINS, module="vllm.core", text=("VALUE",))
    pack = _pack(passing=(holds, holds), failing=(holds, lacks, elsewhere), partial=(holds, elsewhere), undeclared=())

    facts = {fact.fact: fact for fact in evaluate_pack(pack, _image(tmp_path))}

    assert {name: fact.status for name, fact in facts.items()} == {
        "passing": Status.PASS,
        "failing": Status.FAIL,
        "partial": Status.UNVERIFIABLE,
        "undeclared": Status.UNVERIFIABLE,
    }
    assert facts["undeclared"].checks == ()
    assert "no image_check is declared" in facts["undeclared"].note
    assert facts["failing"].project == "project of failing"
    assert [check.status for check in facts["failing"].checks] == [Status.PASS, Status.FAIL, Status.UNVERIFIABLE]


def _reference_image(tmp_path: Path, merge_line: str) -> HostImage:
    _package(
        tmp_path,
        "aiter",
        {
            "jit/core.py": f'AITER_LOG_TUNED_CONFIG = int(os.getenv("AITER_LOG_TUNED_CONFIG", "0"))\n'
            f"{_AITER_META}\n"
            f"class AITER_CONFIG:\n    def update_config_files(self, file_path, merge_name, env_name=None):\n        {merge_line}\n",
            "tuned_gemm.py": 'f"found padded_M: {m} is tuned on cu_num = {c} in {t}, libtype is {l}, kernel name is {k}"\n',
        },
    )
    _package(tmp_path, "sglang")
    _package(tmp_path, "vllm")
    _package(
        tmp_path,
        "torch",
        {"cuda/tunable.py": "PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING PYTORCH_TUNABLEOP_FILENAME\n"},
    )
    _package(
        tmp_path,
        "transformers",
        {"models/auto/configuration_auto.py": '"but Transformers does not recognize this architecture. It could be"\n'},
    )
    header = "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName\n"
    _write(tmp_path / "sgl-workspace" / "aiter" / "aiter" / "configs" / "bf16_tuned_gemm.csv", header)
    _write(tmp_path / "ix" / ".gitmodules", '[submodule "utils/aiperf"]\n  path = utils/aiperf\n')
    helps = _Helps(
        {
            ("sglang.launch_server", "--help"): _ok(_LISTING),
            ("vllm.entrypoints.openai.api_server", "--help"): _ok(_SUMMARY),
            ("vllm.entrypoints.openai.api_server", "--help=all"): _ok(
                _LISTING.replace("--context-length", "--max-model-len")
            ),
        }
    )
    return _image(tmp_path, helps, INFERENCEX_PATH="/ix")


def _by_fact(report: image_check.ImageCheckReport) -> dict[str, Status]:
    return {fact.fact: fact.status for fact in report.facts}


def test_the_reference_pack_over_an_image_that_holds_every_fact_it_can_confirm(tmp_path: Path) -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    report = build_report(pack, _reference_image(tmp_path, _OLD_AITER))

    held = _by_fact(report)
    assert report.failed() == ()
    assert [name for name, status in held.items() if status is Status.PASS] == [
        name for name, entry in pack.entries.items() if entry.image_checks
    ]
    assert all(held[name] is Status.UNVERIFIABLE for name, entry in pack.entries.items() if not entry.image_checks)
    assert report.counts() == {"pass": 12, "fail": 0, "unverifiable": 33}
    assert report.pack == {"version": pack.version, "digest": pack.digest}


def test_an_aiter_that_no_longer_holds_the_merge_directory_literal_fails_that_fact_and_no_other(tmp_path: Path) -> None:
    report = build_report(load_pack(DEFAULT_PACK_PATH), _reference_image(tmp_path, _NEW_AITER))

    assert [fact.fact for fact in report.failed()] == ["aiter_merged_view"]
    merged = report.failed()[0]
    assert [(check.check, check.status) for check in merged.checks] == [
        ("source_contains", Status.FAIL),
        ("module_attr", Status.PASS),
    ]
    assert merged.reasons() == [
        "aiter.jit.core holds ['/tmp/aiter_configs']: its source does not hold ['/tmp/aiter_configs']"
    ]
    assert report.summary() == {
        "mode": "warn",
        "pack_digest": report.pack["digest"],
        "pass": 11,
        "unverifiable": 33,
        "fail": ["aiter_merged_view"],
    }


def test_the_report_is_json_for_close_and_lines_for_people_and_is_written_whole(tmp_path: Path) -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    report = build_report(pack, _reference_image(tmp_path, _NEW_AITER))

    document = json.loads(json.dumps(report.to_json()))
    assert document["counts"] == {"pass": 11, "fail": 1, "unverifiable": 33}
    assert document["environment"]["python_version"].count(".") == 2
    assert document["environment"]["packages"]["vllm"] == {"installed": True, "version": None}
    assert {fact["fact"] for fact in document["facts"]} == set(pack.entries)
    lines = report.markdown_lines()
    assert lines[0].startswith(f"Image check (warn) of rule pack {pack.version}: 11 pass, 1 fail, 33 unverifiable")
    assert lines[1] == "- FAIL aiter_merged_view (aiter)"
    assert lines[-1].startswith("- Unverifiable here: python_import_errors, ")
    target = tmp_path / "attached" / "image_check.json"
    report.write(target)
    assert json.loads(target.read_text(encoding="utf-8")) == document


def test_a_failing_fact_is_one_warning_by_default_and_a_healthy_image_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        healthy = run_image_check(mode=ImageCheckMode.WARN, pack=pack, image=_reference_image(tmp_path, _OLD_AITER))
    assert healthy.failed() == ()
    assert [record for record in caplog.records if record.name == _LOGGER] == []

    other = tmp_path / "other"
    other.mkdir()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        report = run_image_check(mode=ImageCheckMode.WARN, pack=pack, image=_reference_image(other, _NEW_AITER))
    assert len(report.failed()) == 1
    (record,) = [record for record in caplog.records if record.name == _LOGGER]
    assert record.levelno == logging.WARNING
    assert "aiter_merged_view (aiter)" in record.getMessage()
    assert "does not hold ['/tmp/aiter_configs']" in record.getMessage()
    assert f"{MODE_ENV}=strict" in record.getMessage()


def test_a_failing_fact_stops_the_session_under_strict_and_carries_the_report(tmp_path: Path) -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    with pytest.raises(ImageCheckFailed, match="aiter_merged_view") as stopped:
        run_image_check(mode=ImageCheckMode.STRICT, pack=pack, image=_reference_image(tmp_path, _NEW_AITER))
    assert [fact.fact for fact in stopped.value.report.failed()] == ["aiter_merged_view"]

    other = tmp_path / "other"
    other.mkdir()
    healthy = run_image_check(mode=ImageCheckMode.STRICT, pack=pack, image=_reference_image(other, _OLD_AITER))
    assert healthy.failed() == ()


class _Untouchable:
    """An environment that records every attribute asked of it, so even a swallowed lookup is seen."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.asked.append(name)
        raise AttributeError(name)


def test_nothing_is_evaluated_under_off_and_the_report_says_so() -> None:
    image = _Untouchable()
    report = run_image_check(mode=ImageCheckMode.OFF, image=image)  # type: ignore[arg-type]
    assert image.asked == [], "the environment was asked for something"
    assert report.facts == ()
    assert report.summary()["mode"] == "off"
    assert report.summary()["fail"] == []


@pytest.mark.parametrize(
    ("environ", "mode"),
    [
        ({}, ImageCheckMode.WARN),
        ({MODE_ENV: ""}, ImageCheckMode.WARN),
        ({MODE_ENV: " Strict "}, ImageCheckMode.STRICT),
        ({MODE_ENV: "off"}, ImageCheckMode.OFF),
    ],
)
def test_the_mode_is_read_from_the_session_input_and_warn_is_the_default(
    environ: dict[str, str], mode: ImageCheckMode
) -> None:
    assert mode_from_environment(environ) is mode


def test_a_mode_that_is_not_one_is_refused_and_not_read_as_the_default() -> None:
    with pytest.raises(ValueError, match=r"HYPERLOOM_IMAGE_CHECK='stict' is not one of \['off', 'warn', 'strict'\]"):
        mode_from_environment({MODE_ENV: "stict"})


def test_the_command_line_prints_the_report_and_exits_one_when_a_fact_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    assert main([], pack=pack, image=_reference_image(tmp_path, _OLD_AITER)) == 0
    assert capsys.readouterr().out.startswith(
        f"Image check (warn) of rule pack {pack.version}: 12 pass, 0 fail, 33 unverifiable"
    )

    other = tmp_path / "other"
    other.mkdir()
    saved = tmp_path / "report.json"
    assert main(["--json", "--output", str(saved)], pack=pack, image=_reference_image(other, _NEW_AITER)) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(saved.read_text(encoding="utf-8"))
    assert printed["counts"]["fail"] == 1


def test_the_module_runs_on_its_own_against_the_pack_the_session_names(tmp_path: Path) -> None:
    document = yaml.safe_load(DEFAULT_PACK_PATH.read_text(encoding="utf-8"))
    check = document["facts"]["argparse_invalid_choice"]["image_check"][0]
    check["text"] = "a sentence argparse never prints"
    check["examples"]["pass"][0]["source"] = "a sentence argparse never prints"
    pack = tmp_path / "pack.yaml"
    pack.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    environment = {
        "PYTHONPATH": os.pathsep.join(str(Path(entry).resolve()) for entry in sys.path if entry),
        PACK_ENV: str(pack),
    }

    done = subprocess.run(
        [sys.executable, "-m", "hyperloom.common.outside_facts.image_check", "--json"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="/",
        env=environment,
        check=False,
    )

    assert done.returncode == 1, done.stderr[-2000:]
    held = {fact["fact"]: fact["status"] for fact in json.loads(done.stdout)["facts"]}
    assert held["argparse_invalid_choice"] == "fail"
    assert held["argparse_unrecognized_arguments"] == "pass"
