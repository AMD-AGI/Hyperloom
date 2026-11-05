# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``image_check`` a rule pack entry may declare is read with the entry and pinned to its own examples.

What is pinned: the closed list of kinds has an evaluator for each member and the reference pack uses each; the reference
pack's checks hold the names the pack holds (a fact is still written once); a check that is of an unknown kind, lacks or
adds a field, cannot say which image it is about, names a reference the pack cannot resolve, or is refused or accepted
by one of its own examples is refused with the entry that carries it; a pack with no ``image_check`` loads as before.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.common import outside_facts
from hyperloom.common.outside_facts import image_check_eval
from hyperloom.common.outside_facts.image_check_spec import CheckKind, FileType, MatchMode
from hyperloom.common.outside_facts.rule_pack import DEFAULT_PACK_PATH, RulePackError, load_pack

Facts = dict[str, Any]

_WITH_CHECKS = {
    "aiter_merged_view",
    "aiter_meta_dir",
    "sglang_image_layout",
    "context_length_flags",
    "pytorch_tunableop_env",
    "argparse_unrecognized_arguments",
    "argparse_invalid_choice",
    "transformers_unknown_architecture",
    "transformers_does_not_recognize",
    "aiter_log_tuned_config",
    "aiter_tuned_hit",
    "aiperf_source",
}


def _variant(tmp_path: Path, change: Callable[[Facts], None]) -> Path:
    document = yaml.safe_load(DEFAULT_PACK_PATH.read_text(encoding="utf-8"))
    change(document["facts"])
    path = tmp_path / "pack.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _first_check(entry: str) -> Callable[[Facts], dict[str, Any]]:
    return lambda facts: facts[entry]["image_check"][0]


def _edit(entry: str, change: Callable[[dict[str, Any]], None]) -> Callable[[Facts], None]:
    return lambda facts: change(_first_check(entry)(facts))


def _set(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda check: check.__setitem__(key, value)


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda check: check.pop(key)


def _pass_example_without_the_text(check: dict[str, Any]) -> None:
    check["examples"]["pass"][0]["source"] = "nothing the check asks for"


def _fail_example_with_the_text(check: dict[str, Any]) -> None:
    check["examples"]["fail"][0]["source"] = check["examples"]["pass"][0]["source"]


def _path_check_anchored_by_nothing(check: dict[str, Any]) -> None:
    check.pop("installed")


def _path_second_part_absolute(check: dict[str, Any]) -> None:
    check["path"] = ["/sgl-workspace", "/aiter"]


def _both_module_and_path(check: dict[str, Any]) -> None:
    check["path"] = [{"env": "INFERENCEX_PATH"}, ".gitmodules"]


def _module_check_with_installed(check: dict[str, Any]) -> None:
    check["installed"] = "aiter"


@pytest.mark.parametrize(
    ("change", "entry", "reason"),
    [
        (_edit("aiter_merged_view", _set("check", "grep")), "aiter_merged_view", "is not one of"),
        (_edit("aiter_merged_view", _set("colour", "red")), "aiter_merged_view", "may add"),
        (_edit("context_length_flags", _drop("flag")), "context_length_flags", "found"),
        (_edit("aiter_merged_view", _set("text", {"ref": "no_such_entry.dir"})), "aiter_merged_view", "names no names"),
        (
            _edit("aiter_merged_view", _set("text", {"ref": "aiter_merged_view.nope"})),
            "aiter_merged_view",
            "holds no name",
        ),
        (
            _edit("aiter_merged_view", _set("text", {"ref": "hip_error_prefix.pattern"})),
            "aiter_merged_view",
            "names no names",
        ),
        (_edit("aiter_merged_view", _set("text", [])), "aiter_merged_view", "at least one text"),
        (_edit("aiter_merged_view", _set("module", "aiter jit")), "aiter_merged_view", "not a valid name"),
        (_edit("aiter_merged_view", _set("match", "regex")), "aiter_merged_view", "is not one of"),
        (_edit("aiter_merged_view", _module_check_with_installed), "aiter_merged_view", "anchored by its own package"),
        (_edit("aiter_merged_view", _both_module_and_path), "aiter_merged_view", "exactly one of"),
        (_edit("aiter_merged_view", _drop("module")), "aiter_merged_view", "exactly one of"),
        (_edit("context_length_flags", _set("flag", "context-length")), "context_length_flags", "not a valid name"),
        (
            _edit("sglang_image_layout", _path_check_anchored_by_nothing),
            "sglang_image_layout",
            "which image it is about",
        ),
        (_edit("sglang_image_layout", _set("path", ["aiter", "configs"])), "sglang_image_layout", "must be absolute"),
        (
            _edit("sglang_image_layout", _path_second_part_absolute),
            "sglang_image_layout",
            "after the first that is absolute",
        ),
        (_edit("sglang_image_layout", _set("type", "socket")), "sglang_image_layout", "is not one of"),
        (
            _edit("aiperf_source", _set("path", [{"env": "NOT AN ENV"}, ".gitmodules"])),
            "aiperf_source",
            "not a valid name",
        ),
        (_edit("aiter_merged_view", _drop("examples")), "aiter_merged_view", "found"),
        (
            _edit("aiter_merged_view", _set("examples", {"pass": [{"source": "x"}]})),
            "aiter_merged_view",
            "pass' and 'fail",
        ),
        (
            _edit("aiter_merged_view", _set("examples", {"pass": [], "fail": []})),
            "aiter_merged_view",
            "at least one example",
        ),
        (
            _edit("aiter_merged_view", _set("examples", {"pass": [{"help": "x"}], "fail": [{"help": "x"}]})),
            "aiter_merged_view",
            "the key 'source'",
        ),
        (
            _edit("aiter_merged_view", _pass_example_without_the_text),
            "aiter_merged_view",
            r"examples\.pass\[0\] came out fail",
        ),
        (
            _edit("aiter_merged_view", _fail_example_with_the_text),
            "aiter_merged_view",
            r"examples\.fail\[0\] came out pass",
        ),
    ],
    ids=[
        "unknown-kind",
        "unknown-field",
        "missing-field",
        "unknown-entry-reference",
        "unknown-role-reference",
        "reference-to-a-pattern",
        "empty-text-list",
        "bad-module-name",
        "bad-match-mode",
        "installed-on-a-module-check",
        "module-and-path",
        "neither-module-nor-path",
        "bad-flag-name",
        "path-without-an-anchor",
        "relative-path",
        "absolute-second-part",
        "bad-file-type",
        "bad-environment-variable-name",
        "no-examples",
        "examples-lack-fail",
        "empty-examples",
        "example-without-content",
        "pass-example-refused-by-the-check",
        "fail-example-accepted-by-the-check",
    ],
)
def test_a_pack_with_a_wrong_image_check_is_refused_naming_the_entry_and_the_check(
    tmp_path: Path, change: Callable[[Facts], None], entry: str, reason: str
) -> None:
    with pytest.raises(RulePackError) as refused:
        load_pack(_variant(tmp_path, change))
    message = str(refused.value)
    assert f"entry {entry!r}" in message
    assert "image_check[0]" in message
    assert str(tmp_path / "pack.yaml") in message
    assert re.search(reason, message), message


def test_a_renamed_fact_whose_check_still_shows_the_old_name_in_its_example_is_refused(tmp_path: Path) -> None:
    def rename(facts: Facts) -> None:
        facts["context_length_flags"]["names"]["sglang"] = "--ctx"

    with pytest.raises(
        RulePackError, match=r"context_length_flags.*examples\.pass\[0\] came out fail.*does not list --ctx"
    ):
        load_pack(_variant(tmp_path, rename))


def test_an_image_check_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    for wrong in (None, {}, "path_exists"):
        with pytest.raises(RulePackError, match="image_check must be a list with at least one check"):
            load_pack(
                _variant(
                    tmp_path, lambda facts, wrong=wrong: facts["aiter_merged_view"].__setitem__("image_check", wrong)
                )
            )


def test_a_pack_without_any_image_check_loads_as_before(tmp_path: Path) -> None:
    def strip(facts: Facts) -> None:
        for entry in facts.values():
            entry.pop("image_check", None)

    pack = load_pack(_variant(tmp_path, strip))
    assert all(entry.image_checks == () for entry in pack.entries.values())


def test_the_closed_list_of_kinds_has_an_evaluator_each_and_the_reference_pack_uses_every_one() -> None:
    assert set(image_check_eval._EVALUATORS) == set(CheckKind)
    used = {spec.kind for entry in load_pack(DEFAULT_PACK_PATH).entries.values() for spec in entry.image_checks}
    assert used == set(CheckKind)


def test_the_reference_pack_declares_checks_for_exactly_the_facts_that_can_be_confirmed_mechanically() -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    assert {entry_id for entry_id, entry in pack.entries.items() if entry.image_checks} == _WITH_CHECKS
    assert all(entry.project for entry in pack.entries.values())


def test_the_checks_of_the_reference_pack_hold_the_names_the_pack_holds_and_do_not_copy_them() -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    text_of = {entry_id: [spec.text for spec in entry.image_checks] for entry_id, entry in pack.entries.items()}
    assert text_of["aiter_merged_view"][0] == (outside_facts.AITER_MERGED_VIEW_DIR,)
    assert text_of["pytorch_tunableop_env"] == [
        (outside_facts.TUNABLEOP_ENABLED_ENV, outside_facts.TUNABLEOP_TUNING_ENV, outside_facts.TUNABLEOP_TABLE_ENV)
    ]
    assert text_of["aiter_log_tuned_config"] == [(outside_facts.AITER_LOG_TUNED_CONFIG_ENV,)]
    assert text_of["aiter_meta_dir"] == [
        (
            outside_facts.AITER_META_DIR_ENV,
            outside_facts.AITER_META_WHEEL_DIR,
            outside_facts.AITER_META_MARKER_DIR,
            outside_facts.AITER_INSTALL_MODE_FILE,
            outside_facts.AITER_DEVELOP_MODE,
        )
    ]
    flags = [spec.flag for spec in pack.entries["context_length_flags"].image_checks]
    assert tuple(flags) == outside_facts.CONTEXT_LENGTH_FLAGS
    layout = pack.entries["sglang_image_layout"].image_checks[0]
    assert layout.path is not None
    assert "/".join(layout.path.parts) == outside_facts.SGLANG_IMAGE_AITER_CONFIGS
    assert layout.file_type is FileType.DIR
    assert layout.installed == "sglang"


def test_a_check_about_a_checkout_is_anchored_by_the_directory_a_variable_names() -> None:
    check = load_pack(DEFAULT_PACK_PATH).entries["aiperf_source"].image_checks[0]
    assert check.path is not None
    assert check.path.base_env == "INFERENCEX_PATH"
    assert check.path.parts == (".gitmodules",)
    assert check.installed is None
    assert check.match is MatchMode.SUBSTRING


def test_the_examples_of_the_reference_pack_pin_each_check_in_both_directions() -> None:
    for entry in load_pack(DEFAULT_PACK_PATH).entries.values():
        for spec in entry.image_checks:
            assert spec.examples.passing
            assert spec.examples.failing
            image_check_eval.check_examples(spec)
