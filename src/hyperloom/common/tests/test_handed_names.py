# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The names a prompt test looks for are found where a text writes them, and only there."""

from __future__ import annotations

from hyperloom.common.outside_facts import NATIVE_AUX_MODEL_FLAGS, kernel_knobs
from hyperloom.common.tests.handed_names import (
    identifiers_named,
    knob_words_named,
    pull_requests_named,
    server_options_named,
)


def test_an_option_is_named_by_any_spelling_the_server_s_help_lists_and_not_by_a_word_that_holds_it():
    long, short, *_ = NATIVE_AUX_MODEL_FLAGS["llamacpp"]

    assert server_options_named(f"set {long} or {short} FILE, then {long}=FILE") == {long, short}
    assert server_options_named(f"a run{short} word, a{long} word and --no-such-option-of-any-server") == set()
    assert server_options_named("python -m pytest -n 4 -c file") == set()


def test_a_knob_is_named_by_its_name_its_file_or_a_name_on_its_axes_in_any_case():
    name, knob = next(iter(kernel_knobs().items()))
    file = knob.file.rsplit("/", 1)[-1]
    axis_value = next(value for _, values in knob.axes for value in values if not value.isdigit())

    assert knob_words_named(f"tune {name.upper()}") >= {name}
    assert file in knob_words_named(f"read {knob.file}")
    assert knob_words_named(f"weights of type {axis_value}") == {axis_value.lower()}
    assert knob_words_named("read the kernels of the GPU backend and their launch constants") == set()


def test_an_identifier_and_a_numbered_pull_request_are_named_as_written():
    assert identifiers_named("call upstream_pr_search, then read calc_rows of the file") == {
        "upstream_pr_search",
        "calc_rows",
    }
    assert pull_requests_named("see #12345, pull/678 and PR 90; 3 of 261 options") == {"#12345", "pull/678", "PR 90"}
