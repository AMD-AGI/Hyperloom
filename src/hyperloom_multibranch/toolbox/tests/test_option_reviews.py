# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The reviews a session's branches record against its option inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.inference_optimizer.native_builds import OptionVerdict
from hyperloom.measure.runner.native.server_options import (
    build_inventory,
)
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.option_reviews import (
    OPTION_BY_ENV_CODE,
    OptionReviews,
    ReviewEntry,
    ReviewRefused,
    ReviewStanding,
)
from hyperloom_multibranch.toolbox.tests.helpers import req

HELP = """\
----- common params -----

-t,    --threads N                      number of CPU threads to use during generation (default:
                                        -1)
-c,    --ctx-size N                     size of the prompt context (default: 0)
                                        (env: LLAMA_ARG_CTX_SIZE)
--jinja, --no-jinja                     whether to use the chat template engine
                                        (env: LLAMA_ARG_JINJA)
--keep N                                number of tokens to keep from the initial prompt
--old-size N                            the argument has been removed. use --ctx-size
"""


def reviewed_toolbox(root: Path, base_args: str = "-c 4096") -> OptionReviews:
    """The reviews of a toolbox directory whose inventory is ``HELP`` read against ``base_args``."""
    program = root / "llama-server"
    program.write_bytes(b"binary")
    return OptionReviews(root, build_inventory("llama-server", program, HELP, base_args))


def entry(
    option: str, verdict: OptionVerdict = OptionVerdict.NO_EFFECT_ON_STEP, note: str | None = None
) -> ReviewEntry:
    """A verdict on ``option`` with ``note``, or with a note of the option's own."""
    return ReviewEntry(option, verdict, f"read where {option} is used" if note is None else note)


def test_any_spelling_of_an_option_covers_its_entry_and_a_later_verdict_replaces_the_earlier(tmp_path: Path) -> None:
    reviews = reviewed_toolbox(tmp_path)

    reviews.record("b1", [entry("-t", OptionVerdict.CANDIDATE), entry("--jinja")])
    standing = reviews.record("b1", [entry("--threads", OptionVerdict.NOT_USABLE_HERE)])

    assert standing == ReviewStanding(3, 2, ("--keep",), ())


def test_a_note_given_to_two_options_is_refused_and_nothing_of_its_call_is_stored(tmp_path: Path) -> None:
    """One answer pasted on every option gives each a verdict and looks at none."""
    reviews = reviewed_toolbox(tmp_path)
    pasted = [entry(option, note="n/a") for option in ("--threads", "--no-jinja", "--keep")]

    with pytest.raises(ReviewRefused) as error:
        reviews.record("b1", pasted)

    assert 'the note "n/a" is given to 3 options (--threads, --no-jinja, --keep)' in str(error.value)
    assert reviews.standing("b1").covered == 0


def test_a_note_an_earlier_call_gave_another_option_is_refused_in_any_case_and_spacing(tmp_path: Path) -> None:
    reviews = reviewed_toolbox(tmp_path)
    reviews.record("b1", [entry("--threads", note="Read in arg.cpp")])

    with pytest.raises(ReviewRefused, match=r"given to 2 options \(--threads, --keep\)"):
        reviews.record("b1", [entry("--keep", note="read in  ARG.cpp")])

    assert reviews.standing("b1").covered == 1
    assert reviews.record("b1", [entry("-t", note="read in arg.cpp")]).covered == 1


def test_a_note_is_kept_as_one_line(tmp_path: Path) -> None:
    reviews = reviewed_toolbox(tmp_path)

    reviews.record("b1", [entry("--keep", note="line one\n  line two")])

    stored = json.loads((tmp_path / "option-reviews" / "b1.json").read_text(encoding="utf-8"))
    assert stored["reviews"] == {"--keep": {"verdict": "no_effect_on_step", "note": "line one line two"}}


def test_a_review_is_read_against_the_inventory_a_later_session_lays(tmp_path: Path) -> None:
    reviews = reviewed_toolbox(tmp_path)
    reviews.record("b1", [entry("--threads"), entry("--no-jinja"), entry("--keep")])

    assert reviewed_toolbox(tmp_path, "-t 4").standing("b1") == ReviewStanding(3, 2, ("--ctx-size",), ())


def _sets(**envs: str):
    return req("b1", change=RecipeDelta(config=ConfigDelta(envs=envs)))


def test_an_explore_that_sets_an_option_through_the_variable_the_server_reads_it_from_is_refused(
    tmp_path: Path,
) -> None:
    """Set through its variable an option passes the review and every check that reads the arguments."""
    reviews = reviewed_toolbox(tmp_path)
    reviews.record("b1", [entry("--threads"), entry("--no-jinja"), entry("--keep")])

    with pytest.raises(RequestRejected) as refused:
        reviews.check(_sets(LLAMA_ARG_JINJA="0", GGML_SCHED_DEBUG="1"))

    assert refused.value.code == OPTION_BY_ENV_CODE
    assert "LLAMA_ARG_JINJA (--no-jinja)" in refused.value.reason and "GGML_SCHED_DEBUG" not in refused.value.reason
    assert "as a server argument" in refused.value.reason
    reviews.check(_sets(GGML_SCHED_DEBUG="1"))
