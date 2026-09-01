# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""JSON-valued server args must survive every compose / dedup / lift hop.

Regression suite for the ATOM GLM-5.2-MXFP4 incident: a valid
``--online_quant_config '{...}'`` reached the server as
``{global_quant_config:ptpc_fp8,...}``, every ``conc_sweep`` launch died on
``json.loads``, and the run stopped with ~9.8h of its 16h budget unspent.

The damage was a POSIX ``shlex.split`` + space-join inside
:func:`remove_server_args`, reached unconditionally because
:func:`compose_server_args` and the ``current_best`` lift both always call
:func:`strip_benchmark_harness_flags`.

Two assertion styles, deliberately split by what each test is about. Tests that
a JSON VALUE survives a hop go through ``json.loads``, so any join point that
stops re-quoting fails here regardless of which normalization shape it produces.
Tests that a REMOVAL did the right thing compare the whole string for equality:
asserting the removed flag is absent also passes on ``""``, and on output with
the flag's leftover operands stranded as bare argv words — two silent-deletion
bugs shipped past exactly that assertion shape before it was tightened here.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.actions.executors import _grid_server_args
from hyperloom.orchestrator.actions.executors._grid_server_args import (
    compact_json_server_args,
    compose_server_args,
    dedup_vllm_server_args,
    merge_server_args,
    remove_server_args,
    strip_benchmark_harness_flags,
    validate_server_args_shell_safe,
)


@pytest.fixture(autouse=True)
def _reset_undeliverable_warning_cache():
    """The undeliverable-token warning is reported once per process.

    That dedupe is what stops it printing on every compose of every variant, and
    it makes the warning order-dependent across tests in one process: whichever
    test first composes a ``--tool-call-parser`` payload consumes the WARNING and
    the next one sees only a DEBUG line.
    """
    _grid_server_args._UNSAFE_TRANSPORT_WARNED.clear()
    yield
    _grid_server_args._UNSAFE_TRANSPORT_WARNED.clear()


# The exact value from the incident. The wildcard barewords (``*.mlp.gate``,
# ``*expert*``) are what made the after-the-fact ``_repair_unquoted_json``
# heuristic give up, so they are load-bearing and must not be simplified.
_ATOM_QUANT_JSON = (
    '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}'
)

_ATOM_ARGS = f"--online_quant_config '{_ATOM_QUANT_JSON}'"

# The one shape the unquoted EXTRA_*_ARGS transport can carry, so the exact form
# every hop below is expected to hand on. Derived rather than transcribed: a
# hand-copied compaction would drift the moment the serializer's separators do.
_ATOM_COMPACT_ARGS = f"--online_quant_config {json.dumps(json.loads(_ATOM_QUANT_JSON), separators=(',', ':'))}"

_VLLM_ARGS = '--compilation-config \'{"cudagraph_mode": "FULL"}\''


def _json_value_of(args: str, flag: str) -> object:
    """Parse ``flag``'s JSON value out of an args string, as the server would.

    Mirrors the sink: Magpie expands ``EXTRA_*_ARGS`` unquoted, so whatever
    single token follows the flag is handed straight to ``json.loads``.
    """
    tokens = args.split()
    assert flag in tokens, f"{flag} missing from {args!r}"
    value = tokens[tokens.index(flag) + 1]
    return json.loads(value)


@pytest.mark.parametrize(
    ("args", "flag"),
    [
        (_ATOM_ARGS, "--online_quant_config"),
        (_VLLM_ARGS, "--compilation-config"),
    ],
)
class TestJsonValueSurvivesEveryHop:
    """A JSON value stays ``json.loads``-parseable after each individual hop."""

    def test_compact_is_parseable(self, args, flag):
        _json_value_of(compact_json_server_args(args, "atom"), flag)

    def test_strip_harness_flags_is_parseable(self, args, flag):
        """The hop that fires unconditionally.

        ``strip_benchmark_harness_flags`` has a non-empty removal list by
        construction, so this ``remove_server_args`` round trip runs on every
        compose and on every ``current_best`` lift even when the operator passed
        no ``remove_args`` at all.
        """
        _json_value_of(strip_benchmark_harness_flags(compact_json_server_args(args, "atom")), flag)

    def test_compose_append_is_parseable(self, args, flag):
        composed = compose_server_args(
            base_extra_args="--max-num-seqs 64",
            variant_extra_args=compact_json_server_args(args, "atom"),
        )
        _json_value_of(composed, flag)

    def test_compose_replace_with_removals_is_parseable(self, args, flag):
        composed = compose_server_args(
            base_extra_args="--max-num-seqs 32 --block-size 16",
            variant_extra_args=compact_json_server_args(args, "atom"),
            remove_args=["--block-size"],
            args_mode="replace",
        )
        assert "--block-size" not in composed
        _json_value_of(composed, flag)

    def test_full_compose_dedup_lift_chain_is_parseable(self, args, flag):
        """The end-to-end path the report asked to be locked down.

        compose (variant) -> dedup -> lift onto the previous ``current_best``
        (which re-strips harness flags and re-merges) -> dedup again.
        """
        composed = compose_server_args(
            base_extra_args="--max-num-seqs 32",
            variant_extra_args=compact_json_server_args(args, "atom"),
        )
        deduped = dedup_vllm_server_args(composed, "atom")

        # _lift_to_current_best re-strips both sides before merging them.
        previous = strip_benchmark_harness_flags("--gpu-memory-utilization 0.8")
        candidate = strip_benchmark_harness_flags(deduped)
        lifted = dedup_vllm_server_args(merge_server_args(previous, candidate), "atom")

        _json_value_of(lifted, flag)
        assert "--gpu-memory-utilization" in lifted

    def test_repeated_round_trips_do_not_erode_the_value(self, args, flag):
        """Each hop used to strip one more quoting layer, so iterate.

        The state.json evidence showed the same value at three different decay
        stages, which is what a per-hop erosion looks like.
        """
        current = compact_json_server_args(args, "atom")
        for _ in range(5):
            current = strip_benchmark_harness_flags(current)
            current = dedup_vllm_server_args(current, "atom")
            _json_value_of(current, flag)


class TestAtomValueContent:
    """The surviving value must be the value that was authored, not merely valid."""

    def test_exclude_layer_wildcards_are_intact(self):
        composed = compose_server_args(
            base_extra_args="--max-num-seqs 64",
            variant_extra_args=compact_json_server_args(_ATOM_ARGS, "atom"),
        )
        parsed = _json_value_of(composed, "--online_quant_config")
        assert parsed == json.loads(_ATOM_QUANT_JSON)
        assert "*expert*" in parsed["exclude_layer"]


class TestRemovalStillWorks:
    """Preserving quotes must not cost the removal semantics.

    Every assertion here is a whole-string equality. ``"--flag" not in out`` was
    the shape these tests used, and it is satisfied by ``out == ""`` and by an
    output whose surviving flags lost their operands, so it hid both silent
    deletions this suite exists to catch.
    """

    def test_removes_json_valued_flag_by_name(self):
        out = remove_server_args(
            compact_json_server_args(f"--max-num-seqs 64 {_ATOM_ARGS}", "atom"),
            ["--online_quant_config"],
        )
        assert out == "--max-num-seqs 64"

    def test_removes_json_valued_flag_by_exact_pair(self):
        compacted = compact_json_server_args(f"--max-num-seqs 64 {_ATOM_ARGS}", "atom")
        spec = compacted.split("--online_quant_config", 1)[1].strip()
        out = remove_server_args(compacted, [f"--online_quant_config {spec}"])
        assert out == "--max-num-seqs 64"

    def test_removes_plain_flag_next_to_a_json_neighbour(self):
        out = remove_server_args(
            compact_json_server_args(f"--max-num-seqs 64 {_ATOM_ARGS}", "atom"),
            ["--max-num-seqs"],
        )
        assert out == _ATOM_COMPACT_ARGS
        _json_value_of(out, "--online_quant_config")

    def test_an_undeliverable_token_does_not_cancel_the_removal(self, caplog):
        """The removal still happens, and the odd token is reported not swallowed.

        Declining wholesale meant one quoted operand anywhere in the string
        silenced every removal on it. This is a launch-path sink, so the flag
        that should have been dropped got served and benchmarked instead, and
        nothing was logged to say so.
        """
        args = "--tool-call-parser 'my parser' --max-num-seqs 64"
        with caplog.at_level("WARNING"):
            out = remove_server_args(args, ["--max-num-seqs"])
        # The token it cannot carry is passed through byte for byte.
        assert out == "--tool-call-parser 'my parser'"
        assert "cannot represent" in caplog.text

    def test_a_quoted_value_without_whitespace_also_still_removes(self):
        """The decline fired on any edge quote, not just a whitespace-bearing one.

        ``'hermes'`` is a single token and perfectly launchable, yet it disabled
        every removal on the string it appeared in.
        """
        out = remove_server_args("--tool-call-parser 'hermes' --max-num-seqs 64", ["--max-num-seqs"])
        assert out == "--tool-call-parser 'hermes'"

    def test_harness_flag_is_stripped_even_beside_a_quoted_operand(self):
        """``strip_benchmark_harness_flags`` rides on the same call.

        A serving-ineligible benchmark flag reaching a served config is the
        worst version of the wholesale decline: the gain gets attributed to a
        configuration nobody can serve.
        """
        out = strip_benchmark_harness_flags("--tool-call-parser 'hermes' --no-enable-prefix-caching --max-num-seqs 64")
        assert out == "--tool-call-parser 'hermes' --max-num-seqs 64"

    def test_removing_a_flag_consumes_its_whole_whitespace_bearing_value(self):
        """Removing the flag must not leave fragments of its value behind.

        The value survives ``shlex`` as one token here, but a JSON value with
        whitespace becomes several; consuming only the first would leave the
        rest as bare argv words the server reads as positional garbage.
        """
        out = remove_server_args(
            '--a {"k":"v with space"} --max-num-seqs 64',
            ["--a"],
        )
        assert out == "--max-num-seqs 64"

    def test_unbalanced_quotes_still_get_the_removal_applied(self):
        """``shlex`` cannot tokenize this at all; the removal is still attempted."""
        out = remove_server_args("--foo 'unclosed --max-num-seqs 64", ["--max-num-seqs"])
        assert out == "--foo 'unclosed"


class TestUndeliverableTokenReport:
    """What the warning says, and how often it says it."""

    def test_it_names_the_option_and_withholds_the_value(self, caplog):
        """A value can be a credential; so can a removal spec.

        The docstring already refused to echo the args string for that reason,
        while the format string interpolated the removal specs — authored through
        the same LLM / operator path and arriving by the same call.
        """
        with caplog.at_level("WARNING"):
            remove_server_args(
                "--api-key 'sk-secret value' --max-num-seqs 64",
                ["--max-num-seqs 64"],
            )
        assert "--api-key" in caplog.text
        assert "sk-secret value" not in caplog.text
        assert "--max-num-seqs 64" not in caplog.text

    def test_it_reports_once_per_option_set_not_once_per_compose(self, caplog):
        """``compose_server_args`` reaches this sink for every variant of every round.

        An unconditional warning here printed the same multi-line block hundreds
        of times in one session, for an args string that was merely carrying a
        legitimate quoted operand.
        """
        args = "--tool-call-parser 'my parser' --max-num-seqs 64"
        with caplog.at_level("WARNING"):
            for _ in range(20):
                compose_server_args(base_extra_args=args, remove_args=["--max-num-seqs"])
        assert caplog.text.count("cannot represent") == 1

    def test_an_unbalanced_quote_is_reported_under_its_own_option(self, caplog):
        """An unbalanced quote names the option that owns it, not a sentinel.

        This used to be the ``<unparseable>`` case: ``shlex`` raised on the whole
        string, so the report could only say "something in here is unparseable"
        and the removal fell through to a lossy whitespace re-split. Splitting on
        whitespace directly -- which is what the unquoted transport does anyway
        -- leaves nothing for a quote to unbalance, so the token is attributed
        like any other and the removal is exact.
        """
        with caplog.at_level("WARNING"):
            out = remove_server_args("--foo 'unclosed --max-num-seqs 64", ["--max-num-seqs"])
        assert "--foo" in caplog.text
        assert "<unparseable>" not in caplog.text
        assert out == "--foo 'unclosed"


class TestValueSpanStopsAtShortOptions:
    """A value span must not swallow the option that follows it.

    Consuming "everything up to the next ``--``" traded the old bug (a removal
    that silently did nothing) for a worse one: a removal that deletes flags
    nobody asked to remove, on the same unconditional launch path.
    """

    def test_a_single_dash_option_is_not_eaten_as_a_value(self):
        out = strip_benchmark_harness_flags("--no-enable-prefix-caching -tp 8 --max-num-seqs 64")
        assert out == "-tp 8 --max-num-seqs 64"

    def test_a_short_option_can_itself_be_removed(self):
        assert remove_server_args("-tp 8 --max-num-seqs 64", ["-tp"]) == "--max-num-seqs 64"

    def test_a_negative_number_is_a_value_not_an_option(self):
        """``-1`` must stay bound to its flag, or the span ends one token early."""
        assert remove_server_args("--a -1 --b 2", ["--b"]) == "--a -1"

    def test_a_whitespace_bearing_json_value_still_spans_its_fragments(self):
        """The span may only end at an option once the JSON braces are balanced."""
        assert remove_server_args('--a {"k":"v with space"} --max-num-seqs 64', ["--a"]) == "--max-num-seqs 64"

    def test_an_option_after_a_fragmented_json_value_survives(self):
        args = '--a {"k":"v with space"} -tp 8 --b 2'
        assert remove_server_args(args, ["--b"]) == '--a {"k":"v with space"} -tp 8'


class TestAnOptionLookalikeInsideAValue:
    """A JSON string value may contain a word spelled exactly like an option.

    The shapes above all carry values whose fragments happen not to start with a
    dash, so the span ended where the value ended by luck rather than by rule.
    One dash inside the string and the span ended mid-value instead: the removal
    left the rest of the blob behind as argv words, and
    ``validate_server_args_shell_safe`` then refused a string the previous
    shlex-based removal had taken out cleanly.
    """

    def test_the_span_covers_a_fragment_spelled_like_an_option(self):
        args = '--tp 8 --tmpl {"t":"Answer --now please"} --max-num-seqs 64'
        assert remove_server_args(args, ["--tmpl"]) == "--tp 8 --max-num-seqs 64"

    def test_the_result_is_still_launchable(self):
        """The regression was visible only at the sink, which is where it aborted."""
        args = '--tp 8 --tmpl {"t":"Answer --now please"} --max-num-seqs 64'
        validate_server_args_shell_safe(remove_server_args(args, ["--tmpl"]))

    def test_a_lookalike_is_not_itself_removable(self):
        """No spec needed: the harness strip runs on every compose."""
        args = '--a {"k":"x --no-enable-prefix-caching y"} --tp 8'
        assert strip_benchmark_harness_flags(args) == args

    def test_a_neighbour_removal_leaves_the_value_verbatim(self):
        args = '--tp 8 --tmpl {"t":"Answer --now please"} --max-num-seqs 64'
        assert remove_server_args(args, ["--tp"]) == '--tmpl {"t":"Answer --now please"} --max-num-seqs 64'

    def test_an_unbalanced_brace_still_stops_the_span(self):
        """The fallback. Nothing balances, so the plain scan stands.

        Letting the span run while the scan is unbalanced is what deleted the
        whole tail of the configuration; it may only run as far as the word that
        balances the blob, and no further when none does.
        """
        assert remove_server_args("--foo a} --tp 8 --max-num-seqs 64", ["--foo"]) == "--tp 8 --max-num-seqs 64"

    def test_a_stray_brace_does_not_make_later_flags_unremovable(self):
        """The same clamp, read from the other side.

        Carrying the negative depth forward would mark every later word as part
        of a value, and a flag inside a value is not removable — so one stray
        ``}`` would have silently disabled every removal after it.
        """
        assert remove_server_args("--foo a} --tp 8 --max-num-seqs 64", ["--tp"]) == "--foo a} --max-num-seqs 64"


class TestExactPairRemoval:
    def test_a_pair_spec_matches_a_flag_with_trailing_operands(self):
        """``--foo bar`` names one operand, not "everything after --foo".

        Comparing the spec against the whole span made the spec a no-op as soon
        as another operand followed. Matching only the first operand and deleting
        two tokens traded that for a worse outcome — see the class below.
        """
        assert remove_server_args("--foo bar baz", ["--foo bar"]) == ""

    def test_a_pair_spec_matches_mid_string(self):
        assert remove_server_args("--a 1 --foo bar baz --b 2", ["--foo bar"]) == "--a 1 --b 2"

    def test_a_pair_spec_leaves_a_different_value_alone(self):
        assert remove_server_args("--foo qux --b 2", ["--foo bar"]) == "--foo qux --b 2"

    def test_a_pair_spec_takes_the_whole_multi_value_span(self):
        """A flag's operand list goes as a unit, or the leftovers become argv words.

        ``--cuda-graph-bs 1 2 4`` with a ``--cuda-graph-bs 1`` spec used to come
        back as ``2 4 --tp 8``: ``validate_server_args_shell_safe`` then rejected
        the launch outright ("must be argv-like flags"), and the paths that skip
        that validator handed ``2`` and ``4`` to the server as positionals.
        """
        out = remove_server_args("--cuda-graph-bs 1 2 4 --tp 8", ["--cuda-graph-bs 1"])
        assert out == "--tp 8"

    @pytest.mark.parametrize(
        ("args", "removes"),
        [
            ("--cuda-graph-bs 1 2 4 --tp 8", ["--cuda-graph-bs 1"]),
            ("--cuda-graph-bs 1 2 4 --tp 8", ["--cuda-graph-bs"]),
            ("--foo bar baz --tp 8", ["--foo bar"]),
            ('--a {"k":"v with space"} --tp 8', ["--a"]),
            ("--a 1 --foo bar baz --b 2", ["--foo bar"]),
        ],
    )
    def test_no_removal_leaves_a_bare_positional_behind(self, args, removes):
        """The sink-side validator is the property, asserted directly.

        Every removal shape has to produce something still launchable, because
        this is the string that reaches ``EXTRA_*_ARGS``.
        """
        validate_server_args_shell_safe(remove_server_args(args, removes))

    def test_json_neighbour_survives_an_untokenizable_sibling(self):
        """One undeliverable sibling must not corrupt a JSON value beside it.

        The POSIX fallback applied to the WHOLE string, so a single
        whitespace-bearing operand anywhere turned ``--online_quant_config``
        into the ``{global_quant_config:ptpc_fp8,...}`` the server rejected --
        the incident's exact signature, still reachable after the tokenizer was
        made quote-preserving.
        """
        compacted = compact_json_server_args(
            f"--tool-call-parser 'my parser' {_ATOM_ARGS} --max-num-seqs 64",
            "atom",
        )
        out = remove_server_args(compacted, ["--max-num-seqs"])
        assert _json_value_of(out, "--online_quant_config") == json.loads(_ATOM_QUANT_JSON)

    def test_compose_keeps_json_intact_beside_an_untokenizable_sibling(self):
        """The same guarantee through the hop that fires on every compose.

        ``compose_server_args`` always ends in ``strip_benchmark_harness_flags``,
        so this reaches ``remove_server_args`` even with no operator removals.
        """
        composed = compose_server_args(
            base_extra_args="--tool-call-parser 'my parser' --max-num-seqs 64",
            variant_extra_args=compact_json_server_args(_ATOM_ARGS, "atom"),
        )
        assert _json_value_of(composed, "--online_quant_config") == json.loads(_ATOM_QUANT_JSON)


class TestEqualsJoinedOperands:
    """``--flag=value`` owns what follows it, like every other option name.

    The span was computed for the space-separated form only, so a removal on the
    equals form dropped that one word and left anything after it stranded as
    bare argv words -- the outcome the whole operands-as-a-unit rule exists to
    prevent, reached through the spelling the rule did not look at.
    """

    @pytest.mark.parametrize("spec", ["--block-size", "--block-size=16", "--block-size 16"])
    def test_every_spelling_removes_the_equals_form(self, spec):
        assert remove_server_args("--block-size=16 --tp 8", [spec]) == "--tp 8"

    def test_a_different_value_is_left_alone(self):
        args = "--block-size=16 --tp 8"
        assert remove_server_args(args, ["--block-size=32"]) == args

    def test_trailing_operands_go_with_the_flag(self):
        """argparse reads these as positionals, so they are the flag's or nobody's."""
        out = remove_server_args("--cuda-graph-bs=1 2 4 --tp 8", ["--cuda-graph-bs"])
        assert out == "--tp 8"
        validate_server_args_shell_safe(out)

    def test_a_neighbouring_removal_leaves_the_equals_form_verbatim(self):
        assert remove_server_args("--cuda-graph-bs=1 2 4 --tp 8", ["--tp"]) == "--cuda-graph-bs=1 2 4"

    def test_widening_past_the_named_operand_is_reported(self, caplog):
        """Same attribution warning the space-separated form already emits."""
        with caplog.at_level("WARNING"):
            remove_server_args("--cuda-graph-bs=1 2 4 --tp 8", ["--cuda-graph-bs=1"])
        assert "--cuda-graph-bs" in caplog.text
        assert "give it 3" in caplog.text


# ---------------------------------------------------------------------------
# Mechanical coverage of remove_server_args.
#
# Three rounds of review found three separate span/quote bugs in this function,
# each in a case the hand-written matrix did not contain, because the matrix was
# written from memory of the PREVIOUS bug. Enumerating the token shapes and
# asserting invariants over the product finds the next one without anybody
# having to guess it first.
# ---------------------------------------------------------------------------

# One representative per token shape that reaches this sink. Each entry is
# ``(flag, operands)`` and every combination is placed at the head, middle and
# tail of an arg string, with every subset of removal spellings applied.
_SHAPES: tuple[tuple[str, str], ...] = (
    ("--tp", "8"),
    ("-tp", "8"),  # single-dash short option, once swallowed as a value
    ("--seed", "-1"),  # negative number, not an option
    ("--cuda-graph-bs", "1 2 4"),  # multi-operand list
    ("--quant", '{"a":1}'),  # compact JSON
    ("--quant2", '{"k":"v with space"}'),  # JSON whose value carries whitespace
    ("--broken", "a}"),  # unbalanced brace: ran the span off the end
    ("--parser", "'hermes'"),  # quoted operand, no whitespace
    ("--parser2", "'my parser'"),  # quoted operand with whitespace
    ("--half", "'unclosed"),  # single edge quote, untokenizable for shlex
    ("--tmpl", '{"t":"a  b"}'),  # double space inside a JSON string value
    ("--tmpl2", '{"t":"a --b c"}'),  # JSON string value carrying an option lookalike
    ("--eqlist=1", "2 4"),  # equals-joined operand, then more operands
)

_NOISE = "--max-num-seqs 64"


def _spellings(flag: str, operands: str) -> list[str]:
    """The removal spellings an operator or LLM plausibly writes for one flag."""
    first = operands.split()[0]
    out = [flag, f"{flag} {operands}", f"{flag} {first}"]
    quoted = len(first) >= 2 and first[0] == first[-1] and first[0] in "\"'"
    if quoted or first[0] not in "\"'":
        # Re-quoting a whole operand must not change what it names. Only a
        # MATCHED pair is quoting; a lone edge quote (``'my`` from a
        # whitespace-bearing value) is a byte of the value under this transport,
        # so re-quoting it would name something else and is not a spelling of
        # the same removal.
        bare = first[1:-1] if quoted else first
        out.append(f'{flag} "{bare}"')
        out.append(f"{flag} '{bare}'")
    return out


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: s[0].lstrip("-"))
@pytest.mark.parametrize("position", ["head", "middle", "tail"])
def test_removing_a_flag_never_disturbs_its_neighbours(shape: tuple[str, str], position: str):
    """Every retained flag keeps its own bytes, whatever was removed.

    The invariant the whole function exists for. ``--online_quant_config``
    reached a server mangled because a neighbouring removal rewrote it, and the
    span bug found in review deleted the entire tail of the string when one
    flag's operand happened to carry an unbalanced brace.
    """
    flag, operands = shape
    target = f"{flag} {operands}"
    if position == "head":
        args = f"{target} {_NOISE} --port 8000"
    elif position == "middle":
        args = f"{_NOISE} {target} --port 8000"
    else:
        args = f"{_NOISE} --port 8000 {target}"

    for spelling in _spellings(flag, operands):
        got = remove_server_args(args, [spelling])
        # The neighbours come back byte for byte, and nothing of the target is
        # left behind -- so this one assertion covers both the removal and the
        # span not reaching past it.
        assert got == f"{_NOISE} --port 8000", f"{spelling!r} on {args!r} gave {got!r}"


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: s[0].lstrip("-"))
def test_an_unrelated_removal_returns_every_other_flag_verbatim(shape: tuple[str, str]):
    """A removal that matches nothing on this flag must not touch its value.

    This is the case the review's whitespace finding lived in: removing
    ``--max-num-seqs`` collapsed the double space inside a retained
    ``{"chat_template":"a  b"}`` because reassembly went through ``" ".join``.
    """
    flag, operands = shape
    args = f"{flag} {operands} {_NOISE}"
    got = remove_server_args(args, ["--max-num-seqs"])
    assert got == f"{flag} {operands}"


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: s[0].lstrip("-"))
def test_removal_is_idempotent_and_leaves_no_bare_words(shape: tuple[str, str]):
    """Applying a removal twice changes nothing, and never orphans an operand.

    Bare argv words are the concrete damage a half-removed operand list does:
    ``validate_server_args_shell_safe`` rejects them outright and the paths that
    skip it hand them to the server as positionals.
    """
    flag, operands = shape
    args = f"{_NOISE} {flag} {operands} --port 8000"
    for spelling in _spellings(flag, operands):
        once = remove_server_args(args, [spelling])
        assert remove_server_args(once, [spelling]) == once
        for word in once.split():
            if word.startswith("-") or "{" in word or "}" in word or word[0] in "\"'":
                continue
            # Anything left that is not an option name must be an operand of the
            # option that precedes it, never a word standing on its own.
            assert once.split().index(word) > 0, f"{once!r} starts with a bare word"


def test_the_whole_shape_matrix_removed_at_once_leaves_nothing_behind():
    """All shapes in one string, all removed together.

    The per-shape cases above each isolate one flag; composing them checks that
    the spans do not interfere, which is where the depth-tracking version failed
    -- one unbalanced operand consumed every flag after it.
    """
    args = " ".join(f"{flag} {operands}" for flag, operands in _SHAPES)
    got = remove_server_args(args, [flag for flag, _ in _SHAPES])
    assert got == ""


def test_removing_one_flag_from_the_shape_matrix_keeps_all_the_others():
    """The inverse: one removal out of the full matrix disturbs nothing else."""
    args = " ".join(f"{flag} {operands}" for flag, operands in _SHAPES)
    for flag, operands in _SHAPES:
        got = remove_server_args(args, [flag])
        expected = " ".join(f"{f} {o}" for f, o in _SHAPES if f != flag)
        assert got == expected, f"removing {flag} gave {got!r}"
