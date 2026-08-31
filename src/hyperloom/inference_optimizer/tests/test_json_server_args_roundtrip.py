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
:func:`strip_benchmark_harness_flags`. These tests assert on ``json.loads``
rather than on exact strings so any join point that stops re-quoting fails here
regardless of which normalization shape it produces.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.actions.executors._grid_server_args import (
    compact_json_server_args,
    compose_server_args,
    dedup_vllm_server_args,
    merge_server_args,
    remove_server_args,
    strip_benchmark_harness_flags,
)


# The exact value from the incident. The wildcard barewords (``*.mlp.gate``,
# ``*expert*``) are what made the after-the-fact ``_repair_unquoted_json``
# heuristic give up, so they are load-bearing and must not be simplified.
_ATOM_QUANT_JSON = (
    '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}'
)

_ATOM_ARGS = f"--online_quant_config '{_ATOM_QUANT_JSON}'"

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
    """Preserving quotes must not cost the removal semantics."""

    def test_removes_json_valued_flag_by_name(self):
        out = remove_server_args(
            compact_json_server_args(f"--max-num-seqs 64 {_ATOM_ARGS}", "atom"),
            ["--online_quant_config"],
        )
        assert "--online_quant_config" not in out
        assert "--max-num-seqs 64" in out

    def test_removes_json_valued_flag_by_exact_pair(self):
        compacted = compact_json_server_args(f"--max-num-seqs 64 {_ATOM_ARGS}", "atom")
        spec = compacted.split("--online_quant_config", 1)[1].strip()
        out = remove_server_args(compacted, [f"--online_quant_config {spec}"])
        assert "--online_quant_config" not in out
        assert "--max-num-seqs 64" in out

    def test_removes_plain_flag_next_to_a_json_neighbour(self):
        out = remove_server_args(
            compact_json_server_args(f"--max-num-seqs 64 {_ATOM_ARGS}", "atom"),
            ["--max-num-seqs"],
        )
        assert "--max-num-seqs" not in out
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
        assert "--max-num-seqs" not in out
        # The token it cannot carry is passed through byte for byte.
        assert "--tool-call-parser 'my parser'" in out
        assert "cannot represent" in caplog.text

    def test_a_quoted_value_without_whitespace_also_still_removes(self):
        """The decline fired on any edge quote, not just a whitespace-bearing one.

        ``'hermes'`` is a single token and perfectly launchable, yet it disabled
        every removal on the string it appeared in.
        """
        out = remove_server_args("--tool-call-parser 'hermes' --max-num-seqs 64", ["--max-num-seqs"])
        assert "--max-num-seqs" not in out
        assert "--tool-call-parser 'hermes'" in out

    def test_harness_flag_is_stripped_even_beside_a_quoted_operand(self):
        """``strip_benchmark_harness_flags`` rides on the same call.

        A serving-ineligible benchmark flag reaching a served config is the
        worst version of the wholesale decline: the gain gets attributed to a
        configuration nobody can serve.
        """
        out = strip_benchmark_harness_flags(
            "--tool-call-parser 'hermes' --no-enable-prefix-caching --max-num-seqs 64"
        )
        assert "--no-enable-prefix-caching" not in out
        assert "--max-num-seqs 64" in out

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
        assert "--max-num-seqs" not in out

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
