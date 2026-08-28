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

    def test_whitespace_bearing_value_keeps_posix_removal(self):
        """Inputs the quote-preserving tokenizer declines fall back, not break."""
        out = remove_server_args("--tool-call-parser 'my parser' --max-num-seqs 64", ["--max-num-seqs"])
        assert "--max-num-seqs" not in out
        assert "my parser" in out
