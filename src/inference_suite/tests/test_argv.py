# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``inference_suite.argv``: JSON-preserving tokenization, JSON normalization, removal, identity form."""

from __future__ import annotations

import json

import pytest

from inference_suite import argv
from inference_suite.argv import (
    compact_json_server_args,
    flag_occurrences,
    last_alias_value,
    normalize_argv,
    remove_server_args,
    speculative_config,
    split_args_preserving_json,
)


class TestCompactJsonServerArgs:
    """JSON-valued flags must be space-free to survive Magpie's unquoted ``$EXTRA_VLLM_ARGS`` splice (otherwise spec-decode / compilation-config explore variants always crash the server at boot)."""

    def test_compilation_config_separator_space_removed(self):
        out = compact_json_server_args('--compilation-config {"full_cuda_graph": true}')
        assert out == '--compilation-config {"full_cuda_graph":true}'
        # the JSON value is now a single shell word under bash word-splitting
        assert len(out.split()) == 2

    def test_speculative_config_multikey(self):
        out = compact_json_server_args('--speculative-config {"method": "eagle", "num_speculative_tokens": 3}')
        assert out == ('--speculative-config {"method":"eagle","num_speculative_tokens":3}')
        assert len(out.split()) == 2

    def test_compact_json_server_args_internal_space_unsupported(self):
        # Spaces inside JSON string values are not collapsed, so the value is not a single shell word under Magpie's
        # unquoted expansion; this flag shape is unsupported.
        out = compact_json_server_args('--speculative-config {"model": "draft model name"}')
        # Value is preserved verbatim (separator space after ':' removed only).
        assert out == '--speculative-config {"model":"draft model name"}'
        # ...but it still splits into MORE than the ideal 2 words: the two internal spaces of "draft model name"
        # survive, so the shell sees ['--speculative-config', '{"model":"draft', 'model', 'name"}'].
        assert len(out.split()) == 4

    def test_quote_stripped_json_is_repaired(self):
        # A prior shlex round-trip (e.g. in the GEMM shape-capture path) can strip the JSON double-quotes, turning a
        # stored-valid ``{"method":"ngram",...}`` into ``{method:ngram,...}`` which vLLM's ``json.loads`` rejects at
        # boot (observed: shape_capture server never boots -> shape_capture_failed). compact must repair the barewords
        # back to valid JSON.
        out = compact_json_server_args(
            "--speculative-config {method:ngram,num_speculative_tokens:7,prompt_lookup_min:2,prompt_lookup_max:8}"
        )
        assert out == (
            "--speculative-config "
            '{"method":"ngram","num_speculative_tokens":7,"prompt_lookup_min":2,"prompt_lookup_max":8}'
        )
        blob = out.split(" ", 1)[1]
        json.loads(blob)  # must be valid JSON now

    def test_quote_stripped_json_with_path_value_repaired(self):
        # Bareword string values containing ``/``, ``.``, ``-`` (e.g. a model id) must also be re-quoted.
        out = compact_json_server_args(
            "--speculative-config {method:eagle3,model:RedHatAI/Llama-3.1-8B-Instruct-speculator}"
        )
        blob = out.split(" ", 1)[1]
        assert json.loads(blob) == {
            "method": "eagle3",
            "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator",
        }

    def test_quote_stripped_nested_json_is_repaired(self):
        out = compact_json_server_args("--compilation-config {method:ngram,nested:{a:1}}")
        blob = out.split(" ", 1)[1]
        assert json.loads(blob) == {
            "method": "ngram",
            "nested": {"a": 1},
        }

    def test_shell_single_quote_wrappers_are_removed(self):
        out = compact_json_server_args("""--speculative-config '{"method":"ngram","num_speculative_tokens":7}' """)
        assert out.strip() == ('--speculative-config {"method":"ngram","num_speculative_tokens":7}')
        json.loads(out.split(" ", 1)[1].strip())

    def test_unrepairable_json_left_verbatim(self):
        # Genuinely broken blobs that cannot be repaired to valid JSON are left verbatim (no worse than before), never
        # raising.
        raw = "--compilation-config {this is : not ] json"
        out = compact_json_server_args(raw)
        assert "{this is" in out

    def test_other_flags_around_json_untouched(self):
        out = compact_json_server_args('--kv-cache-dtype fp8 --compilation-config {"level": 3}')
        assert out == '--kv-cache-dtype fp8 --compilation-config {"level":3}'

    def test_sglang_json_is_normalized_for_unquoted_transport(self):
        raw = '--speculative-config {"method": "eagle"}'
        assert compact_json_server_args(raw) == ('--speculative-config {"method":"eagle"}')

    def test_legacy_shell_wrappers_removed(self):
        raw = """--json-model-override-args '{"rope_scaling":null}'"""
        assert compact_json_server_args(raw) == '--json-model-override-args {"rope_scaling":null}'

    def test_no_json_is_noop(self):
        raw = "--block-size 128 --no-enable-prefix-caching"
        assert compact_json_server_args(raw) == raw

    def test_malformed_json_left_verbatim(self):
        raw = "--compilation-config {not json}"
        assert compact_json_server_args(raw) == raw

    def test_malformed_wrapped_json_keeps_both_shell_wrappers(self):
        raw = "--compilation-config '{not json}'"
        assert compact_json_server_args(raw) == raw

    def test_empty_is_noop(self):
        assert compact_json_server_args("") == ""
        assert compact_json_server_args(None) == ""


class TestRemoveServerArgsPreservesJson:
    """Removal tokenizes with the JSON-preserving split, so a sibling JSON value stays parseable."""

    def test_remove_port_keeps_sibling_json_valid(self):
        raw = '--compilation-config {"cudagraph_mode":"FULL"} --port 8888'
        out = remove_server_args(raw, ["--port"])
        assert "--port" not in out
        blob = out.split("--compilation-config ", 1)[1].strip()
        assert json.loads(blob) == {"cudagraph_mode": "FULL"}

    def test_remove_keeps_multikey_speculative_config_valid(self):
        raw = '--speculative-config {"method":"ngram","num_speculative_tokens":7} --port 8888'
        out = remove_server_args(raw, ["--port"])
        blob = out.split("--speculative-config ", 1)[1].strip()
        assert json.loads(blob) == {"method": "ngram", "num_speculative_tokens": 7}

    def test_remove_noop_when_nothing_matches_keeps_json(self):
        raw = '--compilation-config {"cudagraph_mode":"FULL"}'
        out = remove_server_args(raw, ["--port"])
        assert json.loads(out.split("--compilation-config ", 1)[1].strip()) == {"cudagraph_mode": "FULL"}

    def test_sign_prefixed_custom_op_survives_removal(self):
        """A ``+``/``-`` prefixed custom-op value must survive the round trip."""
        raw = (
            "--compilation-config "
            '{"mode":3,"custom_ops":["+fused_rms_norm_gated"],'
            '"cudagraph_capture_sizes":[1,2,3]} --port 8888'
        )
        out = remove_server_args(raw, ["--port"])
        assert "--port" not in out
        blob = json.loads(out.split("--compilation-config ", 1)[1].strip())
        assert blob["custom_ops"] == ["+fused_rms_norm_gated"]
        assert blob["cudagraph_capture_sizes"] == [1, 2, 3]

    def test_shell_quoted_plain_operand_loses_its_wrappers(self):
        """Magpie expands ``EXTRA_*_ARGS`` unquoted, so wrappers must not persist."""
        out = remove_server_args("--tool-call-parser 'kimi_k3' --port 8888", ["--port"])
        assert out == "--tool-call-parser kimi_k3"

    def test_an_unbalanced_quote_leaves_the_string_alone(self):
        """Untokenizable input is returned untouched rather than guessed at."""
        broken = "--served-model-name 'oops --max-num-seqs 512"
        assert remove_server_args(broken, ["--max-num-seqs"]) == broken

    def test_a_json_value_survives_a_removal(self):
        args = '--compilation-config {"mode":3} --max-num-seqs 512'
        out = remove_server_args(args, ["--max-num-seqs"])
        assert out == '--compilation-config {"mode":3}'


def test_remove_server_args_accepts_multi_flag_string() -> None:
    out = remove_server_args(
        "--flag-a --flag-b --flag-c 3 --keep 4",
        "--flag-a --flag-b --flag-c",
    )
    assert out == "--keep 4"


def test_remove_server_args_keeps_a_sibling_json_value_parseable() -> None:
    """Removing one flag must not corrupt a JSON-valued sibling."""
    args = (
        '--online_quant_config {"global_quant_config":"ptpc_fp8",'
        '"exclude_layer":["*.mlp.gate","*expert*"]} '
        "--no-enable-prefix-caching --tp 8"
    )
    out = remove_server_args(args, ["--no-enable-prefix-caching"])
    tokens = out.split(" ")
    assert tokens[0] == "--online_quant_config"
    assert json.loads(tokens[1]) == {
        "global_quant_config": "ptpc_fp8",
        "exclude_layer": ["*.mlp.gate", "*expert*"],
    }
    assert tokens[2:] == ["--tp", "8"]


def test_remove_server_args_unbalanced_brace_drops_only_its_own_flag() -> None:
    """A stray ``}`` is not a JSON blob and must not take the tail with it."""
    out = remove_server_args("--foo a} --tp 8 --max-num-seqs 64", ["--foo"])
    assert out == "--tp 8 --max-num-seqs 64"


def test_unwrap_plain_quoted_value():
    assert argv._unwrap_shell_quotes("'kimi_k3'") == "kimi_k3"


def test_unwrap_eq_form_strips_value_quotes():
    assert argv._unwrap_shell_quotes("--tool-call-parser='kimi_k3'") == "--tool-call-parser=kimi_k3"


def test_unwrap_eq_form_leaves_json_value_intact():
    token = "--compilation-config='{\"mode\":3}'"
    result = argv._unwrap_shell_quotes(token)
    # The value starts with { so quotes must not be stripped.
    assert result == token


def test_split_args_unwraps_eq_form_plain_value():
    tokens = split_args_preserving_json("--tool-call-parser='kimi_k3'")
    assert tokens is not None
    assert tokens == ["--tool-call-parser=kimi_k3"]


def test_split_args_leaves_json_double_quotes_intact():
    text = "--compilation-config '{\"mode\":3}'"
    tokens = split_args_preserving_json(text)
    assert tokens is not None
    # The inner double quotes must survive.
    assert any('"mode"' in t for t in tokens), tokens


@pytest.mark.parametrize(
    ("assigned", "spaced"),
    [
        (["--max-num-seqs=64"], ["--max-num-seqs", "64"]),
        (['--speculative-config={"method": "eagle"}'], ["--speculative-config", '{"method":"eagle"}']),
        (["--tp=4", "--enable-x"], ["--tp", "4", "--enable-x"]),
    ],
)
def test_normalize_argv_reads_the_assigned_and_spaced_forms_alike(assigned, spaced):
    assert normalize_argv(assigned) == normalize_argv(spaced)


def test_normalize_argv_keeps_the_order_aliases_resolve_by():
    first = ["--tp", "4", "--tensor-parallel-size", "8"]
    second = ["--tensor-parallel-size", "8", "--tp", "4"]
    assert normalize_argv(first) != normalize_argv(second)
    assert normalize_argv(first) == tuple(first)


def test_normalize_argv_leaves_a_negative_value_and_a_non_json_value_alone():
    assert normalize_argv(["--offset", "-1", "--name", "a=b"]) == ("--offset", "-1", "--name", "a=b")


def test_aliases_resolve_to_the_last_spelling():
    occurrences = flag_occurrences("--tp 4 --tensor-parallel-size 8")
    assert last_alias_value(occurrences, frozenset({"tp", "tensor-parallel-size"})) == "8"


def test_speculative_config_merges_the_per_key_flags_over_the_object():
    occurrences = flag_occurrences(
        '--speculative-config {"method":"mtp","num_speculative_tokens":2} --speculative-config.num-speculative-tokens 3'
    )
    assert speculative_config(occurrences) == {"method": "mtp", "num_speculative_tokens": "3"}
