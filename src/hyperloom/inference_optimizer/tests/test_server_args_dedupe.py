# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Server-args deduplication and cumulative-KEEP merging."""

from __future__ import annotations

from hyperloom.inference_optimizer.grid_server_args import dedupe_extra_server_args, merge_cumulative_extra_server_args


def test_dedupe_empty():
    assert dedupe_extra_server_args("") == ""


def test_dedupe_keeps_last_value():
    out = dedupe_extra_server_args("--tp 1 --tp 8")
    assert out == "--tp 8"


def test_dedupe_multi_value_flag():
    out = dedupe_extra_server_args("--cuda-graph-bs 1 2 4 --tp 8")
    assert "--cuda-graph-bs 1 2 4" in out
    assert "--tp 8" in out


def test_dedupe_positional_token():
    out = dedupe_extra_server_args("foo --tp 8")
    assert out == "foo --tp 8"


def test_dedupe_normalizes_equals_form():
    out = dedupe_extra_server_args("--attention-backend=ROCM_ATTN --attention-backend ROCM_AITER_FA")
    assert out == "--attention-backend ROCM_AITER_FA"


def test_dedupe_preserves_json_and_collapses_other_flags():
    args = (
        '--json-model-override-args {"rope_scaling":null} '
        "--attention-backend ROCM_ATTN --attention-backend ROCM_AITER_FA"
    )
    assert dedupe_extra_server_args(args) == (
        '--json-model-override-args {"rope_scaling":null} --attention-backend ROCM_AITER_FA'
    )


def test_dedupe_warm_replay_json_flags_without_skipping_other_dedup():
    args = (
        '--speculative-config \'{"method":"ngram","num_speculative_tokens":7}\' '
        '--compilation-config \'{"pass_config":{"enable_sp":true}}\' '
        "--max-num-seqs 512 --max-num-seqs 1024"
    )
    out = dedupe_extra_server_args(args)
    assert out == (
        '--speculative-config {"method":"ngram","num_speculative_tokens":7} '
        '--compilation-config {"pass_config":{"enable_sp":true}} '
        "--max-num-seqs 1024"
    )


# ---- merge_cumulative_extra_server_args ----

_merge = merge_cumulative_extra_server_args


def test_merge_prefers_full():
    assert _merge("--a 1", "--b 2", "--a 1 --b 2") == "--a 1 --b 2"


def test_merge_candidate_and_base_disjoint():
    out = _merge("--a 1", "--b 2", "")
    assert "--a 1" in out and "--b 2" in out


def test_merge_candidate_contains_base():
    out = _merge("--a 1", "--a 1 --b 2", "")
    assert out == "--a 1 --b 2"


def test_merge_candidate_only():
    assert _merge("", "--b 2", "") == "--b 2"


def test_merge_dedupes_attention_backend_equals_space_mix():
    out = _merge(
        "--attention-backend=ROCM_ATTN",
        "--attention-backend ROCM_AITER_FA",
        "",
    )
    assert out == "--attention-backend ROCM_AITER_FA"


def test_merge_preserves_json_while_deduping_other_flags():
    args = (
        '--json-model-override-args {"rope_scaling":null} '
        "--attention-backend ROCM_ATTN --attention-backend ROCM_AITER_FA"
    )
    assert _merge("", args, "") == (
        '--json-model-override-args {"rope_scaling":null} --attention-backend ROCM_AITER_FA'
    )
