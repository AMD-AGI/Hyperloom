# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lifting a launch from a shell script: kept, unresolved and runner-owned flags, the model, and literal exports."""

from __future__ import annotations

from inference_suite.recipe.shell_launch import parse_shell_launch

_VLLM_SCRIPT = """#!/usr/bin/env bash
source "$(dirname "$0")/../benchmark_lib.sh"
export VLLM_ROCM_USE_AITER=1
export HF_TOKEN=secret
PORT=${PORT:-8888}
vllm serve $MODEL --port $PORT \\
--tensor-parallel-size=$TP \\
--block-size 128 \\
--no-enable-prefix-caching \\
--served-model-name $NAME \\
--max-model-len $MAX_MODEL_LEN \\
--result-dir=/tmp/out \\
--trust-remote-code > $SERVER_LOG 2>&1 &
"""


def test_static_flags_are_kept_in_script_order():
    launch = parse_shell_launch(_VLLM_SCRIPT, framework="vllm")

    assert launch.has_entrypoint
    assert launch.args == ("--block-size", "128", "--no-enable-prefix-caching", "--trust-remote-code")


def test_flags_valued_by_shell_variables_are_listed_not_kept():
    launch = parse_shell_launch(_VLLM_SCRIPT, framework="vllm")

    assert launch.unresolved_flags == ("--tensor-parallel-size", "--max-model-len")
    assert not any("$" in token for token in launch.args)


def test_runner_owned_flags_are_stripped_with_their_values():
    launch = parse_shell_launch(_VLLM_SCRIPT, framework="vllm")

    assert launch.stripped == ("--port", "--served-model-name", "--result-dir")
    assert "/tmp/out" not in launch.args


def test_exports_are_lifted_unfiltered():
    launch = parse_shell_launch(_VLLM_SCRIPT, framework="vllm")

    assert launch.env == {"VLLM_ROCM_USE_AITER": "1", "HF_TOKEN": "secret"}


def test_continuation_lines_parse_like_the_single_line_form():
    single = (
        "vllm serve $MODEL --block-size 128 --no-enable-prefix-caching --trust-remote-code "
        "--tensor-parallel-size=$TP --max-model-len $MAX_MODEL_LEN\n"
    )

    lifted = parse_shell_launch(single, framework="vllm")
    assert lifted.args == parse_shell_launch(_VLLM_SCRIPT, framework="vllm").args


def test_flag_equals_literal_stays_one_token():
    launch = parse_shell_launch("vllm serve $MODEL --tensor-parallel-size=$TP --block-size=32\n", framework="vllm")

    assert launch.args == ("--block-size=32",)
    assert launch.unresolved_flags == ("--tensor-parallel-size",)


def test_sglang_launch_server_entrypoint():
    text = (
        "export SGLANG_USE_AITER=1\n"
        "python3 -m sglang.launch_server --model-path=$MODEL --port $PORT --tp $TP --chunked-prefill-size 8192\n"
    )

    launch = parse_shell_launch(text, framework="sglang")

    assert launch.args == ("--chunked-prefill-size", "8192")
    assert launch.unresolved_flags == ("--tp",)
    assert launch.model is None
    assert launch.env == {"SGLANG_USE_AITER": "1"}


def test_sglang_serve_entrypoint_as_the_cookbook_writes_it():
    text = (
        "export SGLANG_OPT_USE_TOPK_V2=1\n"
        "sglang serve \\\n"
        "  --trust-remote-code \\\n"
        "  --model-path amd/GLM-5.2-MXFP4 \\\n"
        "  --tp 4 \\\n"
        "  --kv-cache-dtype fp8_e4m3 \\\n"
        "  --host 0.0.0.0 \\\n"
        "  --port 30000\n"
    )

    launch = parse_shell_launch(text, framework="sglang")

    assert launch.args == ("--trust-remote-code", "--tp", "4", "--kv-cache-dtype", "fp8_e4m3")
    assert launch.model == "amd/GLM-5.2-MXFP4"
    assert launch.stripped == ("--host", "--port")


def test_serve_cli_name_is_not_taken_for_the_model():
    launch = parse_shell_launch("sglang serve --model-path=$MODEL --tp 8\n", framework="sglang")

    assert launch.args == ("--tp", "8")
    assert launch.model is None


def test_positional_model_is_kept_literal():
    launch = parse_shell_launch("vllm serve /models/Kimi-K3 --block-size 64\n", framework="vllm")

    assert launch.model == "/models/Kimi-K3"


def test_script_without_the_entrypoint_carries_exports_only():
    launch = parse_shell_launch("export SGLANG_USE_AITER=1\necho hello\n", framework="sglang")

    assert not launch.has_entrypoint
    assert launch.args == ()
    assert launch.env == {"SGLANG_USE_AITER": "1"}


def test_self_defaults_resolve_and_other_expansions_are_unresolved():
    text = """\
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE-1}"
export VLLM_ROCM_USE_AITER=${OTHER:-1}
export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-$DEFAULT}
"""

    launch = parse_shell_launch(text, framework="vllm")

    assert launch.env == {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0", "NCCL_DMABUF_ENABLE": "1"}
    assert launch.unresolved_env == ("VLLM_ROCM_USE_AITER", "VLLM_USE_RUST_FRONTEND")


def test_literal_dollars_are_kept_and_dynamic_commands_unresolved():
    text = """export AITER_CONFIG_GEMM_BF16='/tables/$literal file.csv'
export DYNAMIC=$HOME
export COMMAND=`printf surprise`
export DEFAULT=${DEFAULT:-`printf surprise`}
export CHAIN=value;true
export MULTI='first
second'
"""

    launch = parse_shell_launch(text, framework="sglang")

    assert launch.env == {"AITER_CONFIG_GEMM_BF16": "/tables/$literal file.csv", "MULTI": "first\nsecond"}
    assert launch.unresolved_env == ("DYNAMIC", "COMMAND", "DEFAULT", "CHAIN")
