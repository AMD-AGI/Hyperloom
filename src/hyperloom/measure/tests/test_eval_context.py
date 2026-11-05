# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The context length a launch serves against the one the accuracy evaluation runs in."""

from __future__ import annotations

import pytest

from hyperloom.measure.eval_context import (
    EvalContextError,
    ServedContext,
    check_eval_context,
    eval_context,
    served_context,
)


@pytest.mark.parametrize(
    ("server_args", "expected"),
    [
        ("--context-length 2304", ServedContext("--context-length", 2304)),
        ("--context-length=2304 --tp 1", ServedContext("--context-length", 2304)),
        ("--max-model-len 4096", ServedContext("--max-model-len", 4096)),
        ("--tp 1 --max-model-len=4096 --kv-cache-dtype fp8_e4m3", ServedContext("--max-model-len", 4096)),
        ("--context-length 2304 --context-length 9000", ServedContext("--context-length", 9000)),
        ("--context-length +2304", ServedContext("--context-length", 2304)),
        ("--context-length 2304 --context-length auto", None),
        ("--context-length 2304 --max-model-len 3000", ServedContext("--max-model-len", 3000)),
        ("--tp 1 --kv-cache-dtype fp8_e4m3", None),
        ("", None),
        ("--context-length", None),
        ("--context-length auto", None),
        ("--context-length-extra 10", None),
    ],
)
def test_the_context_a_launch_serves_is_the_last_context_flag_with_a_whole_number(
    server_args: str, expected: ServedContext | None
) -> None:
    assert served_context(server_args) == expected


@pytest.mark.parametrize(
    ("envs", "expected"),
    [({"MAX_MODEL_LEN": 8192}, 8192), ({"MAX_MODEL_LEN": "4096"}, 4096), ({}, None), ({"MAX_MODEL_LEN": "0"}, None)],
)
def test_the_context_of_the_evaluation_is_the_max_model_len_of_the_point(envs: dict, expected: int | None) -> None:
    assert eval_context(envs) == expected


@pytest.mark.parametrize(
    ("envs", "server_args"),
    [
        ({"MAX_MODEL_LEN": 8192}, "--context-length 8192"),
        ({"MAX_MODEL_LEN": 8192}, "--max-model-len=16384"),
        ({"MAX_MODEL_LEN": 8192}, "--tp 1"),
        ({}, "--context-length 128"),
    ],
)
def test_a_launch_that_serves_the_evaluation_context_or_names_none_is_left_alone(envs: dict, server_args: str) -> None:
    check_eval_context(envs, server_args)


@pytest.mark.parametrize(
    ("limit", "flag", "tokens"), [(8192, "--context-length", 2304), (3000, "--max-model-len", 2999)]
)
def test_a_launch_that_serves_less_is_refused_naming_the_least_the_evaluation_needs(
    limit: int, flag: str, tokens: int
) -> None:
    with pytest.raises(EvalContextError) as refused:
        check_eval_context({"MAX_MODEL_LEN": limit}, f"--tp 1 {flag} {tokens}")

    message = str(refused.value)
    assert f"Serve at least {limit} tokens" in message
    assert f"{flag} {tokens}" in message
