# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A failed test's reason names the cause its server log or its benchmark's stderr states.

The logs are sanitised copies of the three kinds that ended a 12 h session as ``no_known_signature``: the server
refusing its own command line, the evaluation client answered ``400``, and a worker dying on a Python exception. The
fourth, an import that failed, is the server.log lines of a real SGLang server (one MI355X card) asked for
``--load-format fastsafetensors``. The two HIP logs are lines of the next 12 h session's server.logs, where twelve
candidates ended as ``hip_kernel_missing`` although no kernel was missing: a patched kernel that read out of bounds and
a CUDA graph capture that was invalidated.
"""

from __future__ import annotations

import pytest

from hyperloom.measure import log_causes
from hyperloom.measure.failures import FailureClass, classify_test_failure
from hyperloom.measure.tests import eval_refusal_logs

ARGPARSE_LOG = (
    "usage: launch_server.py [-h] --model-path MODEL_PATH [--tokenizer-path TOKENIZER_PATH]\n"
    "                        [--tokenizer-mode {auto,slow}] [--chunked-prefill-size CHUNKED_PREFILL_SIZE]\n"
    "                        [--chunked-prefill-with-prefix-cache] [--tp-size TP_SIZE]\n"
    "launch_server.py: error: ambiguous option: --chunked-prefill could match --chunked-prefill-size, "
    "--chunked-prefill-with-prefix-cache\n"
)

SERVER_LOG_WITH_BAD_REQUEST = (
    "[2026-10-20 23:41:01 TP0] Decode batch. #running-req: 64, #token: 71232, token usage: 0.21, gen throughput (token/s): 4812.4\n"
    'INFO:     127.0.0.1:52144 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request\n'
    "[2026-10-20 23:41:02 TP0] Decode batch. #running-req: 0, #token: 0, token usage: 0.00, gen throughput (token/s): 0.0\n"
)
LM_EVAL_STDERR = (
    "2026-10-20:23:41:00 INFO     [evaluator:305] Running generate_until requests\n"
    "2026-10-20:23:41:03 WARNING  [models.api_models:489] API request failed with error message: "
    '{"object":"error","message":"This model\'s maximum context length is 5120 tokens. However, you requested 5900 '
    'tokens in the messages.","type":"BadRequestError","param":null,"code":400}. Retrying...\n'
    "Traceback (most recent call last):\n"
    '  File "/opt/venv/lib/python3.12/site-packages/lm_eval/models/api_models.py", line 512, in amodel_call\n'
    "    response.raise_for_status()\n"
    "aiohttp.client_exceptions.ClientResponseError: 400, message='Bad Request', "
    "url='http://127.0.0.1:30000/v1/chat/completions'\n"
)

PYTHON_EXCEPTION_LOG = (
    "[2026-10-20 23:44:09 TP0] Scheduler hit an exception: Traceback (most recent call last):\n"
    '  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 2531, in run_scheduler_process\n'
    "    scheduler.event_loop_overlap()\n"
    '  File "/sgl-workspace/aiter/aiter/ops/gemm_op_a8w8.py", line 301, in gemm_a8w8_bpreshuffle\n'
    '    assert m % 16 == 0, f"M must be a multiple of 16, got {m}"\n'
    "AssertionError: M must be a multiple of 16, got 63\n"
    "\n"
    "[2026-10-20 23:44:09] Received sigquit from a child process. It usually means the child failed.\n"
)
VLLM_PREFIXED_EXCEPTION_LOG = (
    "(EngineCore_0 pid=4417) ERROR 10-20 23:44:09 [core.py:700] Traceback (most recent call last):\n"
    '(EngineCore_0 pid=4417) ERROR 10-20 23:44:09 [core.py:700]   File "vllm/v1/engine/core.py", line 691, in run\n'
    "(EngineCore_0 pid=4417) ERROR 10-20 23:44:09 [core.py:700] ValueError: 8 attention heads cannot be split into 3 shards\n"
)
FASTSAFETENSORS_LOG = (
    "[2026-10-21 06:23:02] Ignore import error when loading sglang.srt.models.falcon_mamba: cannot import name "
    "'causal_conv1d_fn' from 'sglang.srt.layers.attention.mamba.mamba' "
    "(/sgl-workspace/sglang/python/sglang/srt/layers/attention/mamba/mamba.py)\n"
    "[2026-10-21 06:23:03] Ignore import error when loading sglang.srt.models.inkling: No module named 'cutlass'\n"
    "[2026-10-21 06:23:03] Ignore import error when loading sglang.srt.models.mamba: cannot import name "
    "'causal_conv1d_fn' from 'sglang.srt.layers.attention.mamba.mamba' "
    "(/sgl-workspace/sglang/python/sglang/srt/layers/attention/mamba/mamba.py)\n"
    "[2026-10-21 06:23:03] Scheduler hit an exception: Traceback (most recent call last):\n"
    "    raise ImportError(\n"
    "ImportError: Please install fastsafetensors via `pip install fastsafetensors`\n"
    "\n"
    "[2026-10-21 06:23:03] Received sigquit from a child process. It usually means the child failed.\n"
    "[2026-10-21 06:23:03] kill_process_tree called: parent_pid=16242, include_parent=True, pid=16242\n"
)
ILLEGAL_ADDRESS_LOG = (
    "[aiter] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.\n"
    "[2026-10-21 14:46:55] Scheduler hit an exception: Traceback (most recent call last):\n"
    '  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py", line 182, '
    "in capture_one\n"
    "torch.AcceleratorError: HIP error: an illegal memory access was encountered\n"
    "Search for `hipErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html for more "
    "information.\n"
    "\n"
    "Exception: Capture cuda graph failed: HIP error: an illegal memory access was encountered\n"
    "Search for `hipErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html for more "
    "information.\n"
    "[2026-10-21 14:46:56] Received sigquit from a child process. It usually means the child failed.\n"
)
CAPTURE_INVALIDATED_LOG = (
    "Exception ignored in: <function CompiledKernel.__del__ at 0x77be94b6acb0>\n"
    "RuntimeError: Triton Error [HIP]:  Code: 900, Messsage: operation not permitted when stream is capturing\n"
    "[2026-10-21 15:29:10] Scheduler hit an exception: Traceback (most recent call last):\n"
    "RuntimeError: HIP runtime error: operation failed due to a previous error during capture. hip_check_error.hpp: 18in "
    "function: hip_check_error\n"
    "torch.AcceleratorError: HIP error: operation failed due to a previous error during capture\n"
    "Search for `hipErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html "
    "for more information.\n"
)
HIP_WARNING_THEN_EXCEPTION_LOG = (
    "UserWarning: HIP warning: an illegal memory access was encountered (Triggered internally at /pytorch/aten/src)\n"
    "torch.AcceleratorError: HIP error: an illegal memory access was encountered\n"
    "Search for `hipErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html\n"
    "HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\n"
)
UNDEFINED_SYMBOL_LOG = (
    "Traceback (most recent call last):\n"
    "ImportError: /opt/sgl-kernel/common_ops.so: undefined symbol: _ZN4aiter8fmha_fwdEv\n"
)
PLAIN_LOG = "INFO server booted\nINFO request finished in 3.1s\nsegfault somewhere\n"


def _classify(*logs: str, returncode: int = 1):
    verdict = classify_test_failure(returncode=returncode, logs=list(logs))
    return verdict.failure_class, verdict.failure_reason


def test_a_refused_flag_is_a_candidate_failure_that_names_the_offending_line() -> None:
    failure_class, reason = _classify("stderr of the launcher", ARGPARSE_LOG)

    assert failure_class is FailureClass.CANDIDATE
    assert reason == (
        "serve_flag: launch_server.py: error: ambiguous option: --chunked-prefill could match "
        "--chunked-prefill-size, --chunked-prefill-with-prefix-cache"
    )


def test_a_flag_the_shared_table_already_knows_also_names_its_line() -> None:
    log = "launch_server.py: error: unrecognized arguments: --no-such-flag\n"

    assert _classify(log) == (
        FailureClass.CANDIDATE,
        "serve_flag: launch_server.py: error: unrecognized arguments: --no-such-flag",
    )


@pytest.mark.parametrize(
    "line",
    [
        "aiter/csrc/gemm.cu:88:5: error: use of undeclared identifier 'foo'",
        "kernel.cpp:3:2: error: argument to __builtin_bit_cast must be a constant expression",
        "gemm.hip:12:9: error: invalid conversion value 'x'",
    ],
)
def test_a_compiler_error_is_not_a_refused_flag(line: str) -> None:
    assert _classify(line + "\n") == (FailureClass.UNCLEAR, "no_known_signature")


def test_a_refused_evaluation_request_stays_unclear_and_shows_the_servers_message() -> None:
    failure_class, reason = _classify("", SERVER_LOG_WITH_BAD_REQUEST, LM_EVAL_STDERR)

    assert failure_class is FailureClass.UNCLEAR
    assert reason.startswith("lm_eval_http_400: ")
    assert "maximum context length is 5120 tokens" in reason


def test_an_evaluation_crash_whose_log_holds_a_refusal_names_the_refusal_and_stays_a_candidate_failure() -> None:
    log = eval_refusal_logs.EVAL_HEAD + eval_refusal_logs.CASCADE + eval_refusal_logs.EVAL_TAIL

    failure_class, reason = _classify(log)

    assert failure_class is FailureClass.CANDIDATE
    assert reason.startswith("lm_eval_http_400: ")
    assert "max_completion_tokens is too large: 4096" in reason and "at most 2304 completion tokens" in reason


def test_an_evaluation_crash_with_no_refusal_in_its_log_is_still_the_generic_end() -> None:
    log = eval_refusal_logs.CASCADE + eval_refusal_logs.EVAL_TAIL

    assert _classify(log) == (FailureClass.CANDIDATE, "eval_runtime_failure")


def test_a_400_in_the_server_access_log_alone_is_named_too() -> None:
    assert _classify(SERVER_LOG_WITH_BAD_REQUEST) == (
        FailureClass.UNCLEAR,
        'lm_eval_http_400: INFO:     127.0.0.1:52144 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request',
    )


def test_a_python_exception_stays_unclear_and_shows_its_last_error_line() -> None:
    assert _classify(PYTHON_EXCEPTION_LOG) == (
        FailureClass.UNCLEAR,
        "python_exception: AssertionError: M must be a multiple of 16, got 63",
    )


def test_an_exception_behind_a_process_prefix_is_found_as_well() -> None:
    failure_class, reason = _classify(VLLM_PREFIXED_EXCEPTION_LOG)

    assert failure_class is FailureClass.UNCLEAR
    assert reason.endswith("ValueError: 8 attention heads cannot be split into 3 shards")


def test_a_log_that_names_nothing_stays_no_known_signature() -> None:
    assert _classify(PLAIN_LOG) == (FailureClass.UNCLEAR, "no_known_signature")


def test_a_signature_the_shared_table_knows_wins_over_the_exception_line() -> None:
    log = PYTHON_EXCEPTION_LOG + "ModuleNotFoundError: No module named 'sglang_patch'\n"

    assert _classify(log) == (
        FailureClass.CANDIDATE,
        "import_error: ModuleNotFoundError: No module named 'sglang_patch'",
    )


def test_a_failed_import_is_a_candidate_failure_that_names_the_import_line() -> None:
    assert _classify("stderr of the launcher", FASTSAFETENSORS_LOG) == (
        FailureClass.CANDIDATE,
        "import_error: ImportError: Please install fastsafetensors via `pip install fastsafetensors`",
    )


def test_an_import_the_server_skipped_on_purpose_is_not_the_cause() -> None:
    skipped = "\n".join(FASTSAFETENSORS_LOG.splitlines()[:3]) + "\n"

    assert _classify(skipped) == (FailureClass.UNCLEAR, "no_known_signature")


def test_a_failed_import_with_no_message_still_ends_as_the_bare_kind() -> None:
    assert _classify("Traceback (most recent call last):\n  raise ImportError(\nImportError:\n") == (
        FailureClass.CANDIDATE,
        "import_error",
    )


def test_a_kernel_that_faulted_is_a_runtime_error_that_names_the_fault_not_a_missing_kernel() -> None:
    assert _classify(ILLEGAL_ADDRESS_LOG) == (
        FailureClass.CANDIDATE,
        "hip_runtime_error: Exception: Capture cuda graph failed: HIP error: an illegal memory access was encountered",
    )


def test_a_graph_capture_that_was_invalidated_says_so_in_the_reason() -> None:
    assert _classify("stderr of the launcher", CAPTURE_INVALIDATED_LOG) == (
        FailureClass.CANDIDATE,
        "hip_runtime_error: torch.AcceleratorError: HIP error: operation failed due to a previous error during capture",
    )


def test_a_hip_runtime_error_prefers_the_exception_to_the_warning_before_it() -> None:
    assert _classify(HIP_WARNING_THEN_EXCEPTION_LOG)[1] == (
        "hip_runtime_error: torch.AcceleratorError: HIP error: an illegal memory access was encountered"
    )


def test_a_hip_runtime_error_with_only_the_token_quotes_the_line_that_names_it() -> None:
    reason = _classify("Search for `hipErrorLaunchFailure' in https://docs.nvidia.com/cuda/\n")[1]

    assert reason == "hip_runtime_error: Search for `hipErrorLaunchFailure' in https://docs.nvidia.com/cuda/"


def test_an_absent_symbol_is_a_missing_kernel_that_names_the_symbol() -> None:
    assert _classify(UNDEFINED_SYMBOL_LOG) == (
        FailureClass.CANDIDATE,
        "hip_kernel_missing: ImportError: /opt/sgl-kernel/common_ops.so: undefined symbol: _ZN4aiter8fmha_fwdEv",
    )


def test_a_port_taken_still_wins_over_a_refused_flag() -> None:
    log = ARGPARSE_LOG + "OSError: [Errno 98] error while attempting to bind on address: address already in use\n"

    assert _classify(log) == (FailureClass.INFRA, "port_in_use")


def test_a_cause_line_is_one_bounded_line() -> None:
    reason = _classify("launch_server.py: error: unrecognized arguments: " + "--x" * 400 + "\r\n")[1]

    assert "\n" not in reason and "\r" not in reason
    assert len(reason) <= len("serve_flag: ") + log_causes.CAUSE_CHARS


def test_a_stalled_server_keeps_its_reason_when_its_log_holds_an_exception() -> None:
    verdict = classify_test_failure(returncode=-911, logs=[PYTHON_EXCEPTION_LOG])

    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.UNCLEAR, "server_stalled")


@pytest.mark.parametrize(
    ("reader", "log", "expected"),
    [
        (log_causes.argparse_error, ARGPARSE_LOG, "launch_server.py: error: ambiguous option:"),
        (log_causes.bad_request, LM_EVAL_STDERR, "API request failed with error message:"),
        (log_causes.python_exception, PYTHON_EXCEPTION_LOG, "AssertionError: M must be"),
        (log_causes.import_error, FASTSAFETENSORS_LOG, "ImportError: Please install fastsafetensors"),
        (log_causes.hip_error, ILLEGAL_ADDRESS_LOG, "HIP error: an illegal memory access"),
        (log_causes.missing_hip_kernel, UNDEFINED_SYMBOL_LOG, "undefined symbol: _ZN4aiter8fmha_fwdEv"),
    ],
)
def test_each_reader_returns_the_line_that_names_the_cause_and_none_for_a_plain_log(reader, log, expected) -> None:
    assert expected in (reader(log) or "")
    assert reader(PLAIN_LOG) is None
