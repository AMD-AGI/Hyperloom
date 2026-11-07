# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the baseline round classifies a server that failed to come up."""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_warmup")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "8")


# _classify_subprocess_error unit tests

from hyperloom.orchestrator.actions.executors.benchmark_run.bringup import (
    _classify_subprocess_error,
)


def test_classify_fast_exit_unknown_backend():
    assert (
        _classify_subprocess_error(5.0, "ValueError: Unknown attention backend: 'ROCM_FLASH'") == "fast_exit_arg_error"
    )


def test_classify_fast_exit_unrecognized_args():
    assert _classify_subprocess_error(2.0, "error: unrecognized arguments: --bogus-flag") == "fast_exit_arg_error"


def test_classify_slow_failure_not_arg_error():
    """A generic runtime ValueError does not establish argument validation."""
    assert _classify_subprocess_error(120.0, "ValueError: some runtime error") == "subprocess_nonzero"


_CACHE_CONFIG_ERROR = (
    "pydantic_core._pydantic_core.ValidationError: 1 validation error for CacheConfig\n"
    "gpu_memory_utilization\n"
    "Input should be less than or equal to 1 [type=less_than_equal, input_value=1.5, input_type=float]"
)


@pytest.mark.parametrize("elapsed", [8.0, 30.0, 31.08, 31.09, 70.74])
def test_cache_config_validation_is_not_a_worker_crash(elapsed):
    assert _classify_subprocess_error(elapsed, _CACHE_CONFIG_ERROR) == "fast_exit_arg_error"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            "ValidationError: 1 validation error for ModelConfig\nModel architecture is not registered",
            "subprocess_nonzero",
        ),
        ("ValidationError: 1 validation error for Sample\nmissing field", "subprocess_nonzero"),
        ("CacheConfig initialized\nRuntimeError: worker process died", "subprocess_nonzero"),
    ],
)
def test_cache_config_validation_preserves_other_failure_routes(error, expected):
    assert _classify_subprocess_error(70.74, error) == expected


def test_classify_fast_exit_without_pattern():
    """A fast exit without arg-error patterns stays subprocess_nonzero."""
    assert _classify_subprocess_error(3.0, "Segmentation fault (core dumped)") == "subprocess_nonzero"


def test_classify_fast_runtime_value_error_not_arg_error():
    """A generic fast runtime ValueError is not enough for arg-error routing."""
    assert _classify_subprocess_error(3.0, "ValueError: tensor shape mismatch during warmup") == "subprocess_nonzero"


def test_classify_value_error_with_argv_dump_not_arg_error():
    """A command/argv dump containing flags is not arg validation by itself."""
    assert (
        _classify_subprocess_error(
            3.0,
            "ValueError: tensor shape mismatch during warmup\nargv: vllm serve --model /models/foo --tp 8",
        )
        == "subprocess_nonzero"
    )


def test_classify_subprocess_error_none_tail_does_not_crash():
    # A slow failure with no captured stderr must not raise.
    assert _classify_subprocess_error(600.0, None) == "subprocess_nonzero"


def test_classify_kv_cache_oom_after_weight_load():
    # KV-cache OOM must be detected regardless of elapsed time.
    tail = (
        "ValueError: Loaded weights leave no GPU memory for the KV cache "
        "under --mem-fraction-static=0.7. Raise --mem-fraction-static above 0.737"
    )
    assert _classify_subprocess_error(600.0, tail) == "kv_cache_oom"


def test_classify_kv_cache_oom_fast_exit():
    tail = "no GPU memory for the KV cache"
    assert _classify_subprocess_error(3.0, tail) == "kv_cache_oom"


def test_classify_non_kv_oom_still_nonzero():
    assert _classify_subprocess_error(600.0, "some other runtime failure") == "subprocess_nonzero"
