# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Hugging Face cache that refused a test with ``[Errno 13] Permission denied`` is ``hf_cache_permission``, an
infrastructure failure naming the path, while a refused path anywhere else keeps the class it had.

The lock path is the shape the ``datasets`` library leaves next to a prepared dataset in a shared cache.
"""

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.failures import (
    HF_CACHE_PERMISSION,
    PYTHON_EXCEPTION_REASON,
    classify_test_failure,
)

LOCK = "/x/hf/datasets/_x_openai___gsm8k_main_0.0.0_e53f048.lock"
TRACEBACK = f"""Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/filelock/_unix.py", line 39, in _acquire
    fd = os.open(self.lock_file, open_flags, self._context.mode)
PermissionError: [Errno 13] Permission denied: '{LOCK}'
"""


@pytest.fixture(autouse=True)
def _no_cache_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        monkeypatch.delenv(name, raising=False)


def test_a_dataset_cache_lock_permission_error_is_infra_and_names_the_path() -> None:
    verdict = classify_test_failure(returncode=1, logs=[TRACEBACK])
    assert verdict.failure_class is FailureClass.INFRA
    assert verdict.failure_reason == f"{HF_CACHE_PERMISSION}: {LOCK}"


@pytest.mark.parametrize(
    "path",
    [
        "/root/.cache/huggingface/hub/.locks/models--x--y/abc.lock",
        "/data/HuggingFace/datasets/gsm8k/main/train.arrow",
    ],
)
def test_a_refused_path_of_a_huggingface_cache_is_infra(path: str) -> None:
    verdict = classify_test_failure(returncode=1, logs=[f"PermissionError: [Errno 13] Permission denied: '{path}'"])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, f"{HF_CACHE_PERMISSION}: {path}")


@pytest.mark.parametrize("variable", ["HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"])
def test_a_refused_path_under_the_cache_a_variable_names_is_infra(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(variable, "/shared/cache/")
    path = "/shared/cache/gsm8k/main/train.arrow"
    verdict = classify_test_failure(returncode=1, logs=[f"PermissionError: [Errno 13] Permission denied: '{path}'"])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, f"{HF_CACHE_PERMISSION}: {path}")


@pytest.mark.parametrize(
    "path",
    [
        "/workspace/out/benchmark_report.json",
        # a dataset directory's file that is not the cache's lock
        "/workspace/candidate/datasets/prompts.json",
        # a sibling of the cache a variable names, not under it
        "/shared/cache-other/x.lock",
    ],
)
def test_an_ordinary_permission_error_keeps_its_old_class(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "/shared/cache")
    line = f"PermissionError: [Errno 13] Permission denied: '{path}'"
    verdict = classify_test_failure(returncode=1, logs=[line])
    assert verdict.failure_class is FailureClass.UNCLEAR
    assert verdict.failure_reason == f"{PYTHON_EXCEPTION_REASON}: {line}"
