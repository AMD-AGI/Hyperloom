# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Hugging Face cache whose ``[Errno 13] Permission denied`` ended a test is ``hf_cache_permission``, an
infrastructure failure naming the path, while a refused path anywhere else keeps the class it had, and a refusal the
process logged and survived is no cause at all.

The lock path is the shape the ``datasets`` library gives the lock it takes in its cache directory for a dataset it
prepares; the ignored line is what ``huggingface_hub`` logs when it cannot record a missing optional file in a read-only
cache and carries on.
"""

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.failures import (
    HF_CACHE_PERMISSION,
    PYTHON_EXCEPTION_REASON,
    FailureEvidence,
    classify_test_failure,
)

LOCK = "/x/hf/datasets/_x_hf_datasets_openai___gsm8k_main_0.0.0_e53f048.lock"
TRACEBACK = f"""Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/filelock/_unix.py", line 39, in _acquire
    fd = os.open(self.lock_file, open_flags, self._context.mode)
PermissionError: [Errno 13] Permission denied: '{LOCK}'
"""
IGNORED = (
    "huggingface_hub.file_download WARNING Could not cache non-existence of file. Will ignore error and continue. "
    "Error: [Errno 13] Permission denied: '/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/.no_exist'"
)
CANDIDATE_CRASH = """Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/layers/attention.py", line 12, in <module>
    from aiter.ops.my_new_kernel import run
ModuleNotFoundError: No module named 'aiter.ops.my_new_kernel'
"""


def _refused(path: str) -> str:
    return f"PermissionError: [Errno 13] Permission denied: '{path}'"


def test_a_dataset_cache_lock_permission_error_is_infra_and_names_the_path() -> None:
    verdict = classify_test_failure(returncode=1, evidence=FailureEvidence(logs=(TRACEBACK,)))
    assert verdict.failure_class is FailureClass.INFRA
    assert verdict.failure_reason == f"{HF_CACHE_PERMISSION}: {LOCK}"


@pytest.mark.parametrize(
    "path",
    [
        "/root/.cache/huggingface/hub/.locks/models--x--y/abc.lock",
        "/data/HuggingFace/datasets/gsm8k/main/train.arrow",
        # the Hub cache's layout wherever HF_HUB_CACHE puts it
        "/shared/cache/hub/.locks/models--Qwen--Qwen3-8B/0f1e.lock",
        "/shared/cache/hub/models--Qwen--Qwen3-8B/blobs/0f1e",
        "/shared/cache/hub/datasets--openai--gsm8k/snapshots/e53f048/main/test.parquet",
        # the lock datasets takes, wherever HF_DATASETS_CACHE puts the cache
        "/shared/hf_ds/_shared_hf_ds_openai___gsm8k_main_0.0.0_e53f048.lock",
    ],
)
def test_a_refused_path_of_a_huggingface_cache_is_infra(path: str) -> None:
    verdict = classify_test_failure(returncode=1, evidence=FailureEvidence(logs=(_refused(path),)))
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, f"{HF_CACHE_PERMISSION}: {path}")


@pytest.mark.parametrize(
    "logs", [[IGNORED + "\n" + CANDIDATE_CRASH], [IGNORED, CANDIDATE_CRASH], [CANDIDATE_CRASH, IGNORED]]
)
def test_an_ignored_cache_refusal_does_not_hide_the_candidates_crash(logs: list[str]) -> None:
    verdict = classify_test_failure(returncode=1, evidence=FailureEvidence(logs=tuple(logs)))
    assert verdict.failure_class is FailureClass.CANDIDATE
    assert verdict.failure_reason == "import_error: ModuleNotFoundError: No module named 'aiter.ops.my_new_kernel'"


def test_a_cache_refusal_that_ends_one_log_does_not_outweigh_a_candidates_crash_that_ends_another() -> None:
    verdict = classify_test_failure(
        returncode=1,
        evidence=FailureEvidence(
            logs=(
                CANDIDATE_CRASH,
                TRACEBACK,
            )
        ),
    )
    assert verdict.failure_class is FailureClass.CANDIDATE


def test_an_ignored_cache_refusal_alone_is_no_cause() -> None:
    verdict = classify_test_failure(returncode=1, evidence=FailureEvidence(logs=(IGNORED,)))
    assert verdict.failure_reason is not None and not verdict.failure_reason.startswith(HF_CACHE_PERMISSION)


def test_a_cache_refusal_survived_before_another_error_is_not_the_cause() -> None:
    later = "ValueError: the prompt is longer than the context"
    verdict = classify_test_failure(
        returncode=1, evidence=FailureEvidence(logs=(TRACEBACK + "retrying without the cache\n" + later,))
    )
    assert (verdict.failure_class, verdict.failure_reason) == (
        FailureClass.UNCLEAR,
        f"{PYTHON_EXCEPTION_REASON}: {later}",
    )


def test_the_cache_refusal_that_ends_the_evaluation_is_infra_whatever_the_server_logged_and_survived() -> None:
    server = f"{IGNORED}\nserver up\nserver down\n"
    verdict = classify_test_failure(
        returncode=1,
        evidence=FailureEvidence(
            logs=(
                server,
                "Running loglikelihood requests\n" + TRACEBACK,
            )
        ),
    )
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, f"{HF_CACHE_PERMISSION}: {LOCK}")


@pytest.mark.parametrize(
    "path",
    [
        "/workspace/out/benchmark_report.json",
        # a dataset directory's file that is not the cache's lock
        "/workspace/candidate/datasets/prompts.json",
        # locks of a candidate's own tree, not named after their directory as the datasets lock is
        "/workspace/candidate/datasets/build/out.lock",
        "/workspace/candidate/datasets/_out.lock",
        # under the cache a variable names, but in no layout of a Hugging Face cache
        "/shared/cache/gsm8k/main/train.arrow",
    ],
)
def test_an_ordinary_permission_error_keeps_its_old_class(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "/shared/cache")
    line = _refused(path)
    verdict = classify_test_failure(returncode=1, evidence=FailureEvidence(logs=(line,)))
    assert verdict.failure_class is FailureClass.UNCLEAR
    assert verdict.failure_reason == f"{PYTHON_EXCEPTION_REASON}: {line}"
