# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import importlib
import time

import pytest

from hyperloom.common import failure_signature
from hyperloom.measure import failures, returncodes
from hyperloom.measure.failures import (
    LOG_TAIL_CHARS,
    Disposition,
    FailureClass,
    FailureClassification,
    classify_test_failure,
    decides_test,
)

IMPORT_ERROR_LOG = (
    "Traceback (most recent call last):\n  File \"x.py\", line 1\nModuleNotFoundError: No module named 'sglang_patch'\n"
)


#: The kind of the signature table, then the line of the log that states it.
IMPORT_ERROR_REASON = f"{failure_signature.IMPORT_ERROR}: ModuleNotFoundError: No module named 'sglang_patch'"


def _classify(**kwargs):
    return classify_test_failure(**kwargs)


def test_the_hosted_return_codes_are_distinct_negative_and_clear_of_the_local_sentinels() -> None:
    codes = sorted(returncodes.HOSTED_RETURNCODES)
    assert len(codes) == 5 and all(code < 0 for code in codes)
    local = set()
    for module in ("_subprocess_kill", "_ray_serving"):
        mod = importlib.import_module(f"hyperloom.orchestrator.actions.executors.{module}")
        local |= {v for k, v in vars(mod).items() if k.endswith(("_RC", "_RETURNCODE")) and isinstance(v, int)}
    assert local, "no local sentinel found: the collection is stale"
    assert not local & set(codes)


def test_the_literal_sentinels_in_failures_equal_their_definitions() -> None:
    kill = importlib.import_module("hyperloom.orchestrator.actions.executors._subprocess_kill")
    from hyperloom.measure import failures

    assert failures._AGENTX_PREFLIGHT_RC == kill.AGENTX_PREFLIGHT_RETURNCODE
    assert failures._EVAL_PROBE_UNPATCHABLE_RC == kill.EVAL_PROBE_UNPATCHABLE_RETURNCODE
    assert failures._SESSION_TIME_EXHAUSTED_RC == kill.SESSION_TIME_EXHAUSTED_RETURNCODE
    assert failures._ORCHESTRATOR_CANCELLED_RC == kill.ORCHESTRATOR_CANCELLED_RETURNCODE
    assert failures._SERVER_STALLED_RC == kill.DETOKENIZER_STALL_RETURNCODE


def test_a_server_that_went_silent_is_an_unclear_stall_unless_its_log_names_the_failure() -> None:
    stalled = _classify(returncode=-911, logs=["server ready but log went silent\n"])
    crashed = _classify(returncode=-911, logs=[IMPORT_ERROR_LOG])

    assert (stalled.failure_class, stalled.failure_reason) == (FailureClass.UNCLEAR, "server_stalled")
    assert (crashed.failure_class, crashed.failure_reason) == (FailureClass.CANDIDATE, IMPORT_ERROR_REASON)


def test_a_server_that_was_not_ready_in_time_is_the_candidates_failure_unless_its_log_names_another_cause() -> None:
    boot = returncodes.SERVER_BOOT_TIMEOUT_RC
    slow = _classify(
        returncode=boot, logs=["[aiter] TunableOp: tuning gemm 17/4096\n"], detail="not ready within 1800s"
    )
    bare = _classify(returncode=boot)
    crashed = _classify(returncode=boot, logs=[IMPORT_ERROR_LOG], detail="not ready within 1800s")
    bound = _classify(returncode=boot, logs=["[Errno 98] address already in use"], detail="not ready within 1800s")

    assert (slow.failure_class, slow.failure_reason) == (
        FailureClass.CANDIDATE,
        "server_boot_timeout: not ready within 1800s",
    )
    assert (bare.failure_class, bare.failure_reason) == (FailureClass.CANDIDATE, "server_boot_timeout")
    assert (crashed.failure_class, crashed.failure_reason) == (FailureClass.CANDIDATE, IMPORT_ERROR_REASON)
    assert (bound.failure_class, bound.failure_reason) == (FailureClass.INFRA, "port_in_use")


#: A full disk as sglang's Triton cache met it in the server's log, and as the runner's own stderr carries it.
QUOTA_IN_SERVER_LOG = (
    'Traceback (most recent call last):\n  File "triton/runtime/cache.py", line 88, in put\n'
    "OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/k.json'\n"
)
NO_SPACE_IN_RUNNER_STDERR = "shutil.Error: [Errno 28] No space left on device: '/work/out/rounds.json'\n"


@pytest.mark.parametrize(
    ("logs", "line"),
    [
        (["", QUOTA_IN_SERVER_LOG], "OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/k.json'"),
        ([NO_SPACE_IN_RUNNER_STDERR, ""], "shutil.Error: [Errno 28] No space left on device: '/work/out/rounds.json'"),
        (["write failed: Disk quota exceeded"], "write failed: Disk quota exceeded"),
        (["cc1plus: fatal error: No space left on device"], "cc1plus: fatal error: No space left on device"),
    ],
    ids=["edquot_server", "enospc_runner", "edquot_words", "enospc_words"],
)
def test_a_full_disk_in_the_runners_or_the_servers_output_is_an_infrastructure_failure(logs, line) -> None:
    verdict = _classify(returncode=1, logs=logs)
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, f"{failures.DISK_FULL}: {line}")


#: What a variant aborted in Magpie's AgentX dependency set-up records as its error (#163): the git fetch of the replay
#: client timed out, before any server started.
GIT_FETCH_STALL = (
    'File "Magpie/modes/benchmark/agentx.py", line 806, in ensure_agentx_dependencies\n'
    "subprocess.TimeoutExpired: Command ['git', '-C', '/w/scratch/cache/InferenceX-e625d03f', 'submodule', 'update', "
    "'--init', '--recursive', '--', 'inferencex-e2e/utils/aiperf'] timed out after 300 seconds\n"
)


def test_a_git_fetch_of_the_benchmarks_own_dependencies_that_timed_out_is_an_infrastructure_failure() -> None:
    verdict = _classify(returncode=1, logs=[GIT_FETCH_STALL])

    assert verdict.failure_class is FailureClass.INFRA
    assert verdict.failure_reason.startswith(f"{failures.GIT_FETCH_TIMEOUT}: subprocess.TimeoutExpired: Command ['git'")


#: What the replay client of an AgentX round printed when the Hub refused its trace dataset (board #161), in the words
#: huggingface_hub raises once its retries are spent; the warning before each retry comes first.
HUB_REFUSAL = "429 Too Many Requests: you have reached your 'api' rate limit."
HUB_REFUSED_CLIENT_LOG = (
    "Loading traces via aiperf public-dataset: semianalysis_cc_traces_weka_062126 (semianalysisai/cc-traces-weka-062126)\n"
    "HTTP Error 429 thrown while requesting GET https://huggingface.co/api/datasets/semianalysisai/cc-traces-weka-062126"
    "/revision/main\n"
    "Error: Hugging Face Hub request failed.\n"
    f"{HUB_REFUSAL}\n"
    "Retry after 212 seconds (0/1000 requests remaining in current 300s window).\n"
    "Failed to load tokenizer deepseek-ai/DeepSeek-V4-Flash\n"
)


def test_a_download_the_hub_refused_for_its_rate_limit_is_an_infrastructure_failure() -> None:
    verdict = _classify(
        returncode=1, logs=["AgentX benchmark client failed with exit code 1"], client_logs=[HUB_REFUSED_CLIENT_LOG]
    )

    assert (verdict.failure_class, verdict.failure_reason) == (
        FailureClass.INFRA,
        f"{failures.HUB_RATE_LIMITED}: {HUB_REFUSAL}",
    )


def test_a_client_log_is_not_read_for_a_candidates_signature_nor_a_retried_429() -> None:
    retried = "HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/x/resolve/main/README.md\n"
    tokenizer = "Failed to load tokenizer deepseek-ai/DeepSeek-V4-Flash\n"

    verdict = _classify(
        returncode=1, logs=["AgentX benchmark client failed with exit code 1"], client_logs=[retried + tokenizer]
    )

    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.UNCLEAR, failures.NO_KNOWN_SIGNATURE_REASON)
    assert _classify(returncode=1, logs=[tokenizer]).failure_class is FailureClass.CANDIDATE, (
        "the same line, from the server"
    )


def test_a_timeout_of_any_other_command_stays_unclear() -> None:
    other = "subprocess.TimeoutExpired: Command ['python3', 'bench.py'] timed out after 300 seconds\n"

    assert _classify(returncode=1, logs=[other]).failure_class is FailureClass.UNCLEAR


def test_a_full_disk_beats_the_candidate_signature_and_the_boot_timeout_it_caused() -> None:
    with_import = _classify(returncode=1, logs=[IMPORT_ERROR_LOG + QUOTA_IN_SERVER_LOG])
    slow = _classify(
        returncode=returncodes.SERVER_BOOT_TIMEOUT_RC, logs=[QUOTA_IN_SERVER_LOG], detail="not ready within 1800s"
    )

    assert with_import.failure_class is FailureClass.INFRA and with_import.failure_reason.startswith("disk_full: ")
    assert slow.failure_class is FailureClass.INFRA and slow.failure_reason.startswith("disk_full: ")


def test_a_full_disk_does_not_turn_a_class_the_runner_reported_or_another_errno_into_infra() -> None:
    rejected = _classify(returncode=0, error_class="spec_rejected", logs=[QUOTA_IN_SERVER_LOG])
    other = _classify(returncode=1, logs=["OSError: [Errno 13] Permission denied: '/x'\n"])

    assert rejected.disposition is Disposition.PROGRAM_ERROR
    assert (other.failure_class, other.failure_reason) == (
        FailureClass.UNCLEAR,
        "python_exception: OSError: [Errno 13] Permission denied: '/x'",
    )


def test_a_runner_that_reports_a_full_disk_is_an_infrastructure_failure() -> None:
    verdict = _classify(returncode=0, error_class=failures.DISK_FULL)
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, "disk_full")
    assert failures.is_disk_full(OSError(errno.EDQUOT, "Disk quota exceeded"))
    assert failures.is_disk_full(OSError(errno.ENOSPC, "No space left on device"))
    assert not failures.is_disk_full(OSError(errno.EACCES, "Permission denied"))
    assert not failures.is_disk_full(ValueError("Disk quota exceeded"))


def test_a_clean_exit_is_a_success() -> None:
    verdict = _classify(returncode=0)
    assert verdict.disposition is Disposition.SUCCEEDED and verdict.failure_class is None


def test_a_known_failure_signature_is_a_candidate_failure_named_by_the_signature() -> None:
    verdict = _classify(returncode=1, logs=["", IMPORT_ERROR_LOG])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.CANDIDATE, IMPORT_ERROR_REASON)


def test_a_non_zero_exit_without_a_known_signature_is_unclear() -> None:
    verdict = _classify(returncode=1, logs=["segfault somewhere\n"])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.UNCLEAR, "no_known_signature")


def test_a_timeout_is_unclear_even_when_the_log_carries_a_signature() -> None:
    verdict = _classify(returncode=-9, timed_out=True, logs=[IMPORT_ERROR_LOG])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.UNCLEAR, "timeout")


@pytest.mark.parametrize(
    "error_class",
    ["node_fault", "preflight_mismatch", "runner_start_failed", "runner_lost", "archive_failed", "no_spec"],
)
def test_runner_and_platform_error_classes_are_infrastructure_whatever_the_log_says(error_class: str) -> None:
    verdict = _classify(returncode=1, error_class=error_class, logs=[IMPORT_ERROR_LOG])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, error_class)


@pytest.mark.parametrize("error_class", ["spec_rejected", "result_rejected"])
def test_a_rejected_spec_or_result_is_a_program_error_with_no_failure_class(error_class: str) -> None:
    verdict = _classify(returncode=1, error_class=error_class, logs=[IMPORT_ERROR_LOG])
    assert verdict.disposition is Disposition.PROGRAM_ERROR
    assert verdict.failure_class is None and verdict.failure_reason == error_class


def test_an_error_class_nobody_handles_is_a_program_error_not_a_candidate_failure() -> None:
    verdict = _classify(returncode=1, error_class="brand_new_thing", logs=[IMPORT_ERROR_LOG])
    assert verdict.disposition is Disposition.PROGRAM_ERROR
    assert verdict.failure_reason == "unknown_error_class:brand_new_thing"


@pytest.mark.parametrize(
    ("returncode", "disposition", "failure_class", "reason"),
    [
        (returncodes.HOSTED_RUNNER_UNAVAILABLE_RC, Disposition.FAILED, FailureClass.INFRA, "runner_unavailable"),
        (returncodes.HOSTED_INFRA_FAILURE_RC, Disposition.FAILED, FailureClass.INFRA, "infra_failure"),
        (returncodes.HOSTED_SPEC_REJECTED_RC, Disposition.PROGRAM_ERROR, None, "spec_rejected"),
        (returncodes.HOSTED_RESULT_REJECTED_RC, Disposition.PROGRAM_ERROR, None, "result_rejected"),
        (-912, Disposition.FAILED, FailureClass.INFRA, "agentx_preflight"),
        (-914, Disposition.PROGRAM_ERROR, None, "eval_probe_unpatchable"),
        (-915, Disposition.STOPPED, None, "session_time_exhausted"),
        (-917, Disposition.STOPPED, None, "orchestrator_cancelled"),
    ],
)
def test_the_sentinel_return_codes_classify_by_what_they_mean(returncode, disposition, failure_class, reason) -> None:
    verdict = _classify(returncode=returncode, logs=[IMPORT_ERROR_LOG])
    assert (verdict.disposition, verdict.failure_class, verdict.failure_reason) == (disposition, failure_class, reason)


def test_the_same_log_gets_the_same_class_however_much_older_log_comes_before_it() -> None:
    short = _classify(returncode=1, logs=[IMPORT_ERROR_LOG])
    long = _classify(returncode=1, logs=["noise line\n" * (LOG_TAIL_CHARS // 4) + IMPORT_ERROR_LOG])
    assert short == long and short.failure_class is FailureClass.CANDIDATE


def test_a_signature_older_than_the_tail_is_not_read() -> None:
    padding = ("y" * 100 + "\n") * (LOG_TAIL_CHARS // 100)
    verdict = _classify(returncode=1, logs=[IMPORT_ERROR_LOG + padding])
    assert verdict.failure_class is FailureClass.UNCLEAR


def test_one_enormous_line_cannot_stall_the_classifier() -> None:
    started = time.monotonic()
    verdict = _classify(returncode=1, logs=["x" * 1_000_000 + "\n" + IMPORT_ERROR_LOG])
    assert time.monotonic() - started < 5
    assert verdict.failure_class is FailureClass.CANDIDATE
    assert _classify(returncode=1, logs=["x" * 1_000_000]).failure_class is FailureClass.UNCLEAR


def test_a_classification_cannot_be_built_incoherent() -> None:
    with pytest.raises(ValueError, match="exactly when"):
        FailureClassification(Disposition.FAILED)
    with pytest.raises(ValueError, match="exactly when"):
        FailureClassification(Disposition.PROGRAM_ERROR, FailureClass.INFRA, "x")
    with pytest.raises(ValueError, match="names its reason"):
        FailureClassification(Disposition.STOPPED)
    with pytest.raises(ValueError, match="no failure reason"):
        FailureClassification(Disposition.SUCCEEDED, failure_reason="x")


BIND_LOG = (
    "INFO launching server\n"
    "ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 39211): address already in use\n"
)


def test_a_port_taken_between_choosing_and_binding_is_an_infrastructure_failure() -> None:
    verdict = _classify(returncode=1, logs=[BIND_LOG])
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, "port_in_use")


def test_either_phrase_of_the_bind_failure_is_enough_and_it_beats_a_candidate_signature() -> None:
    assert _classify(returncode=1, logs=["OSError: Address Already In Use"]).failure_class is FailureClass.INFRA
    assert _classify(returncode=1, logs=["OSError: [Errno 98]"]).failure_class is FailureClass.INFRA
    both = _classify(returncode=1, logs=[IMPORT_ERROR_LOG + BIND_LOG])
    assert both.failure_class is FailureClass.INFRA


def test_a_successful_exit_is_not_changed_by_a_bind_message_in_its_log() -> None:
    assert _classify(returncode=0, logs=[BIND_LOG]).disposition is Disposition.SUCCEEDED


@pytest.mark.parametrize(
    ("verdict", "decides"),
    [
        (FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, "hip_runtime_error: HIP error: x"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, "import_error"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, "serve_flag: bad flag"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, "eval_runtime_failure"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, "lm_eval_http_400: refused"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.UNCLEAR, "lm_eval_http_400: refused"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.UNCLEAR, "python_exception: ValueError: x"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, "server_boot_timeout: 900s"), True),
        (FailureClassification(Disposition.FAILED, FailureClass.UNCLEAR, "no_known_signature"), False),
        (FailureClassification(Disposition.FAILED, FailureClass.UNCLEAR, "server_stalled"), False),
        (FailureClassification(Disposition.FAILED, FailureClass.INFRA, "port_in_use"), False),
        (FailureClassification(Disposition.STOPPED, failure_reason="session_time_exhausted"), False),
        (FailureClassification(Disposition.SUCCEEDED), False),
    ],
)
def test_only_a_crashed_or_refusing_server_or_a_failed_evaluation_settles_a_test(verdict, decides) -> None:
    assert decides_test(verdict) is decides
