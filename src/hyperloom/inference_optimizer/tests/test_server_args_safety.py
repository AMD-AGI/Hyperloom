# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for multi-node server-args denylist validation."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.multi_node._internal import server_args_safety as sas


@pytest.mark.parametrize(
    "args",
    [
        pytest.param("--mem-fraction-static 0.9 --moe-runner-backend aiter", id="perf-flags"),
        pytest.param("--max-model-len 8192", id="max-model-len"),
        pytest.param('--speculative-config {"method":"deepseek_mtp"}', id="speculative-config-json"),
        pytest.param("--speculative-draft-model-path=/wekafs/models/draft", id="draft-model-path-equals-form"),
    ],
)
def test_allows_normal_perf_flags(args: str):
    sas.validate_server_args(args)


def test_allows_speculative_draft_model_path():
    # Regression: a legitimate tuning flag ending in ``-path`` must not be rejected by the broad suffix guard (eagle3
    # speculative-decoding sweep).
    sas.validate_server_args(
        "--speculative-algorithm EAGLE3 --speculative-draft-model-path /wekafs/models/draft --speculative-num-steps 3"
    )
    assert not sas.is_denied_server_flag("--speculative-draft-model-path")


@pytest.mark.parametrize(
    "value",
    [
        "Qwen/EAGLE3-Qwen3-30B",  # bare repo id: every pod would download its own
        "hf://org/draft",
        "http://evil.example/draft",
        "./draft",
        "/wekafs/models/../../etc/passwd",
    ],
)
def test_denies_unsafe_speculative_draft_model_path_values(value: str):
    # The name-level exemption must not carry over to arbitrary values.
    with pytest.raises(sas.ServerArgsRejected):
        sas.validate_server_args(f"--speculative-draft-model-path {value}")
    with pytest.raises(sas.ServerArgsRejected):
        sas.validate_server_args(f"--speculative-draft-model-path={value}")


@pytest.mark.parametrize(
    "args",
    [
        "--speculative-draft-model-path --speculative-num-steps 3",
        "--mem-fraction-static 0.9 --speculative-draft-model-path",
    ],
)
def test_denies_dangling_speculative_draft_model_path(args: str):
    # A value-less path flag would otherwise swallow the next flag as its value.
    with pytest.raises(sas.ServerArgsRejected):
        sas.validate_server_args(args)


def test_unsafe_flag_value_report_names_flag_and_reason():
    assert sas.find_unsafe_flag_values("--speculative-draft-model-path relative/draft") == [
        "--speculative-draft-model-path: must be an absolute path, not a repo id or URI"
    ]


def test_explicit_deny_still_wins_over_suffix_exemption():
    # Hard-denied flags must never be re-enabled via the exemption path.
    assert sas.is_denied_server_flag("--model-path")
    assert sas.is_denied_server_flag("--adapter-model-path")


@pytest.mark.parametrize(
    "args",
    [
        "--download-dir /tmp/evil",
        "--tokenizer-path /etc/passwd",
        "--tokenizer /etc/passwd",
        "--model-path /weights/evil",
        "--model /weights/evil",
        "--chat-template /etc/passwd",
        "--lora-modules name=/evil",
        "--lora-paths /evil",
        '--hf-overrides \'{"architectures":["Evil"]}\'',
        "--config /etc/passwd",
        "--revision evil-branch",
        "--custom-weight-path /evil",
    ],
)
def test_denies_path_and_model_injection_vectors(args: str):
    with pytest.raises(sas.ServerArgsRejected):
        sas.validate_server_args(args)


def test_is_denied_server_flag_suffix_rule():
    assert sas.is_denied_server_flag("--custom-weight-path")
    assert not sas.is_denied_server_flag("--max-model-len")


def test_find_denied_flags_empty_for_blank():
    assert sas.find_denied_flags("") == []


def test_shell_safe_extra_args_quotes_metacharacters():
    out = sas.shell_safe_extra_args("--foo 1; touch /tmp/x")
    assert "'1;'" in out
    assert "touch" in out


def test_prepare_shell_safe_extra_args_rejects_denied():
    with pytest.raises(sas.ServerArgsRejected):
        sas.prepare_shell_safe_extra_args("--model-path /evil")
