# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for multi-node server-args denylist validation."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.multi_node._internal import server_args_safety as sas


def test_allows_normal_perf_flags():
    sas.validate_server_args("--mem-fraction-static 0.9 --moe-runner-backend aiter")


def test_allows_max_model_len():
    sas.validate_server_args("--max-model-len 8192")


def test_allows_speculative_config_json():
    sas.validate_server_args('--speculative-config {"method":"deepseek_mtp"}')


def test_allows_speculative_draft_model_path():
    # Regression: a legitimate tuning flag ending in ``-path`` must not be
    # rejected by the broad suffix guard (eagle3 speculative-decoding sweep).
    sas.validate_server_args(
        "--speculative-algorithm EAGLE3 "
        "--speculative-draft-model-path /wekafs/models/draft "
        "--speculative-num-steps 3"
    )
    assert not sas.is_denied_server_flag("--speculative-draft-model-path")


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
