# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
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
        "--hf-overrides '{\"architectures\":[\"Evil\"]}'",
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
