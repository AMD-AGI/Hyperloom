# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for multi-node server-args denylist validation."""

from __future__ import annotations

import pytest

from inference_optimizer.multi_node._internal import server_args_safety as sas


def test_allows_normal_perf_flags():
    sas.validate_server_args("--mem-fraction-static 0.9 --moe-runner-backend aiter")


def test_denies_download_dir():
    with pytest.raises(sas.ServerArgsRejected, match="--download-dir"):
        sas.validate_server_args("--download-dir /tmp/evil")


def test_denies_tokenizer_path():
    with pytest.raises(sas.ServerArgsRejected, match="--tokenizer-path"):
        sas.validate_server_args("--tokenizer-path /etc/passwd")


def test_find_denied_flags_empty_for_blank():
    assert sas.find_denied_flags("") == []
