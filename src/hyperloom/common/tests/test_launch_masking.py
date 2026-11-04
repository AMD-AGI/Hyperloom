# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.common.launch_masking import mask_server_args


@pytest.mark.parametrize(
    "args",
    [
        "--max-total-tokens 100000",
        "--max-prefill-tokens 8192 --max-num-batched-tokens 16384",
        "--speculative-num-draft-tokens 4",
        "--tokenizer-path /models/m --skip-tokenizer-init --tp 8",
        "--tp 8 --port 1 --kv-cache-dtype fp8",
    ],
)
def test_tuning_flags_pass_through_unchanged(args):
    assert mask_server_args(args) == args


def test_credential_flags_are_masked_by_whole_name():
    out = mask_server_args("--tp 8 --api-key abc123 --admin-api-key adm456 --hf-token=zzz9 --port 1")
    assert out == "--tp 8 --api-key [REDACTED] --admin-api-key [REDACTED] --hf-token=[REDACTED] --port 1"


def test_a_following_flag_is_never_consumed():
    assert mask_server_args("--api-key --tp 8") == "--api-key --tp 8"
    assert mask_server_args("--hf-token --max-total-tokens 5") == "--hf-token --max-total-tokens 5"
