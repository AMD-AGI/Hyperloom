# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A credential is named by a marker anywhere in the name, or TOKEN as a whole segment; a tuning knob is not."""

from __future__ import annotations

import pytest

from hyperloom_common.credential_names import is_credential_env_name


@pytest.mark.parametrize(
    "name", ["HF_TOKEN", "hf_token", "HF_TOKEN_2", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD"]
)
def test_a_credential_name_is_one(name):
    assert is_credential_env_name(name)


@pytest.mark.parametrize(
    "name", ["SGLANG_MAX_NUM_BATCHED_TOKENS", "TOKENIZERS_PARALLELISM", "SGLANG_USE_AITER_FP8_PER_TOKEN", "HIP_FORCE"]
)
def test_a_tuning_knob_is_not_a_credential(name):
    assert not is_credential_env_name(name)
