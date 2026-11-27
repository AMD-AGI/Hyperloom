# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.inspector.redact import MASK, is_secret_key, redact, redact_text
from hyperloom.inspector.tests.conftest import FAKE_KEY, FAKE_SECRET


@pytest.mark.parametrize("key", ["api_key", "ANTHROPIC_API_KEY", "HF_TOKEN", "db-password", "credentials", "secret"])
def test_secret_keys(key):
    assert is_secret_key(key)


@pytest.mark.parametrize("key", ["input_tokens", "output_tokens", "max_tokens", "tokenizer", "keyword", "path"])
def test_counters_and_plain_keys_stay_visible(key):
    assert not is_secret_key(key)


@pytest.mark.parametrize(
    "text",
    [
        f"key {FAKE_KEY} here",
        "hf_" + "a" * 30,
        "ghp_" + "b" * 30,
        "Authorization: Bearer abcdefgh12345678",
        f"x-api-key: {FAKE_SECRET}",
        f"ANTHROPIC_API_KEY={FAKE_SECRET}",
        '"HF_TOKEN": "hfvalue123"',
    ],
)
def test_text_patterns_are_masked(text):
    out = redact_text(text)
    assert MASK in out
    assert "abcdefgh12345678" not in out and FAKE_SECRET not in out and FAKE_KEY not in out and "hfvalue123" not in out


def test_url_credentials_keep_user_and_host():
    assert redact_text("https://bob:hunter2@example.com/x") == f"https://bob:{MASK}@example.com/x"


def test_plain_text_is_unchanged():
    text = "max_num_batched_tokens=8192 --tp 4 tokens=100"
    assert redact_text(text) == text


def test_redact_recurses_and_keeps_counters():
    value = {
        "api_key": "abc",
        "input_tokens": 1000,
        "nested": [{"HF_TOKEN": "zzz", "msg": f"use {FAKE_KEY}"}],
        "empty_token": "",
    }
    out = redact(value)
    assert out["api_key"] == MASK
    assert out["input_tokens"] == 1000
    assert out["nested"][0]["HF_TOKEN"] == MASK
    assert FAKE_KEY not in out["nested"][0]["msg"]
    assert out["empty_token"] == ""
    assert value["api_key"] == "abc", "the input must not be mutated"


def test_hyperloom_mask_shows_as_the_inspector_s():
    assert redact_text(f"authorization={FAKE_SECRET}") == f"authorization={MASK}"


@pytest.mark.parametrize(
    "text",
    [
        f"--api-key {FAKE_SECRET} --tp 4",
        f"--hf-token={FAKE_SECRET}",
        f"AWS_SECRET_ACCESS_KEY={FAKE_SECRET}",
        f'"LLM_GATEWAY_KEY": "{FAKE_SECRET}"',
        f"Authorization: Basic {FAKE_SECRET}",
    ],
)
def test_credential_flags_names_and_headers_are_masked_in_text(text):
    out = redact_text(text)
    assert FAKE_SECRET not in out and MASK in out


def test_tuning_flags_stay_visible():
    text = "--max-num-batched-tokens 8192 --tp 4 --model /models/Qwen3-8B --kv-cache-dtype fp8"
    assert redact_text(text) == text


def test_an_environment_mapping_masks_every_credential_name_hyperloom_does():
    value = {
        "launch_env": {"MY_SERVICE_KEY": "v1", "HYPERLOOM_SANDBOX_KEY_DEV": "v2", "VLLM_USE_V1": "1"},
        "operator_extra_env": {"MY_CUSTOM_HEADERS": "x-key: v3"},
        "idempotency_key": "baseline-1",
    }
    out = redact(value)
    assert out["launch_env"] == {"MY_SERVICE_KEY": MASK, "HYPERLOOM_SANDBOX_KEY_DEV": MASK, "VLLM_USE_V1": "1"}
    assert out["operator_extra_env"]["MY_CUSTOM_HEADERS"] == MASK
    assert out["idempotency_key"] == "baseline-1", "a key outside an environment mapping is a document field"


def test_hyperloom_secret_names_are_masked_as_keys_anywhere():
    assert redact({"task": {"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE"}})["task"]["AWS_ACCESS_KEY_ID"] == MASK
