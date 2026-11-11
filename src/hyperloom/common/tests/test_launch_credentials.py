# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A frozen base keeps every credential out: a credential-named env is redacted, any other place is reported."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from dataclasses import replace

import pytest

from hyperloom.common.env_safety import credential_arg_indices, is_credential_name
from hyperloom.common.launch_credentials import credentials_in_the_clear, redact_credential_env
from hyperloom.measure.tests.bundle_factory import base_launch
from inference_suite.contracts import GoldenAcceptance, GoldenCurve, Redacted

PROBE_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "LLM_GATEWAY_KEY",
    "HYPERLOOM_SANDBOX_KEY_MEASUREMENT",
    "DB_PASSWD",
    "SSH_PRIVATE_KEY",
    "GPG_PASSPHRASE",
    "HF_AUTH",
    "MODELSCOPE_ACCESS_KEY",
    "HF_TOKEN",
)

# Named a credential by the old substring rule too, so a rule that drops them would let a base freeze them in the clear.
RUN_TOGETHER_CREDENTIAL_NAMES = ("HFTOKEN", "GITHUBTOKEN", "SERVICE_TOKEN_CACHE", "GITHUB_TOKEN_BACKUP", "DB_KEY")


@pytest.mark.parametrize("name", PROBE_ENV_NAMES + RUN_TOGETHER_CREDENTIAL_NAMES)
def test_a_credential_named_env_is_frozen_as_its_sha256_and_never_reported_in_the_clear(name):
    base = redact_credential_env(replace(base_launch(), env={**base_launch().env, name: "plain-value"}))

    assert base.env[name] == Redacted(hashlib.sha256(b"plain-value").hexdigest())
    assert credentials_in_the_clear(base) == []


@pytest.mark.parametrize(
    "name",
    [
        "MAX_NUM_BATCHED_TOKENS",
        "TOKENIZERS_PARALLELISM",
        "SGLANG_USE_AITER_FP8_PER_TOKEN",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE",
        "SGLANG_CHAT_SEGMENT_TOKEN_CACHE",
        "SGLANG_CHAT_SEGMENT_TOKEN_CACHE_MAX_TOKENS",
        "SGLANG_TOKEN_REUSE",
        "NCCL_IB_HCA",
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD",
        "AITER_PER_TOKEN_2",
        "DEEPEP_COMBINE_TOKEN_BALANCED",
    ],
)
def test_a_tuning_knob_is_not_a_credential(name):
    assert not is_credential_name(name)


@pytest.mark.parametrize("name", PROBE_ENV_NAMES + RUN_TOGETHER_CREDENTIAL_NAMES)
def test_an_env_and_a_flag_of_the_same_name_are_both_credentials(name):
    flag = "--" + name.lower().replace("_", "-")

    assert is_credential_name(name) and is_credential_name(flag)
    assert credential_arg_indices((flag, "opaque")) == [1]
    assert credential_arg_indices((f"{flag}=opaque",)) == [0]


_SNAPSHOTS = Path(__file__).parents[3] / "inference_suite" / "tests" / "fixtures"
_SNAPSHOT_CREDENTIALS = {"--api-key", "ANTHROPIC_AUTH_TOKEN"}


def _names_in(node, names: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if re.fullmatch(r"[A-Z][A-Z0-9_]+", key):
                names.add(key)
            _names_in(value, names)
    elif isinstance(node, list):
        for value in node:
            _names_in(value, names)
    elif isinstance(node, str):
        names.update(m.split("=")[0] for m in re.findall(r"(?<![\w-])--[A-Za-z][\w-]*", node))
        names.update(re.findall(r"\b([A-Z][A-Z0-9_]{2,})=", node))


def test_no_knob_in_the_shipped_recipe_snapshots_is_a_credential():
    names: set[str] = set()
    for snapshot in _SNAPSHOTS.glob("*/*.json.gz"):
        _names_in(json.loads(gzip.decompress(snapshot.read_bytes())), names)

    assert len(names) > 300
    assert {n for n in names if is_credential_name(n)} == _SNAPSHOT_CREDENTIALS


@pytest.mark.parametrize(
    ("argv", "indices"),
    [
        (("--api-key", "X"), [1]),
        (("--api-key=X",), [0]),
        (("--hf-token=hf_" + "q" * 34,), [0]),
        (("--hf-token", "hf_" + "q" * 34), [1]),
        (("--db_password", "p"), [1]),
        (("--aws-secret-access-key-id", "p"), [1]),
        (("--admin-api-key=X", "--tp", "8"), [0]),
        (("python3", "-m", "sglang.launch_server", "--api-key=sk-" + "q" * 20), [3]),
        (("--trust", "Bearer " + "q" * 20), [1]),
        (("--hf-auth", "opaque"), [1]),
        (("--auth", "x"), [1]),
        (("--apiKey=X",), [0]),
        (("--accessKey", "x"), [1]),
        (("--hfToken", "x"), [1]),
    ],
)
def test_a_credential_flag_value_is_found_in_both_forms(argv, indices):
    assert credential_arg_indices(argv) == indices


@pytest.mark.parametrize(
    "argv",
    [
        ("--max-num-batched-tokens", "8192"),
        ("--ssl-keyfile", "/etc/key.pem"),
        ("--api-key",),
        ("--api-key", "--tp", "8"),
        ("--tokenizer", "m"),
        ("--return-tokens-as-token-ids",),
        ("--long-prefill-token-threshold", "512"),
        ("--speculative-token-map", "/m/map.pt"),
        ("--allowed-headers", '["*"]'),
        ("--bytes-per-token", "4"),
        ("--returned-token-ids", "1"),
        ("--sp-min-token-num", "4"),
        ("--encoder-cudagraph-token-budgets", "4"),
        ("--suffix-decoding-min-token-prob", "0.1"),
        ("--prefill-delayer-token-usage-low-watermark", "0.7"),
        ("-", "x"),
    ],
)
def test_a_flag_that_carries_no_credential_is_not_found(argv):
    assert credential_arg_indices(argv) == []


_GOLDEN = GoldenAcceptance(
    value=2.5,
    method="simulated",
    token_mode="t",
    spec_method="eagle",
    draft_length=3,
    curve=GoldenCurve("r", "a" * 40, "p", "b" * 64),
    args=("--hf-token", "x"),
    env={"GOLDEN_SECRET": "x"},
)


@pytest.mark.parametrize(
    ("unredacted", "named"),
    [
        (lambda b: replace(b, args=(*b.args, "--api-key", "X")), "arg"),
        (lambda b: replace(b, args=(*b.args, "--api-key=X")), "arg"),
        (lambda b: replace(b, entrypoint=(*b.entrypoint, "--api-key=sk-" + "q" * 20)), "entrypoint word"),
        (lambda b: replace(b, client=replace(b.client, args=(*b.client.args, "--api-key=X"))), "client arg"),
        (lambda b: replace(b, client=replace(b.client, env={"HF_AUTH": "x"})), "client env HF_AUTH"),
        (lambda b: replace(b, env={**b.env, "DB_PASSWD": "x"}), "env DB_PASSWD"),
        (lambda b: replace(b, golden_al=_GOLDEN), "golden arg 1, golden env GOLDEN_SECRET"),
    ],
)
def test_a_credential_outside_the_base_env_is_reported_by_place(unredacted, named):
    found = ", ".join(credentials_in_the_clear(unredacted(base_launch())))

    assert named in found and "X" not in found and "q" not in found
