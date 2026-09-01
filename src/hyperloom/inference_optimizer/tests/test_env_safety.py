# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for multi-node SSH env key validation.

``env_safety`` is the last-mile gate on keys that already entered a forward
dict: it blocks shell/loader injection vectors (``BLOCKED_UNTRUSTED_ENV_NAMES``
from ``hyperloom.common.env_safety``) and invalid key shapes. Credential
exclusion happens upstream in ``infera._collect_forward_env`` (prefix whitelist)
and the platform's pod env (operator ``--extra-env`` only); those keys are never
placed into the forward dict, so they are not SSH-forwarded to inference pods.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.multi_node._internal import env_safety
from hyperloom.common import env_safety as common_env_safety


def test_valid_mori_and_sglang_keys_allowed():
    assert env_safety.is_forward_env_key_allowed("MORI_DISPATCH_FOO")
    assert env_safety.is_forward_env_key_allowed("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK")
    assert env_safety.is_forward_env_key_allowed("SGLANG_USE_AITER")
    assert env_safety.is_forward_env_key_allowed("SGLANG_TORCH_PROFILER_DIR")


def test_invalid_key_shape_rejected():
    assert not env_safety.is_forward_env_key_allowed("X Y")
    assert not env_safety.is_forward_env_key_allowed("bad-key")


def test_denylist_keys_rejected():
    """Names in BLOCKED_UNTRUSTED_ENV_NAMES are blocked (loader/python/PATH/shell vectors)."""
    assert not env_safety.is_forward_env_key_allowed("LD_PRELOAD")
    assert not env_safety.is_forward_env_key_allowed("PYTHONPATH")
    assert not env_safety.is_forward_env_key_allowed("PATH")
    assert not env_safety.is_forward_env_key_allowed("IFS")


def test_non_denylist_tuning_keys_pass_shape_gate():
    """Keys outside BLOCKED_UNTRUSTED_ENV_NAMES pass the forward gate when shape is valid."""
    assert env_safety.is_forward_env_key_allowed("NCCL_IB_HCA")
    assert env_safety.is_forward_env_key_allowed("SGLANG_USE_AITER")


def test_filter_forward_env_drops_bad_keys():
    out = env_safety.filter_forward_env(
        {
            "MORI_FOO": "1",
            "LD_PRELOAD": "/evil.so",
            "X Y": "nope",
        },
        warn_on_drop=False,
    )
    assert out == {"MORI_FOO": "1"}


def test_assert_forward_env_keys_raises():
    with pytest.raises(ValueError, match="disallowed SSH forward env keys"):
        env_safety.assert_forward_env_keys({"LD_PRELOAD": "/tmp/x.so"})


def test_common_env_safety_filters_dotenv_and_kernel_agent_keys_only():
    assert common_env_safety.is_allowed_dotenv_key("OPENAI_API_KEY")
    assert common_env_safety.is_allowed_dotenv_key("HF_TOKEN")
    assert common_env_safety.is_allowed_dotenv_key("HTTPS_PROXY")
    assert common_env_safety.is_allowed_dotenv_key("HYPERLOOM_RUNTIME_DIR")
    # hyperloom-setup writes the gateway auth headers into .env, so the .env
    # loader must read them back instead of dropping them as unsupported.
    assert common_env_safety.is_allowed_dotenv_key("ANTHROPIC_CUSTOM_HEADERS")
    assert common_env_safety.is_allowed_dotenv_key("OPENAI_CUSTOM_HEADERS")
    assert not common_env_safety.is_allowed_dotenv_key("PYTHONPATH")
    assert not common_env_safety.is_allowed_dotenv_key("BAD-NAME")

    assert common_env_safety.is_allowed_kernel_agent_env_key("TRACELENS_ROOT")
    # install.sh persists the Anthropic header into kernel-agent.env.sh, so the
    # reader must accept it; the OpenAI one is read on the same terms as the
    # OpenAI URL and key already are.
    assert common_env_safety.is_allowed_kernel_agent_env_key("ANTHROPIC_CUSTOM_HEADERS")
    assert common_env_safety.is_allowed_kernel_agent_env_key("OPENAI_CUSTOM_HEADERS")
    assert common_env_safety.is_allowed_kernel_agent_env_key("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV")
    assert common_env_safety.is_allowed_kernel_agent_env_key("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS")
    # Dropped keys never reach the kernel-agent child, so an opt-in route switch
    # is inert until it is listed here.
    assert common_env_safety.is_allowed_kernel_agent_env_key("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL")
    assert not common_env_safety.is_allowed_kernel_agent_env_key("TRACELENS_TOKEN")

    allowed, dropped = common_env_safety.filter_untrusted_env_mapping(
        {
            "bench_foo": 1,
            "custom_tuning_knob": "enabled",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "LD_PRELOAD": "/tmp/agent-provided.so",
            "OPENAI_API_KEY": "secret",
            "PYTHONPATH": "/tmp/agent-provided",
            "bad key": "nope",
            "": "empty",
        },
        allow_predicate=common_env_safety.valid_env_key,
    )
    assert allowed == {
        "bench_foo": "1",
        "custom_tuning_knob": "enabled",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "LD_PRELOAD": "/tmp/agent-provided.so",
        "OPENAI_API_KEY": "secret",
        "PYTHONPATH": "/tmp/agent-provided",
    }
    assert dropped == {
        "bad key": "invalid_env_key",
        "<empty>": "invalid_env_key",
    }

    env = {"LD_PRELOAD": "evil.so", "PATH": "/bin", "SAFE": "1"}
    assert common_env_safety.scrub_child_process_env(env) is env
    assert env == {"PATH": "/bin", "SAFE": "1"}


def test_scrub_benchmark_process_env_removes_control_plane_credentials():
    env = {
        "AMD_API_KEY": "amd-secret",
        "AMD_LLM_API_KEY": "amd-llm-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "LLM_GATEWAY_KEY": "gateway-secret",
        "LLM_PROXY_API_KEY": "proxy-secret",
        "OPENAI_API_KEY": "openai-secret",
        "SAFE_API_KEY": "safe-secret",
        "HF_TOKEN": "model-download-token",
        "PATH": "/bin",
        "RUN_EVAL": "true",
    }

    assert common_env_safety.scrub_benchmark_process_env(env) is env
    assert env == {
        "HF_TOKEN": "model-download-token",
        "PATH": "/bin",
        "RUN_EVAL": "true",
    }


def test_variant_env_key_allows_workload_pins_and_blocks_hijacks():
    # Sweep, conc-sweep and shape-capture grids set these from code, so an
    # allowlist that dropped them would silently flatten every variant.
    for pinned in ("CONC", "ISL", "OSL", "NUM_PROMPTS", "RUN_EVAL", "PORT", "TP", "MAX_MODEL_LEN"):
        assert common_env_safety.is_allowed_variant_env_key(pinned)
    for knob in ("SGLANG_USE_AITER", "VLLM_USE_MTP", "AITER_CONFIG_GEMM_A8W8", "PYTORCH_TUNABLEOP_ENABLED"):
        assert common_env_safety.is_allowed_variant_env_key(knob)
    # Name-shape matching would read this as a credential; it is the private
    # model download token and has to survive.
    assert common_env_safety.is_allowed_variant_env_key("HF_TOKEN")

    for hijack in ("LD_PRELOAD", "PATH", "PYTHONPATH", "BASH_ENV", "LD_AUDIT", "PYTHONSTARTUP"):
        assert not common_env_safety.is_allowed_variant_env_key(hijack)
    for secret in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_GATEWAY_KEY"):
        assert not common_env_safety.is_allowed_variant_env_key(secret)
    assert not common_env_safety.is_allowed_variant_env_key("bad key")
    assert not common_env_safety.is_allowed_variant_env_key("")


def test_build_benchmark_env_layers_over_parent_and_normalizes(monkeypatch):
    monkeypatch.setenv("INHERITED_KNOB", "from-parent")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-benchmark")
    monkeypatch.setenv("SGLANG_USE_AITER", "0")

    env = common_env_safety.build_benchmark_env(
        {"SGLANG_USE_AITER": "1", "TP": 8, "RANDOM_RANGE_RATIO": 0.8},
        None,
        {"TP": 4, "lowercase_knob": "on"},
    )

    assert env["INHERITED_KNOB"] == "from-parent"
    assert env["SGLANG_USE_AITER"] == "1"
    assert env["TP"] == "4"
    assert env["RANDOM_RANGE_RATIO"] == "0.8"
    # Env names are conventionally upper case; a lower-case key would be inert.
    assert env["LOWERCASE_KNOB"] == "on"
    assert "OPENAI_API_KEY" not in env


def test_redact_secret_values_masks_assignments_and_bearer_tokens():
    text = "OPENAI_API_KEY=ak-sensitive-value Authorization: Bearer sensitive-token"

    redacted = common_env_safety.redact_secret_values(text)

    assert "sensitive-value" not in redacted
    assert "sensitive-token" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_redact_secret_values_masks_quoted_assignments():
    """A quoted value is the common shape: shells quote it, json.dumps escapes it."""
    double = common_env_safety.redact_secret_values('export MYAPP_PASSWORD="hunter2"')
    single = common_env_safety.redact_secret_values("export MYAPP_PASSWORD='hunter2'")
    escaped = common_env_safety.redact_secret_values('{"command": "export MYAPP_PASSWORD=\\"hunter2\\""}')

    assert double == 'export MYAPP_PASSWORD="[REDACTED]"'
    assert single == "export MYAPP_PASSWORD='[REDACTED]'"
    assert escaped == '{"command": "export MYAPP_PASSWORD=\\"[REDACTED]\\""}'
    assert common_env_safety.redact_secret_values(r"PASSWORD=C:\foo\bar") == "PASSWORD=[REDACTED]"


def test_redact_secret_values_spares_tokenizer_and_max_tokens():
    """TOKEN as a name fragment must not mask workload knobs in an optimizer."""
    for text in (
        '--tokenizer="/models/x"',
        "--tokenizer=/models/x",
        "HYPERLOOM_EVAL_BOUNDS max_tokens=4096 stop=[</s>]",
        '--max-tokens="4096"',
        "num_speculative_tokens=5",
        "TOKENIZERS_PARALLELISM=false",
        "eos_string=</s>,max_retries=5,tokenized_requests=False,max_length=13312",
        "HYPERLOOM_EVAL_BOUNDS max_tokens=4096 stop_token_ids=[154820, 154827, 154829]",
        "token_budget=128",
        "tokens_per_second=12.3",
    ):
        assert common_env_safety.redact_secret_values(text) == text

    still_secret = common_env_safety.redact_secret_values(
        'HF_TOKEN=abc HF_TOKEN_2=def ANTHROPIC_AUTH_TOKEN="tok-abcdef" api_key=real-secret'
    )
    assert "abc" not in still_secret
    assert "def" not in still_secret
    assert "tok-abcdef" not in still_secret
    assert "real-secret" not in still_secret
    assert still_secret.count("[REDACTED]") == 4


def test_redact_secret_values_masks_aws_key_and_jwt_shapes():
    """AWS access-key ids and compact JWTs are masked by shape, not by assignment."""
    access_id = "AKIA" + "IOSFODNN7EXAMPLE"
    jwt = "eyJ" + "hbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxIn0." + "aaaaaaaaaa"
    text = f"aws --secret-key {access_id} Authorization: {jwt}"

    redacted = common_env_safety.redact_secret_values(text)

    assert access_id not in redacted
    assert jwt not in redacted
    assert redacted.count("[REDACTED]") == 2
