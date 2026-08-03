# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session-breakdown env filter: record the eval prompt wire-format, still drop secrets.

MAGPIE_EVAL_TOKENIZED_REQUESTS records whether the accuracy eval sent string
prompts (``false``, forced on PD so the sglang_router does not 422) or lm_eval's
default token-id prompts. It must survive into the breakdown so a PD run's
accuracy is distinguishable from an aggregated one's -- but its name contains
"TOKEN", which the credential denylist would otherwise strip.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.collectors.sessions import _filter_envs


def test_eval_wire_format_env_is_recorded_despite_the_token_substring() -> None:
    """MAGPIE_EVAL_TOKENIZED_REQUESTS is force-kept even though it contains TOKEN."""
    out = _filter_envs({"MAGPIE_EVAL_TOKENIZED_REQUESTS": "false"})

    assert out == {"MAGPIE_EVAL_TOKENIZED_REQUESTS": "false"}


def test_eval_wire_format_absent_stays_absent() -> None:
    """Aggregated runs never set it, so it simply does not appear (no injection)."""
    out = _filter_envs({"TP": "8"})

    assert "MAGPIE_EVAL_TOKENIZED_REQUESTS" not in out
    assert out == {"TP": "8"}


def test_real_secret_keys_are_still_dropped() -> None:
    """The force-list is one exact key; genuine credential-shaped keys still go."""
    out = _filter_envs(
        {
            "HYPERLOOM_API_TOKEN": "sk-should-not-appear",
            "SGLANG_API_KEY": "secret",
            "HF_TOKEN": "hf_secret",
            "MAGPIE_EVAL_TOKENIZED_REQUESTS": "false",
        }
    )

    assert out == {"MAGPIE_EVAL_TOKENIZED_REQUESTS": "false"}


def test_non_allowlisted_keys_are_dropped() -> None:
    """Anything outside the allowlist/force-list is not recorded."""
    out = _filter_envs({"HOME": "/root", "PATH": "/usr/bin", "RANDOM_KNOB": "1"})

    assert out == {}
