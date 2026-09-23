# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract for the canonical ``env_*`` readers: unset falls back, malformed raises."""

from __future__ import annotations

import pytest

from hyperloom.common.env import (
    EnvValueError,
    env_bool,
    env_flag,
    env_float,
    env_int,
    env_str,
    is_truthy,
)


@pytest.mark.parametrize("token", ["1", "true", "TRUE", " yes ", "on"])
def test_true_vocabulary(token: str) -> None:
    assert is_truthy(token) is True


@pytest.mark.parametrize("token", ["0", "false", "FALSE", " no ", "off", ""])
def test_false_vocabulary(token: str) -> None:
    assert is_truthy(token, default=True) is False


def test_unset_takes_the_default() -> None:
    assert is_truthy(None) is False
    assert is_truthy(None, default=True) is True


@pytest.mark.parametrize("token", ["ture", "enabled", "2", "y"])
def test_unrecognised_boolean_raises_rather_than_taking_the_default(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv("HL_FLAG", token)
    with pytest.raises(EnvValueError) as err:
        env_bool("HL_FLAG", default=True)
    assert "HL_FLAG" in str(err.value)
    assert token in str(err.value)


@pytest.mark.parametrize("token", ["ture", "enabled", "2", "y"])
def test_is_truthy_stays_lenient_for_values_it_did_not_read(token: str) -> None:
    """``is_truthy`` interprets LLM-authored Intent params and operator-supplied
    grid ``extra_envs``, not the environment: an arbitrary value there is data
    to fall back on, not a configuration error that should abort the run."""
    assert is_truthy(token) is False
    assert is_truthy(token, default=True) is True


def test_env_flag_is_the_opt_in_lenient_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that picked ``env_flag`` asked for the fallback ``env_bool`` refuses."""
    monkeypatch.setenv("HL_FLAG", "ture")
    assert env_flag("HL_FLAG", default=True) is True
    assert env_flag("HL_FLAG") is False


def test_env_bool_reads_and_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_TEST_BOOL", raising=False)
    assert env_bool("HL_TEST_BOOL", default=True) is True
    monkeypatch.setenv("HL_TEST_BOOL", " YES ")
    assert env_bool("HL_TEST_BOOL") is True
    monkeypatch.setenv("HL_TEST_BOOL", "0")
    assert env_bool("HL_TEST_BOOL", default=True) is False
    monkeypatch.setenv("HL_TEST_BOOL", "ture")
    with pytest.raises(EnvValueError, match="HL_TEST_BOOL"):
        env_bool("HL_TEST_BOOL", default=True)


def test_env_int_unset_or_blank_takes_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_TEST_INT", raising=False)
    assert env_int("HL_TEST_INT", 5) == 5
    monkeypatch.setenv("HL_TEST_INT", "   ")
    assert env_int("HL_TEST_INT", 5) == 5
    monkeypatch.setenv("HL_TEST_INT", " 7 ")
    assert env_int("HL_TEST_INT") == 7


@pytest.mark.parametrize("raw", ["bad", "30s", "7.5", "1_0_"])
def test_env_int_malformed_raises(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("HL_TEST_INT", raw)
    with pytest.raises(EnvValueError, match="HL_TEST_INT"):
        env_int("HL_TEST_INT", 3)


def test_env_float_unset_or_blank_takes_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_TEST_FLOAT", raising=False)
    assert env_float("HL_TEST_FLOAT", 1.25) == pytest.approx(1.25)
    monkeypatch.setenv("HL_TEST_FLOAT", "")
    assert env_float("HL_TEST_FLOAT", 1.25) == pytest.approx(1.25)
    monkeypatch.setenv("HL_TEST_FLOAT", " 2.5 ")
    assert env_float("HL_TEST_FLOAT") == pytest.approx(2.5)


def test_env_float_malformed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HL_TEST_FLOAT", "not-a-float")
    with pytest.raises(EnvValueError, match="HL_TEST_FLOAT"):
        env_float("HL_TEST_FLOAT", 3.5)


def test_env_str_strips_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_TEST_STR", raising=False)
    assert env_str("HL_TEST_STR", "fallback") == "fallback"
    monkeypatch.setenv("HL_TEST_STR", "  value  ")
    assert env_str("HL_TEST_STR") == "value"


def test_env_value_error_is_a_value_error() -> None:
    assert issubclass(EnvValueError, ValueError)
