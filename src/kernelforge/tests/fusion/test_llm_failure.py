# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An unreachable LLM must never become a verdict about the kernel."""

from __future__ import annotations

import json

import pytest

from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.llm_failure import (
    AGENT_SAFETY_REJECTION_ATTR,
    API_ERROR,
    AUTH,
    CONTEXT_LENGTH,
    TIMEOUT,
    LlmUnavailableError,
    classify_llm_error,
    env_setting,
    is_agent_safety_error,
    is_agent_timeout_error,
    retry_delay,
)
from kernelforge.fusion.report import (
    FUSION_MANIFEST_SCHEMA_VERSION,
    LLM_UNAVAILABLE_VERDICT,
    build_manifest,
    write_manifest,
)


class _Status(Exception):
    """An SDK-style error carrying an HTTP status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_Status("nope", 401), AUTH),
        (_Status("nope", 403), AUTH),
        (_Status("too big", 413), CONTEXT_LENGTH),
        (RuntimeError("missing subscription key"), AUTH),
        (RuntimeError("prompt is too long for this model"), CONTEXT_LENGTH),
        (RuntimeError("request timed out"), TIMEOUT),
        (_Status("Bad Request", 400), API_ERROR),
        (_Status("overloaded", 529), API_ERROR),
        (ConnectionError("connection reset by peer"), API_ERROR),
    ],
)
def test_classification_decides_whether_a_retry_can_help(error, expected):
    assert classify_llm_error(error) == expected


def _marked(message: str, rejection: bool) -> Exception:
    class ProviderSafetyError(RuntimeError):
        pass

    error = ProviderSafetyError(message)
    setattr(error, AGENT_SAFETY_REJECTION_ATTR, rejection)
    return error


def test_a_safety_class_raised_for_io_is_not_a_safety_verdict():
    assert is_agent_safety_error(_marked("protected files changed", True)) is True
    assert is_agent_safety_error(_marked("Could not snapshot /x", False)) is False

    class ProviderSafetyError(RuntimeError):
        pass

    assert is_agent_safety_error(ProviderSafetyError("something")) is False


def test_a_timeout_survives_a_wrapper_raised_while_unwinding_it():
    try:
        try:
            raise TimeoutError("Codex timed out after 3600s")
        except TimeoutError:
            raise _marked("workspace state could not be restored", False)
    except RuntimeError as exc:
        wrapped = exc

    assert is_agent_timeout_error(wrapped) is True
    assert is_agent_timeout_error(RuntimeError("gateway reset the connection")) is False


def test_a_timeout_is_retryable_but_a_rejection_is_not():
    assert LlmUnavailableError("x", kind=API_ERROR).retryable is True
    assert LlmUnavailableError("x", kind=TIMEOUT).retryable is True
    assert LlmUnavailableError("x", kind=AUTH).retryable is False
    assert LlmUnavailableError("x", kind=CONTEXT_LENGTH).retryable is False


def test_backoff_reaches_minutes_not_seconds():
    delays = [retry_delay(attempt, base_sec=5.0, max_sec=120.0, rng=lambda: 1.0) for attempt in range(1, 6)]
    assert delays == [5.0, 15.0, 45.0, 120.0, 120.0]
    assert sum(delays) > 180


def test_backoff_jitter_spreads_a_batch_of_pods():
    low = retry_delay(3, base_sec=5.0, max_sec=120.0, rng=lambda: 0.0)
    high = retry_delay(3, base_sec=5.0, max_sec=120.0, rng=lambda: 1.0)
    assert low == pytest.approx(22.5)
    assert high == pytest.approx(45.0)


def test_env_setting_ignores_unparseable_and_negative_overrides(monkeypatch):
    monkeypatch.setenv("FF_TEST_SETTING", "not-a-number")
    assert env_setting("FF_TEST_SETTING", 7, cast=int) == 7
    monkeypatch.setenv("FF_TEST_SETTING", "-3")
    assert env_setting("FF_TEST_SETTING", 7, cast=int) == 7
    monkeypatch.setenv("FF_TEST_SETTING", "2")
    assert env_setting("FF_TEST_SETTING", 7, cast=int) == 2
    monkeypatch.delenv("FF_TEST_SETTING")
    assert env_setting("FF_TEST_SETTING", 7, cast=int) == 7


def _launch_bound_diagnosis():
    return diagnose_from_shares(
        {"gemm": 0.4, "add": 0.14, "elementwise": 0.14, "cast": 0.13, "mul": 0.08},
        busy_fraction_of_wall=0.21,
    )


def test_a_launch_bound_model_with_no_recipe_is_still_no_opportunity():
    manifest = build_manifest(
        framework="sglang",
        model_path="/m",
        model_type="mixtral",
        diagnosis=_launch_bound_diagnosis(),
        recipe=None,
    )
    assert manifest["verdict"] == "no_opportunity"
    assert manifest["error"] is None


def test_an_unreachable_llm_is_not_reported_as_no_opportunity():
    diagnosis = _launch_bound_diagnosis()
    error = LlmUnavailableError("gateway 400 x4", kind=API_ERROR, attempts=4)
    manifest = build_manifest(
        framework="sglang",
        model_path="/m",
        model_type="mixtral",
        diagnosis=diagnosis,
        recipe=None,
        verdict_override=LLM_UNAVAILABLE_VERDICT,
        error=error.to_dict(),
    )
    assert manifest["verdict"] == LLM_UNAVAILABLE_VERDICT
    assert manifest["error"]["kind"] == API_ERROR
    assert manifest["error"]["attempts"] == 4
    assert manifest["error"]["stage"] == "discovery"


def test_the_wider_verdict_enum_declares_itself_as_schema_v2():
    manifest = build_manifest(
        framework="sglang",
        model_path="/m",
        model_type="mixtral",
        diagnosis=_launch_bound_diagnosis(),
        recipe=None,
    )
    assert FUSION_MANIFEST_SCHEMA_VERSION == 2
    assert manifest["schema_version"] == 2
    assert manifest["error"] is None


def test_the_manifest_lands_whole_with_the_bytes_its_readers_parse(tmp_path):
    manifest = build_manifest(
        framework="sglang",
        model_path="/m",
        model_type="mixtral",
        diagnosis=_launch_bound_diagnosis(),
        recipe=None,
    )
    path = write_manifest(manifest, tmp_path / "out")
    raw = path.read_bytes()
    assert raw == (json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".")]
