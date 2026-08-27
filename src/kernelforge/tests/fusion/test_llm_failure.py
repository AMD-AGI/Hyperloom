# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An unreachable LLM must never become a verdict about the kernel.

The incident these pin: discovery flagged a model as launch-bound
(``candidate=True``), every LLM attempt failed with a bare gateway 400, and the
run wrote ``verdict: no_opportunity`` and exited 0 — a wrong optimization
conclusion that no failure dashboard could see.
"""

from __future__ import annotations

import json

import pytest

from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.discover import complete_with_retry
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
    """An OpenAI-SDK style error carrying an HTTP status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _client(responses):
    """A chat client that replays ``responses`` (an exception raises, else text)."""
    calls = {"n": 0, "max_tokens": []}

    class _Completions:
        def create(self, **kwargs):
            calls["max_tokens"].append(kwargs.get("max_tokens"))
            item = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            if isinstance(item, Exception):
                raise item
            message = type("M", (), {"content": item})()
            return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    return client, calls


# ── classification ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_Status("nope", 401), AUTH),
        (_Status("nope", 403), AUTH),
        (_Status("too big", 413), CONTEXT_LENGTH),
        (RuntimeError("missing subscription key"), AUTH),
        (RuntimeError("prompt is too long for this model"), CONTEXT_LENGTH),
        (RuntimeError("request timed out"), TIMEOUT),
        # The incident's own error: a 400 whose body carries no usable reason.
        (
            _Status(
                "Error code: 400 - litellm.BadRequestError: AnthropicException - "
                "Prediction on deployed model failed with error: Bad Request",
                400,
            ),
            API_ERROR,
        ),
        (_Status("overloaded", 529), API_ERROR),
        (ConnectionError("connection reset by peer"), API_ERROR),
    ],
)
def test_classification_decides_whether_a_retry_can_help(error, expected):
    assert classify_llm_error(error) == expected


def _marked(message: str, rejection: bool) -> Exception:
    """A provider safety error carrying the explicit rejection marker."""

    class ProviderSafetyError(RuntimeError):
        pass

    error = ProviderSafetyError(message)
    setattr(error, AGENT_SAFETY_REJECTION_ATTR, rejection)
    return error


def test_a_safety_class_raised_for_io_is_not_a_safety_verdict():
    """The class name is shared; only the marker says which of the two this is.

    A backend raises its safety class both for "the session edited a protected
    file" and for "the guard could not read a file", and the first is fatal while
    the second is weather. Matching the name made a stalled ``git`` call abandon
    the caller's whole recipe.
    """
    assert is_agent_safety_error(_marked("protected files changed", True)) is True
    assert is_agent_safety_error(_marked("Could not snapshot /x", False)) is False

    class ProviderSafetyError(RuntimeError):
        """Unmarked, so it claims nothing about what the session did."""

    assert is_agent_safety_error(ProviderSafetyError("something")) is False


def test_a_timeout_survives_a_wrapper_raised_while_unwinding_it():
    """A rollback that failed on the way out replaces the timeout it recovered from.

    The expired clock then exists only in ``__context__``, so a classifier reading
    the outermost error alone reports the session as something other than out of
    time.
    """
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
    """A timeout says the request did not come back THIS time; credentials and an
    over-long prompt fail the same way forever."""
    assert LlmUnavailableError("x", kind=API_ERROR).retryable is True
    assert LlmUnavailableError("x", kind=TIMEOUT).retryable is True
    assert LlmUnavailableError("x", kind=AUTH).retryable is False
    assert LlmUnavailableError("x", kind=CONTEXT_LENGTH).retryable is False


# ── retry policy ──────────────────────────────────────────────────────────────


def test_backoff_reaches_minutes_not_seconds():
    """The old fixed 3s steps covered ~30s — shorter than the outage every time."""
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


# ── complete_with_retry ───────────────────────────────────────────────────────


def test_transient_failure_then_success_returns_the_answer():
    client, calls = _client([_Status("flaky", 400), '[{"name":"x"}]'])
    out = complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=4, sleep=lambda _: None)
    assert out == '[{"name":"x"}]'
    assert calls["n"] == 2


def test_max_tokens_is_not_shrunk_between_attempts():
    """Shrinking only truncated the answer; the 400s recur at every cap."""
    client, calls = _client([_Status("flaky", 400), _Status("flaky", 400), "[]"])
    complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=4, sleep=lambda _: None)
    assert calls["max_tokens"] == [2400, 2400, 2400]


def test_exhausted_retries_raise_rather_than_return_empty():
    client, calls = _client([_Status("flaky", 400)])
    with pytest.raises(LlmUnavailableError) as excinfo:
        complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=4, sleep=lambda _: None)
    assert calls["n"] == 4
    assert excinfo.value.attempts == 4


def test_a_non_retryable_failure_gives_up_immediately():
    client, calls = _client([_Status("bad key", 401)])
    with pytest.raises(LlmUnavailableError) as excinfo:
        complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=4, sleep=lambda _: None)
    assert calls["n"] == 1
    assert excinfo.value.kind == AUTH


def test_a_transient_timeout_is_retried():
    """A timeout used to abort the chain on the first one, dropping the retry the
    previous implementation had: one slow response published "unreachable"."""
    client, calls = _client([RuntimeError("request timed out"), '[{"name":"x"}]'])

    out = complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=4, sleep=lambda _: None)

    assert out == '[{"name":"x"}]'
    assert calls["n"] == 2


def test_an_exhausted_timeout_chain_reports_the_timeout_kind():
    client, calls = _client([RuntimeError("request timed out")])

    with pytest.raises(LlmUnavailableError) as excinfo:
        complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=3, sleep=lambda _: None)

    assert calls["n"] == 3
    assert excinfo.value.kind == TIMEOUT


def test_the_retry_chain_stops_at_its_deadline():
    """Attempts alone do not bound wall clock: each one may sit on the client's own
    read timeout, so five could hold discovery for over an hour."""
    client, calls = _client([RuntimeError("request timed out")])
    # First read anchors the start; every later read is past the deadline.
    reads = iter([0.0])

    with pytest.raises(LlmUnavailableError) as excinfo:
        complete_with_retry(
            client,
            "prompt",
            model="m",
            max_tokens=2400,
            attempts=5,
            deadline_sec=600.0,
            sleep=lambda _: None,
            monotonic=lambda: next(reads, 5000.0),
        )

    assert calls["n"] == 1, "past the deadline, stop rather than retry"
    assert excinfo.value.attempts == 1
    assert "deadline" in str(excinfo.value)


def test_deadline_zero_lifts_the_bound():
    client, calls = _client([RuntimeError("request timed out"), "[]"])

    out = complete_with_retry(
        client,
        "prompt",
        model="m",
        max_tokens=2400,
        attempts=4,
        deadline_sec=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 10**9,
    )

    assert out == "[]"
    assert calls["n"] == 2


def test_an_empty_completion_is_a_failure_not_an_answer():
    """Discovery's prompt demands JSON: a model with nothing to propose says [];."""
    client, calls = _client([""])
    with pytest.raises(LlmUnavailableError):
        complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=2, sleep=lambda _: None)
    assert calls["n"] == 2


def test_an_empty_json_array_is_a_real_answer():
    client, _ = _client(["[]"])
    assert complete_with_retry(client, "prompt", model="m", max_tokens=2400, attempts=2, sleep=lambda _: None) == "[]"


# ── the manifest verdict ──────────────────────────────────────────────────────


def _launch_bound_diagnosis():
    return diagnose_from_shares(
        {"gemm": 0.4, "add": 0.14, "elementwise": 0.14, "cast": 0.13, "mul": 0.08},
        busy_fraction_of_wall=0.21,
    )


def test_a_launch_bound_model_with_no_recipe_is_still_no_opportunity():
    diagnosis = _launch_bound_diagnosis()
    manifest = build_manifest(
        framework="sglang",
        model_path="/m",
        model_type="mixtral",
        diagnosis=diagnosis,
        recipe=None,
    )
    assert manifest["verdict"] == "no_opportunity"
    assert manifest["error"] is None


def test_an_unreachable_llm_is_not_reported_as_no_opportunity():
    diagnosis = _launch_bound_diagnosis()
    assert diagnosis.is_candidate, "the incident's precondition: the trace WAS a candidate"
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
    """Adding a value to an enum is not additive for a consumer that switches on
    it, so the version has to move even though every v1 field is untouched."""
    manifest = build_manifest(
        framework="sglang",
        model_path="/m",
        model_type="mixtral",
        diagnosis=_launch_bound_diagnosis(),
        recipe=None,
    )

    assert FUSION_MANIFEST_SCHEMA_VERSION == 2
    assert manifest["schema_version"] == 2
    # v1's fields keep their names, types and meaning, so a v2 reader handles a v1
    # payload and a v1 reader that only reads known keys is unaffected.
    for key in (
        "tool",
        "version",
        "verdict",
        "framework",
        "model",
        "diagnosis",
        "fusion",
        "fusion_candidates",
        "validation",
        "fusion_loop",
        "artifacts",
    ):
        assert key in manifest, key
    assert manifest["error"] is None, "null on every verdict but llm_unavailable"


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
