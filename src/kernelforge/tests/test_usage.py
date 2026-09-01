"""Tests for LLM token-usage accumulation + persistence."""

import tempfile
from dataclasses import dataclass
from typing import Any

from kernelforge.tracker import ExperimentTracker, UsageAccumulator


# Minimal stand-ins for the claude-agent-sdk message types.
@dataclass
class FakeResultMessage:
    """Mirrors the SDK ResultMessage: carries usage + total_cost_usd."""

    usage: dict
    total_cost_usd: float


@dataclass
class FakeAssistantMessage:
    """Mirrors the SDK AssistantMessage: has usage but NO total_cost_usd."""

    usage: dict
    content: Any = None


def test_accumulator_sums_result_messages():
    acc = UsageAccumulator()
    acc.add_from_message(
        FakeResultMessage(
            usage={
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 5,
            },
            total_cost_usd=0.12,
        )
    )
    acc.add_from_message(
        FakeResultMessage(
            usage={"input_tokens": 200, "output_tokens": 60},
            total_cost_usd=0.18,
        )
    )
    totals = acc.totals()
    assert totals["input_tokens"] == 300
    assert totals["output_tokens"] == 100
    assert totals["cache_creation_input_tokens"] == 10
    assert totals["cache_read_input_tokens"] == 5
    assert totals["total_cost_usd"] == 0.3
    assert totals["cost_available"] is True
    assert totals["cost_source"] == "provider"
    assert totals["calls"] == 2
    assert bool(acc) is True


def test_accumulator_ignores_assistant_messages():
    # AssistantMessage.usage must NOT be double-counted (only ResultMessage is).
    acc = UsageAccumulator()
    counted = acc.add_from_message(
        FakeAssistantMessage(
            usage={"input_tokens": 999, "output_tokens": 999},
        )
    )
    assert counted is False
    assert acc.totals()["calls"] == 0
    assert bool(acc) is False


def test_accumulator_tolerates_missing_and_bad_usage():
    acc = UsageAccumulator()
    # No usage dict at all (still a counted call with a cost).
    acc.add_from_message(FakeResultMessage(usage=None, total_cost_usd=0.05))
    # Garbage token value degrades to a skip, not a crash.
    acc.add_from_message(
        FakeResultMessage(
            usage={"input_tokens": "oops", "output_tokens": 7},
            total_cost_usd=None,
        )
    )
    totals = acc.totals()
    assert totals["calls"] == 2
    assert totals["input_tokens"] == 0  # "oops" skipped
    assert totals["output_tokens"] == 7
    assert totals["total_cost_usd"] == 0.05
    assert totals["cost_available"] is False
    assert totals["cost_source"] == "partial"


def test_accumulator_marks_missing_provider_cost_unavailable():
    """Distinguish missing provider pricing from a real zero-dollar charge."""
    acc = UsageAccumulator()
    acc.add_usage({"input_tokens": 10, "output_tokens": 4})
    totals = acc.totals()
    assert totals["total_cost_usd"] == 0.0
    assert totals["cost_available"] is False
    assert totals["cost_source"] == "unavailable"


def test_accumulator_rejects_boolean_provider_cost():
    """Treat a boolean cost field as unavailable provider pricing."""
    acc = UsageAccumulator()
    acc.add_usage({"input_tokens": 10}, total_cost_usd=True)
    totals = acc.totals()
    assert totals["total_cost_usd"] == 0.0
    assert totals["cost_available"] is False
    assert totals["cost_source"] == "unavailable"


def test_set_llm_usage_persists_to_experiment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        exp = tracker.create(task_id="t", backend="forge")
        acc = UsageAccumulator()
        acc.add_from_message(
            FakeResultMessage(
                usage={"input_tokens": 50, "output_tokens": 20},
                total_cost_usd=0.4,
            )
        )
        tracker.set_llm_usage(exp.experiment_id, acc.totals())

        loaded = tracker.get(exp.experiment_id)
        assert loaded.llm_usage["input_tokens"] == 50
        assert loaded.llm_usage["output_tokens"] == 20
        assert loaded.llm_usage["total_cost_usd"] == 0.4
        assert loaded.llm_usage["calls"] == 1


def test_set_llm_usage_noop_on_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        exp = tracker.create(task_id="t", backend="forge")
        tracker.set_llm_usage(exp.experiment_id, {})  # no-op
        assert tracker.get(exp.experiment_id).llm_usage == {}


def test_experiment_llm_usage_round_trips():
    from kernelforge.tracker import Experiment

    exp = Experiment(experiment_id="x", llm_usage={"input_tokens": 9, "calls": 1})
    d = exp.to_dict()
    assert d["llm_usage"]["input_tokens"] == 9
    back = Experiment.from_dict(d)
    assert back.llm_usage == {"input_tokens": 9, "calls": 1}
