# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import random
import statistics

import pytest

from hyperloom.common import perf_metric
from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.toolbox.fake_model import (
    DEFAULT_BASELINE,
    TPOT_P50,
    TTFT_P50,
    Direction,
    Fate,
    FailureRatios,
    AttemptConstraints,
    FakeMeasurer,
    FakeToolboxConfig,
    NoiseModel,
)


def _request(**overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.EXPLORE,
        "branch_id": "b1",
        "parent_commit": "c0",
    }
    return TestRequest(**{**fields, **overrides})


def test_default_node_bias_matches_the_measured_cross_node_spread() -> None:
    model = NoiseModel()
    magnitudes = sorted(abs(model.node_bias(7, f"node-{index}")) for index in range(4000))
    assert statistics.median(magnitudes) == pytest.approx(0.004, rel=0.15)
    assert magnitudes[int(0.9 * len(magnitudes))] == pytest.approx(0.033, rel=0.2)
    signs = {model.node_bias(7, f"node-{index}") > 0 for index in range(50)}
    assert signs == {True, False}


def test_node_bias_is_fixed_per_node_and_seed() -> None:
    model = NoiseModel()
    assert model.node_bias(1, "n") == model.node_bias(1, "n")
    assert model.node_bias(1, "n") != model.node_bias(2, "n")
    assert model.node_bias(1, "n") != model.node_bias(1, "m")


def test_zero_noise_gives_exact_values() -> None:
    model = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
    assert model.node_bias(1, "n") == 0.0
    assert model.test_noise(random.Random(1)) == 0.0


def test_noise_rejects_an_impossible_spread() -> None:
    with pytest.raises(ValueError, match="node_bias_median <= node_bias_p90"):
        NoiseModel(node_bias_median=0.05, node_bias_p90=0.01)


def test_failure_ratios_must_be_a_distribution() -> None:
    with pytest.raises(ValueError, match="add up to at most 1"):
        FailureRatios(infra=0.6, candidate=0.5)
    with pytest.raises(ValueError, match="lie in"):
        FailureRatios(unclear=-0.1)


def test_failure_draws_follow_the_ratios() -> None:
    ratios = FailureRatios(infra=0.1, unclear=0.2, candidate=0.3, measurement=0.1)
    rng = random.Random(3)
    draws = [ratios.draw(rng) for _ in range(20000)]
    share = {fate: draws.count(fate) / len(draws) for fate in Fate}
    assert share[Fate.INFRA] == pytest.approx(0.1, abs=0.01)
    assert share[Fate.UNCLEAR] == pytest.approx(0.2, abs=0.01)
    assert share[Fate.CANDIDATE] == pytest.approx(0.3, abs=0.01)
    assert share[Fate.MEASUREMENT] == pytest.approx(0.1, abs=0.01)
    assert share[Fate.OK] == pytest.approx(0.3, abs=0.01)


def _tps(outcome) -> list[float]:
    return [r.metrics[GRADED_OUTPUT] for r in outcome.rounds if r.phase == "measure"]


def test_values_without_noise_are_the_baseline_times_the_effect() -> None:
    quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
    request = _request()
    config = FakeToolboxConfig(noise=quiet, default_effect=0.05, measure_rounds=2)
    outcome = FakeMeasurer(config).run(request, "r1", 0)
    assert _tps(outcome) == [pytest.approx(1050.0)] * 2
    warm = [r for r in outcome.rounds if r.phase == "warmup"]
    assert warm[0].metrics[GRADED_OUTPUT] == pytest.approx(840.0)
    assert outcome.rounds[1].metrics[TTFT_P50] == pytest.approx(200.0 / 1.05)


def test_effect_by_fingerprint_and_by_function() -> None:
    delta = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 8"))
    quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
    keyed = FakeToolboxConfig(noise=quiet, effects={delta.fingerprint(): 0.1})
    run = FakeMeasurer(keyed).run
    assert _tps(run(_request(delta=delta), "r", 0))[0] == pytest.approx(1100)
    assert _tps(run(_request(), "r", 0))[0] == pytest.approx(1000)
    fn = FakeToolboxConfig(noise=quiet, effect_fn=lambda request: -0.5, effects={delta.fingerprint(): 0.1})
    got = FakeMeasurer(fn).run(_request(delta=delta), "r", 0)
    assert _tps(got)[0] == pytest.approx(500)


def test_a_node_keeps_its_bias_across_tests() -> None:
    config = FakeToolboxConfig(nodes=("only",), noise=NoiseModel(test_noise_sigma=0.0), measure_rounds=1)
    measurer = FakeMeasurer(config)
    first = measurer.run(_request(), "r1", 0)
    second = measurer.run(_request(), "r2", 0)
    assert _tps(first) == pytest.approx(_tps(second), rel=1e-3)
    assert _tps(first)[0] != 1000.0


def test_same_seed_same_values_and_another_seed_other_values() -> None:
    def values(seed: int) -> list[float]:
        measurer = FakeMeasurer(FakeToolboxConfig(seed=seed))
        return _tps(measurer.run(_request(), "r1", 0))

    assert values(5) == values(5)
    assert values(5) != values(6)


def test_a_fresh_sandbox_avoids_the_given_nodes_and_flags_when_it_cannot() -> None:
    config = FakeToolboxConfig(nodes=("a", "b"))
    measurer = FakeMeasurer(config)
    away = measurer.run(_request(), "r", 0, AttemptConstraints(avoid=frozenset({"a"})))
    assert (away.sandbox.node_name, away.sandbox.same_node) == ("b", False)
    stuck = measurer.run(_request(), "r", 0, AttemptConstraints(avoid=frozenset({"a", "b"})))
    assert stuck.sandbox.same_node is True


def test_quarantined_nodes_are_never_used_and_no_node_left_is_an_infra_failure() -> None:
    measurer = FakeMeasurer(FakeToolboxConfig(nodes=("a", "b")))
    outcome = measurer.run(_request(), "r", 0, AttemptConstraints(unavailable=frozenset({"a"})))
    assert outcome.sandbox.node_name == "b"
    none_left = measurer.run(_request(), "r", 0, AttemptConstraints(unavailable=frozenset({"a", "b"})))
    assert (none_left.fate, none_left.sandbox) == (Fate.INFRA, None)


def test_an_accuracy_test_has_scores_and_no_rounds() -> None:
    config = FakeToolboxConfig(accuracy_score=0.9)
    request = _request(kind=TestKind.ACCURACY, purpose=Purpose.ACCURACY, accuracy=True)
    outcome = FakeMeasurer(config).run(request, "r", 0)
    assert outcome.rounds == ()
    assert outcome.accuracy[0].score == pytest.approx(0.9, abs=0.01)
    plain = FakeMeasurer(config).run(_request(), "r", 0)
    assert plain.accuracy == ()


def test_concurrency_of_a_request_sets_the_rounds() -> None:
    config = FakeToolboxConfig(measure_rounds=2)
    outcome = FakeMeasurer(config).run(_request(concurrency=(8, 16)), "r", 0)
    assert [(r.phase, r.concurrency) for r in outcome.rounds] == [
        ("warmup", 8),
        ("measure", 8),
        ("measure", 8),
        ("warmup", 16),
        ("measure", 16),
        ("measure", 16),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"nodes": ()},
        {"measure_rounds": 0},
        {"branch_queue_limit": 0},
        {"main_concurrency": (0,)},
        {"default_k": 0},
        {"baseline": {"mystery": 1.0}},
        {"error_rates": {"x": 101.0}},
    ],
)
def test_config_refuses_nonsense(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FakeToolboxConfig(**overrides)


def _measure(config: FakeToolboxConfig, request: TestRequest) -> dict[str, float]:
    outcome = FakeMeasurer(config).run(request, "r", 0)
    return dict(next(r for r in outcome.rounds if r.phase == "measure").metrics)


def test_the_baseline_is_keyed_by_the_benchmark_report_names() -> None:
    assert set(DEFAULT_BASELINE) == {
        perf_metric.GRADED_OUTPUT,
        perf_metric.GRADED_INTVTY_P50,
        perf_metric.GRADED_INTVTY,
        perf_metric.GRADED_DURATION,
        perf_metric.GRADED_ERROR_RATE,
        TTFT_P50,
        TPOT_P50,
    }
    assert {TTFT_P50, TPOT_P50} <= set(perf_metric.GRADED_AXIS_KEYS)


def test_a_faster_candidate_measures_more_throughput_and_interactivity_and_less_latency() -> None:
    quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
    slow = _measure(FakeToolboxConfig(noise=quiet), _request())
    fast = _measure(FakeToolboxConfig(noise=quiet, default_effect=0.1), _request())
    for higher in (GRADED_OUTPUT, GRADED_INTVTY, GRADED_INTVTY_P50):
        assert fast[higher] > slow[higher], higher
    for lower in (TTFT_P50, TPOT_P50):
        assert fast[lower] < slow[lower], lower
    assert fast[GRADED_DURATION] == slow[GRADED_DURATION] == DEFAULT_BASELINE[GRADED_DURATION]
    assert fast[GRADED_ERROR_RATE] == slow[GRADED_ERROR_RATE] == 0.0


def test_a_metric_moves_as_its_declared_direction_says_not_as_its_name_says() -> None:
    quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
    config = FakeToolboxConfig(
        noise=quiet,
        default_effect=0.25,
        baseline={"custom_up": 100.0, "custom_down_tps": 100.0},
        directions={"custom_up": Direction.UP, "custom_down_tps": Direction.DOWN},
    )
    assert _measure(config, _request()) == {"custom_up": pytest.approx(125.0), "custom_down_tps": pytest.approx(80.0)}


def test_the_error_rate_follows_the_delta_and_fills_the_failed_requests() -> None:
    delta = RecipeDelta(config=ConfigDelta(server_args="--flaky"))
    config = FakeToolboxConfig(error_rates={delta.fingerprint(): 2.5})
    flaky = FakeMeasurer(config).run(_request(delta=delta), "r", 0)
    measured = [r for r in flaky.rounds if r.phase == "measure"]
    assert {r.metrics[GRADED_ERROR_RATE] for r in measured} == {2.5}
    assert {r.failed_requests for r in measured} == {25}
    assert _measure(config, _request())[GRADED_ERROR_RATE] == 0.0


def test_the_accuracy_metric_defaults_to_the_one_the_rule_gate_reads_first() -> None:
    assert FakeToolboxConfig().accuracy_metric == "exact_match,strict-match"
