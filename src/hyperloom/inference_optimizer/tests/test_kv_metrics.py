# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the ``/metrics`` KV sampler.

Fixtures are verbatim expositions captured off MI355X nodes, so a name or unit
that drifts upstream fails here rather than in a breakdown three weeks later.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._kv_metrics import (
    DEFAULT_METRICS_PORT,
    KvMetricsPoller,
    canonical_label_key,
    parse_prometheus_text,
    resolve_metrics_port,
    sample_from_families,
)


# Captured from a saturated SGLang 0.5.17 server. The two retract names differ
# by one word and mean different things; both are here on purpose.
SGLANG_METRICS = """\
# HELP sglang:token_usage Pool usage.
# TYPE sglang:token_usage gauge
sglang:token_usage 0.85
sglang:full_token_usage 0.8506
sglang:swa_token_usage 0.0
sglang:kv_used_tokens 27876.0
sglang:kv_available_tokens 2888.0
sglang:kv_evictable_tokens 2004.0
sglang:swa_evictable_tokens 0.0
sglang:mamba_evictable_tokens 0.0
sglang:kv_cache_memory_usage_gb 180.013
sglang:num_retracted_requests_total 48.0
sglang:num_retracted_input_tokens_total 38205.0
sglang:num_retracted_reqs{pid="4741"} 1.0
sglang:cached_tokens_total 4600439.0
sglang:cache_hit_rate 0.0
"""

VLLM_METRICS = """\
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="qwen"} 458.0
vllm:kv_cache_usage_perc 0.969
vllm:prefix_cache_queries 1000.0
vllm:prefix_cache_hits 529.0
"""


def test_parses_real_sglang_exposition():
    fam = parse_prometheus_text(SGLANG_METRICS)
    s = sample_from_families(fam)

    assert s.engine == "sglang"
    assert s.used_tokens == 27876.0
    assert s.available_tokens == 2888.0
    assert s.evictable_tokens == 2004.0
    assert s.capacity_tokens == 32768.0
    assert s.capacity_gb == 180.013
    assert s.has_readings()


def test_active_and_physical_usage_diverge():
    """The whole reason occupancy is two fields and not one.

    27876 held by running requests, 2004 more held by the prefix cache but
    releasable. Reporting only the physical 0.91 would call this pool nearly
    full when the pressure reading is 0.85.
    """
    s = sample_from_families(parse_prometheus_text(SGLANG_METRICS))

    assert s.active_pool_usage == pytest.approx(0.8506)
    assert s.physical_pool_usage == pytest.approx(29880 / 32768)
    assert s.physical_pool_usage > s.active_pool_usage


def test_reads_cumulative_retracts_not_the_instantaneous_gauge():
    """``num_retracted_requests_total`` is 48; ``num_retracted_reqs`` is 1."""
    s = sample_from_families(parse_prometheus_text(SGLANG_METRICS))

    assert list(s.retract_total.values()) == [48.0]
    assert 1.0 not in s.retract_total.values()


def test_counters_are_kept_per_label_series():
    """An engine restart resets one series; a flat total would hide that."""
    fam = parse_prometheus_text(VLLM_METRICS)
    s = sample_from_families(fam)

    assert s.preempt_total == {'engine="0",model_name="qwen"': 458.0}


def test_vllm_shape_and_legacy_usage_alias():
    s = sample_from_families(parse_prometheus_text(VLLM_METRICS))
    assert s.engine == "vllm"
    assert s.active_pool_usage == pytest.approx(0.969)
    assert s.prefix_cache_queries == 1000.0
    assert s.prefix_cache_hits == 529.0

    legacy = sample_from_families(parse_prometheus_text("vllm:gpu_cache_usage_perc 0.5\n"))
    assert legacy.active_pool_usage == pytest.approx(0.5)


def test_absent_metric_is_none_not_zero():
    """vLLM cannot express physical occupancy; that must read as unknown."""
    s = sample_from_families(parse_prometheus_text(VLLM_METRICS))

    assert s.physical_pool_usage is None
    assert s.evictable_tokens is None
    assert s.capacity_tokens is None
    assert s.capacity_gb is None


def test_prefix_cache_off_omits_cached_tokens_entirely():
    """The metric disappears rather than reading 0, so consumers must see None."""
    without = SGLANG_METRICS.replace("sglang:cached_tokens_total 4600439.0\n", "")
    s = sample_from_families(parse_prometheus_text(without))

    assert s.cached_tokens_total is None


def test_idle_zero_is_a_reading_not_an_absence():
    """An engine nobody has queried reports a true 0.0 for minutes."""
    s = sample_from_families(parse_prometheus_text("sglang:token_usage 0.0\nsglang:kv_used_tokens 0.0\n"))

    assert s.active_pool_usage == 0.0
    assert s.used_tokens == 0.0
    assert s.has_readings()


def test_hybrid_pool_takes_the_max_subpool_not_the_full_one():
    """On a hybrid model the full-attention subpool understates real pressure."""
    hybrid = "sglang:token_usage 0.10\nsglang:full_token_usage 0.10\nsglang:swa_token_usage 0.93\n"
    s = sample_from_families(parse_prometheus_text(hybrid))

    assert s.active_pool_usage == pytest.approx(0.93)


def test_labels_comments_and_timestamps_are_handled():
    text = '# HELP x help text\n# TYPE x gauge\n\nsglang:kv_used_tokens{tp_rank="0",note="a,b=c"} 12.0 1756880000123\n'
    fam = parse_prometheus_text(text)

    assert fam["sglang:kv_used_tokens"][0][0] == {"tp_rank": "0", "note": "a,b=c"}
    assert fam["sglang:kv_used_tokens"][0][1] == 12.0


def test_non_finite_values_are_dropped():
    """NaN carries no occupancy meaning and would poison every max taken over it."""
    fam = parse_prometheus_text("sglang:token_usage NaN\nsglang:kv_used_tokens +Inf\nsglang:kv_available_tokens 5.0\n")

    assert "sglang:token_usage" not in fam
    assert "sglang:kv_used_tokens" not in fam
    assert fam["sglang:kv_available_tokens"][0][1] == 5.0


def test_exposition_without_kv_metrics_has_no_readings():
    """A reachable endpoint exposing only unrelated metrics is not a KV sample."""
    s = sample_from_families(parse_prometheus_text("python_gc_objects_collected_total 12.0\n"))

    assert s.engine == ""
    assert not s.has_readings()


def test_canonical_label_key_is_order_independent():
    assert canonical_label_key({"b": "2", "a": "1"}) == canonical_label_key({"a": "1", "b": "2"})
    assert canonical_label_key({}) == ""


def test_port_resolution_prefers_config_over_env(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    assert resolve_metrics_port({"PORT": 31234}) == 31234
    assert resolve_metrics_port({}) == 9999
    monkeypatch.delenv("PORT", raising=False)
    assert resolve_metrics_port({}) == DEFAULT_METRICS_PORT
    assert resolve_metrics_port({"PORT": "not-a-port"}) == DEFAULT_METRICS_PORT


def test_poller_availability_is_tristate_and_gives_up(monkeypatch):
    """Unknown until proven; parked once an engine shows it has metrics off."""
    poller = KvMetricsPoller(port=1)
    assert poller.available is None

    def _boom(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._kv_metrics.urllib.request.urlopen",
        _boom,
    )
    for _ in range(3):
        assert poller.fetch() is None

    assert poller.available is False

    # Parked: no further requests are issued, so a later call cannot revive it.
    def _explode(*_args, **_kwargs):
        raise AssertionError("poller kept scraping after giving up")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._kv_metrics.urllib.request.urlopen",
        _explode,
    )
    assert poller.fetch() is None


def test_poller_sample_returns_none_when_endpoint_has_no_kv(monkeypatch):
    poller = KvMetricsPoller(port=1)
    monkeypatch.setattr(poller, "fetch", lambda: "python_gc_objects_collected_total 12.0\n")

    assert poller.sample() is None


def test_poller_sample_normalises_a_good_scrape(monkeypatch):
    poller = KvMetricsPoller(port=1)
    monkeypatch.setattr(poller, "fetch", lambda: SGLANG_METRICS)

    sample = poller.sample()

    assert sample is not None
    assert sample.used_tokens == 27876.0
    assert sample.mono > 0
