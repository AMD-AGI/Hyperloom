"""Tests for the in-process metrics shim."""

from __future__ import annotations

from hyperloom.agents.critic.runtime.metrics import (
    CRITIC_KB_WRITE_TOTAL,
    MetricsRegistry,
    get_registry,
)


def test_counter_increments_per_label_set():
    reg = MetricsRegistry()
    reg.counter("c").inc({"endpoint": "upsert", "status": "200"})
    reg.counter("c").inc({"endpoint": "upsert", "status": "200"})
    reg.counter("c").inc({"endpoint": "upsert", "status": "503"})
    snap = reg.snapshot()
    counter = snap["counters"]["c"]
    assert counter[(("endpoint", "upsert"), ("status", "200"))] == 2
    assert counter[(("endpoint", "upsert"), ("status", "503"))] == 1


def test_histogram_collects_samples():
    reg = MetricsRegistry()
    reg.histogram("h").observe(0.1, {"endpoint": "upsert"})
    reg.histogram("h").observe(0.2, {"endpoint": "upsert"})
    snap = reg.snapshot()
    h = snap["histograms"]["h"]
    assert h[(("endpoint", "upsert"),)] == [0.1, 0.2]


def test_singleton_registry_persists_across_calls():
    a = get_registry()
    a.reset()
    a.counter(CRITIC_KB_WRITE_TOTAL).inc({"endpoint": "upsert", "status": "200"})
    b = get_registry()
    assert a is b
    assert b.snapshot()["counters"][CRITIC_KB_WRITE_TOTAL]
    a.reset()
