"""Smoke coverage for the Settings model."""

from __future__ import annotations

import pytest

from robustness_server.config import Settings, get_settings, reset_settings_for_test


def test_defaults_are_sane() -> None:
    reset_settings_for_test()
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8080
    assert s.nats_subject_filter == "events.>"
    assert s.nats_kv_bucket == "BRAIN_REGISTRY"
    assert s.database_schema == "hyperloom_robustness"
    assert s.workload_reconcile_interval_seconds > 0
    assert s.database_pool_min <= s.database_pool_max
    reset_settings_for_test()


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBUSTNESS_SERVER_PORT", "9090")
    monkeypatch.setenv("ROBUSTNESS_SERVER_NATS_DURABLE_NAME", "test-consumer")
    reset_settings_for_test()
    s = Settings()
    assert s.port == 9090
    assert s.nats_durable_name == "test-consumer"
    reset_settings_for_test()


def test_get_settings_caches() -> None:
    reset_settings_for_test()
    a = get_settings()
    b = get_settings()
    assert a is b
    reset_settings_for_test()


def test_pool_min_validation() -> None:
    with pytest.raises(ValueError):
        Settings(database_pool_max=0)
