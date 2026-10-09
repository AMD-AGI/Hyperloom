# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the predictor's operator interface: CLI flags and environment."""

from __future__ import annotations

import argparse

import pytest

from hyperloom.orchestrator.predictor import config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (config.ENV_ENDPOINT, config.ENV_MODE, config.ENV_TIMEOUT_SEC):
        monkeypatch.delenv(name, raising=False)


def test_no_endpoint_means_disabled_whatever_the_mode(monkeypatch):
    monkeypatch.setenv(config.ENV_MODE, "active")
    assert not config.load().enabled


def test_an_endpoint_alone_is_shadow(monkeypatch):
    monkeypatch.setenv(config.ENV_ENDPOINT, " http://p:8973 ")
    conf = config.load()
    assert (conf.endpoint, conf.mode, conf.enabled, conf.enqueues) == ("http://p:8973", "shadow", True, False)
    assert conf.timeout_sec == config.DEFAULT_TIMEOUT_SEC


@pytest.mark.parametrize(("mode", "enabled", "enqueues"), [("active", True, True), ("off", False, False)])
def test_mode_switches(monkeypatch, mode, enabled, enqueues):
    monkeypatch.setenv(config.ENV_ENDPOINT, "http://p:8973")
    monkeypatch.setenv(config.ENV_MODE, mode.upper())
    conf = config.load()
    assert (conf.enabled, conf.enqueues) == (enabled, enqueues)


@pytest.mark.parametrize(
    ("name", "raw", "field", "expected"),
    [
        (config.ENV_MODE, "loud", "mode", config.DEFAULT_MODE),
        (config.ENV_TIMEOUT_SEC, "soon", "timeout_sec", config.DEFAULT_TIMEOUT_SEC),
        (config.ENV_TIMEOUT_SEC, "0.5", "timeout_sec", config.DEFAULT_TIMEOUT_SEC),
        (config.ENV_TIMEOUT_SEC, "inf", "timeout_sec", config.DEFAULT_TIMEOUT_SEC),
        (config.ENV_TIMEOUT_SEC, "30", "timeout_sec", 30.0),
    ],
)
def test_a_bad_value_falls_back_to_its_default(monkeypatch, name, raw, field, expected):
    monkeypatch.setenv(name, raw)
    assert getattr(config.load(), field) == expected


def _optimize_args(*argv: str) -> argparse.Namespace:
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    return _build_parser().parse_args(["optimize", "--model", "m", *argv])


def test_flags_parse_and_reject_an_unknown_mode():
    args = _optimize_args("--primatune-endpoint", "http://p:8973", "--primatune-mode", "active")
    assert (args.primatune_endpoint, args.primatune_mode) == ("http://p:8973", "active")
    with pytest.raises(SystemExit):
        _optimize_args("--primatune-mode", "loud")


def test_flags_are_exported_and_an_absent_flag_keeps_the_shell_value(monkeypatch, capsys):
    from hyperloom.inference_optimizer.cli import _export_predictor_settings

    monkeypatch.setenv(config.ENV_MODE, "active")
    _export_predictor_settings(_optimize_args("--primatune-endpoint", "http://p:8973"))
    conf = config.load()
    assert (conf.endpoint, conf.mode) == ("http://p:8973", "active")
    assert "predictor          : active at http://p:8973" in capsys.readouterr().out

    _export_predictor_settings(_optimize_args("--primatune-mode", "off"))
    assert not config.load().enabled
