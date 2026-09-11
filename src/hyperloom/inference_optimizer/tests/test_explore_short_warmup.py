"""The warmup round has to warm, not measure.

Every explore variant pays for two full benchmarks. The first one boots the
server, runs the accuracy gate, and has its throughput read only for
success/failure before being discarded -- yet it runs at the same length as the
round that is kept, which ``_workload_envs`` sizes at five to ten waves of the
concurrency.

Cutting it to one wave is the largest safe saving in an explore round, so these
tests pin the two things that make it safe: it must be short enough to matter,
and it must fall back to the long warmup rather than to a broken one whenever
the concurrency cannot be established.
"""
from __future__ import annotations

import textwrap

import pytest

from hyperloom.orchestrator.actions.executors.explore import (
    WARMUP_WAVES,
    _short_warmup_enabled,
    _warmup_num_prompts,
)


def _cfg(tmp_path, body: str):
    p = tmp_path / "bench.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_warmup_is_one_wave_of_the_concurrency(tmp_path):
    cfg = _cfg(tmp_path, """
        benchmark:
          envs:
            TP: 8
            CONC: 64
            ISL: 1024
            OSL: 1024
    """)
    assert _warmup_num_prompts(cfg) == 64 * WARMUP_WAVES


def test_warmup_is_shorter_than_the_measured_round(tmp_path):
    """A measured round at ISL+OSL=2048 is CONC*5; the warmup must be well under."""
    cfg = _cfg(tmp_path, """
        benchmark:
          envs:
            CONC: 64
            ISL: 1024
            OSL: 1024
    """)
    measured_num_prompts = 64 * 5
    assert _warmup_num_prompts(cfg) < measured_num_prompts


@pytest.mark.parametrize("body", [
    # No concurrency key at all.
    "benchmark:\n  envs:\n    TP: 8\n",
    # Concurrency present but unusable.
    "benchmark:\n  envs:\n    CONC: 0\n",
    "benchmark:\n  envs:\n    CONC: 'not-a-number'\n",
    # No benchmark section.
    "something_else: 1\n",
    # Empty file.
    "",
])
def test_unreadable_concurrency_keeps_the_long_warmup(tmp_path, body):
    """Falling back must mean the warmup we already run, never a shorter one."""
    assert _warmup_num_prompts(_cfg(tmp_path, body)) is None


def test_missing_file_keeps_the_long_warmup(tmp_path):
    assert _warmup_num_prompts(tmp_path / "does-not-exist.yaml") is None


def test_short_warmup_is_on_by_default(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_EXPLORE_SHORT_WARMUP", raising=False)
    assert _short_warmup_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "FALSE", "Off"])
def test_short_warmup_can_be_turned_off(monkeypatch, val):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXPLORE_SHORT_WARMUP", val)
    assert _short_warmup_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_short_warmup_stays_on_for_truthy_values(monkeypatch, val):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXPLORE_SHORT_WARMUP", val)
    assert _short_warmup_enabled() is True


def test_warmup_never_shorter_than_one_full_batch(tmp_path):
    """Under-filling the batch would warm a different shape than we measure."""
    for conc in (1, 8, 64, 256, 1024):
        cfg = _cfg(tmp_path, f"benchmark:\n  envs:\n    CONC: {conc}\n")
        assert _warmup_num_prompts(cfg) >= conc
