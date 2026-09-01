# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the small env/server-args helpers in ``roofline_ceiling``.

These are pure dict/attr readers used to resolve runtime server args and
benchmark geometry; the larger ceiling tests do not cover them directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import roofline_ceiling as rc


def test_benchmark_envs():
    assert rc._benchmark_envs({}) == {}
    assert rc._benchmark_envs({"envs": "nope"}) == {}
    assert rc._benchmark_envs({"envs": {"A": 1}}) == {"A": 1}


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 0),
        ("8", 8),
        (8, 8),
        (0, 0),
        (-3, 0),
        ("bad", 0),
    ],
)
def test_env_int(raw, expected):
    assert rc._env_int({"K": raw} if raw is not None else {}, "K") == expected


def test_server_args_from_envs():
    assert rc._server_args_from_envs({}) == ""
    envs = {
        "EXTRA_SGLANG_ARGS": " --tp 8 ",
        "EXTRA_VLLM_ARGS": "",
        "EXTRA_ATOM_ARGS": "--foo",
        "IGNORED": "--bar",
    }
    out = rc._server_args_from_envs(envs)
    assert out == "--tp 8 --foo"


def test_server_args_env_override_and_payload():
    assert rc._server_args_env_override("not-a-dict") == ""
    # extra_envs present but not a dict → "".
    assert rc._server_args_env_override({"extra_envs": "nope"}) == ""
    assert rc._server_args_env_override({"extra_envs": {"EXTRA_VLLM_ARGS": "--x"}}) == "--x"

    assert rc._server_args_payload("nope") == ""
    assert rc._server_args_payload({"extra_args": " --z "}) == "--z"
    assert (
        rc._server_args_payload(
            {"candidate_extra_server_args": "--first", "extra_server_args": "--second"},
        )
        == "--first"
    )

    # env override wins over payload.
    entry = {"extra_envs": {"EXTRA_ATOM_ARGS": "--env"}, "extra_args": "--payload"}
    assert rc._server_args_from(entry) == "--env"
    assert rc._server_args_from({"extra_args": "--payload"}) == "--payload"


def test_read_baseline_yaml_benchmark(tmp_path: Path):
    # Non-dict / missing materialized_config short-circuits to {}.
    assert rc._read_baseline_yaml_benchmark(SimpleNamespace(last_baseline=None)) == {}
    # last_baseline is a non-dict truthy value → {}.
    assert rc._read_baseline_yaml_benchmark(SimpleNamespace(last_baseline=[1])) == {}
    assert (
        rc._read_baseline_yaml_benchmark(
            SimpleNamespace(last_baseline={"extras": {}}),
        )
        == {}
    )

    cfg = tmp_path / "materialized.yaml"
    cfg.write_text("benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: '--tp 8'\n", encoding="utf-8")
    state = SimpleNamespace(
        last_baseline={"extras": {"materialized_config": str(cfg)}},
    )
    bench = rc._read_baseline_yaml_benchmark(state)
    assert bench.get("envs", {}).get("EXTRA_SGLANG_ARGS") == "--tp 8"

    # Full chain: baseline yaml -> server args string.
    assert rc._read_baseline_yaml_server_args(state) == "--tp 8"


def test_resolve_compute_peak_provenance():
    # Achievable-table hit wins over the vendor dense peak.
    achievable = rc.resolve_compute_peak_provenance("mi300x", "bf16")
    assert achievable == {
        "compute_peak_convention": "achievable",
        "compute_peak_tflops": rc._resolve_achievable_tflops("mi300x", "bf16"),
        "compute_peak_source": "TraceLens arch JSON (max-achievable sustained)",
    }
    assert achievable["compute_peak_tflops"] > 0

    # mi308x has no HW_SPECS_ACHIEVABLE entry -> falls back to the vendor
    # dense peak (mi308x shares mi300x's peak_tflops table).
    vendor = rc.resolve_compute_peak_provenance("mi308x", "bf16")
    assert rc._resolve_achievable_tflops("mi308x", "bf16") == 0.0
    assert vendor == {
        "compute_peak_convention": "vendor",
        "compute_peak_tflops": rc._resolve_peak_tflops("mi308x", "bf16"),
        "compute_peak_source": "vendor dense peak (achievable-table miss fallback)",
    }
    assert vendor["compute_peak_tflops"] > 0

    # Unknown gpu/precision misses both tables -> unknown/0.0/"unavailable".
    unknown = rc.resolve_compute_peak_provenance("h100", "bf16")
    assert unknown == {
        "compute_peak_convention": "unknown",
        "compute_peak_tflops": 0.0,
        "compute_peak_source": "unavailable",
    }
