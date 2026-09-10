# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX handoff must carry the shape GEAK should optimize for.

A trace replay has no single ISL -- the corpus spans roughly 89k at p50 past
500k at p99 -- so ``workload.isl/osl`` stay at the CLI's synthetic 1024/1024.
GEAK does not MEASURE with them (the aiperf client replays the corpus and
ignores them), but its kernel agents read isl/osl as the analytic serving call
model when they synthesize GEMM/attention shapes. Left at 1024 the whole search
targets a regime two orders of magnitude below the served load.

So the handoff carries the average shape THIS run measured on its own baseline,
derived from the canonical result rather than hardcoded corpus percentiles.
"""

import json

from hyperloom.orchestrator.phases.kernel import KernelPhase


def _result(dirpath, *, tput, completed, tin, tout):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "inferencex_result.json").write_text(
        json.dumps(
            {
                "output_throughput": tput,
                "completed": completed,
                "total_input_tokens": tin,
                "total_output_tokens": tout,
            }
        ),
        encoding="utf-8",
    )


def _coord(tmp_path):
    phase = KernelPhase.__new__(KernelPhase)
    phase.session_dir = tmp_path
    return phase


def test_shape_is_derived_from_the_measured_baseline(tmp_path):
    """Campaign numbers: 86,590,710 input tokens over 773 requests => ~112k."""
    _result(
        tmp_path / "runs" / "baseline" / "abc" / "bench",
        tput=168.99,
        completed=773,
        tin=86_590_710,
        tout=615_152,
    )
    got = _coord(tmp_path)._observed_replay_shape(168.99)
    assert got["observed_isl"] == 112_019
    assert got["observed_osl"] == 796
    assert got["observed_requests"] == 773
    assert got["observed_tput_match_err"] == 0.0


def test_the_result_matching_the_handed_over_baseline_wins(tmp_path):
    """Several benchmarks exist; the shape must describe the SAME measurement."""
    _result(
        tmp_path / "runs" / "explore" / "a" / "bench",
        tput=300.0,
        completed=1000,
        tin=1_000_000,
        tout=500_000,
    )
    _result(
        tmp_path / "runs" / "baseline" / "b" / "bench",
        tput=169.0,
        completed=773,
        tin=86_590_710,
        tout=615_152,
    )
    got = _coord(tmp_path)._observed_replay_shape(169.0)
    assert got["observed_isl"] == 112_019


def test_geak_and_overlay_results_are_never_the_orchestrator_baseline(tmp_path):
    """GEAK's own numbers would make the handoff describe itself."""
    _result(
        tmp_path / "runs" / "geak" / "e2e_cycle0" / "bench",
        tput=169.0,
        completed=10,
        tin=10_240,
        tout=10_240,
    )
    _result(
        tmp_path / "runs" / "baseline" / "_baseline_source_overlay" / "bench",
        tput=169.0,
        completed=10,
        tin=10_240,
        tout=10_240,
    )
    assert _coord(tmp_path)._observed_replay_shape(169.0) == {}


def test_an_ancestor_named_like_geak_does_not_exclude_a_real_baseline(tmp_path):
    """The exclusions match path COMPONENTS, not the path as one string.

    A substring test throws away every result as soon as some directory ABOVE
    the session contains "geak". That is not hypothetical: the campaign this
    method was written for ran out of ``.../hlgeak_24h_run/sessions/...``, where
    a substring check discards the very baseline it is meant to read and the
    handoff silently falls back to the synthetic 1024 shape.
    """
    session = tmp_path / "hlgeak_24h_run" / "sessions" / "Kimi-K3" / "s1"
    _result(
        session / "runs" / "baseline" / "abc" / "bench",
        tput=168.99,
        completed=773,
        tin=86_590_710,
        tout=615_152,
    )
    phase = KernelPhase.__new__(KernelPhase)
    phase.session_dir = session
    got = phase._observed_replay_shape(168.99)
    assert got["observed_isl"] == 112_019
    assert got["observed_requests"] == 773


def test_absent_results_degrade_to_no_claim(tmp_path):
    assert _coord(tmp_path)._observed_replay_shape(169.0) == {}


def test_a_zero_request_result_is_not_a_shape(tmp_path):
    _result(
        tmp_path / "runs" / "baseline" / "a" / "bench",
        tput=169.0,
        completed=0,
        tin=0,
        tout=0,
    )
    assert _coord(tmp_path)._observed_replay_shape(169.0) == {}


def test_malformed_result_is_skipped_without_raising(tmp_path):
    bench = tmp_path / "runs" / "baseline" / "a" / "bench"
    bench.mkdir(parents=True)
    (bench / "inferencex_result.json").write_text("{not json", encoding="utf-8")
    assert _coord(tmp_path)._observed_replay_shape(169.0) == {}


def test_without_a_baseline_tput_the_best_available_replay_is_used(tmp_path):
    """A resumed run may not have a baseline number to match against."""
    _result(
        tmp_path / "runs" / "baseline" / "a" / "bench",
        tput=169.0,
        completed=773,
        tin=86_590_710,
        tout=615_152,
    )
    got = _coord(tmp_path)._observed_replay_shape(0.0)
    assert got["observed_isl"] == 112_019
    # Flagged as unconfirmed rather than presented as a matched measurement.
    assert got["observed_tput_match_err"] == 1.0


def test_the_repro_run_agrees_with_the_campaign_within_a_fraction_of_a_percent(tmp_path):
    """Independent confirmation that the derivation is stable, not incidental."""
    _result(
        tmp_path / "runs" / "baseline" / "a" / "bench",
        tput=166.71,
        completed=760,
        tin=85_385_881,
        tout=604_881,
    )
    got = _coord(tmp_path)._observed_replay_shape(166.71)
    assert got["observed_isl"] == 112_350
    assert abs(got["observed_isl"] - 112_019) / 112_019 < 0.005
