# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cycle bench's arithmetic: its launch, its log reading, its weights and its verdicts."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.measurement import cycle_bench as cb

_ARGV = [
    "python3", "-m", "sglang.launch_server", "--model-path", "/m", "--port", "8888",
    "--enable-hierarchical-cache", "--hicache-size", "180", "--hicache-write-policy", "write_through",
    "--tensor-parallel-size", "4",
]  # fmt: skip


def _decode(bs: int, accept: float, tput: float, queue: int = 0) -> str:
    return (
        f"[2026-10-26 10:00:00 TP0] Decode batch, #running-req: {bs}, #token: 300000, token usage: 0.10, "
        f"accept len: {accept:.2f}, cuda graph: True, gen throughput (token/s): {tput:.2f}, #queue-req: {queue}"
    )


def test_the_bench_server_drops_hicache_and_takes_its_own_port():
    assert cb.bench_argv(_ARGV, port=8891) == [
        "python3", "-m", "sglang.launch_server", "--model-path", "/m", "--port", "8891", "--tensor-parallel-size", "4",
    ]  # fmt: skip


def test_simulated_acceptance_is_pinned_to_a_fixed_token_stream():
    env = {"SGLANG_SIMULATE_ACC_LEN": "3.61", "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token", "X": "1"}

    assert cb.bench_env(env) == {
        "SGLANG_SIMULATE_ACC_LEN": "3.61",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed",
        "X": "1",
    }
    assert cb.bench_env({"X": "1"}) == {"X": "1"}


def test_a_decode_line_gives_one_cycle_at_its_batch_size_only():
    lines = [
        _decode(2, 3.6, 360.0),
        _decode(2, 3.6, 360.0, queue=1),
        _decode(4, 3.6, 600.0),
        "unrelated line",
        _decode(2, 3.0, 300.0),
    ]

    assert cb.decode_cycles_ms(lines, bs=2) == [pytest.approx(20.0), pytest.approx(20.0)]


def test_batch_weights_follow_where_the_replay_ran():
    lines = [_decode(1, 3.6, 200.0)] * 2 + [_decode(2, 3.6, 300.0)] * 5 + [_decode(3, 3.6, 300.0)] * 2
    lines += [_decode(9, 3.6, 300.0)]

    assert cb.batch_weights(lines, grid=(1, 2, 4)) == {1: 0.2, 2: 0.5, 4: 0.3}


def test_the_weighted_gain_is_the_share_of_cycle_the_candidate_saves():
    weights = {1: 0.5, 2: 0.5}
    base = {1: 10.0, 2: 20.0}
    cand = {1: 9.0, 2: 20.0}

    assert cb.weighted_gain_pct(base, cand, weights) == pytest.approx(5.0)


def test_a_batch_is_valid_only_when_its_two_bases_agree_within_the_band():
    weights = {1: 0.5, 2: 0.5}

    assert cb.batch_valid({1: 10.0, 2: 20.0}, {1: 10.1, 2: 20.1}, weights, band_pct=1.0)
    assert not cb.batch_valid({1: 10.0, 2: 20.0}, {1: 10.4, 2: 20.6}, weights, band_pct=1.0)


@pytest.mark.parametrize(
    ("gain", "verdict"),
    [(0.4, cb.VERDICT_PASS), (0.0, cb.VERDICT_NEUTRAL), (-1.5, cb.VERDICT_NEUTRAL), (-2.5, cb.VERDICT_REGRESS)],
)
def test_any_positive_estimate_enters_the_pool_and_only_a_slowdown_past_the_band_is_rejected(gain, verdict):
    assert cb.screen_verdict(gain, band_pct=2.0) == verdict
