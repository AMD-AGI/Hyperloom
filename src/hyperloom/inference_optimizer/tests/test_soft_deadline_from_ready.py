# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's wall-clock split into boot and benchmark, read from the ready stamp the round writes."""

from __future__ import annotations

import time

import pytest

from hyperloom.orchestrator.actions.executors.benchmark_run.records import _round_post_ready_sec
from inference_suite.contracts import ReadyStamp, RunLayout


class TestARoundIsPricedByItsTwoParts:
    """What a round spent booting and what it spent benchmarking, told apart."""

    def test_the_boot_is_not_charged_to_the_benchmark(self, tmp_path):
        """A round that boots for 3s and then benchmarks for 1s has the 3s priced as boot and the 1s as benchmark."""
        started_unix = time.time()
        RunLayout(tmp_path).write_ready_stamp(ReadyStamp(started_unix + 3.0, 3.0))

        assert _round_post_ready_sec(tmp_path, started_unix=started_unix, runtime_sec=4.0) == pytest.approx(1.0)

    def test_a_reader_on_another_clock_gets_the_same_boot(self, tmp_path):
        """The boot is a duration, so a reader whose clock runs behind the writer's prices it the same."""
        started_unix = time.time()
        RunLayout(tmp_path).write_ready_stamp(ReadyStamp(started_unix + 3.0, 3.0))

        agreed = _round_post_ready_sec(tmp_path, started_unix=started_unix, runtime_sec=4.0)
        skewed = _round_post_ready_sec(tmp_path, started_unix=started_unix - 5.0, runtime_sec=4.0)

        assert agreed is not None and skewed == pytest.approx(agreed)

    def test_a_round_that_never_came_up_is_not_priced(self, tmp_path):
        """No ready stamp means no split, reported as unknown rather than guessed."""
        assert _round_post_ready_sec(tmp_path, started_unix=0.0, runtime_sec=1.0) is None
        assert _round_post_ready_sec(None, started_unix=0.0, runtime_sec=1.0) is None

    def test_an_earlier_rounds_stamp_is_not_read_as_this_ones(self, tmp_path):
        """A stamp predating the round is unknown, not \"it never booted\"."""
        log_path = tmp_path / "server.log"
        log_path.write_text("INFO loading weights\n", encoding="utf-8")
        # A complete stamp, so what rejects it can only be its instant.
        (tmp_path / "server_ready_at").write_text(f"{time.time() - 600.0:.3f} 30.000\n", encoding="utf-8")

        assert (
            _round_post_ready_sec(
                log_path.parent,
                started_unix=time.time(),
                runtime_sec=90.0,
            )
            is None
        )

    def test_a_stamp_is_cleared_so_the_next_round_starts_blind(self, tmp_path):
        """Clearing is what keeps the case above from arising in a reused dir."""
        log_path = tmp_path / "server.log"
        stamp = tmp_path / "server_ready_at"
        stamp.write_text(f"{time.time():.3f} 30.000\n", encoding="utf-8")
        assert _round_post_ready_sec(log_path.parent, started_unix=0.0, runtime_sec=1.0) is not None

        RunLayout(log_path.parent).clear_ready_stamp()

        assert not stamp.exists()
        assert _round_post_ready_sec(log_path.parent, started_unix=0.0, runtime_sec=1.0) is None
        # Idempotent: a round whose dir never had one must not fail to start.
        RunLayout(log_path.parent).clear_ready_stamp()

    def test_a_boot_longer_than_the_round_cannot_produce_a_negative_price(self, tmp_path):
        """A corrupt stamp must cost a measurement, never produce a nonsense one."""
        log_path = tmp_path / "server.log"
        started_unix = time.time()
        (tmp_path / "server_ready_at").write_text(f"{started_unix + 10.0:.3f} 500.000\n", encoding="utf-8")

        priced = _round_post_ready_sec(
            log_path.parent,
            started_unix=started_unix,
            runtime_sec=100.0,
        )

        assert priced == 0.0

    def test_a_stamp_with_no_boot_recorded_is_no_stamp_at_all(self, tmp_path):
        """A missing boot must not read as a round that never booted."""
        log_path = tmp_path / "server.log"
        started_unix = time.time()
        (tmp_path / "server_ready_at").write_text(f"{started_unix + 10.0:.3f}\n", encoding="utf-8")

        assert (
            _round_post_ready_sec(
                log_path.parent,
                started_unix=started_unix,
                runtime_sec=100.0,
            )
            is None
        )
