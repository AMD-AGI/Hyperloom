# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the two things measured from the server-ready marker."""

from __future__ import annotations

import sys
import time

import pytest

from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    DETOKENIZER_STALL_RETURNCODE,
    RoundWatch,
    run_with_session_kill,
)
from hyperloom.orchestrator.actions.executors.benchmark_run.records import _round_post_ready_sec
from inference_suite.contracts import RunLayout


def test_from_ready_excludes_pre_ready_phase(tmp_path):
    """Pre-ready time is NOT counted: a child that spends > deadline BEFORE the ready marker but only a little AFTER it finishes normally."""
    log_path = tmp_path / "server.log"
    # 3s pre-ready boot, then ready, then ~1s post-ready client.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('INFO loading weights\\n'); f.flush()\n"
        "time.sleep(3)\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "time.sleep(1)\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        watch=RoundWatch(timeout=30, server_log_path=str(log_path), silence_timeout_sec=2.0),
    )
    elapsed = time.monotonic() - start
    # Post-ready (~1s) < 2.0s deadline -> not killed despite ~4s wall-clock.
    assert cp.returncode == 0, f"unexpected returncode={cp.returncode}"
    assert elapsed >= 3.0, f"child should have run its full pre-ready sleep, got {elapsed:.2f}s"


def test_from_ready_fires_after_ready(tmp_path):
    """Post-ready overrun IS killed: once ready, exceeding the deadline in the client phase reaps the tree with the silence sentinel."""
    log_path = tmp_path / "server.log"
    # Ready immediately, then a post-ready run that overruns the deadline.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "time.sleep(30)\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        watch=RoundWatch(timeout=60, server_log_path=str(log_path), silence_timeout_sec=1.0),
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == DETOKENIZER_STALL_RETURNCODE
    # Killed shortly after ready, not at 30s.
    assert elapsed < 15.0, f"from-ready silence deadline took {elapsed:.2f}s"


def test_scriptable_silence_is_disarmed():
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(.3)"],
        watch=RoundWatch(timeout=5, silence_timeout_sec=0.1),
    )
    assert cp.returncode == 0


class TestARoundIsPricedByItsTwoParts:
    """What a round spent booting and what it spent benchmarking, told apart."""

    def test_the_boot_is_not_charged_to_the_benchmark(self, tmp_path):
        """A round that boots for 3s and then benchmarks for 1s has the 3s priced as boot and the 1s as benchmark."""
        log_path = tmp_path / "server.log"
        # The benchmark's second starts once the watchdog has stamped the ready marker, so it lands after the stamp
        # however late a loaded host runs the watchdog's poll.
        script = (
            "import os, sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(3)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "deadline = time.monotonic() + 20\n"
            "while not os.path.exists(sys.argv[2]) and time.monotonic() < deadline:\n"
            "    time.sleep(0.05)\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        started_unix = time.time()
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path), str(tmp_path / "server_ready_at")],
            watch=RoundWatch(timeout=30, server_log_path=str(log_path)),
        )
        runtime_sec = time.time() - started_unix
        assert cp.returncode == 0

        post_ready = _round_post_ready_sec(
            log_path.parent,
            started_unix=started_unix,
            runtime_sec=runtime_sec,
        )
        assert post_ready is not None, "the round reported ready but nothing recorded when"
        # Floors only. A ceiling on either share would time the host -- when the watchdog's poll ran, and how long the
        # round took to wind down after its child exited -- rather than the split.
        boot_sec = runtime_sec - post_ready
        assert boot_sec >= 3.0, f"boot share read as {boot_sec:.2f}s for a round that slept 3s before it was ready"
        # Less the millisecond the stamp rounds the boot to.
        assert post_ready >= 0.99, f"benchmark share read as {post_ready:.2f}s for a round that ran 1s after its stamp"

    def test_a_previous_attempts_log_does_not_time_this_rounds_boot(self, tmp_path):
        """Only bytes this round writes may say when its server came up."""
        stale = tmp_path / "benchmark_vllm_20200101" / "server.log"
        stale.parent.mkdir(parents=True)
        stale.write_text("Application startup complete\n", encoding="utf-8")

        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(3)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        started_unix = time.time()
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            watch=RoundWatch(timeout=30, server_log_path=str(log_path)),
        )
        runtime_sec = time.time() - started_unix
        assert cp.returncode == 0

        post_ready = _round_post_ready_sec(
            log_path.parent,
            started_unix=started_unix,
            runtime_sec=runtime_sec,
        )
        assert post_ready is not None
        boot_sec = runtime_sec - post_ready
        assert boot_sec >= 2.5, (
            f"the stale log timed the boot: read {boot_sec:.2f}s of boot for a round that spent 3s on it"
        )

    def test_the_split_does_not_depend_on_an_unrelated_watchdog(self, tmp_path):
        """The stall grace is a hang backstop, not a switch for the cost model."""
        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(2)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            watch=RoundWatch(timeout=30, server_log_path=str(log_path), session_deadline_sec=time.monotonic() + 30.0),
        )
        assert cp.returncode == 0
        assert _round_post_ready_sec(log_path.parent, started_unix=0.0, runtime_sec=1.0) is not None, (
            "the boot/benchmark split was withdrawn along with the stall watchdog"
        )

    def test_a_reader_on_another_clock_gets_the_same_boot(self, tmp_path):
        """The split survives being read where it was not written."""
        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(3)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        started_unix = time.time()
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            watch=RoundWatch(timeout=30, server_log_path=str(log_path)),
        )
        runtime_sec = time.time() - started_unix
        assert cp.returncode == 0

        agreed = _round_post_ready_sec(
            log_path.parent,
            started_unix=started_unix,
            runtime_sec=runtime_sec,
        )
        # A caller whose clock runs five seconds behind the writer's.
        skewed = _round_post_ready_sec(
            log_path.parent,
            started_unix=started_unix - 5.0,
            runtime_sec=runtime_sec,
        )
        assert agreed is not None
        assert skewed == pytest.approx(agreed), f"a five-second clock disagreement moved the boot: {skewed} vs {agreed}"

    def test_a_round_that_never_came_up_is_not_priced(self, tmp_path):
        """No ready marker means no split, reported as unknown rather than guessed."""
        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(1)\n"
        )
        started_unix = time.time()
        run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            watch=RoundWatch(timeout=30, server_log_path=str(log_path)),
        )
        assert _round_post_ready_sec(log_path.parent, started_unix=0.0, runtime_sec=1.0) is None
        assert (
            _round_post_ready_sec(
                log_path.parent,
                started_unix=started_unix,
                runtime_sec=time.time() - started_unix,
            )
            is None
        )

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
