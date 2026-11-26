"""Exit-code mapping for terminal stop_reasons."""

from __future__ import annotations

from hyperloom.inference_optimizer.cli import _exit_code_for_stop_reason


def test_cycle_end_completions_exit_zero():
    assert _exit_code_for_stop_reason("cycle_end_done", 1.0) == 0
    assert _exit_code_for_stop_reason("cycle_end_done", 1.0) == 0


def test_established_success_reasons_still_exit_zero():
    for reason in ("target_reached", "global_converged", "time_exhausted", "max_ticks"):
        assert _exit_code_for_stop_reason(reason, 1.0) == 0


def test_a_success_shaped_reason_with_no_baseline_measurement_exits_nonzero():
    for reason in ("time_exhausted", "max_ticks", "cycle_end_done"):
        assert _exit_code_for_stop_reason(reason, 0.0) == 1


def test_failure_reasons_exit_nonzero():
    for reason in (
        "prelude_baseline_failed",
        "enablement_attempts_exhausted",
        "",
        None,
    ):
        assert _exit_code_for_stop_reason(reason, 1.0) == 1
