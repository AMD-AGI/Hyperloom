"""Exit-code mapping for terminal stop_reasons.

Regression: a clean no-kernel run closes with stop_reason ``sweep_done``
(and a shape-grid run with ``sweep_done``). Both mean the optimizer ran and
closed normally, yet the CLI used to return exit code 1 for them, so CI (with
backoffLimit: 0) flagged a successful run as failed.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.cli import _exit_code_for_stop_reason


def test_sweep_completions_exit_zero():
    # The bug: these clean SWEEP terminals were mapped to 1.
    assert _exit_code_for_stop_reason("sweep_done") == 0
    assert _exit_code_for_stop_reason("sweep_done") == 0


def test_established_success_reasons_still_exit_zero():
    for reason in ("target_reached", "global_converged", "time_exhausted", "max_ticks"):
        assert _exit_code_for_stop_reason(reason) == 0


def test_failure_reasons_exit_nonzero():
    for reason in (
        "prelude_baseline_failed",
        "enablement_stalled",
        "sweep_failed",
        "crash_threshold_exceeded",
        "",
        None,
    ):
        assert _exit_code_for_stop_reason(reason) == 1
