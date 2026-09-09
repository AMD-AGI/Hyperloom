"""The GEMM e2e validator must not do its file I/O on the event loop."""

from __future__ import annotations

import inspect
import re

from hyperloom.orchestrator.phases import kernel as kernel_mod


def _validator_source() -> str:
    return inspect.getsource(kernel_mod.KernelPhase._validate_gemm_tuning_e2e)


def test_the_validator_is_still_a_coroutine():
    # The whole premise: if it ever becomes sync, it is not on the loop and these assertions are measuring nothing.
    assert inspect.iscoroutinefunction(kernel_mod.KernelPhase._validate_gemm_tuning_e2e)


def test_the_fused_moe_probe_runs_off_the_loop():
    src = _validator_source()
    assert "self._runtime_uses_aiter_fused_moe" in src
    assert re.search(
        r"await\s+asyncio\.to_thread\(\s*\n?\s*self\._runtime_uses_aiter_fused_moe",
        src,
    ), src


def test_the_coverage_replay_runs_off_the_loop():
    src = _validator_source()
    assert "self._gemm_tuned_config_coverage" in src
    assert re.search(
        r"await\s+asyncio\.to_thread\(\s*self\._gemm_tuned_config_coverage\s*,",
        src,
    ), src


def test_neither_is_also_called_bare():
    """A ``to_thread`` wrap elsewhere does not excuse a second inline call."""
    src = _validator_source()
    for name in ("_runtime_uses_aiter_fused_moe", "_gemm_tuned_config_coverage"):
        # A bare call is the attribute immediately followed by "(" -- the to_thread form passes the bound method as an
        # argument instead, so it is followed by "," or ")".
        assert not re.search(rf"self\.{name}\s*\(", src), f"{name} is called inline"
