"""D1 — log-pattern extension tests.

The patterns themselves live in ``local_probe._DEFAULT_LOG_ERROR_PATTERNS``
and ``local_health._HIGH_SEVERITY_PATTERNS``. These tests verify each new
pattern is detected and routed to the right severity tier.
"""

from __future__ import annotations

import pytest

from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals import SymptomSeverity
from hyperloom.agents.robustness.signals.local_health import (
    evaluate_local_health_signals,
)
from hyperloom.agents.robustness.sources.base import SourceData
from hyperloom.agents.robustness.sources.local_probe import (
    _DEFAULT_LOG_ERROR_PATTERNS,
    _extract_log_errors,
)


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(),
        inbox=[],
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# D1 — new patterns get matched in raw log lines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        ("Engine core EngineCore-1 died unexpectedly",
         r"Engine core .* died"),
        ("RuntimeError: Engine core initialization failed",
         r"RuntimeError: Engine core initialization failed"),
        ("OSError: [Errno 98] Address already in use",
         r"Address already in use"),
        ("sglang tokenizer worker tw-3 died (signal 9)",
         r"tokenizer worker .* died"),
        ("MLA-style attention not supported in this checkpoint",
         r"MLA.*not supported"),
        ("MTP draft model unavailable for spec decoding",
         r"MTP draft .* unavailable"),
        ("aiter rms_norm compilation failed: nvcc returned exit 1",
         r"aiter .* compilation failed"),
        ("hipcc died with signal 9 (SIGKILL)",
         r"hipcc .* signal"),
        ("accuracy MMLU gate failed; reverting integrate",
         r"accuracy .* gate failed"),
        ("Eval result: MMLU 67.3% below threshold (74%)",
         r"MMLU .* below threshold"),
        ("ROCblas internal error: rocblasStatus 2",
         r"ROCblas.*Status\s*\d+"),
        ("hipBLAS Error: handle is in invalid state",
         r"hipBLAS.*Error"),
        ("NCCL WARN [Worker 3] timeout after 600 seconds",
         r"NCCL WARN .* timeout"),
        ("Failed to load checkpoint /weights/dsr1/safetensors",
         r"Failed to load checkpoint"),
        ("runtime.cli prepare-review timed out after 30s",
         r"runtime\.cli .* timed out after \d+s"),
        ("cudaErrorOutOfDevice while allocating KV cache",
         r"cudaErrorOutOfDevice"),
        ("HSA_STATUS_ERROR_OUT_OF_RESOURCES at hipDeviceAlloc",
         r"HSA_STATUS_ERROR_OUT_OF_RESOURCES"),
    ],
)
def test_d1_new_pattern_matches(line, expected_pattern):
    hits = _extract_log_errors([line], _DEFAULT_LOG_ERROR_PATTERNS, window=10)
    matched_patterns = {h["pattern"] for h in hits}
    assert expected_pattern in matched_patterns


# ---------------------------------------------------------------------------
# Severity routing — D1 high-severity patterns produce HIGH symptoms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        # Existing high-severity patterns still work.
        ("CUDA out of memory at allocator.cc:42", "CUDA out of memory"),
        # D1 new HIGH patterns.
        ("Engine core EngineCore-1 died unexpectedly",
         r"Engine core .* died"),
        ("RuntimeError: Engine core initialization failed",
         r"RuntimeError: Engine core initialization failed"),
        ("aiter fused_moe compilation failed: hipcc exit 1",
         r"aiter .* compilation failed"),
        ("Failed to load checkpoint /weights/dsr1",
         r"Failed to load checkpoint"),
        ("runtime.cli commit-review timed out after 30s",
         r"runtime\.cli .* timed out after \d+s"),
    ],
)
def test_d1_high_severity_pattern_emits_high(line, expected_pattern):
    data = SourceData(
        local_log_errors=[{"pattern": expected_pattern, "line": line}],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "log_error_pattern")
    assert sym.severity is SymptomSeverity.HIGH


@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        # D1 medium-severity new patterns.
        ("OSError: Address already in use", r"Address already in use"),
        ("tokenizer worker tw-2 died (signal 11)",
         r"tokenizer worker .* died"),
        ("NCCL WARN [Worker 0] timeout after 600s",
         r"NCCL WARN .* timeout"),
    ],
)
def test_d1_medium_severity_pattern_emits_medium(line, expected_pattern):
    data = SourceData(
        local_log_errors=[{"pattern": expected_pattern, "line": line}],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "log_error_pattern")
    assert sym.severity is SymptomSeverity.MEDIUM
