# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for framework_agent.enablement (failure classifier + request model).

Pure-Python, GPU-free: every case is a canned log string in → structured
:class:`FailureSignature` out.
"""

from __future__ import annotations

import pytest

from framework_agent.enablement import (
    CAPABILITY_DISABLED,
    HIP_KERNEL_MISSING,
    IMPORT_ERROR,
    MISSING_MODEL_ARCH,
    NOT_IMPLEMENTED,
    SHAPE_MISMATCH,
    UNKNOWN,
    UNSUPPORTED_DTYPE,
    EnablementRequest,
    classify_failure,
    runnable_decision,
)


# --- classify_failure: kind detection --------------------------------------


def test_missing_model_arch() -> None:
    """Unsupported architecture message -> missing_model_arch + arch symbol."""
    log = "ValueError: Model architecture 'Glm5ForCausalLM' is not supported for now."
    sig = classify_failure(log)
    assert sig.kind == MISSING_MODEL_ARCH
    assert sig.offending_symbol == "Glm5ForCausalLM"
    assert sig.bridge_layer == "framework"
    assert sig.confidence > 0.9
    assert sig.is_actionable


def test_hip_kernel_missing_no_binary() -> None:
    """hipErrorNoBinaryForGpu -> hip_kernel_missing at the rocm_hip layer."""
    sig = classify_failure("RuntimeError: hipErrorNoBinaryForGpu: no kernel image is available")
    assert sig.kind == HIP_KERNEL_MISSING
    assert sig.bridge_layer == "rocm_hip"


def test_hip_undefined_symbol_captured() -> None:
    """undefined symbol is classified as a HIP kernel gap and the symbol captured."""
    sig = classify_failure("ImportError: /x/_C.so: undefined symbol: _ZN4aiter8fmha_fwdEv")
    assert sig.kind == HIP_KERNEL_MISSING
    assert sig.offending_symbol == "_ZN4aiter8fmha_fwdEv"


def test_unsupported_dtype() -> None:
    """fp8 unsupported -> unsupported_dtype."""
    sig = classify_failure('RuntimeError: "addmm" not implemented for \'Float8_e4m3fn\'')
    assert sig.kind == UNSUPPORTED_DTYPE
    assert sig.offending_symbol == "Float8_e4m3fn"


def test_shape_mismatch() -> None:
    """A tensor shape error -> shape_mismatch."""
    sig = classify_failure("RuntimeError: shape '[2, 4096]' is invalid for input of size 4096")
    assert sig.kind == SHAPE_MISMATCH


def test_not_implemented() -> None:
    """NotImplementedError -> not_implemented with the trailing message."""
    sig = classify_failure("NotImplementedError: sliding window attention on ROCm")
    assert sig.kind == NOT_IMPLEMENTED
    assert "sliding window" in sig.offending_symbol


def test_capability_disabled_supported_predicate() -> None:
    """A *_supported() predicate returning False -> capability_disabled."""
    sig = classify_failure("INFO: fp8_mfma_supported() returned False; using reference path")
    assert sig.kind == CAPABILITY_DISABLED
    assert sig.offending_symbol == "fp8_mfma_supported"


def test_import_error_module_not_found() -> None:
    """ModuleNotFoundError -> import_error at the build layer with module name."""
    sig = classify_failure("ModuleNotFoundError: No module named 'aiter.ops'")
    assert sig.kind == IMPORT_ERROR
    assert sig.offending_symbol == "aiter.ops"
    assert sig.bridge_layer == "build"


def test_unknown_on_unrecognized() -> None:
    """Text with no known signature -> unknown, not actionable, zero confidence."""
    sig = classify_failure("everything is fine, server ready on port 30000")
    assert sig.kind == UNKNOWN
    assert not sig.is_actionable
    assert sig.confidence == 0.0


def test_empty_log_is_unknown() -> None:
    """Blank input -> unknown without raising."""
    assert classify_failure("").kind == UNKNOWN
    assert classify_failure("   \n  ").kind == UNKNOWN


# --- offending-file extraction ---------------------------------------------


def test_offending_file_from_last_traceback_frame() -> None:
    """The last traceback frame (closest to raise) wins over earlier frames."""
    log = (
        'Traceback (most recent call last):\n'
        '  File "/opt/vllm/entrypoint.py", line 10, in main\n'
        '  File "/opt/vllm/model_executor/registry.py", line 88, in resolve\n'
        "ValueError: Model architecture 'FooForCausalLM' is not supported"
    )
    sig = classify_failure(log)
    assert sig.kind == MISSING_MODEL_ARCH
    assert sig.offending_file == "/opt/vllm/model_executor/registry.py"


def test_offending_file_from_inline_cpp_path() -> None:
    """Compiler-style inline path is picked up when no traceback frame exists."""
    sig = classify_failure("/opt/rocm/aiter/csrc/fmha.hip:210:5: error: no kernel image is available")
    assert sig.kind == HIP_KERNEL_MISSING
    assert sig.offending_file.endswith("fmha.hip")


# --- rule ordering ---------------------------------------------------------


def test_hip_symbol_beats_import_error_ordering() -> None:
    """An ImportError caused by an undefined HIP symbol resolves to the more
    actionable hip_kernel_missing, not the generic import_error."""
    log = "ImportError: /lib/_C.so: undefined symbol: hipLaunchKernel"
    sig = classify_failure(log)
    assert sig.kind == HIP_KERNEL_MISSING


# --- multi-signature (secondary_kinds) ------------------------------------


def test_stacked_import_error_masking_hip_symbol() -> None:
    """A stacked traceback (ImportError wrapping an undefined HIP symbol) keeps
    the actionable hip_kernel_missing as primary and surfaces import_error as a
    secondary kind rather than discarding it."""
    log = (
        "Traceback (most recent call last):\n"
        '  File "/opt/vllm/_custom_ops.py", line 5, in <module>\n'
        "ImportError: cannot import name '_C' from partially initialized module\n"
        "The above exception was the direct cause of the following:\n"
        "ImportError: /opt/vllm/_C.so: undefined symbol: _ZN4aiter8fmha_fwdEv"
    )
    sig = classify_failure(log)
    assert sig.kind == HIP_KERNEL_MISSING
    assert IMPORT_ERROR in sig.secondary_kinds
    assert HIP_KERNEL_MISSING not in sig.secondary_kinds
    # Corroborating rules nudge confidence above the bare rule constant.
    assert sig.confidence > 0.85


def test_single_signature_has_empty_secondary() -> None:
    """A log matching exactly one rule reports no secondary kinds."""
    sig = classify_failure("ModuleNotFoundError: No module named 'aiter.ops'")
    assert sig.kind == IMPORT_ERROR
    assert sig.secondary_kinds == ()


def test_offending_file_prefers_frame_near_primary_hit() -> None:
    """With two matching rules, the offending file is taken near the primary
    (earlier, more-specific) hit rather than the last frame overall."""
    log = (
        'Traceback (most recent call last):\n'
        '  File "/opt/vllm/loader.py", line 3, in load\n'
        "RuntimeError: hipErrorNoBinaryForGpu: no kernel image is available\n"
        '  File "/opt/vllm/fallback.py", line 9, in retry\n'
        "ImportError: cannot import name '_C'"
    )
    sig = classify_failure(log)
    assert sig.kind == HIP_KERNEL_MISSING
    assert sig.offending_file == "/opt/vllm/loader.py"


# --- EnablementRequest -----------------------------------------------------


def test_request_from_dict_minimal() -> None:
    """Minimal payload parses and exposes a lazily-classified signature."""
    req = EnablementRequest.from_dict(
        {
            "framework": "SGLang",
            "model": "zai-org/GLM-5",
            "repo_url": "https://github.com/sgl-project/sglang.git",
            "launch_log": "ValueError: Model architecture 'Glm5ForCausalLM' is not supported",
        }
    )
    assert req.framework == "sglang"  # normalized lower
    assert req.signature.kind == MISSING_MODEL_ARCH


@pytest.mark.parametrize("missing", ["framework", "model", "repo_url"])
def test_request_requires_core_fields(missing: str) -> None:
    """Each of framework/model/repo_url is mandatory."""
    payload = {
        "framework": "sglang",
        "model": "m",
        "repo_url": "https://x/y.git",
    }
    del payload[missing]
    with pytest.raises(ValueError, match=missing):
        EnablementRequest.from_dict(payload)


# --- runnable_decision (the enablement gate) -------------------------------


def test_runnable_pass_probe_only() -> None:
    """Probe exit 0 with no correctness check -> runs."""
    runs, reason = runnable_decision(probe_returncode=0, correctness_ok=None)
    assert runs is True
    assert "launches" in reason


def test_runnable_pass_with_correctness() -> None:
    """Probe exit 0 + correctness pass -> runs, reason mentions correctness."""
    runs, reason = runnable_decision(probe_returncode=0, correctness_ok=True)
    assert runs is True
    assert "correctness" in reason


def test_runnable_fail_nonzero_probe() -> None:
    """Non-zero probe exit -> still not runnable."""
    runs, reason = runnable_decision(probe_returncode=1, correctness_ok=None)
    assert runs is False
    assert "still not runnable" in reason


def test_runnable_fail_timeout() -> None:
    """Probe timeout is a hard fail regardless of return code."""
    runs, reason = runnable_decision(probe_returncode=0, correctness_ok=True, probe_timed_out=True)
    assert runs is False
    assert "timed out" in reason


def test_runnable_fail_probe_not_run() -> None:
    """A missing probe result cannot be promoted."""
    runs, reason = runnable_decision(probe_returncode=None, correctness_ok=None)
    assert runs is False
    assert "did not run" in reason


def test_runnable_fail_correctness() -> None:
    """Boots but fails correctness -> rejected."""
    runs, reason = runnable_decision(probe_returncode=0, correctness_ok=False)
    assert runs is False
    assert "correctness check failed" in reason


def test_runnable_fail_same_signature_persists() -> None:
    """Same actionable failure after the patch -> not fixed even if probe rc==0."""
    before = classify_failure("ValueError: Model architecture 'FooForCausalLM' is not supported")
    after = classify_failure("ValueError: Model architecture 'FooForCausalLM' is not supported")
    runs, reason = runnable_decision(
        probe_returncode=0,
        correctness_ok=None,
        before_signature=before,
        after_signature=after,
    )
    assert runs is False
    assert "persists after patch" in reason


def test_runnable_pass_when_post_signature_clean() -> None:
    """A clean (unknown) post-patch signature does not block a rc==0 probe."""
    before = classify_failure("ValueError: Model architecture 'FooForCausalLM' is not supported")
    after = classify_failure("server ready on port 30000")
    assert after.kind == UNKNOWN
    runs, _ = runnable_decision(
        probe_returncode=0,
        correctness_ok=True,
        before_signature=before,
        after_signature=after,
    )
    assert runs is True
