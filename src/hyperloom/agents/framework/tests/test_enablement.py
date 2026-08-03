# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.enablement (failure classifier + request model).

Pure-Python, GPU-free: canned log string in → structured
:class:`FailureSignature` out.
"""

from __future__ import annotations

import pytest

from hyperloom.agents.framework.enablement import (
    ACCURACY_BELOW_FLOOR,
    CAPABILITY_DISABLED,
    EVAL_RUNTIME_FAILURE,
    HIP_KERNEL_MISSING,
    IMPORT_ERROR,
    MISSING_MODEL_ARCH,
    MISSING_WEIGHT,
    NOT_IMPLEMENTED,
    RESOURCE_CONSTRAINT,
    SERVE_FLAG,
    SHAPE_MISMATCH,
    TOKENIZER_ERROR,
    UNKNOWN,
    UNSUPPORTED_DTYPE,
    CapabilityGap,
    EnablementRequest,
    FailureSignature,
    _extract_offending_file,
    _failure_identity,
    classify_failure,
    enablement_made_progress,
    is_targeted_build_candidate,
    runnable_decision,
)


# --- classify_failure: kind detection --------------------------------------


def test_accuracy_below_floor_kind() -> None:
    sig = classify_failure("baseline accuracy did not meet floor: accuracy=0.12 floor=0.30 task=gsm8k")
    assert sig.kind == ACCURACY_BELOW_FLOOR
    assert sig.bridge_layer == ""


def test_eval_runtime_failure_kind() -> None:
    sig = classify_failure("benchmark_stderr.log: ERROR: run_eval failed with exit code 1")
    assert sig.kind == EVAL_RUNTIME_FAILURE


def test_eval_crash_with_import_error_classifies_as_import_error() -> None:
    """The generic eval rule is lowest priority: a real root cause in the same
    log (import/serve-flag) must win over eval_runtime_failure."""
    log = "run_eval failed with exit code 1\nModuleNotFoundError: No module named 'lm_eval'"
    assert classify_failure(log).kind == IMPORT_ERROR
    log2 = "run_eval failed with exit code 1\nvllm: error: unrecognized arguments: --bad"
    assert classify_failure(log2).kind == SERVE_FLAG


def test_missing_model_arch() -> None:
    """Unsupported architecture message -> missing_model_arch + arch symbol."""
    log = "ValueError: Model architecture 'Glm5ForCausalLM' is not supported for now."
    sig = classify_failure(log)
    assert sig.kind == MISSING_MODEL_ARCH
    assert sig.offending_symbol == "Glm5ForCausalLM"
    assert sig.bridge_layer == "framework"
    assert sig.confidence > 0.9
    assert sig.is_actionable


def test_missing_model_arch_transformers_unrecognized() -> None:
    """Transformers 'does not recognize this architecture' (the DeepSeek-V4
    brand-new-arch-on-old-stack signature, wrapped in a vLLM ModelConfig
    ValidationError) -> missing_model_arch with the model_type as symbol."""
    log = (
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig\n"
        "  Value error, The checkpoint you are trying to load has model type "
        "`deepseek_v4` but Transformers does not recognize this architecture."
    )
    sig = classify_failure(log)
    assert sig.kind == MISSING_MODEL_ARCH
    assert sig.offending_symbol == "deepseek_v4"
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
    sig = classify_failure("RuntimeError: \"addmm\" not implemented for 'Float8_e4m3fn'")
    assert sig.kind == UNSUPPORTED_DTYPE
    assert sig.offending_symbol == "Float8_e4m3fn"


def test_shape_mismatch() -> None:
    """A tensor shape error -> shape_mismatch."""
    sig = classify_failure("RuntimeError: shape '[2, 4096]' is invalid for input of size 4096")
    assert sig.kind == SHAPE_MISMATCH


def test_shape_mismatch_narrow_bounds() -> None:
    """A torch .narrow() bounds error (fused-projection width mismatch) -> shape_mismatch."""
    sig = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    assert sig.kind == SHAPE_MISMATCH


def test_missing_weight_not_initialized() -> None:
    """A strict weight-init failure -> missing_weight (distinct from shape_mismatch)."""
    log = (
        "ValueError: Following weights were not initialized from checkpoint: "
        "{'model.layers.19.self_attn.indexer.k_norm.weight', "
        "'model.layers.20.self_attn.indexer.k_norm.bias'}"
    )
    sig = classify_failure(log)
    assert sig.kind == MISSING_WEIGHT
    assert sig.is_actionable
    assert sig.bridge_layer == "framework"


def test_missing_weight_state_dict_keys() -> None:
    """torch load_state_dict missing/unexpected keys -> missing_weight."""
    assert classify_failure("RuntimeError: Missing key(s) in state_dict: 'x.weight'").kind == MISSING_WEIGHT
    assert classify_failure("Error(s) in loading state_dict for Model:").kind == MISSING_WEIGHT


def test_shape_mismatch_and_missing_weight_are_distinct() -> None:
    """The two serial GLM-5.2 gaps classify as different kinds (progress detectable)."""
    gap1 = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    gap2 = classify_failure(
        "ValueError: Following weights were not initialized from checkpoint: {'m.indexer.k_norm.weight'}"
    )
    assert gap1.kind == SHAPE_MISMATCH
    assert gap2.kind == MISSING_WEIGHT
    assert gap1.kind != gap2.kind


def test_enablement_made_progress_different_kind() -> None:
    """A patch that moves the boot to a different actionable failure = progress."""
    before = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    after = classify_failure(
        "ValueError: Following weights were not initialized from checkpoint: {'m.indexer.k_norm.weight'}"
    )
    assert enablement_made_progress(before, after) is True


def test_enablement_no_progress_same_failure() -> None:
    """The identical unfixed failure re-appearing is NOT progress."""
    before = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    after = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    assert enablement_made_progress(before, after) is False


def test_enablement_no_progress_clean_boot() -> None:
    """A clean boot (non-actionable UNKNOWN after) is runnable, not 'progress'."""
    before = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    after = FailureSignature(kind=UNKNOWN)
    assert enablement_made_progress(before, after) is False


def test_enablement_progress_first_actionable_when_no_prior() -> None:
    """With no prior actionable signature, any actionable post-patch failure is a step."""
    after = classify_failure("ValueError: weights were not initialized from checkpoint")
    assert enablement_made_progress(None, after) is True


def test_enablement_progress_unknown_to_different_unknown() -> None:
    """Q1: two DIFFERENT unknown failures still count as progress (taxonomy-independent)."""
    before = classify_failure("novel error A raised in foo.py during widget init")
    after = classify_failure("completely different novel error B in bar.py during frobnicate")
    assert enablement_made_progress(before, after) is True


def test_enablement_no_progress_same_unknown_text() -> None:
    """Q1: the same unknown failure re-appearing (even numeric operands vary) is NOT progress."""
    before = classify_failure("weird failure at offset 128 in module qux")
    after = classify_failure("weird failure at offset 256 in module qux")
    assert enablement_made_progress(before, after) is False


def test_enablement_no_progress_clean_boot_unknown_signature() -> None:
    """A bare UNKNOWN signature (no error text) is a clean boot, not progress."""
    before = classify_failure("RuntimeError: start (0) + length (704) exceeds dimension size (576).")
    assert enablement_made_progress(before, FailureSignature(kind=UNKNOWN)) is False


def test_enablement_setup_guidance_in_mandate() -> None:
    """Q3: the authored mandate authorizes env setup and asks to record setup_commands."""
    from hyperloom.agents.framework.enablement import EnablementRequest
    from hyperloom.agents.framework.enablement_ops import ENABLEMENT_SETUP_GUIDANCE, build_mandate

    req = EnablementRequest(
        framework="vllm",
        model="GLM-5.2",
        repo_url="https://github.com/ROCm/vllm.git",
        launch_log="ValueError: weights were not initialized from checkpoint",
        gpu_type="mi300x",
    )
    m = build_mandate(req)
    assert "ENVIRONMENT SETUP" in m.task_description
    assert "setup_commands" in m.task_description
    assert ENABLEMENT_SETUP_GUIDANCE


def test_enablement_progress_contract_in_mandate() -> None:
    """Serial-enablement contract: the mandate must tell the specialist that a
    patch which only ADVANCES the boot one step is a valid KEPT deliverable, so
    a large gap yields incremental progress instead of a wholesale empty=true.

    This is the specialist-side counterpart of ``enablement_made_progress`` /
    integrate_patch ``status="advanced"`` — without it the incremental stacking
    machinery is never fed any patches (observed on DeepSeek-V4-Flash: every
    round returned empty=true and nothing was ever stacked)."""
    from hyperloom.agents.framework.enablement import EnablementRequest
    from hyperloom.agents.framework.enablement_ops import (
        ENABLEMENT_PROGRESS_GUIDANCE,
        build_mandate,
    )

    req = EnablementRequest(
        framework="vllm",
        model="deepseek-ai-DeepSeek-V4-Flash",
        repo_url="https://github.com/ROCm/vllm.git",
        launch_log=(
            "The checkpoint you are trying to load has model type `deepseek_v4` "
            "but Transformers does not recognize this architecture."
        ),
        gpu_type="mi355x",
    )
    m = build_mandate(req)
    assert "PROGRESS DELIVERABLE" in m.task_description
    # The contract must explicitly permit an advance-one-step patch and reserve
    # empty=true for "cannot advance even one step".
    assert "ADVANCES the boot" in m.task_description
    assert "empty=true" in m.task_description
    assert ENABLEMENT_PROGRESS_GUIDANCE


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
        "Traceback (most recent call last):\n"
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
        "Traceback (most recent call last):\n"
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


# --- New kinds: tokenizer_error, serve_flag, resource_constraint -----------


def test_tokenizer_error_classified() -> None:
    """A vLLM --tokenizer-mode not supported message -> tokenizer_error."""
    sig = classify_failure("Error: Tokenizer mode 'deepseek_v4' is not supported")
    assert sig.kind == TOKENIZER_ERROR


def test_tokenizer_unknown_backend() -> None:
    sig = classify_failure("Unknown tokenizer class: FastTokenizerV2")
    assert sig.kind == TOKENIZER_ERROR


def test_serve_flag_classified() -> None:
    """Argparse unrecognized flag -> serve_flag."""
    sig = classify_failure("unrecognized arguments: --enable-mtp-speculative-decoding")
    assert sig.kind == SERVE_FLAG


def test_serve_flag_invalid_choice() -> None:
    sig = classify_failure("error: argument --tokenizer-mode: invalid choice: 'deepseek_v4'")
    assert sig.kind in (SERVE_FLAG, TOKENIZER_ERROR)


def test_resource_constraint_oom() -> None:
    """Out-of-memory -> resource_constraint."""
    sig = classify_failure("RuntimeError: CUDA out of memory. Tried to allocate 20 GiB")
    assert sig.kind == RESOURCE_CONSTRAINT


def test_resource_constraint_hip_oom() -> None:
    sig = classify_failure("RuntimeError: HIP out of memory.")
    assert sig.kind == RESOURCE_CONSTRAINT


def test_resource_constraint_tp() -> None:
    sig = classify_failure("requires at least 8 GPUs for tensor parallel size 8, but only 4 are available")
    assert sig.kind == RESOURCE_CONSTRAINT


def test_resource_constraint_no_kv_cache() -> None:
    sig = classify_failure("No GPU memory left for the KV cache. Please try enabling quantization")
    assert sig.kind == RESOURCE_CONSTRAINT


# --- CapabilityGap projection -----------------------------------------------


def test_capability_gap_from_resource_constraint() -> None:
    """resource_constraint -> requires_code_acquisition=False."""
    sig = FailureSignature(kind=RESOURCE_CONSTRAINT, confidence=0.88)
    gap = CapabilityGap.from_signature(sig)
    assert gap.kind == RESOURCE_CONSTRAINT
    assert gap.requires_code_acquisition is False


def test_capability_gap_from_import_error() -> None:
    """import_error -> requires_code_acquisition=True."""
    sig = FailureSignature(kind=IMPORT_ERROR, confidence=0.7, bridge_layer="build")
    gap = CapabilityGap.from_signature(sig)
    assert gap.requires_code_acquisition is True
    assert gap.bridge_layer == "build"


def test_capability_gap_from_tokenizer_error() -> None:
    sig = FailureSignature(kind=TOKENIZER_ERROR, confidence=0.75, bridge_layer="framework")
    gap = CapabilityGap.from_signature(sig)
    assert gap.requires_code_acquisition is True


def test_capability_gap_to_dict() -> None:
    gap = CapabilityGap(kind=RESOURCE_CONSTRAINT, requires_code_acquisition=False)
    d = gap.to_dict()
    assert d["kind"] == RESOURCE_CONSTRAINT
    assert d["requires_code_acquisition"] is False


def test_oom_classify_and_gap_no_code_acquisition() -> None:
    """End-to-end: OOM log -> classify -> CapabilityGap.requires_code_acquisition is False."""
    sig = classify_failure("RuntimeError: Out of memory on GPU. Tried to allocate 80 GiB")
    assert sig.kind == RESOURCE_CONSTRAINT
    gap = CapabilityGap.from_signature(sig)
    assert gap.requires_code_acquisition is False


# --- is_targeted_build_candidate (Rung 5 eligibility) ----------------------


def test_targeted_build_candidate_build_bridge_layer() -> None:
    sig = FailureSignature(kind=IMPORT_ERROR, bridge_layer="build")
    assert is_targeted_build_candidate(sig) is True


def test_targeted_build_candidate_rocm_hip_bridge_layer() -> None:
    sig = FailureSignature(kind=HIP_KERNEL_MISSING, bridge_layer="rocm_hip")
    assert is_targeted_build_candidate(sig) is True


def test_targeted_build_candidate_hip_kernel_missing_kind() -> None:
    sig = FailureSignature(kind=HIP_KERNEL_MISSING, bridge_layer="")
    assert is_targeted_build_candidate(sig) is True


def test_targeted_build_candidate_pure_python_dtype_rejected() -> None:
    """A dtype guard with no native evidence stays Rung 4 (pure Python)."""
    sig = FailureSignature(kind=UNSUPPORTED_DTYPE, bridge_layer="framework", raw_excerpt="dtype bf16 is not supported")
    assert is_targeted_build_candidate(sig) is False


def test_targeted_build_candidate_native_dtype_from_symbol() -> None:
    sig = FailureSignature(
        kind=UNSUPPORTED_DTYPE,
        bridge_layer="framework",
        offending_symbol="aiter::fp4_moe",
    )
    assert is_targeted_build_candidate(sig) is True


def test_targeted_build_candidate_native_dtype_from_log() -> None:
    sig = FailureSignature(kind=UNSUPPORTED_DTYPE, bridge_layer="framework")
    log = "torch.ops._C.fp4_gemm undefined symbol: _ZN5aiter..."
    assert is_targeted_build_candidate(sig, log) is True


def test_targeted_build_candidate_framework_python_gap_rejected() -> None:
    sig = FailureSignature(kind=MISSING_MODEL_ARCH, bridge_layer="framework")
    assert is_targeted_build_candidate(sig) is False


def test_targeted_build_candidate_none_signature() -> None:
    assert is_targeted_build_candidate(None) is False  # type: ignore[arg-type]


# --- _extract_offending_file / _failure_identity ---------------------------


def test_extract_offending_file_falls_back_to_last_traceback_frame() -> None:
    """With no ``near`` offset, the last traceback ``File "..."`` frame wins."""
    text = (
        'Traceback (most recent call last):\n'
        '  File "/opt/vllm/first.py", line 10, in boot\n'
        '  File "/opt/vllm/last.py", line 42, in load_model\n'
        "RuntimeError: boom"
    )
    assert _extract_offending_file(text) == "/opt/vllm/last.py"


def test_failure_identity_none_signature_is_empty_triple() -> None:
    """A ``None`` signature yields three empty strings (dedup key for no-sig)."""
    assert _failure_identity(None) == ("", "", "")


def test_failure_signature_to_dict_round_trips_fields() -> None:
    """``FailureSignature.to_dict`` serializes all dataclass fields."""
    sig = FailureSignature(kind=MISSING_MODEL_ARCH, confidence=0.9, offending_file="m.py")
    d = sig.to_dict()
    assert d["kind"] == MISSING_MODEL_ARCH
    assert d["offending_file"] == "m.py"
    assert d["confidence"] == 0.9
