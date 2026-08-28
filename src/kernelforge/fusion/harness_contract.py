# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The contract the fusion harness must satisfy.

Two prompts state this contract -- the per-recipe authoring prompt, which knows
the exact path and env flags, and the fusion kernel backend's system prompt, which does
not. They render from here so the harness an author writes and the harness the
loop parses cannot describe different files.
"""

from __future__ import annotations

from .validate import DEFAULT_SNR_THRESHOLD_DB, DEFAULT_TARGET_SPEEDUP


def harness_contract(harness_path: str = "", env_flags: str = "") -> str:
    """Render the contract, naming the path and flags when the caller knows them.

    :class:`~kernelforge.fusion.validate.HarnessKernelRunner` runs this exact
    file and parses ONE JSON object from its stdout, so an author who does not
    produce it fails every validation attempt with "harness not found".
    """
    where = f"at EXACTLY:\n    {harness_path}" if harness_path else "at the harness path the task gives you."
    flags = f"`{env_flags}`" if env_flags else "the fusion env flag(s)"
    return f"""
## Kernel-validation harness (MANDATORY — the loop runs THIS to score you)
Write a self-contained Python script {where}
The loop RUNS this script from a different directory than the one you write it
in, so a framework path derived from `__file__` will not exist at run time.
Locate the framework tree through `$FORGE_FUSION_FRAMEWORK_ROOT` (exported for
the run, and also the process's cwd) and never through `__file__`.
It must, guarded by {flags}:
  1. import the fused module AND the REAL eager op (per the reference hint),
  2. build representative decode tensors from the shapes above,
  3. run the fused kernel vs the eager op, compute per-shape parity
     (snr_db = 10*log10(sum(ref^2)/sum((ref-fused)^2)); also max_abs_err),
  4. microbench eager vs fused in microseconds. Warm up EACH arm with at least
     500 iterations BEFORE timing it, then time >= 200 iterations and report the
     median. The warm-up size is not a detail to trim: measured on this hardware,
     a 25-iteration warm-up leaves the chip below its steady clock and whichever
     arm is timed SECOND comes out ~3% slower from heat alone -- the same size as
     the speedup gate you are being judged against, and always against the fused
     arm if you time eager first,
  5. print, as the LAST stdout line, ONE JSON object (and nothing after it):
     {{"compiled": true/false, "is_triton": true/false, "error": "",
       "parity": [{{"snr_db": <float or null>, "max_abs_err": <float or null>, "label": "<shape>"}}],
       "eager_us": <float or null>, "fused_us": <float or null>,
       "skipped": false, "skip_reason": ""}}
  - On a hybrid/Mamba model where the decode microbench cannot init on ROCm, set
    "skipped": true + "skip_reason" (parity still required); on compile failure set
    "compiled": false + "error" with the real message.
  - The loop runs this file ONCE BEFORE any fusion exists, to anchor the speedup
    on the unfused framework. With the fused module missing or empty, time the
    eager op for BOTH arms rather than failing, and still report
    "compiled": true -- that run IS the baseline. "compiled": false means a real
    compile failure: the driver reports it as a crash, so the loop starts with no
    per-case timings and aborts before its first iteration.
Do NOT hard-code metrics; compute them live. Parity uses an \
SNR>={DEFAULT_SNR_THRESHOLD_DB:g} dB gate and the keep bar is \
>={DEFAULT_TARGET_SPEEDUP:g}x.
"""
