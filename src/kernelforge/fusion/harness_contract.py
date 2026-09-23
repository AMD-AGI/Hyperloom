# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The contract the fusion harness must satisfy."""

from __future__ import annotations

from typing import Any, Optional

from .validate import DEFAULT_SNR_THRESHOLD_DB, DEFAULT_TARGET_SPEEDUP

TRACE_KERNELS_HEADING = "Kernel sequence the trace recorded"


def trace_kernels_block(evidence: Optional[dict[str, Any]]) -> str:
    """The recorded launches the eager reference has to reproduce.

    Source symbol names are not evidence. A framework routinely ships two or three
    implementations of the same chain -- a Triton one, a JIT C++ one, a torch
    fallback -- behind names that all read like the right thing, and only one is
    reachable from the served forward pass. Naming the kernels the trace ACTUALLY
    recorded turns "which implementation is this" from a judgement call into a
    comparison the agent can run. Renders nothing without an anchor: an empty
    heading reads as "the trace recorded none", which is a stronger and false claim.
    """
    evidence = evidence or {}
    anchor = str(evidence.get("anchor") or "").strip()
    if not anchor:
        return ""

    def listing(title: str, key: str) -> str:
        names = [str(n) for n in (evidence.get(key) or [])]
        if not names:
            return f"{title}: nothing recorded (stream boundary)\n"
        return title + ":\n" + "".join(f"    {n}\n" for n in names)

    span_block = ""
    for item in evidence.get("span") or []:
        # Tolerate a bare name: the span is also read back from run artifacts.
        name = str(item.get("name", "")) if isinstance(item, dict) else str(item)
        is_anchor = bool(item.get("is_anchor")) if isinstance(item, dict) else name == anchor
        span_block += f"    {'>> ' if is_anchor else '   '}{name}\n"
    return f"""
## {TRACE_KERNELS_HEADING} (GROUND TRUTH)
These are the GPU kernels this fusion has to account for. They come from the
profile, not from reading source, so where they disagree with any name above, they
win.

ANCHOR (the kernel the operator pinned):
    {anchor}

{listing("Immediately before the anchor", "before")}
{listing("Immediately after the anchor", "after")}
Compute-bounded span of one representative launch (anchor marked `>>`):
{span_block}
Use this list to CHECK which framework code path you are looking at. A module that
defines the right-sounding function is not necessarily the one that ran: if the
functions you picked do not launch these kernels, you are reading a code path the
served model never reaches, and everything measured against it is measured against
the wrong baseline.
"""


def harness_contract(harness_path: str = "", env_flags: str = "") -> str:
    """Render the contract, naming the path and flags when the caller knows them."""
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
  1. build the eager arm out of the framework's REAL forward path. Do not pick the
     functions by name. Start at the model's forward pass and follow the actual
     calls -- through the module the layer instantiates, the mixin its attention
     backend really inherits, the dispatch branch this configuration takes -- down
     to the functions that issue the chain, and call THOSE. A framework typically
     ships more than one implementation of the same chain, and the one whose name
     reads best is regularly the dead one: resolve every import to the module it
     actually binds, and prefer a symbol you traced from the forward pass over a
     symbol that merely matches the hint. If a hinted symbol turns out not to be on
     the path, print a line saying which symbol you used instead and why, then use
     it -- the hints were read off source, and this step is how they get corrected,
  2. PROVE the eager arm is the right code path before anything else. Run one
     untimed eager iteration under
     `torch.profiler.profile(activities=[ProfilerActivity.CUDA])`, collect the
     device kernel names, and compare them against the kernel list the task gives
     you under "{TRACE_KERNELS_HEADING}" -- the anchor and its recorded neighbours
     must appear. (If the task lists no kernels, check them against the chain you
     set out to replace instead, and report what you saw.) If they do not appear,
     your eager arm is a different implementation from the one that runs in
     production: go back to step 1, follow the calls again, and fix it. Do not
     measure a baseline you have not matched -- a parity and a speedup computed
     against the wrong chain are worth less than no numbers at all, because they
     look like evidence. Report the names you observed in "eager_kernels" and
     whether they matched in "eager_matches_trace",
  3. build the fused arm as ONE call to the fused module's single entry point. You
     are writing this before that module exists, so resolve the function at run time
     rather than naming it: import the fused module and take
     `getattr(mod, mod.__forge_fused_entry__)`, the one public function the author is
     required to export. Treat a missing module as the baseline case below, not an
     error. The harness MEASURES the fusion, it does not implement any part of it:
     if making the fused call work needs a differently-produced input (a GEMM invoked
     with another output dtype, a tensor kept in its original layout), that step is
     the author's to put inside the entry point or at the framework call site. A
     setup line here is a piece of the fusion that will not exist in the served
     model, and it silently moves work out of the arm being measured,
  4. build representative decode tensors from the shapes above,
  5. run the fused kernel vs the eager op, compute per-shape parity
     (snr_db = 10*log10(sum(ref^2)/sum((ref-fused)^2)); also max_abs_err),
  6. microbench eager vs fused in microseconds. Warm up EACH arm with at least
     500 iterations BEFORE timing it, then time >= 200 iterations and report the
     median. The warm-up size is not a detail to trim: measured on this hardware,
     a 25-iteration warm-up leaves the chip below its steady clock and whichever
     arm is timed SECOND comes out ~3% slower from heat alone -- the same size as
     the speedup gate you are being judged against, and always against the fused
     arm if you time eager first,
  7. count the GPU kernel launches EACH arm issues for ONE decode step, over the
     whole step and not just the chain you replaced. Measure them, do not reason
     about them: run one untimed iteration of each arm under
     `torch.profiler.profile(activities=[ProfilerActivity.CUDA])` and count the
     device kernel events. If you genuinely cannot count on this setup, report
     null for both rather than a guess -- a wrong number is worse than none.
  8. print, as the LAST stdout line, ONE JSON object (and nothing after it):
     {{"compiled": true/false, "is_triton": true/false, "error": "",
       "parity": [{{"snr_db": <float or null>, "max_abs_err": <float or null>, "label": "<shape>"}}],
       "eager_us": <float or null>, "fused_us": <float or null>,
       "eager_launches": <int or null>, "fused_launches": <int or null>,
       "eager_kernels": ["<device kernel name observed on the eager arm>"],
       "eager_matches_trace": true/false,
       "skipped": false, "skip_reason": ""}}
  - ``eager_matches_trace`` false means the whole run measured the wrong code path,
    so fix the eager arm rather than reporting it and moving on. Report it honestly
    if you cannot: a false here is a repairable mistake, a true you did not verify
    is a fabricated baseline that the campaign will build on for hours.
  - ``fused_launches`` MUST be strictly less than ``eager_launches``. That is the
    point of the fusion, and it is gated on: a candidate that does not reduce the
    launch count is REJECTED even when parity holds and the microbench is faster.
    A scratch-fill, a separate cast or a contiguous copy added to feed the fused
    kernel can cancel every launch it saved while the chain alone still times
    better. Two null counts leave the fusion unverified on this gate, not passed.
  - On a hybrid/Mamba model where the decode microbench cannot init on ROCm, set
    "skipped": true + "skip_reason" (parity still required); on compile failure set
    "compiled": false + "error" with the real message.
  - "fused_us" may be null ONLY together with "skipped": true and a "skip_reason"
    saying why. With "skipped": false the loop needs a number: a null fused time
    there claims the microbench ran and measured nothing, and the iteration fails
    on it.
  - The loop runs this file ONCE BEFORE any fusion exists, to anchor the speedup
    on the unfused framework. With the fused module missing or empty, time the
    eager op for BOTH arms rather than failing, report the eager launch count for
    both, and still report "compiled": true -- that run IS the baseline, and the
    launch gate does not apply to it. The driver reads the fused module to see that
    nothing is fused yet and anchors on the eager time there, so the pristine run
    is never the failure above. "compiled": false means a real compile failure: the
    driver reports it as a crash, so the loop starts with no
    per-case timings and aborts before its first iteration.
Do NOT hard-code metrics; compute them live. Parity uses an \
SNR>={DEFAULT_SNR_THRESHOLD_DB:g} dB gate and the keep bar is \
>={DEFAULT_TARGET_SPEEDUP:g}x.
"""
