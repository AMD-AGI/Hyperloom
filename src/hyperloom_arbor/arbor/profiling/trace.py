"""Chrome-trace reading: kernel totals, step segmentation, phase classification.

Pure stdlib. No torch, no vLLM — a trace is JSON and this module treats it as
JSON, so every rule below is testable without a GPU.

Two jobs:

* **Totals.** Sum GPU time per kernel name across every worker's trace. With
  TP>1 each rank writes its own file; summing them preserves ratios, which is
  all a component *share* needs.
* **Phase.** Say whether the captured steps were decode, prefill, or a mix.
  The engine is what *isolates* a phase (vLLM's ``delay_iterations`` skips
  prefill outright); this module's job is to **check that the isolation held**.
  A profile that assumed decode and silently caught a prefill chunk will
  overstate every attention component, and nothing downstream would notice.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

# Categories torch emits for device-side work. "kernel" is the interesting one;
# the memcpy/memset categories are included because a page-table build that
# spends its time in a broadcast copy is still spending it.
GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")

# Attention kernel families, used only to CLASSIFY a step, never to attribute
# time. Prefill and decode run different attention kernels, so the presence of
# a prefill-family kernel inside a step is positive evidence that the step was
# not pure decode.
#
# Starting table for DSv4.x on ROCm/vLLM. Extend per model — and when you do,
# extend `PREFILL_KERNELS` first: a missing prefill pattern makes a mixed step
# look clean, which is the failure that matters.
PREFILL_KERNELS = (
    "flash_fwd",
    "fmha",
    "mha_fwd",
    "_attn_fwd",
    "chunked_prefill",
    "prefill",
    "varlen",
)

DECODE_KERNELS = (
    "paged_mqa_logits",
    "paged_attention",
    "mla_decode",
    "fwd_kvcache",
    "decode_attention",
    "topk_transform",
    # DSv4.1-Flash / DSA on gfx950. Verified against a real decode capture
    # (sessions/decode_share_reparsed.json): the kernel is spelled
    # `_sparse_attn_decode_gfx950_partial_kernel.kd`, which matches none of the
    # patterns above — "attn_decode", not "decode_attention". Without this the
    # model's own decode attention is invisible to the classifier and a clean
    # decode capture classifies as `prefill`.
    "attn_decode",
    "sparse_attn_decode",
)

# Kernels that look like prefill by name but legitimately run during decode when
# speculative decoding is on. DSpark verifies (1 + num_spec_tokens) tokens per
# request per step, and that multi-token verify uses the RAGGED attention
# kernel — the same one prefill uses. It shows up as a small share of decode GPU
# time in a clean DSpark capture.
#
# Treating these as proof of prefill makes `phase_report` return `contaminated`
# on a perfectly good profile, and the pack tells the agent that a contaminated
# profile overstates every attention share — so a false positive here does not
# merely mislead, it causes valid work to be discarded.
SPEC_DECODE_AMBIGUOUS_KERNELS = (
    "sparse_attn_prefill_ragged",
)

# Events that plausibly delimit one engine step. vLLM annotates spans via
# torch.profiler.record_function; which name is used drifts between builds, so
# this is a list of candidates rather than a single constant.
STEP_ANNOTATION_PATTERNS = (
    r"execute_model",
    r"model_forward",
    r"EngineCore\.step",
    r"Worker\.execute_model",
)

ANNOTATION_CATEGORIES = ("user_annotation", "cpu_op", "python_function")


@dataclass(frozen=True)
class KernelTotals:
    """GPU time per kernel name, summed across ranks.

    ``by_name`` maps kernel name -> (microseconds, launch count). ``total_us``
    is the sum over **every** kernel, which is the only correct denominator for
    a component share.

    The launch count is a float, not an int: once :meth:`scaled` has divided a
    capture by its step count, "1.5 launches per step" is the honest answer.
    """

    by_name: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_files: int = 0
    n_events: int = 0

    @property
    def total_us(self) -> float:
        return sum(us for us, _ in self.by_name.values())

    def top(self, n: int) -> list[tuple[str, float, float]]:
        """The n most expensive kernels, for display only."""
        ranked = sorted(self.by_name.items(), key=lambda kv: -kv[1][0])
        return [(name, us, cnt) for name, (us, cnt) in ranked[:n]]

    def scaled(self, divisor: float) -> "KernelTotals":
        """Divide every total by ``divisor`` — e.g. to get per-step cost."""
        if divisor <= 0:
            raise ValueError(f"divisor must be positive, got {divisor}")
        return KernelTotals(
            by_name={k: (us / divisor, cnt / divisor) for k, (us, cnt) in self.by_name.items()},
            n_files=self.n_files,
            n_events=self.n_events,
        )


@dataclass(frozen=True)
class Step:
    """One engine step, as a half-open time span on the trace clock."""

    ts: float
    dur: float
    name: str

    @property
    def end(self) -> float:
        return self.ts + self.dur

    def contains(self, ts: float) -> bool:
        return self.ts <= ts < self.end


def load_events(trace_dir: str) -> tuple[list[dict], int]:
    """Read every trace file under ``trace_dir``, return (events, n_files).

    Handles plain and gzipped chrome traces. An unreadable file is skipped with
    a note rather than aborting the run: with TP=4, three good ranks still give
    correct ratios, and losing the whole profile to one truncated file is the
    worse outcome.
    """
    events: list[dict] = []
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True))
    n_read = 0
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as fh:
                trace = json.load(fh)
        except Exception as exc:  # truncated / still-flushing / not a trace
            print(f"[profiling] skipping unreadable trace {os.path.basename(path)}: {exc}")
            continue
        events.extend(trace.get("traceEvents", []))
        n_read += 1
    return events, n_read


def aggregate_kernels(events: list[dict], n_files: int = 0) -> KernelTotals:
    """Sum GPU time by kernel name over ``events``."""
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    n = 0
    for e in events:
        if e.get("cat") in GPU_CATEGORIES and "dur" in e:
            slot = by_name[e["name"]]
            slot[0] += float(e["dur"])
            slot[1] += 1
            n += 1
    return KernelTotals(
        by_name={k: (v[0], int(v[1])) for k, v in by_name.items()},
        n_files=n_files,
        n_events=n,
    )


def segment_steps(
    events: list[dict],
    patterns: tuple[str, ...] = STEP_ANNOTATION_PATTERNS,
) -> list[Step]:
    """Find per-step spans on the host timeline.

    Returns ``[]`` when no annotation matches. That is a real answer, not a
    failure to be papered over: it means this build does not annotate steps in
    a way we recognise, so phase isolation cannot be verified from the trace
    and the caller must say ``unverified``.
    """
    rx = re.compile("|".join(patterns))
    spans = [
        Step(ts=float(e["ts"]), dur=float(e["dur"]), name=str(e["name"]))
        for e in events
        if e.get("cat") in ANNOTATION_CATEGORIES
        and "dur" in e
        and "ts" in e
        and rx.search(str(e.get("name", "")))
    ]
    spans.sort(key=lambda s: s.ts)

    # Nested annotations (execute_model inside EngineCore.step) would double
    # count. Keep only outermost spans.
    outermost: list[Step] = []
    for s in spans:
        if outermost and s.end <= outermost[-1].end:
            continue
        outermost.append(s)
    return outermost


def classify_phase(
    kernel_names: list[str],
    prefill: tuple[str, ...] = PREFILL_KERNELS,
    decode: tuple[str, ...] = DECODE_KERNELS,
    spec_decode: bool = False,
) -> str:
    """Label a set of kernel names ``prefill`` / ``decode`` / ``mixed`` / ``unknown``.

    ``spec_decode`` must be True when the capture ran with speculative decoding.
    It suppresses the kernels in :data:`SPEC_DECODE_AMBIGUOUS_KERNELS` as
    *prefill evidence* — they are expected in a decode step under multi-token
    verification. They still count as decode evidence via ``decode`` patterns if
    they match any; they simply stop being proof of the other phase.
    """
    lowered = [n.lower() for n in kernel_names]
    if spec_decode:
        ambiguous = [
            n for n in lowered
            if any(a in n for a in SPEC_DECODE_AMBIGUOUS_KERNELS)
        ]
        prefill_candidates = [n for n in lowered if n not in ambiguous]
    else:
        prefill_candidates = lowered
    has_prefill = any(p in n for n in prefill_candidates for p in prefill)
    has_decode = any(d in n for n in lowered for d in decode)
    if has_prefill and has_decode:
        return "mixed"
    if has_prefill:
        return "prefill"
    if has_decode:
        return "decode"
    return "unknown"


def phase_report(events: list[dict], expected: str = "decode",
                 spec_decode: bool = False) -> dict:
    """Check that the captured steps are the phase the caller believes they are.

    The engine isolates the phase; this verifies it. Returns a dict carrying
    ``isolation`` — one of:

    ``clean``        every classified step matched ``expected``
    ``contaminated`` at least one step carried work from another phase
    ``unverified``   steps could not be segmented, or none could be classified

    ``unverified`` is not a soft pass. It means the profile's phase label rests
    on the launch arguments alone, exactly as it did before this module existed.
    """
    steps = segment_steps(events)
    if not steps:
        return {
            "isolation": "unverified",
            "reason": "no recognised step annotation in trace",
            "expected": expected,
            "n_steps": 0,
        }

    gpu = sorted(
        (
            (float(e["ts"]), str(e["name"]))
            for e in events
            if e.get("cat") in GPU_CATEGORIES and "ts" in e
        ),
        key=lambda x: x[0],
    )

    per_step: list[list[str]] = [[] for _ in steps]
    i = 0
    for ts, name in gpu:
        while i < len(steps) and ts >= steps[i].end:
            i += 1
        if i >= len(steps):
            break
        if steps[i].contains(ts):
            per_step[i].append(name)

    labels = [classify_phase(names, spec_decode=spec_decode)
              for names in per_step if names]
    counts: dict[str, int] = defaultdict(int)
    for lab in labels:
        counts[lab] += 1

    classified = [lab for lab in labels if lab != "unknown"]
    if not classified:
        isolation = "unverified"
        reason = "no step contained a recognised attention kernel"
    elif all(lab == expected for lab in classified):
        isolation = "clean"
        reason = ""
    else:
        isolation = "contaminated"
        off = sorted({lab for lab in classified if lab != expected})
        reason = f"{len(classified) - counts[expected]}/{len(classified)} steps were {', '.join(off)}"

    out = {
        "isolation": isolation,
        "expected": expected,
        "n_steps": len(steps),
        "n_classified": len(classified),
        "step_labels": dict(counts),
    }
    if reason:
        out["reason"] = reason
    return out
