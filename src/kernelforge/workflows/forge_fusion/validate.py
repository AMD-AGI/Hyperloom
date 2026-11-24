# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-level validation of an authored fusion (Phase 4; e2e is out of scope).

forge-fuse validates at the KERNEL level, NOT full serving e2e (that is
Hyperloom's job). The validator this module provides is:

* :func:`validate_recipe` -- the fine-grained, GPU-optional KERNEL validator used
  by the autoloop (see ``loop.py``). Given a :class:`~kernelforge.workflows.forge_fusion.models.Recipe`
  and an injectable :class:`KernelValidationRunner`, it runs three gates and
  returns a :class:`~kernelforge.workflows.forge_fusion.models.ValidationResult`:

    (a) COMPILE/IMPORT -- the fused kernel module must import and, if Triton,
        JIT-compile on this GPU arch. "Diagnosed headroom that cannot build on
        ROCm" is a hard FAIL (e.g. reusing a framework CUDA-only op such as
        ``fused_qk_norm_rope`` which pulls in ``cuda_bf16.h``).
    (b) NUMERICAL PARITY vs the REAL eager op -- compared with the shared SNR
        gate or an rtol fallback, NEVER strict allclose (bf16 + fp32-accum is
        not bit-exact).
    (c) MICROBENCH speedup -- ``eager_us`` vs ``fused_us``; the speedup must
        clear ``target_speedup`` and stay under this module's own absolute
        plausibility ceiling.
    (d) LAUNCH COUNT -- ``fused_launches`` must be strictly below
        ``eager_launches``. Launches are what a fusion buys, and a chain that
        times faster while adding a scratch-fill or a cast elsewhere has bought
        nothing; counts the harness could not measure leave this unverified
        rather than passed.

``kept`` requires (a) through (d).

The GPU/import work lives entirely behind the injectable ``KernelValidationRunner``
so the orchestration + parity math + ROCm failure-mode classification are unit
testable WITHOUT a GPU (tests pass a fake runner). Known ROCm failure modes are
encoded as first-class classifiers (:func:`classify_compile_error`,
:func:`classify_bench_skip`) so the loop's experience ledger learns the right
lesson (author Triton, not the framework CUDA op; skip the microbench when the
Mamba backend cannot init on ROCm).
"""

from __future__ import annotations

import json
import math
import ast
import os
import subprocess
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB

from .models import Recipe, ValidationResult


# ─────────────────────────── kernel-level validation ────────────────────────
# Phase 4 (kernel level). The three gates below are orchestrated by
# ``validate_recipe`` and exercised through an injectable ``KernelValidationRunner``
# so the decision logic is unit-testable without a GPU.

# Absolute-error fallback, used only when SNR is unavailable.
DEFAULT_RTOL = 2e-2
DEFAULT_TARGET_SPEEDUP = 1.03

# Absolute plausibility ceiling for the microbench speedup. The kernel-rewrite
# loop dropped its equivalent bound because it measures every candidate three
# times and can therefore judge a gain against the candidate's own noise. This
# validator has no such luxury: the harness self-reports one ``eager_us`` and
# one ``fused_us``, so there is no spread to compare against and nothing else
# stands between a broken timing path -- a load-independent floor, a fused arm
# that never ran -- and a KEEP. The highest speedup ever produced by a
# legitimate optimization here is 5.72x.
MAX_PLAUSIBLE_SPEEDUP = 20.0

# Known ROCm compile-failure signatures. A framework "fused" op written for CUDA
# pulls in CUDA-only headers/intrinsics and will NOT build on ROCm; the lesson the
# loop must learn is "author a ROCm-native Triton kernel, do not reuse the CUDA op".
_CUDA_ONLY_MARKERS = (
    "cuda_bf16.h",
    "cuda_fp16.h",
    "cuda_runtime",
    "nvcc",
    "__nv_",
    "sm_80",
    "sm_90",
    "cutlass",
    "mma.sync",
    "device_functions.h",
)
# Triton JIT/compile failures on this GPU arch — actionable but distinct
# from the CUDA-only case (the kernel IS ROCm-native, it just doesn't build yet).
_TRITON_BUILD_MARKERS = (
    "out of resource",
    "shared memory",
    "invalid argument",
    "passmanager",
    "llvm error",
    "triton",
    "cannot compile",
    "no kernel image",
)
# The decode microbench relies on ``bench_one_batch``; on ROCm it cannot init the
# Mamba/SSM backend, so for hybrid models the microbench must fall back / be
# skipped with a note rather than counting as a failure.
_MAMBA_MARKERS = ("mamba", "causal_conv1d", "selective_scan", "ssm", "hybrid")


@dataclass
class CompileOutcome:
    """Result of the compile/import + JIT gate (gate a)."""

    ok: bool
    is_triton: bool = False
    error: str = ""


@dataclass
class ParitySample:
    """One shape's parity metrics vs the imported real eager op (gate b).

    ``snr_db`` is the primary metric; ``max_abs_err`` is the rtol fallback used
    only when ``snr_db`` is unavailable. A runner computes these ON-DEVICE (where
    the tensors live) so only the scalars cross the boundary.
    """

    snr_db: Optional[float] = None
    max_abs_err: Optional[float] = None
    label: str = ""


@dataclass
class BenchOutcome:
    """Result of the microbench gate (gate c).

    ``skipped`` marks a benign unavailability (e.g. the Mamba backend cannot init
    on ROCm) — correctness still counts, but the speedup is unverified.

    The launch counts are per decode step, over the whole step rather than the
    replaced chain. ``None`` means the harness could not count them, which leaves
    the launch gate unverified rather than passed.
    """

    eager_us: Optional[float] = None
    fused_us: Optional[float] = None
    skipped: bool = False
    skip_reason: str = ""
    eager_launches: Optional[int] = None
    fused_launches: Optional[int] = None


@runtime_checkable
class KernelValidationRunner(Protocol):
    """Injectable boundary for all GPU/import work in :func:`validate_recipe`.

    Production code passes a runner that actually compiles + runs the kernel on
    the GPU (see :class:`HarnessKernelRunner`); unit tests pass a fake so the
    orchestration, parity math, and ROCm failure-mode classification are exercised
    without a GPU or an LLM.
    """

    def compile_check(self, recipe: Recipe) -> CompileOutcome:
        """Import the fused module and, if Triton, JIT-compile it on this arch."""

    def parity_samples(self, recipe: Recipe) -> list[ParitySample]:
        """Compare fused vs the imported REAL eager op on representative shapes."""

    def microbench(self, recipe: Recipe) -> BenchOutcome:
        """Time eager vs fused on the decode shape (may be skipped, see above)."""


def classify_compile_error(error: str, recipe: Optional[Recipe] = None) -> str:
    """Map a compile/import error to a crisp, reusable lesson (mirrors the
    forge-loop experience ledger's ``_CONSTRAINT_RULES``).

    The CUDA-only case is first-class: a framework "fused" op authored for CUDA
    fails to build on ROCm, and the loop must learn to author a ROCm-native Triton
    kernel instead of reusing it.
    """
    e = (error or "").lower()
    if any(m in e for m in _CUDA_ONLY_MARKERS):
        return (
            "The fused op is CUDA-only (pulls in e.g. cuda_bf16.h, like sglang's "
            "fused_qk_norm_rope) and cannot build on ROCm. Author a ROCm-native "
            "Triton kernel instead of reusing the framework CUDA fused op."
        )
    if any(m in e for m in _TRITON_BUILD_MARKERS):
        return (
            "The Triton kernel failed to JIT-compile on this GPU arch. "
            "Reduce BLOCK size / shared-memory usage or fix tl.constexpr shapes so "
            "it builds, and keep an eager fallback when Triton is unavailable."
        )
    return (
        "The fused module failed to import/compile on ROCm. Ensure the kernel is "
        "ROCm-native and falls back to eager when Triton is unavailable."
    )


def classify_bench_skip(reason: str) -> str:
    """Map a microbench skip reason to a reusable note.

    The Mamba/SSM-backend case is first-class: ``bench_one_batch`` cannot init the
    backend on ROCm for hybrid models, so the microbench is unavailable and the
    speedup is treated as unverified (NOT a failure) — parity remains the gate.
    """
    r = (reason or "").lower()
    if any(m in r for m in _MAMBA_MARKERS):
        return (
            "bench_one_batch cannot initialize the Mamba/SSM backend on ROCm, so "
            "the decode microbench is unavailable for hybrid models. Gate on "
            "kernel-level parity and treat the speedup as unverified, not failed."
        )
    return "Microbench unavailable; kernel-level parity is the gate for this attempt."


def implausible_speedup_reason(speedup: float) -> str:
    """Why this microbench speedup cannot be real, or "" when it can be.

    Fusion owns this bound rather than borrowing the rewrite loop's KEEP policy:
    the two answer different questions from different evidence, and the shared
    import made them look like one policy until the loop, which repeats every
    measurement, dropped its ceiling and silently took fusion's only anomaly
    check with it.
    """
    value = float(speedup)
    if not math.isfinite(value) or value > MAX_PLAUSIBLE_SPEEDUP:
        return f"microbench speedup {value:.6f}x exceeds the {MAX_PLAUSIBLE_SPEEDUP}x absolute plausibility ceiling"
    return ""


def launch_regression_reason(eager_launches: Optional[int], fused_launches: Optional[int]) -> str:
    """Why this fusion did not remove launches, or "" when it did.

    The whole lever is launch count, so a candidate that collapses its chain but
    adds a scratch-fill, a cast or a copy to feed the fused kernel can time
    faster on the chain and cost more over the decode step. Unknown counts are
    NOT a regression: the harness may be unable to profile, and failing a correct
    fusion for that would teach the author to game the number instead of the
    kernel. :func:`launch_gate_unverified` reports that case separately.
    """
    if eager_launches is None or fused_launches is None:
        return ""
    eager, fused = int(eager_launches), int(fused_launches)
    if fused < eager:
        return ""
    verdict = "removes no launches" if fused == eager else "ADDS launches"
    return (
        f"the fused path {verdict}: {eager} -> {fused} per decode step. Fusion is "
        f"paid for in launches, not in the timing of the chain alone; find what "
        f"the fused path launches in addition to the kernel you wrote"
    )


def launch_gate_unverified(eager_launches: Optional[int], fused_launches: Optional[int]) -> bool:
    """Whether the harness left the launch gate unmeasured."""
    return eager_launches is None or fused_launches is None


def launch_count(value: Any) -> Optional[int]:
    """One launch count out of harness JSON, or ``None`` when it is not usable.

    A harness that reports a count it did not measure is the failure this guards:
    anything that is not a non-negative integer reads as "not counted", which
    leaves the gate unverified instead of silently passing it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    count = int(value)
    return count if count >= 0 else None


def _tail(text: str, n: int = 400) -> str:
    """Last ``n`` chars of an error blob, single-lined for compact notes."""
    return " ".join((text or "").split())[-n:]


def _kernel_match_key(name: str) -> str:
    """One kernel name reduced to a form two profilers can agree on."""
    return " ".join(str(name or "").split()).lower()


def _same_kernel(expected: str, observed: str) -> bool:
    """Whether an observed launch is the kernel the trace recorded.

    Kineto and the in-process profiler render the same symbol slightly
    differently (a dropped ``void``, a differently spelled template argument), so
    containment either way counts as a match. The names are long mangled
    signatures, which makes an accidental containment match unlikely enough that
    tolerance costs less here than a false mismatch would.
    """
    want, got = _kernel_match_key(expected), _kernel_match_key(observed)
    if not want or not got:
        return False
    return want == got or want in got or got in want


def eager_trace_alignment(
    observed: Sequence[str],
    trace_kernels: Optional[dict[str, Any]],
) -> tuple[Optional[bool], str]:
    """Whether the harness's eager arm ran the code path the trace recorded.

    Everything the campaign reports -- parity, speedup, both launch counts -- is
    measured against whatever the eager arm happens to call, and nothing
    downstream re-checks that choice. A framework that ships two implementations
    of one chain (a Triton path and a JIT C++ path, say) will happily export
    plausible names for the dead one, and a harness built against it produces
    numbers that are internally consistent and describe software nobody runs.
    Comparing the arm's actual launches against the anchor's recorded
    neighbourhood is the one cheap check that catches it.

    Returns ``None`` when there is nothing to check -- unanchored discovery
    records no neighbourhood, and a harness that reports no kernel names leaves
    the question open rather than answering it "no".
    """
    evidence = trace_kernels or {}
    anchor = str(evidence.get("anchor") or "").strip()
    if not anchor:
        return None, "no anchored trace evidence to check the eager arm against"
    if not observed:
        return None, "harness reported no eager kernel names; alignment unchecked"

    neighbours: list[str] = []
    for key in ("before", "after"):
        for name in evidence.get(key) or []:
            text = str(name).strip()
            if text and text != anchor and text not in neighbours:
                neighbours.append(text)

    if not any(_same_kernel(anchor, name) for name in observed):
        return False, (
            f"eager arm never launched the anchor kernel ({_tail(anchor, 120)}); "
            f"it issued {len(observed)} other kernel(s), so it is a different code path"
        )
    matched = [n for n in neighbours if any(_same_kernel(n, name) for name in observed)]
    if neighbours and not matched:
        return False, (
            f"eager arm launched the anchor but none of its {len(neighbours)} recorded "
            "neighbour kernels; it reproduces the anchor in isolation, not the chain"
        )
    return True, f"eager arm matched the anchor and {len(matched)}/{len(neighbours)} recorded neighbours"


@dataclass(frozen=True)
class WiringEvidence:
    """What a static read of the framework edit proved about the fused call site."""

    verdict: str
    """``"wired"``, ``"not_wired"``, or ``"unchecked"`` when the gate could not judge."""

    reason: str


def _file_invocation_evidence(source_file: str) -> tuple[Optional[bool], str]:
    """One file's verdict: ``True`` wired, ``False`` dead import, ``None`` no evidence."""
    from .emit import _is_fused_module_name

    label = Path(source_file).name
    try:
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError) as exc:
        return None, f"{label} unchecked ({type(exc).__name__}: {exc})"

    # Names the wiring edit binds from a fused-kernel module, at any nesting
    # depth: a lazy import inside ``forward`` is a legitimate wiring style.
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            leaf = (node.module or "").rsplit(".", 1)[-1]
            if leaf and _is_fused_module_name(f"{leaf}.py"):
                bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_fused_module_name(f"{alias.name.rsplit('.', 1)[-1]}.py"):
                    bound.add(alias.asname or alias.name.split(".")[0])
    if not bound:
        # A fusion authored INLINE in the framework file imports nothing, and is
        # wired by construction. Only a bound-but-unused import is provable, so
        # this branch yields no evidence rather than a verdict.
        return None, f"{label} imports no fused-kernel module"

    # An ``import`` statement contributes ast.alias, never ast.Name, so any Name
    # load of a bound identifier is by construction a use outside the import.
    used = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bound
        }
    )
    if used:
        return True, f"{label} references {', '.join(used)}"
    return False, (
        f"{label} imports {', '.join(sorted(bound))} from a fused-kernel "
        f"module and never references it -- the fused kernel is dead code in the served model"
    )


def fused_symbol_invocation_evidence(
    source_file: str,
    extra_files: Sequence[str] = (),
) -> WiringEvidence:
    """Whether the framework edit CALLS the fused module, or only imports it.

    A fusion is delivered as two edits: a new fused-kernel module, and a wiring
    edit that makes the framework's forward path use it. Everything downstream
    measures only the first. The harness imports the fused entry point and times
    it against its own eager reference, so a 37x microbench is fully explained by
    a module that nothing calls; and the serving smoke boots the framework and
    sends real decodes, which succeed exactly as they did before because the
    unwired kernel never runs. Both report success for zero end-to-end gain --
    the failure this module already names elsewhere as "a PASS reported for a
    kernel that was never loaded, which is worse than a failure".

    This is the missing wiring check, and it is deliberately static: an import
    bound by a name that appears nowhere else in the file (the ``# noqa: F401``
    shape an agent produces when it authors the kernel but forgets the call site)
    cannot execute, whatever the runtime does. Everything else is ``unchecked``,
    which does not block a KEEP -- an unreadable or unparseable source, and
    equally a source that imports no fused module at all, which is what an INLINE
    fusion (the fused call written straight into the framework file) legitimately
    looks like. The gate exists to catch one provable defect, not to demote a KEEP
    it could not inspect, and equally not to claim it inspected one it did not.

    ``extra_files`` are the other files a multi-file fusion edits. The fused module
    only has to be called from ONE of them, so a single file's dead import is not
    the fusion's verdict: the check passes as soon as any file references it, and
    fails only when every file that imports it leaves it unused.
    """
    files = [p for p in [source_file, *extra_files] if p]
    if not files:
        return WiringEvidence("unchecked", "unchecked (no framework file recorded)")

    dead: list[str] = []
    silent: list[str] = []
    for path in files:
        verdict, reason = _file_invocation_evidence(path)
        if verdict is True:
            return WiringEvidence("wired", reason)
        (dead if verdict is False else silent).append(reason)
    if dead:
        return WiringEvidence("not_wired", "; ".join(dead))
    return WiringEvidence("unchecked", f"unchecked ({'; '.join(silent)})")


def validate_recipe(
    recipe: Recipe,
    runner: KernelValidationRunner,
    *,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    snr_threshold_db: float = DEFAULT_SNR_THRESHOLD_DB,
    rtol: float = DEFAULT_RTOL,
) -> ValidationResult:
    """Kernel-level validation of one authored fusion (gates a -> b -> c).

    The gates run in order and short-circuit on the first failure, so a compile
    failure never wastes a parity/bench run. ``kept`` is True only when the kernel
    COMPILES, matches the eager reference (parity), AND is at least
    ``target_speedup`` faster than eager.

    Args:
        recipe: The localized fusion plan (source of shapes, env flag, and the
            eager-reference hint). No per-model literals are read here.
        runner: The injectable GPU/import boundary (mock it in unit tests).
        target_speedup: Microbench speedup required to KEEP.
        snr_threshold_db: Numerical-parity SNR floor in dB.
        rtol: Absolute-error fallback used only when SNR is unavailable.

    Returns:
        A :class:`~kernelforge.workflows.forge_fusion.models.ValidationResult`. On any failure the
        ``note`` carries a compressed error signature plus a reusable LESSON so
        the loop's experience ledger can inject it into the next attempt.
    """
    # ── gate (a): compile / import (+ Triton JIT on this arch) ───────────────
    comp = runner.compile_check(recipe)
    if not comp.ok:
        lesson = classify_compile_error(comp.error, recipe)
        kind = "triton JIT" if comp.is_triton else "module import"
        return _unmeasured_rejection(f"COMPILE FAILED ({kind}): {_tail(comp.error)} | LESSON: {lesson}", rtol)

    # ── gate (b): numerical parity vs the imported REAL eager op ─────────────
    samples = runner.parity_samples(recipe)
    if not samples:
        return _unmeasured_rejection(
            "PARITY UNAVAILABLE: the runner returned no samples — the eager "
            "reference could not be imported/executed. LESSON: import the REAL "
            "eager op per the recipe's eager_reference_hint and assert parity.",
            rtol,
        )
    worst_err = max((s.max_abs_err for s in samples if s.max_abs_err is not None), default=None)
    min_snr = min((s.snr_db for s in samples if s.snr_db is not None), default=None)
    # SNR is the primary gate; rtol is the fallback only when SNR is unavailable.
    if min_snr is not None:
        parity_ok = min_snr >= snr_threshold_db
    elif worst_err is not None:
        parity_ok = worst_err <= rtol
    else:
        parity_ok = False
    if not parity_ok:
        snr_txt = f"{min_snr:.1f} dB" if min_snr is not None else "n/a"
        err_txt = f"{worst_err:.3e}" if worst_err is not None else "n/a"
        return ValidationResult(
            correctness_passed=False,
            max_abs_err=worst_err,
            rtol=rtol,
            kernel_speedup=None,
            eager_us=None,
            fused_us=None,
            kept=False,
            note=(
                f"PARITY FAILED: min SNR={snr_txt} (< {snr_threshold_db:.0f} dB), "
                f"max_abs_err={err_txt}. bf16 + fp32-accum is not bit-exact; check "
                f"the accumulation dtype and the fused math against the eager op."
            ),
        )
    return _microbench_verdict(runner.microbench(recipe), worst_err, rtol, target_speedup)


def _unmeasured_rejection(note: str, rtol: float) -> ValidationResult:
    """A rejection decided before anything compared the fused path against eager."""
    return ValidationResult(
        correctness_passed=False,
        correctness_measured=False,
        max_abs_err=None,
        rtol=rtol,
        kernel_speedup=None,
        eager_us=None,
        fused_us=None,
        kept=False,
        note=note,
    )


def _microbench_verdict(
    bench: BenchOutcome, worst_err: Optional[float], rtol: float, target_speedup: float
) -> ValidationResult:
    """Gates (c) and (d) for a kernel that passed parity: its microbench speedup and launch count decide ``kept``."""
    # ── gate (c): microbench speedup (eager vs fused) ────────────────────────
    if bench.skipped:
        note = classify_bench_skip(bench.skip_reason)
        return ValidationResult(
            correctness_passed=True,
            max_abs_err=worst_err,
            rtol=rtol,
            kernel_speedup=None,
            eager_us=bench.eager_us,
            fused_us=bench.fused_us,
            kept=False,
            note=f"PARITY OK; MICROBENCH SKIPPED ({bench.skip_reason}): {note}",
        )
    speedup: Optional[float] = None
    if bench.eager_us and bench.fused_us and bench.fused_us > 0:
        speedup = bench.eager_us / bench.fused_us
    implausible = implausible_speedup_reason(speedup) if speedup is not None else ""
    # ── gate (d): launch count ───────────────────────────────────────────────
    regression = launch_regression_reason(bench.eager_launches, bench.fused_launches)
    unverified = launch_gate_unverified(bench.eager_launches, bench.fused_launches)
    kept = speedup is not None and not implausible and not regression and speedup >= target_speedup
    if speedup is None:
        note = (
            "PARITY OK but microbench produced no timing (eager_us/fused_us "
            "missing) — cannot confirm the speedup; treat as not kept."
        )
    elif implausible:
        note = f"PARITY OK but the microbench is not believable: {implausible}"
    elif regression:
        note = (
            f"PARITY OK and {speedup:.3f}x on the chain, but {regression}. "
            f"LESSON: count the launches of both arms before optimizing the "
            f"schedule; a faster chain that costs a launch is not a fusion."
        )
    elif kept:
        note = (
            f"KEPT: parity OK and {speedup:.3f}x >= {target_speedup:.2f}x target "
            f"(eager={bench.eager_us} us, fused={bench.fused_us} us"
            + (
                "; launches UNVERIFIED — the harness reported no count"
                if unverified
                else f"; launches {bench.eager_launches} -> {bench.fused_launches}"
            )
            + ")."
        )
    else:
        note = (
            f"PARITY OK but only {speedup:.3f}x (< {target_speedup:.2f}x target) — "
            f"correct yet not fast enough; try a cheaper fused schedule."
        )
    return ValidationResult(
        correctness_passed=True,
        max_abs_err=worst_err,
        rtol=rtol,
        kernel_speedup=round(speedup, 4) if speedup is not None else None,
        eager_us=bench.eager_us,
        fused_us=bench.fused_us,
        kept=kept,
        note=note,
        eager_launches=bench.eager_launches,
        fused_launches=bench.fused_launches,
    )


@dataclass(eq=False)
class HarnessKernelRunner:
    """Production :class:`KernelValidationRunner` backed by an author-written harness.

    The GPU/import work is genuinely environment-specific, so it is kept at the
    process boundary: this runner executes a kernel-validation harness script (the
    author is instructed to write a parity self-check) in a subprocess and parses a
    single JSON object from its stdout. Unit tests never touch this class — they
    inject a fake runner — so it is deliberately defensive and NEVER raises: a
    missing or malformed harness degrades to a compile failure / skipped microbench
    with an actionable note.

    Harness JSON contract (one object on stdout)::

        {"compiled": bool, "is_triton": bool, "error": str,
         "parity": [{"snr_db": float|null, "max_abs_err": float|null, "label": str}],
         "eager_us": float|null, "fused_us": float|null,
         "eager_launches": int|null, "fused_launches": int|null,
         "eager_kernels": [str], "eager_matches_trace": bool,
         "skipped": bool, "skip_reason": str}
    """

    harness_path: str
    _: KW_ONLY
    workdir: str = "."
    framework_root: str = ""
    gpu: str = "0"
    env_flags: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 1800
    _cache: Optional[dict] = field(default=None, init=False, repr=False)

    def _load(self, recipe: Recipe) -> dict:
        """Run the harness once (cached) and return its parsed JSON dict."""
        if self._cache is not None:
            return self._cache
        result: dict
        if not self.harness_path or not Path(self.harness_path).is_file():
            result = {
                "compiled": False,
                "is_triton": False,
                "error": f"kernel harness not found: {self.harness_path!r}",
            }
            self._cache = result
            return result
        env = dict(os.environ)
        env["HIP_VISIBLE_DEVICES"] = self.gpu
        # The harness is authored inside the framework tree and then published to
        # the run's output directory, so a path the author derived from
        # ``__file__`` points somewhere else by the time it runs here. Name the
        # tree outright rather than leaving it to be inferred.
        if self.framework_root:
            env["FORGE_FUSION_FRAMEWORK_ROOT"] = self.framework_root
        env.update({k: str(v) for k, v in self.env_flags.items()})
        try:
            proc = subprocess.run(
                ["python3", self.harness_path],
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            result = _parse_harness_json(proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired:
            result = {
                "compiled": False,
                "is_triton": False,
                "error": f"kernel harness timed out after {self.timeout_s}s",
            }
        except OSError as e:
            result = {"compiled": False, "is_triton": False, "error": f"could not run kernel harness: {e}"}
        self._cache = result
        return result

    def report(self, recipe: Recipe) -> dict:
        """The harness's raw JSON, for gates that read fields beyond the protocol."""
        return dict(self._load(recipe))

    def compile_check(self, recipe: Recipe) -> CompileOutcome:
        d = self._load(recipe)
        return CompileOutcome(
            ok=bool(d.get("compiled")),
            is_triton=bool(d.get("is_triton")),
            error=str(d.get("error") or ""),
        )

    def parity_samples(self, recipe: Recipe) -> list[ParitySample]:
        d = self._load(recipe)
        out: list[ParitySample] = []
        for p in d.get("parity") or []:
            if not isinstance(p, dict):
                continue
            out.append(
                ParitySample(
                    snr_db=p.get("snr_db"),
                    max_abs_err=p.get("max_abs_err"),
                    label=str(p.get("label") or ""),
                )
            )
        return out

    def microbench(self, recipe: Recipe) -> BenchOutcome:
        d = self._load(recipe)
        return BenchOutcome(
            eager_us=d.get("eager_us"),
            fused_us=d.get("fused_us"),
            skipped=bool(d.get("skipped")),
            skip_reason=str(d.get("skip_reason") or ""),
            eager_launches=launch_count(d.get("eager_launches")),
            fused_launches=launch_count(d.get("fused_launches")),
        )


def _parse_harness_json(stdout: str, stderr: str, returncode: int) -> dict:
    """Best-effort parse of the LAST JSON object printed by the harness.

    On any parse failure the harness output tail becomes a compile error so the
    loop still learns something instead of crashing.
    """
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = _tail((stdout or "") + "\n" + (stderr or ""))
    return {"compiled": False, "is_triton": False, "error": f"harness produced no JSON (rc={returncode}): {tail}"}
