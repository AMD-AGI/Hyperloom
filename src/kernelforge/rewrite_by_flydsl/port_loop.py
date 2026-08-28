# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PORT phase — translate the source kernel into a CORRECT FlyDSL kernel.

Reuses the forge building blocks with a correctness-ONLY gate:
  * ``make_agent_fn(insession_gate=True, correctness_only=True, ...)`` runs the
    in-session Stop gate in correctness-only mode (the perf benchmark is skipped
    entirely), so each session drives edit -> build -> test -> fix until the FlyDSL
    output matches the source oracle (SNR gate).
  * after each session the driver's complete correctness suite confirms the
    port; on failure the compact error is fed into the
    next attempt (mirrors the forge experience-ledger pattern).

The source kernel (the port's reference AND the live oracle) and the driver are
protected from edits; only the FlyDSL kernel file is editable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from kernelforge.config import Config
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.rewrite_by_flydsl.prompts import build_port_program_md
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

log = logging.getLogger(__name__)


@dataclass
class PortResult:
    ok: bool
    attempts: int
    snr_db: float | None = None
    error_tail: str = ""


def _validation_error_tail(report) -> str:
    """Compact failure signal from a validation report for the next attempt."""
    if report.all_passed:
        return ""
    tail = report.failed_output or report.summary()
    return tail[-1500:]


# Triton ships alongside FlyDSL in every rewrite environment, so reimplementing
# the op in it is a cheat available whatever the source was written in.
_BANNED_PORT_MODULES: frozenset[str] = frozenset({"triton"})


def check_flydsl_port(spec: RewriteSpec) -> str:
    """Reject a port that is not a genuine FlyDSL rewrite. Returns "" if OK.

    Numeric correctness alone cannot tell a real FlyDSL port from one that cheats
    by importing the source module and re-calling the original kernel, or by
    reimplementing the op in another GPU DSL. The rule is the same for every
    source language, so this gate takes no language argument. Returns a compact
    human-readable reason on violation (fed back to the next attempt), or "" when
    the port is acceptable.
    """
    import ast

    path = spec.flydsl_kernel
    src_stem = Path(spec.source_kernel).stem  # e.g. "softmax" for softmax.py
    forbidden = _BANNED_PORT_MODULES | {src_stem}  # + the source module (re-call cheat)
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError) as e:
        return f"could not parse the FlyDSL kernel {spec.flydsl_kernel_name}: {e}"

    imported_roots: set[str] = set()  # top-level package of every static import
    imported_names: set[str] = set()  # bare names bound by `from X import name`
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
            # `from . import softmax` (relative; node.module is None) and
            # `from pkg import softmax` both BIND the name `softmax` — catch the
            # source module imported as a name, not just as a module root.
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])

    if "flydsl" not in imported_roots:
        return (
            f"{spec.flydsl_kernel_name} does not import `flydsl` — the port MUST be "
            "implemented in FlyDSL (import flydsl...). A kernel that does not use "
            "FlyDSL is not a valid rewrite."
        )
    banned = _BANNED_PORT_MODULES & imported_roots
    if banned:
        return (
            f"{spec.flydsl_kernel_name} imports {sorted(banned)} — the port MUST NOT "
            "compute through another GPU DSL or reach back into the source "
            "language. Reimplement the op in FlyDSL only."
        )
    if src_stem in imported_roots or src_stem in imported_names:
        return (
            f"{spec.flydsl_kernel_name} imports the source module `{src_stem}` — the "
            "port MUST NOT call the original kernel as its implementation (that "
            "defeats the rewrite). Compute the result in FlyDSL only."
        )

    # Dynamic imports evade the static import scan above. A genuine FlyDSL port has
    # no need for `importlib.import_module(...)` / `__import__(...)`; treat one that
    # names a forbidden module as a cheat, and one whose target cannot be resolved
    # statically as unverifiable (reject rather than trust it).
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_dunder_import = isinstance(fn, ast.Name) and fn.id == "__import__"
        is_import_module = isinstance(fn, ast.Attribute) and fn.attr == "import_module"
        if not (is_dunder_import or is_import_module):
            continue
        arg = node.args[0] if node.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            target = arg.value.split(".")[0]
            if target in forbidden:
                return (
                    f"{spec.flydsl_kernel_name} dynamically imports `{arg.value}` — the "
                    "port MUST NOT pull in the source language / source module by any "
                    "means. Compute the result in FlyDSL only."
                )
        else:
            return (
                f"{spec.flydsl_kernel_name} uses a dynamic import with a non-literal "
                "target — a FlyDSL port must import FlyDSL statically and not hide what "
                "it loads. Remove the dynamic import."
            )
    return ""


async def run_port_loop(
    spec: RewriteSpec,
    driver_path: str,
    config: Config,
    *,
    kernel_backend: str = "flydsl",
    max_attempts: int = 3,
    permission_mode: str | None = None,
    validate_stage_timeout_sec: int = 1800,
    usage=None,
    stop_at_unix: float | None = None,
    pre_task_context: str = "",
) -> PortResult:
    """Run the correctness-only port loop; return whether a correct port emerged."""
    from kernelforge.orchestrator.agent import make_agent_fn

    if stop_at_unix and stop_at_unix > 0 and time.time() >= stop_at_unix:
        return PortResult(
            ok=False,
            attempts=0,
            error_tail="PORT stopped at the 20-minute finalization reserve",
        )

    program_md = build_port_program_md(spec, driver_path)

    agent_fn = make_agent_fn(
        config=config,
        program_md=program_md,
        kernel_backend_name=kernel_backend,
        pre_task_context=pre_task_context,
        insession_gate=True,
        # PORT is correctness-only: the in-session gate must NOT impose a perf
        # requirement (a correct FlyDSL port is the goal; OPTIMIZE tunes speed later).
        correctness_only=True,
        driver_script=driver_path,
        snr_threshold=spec.snr_threshold,
        validation_timeout_sec=validate_stage_timeout_sec,
        permission_mode=permission_mode,
        # Single-file target: only the FlyDSL kernel is editable.
        source_files=[spec.flydsl_kernel],
        target_functions=[spec.builder_symbol],
        # Protect the source kernel we port FROM — the driver imports it as the live
        # correctness oracle + baseline, so it must not be editable during PORT.
        # Exact absolute path (same tier as the driver); the basename glob stays as a
        # fallback for edits the hook can only see as an unresolved relative path.
        extra_protected_paths=[spec.source_kernel],
        extra_protected_globs=[spec.source_kernel_name],
        usage=usage,
    )

    history = ""
    last_report = None

    def restore_integrity(session_sink: dict) -> str:
        """Restore protected PORT inputs and return the rejection detail."""

        if session_sink.get("integrity_violation") is not True:
            return ""
        reason = str(session_sink.get("integrity_reason") or "protected PORT driver/source state changed")
        restore = session_sink.get("integrity_restore")
        if not callable(restore):
            return reason + "; protected snapshot restore callback unavailable"
        try:
            restore()
        except Exception as error:  # noqa: BLE001 - report without validating
            return reason + "; protected snapshot restore failed: " + f"{type(error).__name__}: {error}"
        return reason

    for attempt in range(1, max_attempts + 1):
        remaining = stop_at_unix - time.time() if stop_at_unix and stop_at_unix > 0 else None
        if remaining is not None and remaining <= 0:
            return PortResult(
                ok=False,
                attempts=attempt - 1,
                error_tail="PORT stopped at the 20-minute finalization reserve",
            )
        log.info("port attempt %d/%d for %s", attempt, max_attempts, spec.op_name)
        sink: dict = {}
        try:
            session = agent_fn(spec.flydsl_kernel, history, session_sink=sink)
            if remaining is None:
                await session
            else:
                await asyncio.wait_for(session, timeout=remaining)
        except asyncio.TimeoutError:
            log.warning("port attempt %d reached the finalization reserve", attempt)
            integrity_error = restore_integrity(sink)
            if integrity_error:
                log.warning(
                    "port attempt %d restored protected inputs after timeout: %s",
                    attempt,
                    integrity_error,
                )
            return PortResult(
                ok=False,
                attempts=attempt,
                error_tail="PORT stopped at the 20-minute finalization reserve",
            )
        except Exception as e:  # noqa: BLE001 - a session crash is one failed attempt
            log.warning("port attempt %d: agent session error: %s", attempt, e)
            integrity_error = restore_integrity(sink)
            history = f"Previous attempt crashed the session: {e}\n"
            if integrity_error:
                history += (
                    "The attempt also violated protected PORT input integrity; "
                    f"the driver/source oracle were restored: {integrity_error}\n"
                )
            continue

        integrity_error = restore_integrity(sink)
        if integrity_error:
            log.info(
                "port attempt %d rejected before validation (integrity): %s",
                attempt,
                integrity_error,
            )
            history = (
                "Your PORT attempt changed protected driver/source-oracle state "
                "and was REJECTED before validation. The protected files were "
                f"restored. Modify only {spec.flydsl_kernel_name}.\n"
                f"{integrity_error}"
            )
            continue

        # FlyDSL-only gate (security/intent): a numerically-correct kernel that
        # cheats by re-calling the source (or reimplementing in Triton) is NOT a
        # valid rewrite. Enforce this statically BEFORE the (more expensive)
        # correctness pipeline so a cheat is caught + fed back immediately.
        flydsl_violation = check_flydsl_port(spec)
        if flydsl_violation:
            log.info("port attempt %d rejected (not FlyDSL): %s", attempt, flydsl_violation)
            history = (
                "Your port is NOT a valid FlyDSL rewrite and was REJECTED before "
                "correctness was even checked:\n" + flydsl_violation + "\n"
                "Reimplement the operator in FlyDSL (import flydsl...) in "
                f"{spec.flydsl_kernel_name}; do NOT import/call the source kernel."
            )
            continue

        # Canonical acceptance: the driver's complete correctness suite.
        remaining = stop_at_unix - time.time() if stop_at_unix and stop_at_unix > 0 else None
        if remaining is not None and remaining <= 0:
            return PortResult(
                ok=False,
                attempts=attempt,
                error_tail="PORT stopped before validation at the finalization reserve",
            )
        validation_timeout = (
            validate_stage_timeout_sec if remaining is None else max(1, min(validate_stage_timeout_sec, int(remaining)))
        )
        validation = run_validation_pipeline(
            driver_script=driver_path,
            snr_threshold=spec.snr_threshold,
            timeout_per_stage=validation_timeout,
        )
        try:
            report = await validation if remaining is None else await asyncio.wait_for(validation, timeout=remaining)
        except asyncio.TimeoutError:
            return PortResult(
                ok=False,
                attempts=attempt,
                error_tail="PORT validation reached the 20-minute finalization reserve",
            )
        last_report = report
        if report.all_passed:
            snr = report.results[-1].snr_db if report.results else None
            log.info("port succeeded on attempt %d (SNR=%s)", attempt, snr)
            return PortResult(ok=True, attempts=attempt, snr_db=snr)

        tail = _validation_error_tail(report)
        log.info(
            "port attempt %d not yet correct: %s",
            attempt,
            report.summary().splitlines()[-1] if report.summary() else "",
        )
        history = (
            "The FlyDSL port is NOT correct yet. Fix it to match the source "
            "kernel's numerics. Latest validation failure:\n" + tail
        )

    return PortResult(
        ok=False,
        attempts=max_attempts,
        error_tail=_validation_error_tail(last_report) if last_report else "",
    )
