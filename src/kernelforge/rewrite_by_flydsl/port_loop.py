# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PORT phase — translate the source kernel into a CORRECT FlyDSL kernel."""

from __future__ import annotations

import ast
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from kernelforge.config import Config
from kernelforge.evaluation.correctness.pipeline import run_validation_pipeline
from kernelforge.rewrite_by_flydsl.prompts import build_port_program_md
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec
from kernelforge.tracker import UsageAccumulator

log = logging.getLogger(__name__)


# The canonical correctness suite's per-stage ceiling; PORT clips it to the time left before the cutoff.
_VALIDATE_STAGE_TIMEOUT_SEC = 1800
_RESERVE_REACHED = "PORT stopped at the 20-minute finalization reserve"


@dataclass(frozen=True)
class PortRequest:
    """One PORT campaign: how many sessions may run, their permission mode, the cutoff and the KB references."""

    max_attempts: int = 3
    permission_mode: str | None = None
    stop_at_unix: float | None = None
    pre_task_context: str = ""


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


class _ReserveReached(Exception):
    """PORT stopped at its finalization reserve; ``args`` are the attempts it made and the error tail it reports."""


def _seconds_left(stop_at_unix: float | None, attempts: int, tail: str = _RESERVE_REACHED) -> float | None:
    """Seconds before the PORT cutoff, or None without one; raises _ReserveReached once it has passed."""
    remaining = stop_at_unix - time.time() if stop_at_unix and stop_at_unix > 0 else None
    if remaining is not None and remaining <= 0:
        raise _ReserveReached(attempts, tail)
    return remaining


# Triton ships alongside FlyDSL in every rewrite environment, so reimplementing the op in it is a cheat available
# whatever the source was written in.
_BANNED_PORT_MODULES: frozenset[str] = frozenset({"triton"})


def check_flydsl_port(spec: RewriteSpec) -> str:
    """Reject a port that is not a genuine FlyDSL rewrite. Returns \"\" if OK."""
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
            # `from . import softmax` (relative; node.module is None) and `from pkg import softmax` both BIND the name
            # `softmax` — catch the source module imported as a name, not just as a module root.
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
    return _dynamic_import_violation(tree, spec.flydsl_kernel_name, forbidden)


def _dynamic_import_violation(tree: ast.AST, kernel_name: str, forbidden: frozenset[str]) -> str:
    """Reject a dynamic import, which evades the static import scan. Returns \"\" if there is none."""
    for node in ast.walk(tree):
        fn = node.func if isinstance(node, ast.Call) else None
        is_dunder_import = isinstance(fn, ast.Name) and fn.id == "__import__"
        is_import_module = isinstance(fn, ast.Attribute) and fn.attr == "import_module"
        if not (is_dunder_import or is_import_module):
            continue
        arg = node.args[0] if node.args else None
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            return (
                f"{kernel_name} uses a dynamic import with a non-literal "
                "target — a FlyDSL port must import FlyDSL statically and not hide what "
                "it loads. Remove the dynamic import."
            )
        if arg.value.split(".")[0] in forbidden:
            return (
                f"{kernel_name} dynamically imports `{arg.value}` — the "
                "port MUST NOT pull in the source language / source module by any "
                "means. Compute the result in FlyDSL only."
            )
    return ""


def _port_agent(
    spec: RewriteSpec, driver_path: str, config: Config, request: PortRequest, usage: UsageAccumulator | None
):
    """The implementer session every PORT attempt runs: only the FlyDSL kernel is editable, correctness is the gate."""
    from kernelforge.loop.api import GateMeasurement, ImplementerSetup, ImplementerTask, SessionOptions, make_agent_fn

    return make_agent_fn(
        config,
        ImplementerSetup(
            ImplementerTask(
                program_md=build_port_program_md(spec, driver_path),
                kernel_backend_name="flydsl",
                pre_task_context=request.pre_task_context,
                # Only the FlyDSL kernel is editable; the kernel it ports FROM is the driver's live oracle and baseline.
                source_files=[spec.flydsl_kernel],
                target_functions=[spec.builder_symbol],
                extra_protected_globs=[spec.source_kernel_name],
                extra_protected_paths=[spec.source_kernel],
            ),
            # PORT is correctness-only: no perf requirement (a correct port is the goal; OPTIMIZE tunes speed later).
            GateMeasurement(
                driver_path, spec.snr_threshold, stage_timeout_sec=_VALIDATE_STAGE_TIMEOUT_SEC, correctness_only=True
            ),
            SessionOptions(permission_mode=request.permission_mode, insession_gate=True),
        ),
        usage=usage,
    )


async def _validate_port(spec: RewriteSpec, driver_path: str, stop_at_unix: float | None, attempt: int):
    """Canonical acceptance: the driver's complete correctness suite, run within the time left before the cutoff."""
    remaining = _seconds_left(stop_at_unix, attempt, "PORT stopped before validation at the finalization reserve")
    stage_timeout = (
        _VALIDATE_STAGE_TIMEOUT_SEC if remaining is None else max(1, min(_VALIDATE_STAGE_TIMEOUT_SEC, int(remaining)))
    )
    validation = run_validation_pipeline(
        driver_script=driver_path, snr_threshold=spec.snr_threshold, timeout_per_stage=stage_timeout
    )
    try:
        return await asyncio.wait_for(validation, timeout=remaining)
    except asyncio.TimeoutError:
        raise _ReserveReached(attempt, "PORT validation reached the 20-minute finalization reserve") from None


async def run_port_loop(
    spec: RewriteSpec,
    driver_path: str,
    config: Config,
    request: PortRequest = PortRequest(),
    *,
    usage: UsageAccumulator | None = None,
) -> PortResult:
    """Run the correctness-only port loop; return whether a correct port emerged."""
    history = ""
    tail = ""

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
        except (RuntimeError, OSError) as error:
            return reason + "; protected snapshot restore failed: " + f"{type(error).__name__}: {error}"
        return reason

    try:
        _seconds_left(request.stop_at_unix, attempts=0)
        agent_fn = _port_agent(spec, driver_path, config, request, usage)
        for attempt in range(1, request.max_attempts + 1):
            remaining = _seconds_left(request.stop_at_unix, attempts=attempt - 1)
            log.info("port attempt %d/%d for %s", attempt, request.max_attempts, spec.op_name)
            sink: dict = {}
            try:
                await asyncio.wait_for(agent_fn(spec.flydsl_kernel, history, session_sink=sink), timeout=remaining)
            except asyncio.TimeoutError:
                log.warning("port attempt %d reached the finalization reserve", attempt)
                if integrity_error := restore_integrity(sink):
                    log.warning("port attempt %d restored protected inputs after timeout: %s", attempt, integrity_error)
                raise _ReserveReached(attempt, _RESERVE_REACHED) from None
            except Exception as e:  # isolation boundary: one PORT agent attempt
                log.warning("port attempt %d: agent session error: %s", attempt, e, exc_info=True)
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
                log.info("port attempt %d rejected before validation (integrity): %s", attempt, integrity_error)
                history = (
                    "Your PORT attempt changed protected driver/source-oracle state "
                    "and was REJECTED before validation. The protected files were "
                    f"restored. Modify only {spec.flydsl_kernel_name}.\n"
                    f"{integrity_error}"
                )
                continue

            # FlyDSL-only gate (security/intent): a numerically-correct kernel that cheats by re-calling the source
            # (or reimplementing in Triton) is NOT a valid rewrite.
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

            report = await _validate_port(spec, driver_path, request.stop_at_unix, attempt)
            if report.all_passed:
                snr = report.results[-1].snr_db
                log.info("port succeeded on attempt %d (SNR=%s)", attempt, snr)
                return PortResult(ok=True, attempts=attempt, snr_db=snr)

            tail = _validation_error_tail(report)
            log.info("port attempt %d not yet correct: %s", attempt, report.summary().splitlines()[-1])
            history = (
                "The FlyDSL port is NOT correct yet. Fix it to match the source "
                "kernel's numerics. Latest validation failure:\n" + tail
            )
    except _ReserveReached as reached:
        attempts, error_tail = reached.args
        return PortResult(ok=False, attempts=attempts, error_tail=error_tail)

    return PortResult(ok=False, attempts=request.max_attempts, error_tail=tail)
