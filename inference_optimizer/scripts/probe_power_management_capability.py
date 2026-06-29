#!/usr/bin/env python3
"""Rung-2 capability probe for the ``power_management`` action.

Verifies the live host has everything the executor needs at runtime:

* ``rocm-smi`` binary on PATH
* Upstream rocm-smi command surface
  (``--showmaxpower`` for the per-GPU manufacturer ceiling and
  ``--setpoweroverdrive`` for the cap setter; rocm-smi's CLI version
  is independent of both the ROCm release and the rocm_smi_lib C
  library version, so the probe checks for flag presence rather
  than guessing a version cutoff)
* ``--showmaxpower --json`` parses into a usable per-GPU ceiling
* Passwordless ``sudo`` for ``rocm-smi`` (only when running as a
  non-root user; the ``rocm/sglang`` / ``rocm/vllm`` Docker images
  Hyperloom typically ships in run as root and don't even install
  ``sudo``, so this whole class of checks is skipped there)
* Every reset command the executor runs in ``finally:`` works
* The clock tables (``--showsclkrange`` / ``--showclkfrq`` /
  ``--showmclkrange``) parse into the GFX determinism ladder + the
  memory-axis capability gate the settle sweep would build — WARNs
  (rather than fails) when the determinism ladder would degenerate to
  no rows, so a degraded sweep is caught BEFORE a multi-hour run

The probe is **safe to run on a healthy node**: it only invokes
probes and the same reset commands the executor itself runs at the
end of every variant. It NEVER sets the power cap to a non-default
value unless you opt-in with ``--exercise-setter``, which re-applies
the manufacturer ceiling (a true no-op against a freshly-booted node
but exercises the same ``--setpoweroverdrive`` path the executor
uses to apply each variant).

Usage::

    python -m inference_optimizer.scripts.probe_power_management_capability
    python -m inference_optimizer.scripts.probe_power_management_capability \\
        --devices 0,1 --exercise-setter --json

Exit codes (machine-readable):

* ``0`` — every check passed; power_management is ready on this host
* ``1`` — one or more checks failed; structured reason in the report
* ``2`` — script-internal error (e.g. JSON encode failure)

This script imports ZERO Hyperloom modules so it can be copied to a
host that doesn't have the full repo installed. The shell-command
shapes are kept in lock-step with ``orchestrator.action_executors.
power_management``; a regression test in
``tests/test_power_management_probe_parity.py`` enforces parity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants — keep in lock-step with orchestrator.action_executors.
# power_management. The parity test enforces this.
# ---------------------------------------------------------------------------
ROCM_SMI_BIN: str = "rocm-smi"
PROBE_TIMEOUT_SEC: int = 30
SET_TIMEOUT_SEC: int = 60

# Reset flags applied between variants and on shutdown. Same order as
# the executor (resetperfdeterminism MUST run before resetclocks
# because perfdeterminism implicitly locks sclk).
_RESET_FLAGS: tuple[str, ...] = (
    "--resetperfdeterminism",
    "--resetclocks",
    "--resetpoweroverdrive",
    "--resetfans",
)


def _sudo_prefix() -> str:
    """Return ``"sudo "`` for non-root callers, ``""`` when already root.

    Mirrors :func:`orchestrator.action_executors.power_management.
    _sudo_prefix` exactly — the parity test enforces equivalence so
    drift here would be caught at unit-test time.
    """
    return "" if os.geteuid() == 0 else "sudo "


def _reset_commands() -> tuple[str, ...]:
    """Live reset-command list, rendered with the current sudo prefix."""
    sp = _sudo_prefix()
    return tuple(
        f"{sp}{ROCM_SMI_BIN} {flag} --autorespond yes"
        for flag in _RESET_FLAGS
    )


# ---------------------------------------------------------------------------
# Clock-table parsers — BYTE-FOR-BYTE mirrors of orchestrator.action_
# executors.power_management. The parity test
# (tests/test_power_management_probe_parity.py) feeds shared fixtures
# through both copies and asserts identical output so this duplicated
# logic can't drift. The standalone probe can't import the executor
# (it imports zero Hyperloom modules so it can be copied to a bare host),
# hence the duplication.
# ---------------------------------------------------------------------------
# Frequency-token regex: matches ``2400Mhz`` / ``2400 MHz`` / ``2.4Ghz``
# (case-insensitive), tolerating a trailing rocm-smi ``*`` current-level
# marker on the surrounding token.
_FREQ_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mhz|ghz)", re.IGNORECASE)

# Per-clock labeled DPM key in ``--showclkfrq --json``: ``mclk[N]`` etc.
_MCLK_IDX_KEY_RE = re.compile(r"^\s*mclk\s*\[\s*(\d+)\s*\]", re.IGNORECASE)


def _freq_tokens_to_mhz(text: str) -> list[int]:
    """Extract every ``<n>Mhz`` / ``<n>Ghz`` frequency in ``text`` as MHz.

    Mirror of the executor helper of the same name (parity-tested).
    """
    out: list[int] = []
    for m in _FREQ_TOKEN_RE.finditer(text or ""):
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            continue
        if m.group(2).lower() == "ghz":
            val *= 1000.0
        mhz = int(round(val))
        if mhz > 0:
            out.append(mhz)
    return out


def _parse_sclkrange_top_mhz(text: str) -> int | None:
    """Parse ``rocm-smi --showsclkrange`` text → top sclk MHz (or None).

    Mirror of the executor helper of the same name (parity-tested). The
    labeled range output looks like
    ``GPU[0]: Valid sclk range: 500Mhz - 2400Mhz``; the top of the clean
    DVFS range is the largest frequency present (the bare DPM-index
    table on some silicon is coarse and scrambled, so the range is the
    PRIMARY top-sclk source).
    """
    freqs = _freq_tokens_to_mhz(text)
    return max(freqs) if freqs else None


def _parse_mclk_levels_from_clkfrq(
    data: dict[str, Any] | None,
) -> list[tuple[int, int]]:
    """Parse selectable memory-clock levels from ``--showclkfrq --json``.

    Mirror of the executor helper of the same name (parity-tested).
    Returns a sorted (by DPM index) list of ``(dpm_index, mhz)`` pairs,
    one per distinct ``mclk[N]`` entry, taking the max MHz seen per
    index across devices.
    """
    if not isinstance(data, dict):
        return []
    by_idx: dict[int, int] = {}
    for _device_key, fields in data.items():
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            m = _MCLK_IDX_KEY_RE.match(str(key))
            if not m:
                continue
            idx = int(m.group(1))
            freqs = _freq_tokens_to_mhz(str(value))
            mhz = max(freqs) if freqs else 0
            if mhz > by_idx.get(idx, -1):
                by_idx[idx] = mhz
    return sorted(by_idx.items())


def _parse_mclkrange(text: str) -> tuple[int, int] | None:
    """Parse ``rocm-smi --showmclkrange`` text → ``(min_mhz, max_mhz)``.

    Mirror of the executor helper of the same name (parity-tested).
    Returns ``None`` when fewer than two frequencies are present.
    """
    freqs = _freq_tokens_to_mhz(text)
    if len(freqs) < 2:
        return None
    return (min(freqs), max(freqs))


# GFX determinism ladder fractions of the probed top sclk — mirror of
# the executor's ``_SETTLE_DETERMINISM_PCTS``. det_100 is always
# emitted; det_95 / det_90 / det_85 are pruned when the workload is
# compute-bound.
_SETTLE_DETERMINISM_PCTS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """Outcome of one probe step."""

    name: str
    status: str           # "pass" | "fail" | "skip" | "warn"
    detail: str = ""
    cmd: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class ProbeReport:
    """Top-level report serialized as JSON when --json is set."""

    steps: list[StepResult] = field(default_factory=list)
    overall: str = "fail"     # "pass" | "fail"
    summary: str = ""

    def add(self, step: StepResult) -> None:
        self.steps.append(step)

    def fail_fast(self) -> bool:
        return any(s.status == "fail" for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "summary": self.summary,
            "steps":   [asdict(s) for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Shell helper — mirrors executor's _run_smi so we test the same path
# ---------------------------------------------------------------------------
def _run(
    cmd: str, *, timeout_sec: int = SET_TIMEOUT_SEC,
) -> tuple[int, str, str]:
    """Run ``cmd`` and return ``(returncode, stdout, stderr)``.

    Uses ``shlex.split`` (same as the executor) so the rendered command
    can be copy-pasted into a shell and behaves identically. Never raises
    on non-zero exit — the caller inspects rc.
    """
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        return 127, "", f"binary not found: {exc}"
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"timed out after {timeout_sec}s: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _tail(s: str, n_lines: int = 3) -> str:
    return "\n".join(s.strip().splitlines()[-n_lines:])


# ---------------------------------------------------------------------------
# Probe steps
# ---------------------------------------------------------------------------
def step_rocm_smi_on_path() -> StepResult:
    path = shutil.which(ROCM_SMI_BIN)
    if path:
        return StepResult(
            name="rocm-smi on PATH", status="pass", detail=path,
            cmd=f"which {ROCM_SMI_BIN}",
        )
    return StepResult(
        name="rocm-smi on PATH", status="fail",
        detail=(
            f"`{ROCM_SMI_BIN}` not found; install rocm-smi (typically "
            "shipped with the ROCm runtime under /opt/rocm/bin/) and "
            "ensure /opt/rocm/bin is in $PATH"
        ),
        cmd=f"which {ROCM_SMI_BIN}",
    )


def step_rocm_smi_version() -> StepResult:
    cmd = f"{ROCM_SMI_BIN} --version"
    rc, out, err = _run(cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    if rc == 0 and out.strip():
        return StepResult(
            name="rocm-smi --version", status="pass",
            detail=_tail(out, 1), cmd=cmd,
        )
    return StepResult(
        name="rocm-smi --version", status="fail",
        detail=f"rc={rc}", cmd=cmd,
        stdout_tail=_tail(out), stderr_tail=_tail(err),
    )


def step_help_mentions(flag: str) -> StepResult:
    """Check the rocm-smi --help text mentions ``flag``.

    The executor targets the upstream command surface only; if either
    ``--showmaxpower`` or ``--setpoweroverdrive`` is missing the action
    will fail at the probe stage. We surface that as an explicit step
    here instead of letting the user discover it at action-execution
    time.
    """
    cmd = f"{ROCM_SMI_BIN} --help"
    rc, out, err = _run(cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    help_text = out + err
    if flag in help_text:
        return StepResult(
            name=f"rocm-smi --help mentions {flag}", status="pass", cmd=cmd,
        )
    return StepResult(
        name=f"rocm-smi --help mentions {flag}", status="fail",
        detail=(
            f"`{flag}` not in --help output; this rocm-smi build lacks "
            "the power-cap command surface that the power_management "
            "executor requires. Upgrade your ROCm install, or — if "
            "this host has migrated to amd-smi — run on a host whose "
            "rocm-smi still exposes --setpoweroverdrive / --showmaxpower."
        ),
        cmd=cmd, stdout_tail=_tail(out), stderr_tail=_tail(err),
    )


def step_showmaxpower_parses(
    devices: tuple[int, ...],
) -> tuple[StepResult, tuple[int, int, int] | None]:
    """Probe + parse ``rocm-smi --showmaxpower --json``.

    Returns ``(step, (min_w, max_w, ref_setter_w))``. ``min_w`` is
    always 0 — the upstream rocm-smi CLI does not expose a hardware
    minimum cap and the executor falls back to the operator/default
    soft floor. ``ref_setter_w`` is the same as ``max_w`` and is used
    by the ``--exercise-setter`` step to invoke
    ``--setpoweroverdrive <ceiling>`` (a true no-op against a
    freshly-booted node).
    """
    cmd = f"{ROCM_SMI_BIN} --showmaxpower --json"
    if devices:
        cmd += "".join(f" -d {int(d)}" for d in devices)
    rc, out, err = _run(cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    if rc != 0:
        return (
            StepResult(
                name="rocm-smi --showmaxpower --json", status="fail",
                detail=f"rc={rc}", cmd=cmd,
                stdout_tail=_tail(out), stderr_tail=_tail(err),
            ),
            None,
        )
    try:
        data = json.loads(out or "{}")
    except json.JSONDecodeError as exc:
        return (
            StepResult(
                name="rocm-smi --showmaxpower --json", status="fail",
                detail=f"json parse failure: {exc}",
                cmd=cmd, stdout_tail=_tail(out), stderr_tail=_tail(err),
            ),
            None,
        )

    # Same reduce logic as orchestrator.action_executors.power_management
    # ._probe_powercap_range: ``min(per_gpu_max)`` for the ceiling.
    maxs: list[int] = []
    for _device_key, fields_dict in data.items():
        if not isinstance(fields_dict, dict):
            continue
        for key, value in fields_dict.items():
            try:
                watts = int(float(value))
            except (TypeError, ValueError):
                continue
            lower = key.lower()
            if "max" in lower and "power" in lower:
                maxs.append(watts)

    if not maxs:
        return (
            StepResult(
                name="rocm-smi --showmaxpower --json", status="fail",
                detail=(
                    "parsed JSON has no 'Max ... Power' field; the "
                    "default-grid path cannot run on this host. "
                    "Power_management will demand an explicit "
                    "task.params.grid"
                ),
                cmd=cmd, stdout_tail=_tail(out), stderr_tail=_tail(err),
            ),
            None,
        )

    max_w = min(maxs)        # most-restrictive ceiling across GPUs
    return (
        StepResult(
            name="rocm-smi --showmaxpower --json", status="pass",
            detail=(
                f"per-node ceiling={max_w}W; rocm-smi does not expose "
                "a hardware minimum, executor will use the operator/"
                "default soft floor"
            ),
            cmd=cmd,
        ),
        (0, max_w, max_w),
    )


def step_root_or_sudo_available() -> StepResult:
    """Verify we can elevate to write rocm-smi knobs.

    Three paths:

    1. Already root (``geteuid() == 0``) — typical in ROCm Docker
       images (``rocm/sglang``, ``rocm/vllm``). Skip sudo entirely.
    2. Non-root user + ``sudo -n true`` succeeds — passwordless sudo
       is configured; can run the executor.
    3. Non-root + sudo fails — return a clean ``fail`` with the
       remediation (add a NOPASSWD sudoers entry for ``rocm-smi``).
    """
    if os.geteuid() == 0:
        return StepResult(
            name="running as root (no sudo needed)",
            status="pass",
            detail=f"uid=0 — executor will invoke {ROCM_SMI_BIN} directly",
            cmd="id -u",
        )

    cmd = "sudo -n true"
    rc, out, err = _run(cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    if rc == 0:
        return StepResult(
            name="sudo -n true (non-interactive auth)",
            status="pass",
            detail=f"non-root (uid={os.geteuid()}) but sudo -n works",
            cmd=cmd,
        )
    return StepResult(
        name="sudo -n true (non-interactive auth)", status="fail",
        detail=(
            "non-root caller and sudo requires a password / TTY; either "
            "(a) run the executor as root (the typical Docker setup) or "
            "(b) add a NOPASSWD sudoers entry, e.g. "
            f"`myuser ALL=(ALL) NOPASSWD: /opt/rocm/bin/{ROCM_SMI_BIN}`"
        ),
        cmd=cmd, stdout_tail=_tail(out), stderr_tail=_tail(err),
    )


def step_elevated_rocm_smi() -> StepResult:
    """Verify the elevation path invokes ``rocm-smi`` correctly.

    Catches the common gotcha where the sudoers NOPASSWD entry is
    written against the wrong binary path (e.g. ``/usr/bin/rocm-smi``
    vs ``/opt/rocm/bin/rocm-smi``). When we're already root we just
    re-run a no-prefix version probe so the step still has something
    to report; for non-root callers we prepend ``sudo -n``.
    """
    sudo_present = bool(_sudo_prefix().strip())
    if sudo_present:
        cmd = f"sudo -n {ROCM_SMI_BIN} --version"
        label = f"sudo -n {ROCM_SMI_BIN} --version"
    else:
        cmd = f"{ROCM_SMI_BIN} --version"
        label = f"{ROCM_SMI_BIN} --version (as root)"
    rc, out, err = _run(cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    if rc == 0:
        return StepResult(
            name=label, status="pass", detail=_tail(out, 1), cmd=cmd,
        )
    extra = (
        " — check that the NOPASSWD sudoers entry resolves to the SAME "
        f"`{ROCM_SMI_BIN}` binary that `which {ROCM_SMI_BIN}` found above"
        if sudo_present else ""
    )
    return StepResult(
        name=label, status="fail",
        detail=f"rc={rc}{extra}", cmd=cmd,
        stdout_tail=_tail(out), stderr_tail=_tail(err),
    )


def step_reset_commands_work() -> list[StepResult]:
    """Run every reset command the executor invokes in its finally block.

    These are no-ops on a freshly-booted node (nothing to reset) but
    they validate the elevation path + command syntax for each flag. If
    ANY reset fails the executor can leave the GPU in a non-default
    state after a crash, which is the worst-case operational failure
    mode.

    Non-root callers get ``-n`` added to every sudo invocation so this
    probe step is strictly non-interactive. Root callers run the
    commands directly.
    """
    out: list[StepResult] = []
    sudo_present = bool(_sudo_prefix().strip())
    for cmd in _reset_commands():
        probe_cmd = (
            cmd.replace("sudo ", "sudo -n ", 1) if sudo_present else cmd
        )
        rc, stdout, stderr = _run(probe_cmd, timeout_sec=SET_TIMEOUT_SEC)
        # Stable label keyed off the reset flag, regardless of prefix.
        flag = next(
            (tok for tok in probe_cmd.split() if tok.startswith("--reset")),
            "--reset?",
        )
        name = f"{ROCM_SMI_BIN} {flag}"
        if rc == 0:
            out.append(StepResult(name=name, status="pass", cmd=probe_cmd))
        else:
            out.append(StepResult(
                name=name, status="fail",
                detail=f"rc={rc}", cmd=probe_cmd,
                stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
            ))
    return out


def step_setter_noop(
    cur_w: int, devices: tuple[int, ...],
) -> StepResult:
    """OPTIONAL: re-apply the manufacturer ceiling to validate the setter.

    Calling ``--setpoweroverdrive <ceiling_w>`` is a true no-op against
    a freshly-booted node (the cap is already at the ceiling) but
    exercises the same shell call the executor uses to apply each
    variant. Only run when the operator passes ``--exercise-setter``
    because some sites prefer to keep the setter path completely cold
    until the action itself runs.
    """
    if cur_w <= 0:
        return StepResult(
            name=f"{ROCM_SMI_BIN} --setpoweroverdrive <ceiling> (no-op)",
            status="skip",
            detail="no ceiling parsed; skipping setter exercise",
        )
    device_args = "".join(f" -d {int(d)}" for d in devices)
    sudo_present = bool(_sudo_prefix().strip())
    prefix = "sudo -n " if sudo_present else ""
    cmd = (
        f"{prefix}{ROCM_SMI_BIN}{device_args} --setpoweroverdrive {cur_w} "
        "--autorespond yes"
    )
    rc, out, err = _run(cmd, timeout_sec=SET_TIMEOUT_SEC)
    label = f"{ROCM_SMI_BIN} --setpoweroverdrive {cur_w} (no-op)"
    if rc == 0:
        return StepResult(
            name=label, status="pass", cmd=cmd, detail=_tail(out, 1),
        )
    return StepResult(
        name=label, status="fail",
        detail=(
            f"rc={rc}; the executor's variant-apply path will fail the "
            "same way at runtime — fix the elevation / GPU driver state "
            "before promoting power_management on this host"
        ),
        cmd=cmd, stdout_tail=_tail(out), stderr_tail=_tail(err),
    )


def step_clock_tables_parse(devices: tuple[int, ...]) -> StepResult:
    """Report the GFX/memory ladder the settle sweep would build here.

    Runs the same clock probes the executor does — ``--showsclkrange``
    (primary top-sclk source) with ``--showclkfrq --json`` as the
    fallback, plus ``--showclkfrq`` / ``--showmclkrange`` for the
    memory-axis capability gate — and applies the SAME parsers
    (``_parse_sclkrange_top_mhz`` / ``_parse_mclk_levels_from_clkfrq`` /
    ``_parse_mclkrange``). It then reports the ladder that WOULD be built:

      * the GFX determinism ladder (det_100 always; det_95 / det_90 /
        det_85 unless the workload is compute-bound — the probe can't know
        the roofline, so it reports the full ladder);
      * the memory-axis status (capable iff >= 2 selectable mclk levels).

    Returns ``status='warn'`` (NOT ``fail`` — the action can still run
    auto/high-only) when the determinism ladder would degenerate to 0
    rows because no top sclk parsed, so a degraded sweep is caught HERE
    instead of after a multi-hour run. A single-mclk-level box (the
    memory axis skipped-with-reason) is the gate firing as intended and
    stays ``pass``.
    """
    # --- top sclk: --showsclkrange primary, --showclkfrq max fallback.
    range_cmd = f"{ROCM_SMI_BIN} --showsclkrange"
    if devices:
        range_cmd += "".join(f" -d {int(d)}" for d in devices)
    _rc, range_out, _err = _run(range_cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    top_sclk = _parse_sclkrange_top_mhz(range_out or "")
    top_sclk_source = "--showsclkrange"

    clkfrq_cmd = f"{ROCM_SMI_BIN} --showclkfrq --json"
    if devices:
        clkfrq_cmd += "".join(f" -d {int(d)}" for d in devices)
    _rc2, clkfrq_out, _err2 = _run(clkfrq_cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    try:
        clkfrq_data = json.loads(clkfrq_out or "{}")
    except json.JSONDecodeError:
        clkfrq_data = {}
    if not top_sclk:
        # Fallback: the maximum sclk frequency in the bare DPM table.
        candidates: list[int] = []
        if isinstance(clkfrq_data, dict):
            for _dk, fields in clkfrq_data.items():
                if not isinstance(fields, dict):
                    continue
                for key, value in fields.items():
                    lower = str(key).lower()
                    if "sclk" not in lower and "gpu clock" not in lower:
                        continue
                    candidates.extend(_freq_tokens_to_mhz(str(value)))
        if candidates:
            top_sclk = max(candidates)
            top_sclk_source = "--showclkfrq (fallback)"

    # --- determinism ladder rows that would be built.
    det_rows: list[str] = []
    if top_sclk and top_sclk > 0:
        seen: set[int] = set()
        for pct in _SETTLE_DETERMINISM_PCTS:
            mhz = int(round(top_sclk * pct))
            if mhz <= 0 or mhz in seen:
                continue
            seen.add(mhz)
            det_rows.append(f"det_{int(round(pct * 100))}={mhz}MHz")

    # --- memory axis capability.
    mclk_levels = _parse_mclk_levels_from_clkfrq(clkfrq_data)
    mclk_count = len(mclk_levels)
    mrange_cmd = f"{ROCM_SMI_BIN} --showmclkrange"
    if devices:
        mrange_cmd += "".join(f" -d {int(d)}" for d in devices)
    _rc3, mrange_out, _err3 = _run(mrange_cmd, timeout_sec=PROBE_TIMEOUT_SEC)
    mclk_rng = _parse_mclkrange(mrange_out or "")
    if mclk_count >= 2:
        # Top mclk is covered by high/auto; the axis steps the rest.
        mem_status = (
            f"capable: {mclk_count} levels {[m for _i, m in mclk_levels]} "
            f"→ would emit {mclk_count - 1} mclk row(s) "
            "(NOT memory-bound only)"
        )
    else:
        mem_status = (
            f"skipped: only {mclk_count} selectable mclk level(s) "
            f"(need >= 2); range={mclk_rng}"
        )

    detail = (
        f"top sclk={top_sclk}MHz via {top_sclk_source}; "
        f"GFX determinism ladder=[auto_baseline, high, "
        f"{', '.join(det_rows) if det_rows else '<none>'}]"
        f" (det_95/det_90/det_85 pruned when compute-bound); "
        f"memory axis: {mem_status}"
    )

    if not det_rows:
        return StepResult(
            name="clock tables → settle ladder", status="warn",
            detail=(
                "no top sclk parsed from --showsclkrange or --showclkfrq "
                "— the GFX determinism ladder would produce 0 rows and the "
                "settle sweep degrades to auto/high-only. " + detail
            ),
            cmd=f"{range_cmd}; {clkfrq_cmd}",
            stdout_tail=_tail(range_out),
        )
    return StepResult(
        name="clock tables → settle ladder", status="pass",
        detail=detail, cmd=f"{range_cmd}; {clkfrq_cmd}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _parse_devices(raw: str) -> tuple[int, ...]:
    if not raw or not raw.strip():
        return ()
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return tuple(out)


def run_probe(
    *, devices: tuple[int, ...], exercise_setter: bool,
) -> ProbeReport:
    rep = ProbeReport()

    # Phase 1: rocm-smi presence + version
    rep.add(step_rocm_smi_on_path())
    if rep.fail_fast():
        return _finalise(rep)
    rep.add(step_rocm_smi_version())

    # Phase 2: command surface
    rep.add(step_help_mentions("--showmaxpower"))
    rep.add(step_help_mentions("--setpoweroverdrive"))
    if rep.fail_fast():
        return _finalise(rep)

    # Phase 3: probe parses
    show_step, parsed = step_showmaxpower_parses(devices)
    rep.add(show_step)
    if rep.fail_fast() or parsed is None:
        return _finalise(rep)
    _min_w, _max_w, cur_w = parsed

    # Phase 4: root-or-sudo elevation path
    rep.add(step_root_or_sudo_available())
    if rep.fail_fast():
        return _finalise(rep)
    rep.add(step_elevated_rocm_smi())
    if rep.fail_fast():
        return _finalise(rep)

    # Phase 5: every reset the executor runs
    for s in step_reset_commands_work():
        rep.add(s)

    # Phase 6: report the GFX/memory ladder the settle sweep would build
    # (warn — not fail — if it would degenerate to no determinism ladder).
    rep.add(step_clock_tables_parse(devices))

    # Phase 7 (opt-in): exercise the setter via a no-op cap re-apply
    if exercise_setter:
        rep.add(step_setter_noop(cur_w, devices))

    return _finalise(rep)


def _finalise(rep: ProbeReport) -> ProbeReport:
    if any(s.status == "fail" for s in rep.steps):
        rep.overall = "fail"
        first_fail = next(s for s in rep.steps if s.status == "fail")
        rep.summary = (
            f"FAIL: {first_fail.name} — {first_fail.detail or first_fail.stderr_tail}"
        )
    else:
        rep.overall = "pass"
        warns = [s for s in rep.steps if s.status == "warn"]
        if warns:
            rep.summary = (
                "PASS with warnings — power_management can run, but: "
                + "; ".join(f"{w.name}: {w.detail}" for w in warns)
            )
        else:
            rep.summary = (
                "All checks passed — power_management is ready to run "
                "on this host"
            )
    return rep


def _format_human(rep: ProbeReport) -> str:
    lines: list[str] = [
        "Rung-2 capability probe — power_management",
        "=" * 60,
    ]
    total = len(rep.steps)
    width = max((len(s.name) for s in rep.steps), default=0)
    for idx, s in enumerate(rep.steps, start=1):
        marker = {
            "pass": "PASS", "fail": "FAIL", "skip": "SKIP", "warn": "WARN",
        }[s.status]
        line = f"[{idx:>2}/{total}] {s.name:<{width}}  {marker}"
        if s.detail:
            line += f"   ({s.detail})"
        lines.append(line)
        if s.status in ("fail", "warn") and s.stderr_tail:
            for err_line in s.stderr_tail.splitlines():
                lines.append(f"          STDERR  {err_line}")
        if s.status in ("fail", "warn") and s.stdout_tail:
            for out_line in s.stdout_tail.splitlines():
                lines.append(f"          STDOUT  {out_line}")
    lines.append("=" * 60)
    lines.append(rep.summary)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_power_management_capability",
        description=(
            "Verify the live host can run the power_management action. "
            "Read-only by default; pass --exercise-setter to re-apply "
            "the manufacturer ceiling as a no-op setter validation."
        ),
    )
    parser.add_argument(
        "--devices", default="",
        help=(
            "Optional comma-separated GPU indices to probe (matches the "
            "executor's task.params.devices). Default: all GPUs."
        ),
    )
    parser.add_argument(
        "--exercise-setter", action="store_true",
        help=(
            "Also re-apply the manufacturer ceiling via sudo + "
            "--setpoweroverdrive <ceiling>. True no-op against a "
            "freshly-booted node, but exercises the same shell path "
            "the executor uses to apply each variant."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the full report as JSON on stdout instead of a human table.",
    )
    args = parser.parse_args(argv)

    try:
        devices = _parse_devices(args.devices)
    except ValueError as exc:
        print(f"error: --devices must be comma-separated ints, got {exc}",
              file=sys.stderr)
        return 2

    rep = run_probe(devices=devices, exercise_setter=args.exercise_setter)

    if args.json:
        try:
            print(json.dumps(rep.to_dict(), indent=2))
        except (TypeError, ValueError) as exc:
            print(f"error: failed to serialise probe report: {exc}",
                  file=sys.stderr)
            return 2
    else:
        print(_format_human(rep))

    return 0 if rep.overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
