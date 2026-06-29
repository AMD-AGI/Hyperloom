"""Real ``power_management`` ActionRunner — host-level rocm-smi sweep.

Mirrors the structure of :class:`SweepExecutor` / :class:`ExploreExecutor`,
but the "knob" is applied *outside* the Magpie subprocess via ``rocm-smi``
before each variant's rebench. Each variant runs as its own one-row
:func:`run_grid` invocation so we inherit the standard workload-contract
materialization, leak-salvage, cold-start timeout, and per-variant
workspace plumbing.

Lifecycle for one task invocation:

    1. probe ``rocm-smi --showmaxpower --json`` to discover the
       hardware ceiling, ``--showsclkrange`` for the top engine clock
       (with the ``--showclkfrq`` max as a fallback), and
       ``--showclkfrq`` / ``--showmclkrange`` for the selectable
       memory-clock levels (the memory-axis capability gate). All clock
       probes are best-effort. The ROCm 4.x-vintage ``rocm-smi`` CLI
       does not expose a *hardware-minimum* cap; the executor falls
       back to the operator/default soft floor for the lower bound.
    2. build the variant list:
         * LLM-supplied ``params.grid`` wins (operator owns shape) and
           is benched as-is.
         * otherwise the Coordinator-internal settle sweep builds a
           single **roofline-routed** grid (see
           ``actions/power_management.md``). It is throughput-only with
           the power cap and fan pinned to MAX on every row, so the
           kernel-climb state and the sweep baseline are the same
           hardware state and power is never a confound. GFX is tuned
           via ``--setperfdeterminism`` ONLY — there is no GFX
           DPM-index pin (the DPM table is coarse and scrambled). Rows:
             - ``auto_baseline`` (incumbent, N reps → median): perflevel
               auto + cap-max + fan-max. The reference the winner gate
               measures against. Always.
             - ``high``: perflevel high + cap-max + fan-max. Always.
             - ``det_100`` (anchor): ``--setperfdeterminism <top>`` +
               cap-max + fan-max. Always (when a top sclk was probed).
             - ``det_95`` / ``det_90`` / ``det_85``:
               ``--setperfdeterminism <pct·top>`` — emitted UNLESS the
               workload is compute-bound.
             - ``mclk_*``: GFX pinned high (manual + sclk at its
               highest-frequency DPM index) while memory is stepped —
               emitted only when NOT memory-bound AND there are >= 2
               selectable mclk levels.
    3. for each row: bench ``auto_baseline`` N times (median), bench the
       challengers once. Each bench:
         a. apply its rocm-smi knobs (with --autorespond yes)
         b. run ONE Magpie rebench through ``run_grid``
         c. reset to defaults so the NEXT variant starts clean
       The winner is the highest median throughput that clears the
       noise floor OVER ``auto_baseline``; otherwise auto is kept. A
       determinism row whose apply fails (HW-gated determinism) is
       dropped via the per-variant failure path.
    4. apply the winner (or keep auto) + cap-max + fan-max so the chosen
       state persists past the action boundary into the remaining
       sweep / close phases (and a resume re-applies it from state).
    5. ``finally:`` always reset on any unhandled exception so a crash
       never leaves the GPU pinned at an unsafe cap

The executor never writes its winner onto ``current_best.extra_server_args``:
power state is a host-level property, not a server flag. The Coordinator's
grid-promotion path lifts ``current_best`` for ``explore`` / ``sweep`` only;
``power_management`` intentionally stays out of it. See
``actions/power_management.md`` for the rationale.

This executor targets AMD's published ``rocm-smi`` Python CLI
surface — ``--showmaxpower`` for the per-GPU manufacturer ceiling and
``--setpoweroverdrive WATTS`` for the cap setter. ``rocm-smi``'s CLI
version is independent of both the ROCm release and the
``rocm_smi_lib`` C library version, so the probe + executor detect
support by flag presence rather than parsing a version string. The
modern ``--setpowercap`` / ``--showpowercap`` aliases that some
documentation references are NOT exposed by the upstream ``rocm-smi``
CLI; using ``--setpoweroverdrive`` keeps us aligned with the actual
binary shipped under ``/opt/rocm/bin/rocm-smi``. Hosts whose CLI
lacks ``--setpoweroverdrive`` fail cleanly via the probe (and the
executor surfaces ``error_class='rocm_smi_set_failed'`` mid-sweep
for the same reason); bring those hosts forward rather than
special-casing further legacy command syntax. ``amd-smi`` is the
forward-looking replacement and is a separate integration target.

Sudo is auto-detected via ``geteuid()``: when the executor runs as
root (the normal case inside the ``rocm/sglang`` / ``rocm/vllm``
container images Hyperloom ships with) the rendered shell commands
omit ``sudo`` entirely — those images often do not install ``sudo`` at
all. Non-root callers (e.g. bare-metal ops) get the canonical
``sudo`` prefix and are expected to have a ``NOPASSWD`` sudoers entry
for the ``rocm-smi`` binary.

This executor is **single-node only**. ``rocm-smi`` mutates the
GPUs of the host it runs on, so a multi-node session would only
apply the chosen state to the head node and leave peer workers at
defaults. The action's YAML declares
``applicable_when: not is_multi_node`` (filters it out of the LLM's
catalogue on multi-node runs) and the ``__call__`` entry point hard-
refuses with ``error_class='multi_node_unsupported'`` as a runtime
backstop. ``dry_run=true`` bypasses the guard so test harnesses can
exercise the grid-resolution code under simulated multi-node config.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from ...session_paths import runs_dir
from ._grid_runner import (
    GridVariant,
    SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
    VariantResult,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)
from ..sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROCM_SMI_BIN: str = "rocm-smi"
ROCM_SMI_PROBE_TIMEOUT_SEC: int = 30
ROCM_SMI_SET_TIMEOUT_SEC: int = 60

# Soft minimum (watts) used when the operator doesn't override
# ``power_cap_floor_w`` AND the probe couldn't read a hardware minimum.
# In practice this is a thermal-headroom guess (well under any modern
# AMD datacenter GPU's idle draw); it exists only as a defensive
# backstop against an LLM proposing nonsense like ``power_cap_w=50``.
# The EFFECTIVE floor used for variant validation is
# ``max(operator_floor, rocm_smi_hardware_min)`` — i.e. the probed
# hardware minimum always wins when it's stricter. See
# :meth:`PowerManagementExecutor.__call__` for the lift logic.
POWER_CAP_DEFAULT_FLOOR_W: int = 150

# Winner gate — per-iteration framework_agent-shape gate.
#
# Each variant in this executor runs ONE Magpie rebench via its own
# single-row :func:`run_grid` call (see :meth:`_run_one_variant`). That
# per-iteration shape — apply knob → bench → check gain → reset — is
# the same as :mod:`framework_agent`, so we use the same gate parameter
# name (``keep_threshold_pct``) and inclusive comparator (``>=``) that
# :mod:`framework_agent` and :mod:`explore` use for their KEEP gates.
#
# The noise-floor *value* is sourced from :mod:`_grid_runner` —
# specifically :data:`SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT` (1.0 %).
# The multi-node default (2.0 %) is intentionally NOT imported: this
# action is single-node-only (see the ``is_multi_node()`` refusal in
# :meth:`PowerManagementExecutor.__call__`), so a multi-node gain
# threshold can never apply. The resolution order is therefore
# simpler than :mod:`explore`'s — ``params["keep_threshold_pct"]``
# (operator override) > single-node default.

POWER_MANAGEMENT_DEFAULT_TIMEOUT_SEC: int = 2400

# KERNEL-phase PM keep cutoff. Power changes carry
# thermal / reproducibility cost (the winning state must be re-applied
# on every restart and on another machine), so the KERNEL-plateau
# settle sweep requires a higher bar than the 1.0 % single-node noise
# floor that ``explore`` / ``framework_agent`` use. The Coordinator passes
# this as ``keep_threshold_pct`` on the internal settle PM task it
# enqueues at the KERNEL -> SWEEP boundary.
KERNEL_PM_KEEP_THRESHOLD_PCT: float = 2.0

# Reps for the settle sweep's vendor-default kernel-only baseline. The
# baseline is the authoritative reference the recipe's power gain is
# measured against, so we bench it a few times and take the median to
# keep one noisy run from skewing the reported power contribution.
_KERNEL_BASELINE_REPS: int = 3

# Valid values for ``params.revalidate_winners``. Default is ``"lazy"``:
# re-validate prior winners only when an entry probe detects the host
# state has drifted from ``SharedState.host_state_applied.measured_state``.
# ``"always"`` re-benches every prior winner on every call (costs one
# bench per prior winner; gives clean cross-call gain trajectories).
# ``"never"`` skips re-validation entirely (cheapest; trusts the cache).
_REVALIDATE_MODES: frozenset[str] = frozenset({"always", "lazy", "never"})


# Reset commands applied between variants and on shutdown. Built
# dynamically because the ``sudo`` prefix depends on ``geteuid()``
# (see :func:`_sudo_prefix`). Order matters: ``--resetperfdeterminism``
# MUST run before ``--resetclocks`` because perfdeterminism implicitly
# locks sclk.
_RESET_FLAGS: tuple[str, ...] = (
    "--resetperfdeterminism",
    "--resetclocks",
    "--resetpoweroverdrive",
    "--resetfans",
)


def _sudo_prefix() -> str:
    """Return ``"sudo "`` for non-root callers, ``""`` when already root.

    Hyperloom is typically run inside ROCm Docker images (``rocm/sglang``,
    ``rocm/vllm``) as the root user — no ``sudo`` binary present and no
    need for one. Bare-metal hosts (or non-root containers) still need
    ``sudo`` for the rocm-smi setter calls. Detection is via
    :func:`os.geteuid` so it's also correct under ``unshare`` / user-ns
    sandboxes. Not cached because the syscall is essentially free and
    avoiding the cache means tests can monkeypatch ``os.geteuid``
    deterministically.
    """
    return "" if os.geteuid() == 0 else "sudo "


def _reset_cmds() -> tuple[str, ...]:
    """Live reset-command list, rendered with the current sudo prefix.

    Returned fresh on every call so tests can flip ``geteuid`` between
    cases. Production hot path calls this exactly once per variant +
    once at the action boundary; the cost is negligible.
    """
    sp = _sudo_prefix()
    return tuple(
        f"{sp}{ROCM_SMI_BIN} {flag} --autorespond yes"
        for flag in _RESET_FLAGS
    )


# ---------------------------------------------------------------------------
# Variant model
# ---------------------------------------------------------------------------
@dataclass
class PowerVariant:
    """One row of the power-knob sweep."""

    name: str
    power_cap_w: int | None = None
    perflevel: str | None = None             # auto / high / manual / profile_* (see _ALLOWED_PERFLEVELS; 'low' rejected)
    sclk_idx: int | None = None              # requires perflevel=manual
    mclk_idx: int | None = None              # requires perflevel=manual
    pcie_idx: int | None = None              # requires perflevel=manual
    perf_deterministic_mhz: int | None = None
    fan_pct: int | None = None
    devices: tuple[int, ...] = ()            # empty = all GPUs
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":                   self.name,
            "power_cap_w":            self.power_cap_w,
            "perflevel":              self.perflevel,
            "sclk_idx":               self.sclk_idx,
            "mclk_idx":               self.mclk_idx,
            "pcie_idx":               self.pcie_idx,
            "perf_deterministic_mhz": self.perf_deterministic_mhz,
            "fan_pct":                self.fan_pct,
            "devices":                list(self.devices),
            "note":                   self.note,
        }


# ---------------------------------------------------------------------------
# Cross-call search ledger (mirror of explore_search)
# ---------------------------------------------------------------------------
# Fields we hash to identify "this is the same power state I've already
# benched". ``name`` and ``note`` are intentionally EXCLUDED so an LLM
# rename of an already-tested variant (e.g. ``cap_80pct_high`` →
# ``my_special_cap``) collapses to the same fingerprint and gets
# deduped by the executor's pre-filter.
_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "power_cap_w",
    "perflevel",
    "sclk_idx",
    "mclk_idx",
    "pcie_idx",
    "perf_deterministic_mhz",
    "fan_pct",
    "devices",
)


def power_variant_fingerprint(power_settings: Any) -> str:
    """Stable 16-char fingerprint over the variant's power knobs.

    Accepts either a :class:`PowerVariant` instance or a plain dict
    matching :meth:`PowerVariant.to_dict` (the shape stored in the
    search ledger and the result payload's ``power_settings`` field).

    The hash covers the eight power knobs that actually mutate host
    state. Two variants with identical knobs but different ``name`` /
    ``note`` collide intentionally — that's the dedup we want when the
    LLM resubmits a previously-tested config under a fresh name.

    16 hex chars matches the explore_search fingerprint width
    (``variant_fingerprint`` in ``_grid_runner.py``) so prompt
    formatters and migration helpers that look for a "fingerprint key"
    by length apply uniformly across both ledgers (``explore_search``
    and ``power_management_search``).
    """
    if isinstance(power_settings, PowerVariant):
        raw = power_settings.to_dict()
    elif isinstance(power_settings, dict):
        raw = power_settings
    else:
        raw = {}
    payload: list[tuple[str, Any]] = []
    for field_name in _FINGERPRINT_FIELDS:
        value = raw.get(field_name)
        if field_name == "devices":
            value = tuple(sorted(int(d) for d in (value or ())))
        payload.append((field_name, value))
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _initial_power_management_search_state() -> dict[str, Any]:
    """Empty :attr:`SharedState.power_management_search` ledger.

    Schema parallels explore_search ledger layout
    but with a power-knob fingerprint (see
    :func:`power_variant_fingerprint`). ``last_round`` records the
    most recent round's tested + winner fingerprints so a downstream
    reader can see the per-round outcome without scanning all of
    ``tested``.
    """
    return {
        "schema_version": 1,
        "accepted": [],
        "rejected": [],
        "tested": {},
        "name_index": {},
        "cursor": 0,
        "last_round": {},
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
# The executor is a *performance* tuner — every variant we accept must
# have a plausible path to non-negative perf delta vs. baseline. AMD
# rocm-smi exposes additional perflevels that pin clocks DOWN
# (``low`` / ``profile_min_sclk`` / ``profile_min_mclk``); accepting
# them here would let an LLM or stale-ledger replay burn a ~40-min
# Magpie cycle proving "slower is slower" and pollute the cross-call
# fingerprint cache with a "tested" row for an option no perf
# optimizer should ever recommend. ``manual`` stays — it's the GATE
# for ``--setsclk``/``--setmclk``/``--setpcie`` to take effect, not a
# perf knob itself (see :func:`_build_variant_from_payload` for the
# auto-injection rule that pairs it with explicit pins).
_ALLOWED_PERFLEVELS: frozenset[str] = frozenset({
    "auto", "high", "manual",
    "profile_standard", "profile_peak", "profile_compute",
})


def _build_variant_from_payload(raw: dict[str, Any], idx: int) -> PowerVariant:
    """Coerce one LLM-supplied grid entry into a typed :class:`PowerVariant`.

    Raises :class:`ValueError` on shape / vocabulary errors so the executor
    can surface ``error_class='bad_param'`` (Coordinator promotes that to a
    policy_denied observation).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"grid[{idx}] must be a dict, got {type(raw).__name__}")
    name = str(raw.get("name") or f"variant_{idx}").strip()
    if not name:
        raise ValueError(f"grid[{idx}].name must be non-empty")

    def _opt_int(key: str) -> int | None:
        v = raw.get(key)
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"grid[{idx}].{key} must be int, got {v!r}") from exc

    perflevel_raw = raw.get("perflevel")
    perflevel = str(perflevel_raw).strip().lower() if perflevel_raw else None
    if perflevel and perflevel not in _ALLOWED_PERFLEVELS:
        raise ValueError(
            f"grid[{idx}].perflevel={perflevel!r} not in "
            f"{sorted(_ALLOWED_PERFLEVELS)!r}"
        )

    devices_raw = raw.get("devices") or ()
    try:
        devices = tuple(int(d) for d in devices_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"grid[{idx}].devices must be list[int], got {devices_raw!r}"
        ) from exc

    fan = _opt_int("fan_pct")
    if fan is not None and not (0 <= fan <= 100):
        raise ValueError(f"grid[{idx}].fan_pct={fan} out of [0,100]")

    sclk_idx = _opt_int("sclk_idx")
    mclk_idx = _opt_int("mclk_idx")
    pcie_idx = _opt_int("pcie_idx")
    # Manual-perflevel guard.
    #
    # On AMD rocm-smi the ``--setsclk`` / ``--setmclk`` / ``--setpcie``
    # setters are silent no-ops unless the GPU is in ``manual``
    # perflevel first — they print a "GPU is not in manual performance
    # level" warning to stderr, exit 0, and leave the clock untouched.
    # Without an upfront guard the executor would happily bench an
    # unmodified GPU for ~40 min, log a ``tested`` ledger row whose
    # ``power_settings`` claims the pin we never actually applied, and
    # then dedup the very same fingerprint next round. Two fixes:
    #
    #   1. ``perflevel`` unset + any pin set → auto-inject ``manual``
    #      so the LLM doesn't have to relearn rocm-smi's gating rule.
    #   2. ``perflevel`` set to anything BUT ``manual`` + any pin →
    #      hard reject. ``auto`` / ``high`` / ``profile_*`` all force
    #      the governor/preset to pick the clocks, so combining them
    #      with explicit pins is contradictory at the hardware level.
    #
    # :func:`_is_contradictory_combo` is the cross-variant statement of
    # the same rocm-smi constraints (used by direct ``PowerVariant``
    # constructions / tests); the guard here is the per-row twin that
    # validates LLM-supplied grid payloads.
    pin_fields = {
        "sclk_idx": sclk_idx,
        "mclk_idx": mclk_idx,
        "pcie_idx": pcie_idx,
    }
    set_pins = {k: v for k, v in pin_fields.items() if v is not None}
    if set_pins:
        if perflevel is None:
            log.info(
                "power_management: grid[%d] auto-injecting perflevel=manual "
                "(required by %s pin%s)",
                idx, "+".join(sorted(set_pins.keys())),
                "" if len(set_pins) == 1 else "s",
            )
            perflevel = "manual"
        elif perflevel != "manual":
            raise ValueError(
                f"grid[{idx}].perflevel={perflevel!r} cannot be combined "
                f"with {sorted(set_pins.keys())!r} pins — clock pins "
                f"require perflevel='manual' (omit perflevel to auto-inject)"
            )

    return PowerVariant(
        name=name,
        power_cap_w=_opt_int("power_cap_w"),
        perflevel=perflevel,
        sclk_idx=sclk_idx,
        mclk_idx=mclk_idx,
        pcie_idx=pcie_idx,
        perf_deterministic_mhz=_opt_int("perf_deterministic_mhz"),
        fan_pct=fan,
        devices=devices,
        note=str(raw.get("note") or ""),
    )


def _enforce_cap_bounds(
    v: PowerVariant, *, floor_w: int, ceiling_w: int,
) -> str | None:
    """Return rejection reason if v.power_cap_w is out of [floor, ceiling].

    Returns ``None`` when the variant has no cap or the cap is in range.
    A clamp would silently mutate the operator's intent so we reject
    instead.
    """
    if v.power_cap_w is None:
        return None
    if v.power_cap_w < floor_w:
        return f"power_cap_w={v.power_cap_w} below floor={floor_w}"
    if ceiling_w > 0 and v.power_cap_w > ceiling_w:
        return f"power_cap_w={v.power_cap_w} above ceiling={ceiling_w}"
    return None


# ---------------------------------------------------------------------------
# rocm-smi shell helpers (sync — fast probes / short setters)
# ---------------------------------------------------------------------------
def _which_rocm_smi() -> bool:
    """Lightweight presence check so dry-run / mock harnesses don't shell out."""
    try:
        return subprocess.run(
            ["which", ROCM_SMI_BIN], capture_output=True, timeout=5,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_smi(
    cmd: str,
    *,
    timeout_sec: int = ROCM_SMI_SET_TIMEOUT_SEC,
    check: bool,
) -> tuple[int, str, str]:
    """Shell out to ``rocm-smi`` and return ``(returncode, stdout, stderr)``.

    Raises :class:`RuntimeError` when ``check=True`` and the command
    exits non-zero. Always uses ``shlex.split`` so an attacker-controlled
    cap value cannot smuggle in a sub-command (every payload field is
    coerced to ``int`` upstream anyway, but defense in depth).
    """
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"rocm-smi not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"rocm-smi command timed out after {timeout_sec}s: {cmd!r}"
        ) from exc
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{cmd!r} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc.returncode, proc.stdout, proc.stderr


def _probe_powercap_range(
    devices: tuple[int, ...],
) -> tuple[int, int] | None:
    """Return ``(min_w, max_w)`` from ``rocm-smi --showmaxpower --json``.

    Aggregates the manufacturer ceiling across the requested device
    subset (or all devices when empty) by taking ``min(per_gpu_max)``
    so a sweep respects the most-restrictive ceiling on the node.

    The ``rocm-smi`` Python CLI does not expose a *hardware-minimum*
    powercap reading: ``--showmaxpower`` only reports the ceiling, and
    ``--showpower`` reports the live draw rather than the cap setpoint.
    We therefore return ``min_w=0`` as a sentinel meaning "no hardware
    floor available" — :meth:`PowerManagementExecutor._resolve_variants`
    treats ``min_w < floor_w`` as the operator/default soft floor
    binding, so the default-grid synthesis still works (its span runs
    from ``floor_w`` up to the probed ceiling).

    Returns ``None`` on parse failure / missing binary / missing
    ``--showmaxpower`` flag. Callers treat ``None`` as "no default
    grid available" and require an explicit ``task.params.grid``.
    """
    cmd = f"{ROCM_SMI_BIN} --showmaxpower --json"
    if devices:
        cmd += "".join(f" -d {int(d)}" for d in devices)
    try:
        rc, stdout, _ = _run_smi(
            cmd, timeout_sec=ROCM_SMI_PROBE_TIMEOUT_SEC, check=True,
        )
    except RuntimeError as exc:
        log.warning("power_management: --showmaxpower probe failed: %s", exc)
        return None
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        log.warning(
            "power_management: --showmaxpower --json parse failure: %s "
            "(stdout=%r)", exc, stdout[:200],
        )
        return None

    # `--showmaxpower --json` emits one entry per device with a key like
    # ``"Max Graphics Package Power (W)"``. Filter loosely so we tolerate
    # minor wording variations across rocm-smi builds (some emit
    # ``"Max Graphics Package Power"``, others ``"max power (W)"``).
    maxs: list[int] = []
    for _device_key, fields in data.items():
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            try:
                watts = int(float(value))
            except (TypeError, ValueError):
                continue
            lower = key.lower()
            if "max" in lower and "power" in lower:
                maxs.append(watts)
    if not maxs:
        log.warning(
            "power_management: --showmaxpower --json had no "
            "'Max ... Power' field (stdout=%r); no default grid",
            stdout[:200],
        )
        return None
    # min_w=0 is the "no hardware minimum available" sentinel; see the
    # docstring + _resolve_variants for how callers consume it.
    return 0, min(maxs)


def _probe_current_state(
    devices: tuple[int, ...],
) -> dict[str, Any]:
    """Probe the currently-applied power state for cache-staleness checks.

    Reads ``--showperflevel`` (current perflevel) from rocm-smi.
    The ROCm 4.x-vintage ``rocm-smi`` CLI does not expose a "current
    cap setpoint" reading: ``--showmaxpower`` reports the
    manufacturer ceiling (which doesn't shift under setter calls),
    and ``--showpower`` reports live workload draw (which is too
    noisy to drift-detect against). ``powercap_w`` therefore stays
    ``None`` in the returned dict and ``_host_state_is_stale`` skips
    that field's comparison — drift detection on this CLI is
    perflevel-only.

    Best-effort: any probe failure returns ``None`` in the affected
    field rather than raising. Used by the executor's lazy-revalidate
    logic to detect when ``SharedState.host_state_applied.measured_state``
    diverges from the live host — typically after an operator-driven
    rocm-smi tweak, a reboot, or a thermal-throttle event that
    silently shifted clocks beneath us.

    Returns ``{"powercap_w": int|None, "perflevel": str|None}``.
    """
    out: dict[str, Any] = {"powercap_w": None, "perflevel": None}

    pl_cmd = f"{ROCM_SMI_BIN} --showperflevel --json"
    if devices:
        pl_cmd += "".join(f" -d {int(d)}" for d in devices)
    try:
        _rc, pl_stdout, _ = _run_smi(
            pl_cmd, timeout_sec=ROCM_SMI_PROBE_TIMEOUT_SEC, check=True,
        )
        pl_data = json.loads(pl_stdout or "{}")
        levels: set[str] = set()
        for _device_key, fields in pl_data.items():
            if not isinstance(fields, dict):
                continue
            for key, value in fields.items():
                if "performance level" in key.lower() or "perflevel" in key.lower():
                    levels.add(str(value).strip().lower())
        # Only stamp a perflevel when it's uniform across devices —
        # mixed levels mean SOMETHING already drifted and we'd rather
        # mark the cache stale than guess.
        if len(levels) == 1:
            out["perflevel"] = next(iter(levels))
    except (RuntimeError, json.JSONDecodeError) as exc:
        log.info("power_management: current-perflevel probe failed: %s", exc)

    return out


def _probe_clkfrq_raw(devices: tuple[int, ...]) -> dict[str, Any] | None:
    """Shell out to ``rocm-smi --showclkfrq --json`` ONCE and return the
    parsed JSON dict, or ``None`` on any failure.

    The clock-derived probes — :func:`_probe_top_sclk_mhz` (the
    fallback for the determinism ladder), :func:`_gfx_high_sclk_idx`
    (the GFX-high pin on memory rows), and
    :func:`_parse_mclk_levels_from_clkfrq` (the memory-axis capability
    gate) — read the same ``--showclkfrq`` payload, so
    :meth:`PowerManagementExecutor.__call__` fetches it once here and
    threads the result into both parsers via their ``data=`` argument
    (one ~30 s probe instead of two identical ones). The parsers retain
    the no-arg path (they fetch via this helper themselves) so their
    direct unit tests keep shelling out independently.

    Best-effort: missing binary / non-zero exit / JSON parse error all
    log at INFO and return ``None``; callers degrade to "omit that row".
    """
    cmd = f"{ROCM_SMI_BIN} --showclkfrq --json"
    if devices:
        cmd += "".join(f" -d {int(d)}" for d in devices)
    try:
        _rc, stdout, _ = _run_smi(
            cmd, timeout_sec=ROCM_SMI_PROBE_TIMEOUT_SEC, check=True,
        )
    except RuntimeError as exc:
        log.info("power_management: --showclkfrq probe failed: %s", exc)
        return None
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        log.info(
            "power_management: --showclkfrq --json parse failure: %s "
            "(stdout=%r)", exc, stdout[:200],
        )
        return None
    return data if isinstance(data, dict) else None


# Frequency-token regex shared by the clock-table text parsers. Matches
# ``2400Mhz`` / ``2400 MHz`` / ``2.4Ghz`` (case-insensitive), tolerating a
# trailing rocm-smi ``*`` current-level marker on the surrounding token.
_FREQ_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mhz|ghz)", re.IGNORECASE)


def _freq_tokens_to_mhz(text: str) -> list[int]:
    """Extract every ``<n>Mhz`` / ``<n>Ghz`` frequency in ``text`` as MHz.

    Pure helper shared by :func:`_parse_sclkrange_top_mhz` and the
    standalone probe (parity-tested). Tolerates the trailing ``*``
    current-level marker rocm-smi prints on the active DPM row because
    the regex only consumes the numeric value + unit, not the marker.
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

    The labeled range output looks like
    ``GPU[0]: Valid sclk range: 500Mhz - 2400Mhz``; the top of the
    *clean* DVFS range is the largest frequency present. This is the
    PRIMARY top-sclk source because the bare ``--showclkfrq`` DPM index
    table on some silicon is coarse and scrambled (e.g. ``0:500``,
    ``1:158``, ``2:2400``) — only the range gives the true ceiling.
    """
    freqs = _freq_tokens_to_mhz(text)
    return max(freqs) if freqs else None


def _probe_sclkrange_raw(devices: tuple[int, ...]) -> str | None:
    """Shell out to ``rocm-smi --showsclkrange`` and return raw stdout.

    Best-effort: missing binary / non-zero exit returns ``None`` so the
    caller can fall back to the ``--showclkfrq`` DPM-index maximum.
    """
    cmd = f"{ROCM_SMI_BIN} --showsclkrange"
    if devices:
        cmd += "".join(f" -d {int(d)}" for d in devices)
    try:
        _rc, stdout, _ = _run_smi(
            cmd, timeout_sec=ROCM_SMI_PROBE_TIMEOUT_SEC, check=True,
        )
    except RuntimeError as exc:
        log.info("power_management: --showsclkrange probe failed: %s", exc)
        return None
    return stdout


def _probe_top_sclk_mhz(
    devices: tuple[int, ...], *,
    data: dict[str, Any] | None = None,
    sclkrange_text: str | None = None,
) -> int | None:
    """Return the top engine-clock frequency (MHz), or None on failure.

    PRIMARY source: ``rocm-smi --showsclkrange`` (the clean DVFS range,
    parsed by :func:`_parse_sclkrange_top_mhz`). FALLBACK: the maximum
    frequency in ``rocm-smi --showclkfrq --json`` (the bare DPM-index
    table). The DPM index table on some silicon is scrambled, so the
    range is preferred; the ``--showclkfrq`` max is kept only as a
    backstop for hosts whose ``--showsclkrange`` is unavailable.

    Used to build the GFX determinism ladder: ``--setperfdeterminism N``
    wants a frequency, not a DPM index, and the ladder pins 100 / 95 /
    90% of this ceiling. Best-effort by design — when both sources fail
    the determinism ladder is skipped and the grid self-check flags
    ``grid_degraded``.

    ``data`` lets the caller pass an already-fetched ``--showclkfrq``
    payload (see :func:`_probe_clkfrq_raw`) for the fallback;
    ``sclkrange_text`` lets the caller pass an already-fetched
    ``--showsclkrange`` stdout. Both are fetched here when ``None``.
    """
    text = (
        sclkrange_text if sclkrange_text is not None
        else _probe_sclkrange_raw(devices)
    )
    if text:
        mhz = _parse_sclkrange_top_mhz(text)
        if mhz:
            return mhz

    # Fallback: the maximum frequency in the bare --showclkfrq table.
    if data is None:
        data = _probe_clkfrq_raw(devices)
    if not isinstance(data, dict):
        return None
    candidates: list[int] = []
    for _device_key, fields in data.items():
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            lower = key.lower()
            if "sclk" not in lower and "gpu clock" not in lower:
                continue
            candidates.extend(_freq_tokens_to_mhz(str(value)))
    if not candidates:
        return None
    return max(candidates)


# Per-clock labeled DPM key in ``--showclkfrq --json``: ``mclk[N]`` etc.
_MCLK_IDX_KEY_RE = re.compile(r"^\s*mclk\s*\[\s*(\d+)\s*\]", re.IGNORECASE)


def _parse_mclk_levels_from_clkfrq(
    data: dict[str, Any] | None,
) -> list[tuple[int, int]]:
    """Parse selectable memory-clock levels from ``--showclkfrq --json``.

    Returns a sorted (by DPM index) list of ``(dpm_index, mhz)`` pairs,
    one per distinct ``mclk[N]`` entry, taking the max MHz seen per
    index across devices. Pure helper shared with the standalone probe
    (parity-tested).
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


# Per-clock labeled DPM key for the engine clock: ``sclk[N]``.
_SCLK_IDX_KEY_RE = re.compile(r"^\s*sclk\s*\[\s*(\d+)\s*\]", re.IGNORECASE)


def _parse_sclk_levels_from_clkfrq(
    data: dict[str, Any] | None,
) -> list[tuple[int, int]]:
    """Parse selectable engine-clock levels from ``--showclkfrq --json``.

    Returns a sorted (by DPM index) list of ``(dpm_index, mhz)`` pairs,
    one per distinct ``sclk[N]`` entry, taking the max MHz seen per index
    across devices. Mirrors :func:`_parse_mclk_levels_from_clkfrq`.
    """
    if not isinstance(data, dict):
        return []
    by_idx: dict[int, int] = {}
    for _device_key, fields in data.items():
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            m = _SCLK_IDX_KEY_RE.match(str(key))
            if not m:
                continue
            idx = int(m.group(1))
            freqs = _freq_tokens_to_mhz(str(value))
            mhz = max(freqs) if freqs else 0
            if mhz > by_idx.get(idx, -1):
                by_idx[idx] = mhz
    return sorted(by_idx.items())


def _gfx_high_sclk_idx(data: dict[str, Any] | None) -> int | None:
    """Return the sclk DPM index with the HIGHEST frequency (GFX-high pin).

    The memory rows pin GFX high via ``--setsclk <idx>`` while stepping
    memory. The sclk DPM index table on some silicon is coarse and
    scrambled — the bracketed index order does NOT track frequency (e.g.
    ``0:500, 1:2400, 2:158``) — so the GFX-high pin must be the index
    whose frequency is maximal, NOT simply the largest index (the latter
    would risk pinning GFX *low* and conflating the GFX + memory axes).
    Ties break to the highest index. Returns ``None`` when no sclk levels
    parse (the memory rows then degrade cleanly + flag ``grid_degraded``).
    """
    levels = _parse_sclk_levels_from_clkfrq(data)
    if not levels:
        return None
    best_idx, _best_mhz = max(levels, key=lambda kv: (kv[1], kv[0]))
    return best_idx


def _parse_mclkrange(text: str) -> tuple[int, int] | None:
    """Parse ``rocm-smi --showmclkrange`` text → ``(min_mhz, max_mhz)``.

    Returns ``None`` when fewer than two frequencies are present.
    """
    freqs = _freq_tokens_to_mhz(text)
    if len(freqs) < 2:
        return None
    return (min(freqs), max(freqs))


def _probe_mclkrange_raw(devices: tuple[int, ...]) -> str | None:
    """Shell out to ``rocm-smi --showmclkrange`` and return raw stdout."""
    cmd = f"{ROCM_SMI_BIN} --showmclkrange"
    if devices:
        cmd += "".join(f" -d {int(d)}" for d in devices)
    try:
        _rc, stdout, _ = _run_smi(
            cmd, timeout_sec=ROCM_SMI_PROBE_TIMEOUT_SEC, check=True,
        )
    except RuntimeError as exc:
        log.info("power_management: --showmclkrange probe failed: %s", exc)
        return None
    return stdout


def _probe_mclk_levels(
    devices: tuple[int, ...], *,
    clkfrq_data: dict[str, Any] | None = None,
    mclkrange_text: str | None = None,
) -> dict[str, Any]:
    """Probe the selectable memory-clock levels for the capability gate.

    Reads the labeled ``mclk[N]`` section of ``--showclkfrq --json``
    (the discrete selectable DPM levels ``--setmclk`` can target) plus
    ``--showmclkrange`` (the continuous range, informational). Returns:

    ``{"indices": [int,...], "mhz": [int,...], "count": int,
       "range": (min,max)|None}``

    ``count`` is the number of distinct selectable mclk levels — the
    memory axis is capability-gated on ``count >= 2``. On silicon with
    a single memory level (e.g. this MI355X, one 2000 MHz level) the
    memory rows are omitted and the reason logged.

    Best-effort: any probe / parse failure yields an empty levels set
    (``count == 0``), which the gate treats as "memory axis
    unavailable".
    """
    if clkfrq_data is None:
        clkfrq_data = _probe_clkfrq_raw(devices)
    levels = _parse_mclk_levels_from_clkfrq(clkfrq_data)
    indices = [idx for idx, _mhz in levels]
    mhz = [m for _idx, m in levels]

    text = (
        mclkrange_text if mclkrange_text is not None
        else _probe_mclkrange_raw(devices)
    )
    rng = _parse_mclkrange(text) if text else None

    return {
        "indices": indices,
        "mhz": mhz,
        "count": len(indices),
        "range": rng,
    }


# GFX determinism ladder fractions of the probed top sclk. ``det_100``
# (1.00, the top-of-range anchor) is always emitted; ``det_95`` / ``det_90``
# / ``det_85`` are emitted unless the workload is compute-bound (lowering
# GFX cannot help a compute-bound shape, so the descent rungs are pruned
# there).
_SETTLE_DETERMINISM_PCTS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85)

# Reps the always-on ``auto_baseline`` incumbent row runs (median taken)
# so one noisy run can't move the reference the whole sweep gates against.
_AUTO_BASELINE_REPS: int = 3


def _det_row_name(pct: float) -> str:
    """Stable ``det_<pct>`` row name (det_100 / det_95 / det_90 / det_85)."""
    return f"det_{int(round(pct * 100))}"


def _build_settle_grid(
    *,
    cap_w: int,
    fan_pct: int = 100,
    bound_kind: str | None,
    top_sclk_mhz: int | None,
    sclk_top_idx: int | None = None,
    mclk_levels: dict[str, Any] | None = None,
    pcts: tuple[float, ...] = _SETTLE_DETERMINISM_PCTS,
) -> tuple[list[PowerVariant], dict[str, str]]:
    """Roofline-routed settle grid (the single power tune of a run).

    Throughput-only, fan + power-cap pinned to max on EVERY row (so the
    kernel phase and this sweep are the same hardware state and power is
    never a confound). GFX is tuned via ``--setperfdeterminism`` only —
    there is no GFX DPM-index downregulation path (the DPM index table
    on this silicon is coarse and scrambled, so only determinism gives
    usable resolution). The rows:

    * ``auto_baseline`` — ``--setperflevel auto`` + cap-max + fan-max.
      The incumbent (also the kernel-climb state) and the reference the
      winner gate measures against. Always emitted.
    * ``high`` — ``--setperflevel high`` + cap-max + fan-max. Always.
    * ``det_100`` — ``--setperfdeterminism <top>`` + cap-max + fan-max.
      The determinism anchor. Always (when a top sclk was probed).
    * ``det_95`` / ``det_90`` / ``det_85`` — ``--setperfdeterminism
      <pct·top>`` + cap-max + fan-max. Emitted UNLESS compute-bound.
    * ``mclk_<mhz>`` — ``--setperflevel manual`` + ``--setsclk <top>``
      (GFX held high) + ``--setmclk <idx>`` (memory stepped) + cap-max +
      fan-max. Emitted only when NOT memory-bound AND there are >= 2
      selectable mclk levels (the capability gate).

    Returns ``(rows, grid_degraded)``. ``grid_degraded`` maps an
    *expected-but-empty* ladder name to a human reason — it is populated
    only when the roofline + capability said a ladder SHOULD have rows
    but the probe couldn't produce any (e.g. the top-sclk probe failed
    so the determinism ladder is empty, or the memory axis was capable
    but the GFX-pin index was unavailable). A capability skip (e.g.
    ``< 2`` mclk levels, or compute-bound pruning) is NOT degradation —
    it is the intended gate firing — so it is logged but not flagged.
    """
    bk = (bound_kind or "unknown").strip().lower()
    grid_degraded: dict[str, str] = {}

    rows: list[PowerVariant] = [
        PowerVariant(
            name="auto_baseline", perflevel="auto",
            power_cap_w=cap_w, fan_pct=fan_pct, note="auto_baseline",
        ),
        PowerVariant(
            name="high", perflevel="high",
            power_cap_w=cap_w, fan_pct=fan_pct, note="high",
        ),
    ]

    # --- GFX determinism ladder (always expected: det_100 is the anchor).
    det_rows: list[PowerVariant] = []
    if top_sclk_mhz and top_sclk_mhz > 0:
        seen: set[int] = set()
        for pct in pcts:
            # det_100 always; det_95 / det_90 / det_85 pruned when
            # compute-bound (a GFX-clock descent can't help a
            # compute-bound shape).
            if pct < 1.0 and bk == "compute":
                continue
            mhz = int(round(top_sclk_mhz * pct))
            if mhz <= 0 or mhz in seen:
                continue
            seen.add(mhz)
            det_rows.append(PowerVariant(
                name=_det_row_name(pct),
                perf_deterministic_mhz=mhz,
                power_cap_w=cap_w, fan_pct=fan_pct,
                note="determinism",
            ))
    if not det_rows:
        # det_100 was expected on every box but no MHz value was probed.
        grid_degraded["gfx_determinism"] = (
            "no top sclk probed (--showsclkrange / --showclkfrq) — "
            "determinism ladder produced 0 rows"
        )
    rows.extend(det_rows)

    # --- Memory axis (capability-gated: NOT memory-bound AND >= 2 levels).
    levels = mclk_levels or {}
    mclk_count = int(levels.get("count") or 0)
    mclk_indices = list(levels.get("indices") or [])
    mclk_mhz = list(levels.get("mhz") or [])
    memory_axis_applicable = bk != "memory" and mclk_count >= 2
    if bk == "memory":
        log.info(
            "power_management: memory axis skipped — workload is "
            "memory-bound (stepping memory down cannot help)",
        )
    elif mclk_count < 2:
        log.info(
            "power_management: memory axis skipped — only %d selectable "
            "mclk level(s) (need >= 2)", mclk_count,
        )
    if memory_axis_applicable:
        mem_rows: list[PowerVariant] = []
        # Hold GFX high by pinning sclk to its highest-FREQUENCY DPM index
        # (``sclk_top_idx``, chosen by frequency via ``_gfx_high_sclk_idx``
        # — NOT the largest index, since the table is scrambled); step
        # memory to each NON-top selectable level. Without a probed GFX-high
        # index we cannot keep GFX pinned, so the row would conflate the
        # two axes — skip + flag degraded rather than emit a misleading
        # measurement.
        if sclk_top_idx is None:
            grid_degraded["memory"] = (
                "memory axis capable (>= 2 mclk levels) but no GFX-high "
                "sclk DPM index probed — cannot pin GFX high while stepping "
                "memory; 0 memory rows produced"
            )
        else:
            top_idx = max(mclk_indices) if mclk_indices else None
            mhz_by_idx = dict(zip(mclk_indices, mclk_mhz))
            for idx in mclk_indices:
                if idx == top_idx:
                    continue  # top mclk is already covered by high/auto
                label_mhz = mhz_by_idx.get(idx)
                name = (
                    f"mclk_{label_mhz}mhz" if label_mhz
                    else f"mclk_idx{idx}"
                )
                mem_rows.append(PowerVariant(
                    name=name, perflevel="manual",
                    sclk_idx=int(sclk_top_idx), mclk_idx=int(idx),
                    power_cap_w=cap_w, fan_pct=fan_pct, note="mclk",
                ))
            if not mem_rows:
                grid_degraded["memory"] = (
                    "memory axis capable but no non-top mclk levels to "
                    "step to; 0 memory rows produced"
                )
            rows.extend(mem_rows)

    return rows, grid_degraded


# ---------------------------------------------------------------------------
# Combo-contradiction guard
# ---------------------------------------------------------------------------
# Encodes rocm-smi's own semantic constraints for stacking knobs. The
# roofline-routed settle grid builds each row explicitly (no cartesian
# merge), but the determinism-vs-manual-pin rule this guard captures is
# exactly the gotcha the determinism ladder must respect
# (``--setperfdeterminism`` sets perflevel=DETERMINISM, which
# contradicts ``perflevel=manual`` + ``sclk_idx``). Kept as the
# canonical statement of that constraint, also exercised directly by
# direct ``PowerVariant`` constructions in tests + future callers.


def _is_contradictory_combo(parts: tuple[PowerVariant, ...]) -> str | None:
    """Return a human-readable reason if these variants can't safely
    stack, else None.

    The rules encode rocm-smi's own semantic constraints:

    * Two variants both setting ``power_cap_w`` to different watts
      can't coexist — last setter wins, the first's measurement intent
      is lost. (Same watts is fine; we tolerate the redundant set.)
    * Two variants setting different non-None ``perflevel`` values
      conflict — perflevel is a single global mode.
    * ``perflevel='high'`` pins clocks to the top of the DVFS table;
      combining it with an explicit ``sclk_idx`` / ``mclk_idx`` is
      either redundant (top index) or contradictory (any lower index).
      ``perflevel='auto'`` similarly precludes manual clock pins. This
      check is defense in depth — :func:`_build_variant_from_payload`
      already auto-injects ``manual`` (or rejects) for pin-bearing
      LLM payloads, but the determinism ladder and ad-hoc direct
      ``PowerVariant`` constructions (tests, future callers) bypass the
      per-row guard and rely on this filter.
    * Two variants setting different ``perf_deterministic_mhz`` values
      conflict — only one deterministic frequency is meaningful.
    * Two variants targeting disjoint ``devices`` lists would need
      separate apply sequences; we keep combos device-symmetric.
    """
    caps = {p.power_cap_w for p in parts if p.power_cap_w is not None}
    if len(caps) > 1:
        return f"conflicting power_cap_w values: {sorted(caps)!r}"

    perflevels = {p.perflevel for p in parts if p.perflevel}
    if len(perflevels) > 1:
        return f"conflicting perflevels: {sorted(perflevels)!r}"

    forces_clocks = any(p.perflevel in {"auto", "high"} for p in parts)
    pins_clocks = any(
        p.sclk_idx is not None or p.mclk_idx is not None
        for p in parts
    )
    if forces_clocks and pins_clocks:
        return "perflevel=auto/high precludes manual sclk_idx/mclk_idx pins"

    det_freqs = {
        p.perf_deterministic_mhz for p in parts
        if p.perf_deterministic_mhz
    }
    if len(det_freqs) > 1:
        return f"conflicting perf_deterministic_mhz: {sorted(det_freqs)!r}"

    device_sets = {tuple(sorted(p.devices)) for p in parts if p.devices}
    if len(device_sets) > 1:
        return f"conflicting devices: {sorted(device_sets)!r}"

    return None


def _apply_variant_cmds(v: PowerVariant) -> list[str]:
    """Build the ordered list of ``rocm-smi`` commands for one variant.

    Order: power_cap first (cheapest, broadest impact), then perflevel
    (must precede sclk/mclk pin), then clock pins, then determinism,
    then fan. Empty variants return ``[]`` (no-op probe row).

    Sets the cap via ``--setpoweroverdrive WATTS`` — the canonical
    upstream ``rocm-smi`` Python CLI flag (the ``--setpowercap`` alias
    some docs reference is not exposed by the binary that ships under
    ``/opt/rocm/bin/rocm-smi``). Pairs with ``--resetpoweroverdrive``
    in :data:`_RESET_FLAGS` for symmetric apply/reset.
    """
    cmds: list[str] = []
    device_args = "".join(f" -d {int(d)}" for d in v.devices)
    sudo_prefix = _sudo_prefix()

    def _smi(flag_and_value: str) -> str:
        return f"{sudo_prefix}{ROCM_SMI_BIN}{device_args} {flag_and_value} --autorespond yes"

    if v.power_cap_w is not None:
        cmds.append(_smi(f"--setpoweroverdrive {int(v.power_cap_w)}"))
    if v.perflevel:
        cmds.append(_smi(f"--setperflevel {v.perflevel}"))
    if v.sclk_idx is not None:
        cmds.append(_smi(f"--setsclk {int(v.sclk_idx)}"))
    if v.mclk_idx is not None:
        cmds.append(_smi(f"--setmclk {int(v.mclk_idx)}"))
    if v.pcie_idx is not None:
        cmds.append(_smi(f"--setpcie {int(v.pcie_idx)}"))
    if v.perf_deterministic_mhz:  # 0 / None both skip
        cmds.append(_smi(f"--setperfdeterminism {int(v.perf_deterministic_mhz)}"))
    if v.fan_pct is not None:
        cmds.append(_smi(f"--setfan {int(v.fan_pct)}%"))
    return cmds


def _apply_variant(v: PowerVariant) -> list[str]:
    """Apply variant knobs; return the list of commands that ran."""
    applied: list[str] = []
    for c in _apply_variant_cmds(v):
        log.info("power_management: applying %s", c)
        _run_smi(c, check=True)
        applied.append(c)
    return applied


def reset_host_power_defaults(
    *, is_multi_node: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Public guarded wrapper around :func:`_reset_defaults` (workstream A).

    Used by the Coordinator for the mandatory run-start reset to vendor
    defaults and on resume-into-KERNEL before re-applying the recorded
    host state. Never raises.

    Returns a small status dict:

      * ``{"status": "skipped", "reason": "multi_node"}`` — PM is
        single-node only; resetting only the head node would lie about
        the cluster state, so we no-op.
      * ``{"status": "skipped", "reason": "dry_run"}`` — test / probe
        path, no GPU touched.
      * ``{"status": "skipped", "reason": "rocm_smi_unavailable"}`` —
        binary not on PATH (bare host without ROCm); nothing to reset.
      * ``{"status": "reset"}`` — reset commands issued (best-effort).
    """
    if is_multi_node:
        return {"status": "skipped", "reason": "multi_node"}
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    if not _which_rocm_smi():
        return {"status": "skipped", "reason": "rocm_smi_unavailable"}
    _reset_defaults()
    return {"status": "reset"}


def reapply_host_power_state(
    snapshot: dict[str, Any] | None,
    *, is_multi_node: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Re-apply a memorized ``host_state_applied`` snapshot (workstream B).

    Runs the exact ``smi_commands`` captured when the state was first
    applied. The caller is expected to have already reset to defaults
    (``reset_host_power_defaults``) so this only layers the memorized
    knobs back on. Best-effort; never raises.

    Returns ``{"status": "applied", "n": <count>}`` on success, or a
    ``{"status": "skipped", "reason": ...}`` dict for the no-op paths.
    """
    if is_multi_node:
        return {"status": "skipped", "reason": "multi_node"}
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    if not isinstance(snapshot, dict):
        return {"status": "skipped", "reason": "no_snapshot"}
    cmds = snapshot.get("smi_commands") or []
    if not isinstance(cmds, list) or not cmds:
        return {"status": "skipped", "reason": "no_commands"}
    if not _which_rocm_smi():
        return {"status": "skipped", "reason": "rocm_smi_unavailable"}
    applied = 0
    for c in cmds:
        try:
            _run_smi(str(c), check=False)
            applied += 1
        except RuntimeError as exc:
            log.warning(
                "power_management: re-apply cmd failed (continuing): %s", exc,
            )
    return {"status": "applied", "n": applied}


def apply_max_climb_state(
    *, devices: tuple[int, ...] = (), is_multi_node: bool = False,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Apply the fixed kernel-phase incumbent state (auto + cap/fan max).

    Called once at KERNEL entry by the Coordinator. Establishes the
    incumbent state — ``perflevel auto`` + the manufacturer ceiling
    power cap + fan at 100% — and holds it for the entire greedy kernel
    climb. Fan + power cap are pinned to max at ALL times (during the
    kernel climb AND the settle sweep) so the kernel phase and the
    sweep's ``auto_baseline`` row are the SAME hardware state — power is
    never a confound for the irreversible KEEP/REVERT/plateau signal.

    GFX is left on the ``auto`` governor here (not ``high`` / not a
    manual DPM pin): the settle sweep tunes GFX exclusively via
    ``--setperfdeterminism``, and its ``auto_baseline`` row reproduces
    exactly this state, so the climb incumbent and the sweep reference
    line up byte-for-byte.

    The ceiling is probed via ``--showmaxpower`` at climb entry; when
    the probe is unavailable the cap is simply omitted (perflevel auto
    + fan still apply). Power is NOT otherwise tuned during the climb —
    there is one power tune per run, the settle sweep at the
    KERNEL -> SWEEP boundary, which starts from THIS state.

    Returns a ``host_state_applied`` snapshot (so the Coordinator can
    record + re-apply it on resume and pass it as the settle sweep's
    incumbent), or ``None`` on the no-op paths (multi-node / dry-run /
    rocm-smi unavailable / apply failure).
    """
    if is_multi_node or dry_run:
        return None
    if not _which_rocm_smi():
        return None
    probed_devices = tuple(devices)
    # Probe the manufacturer ceiling so the climb runs at full power
    # headroom. Best-effort: omit the cap when the probe is unavailable.
    ceiling_w: int | None = None
    probed_range = _probe_powercap_range(probed_devices)
    if probed_range:
        _min_w, max_w = probed_range
        if max_w > 0:
            ceiling_w = max_w
    variant = PowerVariant(
        name="climb_max_state", perflevel="auto", power_cap_w=ceiling_w,
        fan_pct=100, devices=probed_devices, note="kernel_climb_max_state",
    )
    try:
        applied = _apply_variant(variant)
    except RuntimeError as exc:
        log.warning(
            "power_management: failed to apply kernel incumbent state "
            "(auto + cap-max + fan-max) at KERNEL entry (%s); leaving "
            "defaults", exc,
        )
        _reset_defaults()
        return None
    return {
        "variant_name":   variant.name,
        "power_settings": variant.to_dict(),
        "smi_commands":   list(applied),
        "device_ids":     list(variant.devices),
        "probed_range_w": list(probed_range) if probed_range else None,
        "measured_state": _probe_current_state(variant.devices),
        "note":           variant.note,
        "ts":             datetime.now(timezone.utc).isoformat(),
    }


def _reset_defaults() -> None:
    """Best-effort reset of every knob; never raises."""
    for c in _reset_cmds():
        try:
            _run_smi(c, check=False)
        except RuntimeError as exc:
            log.warning("power_management: reset cmd failed (continuing): %s", exc)


def _snapshot_state() -> str:
    """Capture ``rocm-smi -a --json`` for the variant audit; never raises."""
    try:
        _, stdout, _ = _run_smi(
            f"{ROCM_SMI_BIN} -a --json",
            timeout_sec=ROCM_SMI_PROBE_TIMEOUT_SEC, check=False,
        )
        return stdout
    except RuntimeError:
        return ""


def _variant_from_accepted(entry: dict[str, Any]) -> PowerVariant | None:
    """Rehydrate a :class:`PowerVariant` from a ``power_management_search.accepted`` row.

    Returns ``None`` when the row is malformed (e.g. missing
    ``power_settings``) — caller drops it silently rather than failing
    the whole re-validation phase. The display name is preserved from
    the ledger entry so audit trails line up across rounds; we tag
    ``note`` with ``prior_winner_revalidate`` so result-table readers
    can distinguish a re-bench from a fresh exploration.
    """
    if not isinstance(entry, dict):
        return None
    ps = entry.get("power_settings")
    # Empty / missing power_settings would yield a no-op PowerVariant
    # (no knobs to apply, nothing to bench). Reject explicitly so the
    # caller drops the row instead of wasting a bench iteration on it.
    if not isinstance(ps, dict) or not ps:
        return None
    name = str(entry.get("name") or ps.get("name") or "").strip()
    if not name:
        fp = str(entry.get("fingerprint") or "")[:8]
        name = f"prior_winner_{fp}" if fp else "prior_winner"

    def _opt_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    devices_raw = ps.get("devices") or ()
    try:
        devices = tuple(int(d) for d in devices_raw)
    except (TypeError, ValueError):
        devices = ()
    return PowerVariant(
        name=name,
        power_cap_w=_opt_int(ps.get("power_cap_w")),
        perflevel=(
            str(ps.get("perflevel")).strip().lower()
            if ps.get("perflevel") else None
        ),
        sclk_idx=_opt_int(ps.get("sclk_idx")),
        mclk_idx=_opt_int(ps.get("mclk_idx")),
        pcie_idx=_opt_int(ps.get("pcie_idx")),
        perf_deterministic_mhz=_opt_int(ps.get("perf_deterministic_mhz")),
        fan_pct=_opt_int(ps.get("fan_pct")),
        devices=devices,
        note="prior_winner_revalidate",
    )


def _build_host_state_snapshot(
    *,
    winner_variant: PowerVariant,
    smi_commands: list[str],
    probed_range: tuple[int, int] | None,
    top_sclk_mhz: int | None,
    gain_pct: float | None,
    session_dir: str,
    task_id: str,
) -> dict[str, Any]:
    """Build the :attr:`SharedState.host_state_applied` payload for a winner.

    Captures everything a final-report renderer (or an operator
    replaying the run) needs to recreate the GPU state without
    consulting the executor's source: the exact rocm-smi commands
    that were issued, the device subset, the probed bounds at apply
    time, and a measured-state readback (best-effort) so the next
    invocation's lazy-revalidate check has something to compare
    against.
    """
    measured = _probe_current_state(winner_variant.devices)
    return {
        "variant_name":   winner_variant.name,
        "power_settings": winner_variant.to_dict(),
        "smi_commands":   list(smi_commands),
        "device_ids":     list(winner_variant.devices),
        "probed_range_w": list(probed_range) if probed_range else None,
        "top_sclk_mhz":   top_sclk_mhz,
        "measured_state": measured,
        "gain_pct":       float(gain_pct) if gain_pct is not None else None,
        "ts":             datetime.now(timezone.utc).isoformat(),
        "session_dir":    str(session_dir),
        "task_id":        str(task_id or ""),
    }


def _host_state_is_stale(
    *, current: dict[str, Any], cached: dict[str, Any] | None,
) -> bool:
    """Return True when ``current`` probe values diverge from ``cached``.

    Used by lazy-revalidate. Missing fields on either side collapse
    to "unknown" and skip that field's comparison — a partial probe
    failure shouldn't force a re-validation cascade. Returns False
    when there's no cache at all (nothing to compare against, which
    callers treat as "fresh start, no re-validation needed").
    """
    if not isinstance(cached, dict) or not cached:
        return False
    if not isinstance(current, dict):
        return False
    for field_name in ("powercap_w", "perflevel"):
        cur_val = current.get(field_name)
        cached_val = cached.get(field_name)
        if cur_val is None or cached_val is None:
            continue
        if cur_val != cached_val:
            return True
    return False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class PowerManagementExecutor:
    """ActionRunner for the ``power_management`` action.

    See module docstring + ``actions/power_management.md`` for the
    end-to-end lifecycle. Reads the same Magpie YAML / workload-contract
    surface as :class:`BaselineExecutor`; the rocm-smi knobs apply
    *outside* the subprocess so each per-variant ``run_grid`` call sees
    the same SGLang/vLLM command line.
    """

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        variant_timeout_sec: int = POWER_MANAGEMENT_DEFAULT_TIMEOUT_SEC,
        default_power_cap_floor_w: int = POWER_CAP_DEFAULT_FLOOR_W,
    ):
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.default_power_cap_floor_w = int(default_power_cap_floor_w)

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------
    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        params = ctx.task.params or {}
        dry_run = bool(params.get("dry_run"))
        # Settle sweep marker (set by the Coordinator at the KERNEL->SWEEP
        # boundary — the single power tune of a run). When set, the
        # executor benches the kernel-combo at TRUE vendor defaults
        # (median of a few reps) to measure the FULL power gain (winner /
        # incumbent vs kernel-only). The kernel-only number is reported
        # for attribution only; it does not influence winner selection.
        # See the post-loop block below.
        measure_kernel_baseline = bool(params.get("measure_kernel_baseline"))

        # Multi-node hard refusal.
        #
        # ``rocm-smi`` is a node-local tool: the executor shells out on
        # the orchestrator host only, so applying a power cap /
        # perflevel / clock pin on a >=2-node RayJob would touch the
        # head node alone and leave peer workers at defaults. The
        # resulting Magpie measurement would benchmark a heterogeneous
        # cluster against a single-node baseline, which is meaningless
        # for KEEP/REJECT decisions and silently pollutes the
        # cross-call ledger with fingerprints that don't describe the
        # state the GPUs were actually in.
        #
        # The YAML's ``applicable_when: not is_multi_node`` predicate
        # is the primary defense (it filters the action out of the
        # Orchestration LLM's catalogue on multi-node sessions); this
        # runtime guard is the backstop for stale state.json resumes,
        # direct executor calls from operator tooling, or ledger
        # replay paths that bypass the catalogue. ``dry_run`` is
        # exempt so unit tests / probe scripts can still exercise the
        # variant-resolution code under simulated multi-node config.
        if not dry_run:
            from ._multi_node_env import is_multi_node
            if is_multi_node():
                return {
                    "status": "failed",
                    "error_class": "multi_node_unsupported",
                    "error": (
                        "power_management is single-node only: rocm-smi "
                        "setters apply to the local node only and would "
                        "leave peer workers at defaults on a >=2-node "
                        "RayJob. Set dry_run=true to validate grid "
                        "shape without touching the GPU, or run this "
                        "action on a single-node session."
                    ),
                }

        # Resolve workspace + config (shared pattern with baseline.py).
        extra = getattr(ctx, "extra", None) or {}
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "power_management", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or default_baseline_config()
        )
        if not dry_run and not config_path.exists():
            return {
                "status": "failed",
                "error_class": "missing_config",
                "error": f"config not found: {config_path}",
            }

        # Sanitize Orchestration-supplied overrides (Magpie leak salvage).
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
            }

        # Cross-call search ledger (Coordinator injects from
        # SharedState.power_management_search). Defaults to an empty
        # ledger so direct executor invocations (tests, ad-hoc CLI
        # calls) continue to work without the Coordinator plumbing.
        search = dict(
            params.get("power_management_search")
            or _initial_power_management_search_state()
        )
        search.setdefault("schema_version", 1)
        search.setdefault("accepted", [])
        search.setdefault("rejected", [])
        search.setdefault("tested", {})
        search.setdefault("name_index", {})
        search.setdefault("cursor", 0)
        prior_accepted: list[dict[str, Any]] = [
            a for a in (search.get("accepted") or [])
            if isinstance(a, dict)
        ]
        prior_tested: dict[str, Any] = dict(search.get("tested") or {})
        prior_tested_fps: set[str] = set(prior_tested.keys())
        prior_accepted_fps: set[str] = {
            str(a.get("fingerprint")
                or power_variant_fingerprint(a.get("power_settings") or {}))
            for a in prior_accepted
        }
        prior_accepted_fps.discard("")

        # Revalidation policy: how to handle prior winners on this call.
        # See ``_REVALIDATE_MODES`` for the contract; default 'lazy'
        # mirrors the answer the operator selected when this feature
        # was designed (cheapest in steady state, automatic recovery
        # under drift).
        revalidate_mode = str(
            params.get("revalidate_winners") or "lazy"
        ).strip().lower()
        if revalidate_mode not in _REVALIDATE_MODES:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": (
                    f"revalidate_winners={revalidate_mode!r} not in "
                    f"{sorted(_REVALIDATE_MODES)!r}"
                ),
            }
        # Explicit bypass for the cross-call dedup. Useful when an
        # operator deliberately wants to re-bench a previously-rejected
        # fingerprint (e.g. after a kernel/driver upgrade changed the
        # hardware's behaviour). Default False so the common case dedupes.
        force_retest = bool(params.get("force_retest"))

        # Roofline bottleneck the Coordinator read at settle time (from
        # ``SharedState.roofline_snapshots[-1].roofline_bound_kind``).
        # Routes the settle grid: compute-bound prunes the
        # det_95/det_90/det_85 rungs (a GFX-clock descent can't help a
        # compute-bound shape);
        # memory-bound omits the memory axis (stepping memory down can't
        # help). Only consumed on the default (no explicit grid) settle
        # path.
        bound_kind = str(params.get("bound_kind") or "unknown").strip().lower()

        # Probe powercap range — drives default grid AND enforces ceiling
        # when the operator didn't pass one.
        probed_devices: tuple[int, ...] = tuple(
            int(d) for d in (params.get("devices") or ())
        )
        floor_w = int(
            params.get("power_cap_floor_w", self.default_power_cap_floor_w)
        )
        ceiling_w = int(params.get("power_cap_ceiling_w") or 0)
        probed_range: tuple[int, int] | None = None
        # ``host_state_applied`` snapshot from the previous successful
        # power_management call (injected by Coordinator from
        # SharedState). Used for lazy cache-staleness detection: if
        # the live probe diverges from ``measured_state``, something
        # (reboot / operator tweak / thermal event) shifted the host
        # underneath the cache and lazy mode escalates to re-validating
        # prior winners.
        prior_host_state = params.get("host_state_applied")
        top_sclk_mhz: int | None = None
        gfx_high_idx: int | None = None
        mclk_levels: dict[str, Any] = {}
        cache_stale = False
        if not dry_run:
            if not _which_rocm_smi():
                return {
                    "status": "failed",
                    "error_class": "rocm_smi_unavailable",
                    "error": (
                        "rocm-smi not found on PATH; set dry_run=true to "
                        "validate the grid shape without touching the GPU"
                    ),
                }
            probed_range = _probe_powercap_range(probed_devices)
            if probed_range:
                hardware_min, hardware_max = probed_range
                # Lift the floor to the GPU's hardware-reported minimum
                # when it's STRICTER than the operator/default soft floor.
                # This eliminates the long-standing "we rejected a sane-
                # looking proposal because rocm-smi would have rejected
                # it anyway" race: e.g. a 175 W proposal on a GPU whose
                # hardware minimum is 200 W is silently doomed — better
                # to fail fast here with a clear "below floor" message
                # tied to the hardware value than to wait for the
                # setter to error out.
                #
                # The current upstream ``rocm-smi`` CLI does not expose
                # a hardware-minimum reading, so :func:`_probe_powercap_range`
                # returns ``hardware_min=0`` as a sentinel meaning
                # "no hardware floor available"; the comparison below
                # is then a no-op and ``floor_w`` keeps the operator/
                # default soft value. Kept as live code so a future
                # CLI revision (or an ``amd-smi`` re-target) that does
                # expose the minimum picks it up automatically.
                if hardware_min > floor_w:
                    log.info(
                        "power_management: lifting floor %dW → %dW "
                        "(rocm-smi reports hardware min)",
                        floor_w, hardware_min,
                    )
                    floor_w = hardware_min
                # Symmetric ceiling-from-probe (unchanged behaviour):
                # widen the ceiling to the hardware max when the
                # operator didn't supply one.
                if ceiling_w == 0:
                    ceiling_w = hardware_max
            # Clock probes are best-effort. ``top_sclk_mhz`` seeds the
            # GFX determinism ladder (100/95/90% of it) and comes from
            # ``--showsclkrange`` (the clean DVFS range) with the
            # ``--showclkfrq`` maximum as a fallback. ``gfx_high_idx`` is
            # the sclk DPM index with the HIGHEST frequency (from the same
            # ``--showclkfrq`` payload) — used ONLY to pin GFX high on the
            # memory rows; the index is chosen by frequency (not by the
            # largest index) because the DPM table is coarse and scrambled,
            # and there is no GFX downregulation-by-index path.
            # ``mclk_levels`` drives the memory-axis capability gate (>= 2
            # selectable mclk levels). Each degrades cleanly when its
            # payload is unavailable.
            clkfrq_raw = _probe_clkfrq_raw(probed_devices)
            top_sclk_mhz = _probe_top_sclk_mhz(probed_devices, data=clkfrq_raw)
            gfx_high_idx = _gfx_high_sclk_idx(clkfrq_raw)
            mclk_levels = _probe_mclk_levels(
                probed_devices, clkfrq_data=clkfrq_raw,
            )
            # Lazy cache-staleness check — only meaningful when there's
            # a prior winner cached AND mode is 'lazy' (always/never
            # short-circuit the check entirely). 'always' re-validates
            # unconditionally; 'never' trusts the cache unconditionally.
            if revalidate_mode == "lazy" and isinstance(prior_host_state, dict):
                current_state = _probe_current_state(probed_devices)
                cached_measured = prior_host_state.get("measured_state") or {}
                cache_stale = _host_state_is_stale(
                    current=current_state, cached=cached_measured,
                )
                if cache_stale:
                    log.warning(
                        "power_management: cached host_state diverged from "
                        "live probe — cached=%s current=%s; lazy mode will "
                        "re-validate prior winners",
                        cached_measured, current_state,
                    )

        # Resolve grid: LLM-supplied wins; otherwise synthesize the
        # roofline-routed settle grid (auto_baseline + high + the GFX
        # determinism ladder + the capability-gated memory rows) from
        # the probed ceiling, the roofline ``bound_kind``, the top sclk,
        # the GFX-high sclk index, and the mclk levels.
        grid_override = params.get("grid")
        try:
            variants, grid_degraded = self._resolve_variants(
                grid_override=grid_override,
                probed_range=probed_range,
                floor_w=floor_w,
                ceiling_w=ceiling_w,
                bound_kind=bound_kind,
                top_sclk_mhz=top_sclk_mhz,
                sclk_top_idx=gfx_high_idx,
                mclk_levels=mclk_levels,
            )
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
            }

        # Grid self-check: when the roofline + capability said a ladder
        # SHOULD have produced rows but it came back empty, surface a
        # structured ``grid_degraded`` field (and log.warning) rather
        # than silently collapsing to an auto/high-only sweep. A
        # capability skip (compute-bound pruning, < 2 mclk levels) is
        # the gate firing as intended and is NOT flagged here.
        if grid_degraded:
            log.warning(
                "power_management: grid degraded — expected ladder(s) "
                "produced 0 rows: %s", grid_degraded,
            )

        # --- Cross-call dedup + prior-winner re-validation -----------------
        # Drop any fresh variant whose fingerprint is already in
        # ``tested`` (unless ``force_retest=true``). Both the LLM grid
        # and the default grid pass through the same filter so an LLM
        # rename of an already-tested config collapses to the row that
        # was rejected last round. Prior accepted winners are EXEMPT
        # from the drop (the re-validation phase below benches them
        # again on purpose).
        dropped_for_dedup: list[dict[str, Any]] = []
        if not force_retest and prior_tested_fps:
            kept: list[PowerVariant] = []
            for v in variants:
                fp = power_variant_fingerprint(v.to_dict())
                # The ``auto_baseline`` row is the reference the whole
                # settle sweep gates against — it MUST run every round,
                # so it is never deduped even if a prior round tested
                # the same perflevel=auto + cap-max + fan-max state.
                if v.note == "auto_baseline":
                    kept.append(v)
                    continue
                if fp in prior_tested_fps and fp not in prior_accepted_fps:
                    dropped_for_dedup.append({
                        "name": v.name,
                        "fingerprint": fp,
                        "reason": "already_tested",
                    })
                    log.info(
                        "power_management: dedup-drop variant=%s "
                        "(fingerprint %s already in tested)",
                        v.name, fp,
                    )
                    continue
                kept.append(v)
            variants = kept

        # Phase-0: rehydrate prior winners for re-validation when the
        # revalidation mode + cache state warrant it. They're prepended
        # to the variant list so they bench BEFORE any fresh exploration
        # (so a quick re-confirmation of "yes, the prior winner still
        # holds" arrives before the slower fresh sweep, and so a
        # promote-flow on the prior winner is the round's natural
        # default when nothing new wins).
        revalidation_variants: list[PowerVariant] = []
        if prior_accepted:
            if revalidate_mode == "always":
                should_revalidate = True
            elif revalidate_mode == "never":
                should_revalidate = False
            else:  # lazy
                # Lazy means "re-validate only on detected drift". When
                # the cache is fresh we trust it; the prior winner is
                # already applied to the host and a re-bench would be
                # redundant work. Drift escalates to a full re-bench.
                should_revalidate = cache_stale
            if should_revalidate:
                seen_fps: set[str] = set()
                for entry in prior_accepted:
                    pv = _variant_from_accepted(entry)
                    if pv is None:
                        continue
                    fp = power_variant_fingerprint(pv.to_dict())
                    if fp in seen_fps:
                        continue
                    seen_fps.add(fp)
                    # Hard-skip prior winners that would violate the
                    # current effective cap bounds — a probe-lifted
                    # floor (e.g. rocm-smi now reports min=300W when
                    # the old winner was at 250W) means the historical
                    # value isn't a valid bench candidate any more.
                    reason = _enforce_cap_bounds(
                        pv, floor_w=floor_w, ceiling_w=ceiling_w,
                    )
                    if reason is not None:
                        log.info(
                            "power_management: skipping revalidate of "
                            "prior winner %s (%s)", pv.name, reason,
                        )
                        continue
                    revalidation_variants.append(pv)
                if revalidation_variants:
                    log.info(
                        "power_management: revalidating %d prior "
                        "winner(s) (mode=%s, cache_stale=%s)",
                        len(revalidation_variants),
                        revalidate_mode, cache_stale,
                    )
                    variants = revalidation_variants + variants

        if not variants:
            # Differentiate the two empty-grid causes so the LLM /
            # operator gets a clear error message instead of a generic
            # "empty_grid". When dedup eliminated everything, the right
            # next move is ``force_retest=true`` or an entirely new
            # grid; when the probe failed AND no operator grid was
            # supplied, it's "supply a grid".
            #
            # Note the asymmetry: ``dropped_for_dedup`` populates only
            # from the ``variants`` list after it was synthesized, so
            # if ``probed_range is None`` (no probe) AND
            # ``grid_override is None``, ``_resolve_variants`` already
            # returned ``[]`` and we never entered the dedup loop —
            # ``dropped_for_dedup`` is empty by construction. Two
            # reachable cases remain: every supplied / synthesized
            # variant was deduped (dropped_for_dedup non-empty), or
            # nothing was synthesized in the first place.
            if dropped_for_dedup:
                error_class = "empty_grid_after_dedup"
                if grid_override:
                    error_msg = (
                        "every supplied grid variant was already tested "
                        "in a previous round; change the knob values to "
                        "produce new fingerprints, or set "
                        "force_retest=true to re-bench the prior set"
                    )
                else:
                    error_msg = (
                        "every default grid variant was already tested "
                        "in a previous round; supply an explicit "
                        "task.params.grid with new fingerprints, or set "
                        "force_retest=true to re-bench the prior set"
                    )
            else:
                error_class = "empty_grid"
                error_msg = (
                    "no power_management variants resolved; supply "
                    "task.params.grid or run on a host where rocm-smi "
                    "--showmaxpower --json reports a per-GPU ceiling"
                )
            return {
                "status": "failed",
                "error_class": error_class,
                "error": error_msg,
                "power_floor_w":   floor_w,
                "power_ceiling_w": ceiling_w,
                "dropped_variants": dropped_for_dedup,
            }

        if dry_run:
            return {
                "status": "succeeded",
                "dry_run": True,
                "base_tput": float(params.get("base_tput") or 0.0),
                "grid_size": len(variants),
                "all_results": [],
                "winners": [],
                "best_variant": None,
                "best_gain_pct": 0.0,
                "power_floor_w":   floor_w,
                "power_ceiling_w": ceiling_w,
                "resolved_grid":   [v.to_dict() for v in variants],
                "final_state":     "untouched",
                "workspace":       output_root.as_posix(),
                "revalidate_mode": revalidate_mode,
                "cache_stale":     cache_stale,
                "dropped_variants": dropped_for_dedup,
                "grid_degraded":   grid_degraded or None,
            }

        # Workload-contract materialization — same pattern as explore/
        # sweep so the per-variant rebench inherits CONC/ISL/OSL/TP/etc.
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            out_name="power_management_base.with_envs.yaml",
        )

        base_extra_args = str(params.get("base_extra_args") or "")
        base_tput = float(params.get("base_tput") or 0.0)
        # Resolution order mirrors ``explore.py:357-359`` (explicit task
        # param > single-node default) — the values come from
        # _grid_runner so we agree with explore on Magpie's noise
        # floor. The PARAMETER NAME ``keep_threshold_pct`` matches
        # :mod:`explore` and :mod:`framework_agent` so the LLM (and any
        # operator script) sees one gate-threshold key across every
        # shallow action's task.params.
        #
        # We unconditionally use the single-node threshold here: the
        # multi-node default (2.0 %) is structurally unreachable
        # because the __call__ entry point refuses multi-node sessions
        # with ``error_class='multi_node_unsupported'`` before we ever
        # reach this line, and dry_run returns earlier still. Keeping
        # an ``is_multi_node()`` branch alive would be defense-in-depth
        # against a future relaxation of the refusal, but it would also
        # paper over the broken state where rocm-smi was applied to one
        # node out of N — we'd rather a future relaxation make an
        # explicit decision about per-node application than silently
        # adopt a noise floor designed for a different shape.
        explicit_gain = params.get("keep_threshold_pct")
        if explicit_gain is not None:
            keep_threshold_pct = float(explicit_gain)
        else:
            keep_threshold_pct = SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT
        variant_timeout_sec = int(
            params.get("variant_timeout_sec", self.variant_timeout_sec)
        )
        # The roofline-routed settle sweep runs on the Coordinator-
        # internal path (no explicit ``params.grid``). It benches a
        # single grid: the ``auto_baseline`` reference row (perflevel
        # auto + cap-max + fan-max, N reps median), then the challenger
        # rows (high + the GFX determinism ladder + the capability-gated
        # memory rows), each once. An explicit LLM grid has no
        # ``auto_baseline`` row — it is benched as-is against
        # ``base_tput`` (operator owns shape).
        run_settle = grid_override is None
        auto_reps = max(1, int(
            params.get("auto_baseline_reps") or _AUTO_BASELINE_REPS
        ))

        # On the settle path the ``auto_baseline`` row IS the kernel
        # baseline (same hardware state the kernel climbed under: auto +
        # cap-max + fan-max), so its median seeds both the winner gate
        # reference and the attribution baseline — no separate
        # vendor-default measurement is taken (the cap + fan stay maxed
        # at all times). The LLM-grid path keeps the legacy
        # ``measure_kernel_baseline`` behaviour.
        kernel_baseline_tput: float | None = None
        kernel_baseline_reps: int = 0

        all_results: list[dict[str, Any]] = []
        best_entry: dict[str, Any] | None = None
        auto_entry: dict[str, Any] | None = None
        final_state = "reset_to_default"

        async def _bench(
            v: PowerVariant, *, rep: int | None = None,
        ) -> dict[str, Any]:
            return await self._run_one_variant(
                v,
                rep=rep,
                base_yaml_path=config_path,
                base_extra_args=base_extra_args,
                base_tput=base_tput,
                output_root=output_root,
                variant_timeout_sec=variant_timeout_sec,
                resolved_model=resolved_model,
                resolved_gpu=resolved_gpu,
                override_script=override_script,
                override_result_dir=override_result_dir,
            )

        def _maybe_promote_best(entry: dict[str, Any]) -> None:
            nonlocal best_entry
            # The auto_baseline row is the reference the gate measures
            # against — never a challenger / winner.
            if (entry.get("power_settings") or {}).get("note") == "auto_baseline":
                return
            if (
                entry["status"] == "succeeded"
                and isinstance(entry.get("output_throughput"), (int, float))
                and entry["output_throughput"] > 0
                and (
                    best_entry is None
                    or entry["output_throughput"]
                    > float(best_entry.get("output_throughput") or 0.0)
                )
            ):
                best_entry = entry

        try:
            for v in variants:
                if v.note == "auto_baseline":
                    # Reference row: apply once, bench N reps, take the
                    # median so one noisy run can't move the denominator
                    # the whole sweep gates against. Collapse the reps to
                    # a single representative entry carrying the median.
                    samples: list[float] = []
                    rep_entries: list[dict[str, Any]] = []
                    for rep in range(auto_reps):
                        e = await _bench(v, rep=rep)
                        rep_entries.append(e)
                        t = e.get("output_throughput")
                        if (
                            e.get("status") == "succeeded"
                            and isinstance(t, (int, float)) and t > 0
                        ):
                            samples.append(float(t))
                    med = median(samples) if samples else 0.0
                    entry = dict(rep_entries[0]) if rep_entries else {
                        "variant_name": v.name,
                        "power_settings": v.to_dict(),
                        "status": "failed",
                        "output_throughput": None,
                    }
                    entry["output_throughput"] = med if med > 0 else None
                    entry["status"] = "succeeded" if med > 0 else "failed"
                    entry["gain_pct"] = 0.0
                    entry["reps"] = len(samples)
                    auto_entry = entry
                    all_results.append(entry)
                else:
                    entry = await _bench(v)
                    all_results.append(entry)
                    _maybe_promote_best(entry)
        except Exception as exc:  # noqa: BLE001 — guarantee reset on any failure
            # Convert the mid-sweep crash into a structured failure
            # payload so the Coordinator's
            # ``_promote_to_shared_state`` branch for power_management
            # can clear ``SharedState.host_state_applied`` (it keys
            # off ``final_state='reset_after_failure'``). If we
            # re-raised, ``SubAgentRunner.run_task`` would surface
            # ``SubAgentResult(state='failed', result={})`` — the
            # empty result dict never reaches the audit branch, so
            # ``host_state_applied`` would stay pointing at a winner
            # whose state we just blew away with ``_reset_defaults``.
            #
            # We still reset defaults inside the handler (defense in
            # depth — the ``finally`` below also resets, but the
            # logger call may fail and skip it on edge cases like a
            # broken stderr fd). Phase results gathered so far are
            # preserved so the LLM can see which variants succeeded
            # before the crash.
            log.exception("power_management: unhandled error mid-sweep")
            _reset_defaults()
            return {
                "status": "failed",
                "error_class": "unhandled_exception",
                "error": repr(exc),
                "base_tput":        base_tput,
                "grid_size":        len(all_results),
                "all_results":      list(all_results),
                "winners":          [],
                "best_variant":     None,
                "best_gain_pct":    0.0,
                "power_floor_w":    floor_w,
                "power_ceiling_w":  ceiling_w,
                "probed_range_w":   list(probed_range) if probed_range else None,
                "final_state":      "reset_after_failure",
                "workspace":        output_root.as_posix(),
                "bound_kind":       bound_kind,
                "grid_degraded":    grid_degraded or None,
                "revalidate_mode":  revalidate_mode,
                "cache_stale":      cache_stale,
                "dropped_variants": dropped_for_dedup,
                # Skip ledger write on crash: a partial-round update
                # would persist tested fingerprints whose
                # ``power_settings`` describe a GPU state that didn't
                # actually run to completion. Next round restarts
                # cleanly from the pre-crash ledger.
                "power_management_search_update": None,
                "host_state_applied": None,
            }
        finally:
            # Always reset between the last variant and "apply winner"
            # so the apply step doesn't accidentally inherit residual
            # determinism / clock pins from the previous loop iteration.
            _reset_defaults()

        # Gain reference. On the settle path the ``auto_baseline`` row's
        # median IS the reference — it is the incumbent (perflevel auto +
        # cap-max + fan-max, the same hardware state the kernel climbed
        # under), re-measured co-timed with the challengers. A challenger
        # must clear the noise floor OVER auto_baseline to win; otherwise
        # we keep auto. Falls back to ``base_tput`` on the LLM-grid path
        # (no auto_baseline row exists — operator owns the shape there)
        # or when the auto_baseline reps all failed.
        reference_tput = base_tput
        reference_source = "base_tput"
        if run_settle and auto_entry is not None:
            mt = auto_entry.get("output_throughput")
            if isinstance(mt, (int, float)) and mt > 0:
                reference_tput = float(mt)
                reference_source = "auto_baseline"
                # The auto_baseline IS the kernel baseline (same hardware
                # state), so its median seeds attribution directly — no
                # separate vendor-default measurement (cap + fan stay
                # maxed at all times).
                kernel_baseline_tput = reference_tput
                kernel_baseline_reps = int(auto_entry.get("reps") or 0)
        # Recompute every succeeded row's gain against the chosen
        # reference so the keep gate, the winners list, and the ledger
        # all agree on the denominator. The auto_baseline row's own gain
        # is held at ~0 (it is the reference), so it never clears the
        # keep gate — auto is preserved via the "keep auto" path below,
        # not as a "fresh winner".
        if reference_tput > 0:
            for entry in all_results:
                if entry is auto_entry:
                    entry["gain_pct"] = 0.0
                    continue
                if entry.get("status") != "succeeded":
                    continue
                t = entry.get("output_throughput")
                if isinstance(t, (int, float)) and t > 0:
                    entry["gain_pct"] = (
                        (float(t) - reference_tput) / reference_tput * 100.0
                    )
                else:
                    entry["gain_pct"] = 0.0

        # LLM-grid path keeps the legacy vendor-default attribution
        # baseline (the settle path already seeded it from auto_baseline
        # above). Best-effort: a failed bench leaves it None.
        if measure_kernel_baseline and not run_settle:
            kernel_baseline_tput, kernel_baseline_reps = (
                await self._measure_kernel_only_baseline(
                    base_yaml_path=config_path,
                    base_extra_args=base_extra_args,
                    output_root=output_root,
                    variant_timeout_sec=variant_timeout_sec,
                    resolved_model=resolved_model,
                    resolved_gpu=resolved_gpu,
                    override_script=override_script,
                    override_result_dir=override_result_dir,
                )
            )

        # Apply the round outcome so the chosen state persists past the
        # action boundary (and into the remaining sweep / close phases;
        # see playbook §"Why rocm-smi, not a server flag"). Terminal
        # states:
        #   applied_best     — a challenger cleared the keep gate over
        #                      auto_baseline: apply it (+ cap-max + fan-max).
        #   kept_incumbent   — no challenger won: keep auto (perflevel
        #                      auto + cap-max + fan-max) — the incumbent.
        #   reset_to_default — LLM-grid path with no winner and no
        #                      incumbent to keep: strip to defaults.
        # ``applied_smi_commands`` captures the EXACT shell invocations
        # that re-established the chosen state — feeds the
        # host_state_applied snapshot so the final report can render a
        # verbatim replication recipe for the GPU.
        applied_smi_commands: list[str] = []
        host_state_applied: dict[str, Any] | None = None
        incumbent_snapshot = params.get("host_state_applied")
        if best_entry is not None and self._is_winner(
            best_entry, reference_tput, keep_threshold_pct,
        ):
            winner_v = self._lookup_variant(variants, best_entry["variant_name"])
            if winner_v is not None:
                try:
                    applied_smi_commands = _apply_variant(winner_v)
                    final_state = "applied_best"
                    host_state_applied = _build_host_state_snapshot(
                        winner_variant=winner_v,
                        smi_commands=applied_smi_commands,
                        probed_range=probed_range,
                        top_sclk_mhz=top_sclk_mhz,
                        gain_pct=best_entry.get("gain_pct"),
                        session_dir=output_root.as_posix(),
                        task_id=ctx.task.task_id,
                    )
                except RuntimeError as exc:
                    log.warning(
                        "power_management: failed to re-apply winning "
                        "variant=%s (%s); leaving defaults",
                        best_entry["variant_name"], exc,
                    )
                    _reset_defaults()
                    host_state_applied = None
                    final_state = "reset_after_failure"
        elif run_settle:
            # No challenger cleared the gate → keep auto: re-apply the
            # auto_baseline state (perflevel auto + cap-max + fan-max),
            # which is the incumbent the kernel climbed under. This state
            # holds through the remaining sweep / close phases and a
            # resume re-applies it from host_state_applied.
            auto_v = self._lookup_variant(variants, "auto_baseline")
            if auto_v is not None:
                try:
                    applied_smi_commands = _apply_variant(auto_v)
                    final_state = "kept_incumbent"
                    host_state_applied = _build_host_state_snapshot(
                        winner_variant=auto_v,
                        smi_commands=applied_smi_commands,
                        probed_range=probed_range,
                        top_sclk_mhz=top_sclk_mhz,
                        gain_pct=0.0,
                        session_dir=output_root.as_posix(),
                        task_id=ctx.task.task_id,
                    )
                    log.info(
                        "power_management: no challenger beat auto_baseline; "
                        "kept auto (perflevel auto + cap-max + fan-max), "
                        "reference=%.2f", reference_tput,
                    )
                except RuntimeError as exc:
                    log.warning(
                        "power_management: failed to re-apply auto_baseline "
                        "state (%s); leaving defaults", exc,
                    )
                    _reset_defaults()
                    host_state_applied = None
                    final_state = "reset_after_failure"
        else:
            # LLM-grid path, no winner: keep the incumbent kernel state
            # (auto + cap-max + fan-max) when it beats the measured
            # vendor-default baseline; otherwise strip to vendor defaults.
            incumbent_beats_defaults = False
            if (
                isinstance(incumbent_snapshot, dict)
                and incumbent_snapshot.get("smi_commands")
            ):
                if kernel_baseline_tput is None or kernel_baseline_tput <= 0:
                    incumbent_beats_defaults = True
                elif reference_tput > kernel_baseline_tput * (
                    1.0 + keep_threshold_pct / 100.0
                ):
                    incumbent_beats_defaults = True
            if incumbent_beats_defaults:
                reapply_host_power_state(incumbent_snapshot)
                final_state = "kept_incumbent"
                host_state_applied = dict(incumbent_snapshot)
                log.info(
                    "power_management: no fresh winner; kept incumbent "
                    "state (reference=%.2f > defaults baseline=%s)",
                    reference_tput, kernel_baseline_tput,
                )
            else:
                _reset_defaults()
                final_state = "reset_to_default"
                host_state_applied = None

        winners = [
            e for e in all_results
            if self._is_winner(e, reference_tput, keep_threshold_pct)
        ]
        if best_entry is None:
            status = "failed" if not all_results else "no_winners"
        elif not winners:
            status = "no_winners"
        else:
            status = "succeeded"

        # --- Build the power_management_search ledger update -------------
        # Mirror of explore_search: append every tested fingerprint to
        # ``tested``, the non-winners to ``rejected``. ``accepted`` is
        # left untouched here — Coordinator owns promote semantics and
        # writes accepted via :meth:`record_power_management_accepted`.
        winner_fp_set = {
            power_variant_fingerprint(e.get("power_settings") or {})
            for e in winners
        }
        tested_update = dict(prior_tested)
        rejected_update = list(search.get("rejected") or [])
        name_index = dict(search.get("name_index") or {})
        round_id = f"power_management-{int(search.get('cursor') or 0) + 1:03d}"
        ts = datetime.now(timezone.utc).isoformat()
        round_tested_fps: list[str] = []
        round_winner_fps: list[str] = []
        for entry in all_results:
            ps = entry.get("power_settings") or {}
            fp = power_variant_fingerprint(ps)
            round_tested_fps.append(fp)
            tested_update[fp] = {
                "name": entry.get("variant_name"),
                "power_settings": dict(ps),
                "note": ps.get("note") or "",
                "status": entry.get("status"),
                "tput": entry.get("output_throughput"),
                "gain_pct": entry.get("gain_pct"),
                "base_tput": base_tput,
                "round_id": round_id,
                "ts": ts,
                "fingerprint": fp,
            }
            if entry.get("variant_name"):
                name_index[str(entry["variant_name"])] = fp
            is_round_winner = (
                entry.get("status") == "succeeded" and fp in winner_fp_set
            )
            if is_round_winner:
                round_winner_fps.append(fp)
                continue
            rejected_update.append({
                "name": entry.get("variant_name"),
                "power_settings": dict(ps),
                "fingerprint": fp,
                "reason": (
                    "failed" if entry.get("status") != "succeeded"
                    else "not_keep"
                ),
                "gain_pct": entry.get("gain_pct"),
                "tput": entry.get("output_throughput"),
            })

        # Dedup rejected by fingerprint (keep the latest reason) and
        # drop anything that's already in ``accepted`` so a previously-
        # rejected variant that later won doesn't appear in both buckets.
        accepted_fps_now = set(prior_accepted_fps)
        rejected_dedup: dict[str, dict[str, Any]] = {}
        for r_entry in rejected_update:
            fp = str(r_entry.get("fingerprint") or "")
            if not fp or fp in accepted_fps_now:
                continue
            rejected_dedup[fp] = r_entry

        search_update = {
            "schema_version": 1,
            "accepted": list(prior_accepted),  # Coordinator owns mutations
            "rejected": list(rejected_dedup.values()),
            "tested": tested_update,
            "name_index": name_index,
            "cursor": len(tested_update),
            "last_round": {
                "round_id": round_id,
                "action": "power_management",
                "base_tput": base_tput,
                "tested": round_tested_fps,
                "round_winners": round_winner_fps,
                "revalidate_mode": revalidate_mode,
                "cache_stale": cache_stale,
                "bound_kind": bound_kind,
                "ts": ts,
            },
        }

        return {
            "status":           status,
            "base_tput":        base_tput,
            # Gain reference the keep gate + reported ``gain_pct`` are
            # measured against. ``auto_baseline`` means the co-timed
            # auto_baseline median supplied it (the roofline-routed
            # settle path); ``base_tput`` means the climb number was used
            # (LLM grid, or the auto_baseline reps all failed).
            "reference_tput":   reference_tput,
            "reference_source": reference_source,
            # The kernel baseline the power attribution is measured
            # against. On the settle path this is the auto_baseline
            # median (same hardware state the kernel climbed under); on
            # the LLM-grid path it is the vendor-default kernel-only
            # median when ``measure_kernel_baseline`` was set, else None.
            "kernel_baseline_tput": kernel_baseline_tput,
            "kernel_baseline_reps": kernel_baseline_reps,
            "grid_size":        len(all_results),
            "all_results":      all_results,
            "winners":          winners,
            "best_variant":     best_entry,
            "best_gain_pct":    float(best_entry["gain_pct"]) if best_entry else 0.0,
            "power_floor_w":    floor_w,
            "power_ceiling_w":  ceiling_w,
            "probed_range_w":   list(probed_range) if probed_range else None,
            "final_state":      final_state,
            "workspace":        output_root.as_posix(),
            # Roofline-routed settle surfaces. ``bound_kind`` echoes the
            # roofline bottleneck that routed the grid (gates the
            # det_95/det_90/det_85 rungs + the memory axis). ``grid_degraded``
            # is a structured map of expected-but-empty ladders (None
            # when every expected ladder materialised) — the grid
            # self-check refuses to silently collapse to auto/high-only.
            "bound_kind":       bound_kind,
            "grid_degraded":    grid_degraded or None,
            # Cross-call deepening surfaces. ``revalidate_mode`` echoes
            # the operator's choice (with the executor's default filled
            # in); ``cache_stale`` tells the next-tick prompt whether
            # the lazy revalidate path tripped; ``dropped_variants``
            # surfaces the dedup output for prompt rendering.
            "revalidate_mode":  revalidate_mode,
            "cache_stale":      cache_stale,
            "dropped_variants": dropped_for_dedup,
            # Cross-call ledger + host-state snapshot. The Coordinator
            # consumes both fields in :meth:`_promote_to_shared_state`:
            # the search update merges into SharedState.power_management_
            # search (driving the next call's dedup + revalidate logic),
            # and host_state_applied becomes the authoritative "what is
            # the GPU set to right now" record surfaced by the final
            # report (cleared to None when no winner was applied).
            "power_management_search_update": search_update,
            "host_state_applied": host_state_applied,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_variants(
        self,
        *,
        grid_override: Any,
        probed_range: tuple[int, int] | None,
        floor_w: int,
        ceiling_w: int,
        bound_kind: str | None = None,
        top_sclk_mhz: int | None = None,
        sclk_top_idx: int | None = None,
        mclk_levels: dict[str, Any] | None = None,
    ) -> tuple[list[PowerVariant], dict[str, str]]:
        """Build the validated variant list + grid-degraded report.

        Returns ``(variants, grid_degraded)``. ``bound_kind`` /
        ``top_sclk_mhz`` / ``sclk_top_idx`` / ``mclk_levels`` are
        forwarded to :func:`_build_settle_grid` so the default (no
        explicit ``grid_override``) settle path builds the
        roofline-routed determinism-only grid; they are ignored on the
        LLM-grid path (the operator owns shape there, so its
        ``grid_degraded`` is always empty).
        """
        if grid_override is not None:
            if not isinstance(grid_override, list):
                raise ValueError(
                    f"grid must be a list, got {type(grid_override).__name__}"
                )
            out: list[PowerVariant] = []
            for idx, raw in enumerate(grid_override):
                v = _build_variant_from_payload(raw, idx)
                reason = _enforce_cap_bounds(
                    v, floor_w=floor_w, ceiling_w=ceiling_w,
                )
                if reason is not None:
                    raise ValueError(f"grid[{idx}]: {reason}")
                out.append(v)
            return out, {}

        if probed_range is None:
            return [], {}
        min_w, max_w = probed_range
        if min_w < floor_w:
            min_w = floor_w
        if ceiling_w > 0 and max_w > ceiling_w:
            max_w = ceiling_w
        if max_w <= min_w:
            return [], {}
        return _build_settle_grid(
            cap_w=max_w,
            bound_kind=bound_kind,
            top_sclk_mhz=top_sclk_mhz,
            sclk_top_idx=sclk_top_idx,
            mclk_levels=mclk_levels,
        )

    @staticmethod
    def _lookup_variant(
        variants: list[PowerVariant], name: str,
    ) -> PowerVariant | None:
        for v in variants:
            if v.name == name:
                return v
        return None

    def _is_winner(
        self,
        entry: dict[str, Any],
        reference_tput: float,
        keep_threshold_pct: float,
    ) -> bool:
        if entry.get("status") != "succeeded":
            return False
        gain = entry.get("gain_pct")
        if not isinstance(gain, (int, float)):
            return False
        if reference_tput <= 0:
            # Without a usable reference we accept any positive throughput
            # as a "winner" so the action surfaces SOME best_variant. The
            # Coordinator never promotes this onto current_best because the
            # action is not in the {explore,sweep} current_best promotion set.
            return float(entry.get("output_throughput") or 0.0) > 0
        # Inclusive (>=) to match framework_agent's per-iteration gate
        # (framework_agent.py:832 ``delta_pct >= keep_threshold_pct``). Each of
        # our iterations is structurally one framework_agent bench-and-
        # check, so using the same comparator here means a variant that
        # exactly clears the noise floor is treated identically by both
        # actions.
        return float(gain) >= keep_threshold_pct

    async def _measure_kernel_only_baseline(
        self,
        *,
        base_yaml_path: Path,
        base_extra_args: str,
        output_root: Path,
        variant_timeout_sec: int,
        resolved_model: str,
        resolved_gpu: str,
        override_script: str | None,
        override_result_dir: str | None,
        reps: int = _KERNEL_BASELINE_REPS,
    ) -> tuple[float | None, int]:
        """Measure the combo's kernel-only throughput at vendor defaults.

        Used by the KERNEL->SWEEP settle sweep so the recorded power
        delta reflects the FULL contribution of power tuning for the
        winning combo (combined vs kernel-only) measured against TRUE
        vendor defaults. Resets host power to vendor defaults, runs
        ``reps`` Magpie benches of the base config with no power knobs
        applied (same representative workload shape the rest of KERNEL
        measures at), takes the **median** so a single noisy run can't
        skew the recipe's power-gain number, then resets again.

        Best-effort: returns ``(median, n_samples)`` — the median
        measured throughput and how many reps produced a usable number.
        ``(None, 0)`` when no rep produced a usable number or every rep
        raised; the caller then falls back to the task-supplied
        ``base_tput`` rather than failing the whole settle sweep.
        """
        samples: list[float] = []
        _reset_defaults()
        try:
            for rep in range(max(1, int(reps))):
                try:
                    results: list[VariantResult] = await run_grid(
                        base_yaml_path=base_yaml_path,
                        base_extra_args=base_extra_args,
                        grid=[GridVariant(
                            name=f"power_kernel_only_baseline_rep{rep}",
                            extra_server_args="",  # no power, no args: pure kernel
                            extra_envs={},
                            note="kernel_only_baseline",
                        )],
                        output_root=output_root,
                        variant_timeout_sec=variant_timeout_sec,
                        model_path=resolved_model,
                        gpu_type=resolved_gpu,
                        benchmark_script=override_script,
                        result_dir=override_result_dir,
                    )
                except Exception as exc:  # noqa: BLE001 — never fail the sweep
                    log.warning(
                        "power_management: kernel-only baseline rep %d failed "
                        "(%r); continuing", rep, exc,
                    )
                    continue
                if not results:
                    continue
                rep_tput = results[0].output_throughput or 0.0
                if rep_tput > 0:
                    samples.append(float(rep_tput))
        finally:
            _reset_defaults()
        if not samples:
            return None, 0
        tput = median(samples)
        return (float(tput) if tput > 0 else None), len(samples)

    async def _run_one_variant(
        self,
        v: PowerVariant,
        *,
        rep: int | None = None,
        base_yaml_path: Path,
        base_extra_args: str,
        base_tput: float,
        output_root: Path,
        variant_timeout_sec: int,
        resolved_model: str,
        resolved_gpu: str,
        override_script: str | None,
        override_result_dir: str | None,
    ) -> dict[str, Any]:
        """Apply ``v``, run one Magpie variant, reset defaults, build the row.

        Failure semantics are split deliberately:

        * A failed ``_apply_variant`` (rocm-smi setter raised) is recoverable:
          we record a ``rocm_smi_set_failed`` row, reset defaults, and the
          caller continues to the next variant. The intuition is that one
          knob being unsupported on this revision shouldn't kill the whole
          sweep — the LLM may still get useful signal from the others.
        * A failed ``run_grid`` (Magpie subprocess crash, missing binary,
          etc.) is escalated: we still reset defaults via the ``finally``,
          but we re-raise so the outer ``__call__`` ``except Exception``
          path marks ``final_state="reset_after_failure"`` and the whole
          action fails. That matches the existing behavior of every other
          grid executor (explore / sweep): a Magpie blow-up is
          never silently swallowed.
        """
        snapshot_before: str = ""
        snapshot_after:  str = ""
        # A ``rep`` suffix keeps multi-rep measurements (the auto_baseline
        # reference row) in distinct per-rep result dirs so they don't
        # clobber each other.
        magpie_variant_name = (
            f"power_{v.name}" if rep is None else f"power_{v.name}_rep{rep}"
        )

        try:
            _apply_variant(v)
        except RuntimeError as exc:
            _reset_defaults()
            return {
                "variant_name":   v.name,
                "power_settings": v.to_dict(),
                "status":         "failed",
                "error_class":    "rocm_smi_set_failed",
                "error":          str(exc),
                "output_throughput": None,
                "gain_pct":       0.0,
                "rocm_smi_state_before": snapshot_before,
                "rocm_smi_state_after":  snapshot_after,
            }

        try:
            snapshot_before = _snapshot_state()
            results: list[VariantResult] = await run_grid(
                base_yaml_path=base_yaml_path,
                base_extra_args=base_extra_args,
                grid=[GridVariant(
                    name=magpie_variant_name,
                    extra_server_args="",     # power_management adds no args
                    extra_envs={},
                    note=v.note or "power_management",
                )],
                output_root=output_root,
                variant_timeout_sec=variant_timeout_sec,
                model_path=resolved_model,
                gpu_type=resolved_gpu,
                benchmark_script=override_script,
                result_dir=override_result_dir,
            )
            snapshot_after = _snapshot_state()
        finally:
            _reset_defaults()

        if not results:
            return {
                "variant_name":   v.name,
                "power_settings": v.to_dict(),
                "status":         "failed",
                "error_class":    "run_grid_empty",
                "error":          "run_grid returned no results",
                "output_throughput": None,
                "gain_pct":       0.0,
                "rocm_smi_state_before": snapshot_before,
                "rocm_smi_state_after":  snapshot_after,
            }
        r = results[0]
        d = r.to_dict()
        tput = r.output_throughput or 0.0
        gain_pct = (
            (tput - base_tput) / base_tput * 100.0
            if base_tput > 0 and tput > 0 else 0.0
        )
        d.update({
            "variant_name":          v.name,
            "power_settings":        v.to_dict(),
            "gain_pct":              gain_pct,
            "rocm_smi_state_before": snapshot_before,
            "rocm_smi_state_after":  snapshot_after,
        })
        return d


power_management_executor = PowerManagementExecutor()


__all__ = [
    "POWER_CAP_DEFAULT_FLOOR_W",
    "POWER_MANAGEMENT_DEFAULT_TIMEOUT_SEC",
    "PowerManagementExecutor",
    "PowerVariant",
    "apply_max_climb_state",
    "power_management_executor",
    "reapply_host_power_state",
    "reset_host_power_defaults",
]
