# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session bootstrap + summary helpers for the CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from hyperloom.common.coerce import to_unix
from hyperloom.common.env import forge_explicitly_enabled
from hyperloom.common.gpu_partition import published_shape
from hyperloom.common.gpu_power_settings import SCLK_MAX_MHZ, DeclaredPowerSettings
from hyperloom.common.perf_metric import agentx_enabled as _agentx_enabled, is_agentx_mode
from hyperloom.common.timeutil import now_iso
from hyperloom.plugs import (
    ManifestSources,
    PlugError,
    PlugHost,
    ResolvedManifest,
    activated,
    builtin_registry,
    resolve_manifest,
)
from hyperloom.plugs.resolved import read_resolved, write_resolved
from hyperloom.orchestrator.phases.machine_state import bank_phase_segment
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.common.workload_defaults import (
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_CONC,
    DEFAULT_TP,
    DEFAULT_EP,
    DEFAULT_PRECISION,
)
from ..session.paths import _SESSION_SKELETON
from .model_gate import _load_model_arch, _load_model_config_tags
from ..model_config_utils import summarize_model_config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostedPlugInputs:
    """What a hosted run gives the plug host: the lock layer and the host inputs."""

    lock: Mapping[str, Mapping[str, str]]
    host_info: object


_hosted_inputs: HostedPlugInputs | None = None


def set_hosted_plug_inputs(inputs: HostedPlugInputs | None) -> None:
    """Called by the hosted entry before ``optimize`` starts; ``None`` clears it.  Unset, plug resolution is unchanged."""
    global _hosted_inputs
    _hosted_inputs = inputs


def resolve_plug_manifest(args: argparse.Namespace) -> ResolvedManifest:
    """Resolve the plug manifest for this run and check it can start; exit 2 on a configuration error.

    Reads and writes nothing in a session: it runs before the session is opened or seeded, so a refused flag leaves
    nothing behind. On a resume, the ``plugs.resolved.json`` of the ``--resume-from`` directory is read back
    before any flag is applied.
    """
    resume_from = getattr(args, "resume_from", None)
    registry = builtin_registry()
    hosted = _hosted_inputs
    try:
        sources = ManifestSources(
            profile=getattr(args, "plug_profile", None),
            manifest_path=getattr(args, "plug_manifest", None) or os.environ.get("HYPERLOOM_MANIFEST") or None,
            overrides_path=os.environ.get("HYPERLOOM_MANIFEST_OVERRIDES") or None,
            cli_plugs=tuple(getattr(args, "plug", None) or ()),
            cli_configs=tuple(getattr(args, "plug_config", None) or ()),
            resumed=read_resolved(Path(resume_from).expanduser().resolve()) if resume_from else None,
            locked=hosted.lock if hosted else {},
        )
        manifest = resolve_manifest(registry, sources)
        PlugHost(manifest, registry, session_dir=None).plan()
    except PlugError as exc:
        print(f"ERROR: plug configuration refused: {exc}", file=sys.stderr)
        sys.exit(2)
    return manifest


def open_plug_host(manifest: ResolvedManifest, session_dir: Path) -> PlugHost:
    """Bind a resolved ``manifest`` to the opened session.

    A run that selects nothing (the built-in local profile) writes nothing to the session and prints nothing. A run
    that selects something records the outcome in ``plugs.resolved.json`` and prints the ``Plugs`` line.
    """
    hosted = _hosted_inputs
    host = PlugHost(
        manifest,
        builtin_registry(),
        session_dir=session_dir,
        host_info=hosted.host_info if hosted else None,
    )
    host.plan()
    if not manifest.is_builtin_default:
        write_resolved(session_dir, manifest, host.fingerprint())
        print(f"Plugs           : profile={manifest.profile} fingerprint={host.fingerprint()}")
    return host


@contextmanager
def running_plug_host(host: PlugHost) -> Iterator[PlugHost]:
    """Start ``host``, make it the process's current host, and close it on the way out."""
    with activated(host):
        try:
            host.start()
            yield host
        finally:
            host.close()


def warn_without_target_gpu(args: argparse.Namespace) -> None:
    """Say on stderr that no target GPU was given, so the target analysis writes only its marker file."""
    if (getattr(args, "compare_against_gpu", None) or "").strip():
        return
    print(
        "[target_analysis] no --compare-against-gpu set; will write a "
        "marker JSON at $SESSION_DIR/target_analysis/target_baseline.json "
        "(reason=no_target_gpu_configured) — set --compare-against-gpu "
        "to fetch real InferenceX reference data.",
        file=sys.stderr,
    )


def parse_operator_extra_env(args: argparse.Namespace) -> dict[str, str]:
    """Parse ``--extra-env NAME=VALUE`` pins into a mapping."""
    pins: dict[str, str] = {}
    for item in getattr(args, "extra_env", None) or []:
        key, sep, value = str(item).partition("=")
        if sep and key.strip():
            pins[key.strip()] = value
    return pins


def resolve_model_display_name(args: argparse.Namespace) -> str:
    """Resolve the canonical model identity used for session naming / display."""
    override = (getattr(args, "model_display_name", "") or "").strip()
    if override:
        return override
    return Path(str(getattr(args, "model", "") or "")).name


# Bump when a change makes previously recorded AgentX measurements incomparable.
# The MLPerf client is a different workload, but it is opt-in: the epoch stays
# so aiperf sessions remain resumable. The backend name is what resume compares.
AGENTX_MEASUREMENT_EPOCH = 1


def _requested_framework(args: Any) -> str:
    """--framework, else $FRAMEWORK, else ""."""
    return (getattr(args, "framework", None) or "").strip() or (os.environ.get("FRAMEWORK", "") or "").strip()


def resolve_framework_version(args: argparse.Namespace) -> str:
    """Resolve ``framework_version``: --framework-version > $FRAMEWORK_VERSION > auto-detect > ""."""
    explicit = (getattr(args, "framework_version", None) or "").strip() or (
        os.environ.get("FRAMEWORK_VERSION", "") or ""
    ).strip()
    if explicit:
        return explicit
    framework = _requested_framework(args)
    if not framework:
        return ""
    from ..recipe_snapshot_constants import (
        DEFAULT_FRAMEWORK_VERSION_SLUG,
        detect_framework_version,
    )

    detected = detect_framework_version(framework)
    # Treat the failure-slug as "no info".
    return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected


def seed_grading(framework: str, benchmark_mode: str) -> dict[str, Any]:
    """Resolve the grading axis and its noise band once, at seed, so they can be recorded.

    The resolution reads ``HYPERLOOM_PERF_METRIC`` and ``HYPERLOOM_PERF_NOISE_PCT``. Deriving it again later -- in a
    resumed process, a re-baseline subprocess, or the breakdown export CLOSE drives from a subprocess that often did
    not inherit them -- can name an axis the session never graded on. This is the same reasoning that put
    ``benchmark_mode`` in the state rather than leaving it to the ambient var.
    """
    from hyperloom.common.perf_metric import (
        GRADED_INTVTY,
        GRADED_OUTPUT,
        intvty_serving_grading_enabled,
        parse_intvty_noise_pct,
    )

    from .. import framework_registry

    on_intvty = intvty_serving_grading_enabled(
        scriptable=framework_registry.is_scriptable(framework),
        benchmark_mode=benchmark_mode,
    )
    return {
        "objective": GRADED_INTVTY if on_intvty else GRADED_OUTPUT,
        "noise_pct": parse_intvty_noise_pct(),
    }


def agentx_state_is_stale(state: Any) -> str:
    """Return why a resumed session's AgentX state is unusable, or ``\"\"``."""
    want_mode = "agentx" if _agentx_enabled() else "synthetic"
    had_mode = str(getattr(state, "benchmark_mode", "") or "")
    if had_mode and had_mode != want_mode:
        return (
            f"session was measured in benchmark_mode={had_mode!r} but this run is "
            f"{want_mode!r}; the KEEP ledger keys on server args only, so the two "
            "sets of measurements would overwrite each other"
        )
    if want_mode == "agentx":
        had_epoch = int(getattr(state, "agentx_epoch", 0) or 0)
        if had_epoch != AGENTX_MEASUREMENT_EPOCH:
            return (
                f"session carries AgentX epoch {had_epoch}, this build measures "
                f"epoch {AGENTX_MEASUREMENT_EPOCH}; the recorded results describe "
                "a different workload and cannot anchor or be compared against"
            )
        from hyperloom.common.agentx_workload import agentic_backend

        # Sessions recorded before the backend was persisted are aiperf.
        had_backend = str(getattr(state, "agentx_backend", "") or "") or "aiperf"
        want_backend = agentic_backend()
        if had_backend != want_backend:
            return (
                f"session was measured with agentic backend {had_backend!r} but this "
                f"run is {want_backend!r}; the recorded results describe a different "
                "workload and cannot anchor or be compared against"
            )
    return ""


def latency_budget_scope_error(framework: str | None, requested_ms: float | None) -> str:
    """Return why ``--max-latency-ms`` does not apply to *framework*, or ``\"\"``.

    Scriptable workloads grade on output throughput alone and are the only frameworks compute partitioning places
    work for, so they are the only place a throughput-only gate can buy throughput with per-request latency. AgentX
    serving sessions already REVERT that trade on interactivity, and the fixed ISL/OSL serving mode takes no new
    capability, so the budget is refused there rather than silently doing nothing.
    """
    from .. import framework_registry

    if requested_ms is None or framework_registry.is_scriptable(framework):
        return ""
    name = str(framework or "").strip() or framework_registry.DEFAULT_FRAMEWORK
    return (
        f"--max-latency-ms applies only to scriptable frameworks (xdit, custom); {name!r} is a serving framework. "
        "On AgentX the interactivity objective already refuses a throughput gain bought with per-request latency"
    )


def _budget_resume_conflict(flag: str, unit: str, archived: float, requested: float | None) -> str:
    """Why a resume-time budget differs from the archived one, or ``\"\"``; omitting the flag keeps the archive."""
    if requested is None or float(requested) == archived:
        return ""
    recorded = f"{archived:g} {unit}" if archived > 0 else "no budget"
    return (
        f"{flag} {float(requested):g} differs from the {recorded} this session was "
        "graded under; its KEEPs would be judged against a constraint they were never measured "
        "for. Resume without the flag to keep the recorded budget, or start a fresh session"
    )


def latency_budget_resume_conflict(state: Any, requested_ms: float | None) -> str:
    """Return why ``--max-latency-ms`` cannot apply to a resumed session, or ``\"\"``.

    The recorded KEEPs were graded under the archived budget, so a different
    value would leave them judged against a constraint the new one does not
    state. Omitting the flag keeps the archived budget.
    """
    return _budget_resume_conflict(
        "--max-latency-ms", "ms", float(getattr(state, "latency_budget_ms", 0.0) or 0.0), requested_ms
    )


def power_budget_resume_conflict(state: Any, requested_w: float | None) -> str:
    """Return why ``--max-power-w`` cannot apply to a resumed session, or ``\"\"``."""
    return _budget_resume_conflict(
        "--max-power-w", "W", float(getattr(state, "power_budget_w", 0.0) or 0.0), requested_w
    )


def per_gpu_power_budget_resume_conflict(state: Any, requested: dict[str, float] | None) -> str:
    """Return why ``--max-per-gpu-power-w`` cannot apply to a resumed session, or ``\"\"``."""
    if requested is None:
        return ""
    archived = {str(g): float(w) for g, w in (getattr(state, "power_budget_per_gpu_w", None) or {}).items()}
    if {str(g): float(w) for g, w in requested.items()} == archived:
        return ""
    recorded = ", ".join(f"GPU {g} {w:g} W" for g, w in sorted(archived.items())) or "no per-GPU limits"
    return (
        f"--max-per-gpu-power-w differs from the {recorded} this session was graded under; its KEEPs would be "
        "judged against a constraint they were never measured for. Resume without the flag to keep the recorded "
        "limits, or start a fresh session"
    )


def power_budget_unmeasurable_error(
    *, total_w: float | None, per_gpu_w: dict[str, float] | None, nodes: int, framework: str | None
) -> str:
    """Why a power budget could never be measured on this session, or ``""``; checked at launch and on resume.

    Every round would be ``power_unmeasured``, so the run would spend a whole baseline only to stop as if it were over
    budget.
    """
    if not (total_w and float(total_w) > 0) and not per_gpu_w:
        return ""
    if nodes >= 2:
        return (
            "--max-power-w / --max-per-gpu-power-w cannot be measured on a multi-node session: the GPU power sampler "
            "reads only this host's cards"
        )
    from .. import framework_registry

    if framework_registry.is_scriptable(framework):
        return (
            f"--max-power-w / --max-per-gpu-power-w cannot be measured on the scriptable framework {framework!r}: the "
            "GPU power sampler follows a server's measured phase, and a scriptable round runs no server"
        )
    from hyperloom.common.gpu_power_settings import gpu_power_sampling_unavailable

    reason = gpu_power_sampling_unavailable()
    return f"--max-power-w / --max-per-gpu-power-w cannot be measured: {reason}" if reason else ""


def per_gpu_power_budget_error(
    per_gpu_w: dict[str, float] | None,
    *,
    total_w: float | None,
    available_gpus: set[int] | None,
) -> str:
    """Why the per-GPU power limits cannot be used, or ``""``; checked at launch, before anything runs.

    *available_gpus* is the session's visible mask, or every card amd-smi reported when there is no mask, or
    ``None`` when neither is known (then a listed card cannot be checked, which refuses).
    """
    if not per_gpu_w:
        return ""
    listed = sorted(int(g) for g in per_gpu_w)
    if available_gpus is None:
        return "--max-per-gpu-power-w cannot be checked: no visible-device mask is set and amd-smi reported no GPUs"
    missing = [g for g in listed if g not in available_gpus]
    if missing:
        return (
            f"--max-per-gpu-power-w names GPU(s) {', '.join(map(str, missing))}, which this session cannot use "
            f"(available: {', '.join(map(str, sorted(available_gpus))) or 'none'})"
        )
    if total_w is not None and sum(per_gpu_w.values()) > float(total_w):
        return (
            f"the --max-per-gpu-power-w limits add up to {sum(per_gpu_w.values()):g} W, more than --max-power-w "
            f"{float(total_w):g} W"
        )
    return ""


_MAX_LATENCY_MS = "max_latency_ms"
_MAX_POWER_W = "max_power_w"
_MAX_PER_GPU_POWER_W = "max_per_gpu_power_w"
_NODES = "nodes"
_GPU_POWER_CAP_W = "gpu_power_cap_w"
_GPU_PERF_LEVEL = "gpu_perf_level"
_GPU_SCLK_MAX_MHZ = "gpu_sclk_max_mhz"
_NODE_POWER_MANAGEMENT = "node_power_management"
_NODE_POWER_LIMIT_W = "node_power_limit_w"
_ALLOW_SHARED_NPM_NODE = "allow_shared_npm_node"
_DECLARED = "declared"


def constraint_budgets_from_args(args: Any) -> dict[str, Any]:
    """The session's latency and power budgets from the CLI: the only copy, archived with the session."""
    return {
        "latency_budget_ms": float(getattr(args, _MAX_LATENCY_MS, None) or 0.0),
        "power_budget_w": float(getattr(args, _MAX_POWER_W, None) or 0.0),
        "power_budget_per_gpu_w": dict(getattr(args, _MAX_PER_GPU_POWER_W, None) or {}),
    }


def constraint_budget_resume_conflict(args: Any, state: Any) -> str:
    """Why a resume cannot keep the latency and power budgets the session was graded under, or ``""``."""
    return (
        latency_budget_resume_conflict(state, getattr(args, _MAX_LATENCY_MS, None))
        or power_budget_resume_conflict(state, getattr(args, _MAX_POWER_W, None))
        or per_gpu_power_budget_resume_conflict(state, getattr(args, _MAX_PER_GPU_POWER_W, None))
        or power_budget_unmeasurable_error(
            total_w=getattr(state, "power_budget_w", 0.0),
            per_gpu_w=getattr(state, "power_budget_per_gpu_w", None),
            nodes=max(int(getattr(args, _NODES, 1) or 1), int(getattr(state, _NODES, 1) or 1)),
            framework=state.framework,
        )
    )


def _established_gpu_power_settings(
    declared: Any, *, nodes: int, apply: bool, owner: str, allow_shared_npm_node: bool
) -> tuple[dict[str, Any], str]:
    """Apply the declared settings when permitted, then check and record them; ``(record, error)``.

    The check after applying is the one an operator-set value gets, so a set that did not take is caught the same way.
    With NPM enabled the session must also have every card on the node, unless it is allowed to share it.
    """
    from hyperloom.common.gpu_power_settings import GpuPowerSettingsError

    from . import gpu_power_apply

    applied: dict[str, Any] = {}
    try:
        if apply:
            applied = gpu_power_apply.apply_for_session(declared, nodes=nodes, owner=owner)
        else:
            gpu_power_apply.warn_about_orphaned_cards()
    except GpuPowerSettingsError as exc:
        return {}, f"cannot apply the declared GPU power settings: {exc}"
    record, error = resolve_gpu_power_settings(declared, nodes=nodes, just_applied=bool(applied))
    if applied:
        record["applied"] = applied
    outside, shared = _shared_npm_node(record.get("node"))
    if shared and allow_shared_npm_node:
        record["shared_npm_node"] = outside
    elif shared:
        refusal = (
            f"{shared}: firmware shares one power budget across every card on the node, so work on the others moves "
            "this session's measurements; give the session every GPU on the node, or pass --allow-shared-npm-node "
            "(on every launch and resume) to measure anyway"
        )
        error = f"{error}; {refusal}" if error else refusal
    return record, error


def _shared_npm_node(node: dict[str, Any] | None) -> tuple[list[int] | None, str]:
    """The cards an NPM-enabled node has outside this session and why it is shared, or ``([], "")`` when NPM is off or
    the session has every card; the cards are ``None`` when the session's own cannot be named."""
    from hyperloom.common.gpu_power_settings import NPM, NPM_ENABLED, GpuPowerSettingsError, gpus_outside_session

    if not node or node.get(NPM) != NPM_ENABLED:
        return [], ""
    try:
        outside = gpus_outside_session()
    except GpuPowerSettingsError as exc:
        return None, f"node power management is enabled and the cards this session uses cannot be named ({exc})"
    if not outside:
        return [], ""
    return (
        outside,
        f"node power management is enabled and GPU(s) {', '.join(map(str, outside))} are outside this session",
    )


def _per_gpu_launch_error(args: Any, available_gpus: set[int] | None) -> str:
    return per_gpu_power_budget_error(
        getattr(args, _MAX_PER_GPU_POWER_W, None),
        total_w=getattr(args, _MAX_POWER_W, None),
        available_gpus=available_gpus,
    )


def resolve_launch_gpu_power(args: Any, *, nodes: int, owner: str) -> dict[str, Any]:
    """The launch's GPU power settings record; exits 2 before anything runs when the declared settings cannot be
    applied or a requested power budget cannot be measured or checked. Nothing is applied when a budget is refused."""
    from hyperloom.common.gpu_power_settings import GpuPowerSettingsError, visible_gpu_indices

    gpu_power: dict[str, Any] = {}
    mask: set[int] | None = None
    error = power_budget_unmeasurable_error(
        total_w=getattr(args, _MAX_POWER_W, None),
        per_gpu_w=getattr(args, _MAX_PER_GPU_POWER_W, None),
        nodes=nodes,
        framework=_requested_framework(args),
    )
    if not error and getattr(args, _MAX_PER_GPU_POWER_W, None):
        try:
            mask = visible_gpu_indices()
        except GpuPowerSettingsError as exc:
            error = f"--max-per-gpu-power-w cannot be checked against the visible cards: {exc}"
        else:
            error = _per_gpu_launch_error(args, mask) if mask is not None else ""
    if not error:
        gpu_power, error = _established_gpu_power_settings(
            DeclaredPowerSettings(
                power_cap_w=getattr(args, _GPU_POWER_CAP_W, None),
                perf_level=getattr(args, _GPU_PERF_LEVEL, None) or "",
                sclk_max_mhz=getattr(args, _GPU_SCLK_MAX_MHZ, None),
                npm=getattr(args, _NODE_POWER_MANAGEMENT, None) or "",
                node_power_limit_w=getattr(args, _NODE_POWER_LIMIT_W, None),
            ),
            nodes=nodes,
            apply=bool(getattr(args, "apply_gpu_power_settings", False)),
            owner=owner,
            allow_shared_npm_node=bool(getattr(args, _ALLOW_SHARED_NPM_NODE, False)),
        )
    if not error and mask is None:
        # No mask, so the cards amd-smi reported are the ones the session may use; only known after reading them.
        error = _per_gpu_launch_error(args, {int(gpu) for gpu in (gpu_power.get("observed") or {})} or None)
    if error:
        print(f"ERROR: {error}.", file=sys.stderr)
        sys.exit(2)
    return gpu_power


def resume_gpu_power_settings(args: Any, state: Any) -> tuple[dict[str, Any], str]:
    """The resumed session's power settings: an omitted flag re-asserts the archived value; ``(record, error)``.

    A session that applied its settings restored them when it exited, so its resume must apply them again.
    """
    archived = dict(getattr(state, "gpu_power_settings", None) or {})
    declared = dict(archived.get(_DECLARED) or {})
    apply = bool(getattr(args, "apply_gpu_power_settings", False))
    record, error = _established_gpu_power_settings(
        DeclaredPowerSettings(
            power_cap_w=getattr(args, _GPU_POWER_CAP_W, None) or declared.get("power_cap_w"),
            perf_level=getattr(args, _GPU_PERF_LEVEL, None) or declared.get("perf_level") or "",
            sclk_max_mhz=getattr(args, _GPU_SCLK_MAX_MHZ, None) or declared.get(SCLK_MAX_MHZ),
            npm=getattr(args, _NODE_POWER_MANAGEMENT, None) or declared.get("npm") or "",
            node_power_limit_w=getattr(args, _NODE_POWER_LIMIT_W, None) or declared.get("node_power_limit_w"),
        ),
        nodes=max(int(getattr(args, _NODES, 1) or 1), int(getattr(state, _NODES, 1) or 1)),
        apply=apply,
        owner=str(getattr(state, "session_id", "") or ""),
        allow_shared_npm_node=bool(getattr(args, _ALLOW_SHARED_NPM_NODE, False)),
    )
    if error and archived.get("applied") and not apply:
        error += (
            "; this session applied its power settings itself and restored them when it exited, so pass "
            "--apply-gpu-power-settings to apply them again"
        )
    return record, error


def _node_power_settings(declared: DeclaredPowerSettings) -> tuple[dict[str, Any] | None, str]:
    """The node's NPM reading and why the declared values do not hold; ``(None, "")`` when it cannot be read and
    nothing about it was declared (an amd-smi before ROCm 7.2, or a GPU without NPM)."""
    from hyperloom.common.gpu_power_settings import (
        GpuPowerSettingsError,
        declared_node_problems,
        read_node_power_settings,
    )

    try:
        node = read_node_power_settings()
    except GpuPowerSettingsError as exc:
        if any(declared.node_row().values()):
            return None, f"the declared node power management settings cannot be checked: {exc}"
        return None, ""
    problems = declared_node_problems(node, declared)
    if problems:
        return node, "the node is not at the declared power management settings (NPM is set on the BMC): " + "; ".join(
            problems
        )
    return node, ""


def resolve_gpu_power_settings(
    declared_settings: DeclaredPowerSettings,
    *,
    nodes: int,
    read: Any = None,
    just_applied: bool = False,
) -> tuple[dict[str, Any], str]:
    """Read the cards' and the node's power settings and check the declared ones; ``(record, error)``.

    ``record`` is what the session stores and the platform fingerprint shows: the declared values, what each card
    reported and, where amd-smi reports it, the node's NPM state and limit. ``error`` is non-empty when a declared value
    does not hold, or cannot be checked, and the launch must stop. Nothing is set here: the operator sets the cap, perf
    level and sclk limit with ``amd-smi set`` before launch, or ``--apply-gpu-power-settings`` set the cap and level
    just before this check; NPM is set on the BMC.
    """
    from hyperloom.common.gpu_power_settings import (
        GpuPowerSettingsError,
        read_gpu_power_settings,
        visible_gpu_indices,
    )

    declared = declared_settings.as_record()
    if nodes >= 2:
        if declared:
            return {}, (
                "--gpu-power-cap-w / --gpu-perf-level / --gpu-sclk-max-mhz / --node-power-management / "
                "--node-power-limit-w cannot be checked on a multi-node session: they describe the benchmark nodes' "
                "cards, which this process cannot read, and an unverifiable assertion is not a satisfied one"
            )
        return {}, ""
    # The node's settings need neither the cards' reading nor the mask, so only a card declaration refuses without them.
    cards_declared = any(declared_settings.card_row().values())
    try:
        observed = (read or read_gpu_power_settings)()
    except GpuPowerSettingsError as exc:
        if cards_declared:
            return {_DECLARED: declared}, f"the declared GPU power settings cannot be checked: {exc}"
        if not declared:
            node, _ = _node_power_settings(declared_settings)
            return ({"node": node} if node is not None else {}), ""
        observed = {}
    try:
        gpus = visible_gpu_indices()
    except GpuPowerSettingsError as exc:
        if cards_declared:
            return {_DECLARED: declared}, f"the declared GPU power settings cannot be checked: {exc}"
        gpus = None
    record = {
        _DECLARED: declared,
        "observed": {str(gpu): row for gpu, row in sorted(observed.items()) if gpus is None or gpu in gpus},
    }
    node, node_error = _node_power_settings(declared_settings)
    if node is not None:
        record["node"] = node
    errors = _card_errors(observed, declared_settings, gpus=gpus, just_applied=just_applied)
    return record, "; ".join(errors + [node_error] if node_error else errors)


def _card_errors(
    observed: dict[int, dict[str, Any]], declared: DeclaredPowerSettings, *, gpus: set[int] | None, just_applied: bool
) -> list[str]:
    """Why the cards are not at the declared settings, worded for who sets each: the clock limit is never applied."""
    from hyperloom.common.gpu_power_settings import declared_setting_problems

    unset = "the GPUs are not at the declared power settings (set them with amd-smi before launch)"
    if not just_applied:
        problems = declared_setting_problems(observed, declared, gpus=gpus)
        return [f"{unset}: " + "; ".join(problems)] if problems else []
    not_taken = declared_setting_problems(observed, replace(declared, sclk_max_mhz=None), gpus=gpus)
    asserted = declared_setting_problems(observed, DeclaredPowerSettings(sclk_max_mhz=declared.sclk_max_mhz), gpus=gpus)
    return [
        f"{why}: " + "; ".join(problems)
        for why, problems in (("the GPUs did not take the power settings just applied", not_taken), (unset, asserted))
        if problems
    ]


def _build_agentx_corpus_shape_seed() -> dict[str, Any]:
    """Return the canonical corpus shape."""
    from hyperloom.common.agentx_workload import MLPERF_CORPUS, is_mlperf_backend, mlperf_trajectories

    if is_mlperf_backend():
        # The MLPerf corpus has no published length distribution, only its trajectory count.
        return {
            "corpus_loader": MLPERF_CORPUS,
            "corpus_entries": mlperf_trajectories(),
            "source": "canonical_mlperf",
        }

    return {}


def _custom_specialist_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The operator-defined specialist's SharedState fields from the parsed ``--custom-specialist-*`` flags."""
    return {
        "custom_specialist_prompt": str(getattr(args, "custom_specialist_prompt", "") or ""),
        "custom_specialist_description": str(getattr(args, "custom_specialist_description", "") or ""),
        "custom_specialist_only": bool(getattr(args, "custom_specialist_only", False)),
    }


def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
    compute_partition: dict[str, Any] | None = None,
    gpu_power_settings: dict[str, Any] | None = None,
) -> SharedState:
    """Construct and persist the initial :class:`SharedState` for a run."""
    # research_lane capacity is locked for the session; clamp to [0, ceiling].
    from hyperloom.common.visible_devices import detect_gpu_count
    from hyperloom.orchestrator.policy.gate import research_lane_ceiling

    research_lane_capacity = int(getattr(args, "research_lane_capacity", 1) or 1)
    research_lane_capacity = max(
        0,
        min(research_lane_ceiling(), research_lane_capacity),
    )
    # An unset (None) or unparsable capacity falls back to the visible GPU count.
    try:
        gpu_specialist_capacity = max(0, int(getattr(args, "gpu_specialist_capacity", None)))
    except (TypeError, ValueError):
        gpu_specialist_capacity = detect_gpu_count()
    # Collect plateau threshold overrides; absent keys use defaults at compute time.
    plateau_overrides: dict[str, Any] = {}
    if getattr(args, "plateau_explore_keep_gain", None) is not None:
        plateau_overrides["explore_keep_gain_pct"] = float(args.plateau_explore_keep_gain)
    if getattr(args, "plateau_explore_empty_streak", None) is not None:
        plateau_overrides["explore_empty_streak"] = int(args.plateau_explore_empty_streak)
    if getattr(args, "plateau_explore_lookback", None) is not None:
        plateau_overrides["explore_lookback"] = int(args.plateau_explore_lookback)

    # Resolve int workload knobs from the CLI arg, applying the shared fallback default when unset.
    def _int_arg(arg_name: str, default: int) -> int:
        """Resolve an int workload knob from ``args``, else the fallback default."""
        val = getattr(args, arg_name, None)
        if val is None:
            return int(default)
        try:
            resolved = int(val)
        except (TypeError, ValueError):
            return int(default)
        return resolved if resolved > 0 else int(default)

    # KB architecture tags from config.json; fresh-launch only.
    _cfg_tags = _load_model_config_tags(str(args.model))

    # The only read of the backend choice; every later decision reads the persisted record.
    _kernel_optimizer_record = "forge" if forge_explicitly_enabled() else "geak"

    # Reference launch recipe (fresh-launch only, fail-soft): lowest-priority base for the baseline server args.
    _ref_args, _ref_envs, _ref_model, _ref_controls = _resolve_reference_recipe(args)

    # Canonical model identity (prefers the quantize prelude's pinned source name).
    _model_identity = resolve_model_display_name(args)
    benchmark_mode = "synthetic"
    agentx_backend = ""
    agentx_epoch = 0
    if _agentx_enabled():
        from hyperloom.common.agentx_workload import agentic_backend

        benchmark_mode = "agentx"
        agentx_backend = agentic_backend()
        agentx_epoch = AGENTX_MEASUREMENT_EPOCH
    state = SharedState(
        session_id=session_id,
        claw_session_id=os.environ.get("CLAW_SESSION_ID", "").strip(),
        sandbox_user_id=os.environ.get("SANDBOX_USER_ID", "").strip(),
        model_name=_model_identity,
        model_path=str(args.model),
        model_class=args.model_class or "",
        # Advisory architecture profile; fresh-launch only. Soft-degrade to {}.
        model_arch=_load_model_arch(
            session_dir,
            _model_identity,
            str(args.model),
        ),
        # Architecture-identity tags from config.json.
        model_architectures=_cfg_tags.get("architectures", []),
        model_type=_cfg_tags.get("model_type", ""),
        # config.json structural summary, persisted for downstream collectors.
        model_info=summarize_model_config(str(args.model)),
        framework=os.environ.get("FRAMEWORK", "sglang"),
        # The only copy of the budgets. Validated at the CLI, so anything that reaches here is usable, and archived
        # with the session so a resume restores them without a second source to reconcile.
        **constraint_budgets_from_args(args),
        gpu_power_settings=dict(gpu_power_settings or {}),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI/env.
        tp=_int_arg("tp", DEFAULT_TP),
        ep=_int_arg("ep", DEFAULT_EP),
        precision=(str(getattr(args, "precision", None) or DEFAULT_PRECISION).strip()),
        framework_version=resolve_framework_version(args),
        conc=_int_arg("conc", DEFAULT_CONC),
        isl=_int_arg("isl", DEFAULT_ISL),
        osl=_int_arg("osl", DEFAULT_OSL),
        profile_osl=_int_arg("profile_osl", 0),
        max_model_len=_int_arg("max_model_len", 0),
        kernel_enabled=not getattr(args, "no_kernel", False),
        kernel_optimizer=_kernel_optimizer_record,
        # AgentX corpus shape: seeded from canonical constants if AgentX is on.
        agentx_corpus_shape=_build_agentx_corpus_shape_seed() if benchmark_mode == "agentx" else {},
        baseline_tput=0.0,
        cumulative_gain_validated=0.0,
        reference_server_args=_ref_args,
        reference_envs=_ref_envs,
        reference_launch_controls=_ref_controls,
        reference_model=_ref_model,
        # Operator launch shape; the process env carries it for one process only, so a resume re-exports it from here
        # rather than from argv.
        operator_server_args=str(getattr(args, "server_args", "") or "").strip(),
        operator_extra_env=parse_operator_extra_env(args),
        **_custom_specialist_fields(args),
        bypass_scripts_dir=os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip(),
        framework_repo_path=os.environ.get("FRAMEWORK_REPO_PATH", "").strip(),
        benchmark_backend=os.environ.get("HYPERLOOM_BENCHMARK_BACKEND", "").strip().lower(),
        xdit_model_name=os.environ.get("XDIT_MODEL_NAME", "").strip(),
        compute_partition=dict(compute_partition if compute_partition is not None else (published_shape() or {})),
        nodes=max(1, int(getattr(args, _NODES, 1))),
        warm_replay_enabled=not bool(getattr(args, "no_warm_replay", False)),
        **(
            {}
            if getattr(args, "warm_replay_min_confidence", None) is None
            else {"warm_replay_min_confidence": float(args.warm_replay_min_confidence)}
        ),
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        gpu_specialist_capacity=gpu_specialist_capacity,
        plateau_overrides=plateau_overrides,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_AGENT phase; --no-framework-agent skips it.
        framework_agent_phase_enabled=not bool(getattr(args, "no_framework_agent", False)),
        # FRAMEWORK local-exploration arm; --no-framework-local-explore opts out.
        framework_local_explore_enabled=not bool(getattr(args, "no_framework_local_explore", False)),
        # Enablement self-heal lanes; --enablement off opts out.
        enablement_mode=str(getattr(args, "enablement", "all")),
        research_scout_enabled=bool(getattr(args, "research_scout", True)),
        research_scout_interval=max(1, int(getattr(args, "research_scout_interval", 3) or 3)),
        static_recon_enabled=bool(getattr(args, "static_recon", True)),
        target_advisory_enabled=bool(getattr(args, "target_advisory", True)),
        recipe_sediment_enabled=bool(getattr(args, "recipe_sediment", True)),
        # SWEEP-phase concurrency sweep: defaults OFF under AgentX because each
        # rung is a 3600s window and the session grades at a fixed CONC.
        # Pass --enable-conc-sweep explicitly to override.
        conc_sweep_enabled=(
            not is_agentx_mode(benchmark_mode) if args.enable_conc_sweep is None else args.enable_conc_sweep
        ),
        benchmark_mode=benchmark_mode,
        agentx_epoch=agentx_epoch,
        agentx_backend=agentx_backend,
        grading=seed_grading(os.environ.get("FRAMEWORK", "sglang"), benchmark_mode),
        conc_sweep_concs=_parse_conc_sweep_concs(args, benchmark_mode),
        conc_sweep_total_budget_sec=int(getattr(args, "conc_sweep_total_budget_sec", 9000)),
    )
    state.save(session_dir)
    return state


def _snapshot_system_prompts(
    session_dir: Path,
    *,
    prompts: dict[str, str],
    macro_cycle: int,
    orchestration_phase: str = "",
) -> None:
    """Persist each agent's effective system prompt via the shared snapshot writer.

    The orchestration role additionally writes a phase-scoped copy when ``orchestration_phase`` is set.
    """
    from hyperloom.orchestrator.prompts import write_prompt_snapshot

    for role, body in prompts.items():
        write_prompt_snapshot(session_dir, role, body, macro_cycle=macro_cycle)
    boot_phase = orchestration_phase.strip()
    if boot_phase and "orchestration" in prompts:
        write_prompt_snapshot(
            session_dir, "orchestration", prompts["orchestration"], phase=boot_phase, macro_cycle=macro_cycle
        )


def _print_session_skeleton(session_dir: Path) -> None:
    """Echo the freshly-created skeleton so launchers see the exact layout."""
    print(f"Session layout under {session_dir}:")
    for sub in _SESSION_SKELETON:
        marker = "ok" if (session_dir / sub).is_dir() else "MISSING"
        print(f"  [{marker}] {sub}/")
    print("  [ok] manifest.json (written first)")


@dataclass(frozen=True)
class LaunchInfo:
    """The optimizer process a launcher started and the session it serves, as harvested from HYPERLOOM_LAUNCH."""

    pid: int
    session_dir: Path
    session_id: str
    run_log: str
    gpu_type: str
    framework: str
    model: str


def _emit_launch_info(info: LaunchInfo, *, launch_info_file: str | None) -> dict[str, Any]:
    """Print the machine-readable HYPERLOOM_LAUNCH stdout line; optionally JSON-dump to ``launch_info_file``."""
    launch_info: dict[str, Any] = {
        "event": "launch",
        "pid": info.pid,
        "session_dir": str(info.session_dir),
        "session_id": info.session_id,
        "run_log": info.run_log,
        "manifest": str(info.session_dir / "manifest.json"),
        "gpu_type": info.gpu_type,
        "framework": info.framework,
        "model": info.model,
    }
    kv_body = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in launch_info.items())
    print(f"HYPERLOOM_LAUNCH {kv_body}")
    if launch_info_file:
        path = Path(launch_info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(launch_info, indent=2))
        print(f"Launch info file: {path}")
    return launch_info


def _print_final_summary(
    state: SharedState,
    stop_reason: str,
    session_dir: Path,
) -> None:
    """Print the end-of-run summary block to stdout."""
    print()
    print("================ Final summary ================")
    print(f"  stop_reason          : {stop_reason}")
    print(f"  session_id           : {state.session_id}")
    print(f"  model                : {state.model_name}")
    from .. import framework_registry

    print(
        f"  baseline             : {framework_registry.format_primary_metric(getattr(state, 'framework', ''), state.baseline_tput)}"
    )
    if stop_reason == "baseline_failed":
        failure_summary = _read_failure_summary(session_dir)
        if failure_summary and failure_summary.get("root_cause"):
            print(
                f"  root_cause           : "
                f"[{failure_summary.get('root_cause_type', 'unknown')}] "
                f"{failure_summary.get('root_cause')}"
            )
            if failure_summary.get("server_log"):
                print(f"  server_log           : {failure_summary.get('server_log')}")
    if state.cumulative_gain_validated_ts:
        stale = " ⚠ stack changed since validation" if state.optimization_stack_has_unvalidated_keeps() else ""
        print(
            f"  cumulative_gain_val  : {state.cumulative_gain_validated:.2f}% "
            f"(validated_at_stack_len={state.cumulative_gain_validated_stack_len}, "
            f"ts={state.cumulative_gain_validated_ts}){stale}"
        )
    else:
        print("  cumulative_gain_val  : 0.00% ⚠ never validated — no `explore` KEEP has landed yet")
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    print("===============================================")


def _bank_previous_leg_phase_segment(state: SharedState) -> None:
    """Bank the phase time the stopped leg spent but never recorded."""
    boundary = state.leg_ended_ts or state.stop_ts
    stop_unix = min(to_unix(boundary, 0.0) or 0.0, time.time())
    if stop_unix <= 0.0:
        return
    bank_phase_segment(state, until_unix=stop_unix)


def _begin_resume_leg(state: SharedState) -> str:
    """Mark the start of a resumed run leg on ``state`` (caller persists).

    Stamps :attr:`SharedState.resumed_ts`, which dates the previous leg's CLOSE
    transition in ``phase_history`` as a previous leg's and stops the phase
    clock charging the gap between the two legs to the phase it stopped in.
    Clears the previous leg's terminal bookkeeping: a stale ``stop_reason``
    makes Orchestration heartbeats think the work is done, a stale closing
    flag resumes straight into a wind-down already finished, and a stale
    ``target_reached_at`` sends the leg to SWEEP against an objective the
    resume may have raised. The Coordinator stamps it again on the first tick
    the restored objective holds.

    The wall-clock budget is untouched. Elapsed time is summed forward across
    legs, so this leg starts from whatever the session has already spent;
    :meth:`SharedState.extend_budget_minutes` is the only way to lengthen it.

    Args:
        state (SharedState): The loaded session state, mutated in place.

    Returns:
        str: The timestamp stamped as this leg's boundary.
    """
    _bank_previous_leg_phase_segment(state)
    state.resumed_ts = now_iso()
    state.stop_reason = ""
    state.stop_ts = ""
    state.leg_ended_ts = ""
    state.closing_phase = False
    state.closing_started_unix = 0.0
    state.closing_report_task_id = ""
    state.close_sequence_done = False
    state.target_reached_at = ""
    state.crash_count = 0
    state.begin_leg()
    return state.resumed_ts


def _resume_budget_lines(state: SharedState, *, extend_hours: float) -> list[str]:
    """Operator-facing notes about what budget the resumed leg actually has.

    Args:
        state: Loaded session state after :func:`_begin_resume_leg`.
        extend_hours: Hours this invocation granted via ``--extend-hours``.

    Returns:
        Lines to print, each already prefixed with ``  → ``.
    """
    elapsed_h = state.elapsed_minutes() / 60.0
    remaining_min = state.remaining_minutes()
    lines = [f"  → {elapsed_h:.2f}h charged to this session across every leg so far"]
    if remaining_min is None:
        lines.append("  → budget: unbounded")
        return lines
    lines.append(f"  → budget: {float(state.max_minutes) / 60.0:.2f}h total, {remaining_min / 60.0:.2f}h left")
    if extend_hours > 0.0:
        lines.append(f"  → --extend-hours added {extend_hours:.2f}h to the session budget")
    if remaining_min <= 0.0:
        lines.append(
            "  → WARNING: the budget is spent; this leg will close almost "
            "immediately. Pass --extend-hours to grant more, or start a fresh session"
        )
    return lines


def _reconcile_crash_count(state: SharedState, session_dir: Path) -> None:
    """Reconcile persisted ``crash_count`` (state.json + final.json) up to the live in-memory value."""
    live = int(getattr(state, "crash_count", 0) or 0)

    # state.json: reload, bump if stale, atomic re-save.
    try:
        disk_state = SharedState.load_or_init(session_dir)
        if int(disk_state.crash_count or 0) < live:
            disk_state.crash_count = live
            disk_state.save(session_dir)
    except (OSError, ValueError, TypeError):
        log.exception("crash_count reconcile (state.json) failed (non-fatal)")

    try:
        from hyperloom.orchestrator.actions.executors.report import reconcile_final_crash_count

        reconcile_final_crash_count(session_dir, live)
    except (OSError, ValueError, TypeError, AttributeError):
        log.exception("crash_count reconcile (final.json) failed (non-fatal)")


def _parse_conc_sweep_concs(args: argparse.Namespace, benchmark_mode: str) -> list[int]:
    """Parse ``--conc-sweep-concs`` into a list[int]; non-integers warned+dropped."""
    from hyperloom.orchestrator.kernel.conc_sweep import default_concs_for_mode

    fallback = default_concs_for_mode(benchmark_mode)
    raw = str(getattr(args, "conc_sweep_concs", "") or "").strip()
    if not raw:
        return fallback
    out: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            log.warning("conc_sweep: ignoring non-integer CONC token %r", t)
    return out or fallback


def _read_failure_summary(session_dir: Path) -> dict | None:
    """Read ``reports/final.json``'s ``failure_summary`` block, if present."""
    try:
        from ..session.session_paths import reports_dir

        final_json = reports_dir(session_dir) / "final.json"
        data = json.loads(final_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    fs = data.get("failure_summary") if isinstance(data, dict) else None
    return fs if isinstance(fs, dict) else None


def _resolve_reference_recipe(
    args: argparse.Namespace,
) -> tuple[str, dict[str, str], str, dict[str, Any]]:
    """Resolve the reference launch recipe for a fresh launch."""
    source = (getattr(args, "reference_script", None) or "").strip()
    if not source:
        return ("", {}, "", {})

    framework = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    from ..reference_script import parse_reference_script

    try:
        recipe = parse_reference_script(source, framework=framework)
    except Exception as exc:
        print(f"ERROR: --reference-script {source!r} could not be parsed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    controls = getattr(recipe, "launch_controls", {})
    if not recipe.server_args and not recipe.envs and not controls:
        print(
            f"ERROR: --reference-script {source!r} lifted no server flags and no env exports",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"Reference script: {source} ({len(recipe.server_args.split())} arg tokens, {len(recipe.envs)} env(s))")
    return (recipe.server_args, dict(recipe.envs), recipe.model or "", dict(controls))
