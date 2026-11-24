# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session setup shared by fresh and resumed launches: card, framework, operator paths, partition shape, lock, budget."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom_common import gpu_card
from hyperloom_common.amd_smi import AmdSmiError
from hyperloom_common.gpu_card import GpuCard
from hyperloom.common.gpu_partition import PartitionModel
from hyperloom.common.perf_metric import agentx_enabled as _agentx_enabled
from hyperloom.common.visible_devices import HIP_LEVEL_VARS, ROCR_LEVEL_VARS, effective_mask_tokens

from .. import framework_registry
from ..session.lock import SessionAlreadyRunning, SessionLock
from .launch_shape import _refuse_multinode_partition_shape
from .parser import DEFAULT_MAX_HOURS

if TYPE_CHECKING:
    from hyperloom.orchestrator.state.shared_state import SharedState


def _normalise_framework_name(value: str | None) -> str:
    """Normalize a framework string for equality checks."""
    return str(value or "").strip().lower().replace("_", "-")


_XDIT_MODEL_NAME_ATTR = "xdit_model_name"
_XDIT_MODEL_NAME_ENV = "XDIT_MODEL_NAME"


def _apply_operator_supplied_paths(args: Any, framework: str) -> None:
    """Publish ``--framework-path`` / ``--benchmark-scripts-dir`` as env."""
    fatal: list[str] = []
    for flag, value, env_name in (
        ("--framework-path", getattr(args, "framework_path", None), "FRAMEWORK_REPO_PATH"),
        (
            "--benchmark-scripts-dir",
            getattr(args, "benchmark_scripts_dir", None),
            "HYPERLOOM_BYPASS_SCRIPTS_DIR",
        ),
    ):
        raw = str(value or "").strip()
        # An exported-but-empty var is not a choice, it is an unset one; treating it as set would let a stray `export
        # FRAMEWORK_REPO_PATH=` silently swallow the flag.
        already = os.environ.get(env_name, "").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_dir():
                fatal.append(f"{flag}={raw!r} is not a directory")
                continue
            if not already:
                os.environ[env_name] = str(path.resolve())
        elif framework == "custom" and not already:
            fatal.append(f"{flag} is required for --framework custom (or export {env_name})")

    if framework == "xdit":
        xdit_name = str(getattr(args, _XDIT_MODEL_NAME_ATTR, None) or "").strip()
        already_xdit = os.environ.get(_XDIT_MODEL_NAME_ENV, "").strip()
        if xdit_name and not already_xdit:
            os.environ[_XDIT_MODEL_NAME_ENV] = xdit_name
        elif not xdit_name and not already_xdit:
            fatal.append(
                "--xdit-model-name is required for xDiT runs. "
                "Pass the HuggingFace repo ID, e.g. 'black-forest-labs/FLUX.2-dev'."
            )

    if fatal:
        for line in fatal:
            print(f"ERROR: {line}", file=sys.stderr)
        sys.exit(2)


def _restore_operator_supplied_paths_from_state(args: Any, state: SharedState) -> None:
    """Re-export the session paths persisted on ``state`` when this resume omitted them."""
    from hyperloom.inference_optimizer.framework_paths import point_session_framework_root

    cli_repo = str(getattr(args, "framework_path", None) or "").strip()
    cli_scripts = str(getattr(args, "benchmark_scripts_dir", None) or "").strip()
    if not cli_repo and not os.environ.get("FRAMEWORK_REPO_PATH", "").strip():
        archived = str(getattr(state, "framework_repo_path", "") or "").strip()
        if archived:
            point_session_framework_root(archived, state.framework)
    if not cli_scripts and not os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip():
        archived = str(getattr(state, "bypass_scripts_dir", "") or "").strip()
        if archived:
            os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] = archived
    if (
        not str(getattr(args, _XDIT_MODEL_NAME_ATTR, None) or "").strip()
        and not os.environ.get(_XDIT_MODEL_NAME_ENV, "").strip()
    ):
        archived = str(getattr(state, _XDIT_MODEL_NAME_ATTR, "") or "").strip()
        if archived:
            os.environ[_XDIT_MODEL_NAME_ENV] = archived


def _read_session_card(nodes: int) -> GpuCard | None:
    """The card this session runs on, a remote node's for a multi-node run: the first one, with a WARN when its cards
    are not one kind."""
    try:
        if nodes >= 2:
            from ..multi_node._internal.gpu_probe import remote_cards

            cards = remote_cards()
        else:
            cards = _visible_cards(gpu_card.read_cards())
    except AmdSmiError as exc:
        print(f"GPU card        : none read ({exc})")
        return None
    try:
        card = gpu_card.one_kind(cards)
    except ValueError as exc:
        card = cards[0]
        print(f"WARN: {exc}; the session takes the first", file=sys.stderr)
    print(f"GPU card        : {card.describe()}")
    return card


def _visible_cards(cards: list[GpuCard]) -> list[GpuCard]:
    """The cards the device masks leave this process: a ROCr-level mask picks among the machine's, by ROCm number or
    UUID, and a HIP-level mask among those. Exits 2 on a mask naming a card the machine does not have."""
    seen = list(range(len(cards)))
    for level in (ROCR_LEVEL_VARS, HIP_LEVEL_VARS):
        tokens = next((found for var in level if (found := effective_mask_tokens(os.environ.get(var)))), [])
        uuids = gpu_card.rocr_uuids() if any(not token.isdigit() for token in tokens) else []
        try:
            seen = [seen[int(t)] if t.isdigit() else uuids.index(t.lower()) for t in tokens] or seen
        except (IndexError, ValueError):
            mask = ",".join(tokens)
            print(f"ERROR: the visible-device mask {mask!r} names a GPU this machine does not have", file=sys.stderr)
            sys.exit(2)
    return [cards[index] for index in seen]


def _session_label(given: str, card: GpuCard | None) -> str:
    """The name the session files its results under: the one given, else the card's model designation, else its arch."""
    named = gpu_card.board_label(card.name) if card else ""
    if given and named and given != named:
        print(
            f"WARN: the GPU type is given as {given!r}, but the card calls itself {named!r} ({card.name}); "
            f"this session's results are filed under {given!r}",
            file=sys.stderr,
        )
    return given or named or (card.gfx if card else "")


def _require_custom_entrypoint(framework: str, gpu_type: str | None = None) -> None:
    """Fail at launch when a custom framework has no entrypoint."""
    if str(framework or "").strip().lower() != "custom":
        return
    from hyperloom.orchestrator.actions.executors.workload._framework_runtime import (
        apply_scriptable_runtime_defaults,
    )
    from hyperloom.orchestrator.actions.executors.explore_prefilter.declared_levers import (
        DeclaredLeversError,
        load_declared_levers,
    )

    try:  # a malformed declaration must fail here, not surface later as empty_grid
        load_declared_levers()
    except DeclaredLeversError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    bench: dict[str, Any] = {"framework": "custom"}
    apply_scriptable_runtime_defaults(bench, {}, gpu_type=gpu_type, explicit_benchmark_script=False)
    if bench.get("benchmark_script"):
        return
    scripts_dir = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    expected = Path(scripts_dir) / f"custom_{bench['runner_type']}.sh"
    print(
        f"ERROR: --framework custom could not resolve a benchmark entrypoint: there is no {expected}, "
        f"and {scripts_dir or 'the scripts dir'} does not hold exactly one .sh script.",
        file=sys.stderr,
    )
    sys.exit(2)


def _persist_operator_supplied_paths(state: SharedState) -> None:
    """Mirror the live custom-workload env back onto ``state`` for the next resume."""
    state.framework_repo_path = os.environ.get("FRAMEWORK_REPO_PATH", "").strip()
    state.bypass_scripts_dir = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    state.xdit_model_name = os.environ.get(_XDIT_MODEL_NAME_ENV, "").strip()


def _registered_framework_or_exit(requested: str, *, hint: str) -> str:
    """Return ``requested`` normalised, or the default when blank; exit 2 when it names no registered framework."""
    framework = requested.strip().lower() or framework_registry.DEFAULT_FRAMEWORK
    if not framework_registry.is_supported(framework):
        print(
            f"ERROR: framework must be one of {', '.join(framework_registry.names())} (got {framework!r}); {hint}",
            file=sys.stderr,
        )
        sys.exit(2)
    return framework


def _enforce_expected_framework(framework: str) -> None:
    """Fail fast when a launcher-pinned expected framework is violated."""
    actual = _normalise_framework_name(framework)
    wanted = _normalise_framework_name(
        os.environ.get("INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK", "") or os.environ.get("EXPECTED_FRAMEWORK", "")
    )
    if not wanted:
        return
    if wanted != actual:
        print(
            "ERROR: framework mismatch: "
            f"EXPECTED_FRAMEWORK={wanted!r} but resolved framework={actual!r}. "
            "Refusing to launch because this would run a different backend "
            "than the operator requested.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _apply_atom_auto_tighten(args: argparse.Namespace) -> None:
    """Validate atom-specific CLI knobs: the ``--nodes>=2`` fail-fast guard (IR-8)."""
    if int(getattr(args, "nodes", 1) or 1) >= 2:
        print(
            "ERROR: --framework atom does not support multi-node "
            "(--nodes >= 2). atom multi-node TP wiring is deferred; "
            "drop to --nodes 1 or pick --framework sglang/vllm.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        "  framework=atom: no auto-disable applied (framework-agent + "
        "profile / roofline / TraceLens all wired for atom); "
        "--nodes>=2 guard active — see SKILL.md IR-8"
    )


# Exit code for "another optimizer already owns this session".
SESSION_BUSY_EXIT_CODE = 3


def _acquire_session_lock_or_exit(session_dir: Path) -> SessionLock:
    """Take the single-optimizer session lock or exit ``SESSION_BUSY_EXIT_CODE``."""
    lock = SessionLock(session_dir)
    try:
        lock.acquire()
    except SessionAlreadyRunning as exc:
        print(
            f"ERROR: {exc}. Refusing to start a second optimizer on the same "
            f"session (would corrupt shared leases / state.json). If the owner "
            f"is truly dead, wait for the OS to drop the lock or remove "
            f"{lock.path} before retrying.",
            file=sys.stderr,
        )
        sys.exit(SESSION_BUSY_EXIT_CODE)
    return lock


def _apply_default_budget(args: argparse.Namespace) -> None:
    """Give a run that neither ``--max-hours`` nor its archive budgeted the default, warning when AgentX is on."""
    if args.max_hours is not None:
        return
    args.max_hours = DEFAULT_MAX_HOURS
    if _agentx_enabled():
        print(
            f"NOTE: AgentX replay is configured and --max-hours is at its default of {DEFAULT_MAX_HOURS}. "
            "One AgentX round (corpus load + warmup + drain + measurement window) "
            "typically exceeds that on its own; pass an explicit --max-hours sized "
            "to the number of candidates you intend to measure.",
            file=sys.stderr,
        )


def _partition_fanout_supported(framework: str | None) -> tuple[bool, str]:
    """Whether this framework's runner can place work per partition."""
    name = str(framework or "").strip().lower()
    if not name:
        return False, (
            "the framework is not resolved yet, so whether its runner places work "
            "per partition could not be checked here"
        )
    if framework_registry.is_scriptable(name):
        return True, ""
    return False, (
        f"{name!r} runs a server, and its benchmark does not place work per partition, "
        f"so the streams-per-partition setting would be ignored"
    )


@dataclass(frozen=True)
class PartitionRequest:
    """The compute-partition flags an operator passed: an asserted mode and the streams each partition runs."""

    declared_mode: str | None = None
    streams_per_partition: int | None = None


def _export_partition_shape(
    request: PartitionRequest,
    *,
    framework: str | None = None,
    nodes: int = 1,
    model: PartitionModel = PartitionModel(),
) -> dict[str, Any]:
    """Validate the session's compute-partition shape and publish it."""
    declared_mode, streams_per_partition = request.declared_mode, request.streams_per_partition
    from hyperloom.common.gpu_partition import PartitionError, parse_mode
    from hyperloom.orchestrator.actions.executors.gpu._partition_shape import (
        DEFAULT_STREAMS_PER_PARTITION,
        PARTITION_COUNT_ENV,
        PARTITION_CU_ENV,
        PARTITION_MODE_ENV,
        PARTITION_STREAMS_ENV,
        PARTITION_TOTAL_STREAMS_ENV,
        runtime_env,
        session_shape_summary,
        validate_session_shape,
    )

    # Cleared first so a second session in the same shell cannot inherit a shape the operator did not ask for this
    # time.
    for key in (
        PARTITION_MODE_ENV,
        PARTITION_COUNT_ENV,
        PARTITION_CU_ENV,
        PARTITION_STREAMS_ENV,
        PARTITION_TOTAL_STREAMS_ENV,
    ):
        os.environ.pop(key, None)

    try:
        mode = parse_mode(declared_mode)
    except PartitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    streams_named = streams_per_partition is not None
    # Tested against None rather than falsiness: `0 or DEFAULT` is DEFAULT, which would quietly honour an invalid
    # request as the default instead of refusing it, and leave the guard below unreachable for the one value most
    # likely to be passed by mistake.
    streams = DEFAULT_STREAMS_PER_PARTITION if streams_per_partition is None else int(streams_per_partition)
    if streams < 1:
        print(
            f"ERROR: --streams-per-partition must be >= 1, got {streams_per_partition}",
            file=sys.stderr,
        )
        sys.exit(2)

    if nodes >= 2:
        _refuse_multinode_partition_shape(nodes, mode)
        return {}

    fanout, fanout_detail = _partition_fanout_supported(framework)
    if (mode or streams_named) and not fanout:
        print(f"WARN: {fanout_detail}.", file=sys.stderr)

    verdict = validate_session_shape(
        declared_mode=mode,
        streams=streams,
        model=model,
        # The footprint refusal is arithmetic about streams sharing a partition.
        fanout_expected=fanout or bool(mode) or streams_named,
    )
    for warning in verdict.warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    if not verdict.ok:
        print(f"ERROR: {verdict.refusal}", file=sys.stderr)
        sys.exit(2)
    for note in verdict.notes:
        print(note)

    if verdict.layout is None:
        return {}
    # The observed shape is published either way -- the platform fingerprint reads it back from here, and provenance
    # is the point.
    os.environ.update(runtime_env(verdict.layout, streams, fanout=fanout))
    return session_shape_summary(verdict.layout, streams, fanout_expected=fanout)


def _publish_gpu_power_settings(record: Mapping[str, Any]) -> None:
    """Expose the recorded power settings to the platform fingerprint, the way the partition shape is."""
    from hyperloom.common.platform_probe import GPU_POWER_SETTINGS_ENV

    if record:
        os.environ[GPU_POWER_SETTINGS_ENV] = json.dumps(record, sort_keys=True)
    else:
        os.environ.pop(GPU_POWER_SETTINGS_ENV, None)


def _restore_partition_shape_from_state(args: Any, state: SharedState) -> None:
    """Fill the partition flags from the archive when this resume omitted them."""
    archived = dict(getattr(state, "compute_partition", None) or {})
    if not str(getattr(args, "compute_partition_mode", None) or "").strip():
        args.compute_partition_mode = str(archived.get("mode") or "")
    # `is None` rather than falsiness, so an explicit `--streams-per-partition 0` still reaches the guard that refuses
    # it instead of being read as "omitted" and quietly replaced by the archived value.
    if getattr(args, "streams_per_partition", None) is None:
        streams = archived.get("streams_per_partition")
        args.streams_per_partition = int(streams) if streams else None


@dataclass(frozen=True)
class _SessionStart:
    """The session this run operates on, however it was opened."""

    session_dir: Path
    session_lock: SessionLock
    manifest: dict[str, Any]
