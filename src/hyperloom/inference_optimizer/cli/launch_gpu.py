# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GPU a launch runs on: its one rocminfo run, the identity and power settings it records, and the checks on it.

The checks are preflight install steps. Preflight passes its own step runner in, so they are recorded with the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.env import forge_explicitly_enabled
from hyperloom.common.provenance import GpuProbe

from .resumed_session import _resumed_session_state

#: Preflight's ``_run_install_step``.
RunStep = Callable[..., Any]

_CHECK = "check"
_NO_KERNEL = "no_kernel"


def publish_gpu_power_settings(record: Mapping[str, Any]) -> None:
    """Expose the recorded power settings to the platform fingerprint, the way the partition shape is."""
    from hyperloom.common.platform_probe import GPU_POWER_SETTINGS_ENV

    if record:
        os.environ[GPU_POWER_SETTINGS_ENV] = json.dumps(record, sort_keys=True)
    else:
        os.environ.pop(GPU_POWER_SETTINGS_ENV, None)


def publish_gpu_identity(record: Any) -> None:
    """Expose the GPU's identity to the manifest and the platform fingerprint, the way the power settings are.

    ``record`` is :func:`launch_gpu_identity`'s answer, or on a resume whatever the manifest holds.
    """
    from hyperloom.common.provenance import GPU_IDENTITY_ENV

    if isinstance(record, Mapping) and any(value is not None for value in record.values()):
        os.environ[GPU_IDENTITY_ENV] = json.dumps(dict(record), sort_keys=True)
    else:
        os.environ.pop(GPU_IDENTITY_ENV, None)


def _node_count(value: Any) -> int:
    try:
        return int(value or 1)
    except (TypeError, ValueError):
        return 1


def gpus_are_remote(args: argparse.Namespace) -> bool:
    """Whether the GPUs are on other nodes, where a local rocminfo sees none of them.

    True for ``--nodes`` >= 2, or a resumed session created with it: the resume block restores the node count only
    after preflight.
    """
    recorded = (_resumed_session_state(args) or {}).get("nodes")
    return max(_node_count(getattr(args, "nodes", 1)), _node_count(recorded)) >= 2


#: The device masks a launch probe answers for; when one changes, the probe runs again.
_GPU_MASK_ENVS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def launch_gpu_probe(args: argparse.Namespace) -> GpuProbe | None:
    """This launch's one rocminfo run, which its GPU checks and the identity it records share.

    ``None`` when rocminfo cannot run or the GPUs are on remote nodes.
    """
    from hyperloom.common.provenance import probe_gpus

    if gpus_are_remote(args):
        return None
    masks = tuple(os.environ.get(name) for name in _GPU_MASK_ENVS)
    cached = getattr(args, "_launch_gpu_probe", None)
    if cached is None or cached[0] != masks:
        cached = (masks, probe_gpus())
        args._launch_gpu_probe = cached
    return cached[1]


def launch_gpu_identity(args: argparse.Namespace) -> dict[str, Any]:
    """The identity of the GPU this launch runs on; only the product when the GPUs are on remote nodes."""
    from hyperloom.common.provenance import detect_gpu_identity

    probe = launch_gpu_probe(args)
    return detect_gpu_identity(getattr(args, "gpu_type", None), probe.agents if probe else None)


def _skipped(reason: str, **fields: Any) -> dict[str, Any]:
    """The outcome of a check that did not run, and why."""
    return {"status": "skipped", "skip_reason": reason, **fields}


def check_gpu_launch(args: argparse.Namespace, install_event: Any, *, run_step: RunStep) -> None:
    """Run the ``check_gpu_product`` step and, for a board whose GEAK checks the card, ``check_geak_target``."""
    from ..gpu_types import geak_checks_the_card

    gpu_product = run_step(
        install_event,
        step_id="check_gpu_product",
        category=_CHECK,
        action=lambda: _check_gpu_product(args),
    )
    gpu_type = (gpu_product or {}).get("detail", {}).get("gpu_type")
    if geak_checks_the_card(gpu_type):
        run_step(
            install_event,
            step_id="check_geak_target",
            category=_CHECK,
            action=lambda: _check_geak_target(args, gpu_type),
        )


def _check_gpu_product(args: argparse.Namespace) -> dict[str, Any]:
    """Refuse a launch the visible board cannot serve, before any install, LLM call or session dir (exit 2).

    The ISA, nameplate and PCI ID come from rocminfo alone, never ``HYPERLOOM_GFX_ARCH`` or ``GPU_TYPE``; a resume
    checks the session's own GPU type and framework, which ``_pin_resumed_session_args`` already applied.
    """
    from hyperloom.inference_optimizer import framework_registry

    from ..gpu_types import _resolve_amd_gpu_type, probed_gpu_type, product_launch_error

    state = _resumed_session_state(args) or {}
    resumed = bool(str(state.get("gpu_type") or "").strip())
    requested = str(state.get("gpu_type") or "").strip().lower() or (
        _resolve_amd_gpu_type(getattr(args, "gpu_type", None), probe=False) or ""
    )
    framework = (
        str(getattr(args, "framework", None) or os.environ.get("FRAMEWORK", "")).strip().lower()
        or framework_registry.DEFAULT_FRAMEWORK
    )
    remote = gpus_are_remote(args)
    probe = launch_gpu_probe(args)
    agents = probe.agents if probe else None
    agent = agents[0] if agents else None
    detail = {
        "gpu_type": requested or probed_gpu_type(agents) or None,
        "marketing_name": (agent.marketing_name or None) if agent else None,
        "pci_device_id": (agent.pci_device_id or None) if agent else None,
        "observed_gfx_arch": agent.gfx if agent else None,
        "kernel_enabled": not getattr(args, _NO_KERNEL, False),
    }
    error = product_launch_error(requested, agents, framework=framework, resumed=resumed, remote=remote)
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    if agents is None:
        return _skipped("remote_gpu_probe" if remote else "rocminfo_unavailable", detail=detail)
    return {"detail": detail}


_GEAK_IDENTITY_TIMEOUT_SEC = 30


def _geak_identity(root: Path | None, rocminfo: str) -> tuple[dict[str, Any] | None, str]:
    """What GEAK's identity helper at ``root`` reads from ``rocminfo``, or ``None`` and why it could not."""
    helper = root / "scripts" / "gpu_identity.py" if root else None
    if helper is None:
        return None, "no GEAK checkout was found (GEAK_E2E_RUNNER, GEAK_ROOT, HYPERLOOM_CACHE_DIR)"
    if not helper.is_file():
        return None, f"the GEAK checkout at {root} has no scripts/gpu_identity.py"
    try:
        out = subprocess.run(  # nosec B603
            [sys.executable, str(helper), "--stdin"],
            input=rocminfo,
            capture_output=True,
            text=True,
            timeout=_GEAK_IDENTITY_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"GEAK's {helper} did not finish ({exc})"
    if out.returncode != 0:
        detail = (out.stderr or out.stdout).strip().splitlines()
        return None, f"GEAK's {helper} exited {out.returncode}" + (f": {detail[-1]}" if detail else "")
    try:
        identity = json.loads(out.stdout)
    except ValueError:
        identity = None
    if not isinstance(identity, dict):
        return None, f"GEAK's {helper} printed no identity"
    return identity, ""


def _check_geak_target(args: argparse.Namespace, gpu_type: str) -> dict[str, Any]:
    """Refuse a kernel phase whose GEAK cannot name the board (exit 2), before any install, LLM call or session dir.

    Handed a board's identity pin, GEAK probes the card at every kernel handoff and refuses one its identity helper
    names otherwise, hours into the run. The helper reads the launch's own rocminfo output here, so the handoff's
    verdict comes now.
    """
    from hyperloom.common.geak_checkout import resolve_e2e_runner

    from ..gpu_types import geak_identity_pin

    if getattr(args, _NO_KERNEL, False):
        return _skipped(_NO_KERNEL)
    if forge_explicitly_enabled():
        return _skipped("forge_enabled")
    probe = launch_gpu_probe(args)
    if probe is None:
        return _skipped("remote_gpu_probe")
    try:
        root: Path | None = Path(resolve_e2e_runner()).resolve().parents[1]
    except FileNotFoundError:
        root = None
    identity, reason = _geak_identity(root, probe.rocminfo)
    if identity is not None:
        target, gfx = identity.get("target"), identity.get("gfx")
        pin = geak_identity_pin(gpu_type)
        if (target, gfx) == (pin["expected_target"], pin["expected_gfx"]):
            return {"target": str(root), "detail": {"geak_target": target, "geak_gfx": gfx}}
        reason = f"GEAK at {root} identifies this GPU as {target} on {gfx}"
    print(
        f"ERROR: the kernel phase hands {gpu_type} to GEAK, which checks the card at every handoff, but {reason}. "
        f"Set GEAK_ROOT to a GEAK checkout that identifies {gpu_type} (its scripts/gpu_identity.py names the card "
        f"{gpu_type}), or pass --no-kernel. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)
