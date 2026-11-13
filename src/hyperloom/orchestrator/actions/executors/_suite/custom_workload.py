# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a custom single-file round adds to a scriptable one.

A custom campaign optimizes one operator-declared module, so its round carries three
things a shipped scriptable round does not: the declared boundary is enforced before any
budget is spent, the export counters are armed and delivered to the benchmark child, and
an AOTI profiling leg asks Inductor for per-launch scopes.

The additions ride as workload params, which the suite hands the script as its
environment beside the names the round sets itself.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

from inference_suite.contracts.config import ScriptableWorkload
from inference_suite.contracts.layout import RunLayout

from hyperloom.common.optimization_scope import aoti_config, optimization_scope
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from hyperloom.orchestrator.framework.optimization_scope import check_patch_scope
from hyperloom.orchestrator.measurement.aoti_evidence import EVIDENCE_DIR_ENV, evidence_dir

from .._framework_rewrite_evidence import probe_asset_dir

log = logging.getLogger(__name__)

CUSTOM_FRAMEWORK = "custom"

#: Where a scriptable script is told to write its torch trace. The suite names the two
#: framework spellings itself; a custom harness reads neither.
PROFILE_TRACE_DIR_ENV = "HYPERLOOM_PROFILE_TRACE_DIR"

_KERNEL_PROFILE_ENV = "TORCHINDUCTOR_CPP_ENABLE_KERNEL_PROFILE"


def is_custom(framework: str) -> bool:
    return str(framework or "").strip().lower() == CUSTOM_FRAMEWORK


def enforce_custom_scope(framework: str) -> None:
    """Fail a custom round whose declared boundary cannot be enforced.

    Checked here, where ``FRAMEWORK_REPO_PATH`` is already exported, so an
    out-of-boundary campaign fails before spending any budget rather than at the first
    patch apply.

    The ``aoti`` block is parsed here too, and because nothing else ever reports it:
    every other caller of ``aoti_config`` swallows the error. A malformed block
    otherwise costs the whole run -- the agent is never told to export, since
    ``scope_instructions`` omits the declaration it could not read, while the gate stays
    shut on the same error.

    Raises:
        ValueError: When the declaration is unreadable or the boundary does not hold.
    """
    if not is_custom(framework) or optimization_scope() is None:
        return
    check_patch_scope(Path(os.environ.get("FRAMEWORK_REPO_PATH", "")), [])
    aoti_config()


def custom_workload(
    workload: ScriptableWorkload, framework: str, run_dir: str | Path, *, profile: bool
) -> ScriptableWorkload:
    """``workload`` with the params a custom round adds; unchanged for any other framework."""
    if not is_custom(framework):
        return workload
    params = dict(workload.params)
    layout = RunLayout(Path(run_dir))
    if profile:
        params[PROFILE_TRACE_DIR_ENV] = str(layout.profile_torch_trace)
        _enable_aoti_kernel_profile(params)
    _arm_aoti_evidence(params, layout.root)
    return replace(workload, params=params)


def _arm_aoti_evidence(params: dict[str, object], workspace: Path) -> None:
    """Deliver the export counters to the benchmark child and point them at its run.

    Both halves are required and only one is obvious. Setting the directory arms
    ``hl_aoti_evidence``, but the shim only runs if Python imports the bundled
    ``sitecustomize``, which needs its directory on ``PYTHONPATH``. Armed without
    delivered, the counters are never written and the export gate -- which has no source
    fallback -- stays shut for the whole campaign, including on ``profile``, the one
    action that delivers the shim by another route.

    Fails open, like the kernel-profile flag: an unreadable scope costs the gate its
    evidence, and the gate already refuses to open without it.
    """
    try:
        config = aoti_config()
    except ValueError:
        return
    if config is None or not config.enabled:
        return
    asset_dir = probe_asset_dir()
    if not asset_dir.is_dir():
        log.warning("custom workload: start-up shim assets missing at %s; the export gate cannot open", asset_dir)
        return
    params[EVIDENCE_DIR_ENV] = str(evidence_dir(workspace))
    # A param replaces the inherited value rather than adding to it, so the operator's
    # and the image's import paths have to be carried forward explicitly.
    prepend_launch_path(params, "PYTHONPATH", str(asset_dir))


def _enable_aoti_kernel_profile(params: dict[str, object]) -> None:
    """Ask Inductor for per-launch profiling scopes, on AOTI profiling runs only.

    An AOTI module compiles its launches into a ``.so``, and without this the trace loses
    roughly half its kernels -- measured 767 vs 1466 events and 7.2 vs 13.1 ms of
    captured GPU time on SmolVLA/gfx1151, at identical end-to-end latency. The missing
    ones are the Inductor-generated kernels, so the blind spot is exactly the exported
    region. Overhead is below noise (under ~1.5%, n=3 per arm), so the profile leg buys
    that visibility for nothing.

    The benchmark leg is held out for a different reason, not cost: the deliverable ships
    without this flag, so a round that compiles with it is not measuring what ships.

    Fails *open*, unlike the export gate -- an unreadable scope costs trace richness,
    which is not worth failing a run over.
    """
    try:
        config = aoti_config()
    except ValueError:
        return
    if config is not None and config.enabled:
        params[_KERNEL_PROFILE_ENV] = "1"


__all__ = [
    "CUSTOM_FRAMEWORK",
    "PROFILE_TRACE_DIR_ENV",
    "custom_workload",
    "enforce_custom_scope",
    "is_custom",
]
