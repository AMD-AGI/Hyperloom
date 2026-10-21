# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The resume step that puts the kept source patches back on the framework tree."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

from ..source_snapshot import install_source_layers

log = logging.getLogger(__name__)


def install_kept_source_layers(state: Any, session_dir: Path, report: dict[str, Any]) -> str | None:
    """Install the kept source patches on a tree that lost them; return why the resume must halt, if it must.

    The patches live on the framework tree and the session records them in durable snapshots, so a resume onto a
    fresh tree would otherwise bench every candidate against the stock source.
    """
    result = install_source_layers(state.optimization_stack or [])
    report["warnings"].extend(
        {"kind": "source_layer_not_reproducible", "variant": variant} for variant in result.unreproducible
    )
    if result.installed:
        log.warning("resume: installed %d kept source file(s) the framework tree did not hold", len(result.installed))
        report["fixes"].append({"kind": "reinstalled_source_layer", "files": list(result.installed)})
    if not result.failed:
        return None
    report["warnings"].append({"kind": "source_layer_install_failed", "failures": list(result.failed)})
    state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
    state.save(session_dir)
    return "the kept source layer cannot be installed on the framework tree: " + "; ".join(result.failed)
