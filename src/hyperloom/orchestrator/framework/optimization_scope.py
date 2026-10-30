# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional operator-owned single-file scope for custom script workloads."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..specialists.patch_safety import parse_patch_targets

log = logging.getLogger(__name__)

# Schema 2 adds the optional "aoti" block: the operator's opt-in to ahead-of-time
# export, plus the dynamic input dims the agent must not guess.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)


@dataclass(frozen=True)
class AotiDim:
    """One input dimension the operator requires to vary at runtime."""

    input: str
    dim: int
    min: int
    max: int
    derivation: str


@dataclass(frozen=True)
class AotiConfig:
    """The operator's ahead-of-time export declaration."""

    enabled: bool
    pin_all_others: bool
    dynamic_dims: tuple[AotiDim, ...]


def _read_scope(directory: str | None = None) -> dict | None:
    """Load and validate the declaration, or return None when none is declared.

    Every failure surfaces as ``ValueError``. The declaration sits on an
    operator-supplied path that is often a network mount, so reading it can raise
    ``OSError``, and a hand-written file can be any shape at all; callers on the
    dispatch and promotion paths should not have to enumerate those.
    """
    if directory is None:
        directory = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    if not directory:
        return None
    declaration = Path(directory) / "optimization_scope.json"
    try:
        if not declaration.exists():
            return None
        scope = json.loads(declaration.read_text())
        path = Path(scope["file"])
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable single-file optimization scope: {exc}") from exc
    if scope.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS or not path.is_absolute() or path.suffix != ".py":
        raise ValueError("Invalid single-file optimization scope")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Optimization file must be an existing regular Python file")
    return scope


def optimization_file(directory: str | None = None) -> Path | None:
    """Read the write boundary from the frozen benchmark directory, if declared."""
    scope = _read_scope(directory)
    if scope is None:
        return None
    return Path(scope["file"]).resolve()


def _parse_dim(entry: object, index: int) -> AotiDim:
    """Validate one dynamic-dimension entry, naming the offender on failure."""
    where = f"aoti.dynamic_dims[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be an object")
    missing = {"input", "dim", "min", "max", "derivation"} - set(entry)
    if missing:
        raise ValueError(f"{where} is missing {sorted(missing)}")
    name = entry["input"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}.input must be a non-empty string")
    derivation = entry["derivation"]
    if not isinstance(derivation, str) or not derivation.strip():
        raise ValueError(f"{where}.derivation must say how the bound was measured")
    for key in ("dim", "min", "max"):
        if not isinstance(entry[key], int) or isinstance(entry[key], bool):
            raise ValueError(f"{where}.{key} must be an integer")
    if entry["dim"] < 0:
        raise ValueError(f"{where}.dim must be a non-negative axis index")
    if entry["min"] < 1:
        raise ValueError(f"{where}.min must be at least 1")
    if entry["min"] >= entry["max"]:
        raise ValueError(f"{where} requires min < max; pin the dimension instead of declaring a degenerate range")
    return AotiDim(
        input=name,
        dim=entry["dim"],
        min=entry["min"],
        max=entry["max"],
        derivation=derivation,
    )


def aoti_config(directory: str | None = None) -> AotiConfig | None:
    """Read the operator's ahead-of-time export declaration, if any."""
    scope = _read_scope(directory)
    if scope is None or "aoti" not in scope:
        return None
    # Schema 1 predates the block. A stray one must not silently enable export.
    if scope["schema_version"] < 2:
        return None
    block = scope["aoti"]
    if not isinstance(block, dict):
        raise ValueError("aoti must be an object")
    enabled = block.get("enabled", False)
    pin_all_others = block.get("pin_all_others", True)
    for key, value in (("enabled", enabled), ("pin_all_others", pin_all_others)):
        if not isinstance(value, bool):
            raise ValueError(f"aoti.{key} must be a boolean")
    raw_dims = block.get("dynamic_dims", [])
    if not isinstance(raw_dims, list):
        raise ValueError("aoti.dynamic_dims must be a list")
    dims = tuple(_parse_dim(entry, index) for index, entry in enumerate(raw_dims))
    seen: set[tuple[str, int]] = set()
    for dim in dims:
        if (dim.input, dim.dim) in seen:
            raise ValueError(f"aoti.dynamic_dims declares {dim.input} axis {dim.dim} twice")
        seen.add((dim.input, dim.dim))
    return AotiConfig(enabled=enabled, pin_all_others=pin_all_others, dynamic_dims=dims)


def check_patch_scope(root: Path, texts: list[str]) -> None:
    """Reject patches outside the sole model artifact before applying them."""
    allowed = optimization_file()
    if allowed is None:
        return
    if root.resolve() != allowed.parent:
        raise ValueError(f"Single-file workload permits changes only to {allowed}")
    for text in texts:
        targets = parse_patch_targets(text).all
        if not targets or set(targets) != {allowed.name}:
            raise ValueError(f"Single-file workload permits changes only to {allowed.name}")


def _aoti_instructions(config: AotiConfig) -> str:
    """Describe the operator's export declaration to optimization agents."""
    if config.dynamic_dims:
        declared = "; ".join(
            f"{dim.input} axis {dim.dim} in [{dim.min}, {dim.max}] ({dim.derivation})" for dim in config.dynamic_dims
        )
        dims = f"The operator requires these dimensions to stay dynamic: {declared}. "
    else:
        dims = "The operator declares no dynamic dimensions; export at fixed shapes. "
    pinned = (
        "Every dimension not listed above must be pinned to its benchmark value. "
        if config.pin_all_others
        else "Dimensions not listed above may vary, but a pinned dimension is one less chance "
        "for the compiler to fail to prove something. "
    )
    return (
        " Ahead-of-time export is the objective for this workload: export the whole model with "
        "torch.export and compile it with torch._inductor.aoti_compile_and_package. Until a KEEP "
        "lands whose module exports, the dispatcher denies explore/sweep/profile/roofline with "
        "rule='export_gate'. Do not fall back to eager on failure -- that records a KEEP for a "
        "path the deployment never takes. " + dims + pinned + "These bounds are baked into the "
        "artifact permanently; do not widen, narrow or invent one. See "
        "references/framework_rewrite_patterns.md (h) for how to get a model tracing."
    )


def scope_instructions() -> str:
    """Describe the operator's artifact boundary to optimization agents."""
    allowed = optimization_file()
    if allowed is None:
        return ""
    base = (
        f"Single-file optimization contract: only {allowed} may change. "
        "It must define hyperloom_optimize(model) returning the optimized callable. "
        "The user script owns inputs, weights, validation and latency measurement. "
        "Do not edit that script or installed packages. All reusable optimization logic, "
        "including custom kernels and runtime setup, must live in this one Python module. "
        "External environment changes, overlays and auxiliary source files are not deliverables."
    )
    # A malformed block must not take down prompt building; the agent is simply not
    # told about an export it could not have been given valid bounds for.
    try:
        config = aoti_config()
    except ValueError:
        log.warning("optimization scope declares an unusable aoti block; omitting export instructions")
        return base
    if config is None or not config.enabled:
        return base
    return base + _aoti_instructions(config)
