# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Route gate for KernelForge's source-to-FlyDSL rewrite of one Forge attempt."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Container, Mapping, Sequence

_TOOLS_DIR = str(Path(__file__).resolve().parent.parent)
_TOOLS_DIR_INSERTED = _TOOLS_DIR not in sys.path
if _TOOLS_DIR_INSERTED:
    sys.path.insert(0, _TOOLS_DIR)
from _collective_names import kernel_name_implies_multigpu  # noqa: E402

if _TOOLS_DIR_INSERTED:
    sys.path.remove(_TOOLS_DIR)

log = logging.getLogger(__name__)

REWRITE_ENV = "HYPERLOOM_FORGE_REWRITE_BY_FLYDSL"
REWRITE_COMMAND = "forge-rewrite-by-flydsl"
CAPABILITIES_FLAG = "--capabilities-json"

# Consumer-side halves of the cross-repo contract.
PROTOCOL_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 2
RESULT_SENTINEL = "__FORGE_RESULT__"

# Which source languages can be rewritten is the producer's to declare, so it arrives through the capability handshake
# rather than living here.
SUPPORTED_FRAMEWORKS = frozenset({"aiter", "vllm", "sglang"})

# Mirrors kernelforge.cli MIN_MAX_HOURS (1.0h): the producer rejects a shorter --max-hours outright, so a budget that
# cannot reach it is ineligible rather than a child-process hard failure.
PRODUCER_MIN_BUDGET_SEC = 3600
# Head-room reserved on top of the producer's own budget so the apply-back commit is published before Hyperloom's
# absolute deadline kills the child.
APPLYBACK_RESERVE_SEC = 900
MIN_BUDGET_SEC = PRODUCER_MIN_BUDGET_SEC + APPLYBACK_RESERVE_SEC

CAPABILITY_PROBE_TIMEOUT_SEC = 60

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

_CAPABILITY_CACHE: dict[str, "RewriteCapabilities"] = {}


@dataclass(frozen=True)
class RewriteCapabilities:
    """What the installed KernelForge advertises for the rewrite route."""

    supported: bool
    reason: str
    detail: str = ""
    frameworks: tuple[str, ...] = ()
    # The file languages and the curated kinds the producer can port from.
    source_languages: tuple[str, ...] = ()
    source_kinds: tuple[str, ...] = ()
    # Optional and additive: a producer predating driver preparation simply omits it, which keeps the route on its own
    # synthesized driver.
    driver_preparation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "detail": self.detail,
            "frameworks": list(self.frameworks),
            "source_languages": list(self.source_languages),
            "source_kinds": list(self.source_kinds),
            "driver_preparation": self.driver_preparation,
        }

    def accepted_sources(self) -> frozenset[str]:
        """Every source name the producer said it can port from."""
        return frozenset(self.source_kinds) | frozenset(self.source_languages)

    def resolved_source(self, *, language: str, kind: str) -> str:
        """The source identity a candidate resolves to, kind before language."""
        return _rewritable_source(language, kind, self.accepted_sources())


@dataclass(frozen=True)
class RewriteCandidateSpec:
    """The candidate identity Hyperloom hands to the rewrite producer."""

    logical_operator: str
    source_kernel: str
    implementation_symbols: tuple[str, ...]
    source_entry: str
    shape_cases: tuple[dict[str, Any], ...]
    framework: str
    gpu_target: str
    driver: str
    branch: str
    attempt_id: str
    # Resolved from the trace, which knows more than the file: a traced Triton kernel lives in a ``.py`` that names no
    # language.
    source_language: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_operator": self.logical_operator,
            "source_kernel": self.source_kernel,
            "implementation_symbols": list(self.implementation_symbols),
            "source_entry": self.source_entry,
            "shape_cases": [dict(case) for case in self.shape_cases],
            "framework": self.framework,
            "gpu_target": self.gpu_target,
            "driver": self.driver,
            "branch": self.branch,
            "attempt_id": self.attempt_id,
            "source_language": self.source_language,
        }


@dataclass(frozen=True)
class RewriteDecision:
    """Whether one attempt may take the rewrite route, and why."""

    eligible: bool
    reason: str
    detail: str = ""
    spec: RewriteCandidateSpec | None = None
    capabilities: RewriteCapabilities | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "eligible": self.eligible,
            "reason": self.reason,
            "detail": self.detail,
        }
        if self.spec is not None:
            payload["spec"] = self.spec.as_dict()
        if self.capabilities is not None:
            payload["capabilities"] = self.capabilities.as_dict()
        return payload

    def with_driver(self, driver: str) -> "RewriteDecision":
        """Return the decision with the generated driver path recorded."""
        return replace(self, spec=replace(self.spec, driver=driver))


def rewrite_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Report whether the operator opted this session into the rewrite route."""
    source = os.environ if env is None else env
    return str(source.get(REWRITE_ENV) or "").strip().lower() in _TRUE_VALUES


def reset_capability_cache() -> None:
    """Drop the per-process capability answer so the next probe re-runs."""
    _CAPABILITY_CACHE.clear()


def _decode_capability_payload(stdout: str) -> dict[str, Any] | None:
    """Extract the capability object from producer stdout that may carry logs."""
    decoder = json.JSONDecoder()
    text = stdout or ""
    index = text.find("{")
    while index >= 0:
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(payload, dict):
            return payload
        index = text.find("{", index + 1)
    return None


def _int_version(payload: Mapping[str, Any], key: str) -> int | None:
    """Read one scalar version, rejecting booleans and malformed values."""
    raw = payload.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _int_versions(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    """Read one declared version list, dropping entries that are not integers."""
    raw = payload.get(key)
    if not isinstance(raw, (list, tuple)):
        return ()
    versions: list[int] = []
    for value in raw:
        # ``True`` would otherwise coerce to a version of 1.
        if isinstance(value, bool):
            continue
        try:
            versions.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(versions)


def _capability_names(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """Read one advertised name list, normalized and de-duplicated."""
    raw = payload.get(key)
    values = raw if isinstance(raw, (list, tuple)) else []
    names: list[str] = []
    for value in values:
        name = str(value or "").strip().lower().replace("-", "_")
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _validated_capabilities(payload: dict[str, Any] | None) -> RewriteCapabilities:
    """Check one capability payload against the versions this consumer reads."""
    if not isinstance(payload, dict):
        return RewriteCapabilities(False, "capability_payload_invalid", "capability output is not a JSON object")
    protocol = _int_version(payload, "rewrite_protocol_version")
    if protocol != PROTOCOL_VERSION:
        return RewriteCapabilities(
            False,
            "capability_protocol_unsupported",
            f"producer rewrite protocol version {protocol!r} is not {PROTOCOL_VERSION}",
        )
    schemas = _int_versions(payload, "artifact_schema_versions")
    if ARTIFACT_SCHEMA_VERSION not in schemas:
        return RewriteCapabilities(
            False,
            "capability_artifact_schema_unsupported",
            f"producer artifact schemas {list(schemas)} exclude {ARTIFACT_SCHEMA_VERSION}",
        )
    sentinel = str(payload.get("result_sentinel") or "").strip()
    if sentinel != RESULT_SENTINEL:
        return RewriteCapabilities(
            False,
            "capability_sentinel_mismatch",
            f"producer result sentinel {sentinel!r} is not {RESULT_SENTINEL!r}",
        )
    frameworks = _capability_names(payload, "frameworks")
    if not frameworks:
        return RewriteCapabilities(
            False,
            "capability_frameworks_missing",
            "producer advertises no apply-back frameworks",
        )
    source_languages = _capability_names(payload, "source_languages")
    source_kinds = _capability_names(payload, "source_kinds")
    if not source_languages and not source_kinds:
        return RewriteCapabilities(
            False,
            "capability_source_languages_missing",
            "producer advertises no source languages or kinds it can port from",
        )
    return RewriteCapabilities(
        True,
        "capability_ok",
        "",
        frameworks,
        source_languages=source_languages,
        source_kinds=source_kinds,
        driver_preparation=payload.get("driver_preparation") is True,
    )


def probe_capabilities() -> RewriteCapabilities:
    """Ask the installed producer what rewrite contract it speaks."""
    cache_key = "<installed>"
    cached = _CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    child_env = dict(os.environ)
    cmd = [sys.executable, "-m", "kernelforge.cli", REWRITE_COMMAND, CAPABILITIES_FLAG]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=child_env,
            timeout=CAPABILITY_PROBE_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        capabilities = RewriteCapabilities(
            False,
            "capability_probe_failed",
            f"{type(exc).__name__}: {exc}",
        )
    else:
        if proc.returncode != 0:
            capabilities = RewriteCapabilities(
                False,
                "capability_probe_failed",
                f"{REWRITE_COMMAND} {CAPABILITIES_FLAG} exited rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[-400:]}",
            )
        else:
            capabilities = _validated_capabilities(_decode_capability_payload(proc.stdout or ""))
    _CAPABILITY_CACHE[cache_key] = capabilities
    return capabilities


# The producer requires --driver to name an existing file before it will decide whether to prepare one, so an attempt
# with no synthesizable contract still has to hand over something.
REWRITE_DRIVER_SEED_TEMPLATE = '''#!/usr/bin/env python3
"""Placeholder rewrite driver, replaced by the producer's preparation stage.

Hyperloom could not rebuild a faithful invocation for this operator from traced
shapes, so it handed the producer the invocation spec and this stub instead.
Nothing here is meant to run.
"""
import sys

print(
    "rewrite_driver_error: placeholder driver; the producer's driver-preparation "
    "stage must author the real one from the invocation spec",
    file=sys.stderr,
)
raise SystemExit(1)
'''


def build_rewrite_driver_seed(
    *,
    workspace: str,
    writer: Callable[[str, str], str],
) -> str:
    """Write the placeholder driver the producer's preparation stage replaces."""
    return writer(workspace, REWRITE_DRIVER_SEED_TEMPLATE)


def _shape_cases(
    shape_cases: Sequence[Any] | None,
    shapes: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Normalize the grouped shape context, falling back to the single case."""
    cases = [dict(case) for case in (shape_cases or []) if isinstance(case, dict)]
    if cases:
        return tuple(cases)
    return (dict(shapes),) if shapes else ()


def _source_entry_hint(candidate: Mapping[str, Any] | None) -> str:
    """Return the optional host-entry hint echoed back by the producer."""
    candidate = candidate or {}
    explicit = str(candidate.get("source_entry") or "").strip()
    if explicit:
        return explicit
    symbol = candidate.get("source_symbol")
    return str(symbol).strip() if isinstance(symbol, str) else ""


def _rewritable_source(language: str, kind: str, accepted: Container[str]) -> str:
    """Resolve which rewrite source a candidate is, kind first then language."""
    return kind if kind in accepted else language


def _is_multi_node() -> bool:
    """Report multi-node fan-out through the apply-side authority."""
    tools_dir = str(Path(__file__).resolve().parent.parent)
    inserted = tools_dir not in sys.path
    if inserted:
        sys.path.insert(0, tools_dir)
    try:
        import apply_kernel_patch

        return bool(apply_kernel_patch._is_multi_node())
    except (ImportError, AttributeError):
        log.warning("forge route: cannot resolve node fan-out; treating the session as multi-node")
        return True
    finally:
        if inserted and tools_dir in sys.path:
            sys.path.remove(tools_dir)


def _mapped_into_workspace(paths: Sequence[str], workspace: str) -> str:
    """Return the first path that does not resolve inside ``workspace``."""
    root = Path(workspace).resolve()
    for raw in paths:
        path = str(raw or "").strip()
        if not path:
            continue
        resolved = Path(path).resolve()
        if resolved != root and not resolved.is_relative_to(root):
            return path
    return ""


def evaluate_rewrite_route(
    *,
    candidate: Mapping[str, Any] | None,
    source_type: str,
    kernel_kind: str,
    logical_operator: str,
    source_kernel: str,
    workspace: str,
    implementation_sources: Sequence[str],
    implementation_symbols: Sequence[str],
    framework: str,
    gpu_target: str,
    shape_cases: Sequence[Any] | None,
    shapes: Mapping[str, Any] | None,
    branch: str,
    attempt_id: str,
    timeout_s: int,
    invocation_spec_file: str = "",
    capability_probe: Callable[..., RewriteCapabilities] | None = None,
) -> RewriteDecision:
    """Decide whether one prepared Forge attempt may take the rewrite route."""
    if not rewrite_enabled():
        return RewriteDecision(False, "route_disabled", f"{REWRITE_ENV} is not set")

    kind = str(kernel_kind or "").strip().lower().replace("-", "_")
    language = str(source_type or "").strip().lower()
    if "flydsl" in kind or language == "flydsl":
        return RewriteDecision(False, "already_flydsl_source", "candidate is already a FlyDSL kernel")
    # Ahead of the handshake: there is nothing to port without readable source, so widening the producer's advertised
    # languages must never reach this.
    if "asm" in kind or "prebuilt" in kind:
        return RewriteDecision(False, "prebuilt_binary_unsupported", f"kernel_kind={kernel_kind}")

    candidate = candidate or {}
    if bool(candidate.get("is_multigpu")) or kernel_name_implies_multigpu(
        logical_operator or str(candidate.get("name") or "")
    ):
        return RewriteDecision(False, "collective_unsupported", "candidate is a multi-GPU collective")

    canonical_framework = str(framework or "").strip().lower()
    if canonical_framework not in SUPPORTED_FRAMEWORKS:
        return RewriteDecision(False, "framework_unsupported", f"framework={framework or 'unresolved'}")

    # Multi-node apply runs a separate stdlib path-safety allowlist that this route does not feed, so it must fail
    # here rather than at apply time.
    if _is_multi_node():
        return RewriteDecision(False, "multi_node_unsupported", "apply-back is single-node only")

    if timeout_s < MIN_BUDGET_SEC:
        return RewriteDecision(
            False,
            "budget_insufficient",
            f"remaining budget {timeout_s}s is below the {MIN_BUDGET_SEC}s rewrite minimum",
        )

    unmapped = _mapped_into_workspace([source_kernel, *implementation_sources], workspace)
    if unmapped:
        return RewriteDecision(
            False,
            "workspace_mapping_unresolved",
            f"source outside the prepared workspace: {unmapped}",
        )

    symbols = tuple(str(symbol).strip() for symbol in implementation_symbols if str(symbol or "").strip())
    if not symbols:
        return RewriteDecision(False, "target_functions_missing", "no implementation symbol resolved")

    probe = capability_probe or probe_capabilities
    capabilities = probe()
    if not capabilities.supported:
        return RewriteDecision(False, capabilities.reason, capabilities.detail, capabilities=capabilities)
    if canonical_framework not in capabilities.frameworks:
        return RewriteDecision(
            False,
            "capability_framework_unsupported",
            f"producer frameworks {list(capabilities.frameworks)} exclude {canonical_framework}",
            capabilities=capabilities,
        )
    # Which languages are portable is the producer's to state: it owns the port prompt and the entry resolution that
    # have to read the source.
    resolved_source = capabilities.resolved_source(language=language, kind=kind)
    if resolved_source not in capabilities.accepted_sources():
        return RewriteDecision(
            False,
            "source_type_unsupported",
            f"source_type={source_type} kernel_kind={kernel_kind} is outside the "
            f"producer's languages {list(capabilities.source_languages)} / kinds "
            f"{list(capabilities.source_kinds)}",
            capabilities=capabilities,
        )
    # An operator's real invocation cannot be rebuilt from traced shapes alone -- quantized and routed operands carry
    # scale and index meanings the trace does not describe -- so the producer authors the driver from the invocation
    # spec.
    if not capabilities.driver_preparation:
        return RewriteDecision(
            False,
            "driver_preparation_unsupported",
            "the producer does not advertise driver preparation",
            capabilities=capabilities,
        )
    # Preparation needs the invocation evidence; without it the producer keeps the placeholder driver, which exits 1
    # after burning the whole budget.
    from hyperloom.common.invocation_spec_readiness import (  # noqa: PLC0415 - keep module import-light
        evaluate_spec_readiness,
    )

    readiness = evaluate_spec_readiness(invocation_spec_file)
    if not readiness.ready:
        return RewriteDecision(
            False,
            readiness.reason,
            readiness.detail,
            capabilities=capabilities,
        )

    spec = RewriteCandidateSpec(
        logical_operator=logical_operator,
        source_kernel=source_kernel,
        implementation_symbols=symbols,
        source_entry=_source_entry_hint(candidate),
        shape_cases=_shape_cases(shape_cases, shapes),
        framework=canonical_framework,
        gpu_target=gpu_target,
        driver="",
        branch=branch,
        attempt_id=attempt_id,
        source_language=resolved_source,
    )
    return RewriteDecision(True, "eligible", "", spec=spec, capabilities=capabilities)


# What ``forge_submit``, the only consumer outside this module, depends on.
__all__ = [
    "APPLYBACK_RESERVE_SEC",
    "PRODUCER_MIN_BUDGET_SEC",
    "RESULT_SENTINEL",
    "REWRITE_COMMAND",
    "RewriteDecision",
    "build_rewrite_driver_seed",
    "evaluate_rewrite_route",
]
