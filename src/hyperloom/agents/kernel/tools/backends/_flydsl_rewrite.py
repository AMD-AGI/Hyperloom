# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Route gate for KernelForge's source-to-FlyDSL rewrite of one Forge attempt.

The generic per-kernel route optimizes a kernel in its own language and
consumes a schema-1 ``best_result.json``. The rewrite route instead asks
KernelForge to port the kernel to FlyDSL and publish a framework apply-back
patch. That is a different producer contract, so an attempt may only switch
routes when the operator opted in, the candidate matches the supported MVP
shape, and the installed producer advertises the protocol/schema/driver
versions this consumer knows how to read.

Every verdict carries a stable reason code. A negative verdict is never a
kernel skip: the attempt stays on the generic forge-loop route untouched.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

# Consumer-side halves of the cross-repo contract. Bumping any of these means
# this module can no longer read what an older producer emits.
PROTOCOL_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2
DRIVER_CONTRACT_VERSION = 1
RESULT_SENTINEL = "__FORGE_RESULT__"

SUPPORTED_SOURCE_TYPES = frozenset({"triton"})
SUPPORTED_FRAMEWORKS = frozenset({"aiter", "vllm", "sglang"})

# Mirrors kernel_agents.cli MIN_MAX_HOURS (1.0h): the producer rejects a shorter
# --max-hours outright, so a budget that cannot reach it is ineligible rather
# than a child-process hard failure.
PRODUCER_MIN_BUDGET_SEC = 3600
# Head-room reserved on top of the producer's own budget so the apply-back
# commit is published before Hyperloom's absolute deadline kills the child.
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "detail": self.detail,
            "frameworks": list(self.frameworks),
        }


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
    operator_family: str
    driver: str
    branch: str
    attempt_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_operator": self.logical_operator,
            "source_kernel": self.source_kernel,
            "implementation_symbols": list(self.implementation_symbols),
            "source_entry": self.source_entry,
            "shape_cases": [dict(case) for case in self.shape_cases],
            "framework": self.framework,
            "gpu_target": self.gpu_target,
            "operator_family": self.operator_family,
            "driver": self.driver,
            "branch": self.branch,
            "attempt_id": self.attempt_id,
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
    """Report whether the operator opted this session into the rewrite route.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        ``True`` when ``$HYPERLOOM_FORGE_REWRITE_BY_FLYDSL`` is truthy.
    """
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


def _capability_frameworks(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the apply-back frameworks the producer advertises."""
    raw = payload.get("frameworks")
    values = raw if isinstance(raw, (list, tuple)) else []
    frameworks: list[str] = []
    for value in values:
        name = str(value or "").strip().lower()
        if name and name not in frameworks:
            frameworks.append(name)
    return tuple(frameworks)


def _validated_capabilities(payload: dict[str, Any] | None) -> RewriteCapabilities:
    """Check one capability payload against the versions this consumer reads."""
    if not isinstance(payload, dict):
        return RewriteCapabilities(False, "capability_payload_invalid", "capability output is not a JSON object")
    protocols = _int_versions(payload, "protocol_versions")
    if PROTOCOL_VERSION not in protocols:
        return RewriteCapabilities(
            False,
            "capability_protocol_unsupported",
            f"producer protocol versions {list(protocols)} exclude {PROTOCOL_VERSION}",
        )
    schemas = _int_versions(payload, "artifact_schema_versions")
    if ARTIFACT_SCHEMA_VERSION not in schemas:
        return RewriteCapabilities(
            False,
            "capability_artifact_schema_unsupported",
            f"producer artifact schemas {list(schemas)} exclude {ARTIFACT_SCHEMA_VERSION}",
        )
    contracts = _int_versions(payload, "driver_contract_versions")
    if DRIVER_CONTRACT_VERSION not in contracts:
        return RewriteCapabilities(
            False,
            "capability_driver_contract_unsupported",
            f"producer driver contracts {list(contracts)} exclude {DRIVER_CONTRACT_VERSION}",
        )
    sentinel = str(payload.get("result_sentinel") or "").strip()
    if sentinel != RESULT_SENTINEL:
        return RewriteCapabilities(
            False,
            "capability_sentinel_mismatch",
            f"producer result sentinel {sentinel!r} is not {RESULT_SENTINEL!r}",
        )
    frameworks = _capability_frameworks(payload)
    if not frameworks:
        return RewriteCapabilities(
            False,
            "capability_frameworks_missing",
            "producer advertises no apply-back frameworks",
        )
    return RewriteCapabilities(True, "capability_ok", "", frameworks)


def probe_capabilities(*, forge_root: str = "") -> RewriteCapabilities:
    """Ask the installed producer what rewrite contract it speaks.

    The answer is cached for the process: it describes the installed
    KernelForge, not the candidate, and the probe must not cost a subprocess
    per attempt. ``--capabilities-json`` is an eager short-circuit option, so a
    failure here is reported as-is and never re-tried with guessed arguments.

    Args:
        forge_root: Directory holding ``kernel_agents``, prepended to the child
            ``PYTHONPATH``; empty relies on an installed package.

    Returns:
        The validated :class:`RewriteCapabilities` for this process.
    """
    cache_key = forge_root or "<installed>"
    cached = _CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    child_env = dict(os.environ)
    if forge_root:
        child_env["PYTHONPATH"] = forge_root + os.pathsep + child_env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "kernel_agents.cli", REWRITE_COMMAND, CAPABILITIES_FLAG]
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


ENV_SOURCE_KERNEL = "KERNELFORGE_REWRITE_SOURCE_KERNEL"
ENV_CANDIDATE_KERNEL = "KERNELFORGE_REWRITE_CANDIDATE_KERNEL"
ENV_BUILDER_SYMBOL = "KERNELFORGE_REWRITE_BUILDER_SYMBOL"
ENV_LOGICAL_OP = "KERNELFORGE_REWRITE_LOGICAL_OP"

_CONTRACT_PLACEHOLDER = "__HYPERLOOM_REWRITE_CONTRACT__"

# Dual-mode driver for one rewrite attempt. The producer re-invokes it per
# iteration with the candidate boundaries in the environment, so the script
# resolves both implementations at run time and never names a producer file.
REWRITE_DRIVER_TEMPLATE = '''#!/usr/bin/env python3
"""Generated dual-mode driver comparing a source kernel with its FlyDSL rewrite."""
import argparse
import functools
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
import time

import torch

CONTRACT = json.loads(__HYPERLOOM_REWRITE_CONTRACT__)

ENV_SOURCE_KERNEL = "KERNELFORGE_REWRITE_SOURCE_KERNEL"
ENV_CANDIDATE_KERNEL = "KERNELFORGE_REWRITE_CANDIDATE_KERNEL"
ENV_BUILDER_SYMBOL = "KERNELFORGE_REWRITE_BUILDER_SYMBOL"
ENV_LOGICAL_OP = "KERNELFORGE_REWRITE_LOGICAL_OP"

TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
# A Triton kernel needs an explicit launch grid, so it is never the host-level
# callable a driver can invoke with plain tensors.
NON_HOST_CALLABLES = ("JITFunction", "Autotuner", "Heuristics")

_MODULES = {}
_LAUNCHERS = {}


class DriverError(RuntimeError):
    """The driver could not run the requested implementation."""


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_module(env_key, module_name):
    if module_name in _MODULES:
        return _MODULES[module_name]
    path = (os.environ.get(env_key) or "").strip()
    if not path:
        raise DriverError(env_key + " is not set")
    if not os.path.isfile(path):
        raise DriverError(env_key + " points at a missing file: " + path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DriverError("cannot import " + path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULES[module_name] = module
    return module


def _is_host_callable(value):
    return callable(value) and type(value).__name__ not in NON_HOST_CALLABLES


def _source_entry():
    module = _load_module(ENV_SOURCE_KERNEL, "_forge_rewrite_source")
    for name in CONTRACT.get("entry_symbols") or []:
        entry = getattr(module, name, None)
        if _is_host_callable(entry):
            return entry
    for name, value in vars(module).items():
        if not name.startswith("_") and inspect.isfunction(value):
            return value
    raise DriverError("no host-level callable found in the source module")


def _resolve_launcher(builder):
    """Return the callable to invoke: a zero-argument builder yields it."""
    try:
        parameters = inspect.signature(builder).parameters.values()
    except (TypeError, ValueError):
        return builder
    required = [
        parameter
        for parameter in parameters
        if parameter.default is parameter.empty
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    ]
    if required:
        return builder
    built = builder()
    return built if callable(built) else builder


def _candidate_entry():
    if "candidate" in _LAUNCHERS:
        return _LAUNCHERS["candidate"]
    module = _load_module(ENV_CANDIDATE_KERNEL, "_forge_rewrite_candidate")
    symbol = (os.environ.get(ENV_BUILDER_SYMBOL) or "").strip()
    if not symbol:
        raise DriverError(ENV_BUILDER_SYMBOL + " is not set")
    builder = getattr(module, symbol, None)
    if not callable(builder):
        raise DriverError("candidate module exposes no callable " + symbol)
    launcher = _resolve_launcher(builder)
    _LAUNCHERS["candidate"] = launcher
    return launcher


def _build_inputs(case):
    """Materialize one case identically for every mode."""
    torch.manual_seed(int(CONTRACT.get("seed") or 0))
    device = _device()
    tensors = []
    for entry in case.get("inputs") or []:
        dtype = TORCH_DTYPES.get(entry.get("dtype"))
        if dtype is None:
            raise DriverError("unsupported dtype " + str(entry.get("dtype")))
        # Draw in fp32 and cast so the random stream does not depend on dtype.
        drawn = torch.randn(tuple(entry["shape"]), device=device, dtype=torch.float32)
        tensors.append(drawn.to(dtype))
    if not tensors:
        raise DriverError("case " + str(case.get("case_id")) + " declares no inputs")
    return tensors


def _output_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, torch.Tensor)]
    return []


def _snr_db(reference, actual):
    ref = reference.detach().float()
    error = (ref - actual.detach().float()).norm().item()
    if not math.isfinite(error):
        return -120.0
    if error == 0.0:
        return 120.0
    return 20.0 * math.log10(max(ref.norm().item(), 1e-12) / error)


def _median_ms(call, warmup, iters):
    for _ in range(max(1, warmup)):
        call()
    samples = []
    if _device() == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(max(1, iters)):
            start.record()
            call()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    else:
        for _ in range(max(1, iters)):
            started = time.perf_counter()
            call()
            samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def _selected_cases(shape):
    cases = CONTRACT.get("cases") or []
    selector_key = str(CONTRACT.get("case_selector_key") or "CASE_ID")
    wanted = ""
    for part in (shape or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().upper() == selector_key:
            wanted = value.strip()
    if not wanted:
        return cases
    matched = [case for case in cases if str(case.get("case_id")) == wanted]
    return matched or cases


def _run_bench(entry, cases, args):
    """Time one implementation over every case and report the workload total."""
    total = 0.0
    for case in cases:
        tensors = _build_inputs(case)
        median = _median_ms(functools.partial(entry, *tensors), args.warmup, args.iters)
        print("[case] %s ms=%.6f" % (case.get("case_id"), median))
        total += median
    print("median_ms: %.6f" % total)
    print("wall_ms: %.6f" % total)


def _run_correctness(cases, args):
    source = _source_entry()
    candidate = _candidate_entry()
    worst_snr = None
    structural_ok = True
    for case in cases:
        case_id = str(case.get("case_id"))
        tensors = _build_inputs(case)
        reference = _output_tensors(source(*[tensor.clone() for tensor in tensors]))
        actual = _output_tensors(candidate(*[tensor.clone() for tensor in tensors]))
        if not reference:
            raise DriverError("source produced no tensor output for " + case_id)
        if len(actual) != len(reference):
            print("[case] %s outputs=%d expected=%d" % (case_id, len(actual), len(reference)))
            structural_ok = False
            continue
        for index, (expected, produced) in enumerate(zip(reference, actual)):
            if tuple(expected.shape) != tuple(produced.shape):
                print(
                    "[case] %s output=%d shape=%s expected=%s"
                    % (case_id, index, tuple(produced.shape), tuple(expected.shape))
                )
                structural_ok = False
                continue
            snr = _snr_db(expected, produced)
            worst_snr = snr if worst_snr is None else min(worst_snr, snr)
            print("[case] %s output=%d snr_db=%.2f" % (case_id, index, snr))
    if worst_snr is None:
        structural_ok = False
    if not structural_ok:
        # An explicit negative; the producer applies its own SNR threshold to a
        # structurally valid comparison, so no allclose line is emitted then.
        print("allclose: False")
        raise DriverError("candidate outputs do not match the source signature")
    print("SNR: %.2f dB" % worst_snr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--warmup", type=int, default=int(CONTRACT.get("warmup") or 10))
    parser.add_argument("--iters", type=int, default=int(CONTRACT.get("iters") or 30))
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--ref-bench-mode", action="store_true")
    args, _ = parser.parse_known_args()

    logical_op = (os.environ.get(ENV_LOGICAL_OP) or "").strip()
    expected_op = str(CONTRACT.get("logical_operator") or "")
    if logical_op and expected_op and logical_op != expected_op:
        print(
            "rewrite_driver_warning: logical operator %r does not match %r" % (logical_op, expected_op),
            file=sys.stderr,
        )

    cases = _selected_cases(args.shape)
    if not cases:
        print("rewrite_driver_error: contract declares no cases", file=sys.stderr)
        raise SystemExit(1)
    try:
        if args.ref_bench_mode:
            _run_bench(_source_entry(), cases, args)
        elif args.bench_mode:
            _run_bench(_candidate_entry(), cases, args)
        else:
            _run_correctness(cases, args)
    except DriverError as error:
        print("rewrite_driver_error: %s" % error, file=sys.stderr)
        raise SystemExit(1)


main()
'''


def build_rewrite_driver(
    contract: Mapping[str, Any],
    *,
    workspace: str,
    writer: Callable[[str, str], str],
) -> str:
    """Write the dual-mode rewrite driver into the producer workspace.

    Args:
        contract: The rewrite driver contract describing cases and entries.
        workspace: The prepared Forge workspace the driver must live in.
        writer: Allocator for a driver file inside ``workspace``, sharing the
            naming and cleanup contract of every other generated driver.

    Returns:
        str: The path of the generated driver.
    """
    payload = json.dumps(dict(contract), sort_keys=True)
    content = REWRITE_DRIVER_TEMPLATE.replace(_CONTRACT_PLACEHOLDER, repr(payload))
    return writer(workspace, content)


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


def _is_multi_node() -> bool:
    """Report multi-node fan-out through the apply-side authority.

    Returns:
        ``True`` when the session fans out over several nodes, and also when
        that cannot be determined: the route runs only where single-node apply
        is proven.
    """
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
    driver_contract: Mapping[str, Any] | None,
    shape_cases: Sequence[Any] | None,
    shapes: Mapping[str, Any] | None,
    branch: str,
    attempt_id: str,
    timeout_s: int,
    forge_root: str = "",
    capability_probe: Callable[..., RewriteCapabilities] | None = None,
) -> RewriteDecision:
    """Decide whether one prepared Forge attempt may take the rewrite route.

    Local candidate facts are checked before the producer is probed, so an
    ineligible candidate never spends a subprocess or any rewrite budget.

    Args:
        candidate: The kernel candidate payload.
        source_type: Detected source language of the candidate.
        kernel_kind: Curated kernel kind that refines ``source_type``.
        logical_operator: Stable workload/KB operator identity.
        source_kernel: Workspace path of the kernel to rewrite.
        workspace: Prepared Forge workspace root.
        implementation_sources: Declared sources remapped into the workspace.
        implementation_symbols: Target functions the rewrite must cover.
        framework: Resolved apply-back framework identity.
        gpu_target: Resolved gfx target.
        driver_contract: Contract describing how a rewrite driver rebuilds this
            operator's invocation; empty when the family is unsupported.
        shape_cases: Grouped shape cases from the task group.
        shapes: Single-case shape mapping used when no group exists.
        branch: Unique branch created for this attempt.
        attempt_id: Unique attempt identity.
        timeout_s: Remaining wall-clock budget for the attempt.
        forge_root: Directory holding ``kernel_agents`` for the probe child.
        capability_probe: Injection point for the capability probe.

    Returns:
        A :class:`RewriteDecision`; ineligible verdicts keep the generic route.
    """
    if not rewrite_enabled():
        return RewriteDecision(False, "route_disabled", f"{REWRITE_ENV} is not set")

    kind = str(kernel_kind or "").strip().lower().replace("-", "_")
    language = str(source_type or "").strip().lower()
    if "flydsl" in kind or language == "flydsl":
        return RewriteDecision(False, "already_flydsl_source", "candidate is already a FlyDSL kernel")
    if "asm" in kind or "prebuilt" in kind:
        return RewriteDecision(False, "prebuilt_binary_unsupported", f"kernel_kind={kernel_kind}")
    if language not in SUPPORTED_SOURCE_TYPES:
        return RewriteDecision(False, "source_type_unsupported", f"source_type={source_type}")

    candidate = candidate or {}
    if bool(candidate.get("is_multigpu")) or kernel_name_implies_multigpu(
        logical_operator or str(candidate.get("name") or "")
    ):
        return RewriteDecision(False, "collective_unsupported", "candidate is a multi-GPU collective")

    canonical_framework = str(framework or "").strip().lower()
    if canonical_framework not in SUPPORTED_FRAMEWORKS:
        return RewriteDecision(False, "framework_unsupported", f"framework={framework or 'unresolved'}")

    # Multi-node apply runs a separate stdlib path-safety allowlist that this
    # route does not feed, so it must fail here rather than at apply time.
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

    contract = dict(driver_contract or {})
    if not contract.get("cases"):
        return RewriteDecision(
            False,
            "driver_unavailable",
            "no rewrite driver contract for this operator family",
        )
    try:
        contract_version = int(contract.get("contract_version") or 0)
    except (TypeError, ValueError):
        contract_version = 0
    if contract_version != DRIVER_CONTRACT_VERSION:
        return RewriteDecision(
            False,
            "driver_contract_unsupported",
            f"driver contract version {contract_version} is not {DRIVER_CONTRACT_VERSION}",
        )

    probe = capability_probe or probe_capabilities
    capabilities = probe(forge_root=forge_root)
    if not capabilities.supported:
        return RewriteDecision(False, capabilities.reason, capabilities.detail, capabilities=capabilities)
    if canonical_framework not in capabilities.frameworks:
        return RewriteDecision(
            False,
            "capability_framework_unsupported",
            f"producer frameworks {list(capabilities.frameworks)} exclude {canonical_framework}",
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
        operator_family=str(contract.get("operator_family") or ""),
        driver="",
        branch=branch,
        attempt_id=attempt_id,
    )
    return RewriteDecision(True, "eligible", "", spec=spec, capabilities=capabilities)


__all__ = [
    "APPLYBACK_RESERVE_SEC",
    "ARTIFACT_SCHEMA_VERSION",
    "DRIVER_CONTRACT_VERSION",
    "ENV_BUILDER_SYMBOL",
    "ENV_CANDIDATE_KERNEL",
    "ENV_LOGICAL_OP",
    "ENV_SOURCE_KERNEL",
    "MIN_BUDGET_SEC",
    "PROTOCOL_VERSION",
    "RESULT_SENTINEL",
    "REWRITE_COMMAND",
    "REWRITE_DRIVER_TEMPLATE",
    "REWRITE_ENV",
    "SUPPORTED_FRAMEWORKS",
    "SUPPORTED_SOURCE_TYPES",
    "RewriteCandidateSpec",
    "RewriteCapabilities",
    "RewriteDecision",
    "build_rewrite_driver",
    "evaluate_rewrite_route",
    "probe_capabilities",
    "reset_capability_cache",
    "rewrite_enabled",
]
