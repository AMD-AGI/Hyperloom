# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one view of a session that every KERNEL lane is built from.

Each lane used to assemble its own inputs straight from ``SharedState``, the
environment and a private set of disk scans. The same fact then reached three
consumers in three shapes -- the GEMM lane read the running server's
``--quantization`` while the rewrite handoff read a session field that may
predate it -- and a fact nobody had wired reached none of them.

One builder, not one instance. A lane KEEPs, ``current_best`` moves, and the
next lane has to describe the stack it will actually measure against, so each
lane builds its own context immediately before it runs. What is shared is the
code path, which is what made the three answers diverge.

Discovery only. Every field here is a read, so building a context can never
cost a GPU or mutate a workspace. Anything that boots a server, writes a CSV,
re-keys a shape table or commits a baseline is materialization: it stays in the
lane that needs it and reaches the projection as an argument.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import is_secret_shaped_env_name, redact_secret_values
from hyperloom.common.io import atomic_write_text

from .kernel_evidence import (
    resolve_forge_server_log,
    resolve_forge_untuned_csv,
    resolve_fp8_quant_type,
    resolve_fusion_decode_trace,
    resolve_trace_shape_manifest,
)

CONTEXT_FILENAME = "kernel-context.json"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _absolute(value: Any) -> str:
    """Resolve a path for a consumer in another process, or "" when unset."""
    raw = _text(value)
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve(strict=False))


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


@dataclass(frozen=True)
class ArtifactRef:
    """One session artifact: where it is, and whether it is actually there.

    A consumer needs both. "Nobody produced this" and "the producer named a
    path that is gone" call for different next steps, and a bare string cannot
    tell them apart -- the rewrite handoff has always reported the difference
    and every lane can now do the same.
    """

    path: str = ""
    available: bool = False

    @classmethod
    def of(cls, value: Any) -> "ArtifactRef":
        resolved = _absolute(value)
        if not resolved:
            return cls()
        return cls(path=resolved, available=Path(resolved).exists())

    @property
    def usable(self) -> str:
        """The path when it exists, else "" -- what a CLI flag wants."""
        return self.path if self.available else ""


@dataclass(frozen=True)
class WorkloadFacts:
    """What is being served, independent of how."""

    model_name: str = ""
    #: What the operator named, which may be a Hugging Face repo id. Kept
    #: because provenance and durable artifact names must stay stable across
    #: revisions, where a resolved snapshot basename is a commit hash.
    model_path: str = ""
    #: A local directory something can read ``config.json`` from, or "" when
    #: the logical path has no materialized snapshot.
    resolved_model_path: str = ""
    model_class: str = ""
    precision: str = ""
    quant_type: str = ""
    gpu_type: str = ""
    tp: int = 0
    ep: int = 0
    isl: int = 0
    osl: int = 0
    conc: int = 0
    max_model_len: int = 0


@dataclass(frozen=True)
class ServingFacts:
    """How it is being served: the launch surface ``current_best`` was measured on."""

    framework: str = ""
    framework_version: str = ""
    launch_recipe: str = ""
    overlay_pythonpath: str = ""
    server_args: str = ""
    extra_server_args: str = ""
    #: Secret-shaped names dropped and values redacted at construction, so a
    #: persisted context and an LLM-facing handoff cannot leak a credential.
    extra_envs: Mapping[str, str] = field(default_factory=dict)
    unset_envs: tuple[str, ...] = ()
    #: Every repository a rewrite could name: the configured checkouts plus the
    #: framework packages resolved from where they are imported. Plural because
    #: a session serves more than one -- sglang and aiter at once is ordinary.
    repository_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceIndex:
    """Every artifact this session produced that a lane might read."""

    profile_trace: ArtifactRef = field(default_factory=ArtifactRef)
    #: The profile trace narrowed to a single file. ``profile_trace`` may name a
    #: directory; fusion's discover stage needs one kineto trace.
    decode_trace: ArtifactRef = field(default_factory=ArtifactRef)
    trace_input: ArtifactRef = field(default_factory=ArtifactRef)
    steady_state_trace: ArtifactRef = field(default_factory=ArtifactRef)
    analysis_md: ArtifactRef = field(default_factory=ArtifactRef)
    kernel_candidates: ArtifactRef = field(default_factory=ArtifactRef)
    kernel_source_resolution: ArtifactRef = field(default_factory=ArtifactRef)
    kernel_roofline: ArtifactRef = field(default_factory=ArtifactRef)
    #: Serving log carrying aiter dispatch lines -- the only shape source
    #: grounded in what the model actually ran.
    server_log: ArtifactRef = field(default_factory=ArtifactRef)
    shape_manifest: ArtifactRef = field(default_factory=ArtifactRef)
    untuned_csv: ArtifactRef = field(default_factory=ArtifactRef)
    trace_health_warnings: tuple[Mapping[str, str], ...] = ()


@dataclass(frozen=True)
class KernelContext:
    """One session view, built the same way for every KERNEL lane."""

    session_dir: Path
    macro_cycle: int = 0
    workload: WorkloadFacts = field(default_factory=WorkloadFacts)
    serving: ServingFacts = field(default_factory=ServingFacts)
    evidence: EvidenceIndex = field(default_factory=EvidenceIndex)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_dir"] = str(self.session_dir)
        return payload


def _workload_context(state: Any) -> Mapping[str, Any]:
    context = state.current_profile_workload_context()
    return context if isinstance(context, Mapping) else {}


def _pick(overrides: Mapping[str, Any], context: Mapping[str, Any], state: Any, key: str) -> Any:
    """Resolve one fact: explicit request, then live profile, then session.

    Absent stays absent. A lane that wants ``tp=1`` rather than "unknown"
    applies that itself, so the handoff can still report what the session never
    stated instead of a default it invented.
    """
    for source in (overrides, context):
        value = source.get(key)
        if value not in (None, "", 0):
            return value
    return getattr(state, key, None)


def _normalize_precision(value: Any) -> str:
    return str(value or "").strip().lower()


def _fp8_quant_type(state: Any, payload: Mapping[str, Any], framework: str) -> str:
    """Resolve the fp8 GEMM path the checkpoint actually runs."""
    model_path = _text(payload.get("model_path") or getattr(state, "model_path", ""))
    gpu_type = _text(payload.get("gpu_type") or getattr(state, "gpu_type", ""))
    return resolve_fp8_quant_type(model_path, gpu_type, framework)


def _declared_quant_type(state: Any, payload: Mapping[str, Any], precision: str, framework: str) -> str:
    """The payload's ``quant_type``, with fp8's ``auto`` resolved against the checkpoint."""
    quant_type = _text(payload.get("quant_type")) or "auto"
    if precision == "fp8" and quant_type.lower() == "auto":
        return _fp8_quant_type(state, payload, framework)
    return quant_type


def resolve_precision_and_quant(state: Any, payload: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve the precision the runtime is serving at, and its GEMM quant path.

    Priority: an explicit request, then ``--quantization`` on the server args
    ``current_best`` was measured with, then the session field -- which may
    predate the runtime and is therefore last.
    """
    from .roofline_ceiling import _parse_server_arg, resolve_runtime_workload

    framework = _text(payload.get("framework") or getattr(state, "framework", "")).lower()

    if payload.get("precision"):
        precision = _normalize_precision(payload["precision"])
        return precision, _declared_quant_type(state, payload, precision, framework)

    current_best = getattr(state, "current_best", None)
    current_best = current_best if isinstance(current_best, Mapping) else {}
    try:
        server_args = resolve_runtime_workload(state, arm="current_best").server_args
    except Exception:  # noqa: BLE001 - best-effort fallback for partial state/test doubles
        server_args = _text(current_best.get("extra_server_args"))
    quantization_arg = _parse_server_arg(server_args, "--quantization").lower()

    if quantization_arg == "fp8":
        # An explicit per-token env wins over the checkpoint's static format:
        # it is the path the server was told to take.
        ref_envs = getattr(state, "reference_envs", None) or {}
        per_token = is_truthy(
            (current_best.get("extra_envs") or {}).get("SGLANG_USE_AITER_FP8_PER_TOKEN")
        ) or is_truthy(ref_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN"))
        return "fp8", "per_token" if per_token else _fp8_quant_type(state, payload, framework)

    if quantization_arg in ("fp4", "mxfp4"):
        return quantization_arg, "fp4"

    precision = _normalize_precision(getattr(state, "precision", "")) or "bf16"
    return precision, _declared_quant_type(state, payload, precision, framework)


def build_workload_facts(state: Any, *, overrides: Mapping[str, Any] | None = None) -> WorkloadFacts:
    """Resolve what is being served, before any lane-specific interpretation."""
    from hyperloom.common.model_paths import resolve_serving_model_path
    from hyperloom.inference_optimizer.model_config_utils import resolve_local_model_dir

    incoming = dict(overrides or {})
    context = _workload_context(state)
    model_path = _text(_pick(incoming, context, state, "model_path"))
    precision, quant_type = resolve_precision_and_quant(state, incoming)
    # Bootstrap already walked HL_MODEL_BASE and the hub cache to decide what to
    # serve; probing only the hub cache here would reject a repo id the running
    # server resolved fine.
    resolved = resolve_local_model_dir(resolve_serving_model_path(model_path) or model_path) if model_path else None
    return WorkloadFacts(
        model_name=_text(getattr(state, "model_name", "")),
        model_path=model_path,
        resolved_model_path=str(resolved) if resolved is not None else "",
        model_class=_text(getattr(state, "model_class", "")),
        precision=precision,
        quant_type=quant_type,
        gpu_type=_text(_pick(incoming, context, state, "gpu_type")).lower(),
        tp=_positive_int(_pick(incoming, context, state, "tp")),
        ep=_positive_int(getattr(state, "ep", 0)),
        isl=_positive_int(_pick(incoming, context, state, "isl")),
        osl=_positive_int(_pick(incoming, context, state, "osl")),
        conc=_positive_int(_pick(incoming, context, state, "conc")),
        max_model_len=_positive_int(_pick(incoming, context, state, "max_model_len")),
    )


def _redacted_envs(context: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (context.get("extra_envs"), config.get("extra_envs")):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            name = _text(key)
            if not name or is_secret_shaped_env_name(name):
                continue
            merged[name] = redact_secret_values(str(value))
    return dict(sorted(merged.items()))


def build_serving_facts(
    state: Any,
    *,
    env_spec: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ServingFacts:
    """Resolve the launch surface ``current_best`` was measured on."""
    from .campaign_baseline import campaign_repositories

    incoming = dict(overrides or {})
    context = _workload_context(state)
    spec = dict(env_spec or {})
    config = spec.get("config") if isinstance(spec.get("config"), Mapping) else {}
    current_best = getattr(state, "current_best", None)
    current_best = current_best if isinstance(current_best, Mapping) else {}
    serving_config = context.get("serving_config")
    serving_config = serving_config if isinstance(serving_config, Mapping) else {}
    unset = context.get("unset_envs")
    return ServingFacts(
        framework=_text(_pick(incoming, context, state, "framework")).lower(),
        framework_version=_text(getattr(state, "framework_version", "")),
        launch_recipe=_absolute(spec.get("launch_recipe") or getattr(state, "baseline_config_path", "")),
        overlay_pythonpath=_absolute(spec.get("overlay_pythonpath")),
        server_args=redact_secret_values(_text(config.get("server_launch_flags") or context.get("server_args"))),
        extra_server_args=redact_secret_values(
            _text(
                config.get("extra_server_args")
                or current_best.get("extra_server_args")
                or serving_config.get("extra_server_args")
            )
        ),
        extra_envs=_redacted_envs(context, config),
        unset_envs=tuple(_text(value) for value in unset if _text(value)) if isinstance(unset, list) else (),
        repository_roots=tuple(str(root) for root in campaign_repositories(state)),
    )


def build_evidence_index(
    state: Any,
    session_dir: Path,
    *,
    workload: WorkloadFacts,
    overrides: Mapping[str, Any] | None = None,
) -> EvidenceIndex:
    """Index every artifact this session produced that a lane might read."""
    incoming = dict(overrides or {})
    analysis = getattr(state, "last_trace_analyze", None)
    analysis = analysis if isinstance(analysis, Mapping) else {}
    candidates = _absolute(analysis.get("candidates_path"))
    warnings = analysis.get("trace_health_warnings")
    return EvidenceIndex(
        profile_trace=ArtifactRef.of(getattr(state, "last_profile_trace", "")),
        decode_trace=ArtifactRef.of(resolve_fusion_decode_trace(state, incoming)),
        trace_input=ArtifactRef.of(analysis.get("trace_input")),
        steady_state_trace=ArtifactRef.of(analysis.get("steady_state_trace")),
        analysis_md=ArtifactRef.of(analysis.get("analysis_md_path")),
        kernel_candidates=ArtifactRef.of(candidates),
        kernel_source_resolution=ArtifactRef.of(
            str(Path(candidates).parent / "kernel_source_resolution.json") if candidates else ""
        ),
        kernel_roofline=ArtifactRef.of(analysis.get("kernel_roofline_path")),
        server_log=ArtifactRef.of(resolve_forge_server_log(state, session_dir)),
        shape_manifest=ArtifactRef.of(resolve_trace_shape_manifest(state, session_dir)),
        untuned_csv=ArtifactRef.of(
            resolve_forge_untuned_csv(
                session_dir,
                workload.precision,
                workload.quant_type,
                workload.resolved_model_path,
            )
        ),
        trace_health_warnings=tuple(warning for warning in warnings if isinstance(warning, Mapping))
        if isinstance(warnings, list)
        else (),
    )


def build_kernel_context(
    state: Any,
    session_dir: Path,
    *,
    env_spec: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> KernelContext:
    """Collect the session view a KERNEL lane is about to be handed.

    ``overrides`` is the request payload. It participates here rather than only
    at projection time because it can redirect discovery itself: a payload that
    names a different precision must be answered with that precision's untuned
    CSV, not the session's.

    Blocking IO: walks the session tree and byte-scans serving logs. Callers on
    the orchestrator reactor must run this in a thread.
    """
    workload = build_workload_facts(state, overrides=overrides)
    return KernelContext(
        session_dir=Path(session_dir),
        macro_cycle=_positive_int(getattr(state, "macro_cycle", 0)),
        workload=workload,
        serving=build_serving_facts(state, env_spec=env_spec, overrides=overrides),
        evidence=build_evidence_index(state, Path(session_dir), workload=workload, overrides=overrides),
    )


def write_kernel_context(context: KernelContext, directory: Path) -> Path:
    """Persist the context beside the run it fed, for audit and triage."""
    path = Path(directory) / CONTEXT_FILENAME
    atomic_write_text(
        path,
        json.dumps(context.to_dict(), indent=2, sort_keys=True) + "\n",
        make_parents=True,
    )
    return path


__all__ = [
    "CONTEXT_FILENAME",
    "ArtifactRef",
    "EvidenceIndex",
    "KernelContext",
    "ServingFacts",
    "WorkloadFacts",
    "build_evidence_index",
    "build_kernel_context",
    "build_serving_facts",
    "build_workload_facts",
    "resolve_precision_and_quant",
    "write_kernel_context",
]
