# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the Markdown handoff from Hyperloom to KernelForge."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import is_credential_name, redact_secret_values
from hyperloom_common.io import WriteOptions, atomic_write_text
from hyperloom.common.kernel_source_contract import SOURCE_RESOLUTION_FILENAME
from hyperloom.inference_optimizer.roofline_snapshot import extract_workload_summary
from hyperloom.orchestrator.kernel.campaign_baseline import campaign_repositories
from hyperloom.orchestrator.kernel.gemm_shape_coverage import parse_aiter_shape_lookups, traced_gemm_shapes
from hyperloom.orchestrator.measurement.runtime_findings import (
    AITER_TUNED_MISS,
    load_runtime_findings,
    render_runtime_findings,
)
from hyperloom.orchestrator.prompts.agentx_context import corpus_lines
from kernelforge.workflows.rewrite_controller import (
    SERVING_CONTEXT_FILENAME,
    TRACE_EVIDENCE_FILENAME,
    WORKLOAD_FILENAME,
)

# Session-scoped, not cycle-scoped: the verdicts that matter are the ones a later macro cycle must not repeat.
E2E_REJECTIONS_FILENAME = "e2e-rejections.json"


def _display(value: Any) -> str:
    text = str(value if value not in (None, "") else "not available")
    return redact_secret_values(text)


def _absolute_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve(strict=False))


def _workload_context(state: Any) -> dict[str, Any]:
    context = state.current_profile_workload_context()
    return dict(context) if isinstance(context, Mapping) else {}


def build_workload_md(state: Any) -> str:
    """Render the active workload without deriving optimization candidates.

    An agentic replay's request shape comes from its corpus, so its measured distribution replaces the ISL/OSL in
    state, which are placeholders there.
    """
    context = _workload_context(state)
    corpus = corpus_lines(getattr(state, "agentx_corpus_shape", None))
    lengths = (
        ()
        if corpus
        else (
            ("Input sequence length", context.get("isl") or getattr(state, "isl", 0)),
            ("Output sequence length", context.get("osl") or getattr(state, "osl", 0)),
        )
    )
    fields = (
        ("Model name", getattr(state, "model_name", "")),
        ("Model path", context.get("model_path") or getattr(state, "model_path", "")),
        ("Model class", getattr(state, "model_class", "")),
        ("Precision", context.get("precision") or getattr(state, "precision", "")),
        ("Tensor parallelism", context.get("tp") or getattr(state, "tp", 0)),
        ("Expert parallelism", getattr(state, "ep", 0)),
        *lengths,
        ("Concurrency", context.get("conc") or getattr(state, "conc", 0)),
        ("Maximum model length", context.get("max_model_len") or getattr(state, "max_model_len", 0)),
    )
    lines = ["# Workload", ""]
    lines.extend(f"- **{label}:** `{_display(value)}`" for label, value in fields)
    if corpus:
        lines += ["", *corpus]
    return "\n".join(lines) + "\n"


def _environment_overrides(
    context: Mapping[str, Any],
    env_spec: Mapping[str, Any],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (
        context.get("extra_envs"),
        (env_spec.get("config") or {}).get("extra_envs") if isinstance(env_spec.get("config"), Mapping) else None,
    ):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            name = str(key or "").strip()
            if not name or is_credential_name(name):
                continue
            merged[name] = redact_secret_values(str(value))
    return dict(sorted(merged.items()))


def _fenced_section(heading: str, *body: str) -> list[str]:
    """A Markdown section, opened by a blank line, whose content is one fenced text block."""
    return ["", heading, "", "```text", *([line for line in body if line] or ["not available"]), "```"]


def _bullet_section(heading: str, *body: str) -> list[str]:
    """A Markdown section, opened by a blank line, whose content is a bullet list."""
    return ["", heading, "", *([f"- {line}" for line in body if line] or ["- not available"])]


def _rejections_path(session_dir: str | Path) -> Path:
    """Where one session keeps the operators its end-to-end gate already turned down."""
    return Path(session_dir) / "kernel-agent" / E2E_REJECTIONS_FILENAME


def load_e2e_rejections(session_dir: str | Path) -> list[dict[str, Any]]:
    """Read the session's end-to-end rejection ledger; an absent or unreadable ledger is empty."""
    try:
        rows = json.loads(_rejections_path(session_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def record_e2e_rejections(
    session_dir: str | Path,
    integration: Mapping[str, Any] | None,
    *,
    macro_cycle: int = 0,
) -> int:
    """Append every patch the end-to-end gate reverted to the session's rejection ledger.

    A forge patch can clear the in-session speed gate and still lose end to end -- the
    microbenchmark measures one operator, the gate measures the served workload. The
    Controller re-analyses from the sealed baseline every macro cycle and carries no
    memory of that verdict, so without this ledger the next cycle re-publishes the same
    rewrite. Returns the number of rows written.
    """
    rows = [
        row
        for row in (integration or {}).get("results") or ()
        if isinstance(row, dict) and str(row.get("status") or "").startswith("reverted_")
    ]
    if not rows:
        return 0
    ledger = {
        str(row.get("operator_id")): row
        for row in load_e2e_rejections(session_dir)
        if str(row.get("operator_id") or "")
    }
    for row in rows:
        operator_id = str(row.get("operator_id") or "")
        if not operator_id:
            continue
        ledger[operator_id] = {
            "operator_id": operator_id,
            "macro_cycle": int(macro_cycle),
            "status": str(row.get("status") or ""),
            "reason": str(row.get("reason") or ""),
            "e2e_gain_pct": row.get("gain_pct"),
            "e2e_tput": row.get("new_tput"),
            "best_commit": str(row.get("best_commit") or ""),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    path = _rejections_path(session_dir)
    atomic_write_text(
        path,
        json.dumps(list(ledger.values()), indent=2, sort_keys=True) + "\n",
        options=WriteOptions(make_parents=True, fsync=True, fsync_dir=True, mode=0o600),
    )
    return len(rows)


def _rejection_section(rejections: Sequence[Mapping[str, Any]]) -> list[str]:
    """The handoff section that keeps a reverted rewrite from being published again."""
    if not rejections:
        return []
    ordered = sorted(
        rejections,
        key=lambda row: abs(row.get("e2e_gain_pct") if isinstance(row.get("e2e_gain_pct"), (int, float)) else 0.0),
        reverse=True,
    )
    bullets = []
    for row in ordered:
        gain = row.get("e2e_gain_pct")
        tput = row.get("e2e_tput")
        status = _display(row.get("status"))
        if isinstance(gain, (int, float)):
            tput_text = f" ({float(tput):.1f} tok/s)" if isinstance(tput, (int, float)) else ""
            verdict = f"end-to-end {float(gain):+.2f}%{tput_text}, {status}"
        else:
            # No end-to-end number: the operator was excluded before integration, so its reason is
            # the whole verdict and dropping it would leave the exclusion unjustified.
            reason = _display(row.get("reason"))
            verdict = f"excluded before integration, {status}" + (f" — {reason}" if reason else "")
        bullets.append(
            f"`{_display(row.get('operator_id'))}` — macro cycle {row.get('macro_cycle') or '?'}: {verdict}"
        )
    return [
        *_bullet_section("## Operators Already Reverted By End-To-End Validation", *bullets),
        "",
        "Each operator above was excluded on a measured number, not a guess: either its patch",
        "cleared the in-session speed gate and still made the full served workload slower, so its",
        "microbenchmark gain did not survive the real batch, or its benchmark cases are so",
        "launch-bound that no kernel-body rewrite can clear the gate at all. Do not re-publish",
        "these operators: prefer an operator that is not listed. If one of them is the only viable",
        "candidate, attack a different bottleneck and state in the plan what evidence contradicts",
        "the verdict recorded here.",
    ]


def build_serving_context_md(
    state: Any,
    env_spec: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None,
    rejections: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Render the framework, serving arguments, and environment overrides."""
    context = _workload_context(state)
    spec = env_spec or {}
    config = spec.get("config") or {}
    resolved_args = str(config.get("server_launch_flags") or context.get("server_args") or "").strip()
    # The launch config, then the promoted best, then the profile's own serving block: first one that names them wins.
    extra_sources = (config, getattr(state, "current_best", None) or {}, context.get("serving_config") or {})
    extra_args = str(next((s["extra_server_args"] for s in extra_sources if s.get("extra_server_args")), "")).strip()
    envs = _environment_overrides(context, spec)
    # Taken from the seal that just ran rather than read back from Git: the
    # commit it returned is the base every rewrite of that repository diffs
    # from, and re-reading HEAD would both cost a subprocess per repository
    # and leave a window in which the two could disagree.
    sealed = {str(repo): getattr(baseline, "commit", "") for repo, baseline in (baselines or {}).items()}
    source_roots = tuple(sorted(sealed)) or tuple(map(str, campaign_repositories(state)))

    lines = [
        "# Serving Context",
        "",
        f"- **Framework:** `{_display(context.get('framework') or getattr(state, 'framework', ''))}`",
        f"- **Framework version:** `{_display(getattr(state, 'framework_version', ''))}`",
        f"- **Launch record:** `{_display(_absolute_path(spec.get('launch_json')))}`",
        f"- **Overlay Python path:** `{_display(_absolute_path(spec.get('overlay_pythonpath')))}`",
        # Reported with the commit each one currently sits at, which the caller
        # sealed just before this was written. That object id is the base every
        # rewrite of this repository diffs from, so a task naming a path Git does
        # not carry there is answerable from this document alone.
        *_bullet_section(
            "## Source Repositories",
            *(f"`{root}` @ `{sealed.get(root) or 'unknown'}`" for root in source_roots),
        ),
        *_fenced_section("## Resolved Server Arguments", redact_secret_values(resolved_args)),
        *_fenced_section("## Additional Server Arguments", redact_secret_values(extra_args)),
        *_fenced_section(
            "## Environment Variable Overrides",
            *(f"{key}={value}" for key, value in envs.items()),
        ),
        *_bullet_section(
            "## Unset Environment Variables",
            *(f"`{_display(value)}`" for value in context.get("unset_envs") or ()),
        ),
        *_rejection_section(rejections),
    ]
    return "\n".join(lines) + "\n"


def _evidence_line(label: str, value: Any) -> str:
    path_text = _absolute_path(value)
    if not path_text:
        return f"- **{label}:** not provided"
    status = "available" if Path(path_text).exists() else "missing"
    return f"- **{label}:** `{path_text}` ({status})"


def build_trace_evidence_md(state: Any) -> str:
    """Render absolute paths to existing trace and TraceLens artifacts."""
    analysis = getattr(state, "last_trace_analyze", None)
    analysis = analysis if isinstance(analysis, Mapping) else {}
    candidates_path = _absolute_path(analysis.get("candidates_path"))
    source_resolution = str(Path(candidates_path).parent / SOURCE_RESOLUTION_FILENAME) if candidates_path else ""
    evidence = (
        ("Profile raw trace", getattr(state, "last_profile_trace", "")),
        ("TraceLens input trace", analysis.get("trace_input")),
        ("TraceLens steady-state trace", analysis.get("steady_state_trace")),
        ("TraceLens analysis", analysis.get("analysis_md_path")),
        ("Kernel candidates", candidates_path),
        ("Kernel source resolution", source_resolution),
        ("Kernel roofline", analysis.get("kernel_roofline_path")),
    )
    lines = ["# Trace Evidence", ""]
    lines.extend(_evidence_line(label, value) for label, value in evidence)

    warnings = analysis.get("trace_health_warnings")
    lines.extend(["", "## Trace Health Warnings", ""])
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            if isinstance(warning, Mapping):
                code = str(warning.get("code") or "warning")
                message = str(warning.get("message") or warning.get("detail") or "")
                lines.append(f"- **{_display(code)}:** {_display(message)}")
            else:
                lines.append(f"- {_display(warning)}")
    else:
        lines.append("- none")

    lines.extend(
        ["", "## Exposed Memcpy", "", "```text", _exposed_memcpy_line(analysis.get("analysis_md_path")), "```"]
    )
    lines.extend(["", "## Runtime Findings", ""])
    measurement = getattr(state, "current_best_measurement", None)
    lines.extend(["```text", render_runtime_findings(measurement) if measurement else "not available", "```"])
    lines.extend(["", "## Hot GEMMs Missing Tuned Config", ""])
    profiled = {"launch_evidence_path": str(getattr(state, "last_profile_launch_evidence_path", "") or "")}
    lines.extend(["```text", *_tuned_miss_lines(candidates_path, profiled), "```"])
    return "\n".join(lines) + "\n"


def _exposed_memcpy_line(analysis_md_path: Any) -> str:
    """Exposed memcpy share read beside analysis.md, which omits it."""
    path_text = _absolute_path(analysis_md_path)
    memcpy_pct = extract_workload_summary(path_text)["memcpy_pct"] if path_text else None
    return "not available" if memcpy_pct is None else f"exposed_memcpy={memcpy_pct}% of GPU time"


def _tuned_miss_lines(candidates_path: str, profiled: Mapping[str, Any]) -> list[str]:
    """Name the candidate GEMM rows whose traced shapes the profiled run's server log reported as untuned.

    Prefill M depends on batching, so only the run that produced the trace can match its shapes.
    """
    findings = load_runtime_findings(profiled)
    entry = next((f for f in (findings or {}).get("findings", []) if f["rule_id"] == AITER_TUNED_MISS), None)
    if entry is None or not candidates_path or not Path(candidates_path).is_file():
        return ["not available"]
    if entry["status"] == "unknown":
        return [f"unknown: {entry['reason']}"]
    if entry["status"] == "not_detected":
        return ["none: the profiled server log reports no tuned-config miss"]
    missed, _hit = parse_aiter_shape_lookups(Path(findings["log_path"]).read_text(encoding="utf-8", errors="replace"))
    rows = json.loads(Path(candidates_path).read_text(encoding="utf-8")).get("hot_kernels", [])
    lines = []
    for row in rows:
        shapes = sorted({shape for shape, _calls in traced_gemm_shapes(row) if shape in missed})
        if shapes:
            dims = ", ".join(f"M={m} N={n} K={k}" for m, n, k in shapes)
            lines.append(
                f"- {_display(row.get('kernel_id'))} {_display(row.get('name'))} gpu_pct={row.get('gpu_pct')}: {dims}"
            )
    if not lines:
        lines.append(f"- no hot GEMM shape matches the {len(missed)} missed shape(s)")
    lines.append("Rows not listed: unknown; a tuned-config hit is only logged under AITER_LOG_TUNED_CONFIG.")
    return lines


def write_forge_handoff(
    state: Any,
    *,
    handoff_dir: Path,
    env_spec: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None,
    rejections: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Atomically write one Forge handoff into ``handoff_dir`` and return it."""
    handoff_dir = Path(handoff_dir)
    documents = {
        WORKLOAD_FILENAME: build_workload_md(state),
        SERVING_CONTEXT_FILENAME: build_serving_context_md(state, env_spec, baselines, rejections),
        TRACE_EVIDENCE_FILENAME: build_trace_evidence_md(state),
    }
    for filename, text in documents.items():
        atomic_write_text(
            handoff_dir / filename,
            text,
            options=WriteOptions(make_parents=True, fsync=True, fsync_dir=True, mode=0o600),
        )
    return handoff_dir


__all__ = [
    "build_serving_context_md",
    "build_trace_evidence_md",
    "build_workload_md",
    "load_e2e_rejections",
    "record_e2e_rejections",
    "write_forge_handoff",
]
