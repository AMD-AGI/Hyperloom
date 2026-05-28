#!/usr/bin/env python3
"""Run TraceLens analysis-orchestrator skill through Claude SDK.

This is the LLM-backed path for issue #124. It deliberately lives outside
``tracelens_analysis.py`` so the deterministic CLI/csv fallback remains
isolated and easy to test.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Task"]


# Upstream-defined kernel category enum (TraceLens-internal v0.3,
# see TraceLens/Agent/Analysis/utils/orchestrator_prepare.py CATEGORY_SKILL_MAP).
# Hyperloom's GEAK pipeline expects a small fixed set of category labels;
# this map keeps the upstream → GEAK translation in one place so a future
# upstream addition (e.g. a new analyzer skill) is trivially extended.
UPSTREAM_CATEGORY_TO_GEAK: dict[str, str] = {
    "cpu_idle": "Other",
    "gemm": "GEMM",
    "groupedgemm_fwd": "GEMM",
    "groupedgemm_bwd": "GEMM",
    "moe_fused": "MoE",
    "moe_unfused": "MoE",
    "sdpa_fwd": "SDPA",
    "sdpa_bwd": "SDPA",
    "inferenceattention": "SDPA",
    "elementwise": "Elementwise",
    "reduce": "Reduction",
    "triton": "Triton",
    "norm": "LayerNorm",
    "norm_fwd": "LayerNorm",
    "norm_bwd": "LayerNorm",
    "rmsnorm": "LayerNorm",
    "convolution": "Convolution",
    "conv_fwd": "Convolution",
    "conv_bwd": "Convolution",
    "other": "Other",
}


def normalize_upstream_category(raw: str) -> str:
    """Normalize a TraceLens category string to a GEAK-facing label."""

    if not raw:
        return "unknown"
    key = raw.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return UPSTREAM_CATEGORY_TO_GEAK.get(key, raw)


@dataclass
class TraceLensSkillRunResult:
    """Artifacts produced by one TraceLens skill run.

    Per ``TraceLens_Report_Interfacing.docx`` §2, ``analysis.md`` is the
    single source of truth. Intermediate JSON sidecars are intentionally not
    surfaced as Hyperloom inputs.
    """

    output_dir: Path
    report_path: Path
    artifact_paths: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


def shell_quote(path: Path | str) -> str:
    return shlex.quote(str(path))


def write_local_cmd_prefix(output_dir: Path, tracelens_root: Path) -> Path:
    """Create the command-prefix cache expected by the TraceLens skill."""

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = cache_dir / "cmd_prefix.txt"
    prefix_path.write_text(
        f"cd {shell_quote(tracelens_root)} && {{CMD}}\n",
        encoding="utf-8",
    )
    return prefix_path


def infer_analysis_mode(framework: str, requested: str) -> str:
    requested = (requested or "").strip().lower()
    if requested and requested != "default":
        return requested
    # atom_gap2.md B3 fix: atom traces come out of the same torch
    # profiler API and the same chrome-trace JSON shape as
    # sglang/vllm, so the inference-mode kernel grouping should
    # apply. Pre-fix, atom fell through to "default" and got generic
    # torch grouping that obscured per-iteration boundaries.
    if (framework or "").strip().lower() in {"vllm", "sglang", "atom"}:
        return "inference"
    return requested or "default"


def discover_capture_folder(trace_input: Path, trace_files: list[Path]) -> Path | None:
    """Find a graph-capture folder near a Magpie torch_trace input."""

    candidates: list[Path] = []
    if trace_input.is_dir():
        candidates.extend([
            trace_input / "capture_traces",
            trace_input / "graph_capture",
        ])
    for trace_file in trace_files[:1]:
        candidates.extend([
            trace_file.parent / "capture_traces",
            trace_file.parent.parent / "capture_traces",
        ])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def build_orchestrator_prompt(
    *,
    skill_path: Path,
    trace_path: Path,
    output_dir: Path,
    tracelens_root: Path,
    platform: str,
    framework: str,
    analysis_mode: str,
    capture_folder: Path | None,
) -> str:
    """Prompt a Claude SDK agent to execute the TraceLens standalone skill."""

    analysis_mode = infer_analysis_mode(framework, analysis_mode)
    if analysis_mode == "inference" and capture_folder is not None:
        exec_mode = "graph_capture"
    elif analysis_mode == "inference":
        exec_mode = "eager"
    else:
        exec_mode = "default"

    capture_text = str(capture_folder) if capture_folder else "N/A"
    return f"""You are running TraceLens standalone analysis for Hyperloom.

Read and follow the FULL instructions in this skill file:
{skill_path}

All required Step 0 inputs are already provided below. Do not ask the user any
questions; proceed with the analysis.

Execution context:
- Environment: local
- TraceLens project root: {tracelens_root}
- Command prefix cache: {output_dir / "cache" / "cmd_prefix.txt"}
- Trace file path: {trace_path}
- Output directory: {output_dir}
- Platform: {platform}
- Framework: {framework or "unknown"}
- Analysis mode: {analysis_mode}
- Inference execution mode: {exec_mode}
- Capture folder path: {capture_text}

Important requirements:
1. Use the provided command prefix cache for all shell commands.
2. Run the analysis-orchestrator workflow through Step 11.
3. If analysis_mode is inference and execution mode is graph_capture, pass the
   capture folder to the inference perf-report CLI exactly as the skill says.
4. Write all TraceLens outputs under the output directory above.
5. Ensure this file exists before you finish:
   - {output_dir / "analysis.md"}  (TraceLens v0.3 final report; REQUIRED)
6. Do not run GEAK, OOB kernel optimization, or modify model/framework source.

When complete, respond with a short summary of the artifacts you wrote.
"""


def _import_sdk() -> tuple[Any, Any]:
    try:
        import claude_agent_sdk as sdk  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via caller fallback
        raise RuntimeError(
            "claude_agent_sdk not installed; run inference_optimizer/kernel-agent install first"
        ) from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions


def _iter_message_text(message: Any) -> Iterable[str]:
    for block in list(getattr(message, "content", None) or []):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            yield text
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            yield block["text"]
    result_text = getattr(message, "result", None)
    if isinstance(result_text, str) and result_text:
        yield result_text


async def run_tracelens_skill(
    *,
    skill_path: Path,
    trace_path: Path,
    output_dir: Path,
    tracelens_root: Path,
    platform: str,
    framework: str,
    analysis_mode: str,
    capture_folder: Path | None,
    budget_minutes: float,
    model: str | None = None,
    sdk_query_factory: Callable[..., Any] | None = None,
    sdk_options_cls: Any | None = None,
    log: Callable[[str], None] | None = None,
) -> TraceLensSkillRunResult:
    """Execute the standalone TraceLens skill with Claude SDK."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = write_local_cmd_prefix(output_dir, tracelens_root)
    prompt = build_orchestrator_prompt(
        skill_path=skill_path,
        trace_path=trace_path,
        output_dir=output_dir,
        tracelens_root=tracelens_root,
        platform=platform,
        framework=framework,
        analysis_mode=analysis_mode,
        capture_folder=capture_folder,
    )

    if sdk_query_factory is None or sdk_options_cls is None:
        query, options_cls = _import_sdk()
        sdk_query_factory = sdk_query_factory or query
        sdk_options_cls = sdk_options_cls or options_cls

    system_prompt = (
        "You are a TraceLens analysis runner inside Hyperloom. Execute only "
        "the requested standalone analysis workflow. Use absolute paths, "
        "write artifacts under the requested output directory, and do not "
        "modify application source code."
    )
    # The standalone TraceLens skill runs deterministic setup plus several
    # subagent-backed analysis steps; keep a fixed, explicit turn budget so
    # smoke and production runs use the same orchestration envelope.
    max_turns = 300
    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "allowed_tools": DEFAULT_ALLOWED_TOOLS,
        "stderr": lambda line: log(f"[claude-sdk] {line.rstrip()}") if log else None,
    }
    if model:
        kwargs["model"] = model
    # Supported by claude-agent-sdk used by the outer role backend; harmless in
    # tests via FakeOptions and keeps Bash relative paths rooted at TraceLens.
    kwargs["cwd"] = str(tracelens_root)

    try:
        options = sdk_options_cls(**kwargs)
    except TypeError:
        # Older SDK builds may not support cwd. The prompt and command-prefix
        # cache still use absolute paths, so retrying without cwd is safe.
        kwargs.pop("cwd", None)
        options = sdk_options_cls(**kwargs)
    chunks: list[str] = []
    sdk_error = ""
    if log:
        log(f"TraceLens SDK runner: prefix cache={prefix_path}")
    try:
        async for message in sdk_query_factory(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001
        # Claude Code may report "max turns reached" after it has already
        # produced the TraceLens artifacts. Treat artifact presence as the
        # source of truth and surface the SDK error as metadata.
        sdk_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {sdk_error}")

    # TraceLens v0.3 ships the final report as ``analysis.md`` per
    # ``TraceLens/Agent/Analysis/utils/templates/analysis_template.md``.
    # We deliberately do NOT accept the v0.2 ``standalone_analysis.md``
    # fallback any longer: in #203 that fallback was found to silently
    # paper over SDK-orchestrator failures by picking up a stale
    # Hyperloom-fabricated bullet list from a prior run. The v0.3 layout
    # has been the contract since #148, so any miss here is a real
    # upstream failure that should surface to the operator.
    #
    # Per ``TraceLens_Report_Interfacing.docx`` §2, ``analysis.md`` is
    # the single source of truth. Do not surface intermediate sidecars as
    # Hyperloom inputs; the caller parses only the final report.
    report_path = output_dir / "analysis.md"
    if not report_path.exists():
        if sdk_error:
            raise RuntimeError(
                f"TraceLens SDK runner failed before writing {report_path}: {sdk_error}"
            )
        raise RuntimeError(f"TraceLens SDK runner did not write {report_path}")

    artifact_paths = {
        "tracelens_agent_report": str(report_path),
        "tracelens_cmd_prefix": str(prefix_path),
    }
    if sdk_error:
        artifact_paths["tracelens_agent_sdk_error"] = sdk_error

    return TraceLensSkillRunResult(
        output_dir=output_dir,
        report_path=report_path,
        raw_text="\n".join(chunks),
        artifact_paths=artifact_paths,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# --- analysis.md parser (TraceLens v0.3 final-report contract) -------------
#
# The reviewer-preferred exit interface for the TraceLens analysis-orchestrator
# is the final ``analysis.md`` report (see PR #155 review by @tsrikris). The
# stable contract for that file is documented in
# ``TraceLens/Agent/Analysis/utils/templates/{analysis_template.md,
# sub_agent_spec.md}`` and consists of:
#
#   1. Per-P-item ``<!-- impact-begin kind=p_item category=<cat> mid=<m> ... -->``
#      markers in the ``## Compute Kernel Optimizations`` section. The
#      ``category=`` attribute is the upstream category enum
#      (``CATEGORY_SKILL_MAP`` keys in orchestrator_prepare.py). P-items are
#      emitted in priority order (rank 1 → N).
#
#   2. Per-P-item ``<!-- reasoning-candidate tier=compute rank=<R> -->`` markers
#      in ``## Detailed Analysis → ### Compute Kernel Insights``, each followed
#      by an ``#### 🔴/🟡/🟢 PN: <title>`` heading and a ``**Data:**`` block
#      that contains the canonical 9-column kernel breakdown table:
#
#        Operation | Args | Kernel Path | Time (ms) | %E2E | Count |
#                  FLOPS/Byte | Efficiency | Bound
#
# This parser is the only place in Hyperloom that reads TraceLens candidate
# data; intermediate files (``priority_data.json``, ``category_data/*.json``)
# are intentionally ignored.
_DATA_TABLE_HEADER_TOKENS = (
    "operation",
    "args",
    "kernel path",
    "time (ms)",
    "%e2e",
    "count",
    "flops/byte",
    "efficiency",
    "bound",
)
_PITEM_MARKER_RE = re.compile(
    r"<!--\s*impact-begin\s+kind=p_item\s+([^>]*?)-->",
    re.IGNORECASE,
)
_REASONING_MARKER_RE = re.compile(
    r"<!--\s*reasoning-candidate\s+tier=(\w+)\s+rank=(\d+)\s*-->",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(
    r"^####\s+(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]+\s+)?P(\d+):\s*(.+?)\s*$",
    re.MULTILINE,
)
_LIBRARY_PARENS_RE = re.compile(r"\(([^()]+)\)\s*$")
_EFFICIENCY_RE = re.compile(
    r"([\d.]+)\s*%\s*of\s*([\d.]+)\s*([A-Za-z/]+)",
    re.IGNORECASE,
)
# TraceLens v0.3 Detailed Analysis blocks include three sibling labels:
#   **Reasoning for Slowdown:** <prose>
#   **Resolution:**             <prose>
#   **Impact estimate:**        Low end ...: <ms> ms savings (<pct>% E2E)
#                               High end ...: <ms> ms savings (<pct>% E2E)
# Extracting these gives GEAK the same hypothesis a human reviewer reads in
# the report; the prose is treated as a *hypothesis to validate*, never as
# imperative guidance — see ``build_prompt`` for the framing.
_IDENTIFICATION_LABEL = "**Identification:**"
_DATA_LABEL = "**Data:**"
_REASONING_LABEL = "**Reasoning for Slowdown:**"
_RESOLUTION_LABEL = "**Resolution:**"
_IMPACT_LABEL = "**Impact estimate:**"
_IMPACT_LOW_RE = re.compile(
    r"Low end[^:\n]*:\s*([0-9.]+)\s*ms savings\s*\(([0-9.]+)%\s*E2E\)",
    re.IGNORECASE,
)
_IMPACT_HIGH_RE = re.compile(
    r"High end[^:\n]*:\s*([0-9.]+)\s*ms savings\s*\(([0-9.]+)%\s*E2E\)",
    re.IGNORECASE,
)


def _parse_marker_attrs(blob: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s>]+)", blob))


def _extract_between(
    text: str, start_marker: str, end_markers: tuple[str, ...],
) -> str:
    """Return the substring between ``start_marker`` and the earliest of
    ``end_markers``. Empty string when ``start_marker`` is absent. When no
    end marker is present, returns the tail of ``text`` after the start
    marker (defensive: TraceLens occasionally truncates the trailing
    section)."""
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end_positions = [text.find(m, start) for m in end_markers]
    end_positions = [pos for pos in end_positions if pos != -1]
    end = min(end_positions) if end_positions else len(text)
    return text[start:end].strip()


def _extract_pitem_prose(body: str) -> dict[str, Any]:
    """Extract Identification / Reasoning / Resolution / Impact-estimate
    fields from a Detailed Analysis P-item body. All fields default to
    empty / 0.0 so the parser stays additive — callers that don't care
    for prose are unaffected.

    The Identification line carries per-rank context that pins the
    P-item back to its source metrics file (e.g.
    ``(source: gemm_metrics.json → operations[].efficiency.efficiency_percent)``);
    surfacing it lets GEAK trace any hypothesis back to the raw
    TraceLens data when it needs to disagree.

    Returns::

        {
          "identification":         str,
          "reasoning_for_slowdown": str,
          "resolution":             str,
          "impact_low_ms":          float,
          "impact_low_e2e_pct":     float,
          "impact_high_ms":         float,
          "impact_high_e2e_pct":    float,
        }
    """
    identification = _extract_between(
        body, _IDENTIFICATION_LABEL,
        (_DATA_LABEL, _REASONING_LABEL, _RESOLUTION_LABEL, _IMPACT_LABEL),
    )
    reasoning = _extract_between(
        body, _REASONING_LABEL, (_RESOLUTION_LABEL, _IMPACT_LABEL),
    )
    resolution = _extract_between(body, _RESOLUTION_LABEL, (_IMPACT_LABEL,))
    low_match = _IMPACT_LOW_RE.search(body)
    high_match = _IMPACT_HIGH_RE.search(body)
    return {
        "identification":         identification,
        "reasoning_for_slowdown": reasoning,
        "resolution":             resolution,
        "impact_low_ms":          _safe_float(low_match.group(1)) if low_match else 0.0,
        "impact_low_e2e_pct":     _safe_float(low_match.group(2)) if low_match else 0.0,
        "impact_high_ms":         _safe_float(high_match.group(1)) if high_match else 0.0,
        "impact_high_e2e_pct":    _safe_float(high_match.group(2)) if high_match else 0.0,
    }


def _extract_pitem_categories(text: str) -> list[dict[str, Any]]:
    """Return per-P-item metadata, in priority order, from p_item markers.

    Each item carries ``category``, ``low``, ``mid``, ``high`` (where present).
    """

    items: list[dict[str, Any]] = []
    for match in _PITEM_MARKER_RE.finditer(text):
        attrs = _parse_marker_attrs(match.group(1))
        if "category" not in attrs:
            continue
        items.append({
            "category": attrs.get("category", ""),
            "impact_score_low": _safe_float(attrs.get("low")),
            "impact_score": _safe_float(attrs.get("mid")),
            "impact_score_high": _safe_float(attrs.get("high")),
        })
    return items


def _split_data_blocks(text: str) -> list[tuple[int, str, str]]:
    """Yield ``(rank, title, body)`` for each compute-tier reasoning block."""

    blocks: list[tuple[int, str, str]] = []
    matches = list(_REASONING_MARKER_RE.finditer(text))
    for idx, match in enumerate(matches):
        tier = match.group(1).lower()
        if tier != "compute":
            continue
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        head_match = _HEADING_RE.search(body)
        if not head_match:
            continue
        rank = int(head_match.group(1))
        title = head_match.group(2).strip()
        blocks.append((rank, title, body))
    return blocks


def _extract_data_table(body: str) -> list[list[str]]:
    """Pull the 9-column markdown table that follows ``**Data:**``.

    Returns the raw cell strings (header + data rows). The table ends at the
    first blank line or at the next bold ``**Field:**`` label.
    """

    marker = body.find("**Data:**")
    if marker < 0:
        return []
    tail = body[marker + len("**Data:**"):]
    rows: list[list[str]] = []
    in_table = False
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_table:
                break
            continue
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        if set(stripped.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        rows.append(cells)
    return rows


def _row_to_candidate(
    headers: list[str],
    cells: list[str],
    *,
    category: str,
    rank: int,
    title: str,
    library: str,
    impact: dict[str, float],
    prose: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if len(cells) != len(headers):
        return None
    record = dict(zip(headers, cells))

    name = record.get("operation", "").strip()
    if not name or name in {"-", "—"}:
        return None
    args = record.get("args", "").replace("<br>", "\n").strip()
    shapes = [s.strip() for s in args.split("\n") if s.strip() and s.strip() not in {"-", "—"}]
    kernel_path = record.get("kernel path", "").strip()
    if kernel_path in {"-", "—"}:
        kernel_path = ""
    # PR: when TraceLens reports a launcher path relative to a known
    # framework package (``aiter/...`` / ``sglang/...`` / ``vllm/...``)
    # — that's what torch.profiler emits after stripping ``sys.path``
    # prefixes — promote it to the absolute on-disk source file so the
    # downstream patchability gate stops emitting ``source not under a
    # reusable framework root`` and GEAK can target real reusable
    # kernels (e.g. ``/sgl-workspace/aiter/aiter/ops/rmsnorm.py``).
    # The verbatim launcher string is preserved on
    # ``tracelens_launcher_path`` below so AST-based source-function
    # aggregation still works.
    resolved_source_file = kernel_path
    if kernel_path:
        resolved = _resolve_launcher_to_abs_source(kernel_path)
        if resolved is not None:
            resolved_source_file, _resolved_line, _resolved_func = resolved
    time_ms = _safe_float(record.get("time (ms)"))
    percent_e2e = _safe_float(record.get("%e2e"))
    count_val = _safe_float(record.get("count"), 1.0)
    flops_per_byte = _safe_float(record.get("flops/byte"))
    bound_raw = record.get("bound", "").strip()
    eff_raw = record.get("efficiency", "").strip()
    eff_match = _EFFICIENCY_RE.search(eff_raw)
    if eff_match:
        eff_pct = _safe_float(eff_match.group(1))
        peak_value = _safe_float(eff_match.group(2))
        peak_unit = eff_match.group(3).strip()
    else:
        eff_pct = _safe_float(eff_raw.rstrip("%")) if eff_raw else 0.0
        peak_value = 0.0
        peak_unit = ""

    candidate: dict[str, Any] = {
        "name": name,
        "duration_us": time_ms * 1000.0,
        "call_count": int(count_val) if count_val else 0,
        "source_file": resolved_source_file,
        # PR-B §1: preserve the raw Kernel Path string verbatim so
        # ``aggregate_by_source_function`` can run AFTER
        # ``_finalize_candidates`` (which overwrites ``source_file``
        # with the locate_source_via_grep result). Without this the
        # launcher string ``<path>(<line>): <fn>`` is lost and AST
        # resolution falls back to ``None``.
        "tracelens_launcher_path": kernel_path,
        "source_type": "tracelens_report",
        "shapes": shapes,
        "tracelens_category": category,
        "tracelens_pitem_rank": rank,
        "tracelens_pitem_title": title,
        "library": library,
        "bound_type": bound_raw,
        "percent_of_total": percent_e2e,
        "flops_per_byte": flops_per_byte,
        "efficiency_percent": eff_pct,
        "efficiency_peak_value": peak_value,
        "efficiency_peak_unit": peak_unit,
        "impact_score": impact.get("impact_score", 0.0),
        "impact_score_low": impact.get("impact_score_low", 0.0),
        "impact_score_high": impact.get("impact_score_high", 0.0),
    }
    if prose:
        # P-item prose is shared across every row in the same Detailed
        # Analysis block; duplicating it onto each candidate keeps the
        # candidate dict self-describing for downstream consumers
        # (build_prompt / source-function aggregation) without forcing
        # them to re-join against a rank-indexed sidecar.
        for key in (
            "identification",
            "reasoning_for_slowdown",
            "resolution",
            "impact_low_ms",
            "impact_low_e2e_pct",
            "impact_high_ms",
            "impact_high_e2e_pct",
        ):
            if key in prose:
                candidate[key] = prose[key]
    return candidate


_IDLE_PCT_TABLE_RE = re.compile(
    r"^\|\s*Idle\s*%\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\|",
    re.IGNORECASE | re.MULTILINE,
)


def extract_idle_pct_from_analysis_md(md_path: Path) -> float | None:
    """Extract ``Idle %`` from the Executive Summary table in ``analysis.md``.

    The TraceLens v0.3 ``analysis_template.md`` always emits an Executive
    Summary table whose rows are ``| Metric | Value |``; the row of
    interest looks exactly like ``| Idle % | 0.25% |``. Per
    ``Report_Interfacing.docx`` §1 (Executive Summary schema) and §2
    (idle-gate sanity check in Possible Approach (Hyperloom v3)), the
    Executive Summary is the workload-level health snapshot that should
    gate any kernel-level optimization recommendation: a high idle
    percentage means the GPU
    spent most of the trace waiting (host stalls, sync, allocator
    contention, …), so per-kernel speedups will not move end-to-end
    latency and the operator should reach for parameter optimization
    (batch size, KV cache shape, prefill/decode split) instead.

    Returns the idle percentage as a float (e.g. ``0.25`` for ``0.25%``),
    or ``None`` when:
      * the file doesn't exist or can't be decoded
      * the Executive Summary table has no ``Idle %`` row (older
        TraceLens templates, or a partial / malformed report)
      * the value is not numerically parseable

    Returning ``None`` rather than raising lets callers downgrade
    gracefully to "skip the idle-gate check" rather than failing the
    whole run on a TraceLens template drift. The runtime gate in
    ``tracelens_analysis.py`` treats ``None`` as "don't warn".
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    match = _IDLE_PCT_TABLE_RE.search(text)
    if not match:
        return None

    raw = match.group(1)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _efficiency_sort_key(candidate: dict[str, Any]) -> float:
    """Per-row sort key for the ``Lower Efficiency`` budget filter.

    ``TraceLens_Report_Interfacing.docx`` §2 Recommended Interfacing
    Approach → Possible Approach (Hyperloom v3):

      > Filter for GEAK based on budget (Higher P-item, Lower Efficiency)

    P-item rank is the outer order, so this key only orders rows *within*
    one P-item. Rows where TraceLens did not report an efficiency value
    (``_row_to_candidate`` defaulted ``efficiency_percent`` to ``0.0``)
    are demoted to last so they don't outrank rows TraceLens actually
    measured. Python's sort is stable, so true-zero / equal-efficiency
    rows preserve TraceLens's original ``Data:`` row order.
    """
    eff = candidate.get("efficiency_percent")
    try:
        value = float(eff)
    except (TypeError, ValueError):
        return float("inf")
    if value <= 0.0:
        return float("inf")
    return value


def parse_analysis_md(md_path: Path, top_k: int = 10) -> list[dict[str, Any]]:
    """Parse the TraceLens v0.3 ``analysis.md`` final report into hot-kernels.

    This is the only place in Hyperloom that reads TraceLens candidate
    data. The returned list follows the priority order required by
    ``TraceLens_Report_Interfacing.docx`` §2 Recommended Interfacing
    Approach ("Filter for GEAK based on budget (Higher P-item,
    Lower Efficiency)"):

    1. **Higher P-item first** — rank=1 rows before rank=2 rows, etc.
    2. **Lower Efficiency first** within the same P-item, so rows with
       more optimization headroom survive the ``top_k`` budget cap.
       Rows with no efficiency value land last (see
       :func:`_efficiency_sort_key`).

    Empty / non-existent reports return an empty list so callers can
    surface that signal upstream (Hyperloom does not fall back to
    intermediate sidecars per docx §2).
    """

    if not md_path.exists():
        return []
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return []

    pitems = _extract_pitem_categories(text)

    blocks = _split_data_blocks(text)
    if not blocks:
        return []

    headers_canonical = [tok.strip().lower() for tok in _DATA_TABLE_HEADER_TOKENS]

    candidates: list[dict[str, Any]] = []
    for rank, title, body in blocks:
        rows = _extract_data_table(body)
        if not rows:
            continue
        header_row = [cell.strip().lower() for cell in rows[0]]
        if header_row != headers_canonical:
            # Tolerate trivial rename / reordering only if the canonical token
            # is a substring of the header cell. Otherwise skip the block —
            # silent wrong-mapping would be worse than a missed candidate.
            normalized: list[str] = []
            for cell in header_row:
                match = next(
                    (canon for canon in headers_canonical if canon in cell),
                    cell,
                )
                normalized.append(match)
            if normalized != headers_canonical:
                continue
            header_row = normalized
        # Pull the matching p_item meta by 1-based rank (P1 → pitems[0]).
        # The orchestrator template guarantees compute-tier P-items are
        # numbered by global rank in the Compute Kernel Optimizations section,
        # so a missing entry is treated as "category unknown".
        pitem_meta = pitems[rank - 1] if rank - 1 < len(pitems) else {}
        category = pitem_meta.get("category", "")
        library_match = _LIBRARY_PARENS_RE.search(title)
        library = library_match.group(1).strip() if library_match else ""
        impact = {
            "impact_score": pitem_meta.get("impact_score", 0.0),
            "impact_score_low": pitem_meta.get("impact_score_low", 0.0),
            "impact_score_high": pitem_meta.get("impact_score_high", 0.0),
        }
        prose = _extract_pitem_prose(body)
        pitem_candidates: list[dict[str, Any]] = []
        for cells in rows[1:]:
            cand = _row_to_candidate(
                header_row,
                cells,
                category=category,
                rank=rank,
                title=title,
                library=library,
                impact=impact,
                prose=prose,
            )
            if cand is None:
                continue
            pitem_candidates.append(cand)
        pitem_candidates.sort(key=_efficiency_sort_key)
        for cand in pitem_candidates:
            candidates.append(cand)
            if len(candidates) >= top_k:
                return candidates
    return candidates


# ---------------------------------------------------------------------------
# PR-B §1: source-function aggregation
# ---------------------------------------------------------------------------
# TraceLens reports each kernel launch site as one Operation row in the
# 9-column Detailed Analysis table. Multiple rows commonly resolve to the
# *same* Python source function (e.g. ``rmsnorm`` called from prefill,
# decode, and capture paths), so dispatching one GEAK task per kernel_id
# burns the LLM budget on the same function 3-5 times.
#
# The feature branch's ``tracelens_geak_task_parser`` solves this by
# parsing the launcher path string (``path.py(76): function_name``),
# resolving the function definition line via Python AST when the file
# exists locally, and grouping every candidate that maps to the same
# ``(source_path, definition_line, function_name)`` triple into a single
# ``task_group``. We fold that logic in here so ``parse_analysis_md``
# remains the single-source-of-truth report consumer, and a sibling
# ``aggregate_by_source_function`` produces the additive ``task_groups[]``
# structure that downstream callers (``kernel_optimization.py``,
# ``kernel_request_handlers._batch_kernel_candidates``) opt into.
#
# The parser is conservative: kernels whose ``kernel_path`` is empty
# (the LLama70B fixture) or malformed simply fall back to the legacy
# per-kernel dispatch. Aggregation never *replaces* the hot_kernels[]
# list — it adds a parallel view.

# TraceLens emits the launcher path as ``<absolute or workspace-relative
# path>(<line>): <function_name>`` for Python frames, and either a bare
# file path or ``<path>#L<line>`` for HIP/.cu rows. Both shapes resolve
# to ``(path, line, function_name | None)``; missing pieces are None.
_LAUNCHER_PATH_RE = re.compile(
    r"(?P<path>.+?)\((?P<line>\d+)\)\s*:\s*(?P<func>[A-Za-z_][A-Za-z0-9_]*)\s*$",
)
# TraceLens emits these for rows whose Kernel Path it cannot resolve
# (Tensile-backed aten ops, vendor closed-source kernels, etc.). They
# must not survive launcher parsing — otherwise aggregation groups
# every placeholder row together under a bogus ``Path("—")``.
_LAUNCHER_PATH_PLACEHOLDERS: frozenset[str] = frozenset({
    "", "-", "—", "–", "n/a", "none", "null", "tbd", "unknown",
})


def _parse_launcher_path(kernel_path: str) -> tuple[str, int | None, str | None]:
    """Parse a TraceLens kernel-path string.

    Returns ``(path, line, function_name)`` where ``line`` and
    ``function_name`` may be ``None``. The two accepted shapes are::

        <path>(<line>): <function_name>      # Python frame, AST-resolvable
        <path>#L<line>                        # generic file ref
        <path>                                # bare file ref

    Anything else falls back to ``(stripped_text, None, None)`` so the
    caller can decide whether to skip the row or treat it as opaque.
    TraceLens placeholders (``-``, ``—``, ``n/a``, etc.) collapse to
    ``("", None, None)`` so source-function aggregation skips them.
    """
    if not kernel_path:
        return "", None, None
    text = kernel_path.strip()
    if text.lower() in _LAUNCHER_PATH_PLACEHOLDERS:
        return "", None, None
    match = _LAUNCHER_PATH_RE.match(text)
    if match:
        return (
            match.group("path").strip(),
            int(match.group("line")),
            match.group("func"),
        )
    path, _, fragment = text.partition("#L")
    if fragment.isdigit():
        return path.strip(), int(fragment), None
    return text, None, None


# ---------------------------------------------------------------------------
# Launcher path → absolute source file resolver.
#
# PyTorch profiler records Python frames with ``__file__`` already made
# relative to ``sys.path`` entries — e.g. ``aiter/ops/rmsnorm.py(62):
# rmsnorm2d_fwd``. TraceLens forwards that string verbatim as the
# ``Kernel Path`` column. Without resolution the downstream patchability
# gate rejects every row with ``source not under a reusable framework
# root`` because the relative segment never matches an absolute
# allowlist prefix, and GEAK never gets to rewrite real reusable
# kernels (e.g. ``/sgl-workspace/aiter/aiter/ops/rmsnorm.py``).
#
# Resolution strategy (most-specific first):
#   1. ``HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` env override, format
#      ``pkg=/abs/parent[,pkg=/abs/parent...]``. Per-package operator
#      escape hatch when the install layout deviates from defaults.
#   2. ``importlib.util.find_spec(pkg)`` walks the live ``sys.path``
#      and returns the absolute origin. Robust to editable installs,
#      wheel installs, and dist-packages layouts alike.
#   3. Hardcoded fallback table for the production image when the
#      package is not yet imported. Keeps the gate working from
#      static-analysis paths where ``import aiter`` might not have run
#      (e.g. CSV-only parses).
#
# Returns the absolute source path when it exists on disk; ``None``
# otherwise so the caller falls back to the original relative string
# and downstream gates emit their normal ``source file not resolved``
# rejection.
_FRAMEWORK_PKG_FALLBACK_ROOTS: dict[str, tuple[str, ...]] = {
    "aiter": ("/sgl-workspace/aiter",),
    "sglang": ("/sgl-workspace/sglang/python", "/sgl-workspace/sglang"),
    "vllm": (
        "/usr/local/lib/python3.12/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/opt/venv/lib/python3.10/site-packages",
        "/sgl-workspace/vllm",
    ),
    # atom_gap2.md B2 fix: kernel-agent's offline source-file
    # resolver walks this fallback table when ``import atom`` hasn't
    # run (CSV-only parses, static-analysis paths). Pre-fix, atom
    # kernels referenced by a TraceLens CSV under ``/app/ATOM/atom/``
    # or ``site-packages/atom/`` fell through to the "could not
    # resolve" branch and the kernel-opt proposal was silently
    # rejected. The roots below mirror the entries Phase 2.5 added
    # to ``inference_optimizer.orchestrator.kernel_request_handlers
    # ._REUSABLE_SOURCE_ROOTS`` and
    # ``kernel-agent/tools/tracelens_analysis._REUSABLE_SOURCE_ROOTS``;
    # the cross-file sync is pinned by
    # ``test_framework_paths_units.py::
    # test_kernel_request_handlers_and_tracelens_analysis_atom_paths_in_sync``.
    # ``/app/ATOM`` (the editable-install parent) is the canonical-
    # case root: when ``atom/model_engine/...`` is the relative path
    # in a TraceLens CSV, joining against this parent recovers the
    # real on-disk file.
    "atom": (
        "/app/ATOM",
        "/usr/local/lib/python3.12/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/opt/venv/lib/python3.10/site-packages",
        "/opt/venv/lib/python3.12/site-packages",
    ),
}
_FRAMEWORK_SOURCE_ROOTS_ENV = "HYPERLOOM_FRAMEWORK_SOURCE_ROOTS"


def _env_framework_source_roots() -> dict[str, tuple[str, ...]]:
    """Parse ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` into ``{pkg: (root,...)}``.

    Format: comma-separated ``pkg=/abs/parent`` entries. Empty / unparseable
    entries are skipped silently so a malformed export doesn't poison
    the whole resolver. Values are not validated for existence here —
    that's the caller's job, so the operator can pre-stage paths.
    """
    raw = os.environ.get(_FRAMEWORK_SOURCE_ROOTS_ENV, "").strip()
    if not raw:
        return {}
    out: dict[str, list[str]] = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        key, sep, value = chunk.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            continue
        out.setdefault(key, []).append(value)
    return {k: tuple(v) for k, v in out.items()}


def _package_root_parent(pkg: str) -> str | None:
    """Return the directory that contains ``pkg/`` on the live ``sys.path``.

    ``importlib.util.find_spec`` is used so editable installs (``aiter``
    at ``/sgl-workspace/aiter/aiter/__init__.py``), wheel installs
    (``vllm`` at ``/usr/local/lib/python3.12/dist-packages/vllm/...``),
    and namespace packages all resolve correctly without hardcoding.
    Returns ``None`` when the package isn't importable in the current
    interpreter.
    """
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        # Regular or namespace package: the first search location is the
        # package directory; its parent is what we want to prepend to
        # ``pkg/rest/of/path``.
        loc = list(spec.submodule_search_locations)[0]
        return os.path.dirname(loc)
    if spec.origin and spec.origin.endswith(".py"):
        # Single-file module (rare for frameworks but kept for safety).
        return os.path.dirname(os.path.dirname(spec.origin))
    return None


def _resolve_launcher_to_abs_source(
    kernel_path: str,
) -> tuple[str, int | None, str | None] | None:
    """Resolve a TraceLens launcher-path string to an absolute file path.

    Returns ``(abs_file, line, function_name)`` when resolution succeeds
    AND the resolved file exists on disk. Returns ``None`` when:

    * the launcher path is empty / a placeholder / has no path component;
    * the path is already absolute (caller can use it verbatim — the
      caller takes care of preserving the original string);
    * no candidate root yields an existing file.

    The resolver is deliberately conservative: a non-existent absolute
    path is treated as a miss so the caller can fall back to the
    grep-based locator in ``tracelens_analysis._finalize_candidates``
    instead of producing a fabricated absolute path that downstream
    consumers would treat as authoritative.
    """
    raw_path, line, func = _parse_launcher_path(kernel_path)
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        # Already absolute — caller can use ``raw_path`` directly; the
        # resolver returns None to signal "no rewrite needed" (preserves
        # the original string verbatim in ``source_file``).
        return None
    # Pick the leading path segment as the candidate package.
    head = raw_path.split("/", 1)[0]
    if not head or head.startswith("."):
        return None

    candidate_roots: list[str] = []
    env_roots = _env_framework_source_roots()
    candidate_roots.extend(env_roots.get(head, ()))
    pkg_parent = _package_root_parent(head)
    if pkg_parent:
        candidate_roots.append(pkg_parent)
    candidate_roots.extend(_FRAMEWORK_PKG_FALLBACK_ROOTS.get(head, ()))

    seen: set[str] = set()
    for root in candidate_roots:
        if not root or root in seen:
            continue
        seen.add(root)
        abs_path = os.path.join(root, raw_path)
        if not os.path.isfile(abs_path):
            continue
        # Validate the resolved path so we never hand GEAK a file whose
        # contents don't actually contain the launcher's function. This
        # catches two real failure modes:
        #
        #   * sys.path shadowing — two installed packages share a leaf
        #     name (e.g. namespace clash from a stale wheel + the
        #     editable checkout) and ``find_spec`` returns the wrong
        #     one. Without this check we'd "resolve" to a real file
        #     that doesn't host the launcher's symbol.
        #   * Operator misconfiguration of
        #     ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` — pointing at a
        #     directory that happens to contain a same-named ``.py``
        #     stub (test fixture, snapshot, doc example) instead of
        #     the live source tree.
        #
        # When the launcher provides no function name (``#L<line>`` /
        # bare-path shapes), or the source is not a ``.py`` file, we
        # fall through to existence-only validation since AST cannot
        # walk it. Verification failures fall through to the next
        # candidate root rather than short-circuiting, so a shadowing
        # spec doesn't block a valid fallback entry.
        if func and abs_path.endswith(".py"):
            if _function_line_from_ast(Path(abs_path), func) is None:
                continue
        return abs_path, line, func
    return None


def _function_line_from_ast(path: Path, function_name: str) -> int | None:
    """Walk ``path`` with :mod:`ast` and return the lineno of the first
    ``FunctionDef`` / ``AsyncFunctionDef`` whose name matches.

    Returns ``None`` when the file is unreadable, doesn't parse, or the
    function is absent — the caller falls back to the launcher's
    reported line. We accept either function flavor because Triton
    kernels may be declared async in user code.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node.lineno
    return None


def _resolve_source_target(
    candidate: dict[str, Any],
    *,
    source_root: Path | None,
) -> dict[str, Any] | None:
    """Resolve a candidate's launcher path to a stable
    ``(source_path, definition_line, function_name)`` triple.

    Returns ``None`` when the launcher path cannot be parsed at all
    (empty / no path component) so the caller falls back to per-kernel
    dispatch. When ``source_root`` is provided, relative paths are
    resolved against it; when the resolved file exists on disk and the
    function name is known, AST resolution overrides the reported line
    number (TraceLens uses the call site's line, not the
    ``def`` site).
    """
    # Prefer the verbatim ``tracelens_launcher_path`` (set by
    # ``_row_to_candidate``) so AST resolution still works after
    # ``_finalize_candidates`` overwrites ``source_file`` with the
    # grep-located absolute path. Fall back to ``source_file`` /
    # ``kernel_path`` for candidates from non-v0.3 sources (raw trace
    # parser, priority_data.json fallback, csv) that never had a
    # launcher-formatted Kernel Path field.
    kernel_path = str(
        candidate.get("tracelens_launcher_path")
        or candidate.get("source_file")
        or candidate.get("kernel_path")
        or ""
    )
    raw_path, reported_line, reported_func = _parse_launcher_path(kernel_path)
    if not raw_path:
        return None
    source_path = Path(raw_path)
    if not source_path.is_absolute() and source_root is not None:
        source_path = source_root / source_path
    function_name = reported_func or source_path.stem
    definition_line = reported_line or 1
    if source_path.exists() and reported_func:
        ast_line = _function_line_from_ast(source_path, reported_func)
        if ast_line is not None:
            definition_line = ast_line
    return {
        "source_path":      str(source_path),
        "definition_line":  definition_line,
        "function_name":    function_name,
        "reported_path":    raw_path,
        "reported_line":    reported_line,
        "reported_func":    reported_func,
        "ast_resolved":     bool(reported_func and source_path.exists()
                                  and reported_func == function_name
                                  and reported_line != definition_line),
    }


def aggregate_by_source_function(
    candidates: list[dict[str, Any]],
    *,
    source_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Group TraceLens candidates by AST-resolved ``(path, line, fn)``.

    Returns a list of ``task_group`` dicts, sorted by aggregate kernel
    time (descending). Each group carries:

    * ``task_group_id`` — stable identifier ``tg<NN>``;
    * ``source_path`` / ``definition_line`` / ``function_name`` — the
      AST-resolved triple shared by every member candidate;
    * ``kernel_ids`` — every member candidate's ``kernel_id``;
    * ``primary_kernel_id`` — the highest-``duration_us`` member, used
      as the representative for prompt assembly + GEAK dispatch;
    * ``rows`` — the full candidate dict for every member (so
      ``build_prompt`` can render the multi-row benchmark cases section
      without re-joining against ``hot_kernels[]``);
    * ``aggregate_duration_us`` / ``aggregate_call_count`` / ``aggregate_gpu_pct``
      — sums across all members.

    Candidates whose ``kernel_path`` cannot be parsed are *not* placed
    in a group; the caller is expected to dispatch them via the legacy
    per-kernel path. Aggregation is additive: callers can ignore
    ``task_groups`` entirely without losing anything.
    """
    if not candidates:
        return []
    root: Path | None = None
    if source_root:
        root = Path(source_root).expanduser()
        if not root.is_dir():
            root = None

    # Grouping key includes the kernel ``operation`` name as the
    # primary component (NOT just the AST-resolved source function).
    # Rationale: TraceLens's "Kernel Path" column reports the calling
    # Python frame (e.g. ``vllm/model_executor/models/gpt_oss.py(283):
    # forward``), not the kernel implementation itself. Multiple
    # semantically distinct kernels — e.g. ``vllm::rocm_unquantized_gemm``
    # (P1 GEMM) and ``vllm::rocm_aiter_triton_add_rmsnorm_pad`` (P2
    # RMSNorm) — can share the same caller. Keying on source function
    # alone would merge them into one task_group with a meaningless
    # "rewrite forward" task. Including ``operation`` keeps each kernel
    # identity intact while still collapsing the same kernel called at
    # different shapes (the Q1 case from the user screenshots).
    groups: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        target = _resolve_source_target(cand, source_root=root)
        if target is None:
            continue
        operation = str(cand.get("name") or "").strip()
        key = (
            operation,
            target["source_path"],
            int(target["definition_line"]),
            str(target["function_name"]),
        )
        bucket = groups.get(key)
        if bucket is None:
            bucket = {
                "task_group_id":          "",  # filled below after sorting
                "operation":              operation,
                "source_path":            target["source_path"],
                "definition_line":        target["definition_line"],
                "function_name":          target["function_name"],
                "ast_resolved":           bool(target.get("ast_resolved")),
                "reported_path":          target["reported_path"],
                "kernel_ids":             [],
                "primary_kernel_id":      "",
                "rows":                   [],
                "aggregate_duration_us":  0.0,
                "aggregate_call_count":   0,
                "aggregate_gpu_pct":      0.0,
                # Cross-P-item prose collection (Q2). When the same
                # operation+source-function legitimately spans multiple
                # TraceLens P-items (e.g. once classified as memory-
                # bound at decode shapes, again as compute-bound at
                # prefill shapes), each P-item contributes its own
                # Identification / Reasoning / Resolution / Impact
                # tuple. We dedupe by ``(rank, title)`` and keep all
                # distinct entries so ``build_prompt`` can render every
                # P-item's hypothesis to GEAK rather than dropping all
                # but the primary's on the floor.
                "all_pitem_prose":        [],
                "_pitem_prose_seen":      set(),  # popped before return
            }
            groups[key] = bucket
        kid = str(cand.get("kernel_id") or "") or cand.get("name") or ""
        if kid and kid not in bucket["kernel_ids"]:
            bucket["kernel_ids"].append(kid)
        bucket["rows"].append(cand)
        # Q2: collect this candidate's P-item prose if we haven't seen
        # the same (rank, title) before in this group. Empty/missing
        # rank/title still produces a valid de-dup key, so candidates
        # from non-Detailed-Analysis paths (raw-trace fallback) without
        # P-item context contribute exactly one entry per group.
        try:
            pitem_rank = int(cand.get("tracelens_pitem_rank") or 0)
        except (TypeError, ValueError):
            pitem_rank = 0
        pitem_title = str(cand.get("tracelens_pitem_title") or "")
        pitem_key = (pitem_rank, pitem_title)
        if pitem_key not in bucket["_pitem_prose_seen"]:
            bucket["_pitem_prose_seen"].add(pitem_key)
            bucket["all_pitem_prose"].append({
                "rank":                    pitem_rank,
                "title":                   pitem_title,
                "identification":          str(cand.get("identification") or "").strip(),
                "reasoning_for_slowdown":  str(cand.get("reasoning_for_slowdown") or "").strip(),
                "resolution":              str(cand.get("resolution") or "").strip(),
                "impact_low_ms":           _safe_float(cand.get("impact_low_ms")),
                "impact_low_e2e_pct":      _safe_float(cand.get("impact_low_e2e_pct")),
                "impact_high_ms":          _safe_float(cand.get("impact_high_ms")),
                "impact_high_e2e_pct":     _safe_float(cand.get("impact_high_e2e_pct")),
            })
        try:
            bucket["aggregate_duration_us"] += float(cand.get("duration_us") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            bucket["aggregate_call_count"] += int(cand.get("call_count") or 0)
        except (TypeError, ValueError):
            pass
        try:
            bucket["aggregate_gpu_pct"] += float(cand.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            pass

    ordered = sorted(
        groups.values(),
        key=lambda g: g["aggregate_duration_us"],
        reverse=True,
    )
    for idx, group in enumerate(ordered, start=1):
        group["task_group_id"] = f"tg{idx:03d}"
        # Sort member rows by duration desc + pick the heaviest as
        # primary; build_prompt renders the primary row's metadata
        # and lists the rest as additional benchmark cases.
        group["rows"].sort(
            key=lambda r: float(r.get("duration_us") or 0.0), reverse=True,
        )
        if group["rows"]:
            primary = group["rows"][0]
            group["primary_kernel_id"] = str(
                primary.get("kernel_id") or primary.get("name") or ""
            )
        group["aggregate_duration_us"] = round(group["aggregate_duration_us"], 3)
        group["aggregate_gpu_pct"] = round(group["aggregate_gpu_pct"], 3)
        # Sort prose entries by rank ascending so P1 reads first; drop
        # any entry that's entirely empty (rank=0 + no prose) — those
        # come from non-P-item paths and add no signal.
        group["all_pitem_prose"].sort(key=lambda e: (e["rank"], e["title"]))
        group["all_pitem_prose"] = [
            e for e in group["all_pitem_prose"]
            if e["rank"]
            or e["identification"]
            or e["reasoning_for_slowdown"]
            or e["resolution"]
            or e["impact_low_ms"]
            or e["impact_high_ms"]
        ]
        # ``_pitem_prose_seen`` is a set; not JSON-serializable. Pop it
        # so summary.json / kernel_candidates.json serialization works.
        group.pop("_pitem_prose_seen", None)
    return ordered


__all__ = [
    "TraceLensSkillRunResult",
    "UPSTREAM_CATEGORY_TO_GEAK",
    "_extract_between",
    "_extract_pitem_prose",
    "_function_line_from_ast",
    "_parse_launcher_path",
    "aggregate_by_source_function",
    "build_orchestrator_prompt",
    "discover_capture_folder",
    "extract_idle_pct_from_analysis_md",
    "infer_analysis_mode",
    "normalize_upstream_category",
    "parse_analysis_md",
    "run_tracelens_skill",
    "write_local_cmd_prefix",
]
