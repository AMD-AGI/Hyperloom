#!/usr/bin/env python3
"""Run TraceLens analysis-orchestrator skill through Claude SDK.

This is the LLM-backed path for issue #124. It deliberately lives outside
``tracelens_analysis.py`` so the deterministic CLI/csv fallback remains
isolated and easy to test.
"""

from __future__ import annotations

import ast
import json
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
    """Artifacts produced by one TraceLens skill run."""

    output_dir: Path
    report_path: Path
    priority_data_path: Path
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
    if (framework or "").strip().lower() in {"vllm", "sglang"}:
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
5. At minimum, ensure these files exist before you finish:
   - {output_dir / "analysis.md"}  (TraceLens v0.3 final report)
   - {output_dir / "priority_data.json"}
   - {output_dir / "category_data" / "category_manifest.json"}
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
    # ``TraceLens/Agent/Analysis/utils/templates/analysis_template.md``. Older
    # branches (v0.2) used ``standalone_analysis.md`` — accept either so the
    # runner stays portable across release switches.
    report_path = output_dir / "analysis.md"
    if not report_path.exists():
        legacy_report = output_dir / "standalone_analysis.md"
        if legacy_report.exists():
            report_path = legacy_report
    priority_data_path = output_dir / "priority_data.json"
    if not report_path.exists():
        if sdk_error:
            raise RuntimeError(
                f"TraceLens SDK runner failed before writing {report_path}: {sdk_error}"
            )
        raise RuntimeError(f"TraceLens SDK runner did not write {report_path}")
    if not priority_data_path.exists():
        if sdk_error:
            raise RuntimeError(
                f"TraceLens SDK runner failed before writing {priority_data_path}: {sdk_error}"
            )
        raise RuntimeError(f"TraceLens SDK runner did not write {priority_data_path}")

    artifact_paths = {
        "tracelens_agent_report": str(report_path),
        "tracelens_priority_data": str(priority_data_path),
        "tracelens_cmd_prefix": str(prefix_path),
    }
    if sdk_error:
        artifact_paths["tracelens_agent_sdk_error"] = sdk_error

    return TraceLensSkillRunResult(
        output_dir=output_dir,
        report_path=report_path,
        priority_data_path=priority_data_path,
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
# This parser is the only place in Hyperloom that reads ``analysis.md``; it
# never consumes intermediate files (``priority_data.json``,
# ``category_data/*.json``) — those remain available via the legacy fallback in
# ``tracelens_analysis.py`` for the non-orchestrator path.
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
    """Extract Reasoning / Resolution / Impact-estimate fields from a
    Detailed Analysis P-item body. All fields default to empty / 0.0 so
    the parser stays additive — callers that don't care for prose are
    unaffected.

    Returns::

        {
          "reasoning_for_slowdown": str,
          "resolution":             str,
          "impact_low_ms":          float,
          "impact_low_e2e_pct":     float,
          "impact_high_ms":         float,
          "impact_high_e2e_pct":    float,
        }
    """
    reasoning = _extract_between(
        body, _REASONING_LABEL, (_RESOLUTION_LABEL, _IMPACT_LABEL),
    )
    resolution = _extract_between(body, _RESOLUTION_LABEL, (_IMPACT_LABEL,))
    low_match = _IMPACT_LOW_RE.search(body)
    high_match = _IMPACT_HIGH_RE.search(body)
    return {
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
        "source_file": kernel_path,
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


def parse_analysis_md(md_path: Path, top_k: int = 10) -> list[dict[str, Any]]:
    """Parse the TraceLens v0.3 ``analysis.md`` final report into hot-kernels.

    This is the reviewer-preferred exit interface: only the final report is
    consumed, not intermediate per-category JSON. The returned list mirrors
    the priority order of ``priority_data.findings`` (P1 first, P2 next, …)
    and, within a P-item, preserves the ``Operation`` rows of the 9-column
    Data table.

    Empty / non-existent reports return an empty list so callers can fall
    back to the legacy ``priority_data.json`` parser.
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


def raw_candidates_from_priority_data(priority_data_path: Path, top_k: int) -> list[dict[str, Any]]:
    """Convert TraceLens priority_data.json into hot-kernel candidate rows.

    ``priority_data.findings[]`` is category-level. Its ``members[]`` entries
    carry operation names and time/impact values, which are the best bridge
    back into Hyperloom's existing hot_kernels schema.
    """

    try:
        payload = json.loads(priority_data_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    findings = payload.get("findings")
    if not isinstance(findings, list):
        return []

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in sorted(
        (f for f in findings if isinstance(f, dict)),
        key=lambda f: _safe_float(f.get("impact_score")),
        reverse=True,
    ):
        category = str(finding.get("category") or "").strip()
        members = finding.get("members")
        if not isinstance(members, list):
            members = []
        for member in members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("operation") or member.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            time_ms = _safe_float(member.get("time_ms"))
            rows.append({
                "name": name,
                "duration_us": time_ms * 1000.0,
                "call_count": int(_safe_float(member.get("operation_count"), 1.0)),
                "source_file": str(member.get("source_path") or member.get("source_file") or ""),
                "source_type": "unknown",
                "shapes": [],
                "tracelens_category": category,
                "impact_score": _safe_float(member.get("impact_score")),
                "impact_score_low": _safe_float(member.get("impact_score_low")),
                "impact_score_high": _safe_float(member.get("impact_score_high")),
                "library": member.get("library") or finding.get("library") or "",
                "bound_type": member.get("bound_type") or finding.get("bound_type") or "",
            })
            if len(rows) >= top_k:
                return rows
    return rows


__all__ = [
    "TraceLensSkillRunResult",
    "UPSTREAM_CATEGORY_TO_GEAK",
    "_extract_between",
    "_extract_pitem_prose",
    "_function_line_from_ast",
    "_parse_launcher_path",
    "build_orchestrator_prompt",
    "discover_capture_folder",
    "infer_analysis_mode",
    "normalize_upstream_category",
    "parse_analysis_md",
    "raw_candidates_from_priority_data",
    "run_tracelens_skill",
    "write_local_cmd_prefix",
]
