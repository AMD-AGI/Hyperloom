#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Run TraceLens analysis-orchestrator skill through Claude SDK.

The LLM-backed path for issue #124; kept outside ``tracelens_analysis.py``
so the deterministic CLI/csv fallback stays isolated.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Task"]


# Upstream TraceLens category enum (orchestrator_prepare.py CATEGORY_SKILL_MAP) → GEAK labels.
UPSTREAM_CATEGORY_TO_GEAK: dict[str, str] = {
    "cpu_idle": "Other",
    "gemm": "GEMM",
    "groupedgemm_fwd": "GEMM",
    "groupedgemm_bwd": "GEMM",
    "moe_fused": "MoE",
    "moe_unfused": "MoE",
    "moe_aux": "MoE",
    "sdpa_fwd": "SDPA",
    "sdpa_bwd": "SDPA",
    "inferenceattention": "SDPA",
    "elementwise": "Elementwise",
    "reduce": "Reduction",
    "triton": "Triton",
    "flydsl": "FlyDSL",
    "norm": "LayerNorm",
    "norm_fwd": "LayerNorm",
    "norm_bwd": "LayerNorm",
    "rmsnorm": "LayerNorm",
    "convolution": "Convolution",
    "conv_fwd": "Convolution",
    "conv_bwd": "Convolution",
    "customcollective": "Communication",
    "other": "Other",
}


def normalize_upstream_category(raw: str) -> str:
    """Normalize a TraceLens category string to a GEAK-facing label.

    The raw value is lower-cased and its separators collapsed to underscores
    before lookup in :data:`UPSTREAM_CATEGORY_TO_GEAK`.

    Args:
        raw (str): The upstream TraceLens category string.

    Returns:
        str: The mapped GEAK-facing label, ``"unknown"`` when ``raw`` is empty,
            or the original ``raw`` value when no mapping exists.
    """

    if not raw:
        return "unknown"
    key = raw.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return UPSTREAM_CATEGORY_TO_GEAK.get(key, raw)


@dataclass
class TraceLensSkillRunResult:
    """Artifacts produced by one TraceLens skill run (per §2: ``analysis.md`` is the single source of truth)."""

    output_dir: Path
    report_path: Path
    artifact_paths: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


def shell_quote(path: Path | str) -> str:
    """Shell-quote a path for safe inclusion in a command string.

    Args:
        path (Path | str): The path to quote.

    Returns:
        str: The string form of ``path`` quoted for POSIX shells.
    """
    return shlex.quote(str(path))


def write_local_cmd_prefix(output_dir: Path, tracelens_root: Path) -> Path:
    """Create the command-prefix cache expected by the TraceLens skill.

    Writes a ``cache/cmd_prefix.txt`` file under ``output_dir`` whose contents
    ``cd <tracelens_root> && {CMD}`` let the skill root every shell command at
    the TraceLens project directory.

    Args:
        output_dir (Path): Directory under which the ``cache`` folder is created.
        tracelens_root (Path): The TraceLens project root the prefix cd's into.

    Returns:
        Path: The path to the written ``cmd_prefix.txt`` file.
    """

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = cache_dir / "cmd_prefix.txt"
    prefix_path.write_text(
        f"cd {shell_quote(tracelens_root)} && {{CMD}}\n",
        encoding="utf-8",
    )
    return prefix_path


def infer_analysis_mode(framework: str, requested: str) -> str:
    """Resolve the effective TraceLens analysis mode for a framework.

    An explicit non-default ``requested`` mode always wins. Otherwise inference
    frameworks (vllm/sglang/atom) default to ``"inference"`` grouping because
    their traces share the chrome-trace shape produced by the torch profiler;
    everything else falls back to the requested value or ``"default"``.

    Args:
        framework (str): The framework that produced the trace (e.g. ``vllm``).
        requested (str): The caller-requested analysis mode, possibly empty or
            ``"default"``.

    Returns:
        str: The resolved analysis mode string.
    """
    requested = (requested or "").strip().lower()
    if requested and requested != "default":
        return requested
    # atom shares the chrome-trace JSON shape of sglang/vllm.
    if (framework or "").strip().lower() in {"vllm", "sglang", "atom"}:
        return "inference"
    return requested or "default"


def discover_capture_folder(trace_input: Path, trace_files: list[Path]) -> Path | None:
    """Find a graph-capture folder near a Magpie torch_trace input.

    Checks the conventional ``capture_traces`` / ``graph_capture`` siblings of
    the trace input directory and of the first trace file, returning the first
    one that exists.

    Args:
        trace_input (Path): The trace input path (file or directory).
        trace_files (list[Path]): Discovered trace files; only the first is used.

    Returns:
        Path | None: The capture folder if one exists nearby, else ``None``.
    """

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
    tracelens_internal_root: Path | None,
    platform: str,
    framework: str,
    analysis_mode: str,
    capture_folder: Path | None,
) -> str:
    """Prompt a Claude SDK agent to execute the TraceLens standalone skill.

    Assembles the full natural-language instruction that pins every Step 0
    input (paths, platform, framework, analysis/execution mode, capture folder)
    so the agent can run the analysis-orchestrator workflow without prompting.

    Args:
        skill_path (Path): Path to the TraceLens skill file to follow.
        trace_path (Path): Path to the trace file to analyze.
        output_dir (Path): Directory where TraceLens outputs must be written.
        tracelens_root (Path): The TraceLens project root.
        platform (str): The target platform string.
        framework (str): The framework that produced the trace.
        analysis_mode (str): The requested analysis mode (resolved internally).
        capture_folder (Path | None): Graph-capture folder for inference runs.

    Returns:
        str: The fully assembled orchestrator prompt text.
    """

    analysis_mode = infer_analysis_mode(framework, analysis_mode)
    if analysis_mode == "inference" and capture_folder is not None:
        exec_mode = "graph_capture"
    elif analysis_mode == "inference":
        exec_mode = "eager"
    else:
        exec_mode = "default"

    internal_root_text = (
        str(tracelens_internal_root) if tracelens_internal_root
        else "(not installed; OSS-only mode)"
    )
    tl_extension_text = (
        "TraceLens_internal" if tracelens_internal_root
        else "(unset)"
    )

    comparison_scope = "standalone"
    capture_text = str(capture_folder) if capture_folder else "N/A"
    return f"""You are running TraceLens standalone analysis for Hyperloom.

Read and follow the FULL instructions in this skill file:
{skill_path}

All required Step 0 inputs are already provided below. Do not ask the user any
questions; proceed with the analysis.

Execution context:
- Environment: local
- TraceLens root: {tracelens_root}
- TraceLens-internal root: {internal_root_text}
- Command prefix cache: {output_dir / "cache" / "cmd_prefix.txt"}
- Trace file path: {trace_path}
- Output directory: {output_dir}
- Platform: {platform}
- Framework: {framework or "unknown"}
- Comparison scope: {comparison_scope}
- Analysis mode: {analysis_mode}
- Inference execution mode: {exec_mode}
- Capture folder path: {capture_text}
- TL_EXTENSION: {tl_extension_text}


Important requirements:
1. Use the provided command prefix cache for all shell commands.
2. Run the analysis-orchestrator workflow through Step 11.
3. If analysis_mode is inference and execution mode is graph_capture, pass the
   capture folder to the inference perf-report CLI exactly as the skill says.
4. Write all TraceLens outputs under the output directory above.
5. Ensure this file exists before you finish:
   - {output_dir / "analysis.md"}  (TraceLens final report; REQUIRED)
6. Do not run GEAK, OOB kernel optimization, or modify model/framework source.

When complete, respond with a short summary of the artifacts you wrote.
"""


def _import_sdk() -> tuple[Any, Any]:
    """Import the Claude Agent SDK and return its query primitives.

    Returns:
        tuple[Any, Any]: The ``(query, ClaudeAgentOptions)`` callables from
            ``claude_agent_sdk``.

    Raises:
        RuntimeError: If the SDK is not installed or lacks the expected
            ``query`` / ``ClaudeAgentOptions`` attributes.
    """
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
    """Yield text fragments from an SDK message.

    Handles both content blocks exposing a ``.text`` attribute or ``"text"``
    dict key, plus a top-level ``.result`` string.

    Args:
        message (Any): An SDK message object.

    Yields:
        str: Each non-empty text fragment found on the message.
    """
    for block in list(getattr(message, "content", None) or []):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            yield text
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            yield block["text"]
    result_text = getattr(message, "result", None)
    if isinstance(result_text, str) and result_text:
        yield result_text


def _json_safe(value: Any) -> Any:
    """Best-effort coercion of an SDK field into a JSON-serializable value.

    Tool inputs / results are usually plain dicts already, but the SDK may
    hand back nested objects; falling back to ``str`` keeps the transcript
    write infallible so a serialization edge case never aborts a TraceLens
    run (#266)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


def _serialize_sdk_block(block: Any) -> dict[str, Any]:
    """Serialize one message content block into a JSON-safe record.

    Blocks are tagged by class name (``TextBlock`` / ``ToolUseBlock`` /
    ``ToolResultBlock`` / ``ThinkingBlock`` / ...) and probed for the
    union of fields those types expose, rather than importing the SDK's
    concrete block classes. This mirrors
    ``ClaudeBackend._iter_blocks`` so the runner stays decoupled from
    claude-agent-sdk internals."""
    if isinstance(block, dict):
        record = _json_safe(block)
        if isinstance(record, dict):
            record.setdefault("block", record.get("type", "dict"))
            return record
        return {"block": "dict", "value": record}
    cls_name = type(block).__name__
    record: dict[str, Any] = {"block": cls_name}
    text = getattr(block, "text", None)
    if isinstance(text, str):
        record["text"] = text
    thinking = getattr(block, "thinking", None)
    if isinstance(thinking, str):
        record["thinking"] = thinking
    name = getattr(block, "name", None)
    if isinstance(name, str):
        record["name"] = name
    block_id = getattr(block, "id", None)
    if isinstance(block_id, str):
        record["id"] = block_id
    tool_input = getattr(block, "input", None)
    if tool_input is not None:
        record["input"] = _json_safe(tool_input)
    tool_use_id = getattr(block, "tool_use_id", None)
    if isinstance(tool_use_id, str):
        record["tool_use_id"] = tool_use_id
    # ``TextBlock`` has no ``content``; only tool-result-style blocks do.
    content = getattr(block, "content", None)
    if content is not None and cls_name != "TextBlock":
        record["content"] = _json_safe(content)
    is_error = getattr(block, "is_error", None)
    if isinstance(is_error, bool):
        record["is_error"] = is_error
    if set(record.keys()) == {"block"}:
        # Unknown block with no recognized fields: keep a bounded repr so
        # the transcript still records that something streamed.
        record["repr"] = str(block)[:2000]
    return record


def _serialize_sdk_message(message: Any, *, seq: int) -> dict[str, Any]:
    """Serialize one SDK stream message into a JSON-safe transcript record.

    The record carries the message class name (``AssistantMessage`` /
    ``ResultMessage`` / ...), its content blocks, and — when present on a
    terminal ``ResultMessage`` — the consolidated ``result`` text and the
    Anthropic ``usage`` dict (token accounting)."""
    record: dict[str, Any] = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": type(message).__name__,
    }
    blocks = getattr(message, "content", None)
    if blocks:
        record["content"] = [_serialize_sdk_block(b) for b in list(blocks)]
    result_text = getattr(message, "result", None)
    if isinstance(result_text, str) and result_text:
        record["result"] = result_text
    usage = getattr(message, "usage", None)
    if isinstance(usage, dict) and usage:
        record["usage"] = _json_safe(usage)
    return record


async def run_tracelens_skill(
    *,
    skill_path: Path,
    trace_path: Path,
    output_dir: Path,
    tracelens_root: Path,
    tracelens_internal_root: Path | None,
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
    """Execute the standalone TraceLens skill with Claude SDK.

    Prepares the command-prefix cache and orchestrator prompt, drives the SDK
    query loop, and treats the presence of ``analysis.md`` as the source of
    truth: an SDK error after the report was written is recorded as metadata
    rather than raised.

    Args:
        skill_path (Path): Path to the TraceLens skill file to follow.
        trace_path (Path): Path to the trace file to analyze.
        output_dir (Path): Directory where TraceLens outputs are written.
        tracelens_root (Path): The TraceLens project root.
        platform (str): The target platform string.
        framework (str): The framework that produced the trace.
        analysis_mode (str): The requested analysis mode.
        capture_folder (Path | None): Graph-capture folder for inference runs.
        budget_minutes (float): Soft time budget for the run (informational).
        model (str | None): Optional model override; defaults to the SDK default.
        sdk_query_factory (Callable[..., Any] | None): Optional injected query
            factory (used by tests); imported from the SDK when ``None``.
        sdk_options_cls (Any | None): Optional injected options class (used by
            tests); imported from the SDK when ``None``.
        log (Callable[[str], None] | None): Optional logging callback.

    Returns:
        TraceLensSkillRunResult: The artifacts produced by the run.

    Raises:
        RuntimeError: If ``analysis.md`` is not written by the run.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = write_local_cmd_prefix(output_dir, tracelens_root)
    prompt = build_orchestrator_prompt(
        skill_path=skill_path,
        trace_path=trace_path,
        output_dir=output_dir,
        tracelens_root=tracelens_root,
        tracelens_internal_root=tracelens_internal_root,
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
    # Fixed turn budget so smoke and production runs share the same orchestration envelope.
    max_turns = 300
    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "allowed_tools": DEFAULT_ALLOWED_TOOLS,
        "stderr": lambda line: log(f"[claude-sdk] {line.rstrip()}") if log else None,
    }
    if model:
        kwargs["model"] = model
    # Roots Bash relative paths at TraceLens; harmless in tests via FakeOptions.
    kwargs["cwd"] = str(tracelens_root)

    try:
        options = sdk_options_cls(**kwargs)
    except TypeError:
        # Older SDK builds lack cwd; absolute paths make retrying without it safe.
        kwargs.pop("cwd", None)
        options = sdk_options_cls(**kwargs)
    chunks: list[str] = []
    sdk_error = ""
    if log:
        log(f"TraceLens SDK runner: prefix cache={prefix_path}")

    # #266: persist a stream-JSON transcript of the agent's turns (text +
    # tool_use/tool_result blocks) so operators can inspect the lifecycle
    # and the artifacts it produced during execution. Capture is
    # best-effort: a serialization or IO error on the logging side must
    # never abort an otherwise-successful TraceLens run, so every write is
    # guarded and the transcript handle open is tolerant of IO failure.
    transcript_path = output_dir / "agent_transcript.jsonl"
    transcript_written = False
    transcript_seq = 0
    try:
        transcript_fh: Any = transcript_path.open("w", encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        transcript_fh = None
        if log:
            log(f"[claude-sdk] WARNING: cannot open transcript "
                f"{transcript_path}: {exc}")
    try:
        async for message in sdk_query_factory(prompt=prompt, options=options):
            if transcript_fh is not None:
                try:
                    record = _serialize_sdk_message(message, seq=transcript_seq)
                    transcript_fh.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    transcript_fh.flush()
                    transcript_seq += 1
                    transcript_written = True
                except Exception:  # noqa: BLE001
                    # Transcript is diagnostic-only; swallow and keep going.
                    pass
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001
        # SDK may error (e.g. "max turns reached") after writing artifacts; treat artifact presence as truth.
        sdk_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {sdk_error}")
    finally:
        if transcript_fh is not None:
            try:
                transcript_fh.close()
            except OSError as exc:
                # Closing the diagnostic transcript must never abort an
                # otherwise-successful run; surface it as a warning instead.
                if log:
                    log(f"[claude-sdk] WARNING: cannot close transcript "
                        f"{transcript_path}: {exc}")

    # Final report is ``analysis.md`` (contract since #148; the v0.2 standalone_analysis.md
    # fallback was dropped in #203 for masking orchestrator failures with stale data).
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
    # #266: surface the stream-JSON transcript so it flows into the
    # kernel-agent status sidecar (``artifacts.update(skill_result
    # .artifact_paths)`` in tracelens_analysis.py) and a launcher/operator
    # can tail it during execution. Only advertise it when at least one
    # turn was actually recorded.
    if transcript_written:
        artifact_paths["tracelens_agent_transcript"] = str(transcript_path)
    if sdk_error:
        artifact_paths["tracelens_agent_sdk_error"] = sdk_error

    return TraceLensSkillRunResult(
        output_dir=output_dir,
        report_path=report_path,
        raw_text="\n".join(chunks),
        artifact_paths=artifact_paths,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float, falling back to a default on failure.

    Args:
        value (Any): The value to coerce.
        default (float): The value returned when ``value`` is ``None`` or not
            convertible.

    Returns:
        float: The parsed float, or ``default`` on ``None`` / parse failure.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# analysis.md parser (TraceLens final-report contract; PR #155): reads p_item markers + compute-tier reasoning blocks with a 9-column **Data:** table. Sole reader of candidate data.
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
# Lowercased canonical header tokens; separates the 9 typed fields from trailing extras.
_DATA_TABLE_CANONICAL_KEY_SET = frozenset(
    tok.strip().lower() for tok in _DATA_TABLE_HEADER_TOKENS
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
# Detailed Analysis sibling labels (Reasoning/Resolution/Impact estimate); extracted
# prose is a hypothesis to validate, not imperative guidance — see ``build_prompt``.
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
    """Parse ``key=value`` attributes from an HTML-comment marker blob.

    Args:
        blob (str): The inner text of a TraceLens marker comment.

    Returns:
        dict[str, str]: A mapping of attribute names to their string values.
    """
    return dict(re.findall(r"(\w+)=([^\s>]+)", blob))


def _extract_between(
    text: str, start_marker: str, end_markers: tuple[str, ...],
) -> str:
    """Return the substring between ``start_marker`` and the earliest ``end_markers`` (tail if none; empty if start absent)."""
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end_positions = [text.find(m, start) for m in end_markers]
    end_positions = [pos for pos in end_positions if pos != -1]
    end = min(end_positions) if end_positions else len(text)
    return text[start:end].strip()


def _extract_pitem_prose(body: str) -> dict[str, Any]:
    """Extract Identification / Reasoning / Resolution / Impact-estimate fields from a P-item body (all default empty/0.0)."""
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
    """Return per-P-item metadata (``category``/``low``/``mid``/``high``), in priority order, from p_item markers."""

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
    """Split the report into compute-tier reasoning blocks.

    Args:
        text (str): The full ``analysis.md`` report text.

    Returns:
        list[tuple[int, str, str]]: One ``(rank, title, body)`` triple per
            compute-tier reasoning-candidate block found.
    """

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
    """Pull the 9-column markdown table that follows ``**Data:**`` (raw header + data cells; ends at blank line or next ``**Field:**``)."""

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
    """Convert one parsed data-table row into a hot-kernel candidate dict.

    Maps the 9 canonical columns into typed candidate fields, preserves any
    trailing extra columns under ``tracelens_extra_columns``, resolves the
    launcher path to an absolute source file where possible, and attaches the
    shared P-item prose.

    Args:
        headers (list[str]): Lower-cased column header names for ``cells``.
        cells (list[str]): The row's cell strings, aligned with ``headers``.
        category (str): The TraceLens category for the owning P-item.
        rank (int): The P-item rank (1-based).
        title (str): The P-item title.
        library (str): The library name parsed from the P-item title.
        impact (dict[str, float]): Impact scores for the owning P-item.
        prose (dict[str, Any] | None): Shared P-item prose to attach, if any.

    Returns:
        dict[str, Any] | None: The candidate dict, or ``None`` when the row is
            malformed (cell count mismatch) or names a placeholder operation.
    """
    if len(cells) != len(headers):
        return None
    record = dict(zip(headers, cells))
    # Trailing extras (spec allows appended columns) preserved verbatim for downstream consumers.
    extra_columns = {
        key: value for key, value in record.items()
        if key not in _DATA_TABLE_CANONICAL_KEY_SET
    }

    name = record.get("operation", "").strip()
    if not name or name in {"-", "—"}:
        return None
    args = record.get("args", "").replace("<br>", "\n").strip()
    shapes = [s.strip() for s in args.split("\n") if s.strip() and s.strip() not in {"-", "—"}]
    kernel_path = record.get("kernel path", "").strip()
    if kernel_path in {"-", "—"}:
        kernel_path = ""
    # Promote a framework-relative launcher path to its absolute on-disk source so the
    # patchability gate passes; verbatim launcher kept on tracelens_launcher_path below.
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
        # PR-B §1: keep raw Kernel Path verbatim so aggregation's AST resolution survives _finalize_candidates' source_file overwrite.
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
    if extra_columns:
        candidate["tracelens_extra_columns"] = extra_columns
    if prose:
        # Duplicate the block-shared P-item prose onto each candidate so it stays self-describing.
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
    """Extract ``Idle %`` from the Executive Summary table in ``analysis.md`` (§1/§2 idle-gate); ``None`` on missing/unparseable so callers skip the gate gracefully."""
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
    """Per-row sort key for the ``Lower Efficiency`` budget filter (§2); rows with no efficiency (0.0) sort last."""
    eff = candidate.get("efficiency_percent")
    try:
        value = float(eff)
    except (TypeError, ValueError):
        return float("inf")
    if value <= 0.0:
        return float("inf")
    return value


def parse_analysis_md(md_path: Path, top_k: int = 10) -> list[dict[str, Any]]:
    """Parse the TraceLens ``analysis.md`` final report into hot-kernels in §2 priority order (P-item, then lower efficiency within); empty/missing → []."""

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
    canonical_width = len(headers_canonical)
    for rank, title, body in blocks:
        rows = _extract_data_table(body)
        if not rows:
            continue
        header_row = [cell.strip().lower() for cell in rows[0]]
        # Validate first 9 cells against the canonical schema; reject narrower/reordered headers (silent wrong-mapping would corrupt candidates).
        if len(header_row) < canonical_width:
            continue
        header_prefix = header_row[:canonical_width]
        if header_prefix != headers_canonical:
            normalized: list[str] = []
            for cell in header_prefix:
                match = next(
                    (canon for canon in headers_canonical if canon in cell),
                    cell,
                )
                normalized.append(match)
            if normalized != headers_canonical:
                continue
            header_prefix = normalized
        # Splice normalized canonical names back into the full header (extras kept verbatim).
        header_row = header_prefix + header_row[canonical_width:]
        # P-item meta by 1-based rank (P1 → pitems[0]); a missing entry => category unknown.
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


# PR-B §1: source-function aggregation — group candidates sharing an AST-resolved (source_path, line, fn) triple to avoid wasting per-kernel GEAK budget; unparseable kernel_path falls back to per-kernel dispatch.

# Launcher path shapes: ``<path>(<line>): <func>`` (Python) or bare / ``<path>#L<line>`` (HIP); missing pieces None.
_LAUNCHER_PATH_RE = re.compile(
    r"(?P<path>.+?)\((?P<line>\d+)\)\s*:\s*(?P<func>[A-Za-z_][A-Za-z0-9_]*)\s*$",
)
# Placeholders for unresolved Kernel Paths; must not survive parsing (else all group under a bogus path).
_LAUNCHER_PATH_PLACEHOLDERS: frozenset[str] = frozenset({
    "", "-", "—", "–", "n/a", "none", "null", "tbd", "unknown",
})


def _parse_launcher_path(kernel_path: str) -> tuple[str, int | None, str | None]:
    """Parse a TraceLens kernel-path to ``(path, line, function_name)`` (accepts ``<path>(<line>): <func>``/``<path>#L<line>``/bare; placeholders → ``("", None, None)``)."""
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


# Launcher path → absolute source file resolver: resolve sys.path-relative Kernel Paths to absolute so the patchability gate accepts them. Strategy (most-specific first): $HYPERLOOM_FRAMEWORK_SOURCE_ROOTS override, importlib find_spec, then hardcoded fallback table.
_FRAMEWORK_PKG_FALLBACK_ROOTS: dict[str, tuple[str, ...]] = {
    "aiter": ("/sgl-workspace/aiter",),
    "sglang": ("/sgl-workspace/sglang/python", "/sgl-workspace/sglang"),
    "vllm": (
        "/usr/local/lib/python3.12/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/opt/venv/lib/python3.10/site-packages",
        "/sgl-workspace/vllm",
    ),
    # atom fallback roots for CSV-only / static-analysis parses (import atom may not have run).
    # Kept in sync with the _REUSABLE_SOURCE_ROOTS in kernel_request_handlers / tracelens_analysis
    # (pinned by test_framework_paths_units.py). /app/ATOM is the editable-install parent.
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
    """Parse ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` (comma-separated ``pkg=/abs/parent``) into ``{pkg: (root,...)}``; unparseable entries skipped."""
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
    """Return the directory containing ``pkg/`` on the live ``sys.path`` via find_spec; ``None`` if not importable."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        # Package dir's parent is what we prepend to ``pkg/rest/of/path``.
        loc = list(spec.submodule_search_locations)[0]
        return os.path.dirname(loc)
    if spec.origin and spec.origin.endswith(".py"):
        # Single-file module.
        return os.path.dirname(os.path.dirname(spec.origin))
    return None


def _resolve_launcher_to_abs_source(
    kernel_path: str,
) -> tuple[str, int | None, str | None] | None:
    """Resolve a TraceLens launcher-path to an absolute file; ``(abs_file, line, function_name)`` only when the file exists, else ``None`` (conservative miss → caller's grep locator)."""
    raw_path, line, func = _parse_launcher_path(kernel_path)
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        # Already absolute — None signals "no rewrite needed".
        return None
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
        # Validate the file actually hosts the launcher's function (guards sys.path shadowing); AST misses try the next root.
        if func and abs_path.endswith(".py"):
            if _function_line_from_ast(Path(abs_path), func) is None:
                continue
        return abs_path, line, func
    return None


def _function_line_from_ast(path: Path, function_name: str) -> int | None:
    """Return the lineno of the first (Async)FunctionDef in ``path`` named ``function_name``; ``None`` if unreadable/unparseable/absent."""
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
    """Resolve a candidate's launcher path to a ``(source_path, definition_line, function_name)`` triple; ``None`` when unparseable. AST line overrides the reported call-site line when resolvable."""
    # Prefer verbatim tracelens_launcher_path so AST resolution survives _finalize_candidates'
    # source_file overwrite; fall back to source_file / kernel_path for non-TraceLens candidates.
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


_NATIVE_SOURCE_SUFFIXES = (
    ".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c",
)


def _is_native_source(path: str) -> bool:
    """True for C/C++/HIP/CUDA source files (#420).

    Native sources have no Python AST to resolve a stable ``def`` line, so
    TraceLens reports the call-site ``#L<line>`` which differs per call —
    keying a task_group on that line splits one device kernel across
    groups. Callers therefore drop the line/function key components for
    these files.
    """
    return str(path).lower().endswith(_NATIVE_SOURCE_SUFFIXES)


def _normalize_operation_key(operation: str) -> str:
    """Canonicalize a TraceLens operation name for task-group keying (#420).

    Strips balanced ``<...>`` template-argument lists (nested-safe) so the
    SAME kernel profiled at different dtypes/shapes — e.g.
    ``rmsnorm_kernel<bf16>`` vs ``rmsnorm_kernel<fp16>`` — groups together,
    while DISTINCT kernels (different base names) stay separate (the Q1
    invariant). Returns the original string when stripping leaves nothing.
    """
    s = str(operation).strip()
    if "<" not in s:
        return s
    out: list[str] = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth > 0:
                depth -= 1
        elif depth == 0:
            out.append(ch)
    normalized = "".join(out).strip()
    return normalized or s


def aggregate_by_source_function(
    candidates: list[dict[str, Any]],
    *,
    source_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Group TraceLens candidates into per-kernel ``task_group`` dicts, sorted by aggregate time (desc). Native (.cu/.hip/.cpp) key on ``(source_path, function)`` only (collapse #420 over-split instantiations); Python key on ``(operation, path, line, function)`` since one caller frame can launch distinct kernels (Q1). Each group carries task_group_id/source_path/definition_line/function_name/kernel_ids/primary_kernel_id/rows/aggregate_*. Unparseable candidates left out for legacy per-kernel dispatch."""
    if not candidates:
        return []
    root: Path | None = None
    if source_root:
        root = Path(source_root).expanduser()
        if not root.is_dir():
            root = None

    # Grouping key: native (.cu/.hip/.cpp) on (source_path, function) only — collapse #420 over-split mangled-symbol instantiations of one __global__ into one composite job; Python on (operation, path, line, function) — TraceLens Kernel Path is the calling frame so distinct kernels share a caller, operation MUST stay in the key (Q1) normalized across dtypes. source_path is normpath-canonicalized.
    groups: dict[tuple, dict[str, Any]] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        target = _resolve_source_target(cand, source_root=root)
        if target is None:
            continue
        operation = str(cand.get("name") or "").strip()
        src_norm = os.path.normpath(str(target["source_path"]))
        function_name = str(target["function_name"])
        if _is_native_source(src_norm):
            # Drop the mangled operation + per-call line: one __global__
            # template == one composite job, keyed on its source TU.
            key: tuple = ("native", src_norm, function_name)
        else:
            # Keep the (normalized) operation so distinct kernels sharing
            # one Python caller frame stay separate (Q1 invariant).
            norm_op = _normalize_operation_key(operation)
            key = (
                "py",
                norm_op,
                src_norm,
                int(target["definition_line"]),
                function_name,
            )
        bucket = groups.get(key)
        if bucket is None:
            bucket = {
                "task_group_id":          "",  # filled below after sorting
                "operation":              operation,
                "source_path":            src_norm,
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
                # Q2: distinct per-P-item prose (deduped by (rank, title)) so build_prompt renders every P-item when one function spans multiple.
                "all_pitem_prose":        [],
                "_pitem_prose_seen":      set(),  # popped before return
            }
            groups[key] = bucket
        kid = str(cand.get("kernel_id") or "") or cand.get("name") or ""
        if kid and kid not in bucket["kernel_ids"]:
            bucket["kernel_ids"].append(kid)
        bucket["rows"].append(cand)
        # Collect P-item prose deduped by (rank, title); missing rank/title still yields a valid key.
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
        # Heaviest row (by duration) becomes primary; the rest are additional benchmark cases.
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
        # Sort prose by rank (P1 first); drop entirely-empty entries.
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
        # ``_pitem_prose_seen`` is a set (not JSON-serializable); pop before return.
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
