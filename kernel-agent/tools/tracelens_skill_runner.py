#!/usr/bin/env python3
"""Run TraceLens standalone-analysis-orchestrator through Claude SDK.

This is the LLM-backed path for issue #124. It deliberately lives outside
``tracelens_analysis.py`` so the deterministic CLI/csv fallback remains
isolated and easy to test.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Task"]


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
2. Run the standalone-analysis-orchestrator workflow through Step 11.
3. If analysis_mode is inference and execution mode is graph_capture, pass the
   capture folder to the inference perf-report CLI exactly as the skill says.
4. Write all TraceLens outputs under the output directory above.
5. At minimum, ensure these files exist before you finish:
   - {output_dir / "standalone_analysis.md"}
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

    report_path = output_dir / "standalone_analysis.md"
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
    "build_orchestrator_prompt",
    "discover_capture_folder",
    "infer_analysis_mode",
    "raw_candidates_from_priority_data",
    "run_tracelens_skill",
    "write_local_cmd_prefix",
]
