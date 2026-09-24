#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the TraceLens analysis-orchestrator skill through an agent runtime.

The LLM-backed path, kept outside ``tracelens_analysis.py`` so the
deterministic CLI/csv fallback stays isolated. The skill itself is plain text
and provider-neutral; only the runtime that executes it differs. Two are
supported, both real agent SDKs: the Claude Agent SDK, and the Codex Agent SDK
for deployments configured with the OpenAI side alone.
"""

from __future__ import annotations

import asyncio
import ast
import importlib.util
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from hyperloom.common.codex_session import (
    CodexSessionError,
    CodexSessionResult,
    run_codex_turn,
)
from hyperloom.common.llm_config import claude_sdk_env_options
from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL

# Sibling import works whether run as a script or loaded via importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _capture_shapes import is_capture_dir_name
from _io_utils import safe_float
from _literal_utils import LITERAL_EVAL_ERRORS as _LITERAL_EVAL_ERRORS
from _literal_utils import safe_literal_eval as _safe_literal_eval
from _task_group_contract import (
    build_operator_identity,
    build_task_group_shape_cases,
    legacy_operator_identity_keys,
    operator_identity_key,
)

sys.path.pop(0)


DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Task"]

# The Codex SDK has no literal Read/Write/Bash/Task tools, and this thread runs
# without sub-agents, so the skill's tool names have to be mapped to Codex's
# native shell and patch capabilities before the agent follows them.
_CODEX_DEVELOPER_INSTRUCTIONS = """\
## Codex runtime mapping
Use shell commands to inspect, search and run files, and your native patch/edit
capability to write them. Where the skill names Read, Write, Edit, Grep, Glob,
Bash or Task, it describes an equivalent capability rather than a literal tool;
carry out any sub-agent step yourself, in sequence.

## System instructions
You are a TraceLens analysis runner inside Hyperloom. Execute only the
requested standalone analysis workflow. Use absolute paths, write artifacts
under the requested output directory, and do not modify application source
code.
"""


# Per-message stream-idle timeout (seconds). The in-process Claude SDK query has
# no client-side read timeout, so a stalled gateway stream would block forever;
# we bound the wait for each next SDK message (inactivity, not total). Env-overridable.
_DEFAULT_STREAM_IDLE_TIMEOUT_SEC = 300.0

# While a tool call is in flight the SDK is silent by design, so the bound above
# would kill a working run. Session 20260803T091144Z lost its roofline exactly
# that way: the agent launched TraceLens_generate_perf_report_pytorch over a
# 896 MB trace and was killed at 300s, while the same command run by hand was
# still making progress 25 minutes in.
_DEFAULT_TOOL_IDLE_TIMEOUT_SEC = 3600.0


def _resolve_stream_idle_timeout_sec() -> float:
    """Resolve the per-message SDK stream-idle timeout in seconds.

    Reads ``HYPERLOOM_TRACELENS_STREAM_IDLE_TIMEOUT_SEC`` and falls back to
    :data:`_DEFAULT_STREAM_IDLE_TIMEOUT_SEC`; floored at 30s. A value <= 0
    disables the idle timeout (legacy unbounded behavior).

    Returns:
        float: The idle timeout in seconds (0 disables it).
    """
    raw = os.environ.get("HYPERLOOM_TRACELENS_STREAM_IDLE_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_STREAM_IDLE_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_STREAM_IDLE_TIMEOUT_SEC
    if value <= 0:
        return 0.0
    return max(30.0, value)


def _resolve_tool_idle_timeout_sec(idle_timeout: float) -> float:
    """Resolve the idle bound that applies while an agent tool call is running.

    The SDK emits nothing between the ``ToolUseBlock`` that launches a tool and
    the result that ends it, so the plain idle timeout cannot tell a dead
    gateway from a working tool. Reads
    ``HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC``; floored at 30s, and a value
    <= 0 removes the bound while a tool is in flight.

    Args:
        idle_timeout: The between-messages idle timeout, used as a floor so the
            tool bound is never the tighter of the two.

    Returns:
        float: The in-flight idle timeout in seconds (0 disables it).
    """
    raw = os.environ.get("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", "").strip()
    if not raw:
        return max(_DEFAULT_TOOL_IDLE_TIMEOUT_SEC, idle_timeout)
    try:
        value = float(raw)
    except ValueError:
        return max(_DEFAULT_TOOL_IDLE_TIMEOUT_SEC, idle_timeout)
    if value <= 0:
        return 0.0
    return max(30.0, value)


def _tool_call_transition(message: Any) -> str | None:
    """Return ``"start"`` / ``"end"`` when a message brackets a tool call.

    Args:
        message: An SDK stream message.

    Returns:
        str | None: ``"start"`` when the message launches a tool, ``"end"`` when
            it delivers a tool result or terminates the run, else ``None``.
    """
    name = type(message).__name__
    if name == "TaskStartedMessage":
        return "start"
    if name == "ResultMessage":
        return "end"
    transition: str | None = None
    for block in list(getattr(message, "content", None) or []):
        block_name = type(block).__name__
        if "ToolUse" in block_name:
            transition = "start"
        elif "ToolResult" in block_name:
            transition = "end"
    return transition


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
    """Artifacts produced by one TraceLens skill run (``analysis.md`` is the single source of truth)."""

    output_dir: Path
    report_path: Path
    # Which runner produced these artifacts, so callers report the provider that
    # actually ran rather than assuming one. Required: a new runner that forgets
    # to declare itself fails at construction instead of mislabelling its output.
    runner: str
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
    if (framework or "").strip().lower() in {"vllm", "sglang", "atom"}:
        return "inference"
    return requested or "default"


def discover_capture_folder(trace_input: Path, trace_files: list[Path]) -> Path | None:
    """Find a graph-capture folder near a Magpie torch_trace input.

    Scans the trace input directory and the first trace file's neighbourhood for
    a subdirectory whose name matches the shared capture-directory shape, so a
    layout that ranking already demotes is also a layout discovery can find.
    Matching by shape rather than by two hard-coded names is what lets an
    unpatched SGLang's ``graph_capture_profile/`` through: it was previously
    missed here, so the capture folder went unpassed even on runs that had
    correctly picked the workload trace.

    Args:
        trace_input (Path): The trace input path (file or directory).
        trace_files (list[Path]): Discovered trace files; only the first is used.

    Returns:
        Path | None: The capture folder if one exists nearby, else ``None``.
    """

    capture_dir_priority = ("capture_traces", "graph_capture_profile", "graph_capture")
    search_roots: list[Path] = []
    if trace_input.is_dir():
        search_roots.append(trace_input)
    for trace_file in trace_files[:1]:
        search_roots.extend([trace_file.parent, trace_file.parent.parent])
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        try:
            children = [child for child in root.iterdir() if child.is_dir()]
        except OSError:
            continue
        children_by_name = {child.name.lower(): child for child in children}
        for name in capture_dir_priority:
            if child := children_by_name.get(name):
                return child
        for child in sorted(children):
            if is_capture_dir_name(child.name):
                return child
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
    """Prompt an agent to execute the TraceLens standalone skill.

    Provider-neutral: the same prompt drives the Claude SDK runner and the
    Codex SDK runner.

    Assembles the full natural-language instruction that pins every required
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

    internal_root_text = str(tracelens_internal_root) if tracelens_internal_root else "(not installed; OSS-only mode)"
    tl_extension_text = "TraceLens_internal" if tracelens_internal_root else "(unset)"

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
2. Run the analysis-orchestrator workflow through Step 12.
3. If analysis_mode is inference and execution mode is graph_capture, pass the
   capture folder to the inference perf-report CLI exactly as the skill says.
4. Write all TraceLens outputs under the output directory above.
5. Ensure this file exists before you finish:
   - {output_dir / "analysis.md"}  (TraceLens final report; REQUIRED)
6. Do not run GEAK, kernel optimization, or modify model/framework source.

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
            "claude_agent_sdk not installed; run src/hyperloom/agents/kernel/scripts/install.sh first"
        ) from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions


def _should_use_codex_runner() -> bool:
    """Return true when the Codex Agent SDK runner should run this skill.

    An OpenAI-only deployment has no Claude credentials to drive the Claude
    Agent SDK, so the Codex runner is the only one that can execute. The choice
    itself belongs to :mod:`hyperloom.common.llm_config`, so this cannot
    disagree with backend selection or the forge kernel_backend.
    """
    from hyperloom.common import llm_config  # local import: keep module import-light

    return llm_config.preferred_agent_backend() == llm_config.AGENT_BACKEND_CODEX


def _iter_message_text(message: Any) -> Iterable[str]:
    from hyperloom.common.claude_oneshot import message_text

    yield from (t for t in message_text(message) if t)


async def _run_tracelens_skill_codex(
    *,
    prompt: str,
    output_dir: Path,
    prefix_path: Path,
    tracelens_root: Path,
    capture_folder: Path | None,
    model: str,
    timeout_sec: float,
    codex_turn_runner: Callable[..., Awaitable[CodexSessionResult]],
    log: Callable[[str], None] | None,
) -> TraceLensSkillRunResult:
    """Run the TraceLens skill on the Codex Agent SDK.

    The session works out of ``tracelens_root``, matching the Claude path, so
    the skill's command-prefix cache and the TraceLens CLIs' own relative paths
    resolve identically on both runners. The write scope is that workspace,
    ``output_dir`` and ``capture_folder``; the rest of the host is readable but
    immutable. The TraceLens checkout has to stay writable because its CLIs
    write caches and intermediates into their own tree, so narrowing the
    workspace to ``output_dir`` alone would break the analysis rather than
    harden it. The capture folder is writable for the same reason: the
    inference perf report classifies it before merging it, and that
    classification writes ``execution_details.json`` into the folder itself.

    Args:
        prompt (str): The orchestrator prompt.
        output_dir (Path): Directory the report is written to.
        prefix_path (Path): The command-prefix cache path, reported as an
            artifact.
        tracelens_root (Path): The TraceLens project root; the session cwd.
        capture_folder (Path | None): Graph-capture folder the inference perf
            report writes into, or ``None`` outside graph-capture runs.
        model (str): The Codex model id.
        timeout_sec (float): Wall-clock budget for the turn.
        codex_turn_runner (Callable[..., Awaitable[CodexSessionResult]]): The
            Codex turn entry point (injected by tests).
        log (Callable[[str], None] | None): Optional logging callback.

    Returns:
        TraceLensSkillRunResult: The artifacts produced by the run.

    Raises:
        RuntimeError: If ``analysis.md`` was not written.
    """
    codex_error = ""
    result = CodexSessionResult()
    try:
        result = await codex_turn_runner(
            prompt=prompt,
            developer_instructions=_CODEX_DEVELOPER_INSTRUCTIONS,
            cwd=tracelens_root,
            model=model,
            timeout_sec=timeout_sec,
            writable_roots=(output_dir, capture_folder) if capture_folder else (output_dir,),
        )
    except CodexSessionError as exc:
        # The SDK can fail after the report landed; artifact presence decides.
        codex_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[codex-sdk] WARNING: {codex_error}")
    codex_error = codex_error or result.error

    report_path = output_dir / "analysis.md"
    if not report_path.exists():
        if codex_error:
            raise RuntimeError(f"TraceLens Codex runner failed before writing {report_path}: {codex_error}")
        raise RuntimeError(f"TraceLens Codex runner did not write {report_path}")

    artifact_paths = {
        "tracelens_agent_report": str(report_path),
        "tracelens_cmd_prefix": str(prefix_path),
    }
    if codex_error:
        artifact_paths["tracelens_agent_sdk_error"] = codex_error
    return TraceLensSkillRunResult(
        output_dir=output_dir,
        report_path=report_path,
        runner="codex",
        raw_text=result.text,
        artifact_paths=artifact_paths,
    )


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
    codex_turn_runner: Callable[..., Awaitable[CodexSessionResult]] | None = None,
    log: Callable[[str], None] | None = None,
) -> TraceLensSkillRunResult:
    """Execute the standalone TraceLens skill on the configured agent runtime.

    Prepares the command-prefix cache and orchestrator prompt, then dispatches
    to the Codex Agent SDK when the deployment has only the OpenAI side
    configured, and to the Claude Agent SDK otherwise. Either way the presence
    of ``analysis.md`` is the source of truth: a runtime error after the report
    was written is recorded as metadata rather than raised.

    Args:
        skill_path (Path): Path to the TraceLens skill file to follow.
        trace_path (Path): Path to the trace file to analyze.
        output_dir (Path): Directory where TraceLens outputs are written.
        tracelens_root (Path): The TraceLens project root.
        platform (str): The target platform string.
        framework (str): The framework that produced the trace.
        analysis_mode (str): The requested analysis mode.
        capture_folder (Path | None): Graph-capture folder for inference runs.
        budget_minutes (float): Time budget for the run. The Codex path spends
            it as the turn's wall-clock timeout (floored at 60s, matching the
            other TraceLens subprocess timeouts); the Claude path bounds each
            SDK message by a stream-idle timeout instead.
        model (str | None): Optional model override. Defaults to
            :data:`DEFAULT_CLAUDE_MODEL` on the Claude SDK path, or ``$CODEX_MODEL`` /
            :data:`DEFAULT_CODEX_MODEL` on the Codex SDK path.
        sdk_query_factory (Callable[..., Any] | None): Optional injected query
            factory (used by tests); imported from the SDK when ``None``.
        sdk_options_cls (Any | None): Optional injected options class (used by
            tests); imported from the SDK when ``None``.
        codex_turn_runner (Callable[..., Awaitable[CodexSessionResult]] | None):
            Optional injected Codex turn entry point (used by tests); defaults
            to :func:`hyperloom.common.codex_session.run_codex_turn`.
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

    resolved_model = (model or "").strip()
    if _should_use_codex_runner() and (
        codex_turn_runner is not None or (sdk_query_factory is None and sdk_options_cls is None)
    ):
        codex_model = resolved_model or (os.environ.get("CODEX_MODEL") or "").strip() or DEFAULT_CODEX_MODEL
        return await _run_tracelens_skill_codex(
            prompt=prompt,
            output_dir=output_dir,
            prefix_path=prefix_path,
            tracelens_root=tracelens_root,
            capture_folder=capture_folder,
            model=codex_model,
            timeout_sec=max(60.0, float(budget_minutes) * 60.0),
            codex_turn_runner=codex_turn_runner or run_codex_turn,
            log=log,
        )

    if sdk_query_factory is None or sdk_options_cls is None:
        query, options_cls = _import_sdk()
        sdk_query_factory = sdk_query_factory or query
        sdk_options_cls = sdk_options_cls or options_cls
        # Only harden the real SDK path; tests inject fakes and skip this.
        os.environ.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        os.environ.setdefault("DISABLE_AUTOUPDATER", "1")

    system_prompt = (
        "You are a TraceLens analysis runner inside Hyperloom. Execute only "
        "the requested standalone analysis workflow. Use absolute paths, "
        "write artifacts under the requested output directory, and do not "
        "modify application source code."
    )
    max_turns = 300
    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "allowed_tools": DEFAULT_ALLOWED_TOOLS,
        "stderr": lambda line: log(f"[claude-sdk] {line.rstrip()}") if log else None,
    }
    resolved_model = resolved_model or DEFAULT_CLAUDE_MODEL
    kwargs["model"] = resolved_model
    # Roots Bash relative paths at TraceLens; harmless in tests via FakeOptions.
    kwargs["cwd"] = str(tracelens_root)
    kwargs.update(
        claude_sdk_env_options(
            model=resolved_model,
            component="tracelens",
            operation="analyze_trace",
        )
    )

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
    # Drive the SDK stream manually so each next message is bounded by a
    # per-message idle timeout (inactivity, not a total budget); the in-process
    # SDK has no client-side read timeout and would otherwise block on a stall.
    idle_timeout = _resolve_stream_idle_timeout_sec()
    tool_idle_timeout = _resolve_tool_idle_timeout_sec(idle_timeout)
    tool_in_flight = False
    stream = sdk_query_factory(prompt=prompt, options=options)
    stream_iter = stream.__aiter__() if hasattr(stream, "__aiter__") else stream
    try:
        while True:
            wait_for = tool_idle_timeout if tool_in_flight else idle_timeout
            try:
                if wait_for > 0:
                    message = await asyncio.wait_for(stream_iter.__anext__(), timeout=wait_for)
                else:
                    message = await stream_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                # Stream went quiet past its bound: abort and tear the generator
                # down so its transport/subprocess does not leak. Name the phase —
                # silence during a tool call means the tool overran its bound, not
                # that the gateway died.
                phase = "while a tool call was in flight" if tool_in_flight else "with no tool call in flight"
                sdk_error = f"stream idle timeout: no SDK message for {wait_for:.0f}s {phase}"
                if log:
                    log(f"[claude-sdk] WARNING: {sdk_error}")
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    try:
                        await asyncio.wait_for(aclose(), timeout=10.0)
                    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                        pass
                break
            transition = _tool_call_transition(message)
            if transition == "start":
                tool_in_flight = True
            elif transition == "end":
                tool_in_flight = False
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001
        # SDK may error after writing artifacts; treat artifact presence as truth.
        sdk_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {sdk_error}")

    # Final report is ``analysis.md``.
    report_path = output_dir / "analysis.md"
    if not report_path.exists():
        if sdk_error:
            raise RuntimeError(f"TraceLens SDK runner failed before writing {report_path}: {sdk_error}")
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
        runner="claude_agent_sdk",
        raw_text="\n".join(chunks),
        artifact_paths=artifact_paths,
    )


# Shared numeric coercion (see _io_utils.safe_float).
_safe_float = safe_float


_IDLE_PCT_TABLE_RE = re.compile(
    r"^\|\s*Idle\s*%\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\|",
    re.IGNORECASE | re.MULTILINE,
)
_COMPUTE_PCT_TABLE_RE = re.compile(
    r"^\|\s*Compute\s*%\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\|",
    re.IGNORECASE | re.MULTILINE,
)
_EXPOSED_COMM_PCT_TABLE_RE = re.compile(
    r"^\|\s*Exposed\s+Communication\s*%\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\|",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_exec_summary_pct(md_path: Path, pattern: re.Pattern[str]) -> float | None:
    """Extract one percentage row from an ``analysis.md`` Executive Summary table.

    Args:
        md_path: Path to the ``analysis.md`` report.
        pattern: Row regex whose first group is the numeric percentage.

    Returns:
        The percentage, or ``None`` when the file or row is missing or
        unparseable, so callers skip their gate gracefully.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    match = pattern.search(text)
    if not match:
        return None

    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_idle_pct_from_analysis_md(md_path: Path) -> float | None:
    """Extract ``Idle %`` from an ``analysis.md`` Executive Summary table.

    Used by the idle gate.

    Args:
        md_path: Path to the ``analysis.md`` report.

    Returns:
        The idle percentage, or ``None`` when missing/unparseable so callers
        skip the gate gracefully.
    """
    return _extract_exec_summary_pct(md_path, _IDLE_PCT_TABLE_RE)


def extract_compute_pct_from_analysis_md(md_path: Path) -> float | None:
    """Extract ``Compute %`` from an ``analysis.md`` Executive Summary table.

    Used by the low-compute gate.

    Args:
        md_path: Path to the ``analysis.md`` report.

    Returns:
        The compute percentage, or ``None`` when missing/unparseable so callers
        skip the gate gracefully.
    """
    return _extract_exec_summary_pct(md_path, _COMPUTE_PCT_TABLE_RE)


def extract_exposed_comm_pct_from_analysis_md(md_path: Path) -> float | None:
    """Extract ``Exposed Communication %`` from an ``analysis.md`` summary table.

    Context for the low-compute gate: it distinguishes a comm-dominated window
    from a host-bound one.

    Args:
        md_path: Path to the ``analysis.md`` report.

    Returns:
        The exposed-communication percentage, or ``None`` when
        missing/unparseable.
    """
    return _extract_exec_summary_pct(md_path, _EXPOSED_COMM_PCT_TABLE_RE)


# Source-function aggregation: group candidates sharing an AST-resolved
# (source_path, line, fn) triple; unparseable kernel_path falls back to per-kernel dispatch.

# Launcher path shapes: ``<path>(<line>): <func>`` (Python) or bare / ``<path>#L<line>`` (HIP).
_LAUNCHER_PATH_RE = re.compile(
    r"(?P<path>.+?)\((?P<line>\d+)\)\s*:\s*(?P<func>[A-Za-z_][A-Za-z0-9_]*)\s*$",
)
# Placeholders for unresolved Kernel Paths; must not survive parsing.
# ``not found`` is TraceLens' own sentinel for an unresolved launcher: it lands
# in ``other_metrics.json`` whenever ``_find_entry_point`` cannot locate the op
# in the call stack (every Synthetic Op), and the report agent copies it
# verbatim into the Kernel Path cell. Letting it through makes it a truthy
# ``source_file`` that silently skips the grep fallback downstream.
_LAUNCHER_PATH_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "—",
        "–",
        "n/a",
        "none",
        "not found",
        "not_found",
        "notfound",
        "null",
        "tbd",
        "unknown",
    }
)


def _launcher_frame_from_dict(obj: dict) -> str | None:
    """Pull the first ``<path>(<line>): <func>`` frame out of a TraceLens launcher dict.

    The dict shape is ``{'entry_point': '<frame>', 'wrappers': "[<frame>, ...]"}``.
    Prefer ``entry_point``; fall back to the first parseable ``wrappers`` frame.
    Returns ``None`` when no frame matches the launcher shape.
    """
    if not isinstance(obj, dict):
        return None
    entry = obj.get("entry_point")
    if isinstance(entry, str) and _LAUNCHER_PATH_RE.match(entry.strip()):
        return entry.strip()
    wrappers = obj.get("wrappers")
    if isinstance(wrappers, str):
        try:
            wrappers = _safe_literal_eval(wrappers)
        except _LITERAL_EVAL_ERRORS:
            wrappers = []
    if isinstance(wrappers, (list, tuple)):
        for frame in wrappers:
            if isinstance(frame, str) and _LAUNCHER_PATH_RE.match(frame.strip()):
                return frame.strip()
    return None


def _parse_launcher_path(kernel_path: str) -> tuple[str, int | None, str | None]:
    """Parse a TraceLens kernel-path into its components.

    Accepts ``<path>(<line>): <func>``, ``<path>#L<line>``, or a bare path.
    Also accepts the newer TraceLens launcher *dict* (or its stringified repr)
    ``{'entry_point': '<frame>', 'wrappers': "[...]"}``, from which the first
    real source frame is extracted before parsing.

    Args:
        kernel_path: The kernel-path string (or launcher dict) to parse.

    Returns:
        A ``(path, line, function_name)`` tuple; placeholders and empty input
        return ``("", None, None)``.
    """
    if not kernel_path:
        return "", None, None
    # TraceLens may hand us the launcher as a dict (or stringified dict) whose
    # real frame lives under entry_point/wrappers; reduce it to that frame first.
    if isinstance(kernel_path, dict):
        frame = _launcher_frame_from_dict(kernel_path)
        if not frame:
            return "", None, None
        kernel_path = frame
    elif isinstance(kernel_path, str):
        stripped = kernel_path.strip()
        if stripped.startswith("{") and stripped.endswith("}") and "entry_point" in stripped:
            try:
                parsed_obj = _safe_literal_eval(stripped)
            except _LITERAL_EVAL_ERRORS:
                parsed_obj = None
            frame = _launcher_frame_from_dict(parsed_obj) if isinstance(parsed_obj, dict) else None
            if not frame:
                return "", None, None
            kernel_path = frame
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


# Launcher path → absolute source file resolver. Strategy (most-specific first):
# $HYPERLOOM_FRAMEWORK_SOURCE_ROOTS override, importlib find_spec, then this fallback table.
_FRAMEWORK_PKG_FALLBACK_ROOTS: dict[str, tuple[str, ...]] = {
    "aiter": ("/sgl-workspace/aiter",),
    "sglang": ("/sgl-workspace/sglang/python", "/sgl-workspace/sglang"),
    "vllm": (
        "/usr/local/lib/python3.12/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/opt/venv/lib/python3.10/site-packages",
        "/sgl-workspace/vllm",
    ),
    # atom fallback roots for CSV-only / static-analysis parses. Kept in sync
    # with the reusable-source roots elsewhere, pinned by test_framework_paths_units.py.
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
    """Parse ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` into a package-root map.

    The variable is a comma-separated list of ``pkg=/abs/parent`` entries.

    Returns:
        A ``{pkg: (root, ...)}`` mapping; unparseable entries are skipped.
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
    """Find the directory containing a package on the live ``sys.path``.

    Args:
        pkg: The importable package name.

    Returns:
        The directory containing ``pkg/``, or ``None`` when not importable.
    """
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        loc = list(spec.submodule_search_locations)[0]
        return os.path.dirname(loc)
    if spec.origin and spec.origin.endswith(".py"):
        return os.path.dirname(os.path.dirname(spec.origin))
    return None


def _resolve_launcher_to_abs_source(
    kernel_path: str,
) -> tuple[str, int | None, str | None] | None:
    """Resolve a TraceLens launcher-path to an absolute source file.

    Args:
        kernel_path: The TraceLens launcher-path to resolve.

    Returns:
        An ``(abs_file, line, function_name)`` tuple for an absolute path or a
        resolvable framework-relative file, else ``None``.
    """
    raw_path, line, func = _parse_launcher_path(kernel_path)
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        # Keep container-resident paths even when this analysis host cannot stat
        # them; the caller still needs the annotation split into separate fields.
        return raw_path, line, func
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
        # Validate the file actually hosts the launcher's function (guards sys.path shadowing).
        if func and abs_path.endswith(".py"):
            if _function_line_from_ast(Path(abs_path), func) is None:
                continue
        return abs_path, line, func
    return None


def _function_line_from_ast(path: Path, function_name: str) -> int | None:
    """Find the line number of a named function definition via AST.

    Args:
        path: The source file to parse.
        function_name: The function name to locate.

    Returns:
        The first matching ``def``/``async def`` line, or ``None`` when the
        file is unreadable, unparseable, or the function is absent.
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
    """Resolve a candidate's launcher path to a source-target triple.

    The AST-derived definition line overrides the reported call-site line when
    resolvable.

    Args:
        candidate: The candidate dict carrying launcher/source paths.
        source_root: Optional root to resolve relative paths against.

    Returns:
        A ``(source_path, definition_line, function_name)`` dict, or ``None``
        when the path is unparseable.
    """
    # Prefer verbatim tracelens_launcher_path so AST resolution survives _finalize_candidates'
    # source_file overwrite; fall back to source_file / kernel_path for non-TraceLens candidates.
    kernel_path = str(
        candidate.get("tracelens_launcher_path") or candidate.get("source_file") or candidate.get("kernel_path") or ""
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
        "source_path": str(source_path),
        "definition_line": definition_line,
        "function_name": function_name,
        "reported_path": raw_path,
        "reported_line": reported_line,
        "reported_func": reported_func,
        "ast_resolved": bool(
            reported_func
            and source_path.exists()
            and reported_func == function_name
            and reported_line != definition_line
        ),
    }


_NATIVE_SOURCE_SUFFIXES = (
    ".cu",
    ".cuh",
    ".hip",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".h",
    ".c",
)


def _is_native_source(path: str) -> bool:
    """True for C/C++/HIP/CUDA source files.

    Native sources have no Python AST to resolve a stable ``def`` line, so
    TraceLens reports the per-call call-site ``#L<line>``. Callers therefore
    drop the line/function key components for these files.

    Args:
        path: The source file path to classify.

    Returns:
        ``True`` for C/C++/HIP/CUDA source files.
    """
    return str(path).lower().endswith(_NATIVE_SOURCE_SUFFIXES)


def aggregate_by_source_function(
    candidates: list[dict[str, Any]],
    *,
    source_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Group TraceLens candidates into per-kernel ``task_group`` dicts.

    Groups are sorted by aggregate time (descending). Native symbols are
    normalized to their logical function before keying by operation and source,
    so template/shape instances merge but different operators stay separate.
    Python candidates key on the same versioned ``(kind, source, operation)``
    identity used by the bypass path. Each group carries ``task_group_id``,
    ``source_path``, ``definition_line``, ``function_name``, ``kernel_ids``,
    ``primary_kernel_id``, ``rows``, and ``aggregate_*`` fields.

    Args:
        candidates: The TraceLens candidate rows to group.
        source_root: Optional root to resolve relative source paths against.

    Returns:
        The task-group dicts. Unparseable candidates are left out for legacy
        per-kernel dispatch.
    """
    if not candidates:
        return []
    root: Path | None = None
    if source_root:
        root = Path(source_root).expanduser()
        if not root.is_dir():
            root = None

    # Versioned identity builder. Operation normalization keeps different
    # kernels in one source separate while template/shape instances of one
    # operator merge.
    groups: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        target = _resolve_source_target(cand, source_root=root)
        if target is None:
            continue
        operation = str(cand.get("name") or "").strip()
        src_norm = os.path.normpath(str(target["source_path"]))
        reported_source = src_norm
        function_name = str(target["function_name"])
        source_kind = "native" if _is_native_source(src_norm) else "py"
        identity = build_operator_identity(
            source_kind=source_kind,
            source_path=reported_source,
            operation=operation,
            function_name=function_name,
        )
        src_norm = str(identity["source_path"])
        norm_op = str(identity["operation"])
        key = operator_identity_key(
            source_kind=source_kind,
            source_path=reported_source,
            operation=operation,
            function_name=function_name,
        )
        legacy_keys = legacy_operator_identity_keys(
            source_kind=source_kind,
            source_path=reported_source,
            operation=operation,
            function_name=function_name,
        )
        bucket = groups.get(key)
        if bucket is None:
            bucket = {
                "task_group_id": "",  # filled below after sorting
                "task_group_key": key,
                "operator_identity": identity,
                "identity_route": "skill",
                "legacy_task_group_keys": legacy_keys,
                "operation": operation,
                "operation_key": norm_op,
                "source_path": src_norm,
                "definition_line": target["definition_line"],
                "function_name": target["function_name"],
                "ast_resolved": bool(target.get("ast_resolved")),
                "reported_path": target["reported_path"],
                "kernel_ids": [],
                "primary_kernel_id": "",
                "rows": [],
                "aggregate_duration_us": 0.0,
                "aggregate_call_count": 0,
                "aggregate_gpu_pct": 0.0,
                # Per-P-item prose (deduped by (rank, title)) so build_prompt renders every P-item.
                "all_pitem_prose": [],
                "_pitem_prose_seen": set(),  # popped before return
            }
            groups[key] = bucket
        else:
            bucket["legacy_task_group_keys"] = list(
                dict.fromkeys(
                    [
                        *(bucket.get("legacy_task_group_keys") or []),
                        *legacy_keys,
                    ]
                )
            )
        kid = str(cand.get("kernel_id") or "") or cand.get("name") or ""
        if kid and kid not in bucket["kernel_ids"]:
            bucket["kernel_ids"].append(kid)
        bucket["rows"].append(cand)
        # Collect P-item prose deduped by (rank, title).
        try:
            pitem_rank = int(cand.get("tracelens_pitem_rank") or 0)
        except (TypeError, ValueError):
            pitem_rank = 0
        pitem_title = str(cand.get("tracelens_pitem_title") or "")
        pitem_key = (pitem_rank, pitem_title)
        if pitem_key not in bucket["_pitem_prose_seen"]:
            bucket["_pitem_prose_seen"].add(pitem_key)
            bucket["all_pitem_prose"].append(
                {
                    "rank": pitem_rank,
                    "title": pitem_title,
                    "identification": str(cand.get("identification") or "").strip(),
                    "reasoning_for_slowdown": str(cand.get("reasoning_for_slowdown") or "").strip(),
                    "resolution": str(cand.get("resolution") or "").strip(),
                    "impact_low_ms": _safe_float(cand.get("impact_low_ms")),
                    "impact_low_e2e_pct": _safe_float(cand.get("impact_low_e2e_pct")),
                    "impact_high_ms": _safe_float(cand.get("impact_high_ms")),
                    "impact_high_e2e_pct": _safe_float(cand.get("impact_high_e2e_pct")),
                }
            )
        try:
            bucket["aggregate_duration_us"] += float(cand.get("duration_us") or 0.0)
        except (TypeError, ValueError):
            # Malformed metric value; skip this contribution.
            pass
        try:
            bucket["aggregate_call_count"] += int(cand.get("call_count") or 0)
        except (TypeError, ValueError):
            # Malformed metric value; skip this contribution.
            pass
        try:
            bucket["aggregate_gpu_pct"] += float(cand.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            # Malformed metric value; skip this contribution.
            pass

    ordered = sorted(
        groups.values(),
        key=lambda g: g["aggregate_duration_us"],
        reverse=True,
    )
    for idx, group in enumerate(ordered, start=1):
        group["task_group_id"] = f"tg{idx:03d}"
        # Heaviest row (by duration) becomes primary; the rest are supplementary.
        group["rows"].sort(
            key=lambda r: float(r.get("duration_us") or 0.0),
            reverse=True,
        )
        if group["rows"]:
            primary = group["rows"][0]
            group["primary_kernel_id"] = str(primary.get("kernel_id") or primary.get("name") or "")
        group["aggregate_duration_us"] = round(group["aggregate_duration_us"], 3)
        group["aggregate_gpu_pct"] = round(group["aggregate_gpu_pct"], 3)
        # Sort prose by rank (P1 first); drop entirely-empty entries.
        group["all_pitem_prose"].sort(key=lambda e: (e["rank"], e["title"]))
        group["all_pitem_prose"] = [
            e
            for e in group["all_pitem_prose"]
            if e["rank"]
            or e["identification"]
            or e["reasoning_for_slowdown"]
            or e["resolution"]
            or e["impact_low_ms"]
            or e["impact_high_ms"]
        ]
        # ``_pitem_prose_seen`` is a set (not JSON-serializable); pop before return.
        group.pop("_pitem_prose_seen", None)
        group["shape_cases"] = build_task_group_shape_cases(group)
    return ordered


__all__ = [
    "TraceLensSkillRunResult",
    "UPSTREAM_CATEGORY_TO_GEAK",
    "_function_line_from_ast",
    "_parse_launcher_path",
    "aggregate_by_source_function",
    "build_orchestrator_prompt",
    "discover_capture_folder",
    "extract_compute_pct_from_analysis_md",
    "extract_exposed_comm_pct_from_analysis_md",
    "extract_idle_pct_from_analysis_md",
    "infer_analysis_mode",
    "normalize_upstream_category",
    "run_tracelens_skill",
    "write_local_cmd_prefix",
]
