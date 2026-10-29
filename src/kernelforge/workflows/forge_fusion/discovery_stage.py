# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forge-fuse discovery stage: locate the fusion recipes a diagnosed decode trace calls for."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .anchor import AnchorReport, AnchorResolutionError, KernelAnchor, discover_anchored_recipes, resolve_anchor
from .discover import DiscoverySession, DiscoveryTarget, discover_recipes, registered_agent_llm_fn
from .errors import FuseUsageError
from .llm_failure import LlmUnavailableError
from .locate import FrameworkInstall, _framework_repo_root, build_recipes, resolve_framework_source_file
from .report import ANCHOR_REPORT_NAME, ANCHOR_RESOLVED_VERDICT, LLM_UNAVAILABLE_VERDICT, write_anchor_report
from .shapes import resolve_decode_shapes

log = logging.getLogger("forge_fusion")


def _discovery_root(source_file: str, framework_root: str) -> str:
    """The directory a discovery session runs in, and repo scope searches within."""
    return (
        _framework_repo_root(source_file, framework_root)
        or framework_root
        or str(Path(source_file).parent if source_file else Path.cwd())
    )


def _resolve_discover_mode(fuse_kernel: str, discover_mode: str) -> str:
    """The discovery mode this run uses, which naming a kernel to fuse around decides."""
    if fuse_kernel:
        return "anchored"
    if discover_mode == "anchored":
        raise FuseUsageError("--discover anchored requires --fuse-kernel to name the kernel to fuse around.")
    return discover_mode


@dataclass
class _Discovery:
    """What the discovery stage located, and why it located nothing when it did not."""

    recipes: list
    anchor_report: AnchorReport | None = None
    llm_error: LlmUnavailableError | None = None


@dataclass(frozen=True)
class _DiscoveryInputs:
    """What a discovery stage works from: the trace and its diagnosis, the model, the install, scope and agent."""

    trace_path: str
    model_path: str
    model_type: str
    install: FrameworkInstall
    decode_batch: int
    attn_tp_size: int
    diagnosis: Any
    repo_scope: bool
    out: Path
    agent_factory: Any

    def resolve_source(self) -> tuple[dict[str, Any], str]:
        """The decode shapes and the framework source file the model resolves to."""
        shapes = resolve_decode_shapes(self.model_path, decode_batch=self.decode_batch, attn_tp_size=self.attn_tp_size)
        source_file, _source_note = resolve_framework_source_file(
            self.model_path,
            self.install.framework,
            framework_root=self.install.framework_root,
            model_type=self.model_type,
        )
        return shapes, source_file


def _discover_anchored(inputs: _DiscoveryInputs, fuse_kernel: str, dry_run: bool) -> _Discovery:
    """Resolve the named kernel in the trace, then discover fusions anchored on it."""
    out = inputs.out
    try:
        anchor_report = resolve_anchor(inputs.trace_path, KernelAnchor(name=fuse_kernel))
    except AnchorResolutionError as exc:
        raise FuseUsageError(str(exc)) from exc
    write_anchor_report(anchor_report, out)
    log.info(
        "anchor: %s | %d launches, %.1fus, %.2f%% of kernel time | %s (%.1f%%)",
        anchor_report.category,
        anchor_report.occurrences,
        anchor_report.total_us,
        anchor_report.share * 100,
        anchor_report.signature,
        anchor_report.consistency * 100,
    )
    for text in anchor_report.warnings:
        log.warning("anchor: %s", text)
    # Ahead of the dry-run exit: resolution is the whole point of a dry run here, so the operator sees which
    # source the launches were selected against before an agent or a GPU is paid for.
    shapes, source_file = inputs.resolve_source()
    discovery_root = _discovery_root(source_file, inputs.install.framework_root)
    if dry_run:
        log.info("dry run: anchor resolved to %s, skipping discovery", out / ANCHOR_REPORT_NAME)
        return _Discovery([], anchor_report)
    try:
        discovery_agent = inputs.agent_factory()
        recipes = discover_anchored_recipes(
            DiscoveryTarget(
                model_type=inputs.model_type,
                install=inputs.install,
                source_file=source_file,
                shapes=shapes,
                repo_scope=inputs.repo_scope,
                repo_root=discovery_root,
            ),
            anchor_report,
            category_shares=inputs.diagnosis.category_shares,
            llm_fn=registered_agent_llm_fn(
                discovery_agent,
                DiscoverySession(
                    model=discovery_agent.runtime.model,
                    workdir=discovery_root,
                    # Discovery is read-only, so the file it is shown is snapshotted
                    # and restored. Under repo scope no file is shown and the whole
                    # tree is off limits, so the session's own read-only spec
                    # carries that instead.
                    protected_files=[source_file] if source_file else [],
                    log_path=str(out / "discovery_llm.txt"),
                ),
            ),
        )
    except LlmUnavailableError as exc:
        log.error(
            "anchored discovery could not reach the LLM (%s after %d attempt(s)): %s",
            exc.kind,
            exc.attempts,
            exc,
        )
        return _Discovery([], anchor_report, exc)
    return _Discovery(recipes, anchor_report)


def _discover_by_llm(inputs: _DiscoveryInputs) -> _Discovery:
    """Let the model read the launch-bound profile and the real source and propose chains itself.

    Not capped to the template library, unlike ``patterns``.
    """
    out = inputs.out
    shapes, source_file = inputs.resolve_source()
    try:
        discovery_agent = inputs.agent_factory()
        discovery_workdir = _discovery_root(source_file, inputs.install.framework_root)
        recipes = discover_recipes(
            inputs.diagnosis,
            DiscoveryTarget(
                model_type=inputs.model_type,
                # The same install build_recipes gets: each proposal is checked against THIS install's compile-pass
                # config, and that verdict rewrites the pattern id.
                install=inputs.install,
                source_file=source_file,
                shapes=shapes,
                repo_scope=inputs.repo_scope,
                repo_root=discovery_workdir,
            ),
            inputs.trace_path,
            llm_fn=registered_agent_llm_fn(
                discovery_agent,
                DiscoverySession(
                    model=discovery_agent.runtime.model,
                    workdir=discovery_workdir,
                    protected_files=[source_file] if source_file else [],
                    log_path=str(out / "discovery_llm.txt"),
                ),
            ),
        )
    except LlmUnavailableError as exc:
        # The model was never reached, so this run knows nothing about the kernel.
        log.error("discovery could not reach the LLM (%s after %d attempt(s)): %s", exc.kind, exc.attempts, exc)
        return _Discovery([], None, exc)
    log.info("discovery(llm) proposed %d fusion(s)", len(recipes))
    return _Discovery(recipes)


def _discover(inputs: _DiscoveryInputs, discover_mode: str, fuse_kernel: str, dry_run: bool) -> _Discovery:
    """Locate the run's fusion recipes the way ``discover_mode`` names."""
    if discover_mode == "anchored":
        return _discover_anchored(inputs, fuse_kernel, dry_run)
    if discover_mode == "llm":
        return _discover_by_llm(inputs)
    if inputs.repo_scope:
        # ``patterns`` matches a fixed template library against the trace and never asks a model anything, so there is
        # nothing in it that could explore a repository.
        raise FuseUsageError("--repo-scope requires --discover llm or --discover anchored")
    return _Discovery(
        build_recipes(
            inputs.diagnosis, model_path=inputs.model_path, install=inputs.install, decode_batch=inputs.decode_batch
        )
    )


def _log_located_recipes(discovery: _Discovery, *, dry_run: bool) -> None:
    """Say what discovery located, and when it located nothing, why -- which is not always a verdict."""
    if discovery.recipes:
        log.info(
            "located %d candidate recipe(s): %s",
            len(discovery.recipes),
            ", ".join(f"{r.pattern_id}({r.trigger_share:.2f})" for r in discovery.recipes),
        )
    elif discovery.llm_error is not None:
        log.error(
            "no fusion recipe located because the LLM was unreachable "
            "(verdict: %s) — this is NOT a no_opportunity result",
            LLM_UNAVAILABLE_VERDICT,
        )
    elif discovery.anchor_report is not None and dry_run:
        log.info(
            "anchor resolved, discovery not run (verdict: %s) — this is NOT a no_opportunity result",
            ANCHOR_RESOLVED_VERDICT,
        )
    else:
        log.info("no fusion recipe located (verdict: no_opportunity)")
