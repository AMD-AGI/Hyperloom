# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point for `kernelforge forge-fuse`."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import click

from kernelforge.llm.git import git
from kernelforge.option_groups import option_group

from . import __version__
from .author import (
    AUTHOR_RC_FAILED,
    AUTHOR_RC_SAFETY,
    AuthorAgent,
    AuthorBrief,
    AuthorWorkspace,
    build_multi_author_prompt,
    run_author,
)
from .campaign import (
    LOOP_CAMPAIGN_STATE,
    CampaignBrief,
    CampaignRuntime,
    CampaignWorkspace,
    _safe_artifact_id,
    fused_module_path,
    run_recipe_campaign,
)
from .anchor import AnchorReport, AnchorResolutionError, KernelAnchor, discover_anchored_recipes, resolve_anchor
from .diagnose import diagnose_trace
from .discover import DiscoverySession, DiscoveryTarget, discover_recipes, registered_agent_llm_fn
from .emit import FusionFiles, _git_tracks, export_artifacts, restore_exported_changes
from .gpu_arch import canon_arch, detect_arch
from .llm_failure import LlmUnavailableError
from .locate import FrameworkInstall, build_recipes, resolve_framework_source_file
from .loop import FusionAbort, LoopConfig, LoopResult, RecipePatch, run_fusion_loop
from .models import CompilePassOutcome, FusionArtifacts, Recipe, ValidationResult
from .shadow_repo import ensure_git_workspace
from .report import (
    ANCHOR_REPORT_NAME,
    ANCHOR_RESOLVED_VERDICT,
    LLM_UNAVAILABLE_VERDICT,
    LocatedFusions,
    ManifestSubject,
    RunOutcome,
    build_manifest,
    write_anchor_report,
    write_manifest,
)
from .shapes import load_model_config, resolve_decode_shapes
from .validate import DEFAULT_TARGET_SPEEDUP, HarnessKernelRunner, validate_recipe
from .agent_selection import _agent_timeout_sec, _campaign_agent_provider, _create_agent_backend
from .compile_pass_claim import ClaimExport, ClaimGate, ClaimServing, _run_single_compile_pass_claim
from .harness_authoring import (
    _author_baseline_harness,
    _author_harness_target,
    _finish_author_harness,
    _prepare_author_harness,
)
from .serving_gate import ServingSession, _run_serving_smoke, apply_serving_gate
from .source_snapshot import _discard_failed_attempt, _needs_discard, _snapshot_fusion_source

log = logging.getLogger("forge_fusion")

# Exit code for "the run never reached the model".
EXIT_LLM_UNAVAILABLE = 3
# Exit code for "infrastructure failure before fusion was attempted": no git workspace, harness could not be authored,
# etc.
EXIT_INFRASTRUCTURE_FAILURE = 4


def _author_module_dirs(source_files: list[str]) -> list[str]:
    """Directories in which the author may create new fused helper modules."""
    dirs: list[str] = []
    for source_file in source_files:
        if not source_file:
            continue
        parent = str(Path(source_file).parent)
        if parent not in dirs:
            dirs.append(parent)
    return dirs


def _discovery_root(source_file: str, framework_root: str) -> str:
    """The directory a discovery session runs in, and repo scope searches within."""
    return (
        _framework_repo_root(source_file, framework_root)
        or framework_root
        or str(Path(source_file).parent if source_file else Path.cwd())
    )


def _recipe_files(recipes) -> list[str]:
    """Every framework file the given recipes edit, order-stable and de-duplicated."""
    files: list[str] = []
    for recipe in recipes:
        for path in recipe.edit_files:
            if path and path not in files:
                files.append(path)
    return files


def _tracked_roots(repo_root: str, source_files: list[str]) -> list[str]:
    """The top-level trees the shadow repo indexes, as the author is told them."""
    if not repo_root:
        return []
    root = Path(repo_root).resolve()
    roots: list[str] = []
    for source_file in source_files:
        with contextlib.suppress(OSError, ValueError):
            rel = Path(source_file).resolve().relative_to(root)
            if not rel.parts:
                continue
            entry = str(root / rel.parts[0])
            if entry not in roots:
                roots.append(entry)
    return roots


def _author_rc_after_harness(rc: int, *, harness_ok: bool) -> int:
    """Fold a harness-finalization failure into the author's return code."""
    if harness_ok or rc == AUTHOR_RC_SAFETY:
        return rc
    return AUTHOR_RC_FAILED


def _append_author_rejection(log_path: str, reason: str) -> None:
    """Append one content-free caller-side rejection to the author progress log."""
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"error: {reason}\n")
    except OSError:
        log.warning("could not append author rejection to %s", log_path)


def _setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging: file (all) + stderr (INFO+/DEBUG)."""
    level = logging.DEBUG if verbose else logging.INFO
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root = logging.getLogger("forge_fusion")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)


def _verdict_override(
    llm_error: Optional[LlmUnavailableError],
    anchor_report: Optional[AnchorReport],
    dry_run: bool,
) -> str:
    """Name the outcomes that are not a judgement about the kernel.

    A run that never asked the model must not report ``no_opportunity``: that reads
    as "there is nothing to fuse here" when nothing was ever looked at.
    """
    if llm_error is not None:
        return LLM_UNAVAILABLE_VERDICT
    if anchor_report is not None and dry_run:
        return ANCHOR_RESOLVED_VERDICT
    return ""


def _require_run_options(trace_path: str, model_path: str, framework: str, output_dir: str) -> None:
    """Refuse a run that is missing an option it cannot proceed without."""
    missing = [
        name
        for name, value in (
            ("--trace", trace_path),
            ("--model-path", model_path),
            ("--framework", framework),
            ("--output-dir", output_dir),
        )
        if not value
    ]
    if missing:
        raise click.UsageError(f"Missing option(s): {', '.join(missing)}.")


def _resolve_discover_mode(fuse_kernel: str, discover_mode: str) -> str:
    """The discovery mode this run uses, which naming a kernel to fuse around decides."""
    if fuse_kernel:
        return "anchored"
    if discover_mode == "anchored":
        raise click.UsageError("--discover anchored requires --fuse-kernel to name the kernel to fuse around.")
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
        raise click.UsageError(str(exc)) from exc
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


@dataclass(frozen=True)
class _FusionRun:
    """The run-wide context the authoring paths share: where it writes, what it serves, who authors, the bar."""

    out: Path
    repo_root: str
    pristine_dir: str
    framework_root: str
    repo_scope: bool
    session: ServingSession
    author: AuthorAgent
    gpu_arch: str
    target_speedup: float

    def claim_serving(self) -> ClaimServing:
        """Where this run's compile-pass claims serve their A/B arms."""
        session = self.session
        return ClaimServing(
            session.framework, self.framework_root, session.model_path, session.gpu, session.isl, session.osl
        )


@dataclass(frozen=True)
class _LoopPlan:
    """How a fusion loop runs: whether it authors, how many recipes it may try (0 = all), whether it combines."""

    author: bool
    max_recipes: int = 0
    combine: bool = False


def _run_author_only(authored: list, run: _FusionRun, ab_hint: str) -> int:
    """Author the fusion and stop there, for a run that asked for no validation."""
    out, repo_root = run.out, run.repo_root
    harness_path = str(out / "kernel_harness.py")
    author_harness_path = _author_harness_target(repo_root, out)
    ready, harness_error, harness_fatal = _prepare_author_harness(author_harness_path)
    if not ready:
        _append_author_rejection(str(out / "author.log"), harness_error)
        log.error("author harness preparation failed: %s", harness_error)
        return AUTHOR_RC_SAFETY if harness_fatal else AUTHOR_RC_FAILED
    prompt_harness_path = author_harness_path or harness_path
    # Every call-site file, not just the primary: the author transaction treats this list as its exact
    # write allowlist, so a multi-file fusion whose second file is missing here cannot be delivered.
    author_sources = _recipe_files(authored)
    prompt = build_multi_author_prompt(
        [r.to_dict() for r in authored],
        AuthorBrief(
            framework=run.session.framework,
            ab_hint=ab_hint,
            target_speedup=run.target_speedup,
            harness_path=prompt_harness_path,
            gpu_arch=run.gpu_arch,
            model_path=run.session.model_path,
        ),
    )
    (out / "author_prompt.md").write_text(prompt, encoding="utf-8")
    rc = run_author(
        prompt,
        AuthorWorkspace(
            workdir=repo_root or ".",
            log_path=str(out / "author.log"),
            target_files=[*author_sources, prompt_harness_path],
            new_module_dirs=_author_module_dirs(author_sources),
        ),
        dataclasses.replace(run.author, timeout_s=_agent_timeout_sec()),
    )
    harness_ok, harness_error = _finish_author_harness(author_harness_path, harness_path, author_ok=rc == 0)
    if not harness_ok:
        rc = _author_rc_after_harness(rc, harness_ok=harness_ok)
        _append_author_rejection(str(out / "author.log"), harness_error)
        log.error("author harness finalization failed: %s", harness_error)
    return rc


def _echo_run_summary(manifest: dict[str, Any], path: Path, recipes: list, compile_pass_outcome, validation) -> None:
    """Print the one-line JSON summary a caller reads instead of the manifest.

    A compile_pass run has no kernel-level ValidationResult, so ITS verdict is
    reported rather than a null that reads as "no validation ran".
    """
    click.echo(
        json.dumps(
            {
                "verdict": manifest["verdict"],
                "manifest": str(path),
                "patterns": [r.pattern_id for r in recipes],
                "speedup": (
                    compile_pass_outcome.speedup
                    if compile_pass_outcome is not None
                    else (validation.kernel_speedup if validation else None)
                ),
                "kept": (
                    compile_pass_outcome.kept
                    if compile_pass_outcome is not None
                    else (validation.kept if validation else None)
                ),
                "error": manifest["error"],
                "agent_backend": manifest.get("agent_backend"),
                "agent_model": manifest.get("agent_model"),
                "agent_sandbox_mode": manifest.get("agent_sandbox_mode"),
            }
        )
    )


def _exit_code_for_run(llm_error, loop_result) -> None:
    """Leave non-zero when the run failed for a reason an exit-code watcher must see.

    The manifest is the contract, but a run that never reached the model, or one
    whose pipeline never had a chance to fuse anything, must also be visible to
    anything that only watches exit codes.
    """
    if llm_error is not None:
        raise SystemExit(EXIT_LLM_UNAVAILABLE)
    if (
        loop_result is not None
        and not loop_result.kept
        and loop_result.termination_reason in ("no_git_workspace", "harness_author_failed", "serving_unconfirmed")
    ):
        raise SystemExit(EXIT_INFRASTRUCTURE_FAILURE)


def _settle_single_patch_workspace(run: _FusionRun, top_recipe, outcome: RunOutcome, exported_ok: bool):
    """Export what a single-patch run produced and put the framework tree back as it was.

    None of this runs for the other two paths. The multi-patch nomination already
    exported each sibling and restored the tree to base inside its own
    transaction, so exporting again would diff the wrong (reset) tree and file a
    bogus discard; a compile_pass claim likewise runs inside its own restore
    transaction and authors no modules.
    """
    artifacts = outcome.artifacts
    if outcome.patches is not None or outcome.compile_pass is not None:
        return artifacts
    repo_root, out, pristine_dir = run.repo_root, run.out, run.pristine_dir
    # Only export a patch when the run produced a USABLE fusion (validate path: kernel parity + speedup AND
    # serving survived).
    if repo_root and exported_ok:
        artifacts = export_artifacts(
            repo_root,
            FusionFiles(top_recipe.source_file, extra_files=top_recipe.extra_files, repo_scope=run.repo_scope),
            out / "fusion.patch",
            pristine_dir=pristine_dir,
        )
    # The exported patch is taken back out of the framework.
    if repo_root and artifacts and artifacts.patch:
        restore_exported_changes(repo_root, artifacts, pristine_dir=pristine_dir)
    if repo_root and pristine_dir and _needs_discard(exported_ok, artifacts):
        # Nothing usable came out, so leave the framework exactly as found rather than carrying unvalidated code
        # into whatever runs next.
        _discard_failed_attempt(
            repo_root,
            top_recipe.source_file,
            out,
            pristine_dir,
            extra_files=top_recipe.extra_files,
        )
    return artifacts


@dataclass(frozen=True)
class HarnessNoiseProbe:
    """The hidden diagnostic that repeats one kernel-validation harness and reports its variance."""

    harness_noise_path: str
    harness_noise_repeat: int
    harness_noise_env: tuple[str, ...]


@dataclass(frozen=True)
class DecodeSampling:
    """The decode batch shapes are taken at, and the decode steps the trace captured."""

    decode_batch: int
    decode_steps: int


@dataclass(frozen=True)
class DiscoveryChoice:
    """How recipes are discovered, and the kernel an anchored discovery fuses around."""

    discover_mode: str
    fuse_kernel: str


@dataclass(frozen=True)
class RunStages:
    """Which stages run: dry run, authoring, validation, and whether every confirmed pattern is fused at once."""

    dry_run: bool
    author: bool
    validate: bool
    fuse_all_confirmed: bool


@dataclass(frozen=True)
class FuseAgentOptions:
    """The agent provider, sandbox, model and turn cap for discovery and authoring."""

    agent_backend: str
    agent_sandbox_mode: str
    llm_model: Optional[str]
    max_turns: int


@dataclass(frozen=True)
class SmokeOptions:
    """The A/B decode lengths, and the extra serving args the smoke launch needs."""

    ab_isl: int
    ab_osl: int
    server_extra: str


@dataclass(frozen=True)
class ServingLayout:
    """How the session shards and sizes the model: TP, attention TP, KV block size and max model length."""

    tp: int
    attn_tp_size: int
    block_size: int
    max_model_len: int


@dataclass(frozen=True)
class FuseRequest:
    """Every option of one ``forge-fuse`` invocation, grouped by concern."""

    trace_path: str
    model_path: str
    framework: str
    output_dir: str
    noise: HarnessNoiseProbe
    framework_root: str
    repo_scope: bool
    decode: DecodeSampling
    discovery: DiscoveryChoice
    stages: RunStages
    gpu: str
    agent: FuseAgentOptions
    max_recipes: int
    smoke: SmokeOptions
    gpu_arch: str
    layout: ServingLayout
    verbose: bool


@click.command("forge-fuse")
@click.version_option(version=__version__)
@option_group(
    "request",
    FuseRequest,
    click.option(
        "--trace",
        "trace_path",
        default="",
        type=click.Path(),
        help="Decode kineto trace (*.trace.json[.gz]), captured with CUDA graphs disabled.",
    ),
    click.option("--model-path", default="", help="Path to the model directory (must contain config.json)."),
    click.option(
        "--framework",
        default=None,
        type=click.Choice(["sglang", "vllm", "vllm-aiter"]),
        help="Target inference framework.",
    ),
    click.option(
        "--output-dir",
        default="",
        type=click.Path(),
        help="Output directory for the manifest + logs.",
    ),
    option_group(
        "noise",
        HarnessNoiseProbe,
        click.option(
            "--harness-noise",
            "harness_noise_path",
            default="",
            hidden=True,
            help="Diagnostic: repeat one kernel-validation harness and report its variance.",
        ),
        click.option(
            "--harness-noise-repeat",
            default=20,
            type=int,
            hidden=True,
            help="Diagnostic: how many times to repeat the harness.",
        ),
        click.option(
            "--harness-noise-env",
            "harness_noise_env",
            multiple=True,
            hidden=True,
            help="Diagnostic: env flag to set for each harness run (repeatable).",
        ),
    ),
    click.option(
        "--framework-root",
        default="",
        help="Explicit framework source root (else auto-detect the installed package).",
    ),
    click.option(
        "--repo-scope/--no-repo-scope",
        "repo_scope",
        default=False,
        help="Give discovery and authoring the whole framework repository instead of one "
        "resolved model file. Discovery embeds no source and explores the tree with its own "
        "read/search tools, may return a fusion whose call sites span several files, and "
        "authoring may edit all of them. Use when the chain is not in the arch-class model "
        "file and you do not want to name its location.",
    ),
    option_group(
        "decode",
        DecodeSampling,
        click.option("--decode-batch", default=16, type=int, help="Representative decode batch size (T) for shapes."),
        click.option(
            "--decode-steps",
            default=0,
            type=int,
            help="Decode steps captured in the trace (to normalize kernels/step).",
        ),
    ),
    option_group(
        "discovery",
        DiscoveryChoice,
        click.option(
            "--discover",
            "discover_mode",
            type=click.Choice(["patterns", "llm", "anchored"]),
            default="patterns",
            help="Recipe discovery: 'patterns' (template library), 'llm' (LLM reads trace+source, autonomous), "
            "or 'anchored' (fuse around the kernel named by --fuse-kernel).",
        ),
        click.option(
            "--fuse-kernel",
            "fuse_kernel",
            default="",
            help="Full GPU kernel name, exactly as the trace spells it. Fusion is then built around "
            "this kernel and its trace neighbours instead of a ranked guess; implies --discover anchored.",
        ),
    ),
    option_group(
        "stages",
        RunStages,
        click.option(
            "--dry-run",
            is_flag=True,
            help="Diagnose + locate only; emit manifest with recipe skeleton (no author/validate).",
        ),
        click.option("--author/--no-author", default=True, help="Author the fused kernel via the LLM (non-dry-run)."),
        click.option("--validate/--no-validate", default=True, help="Run the A/B decode validation (non-dry-run)."),
        click.option(
            "--fuse-all-confirmed",
            is_flag=True,
            help="Author ALL source-confirmed patterns together (not just the top), and "
            "A/B all their flags. A compile-pass candidate cannot be authored with "
            "them, so it is claimed alone and the rest wait for a later round.",
        ),
    ),
    click.option("--gpu", default="0", help="HIP device id for author + A/B."),
    option_group(
        "agent",
        FuseAgentOptions,
        click.option(
            "--agent-backend",
            type=click.Choice(["auto", "claude", "codex"]),
            default="auto",
            show_default=True,
            help="Registered Agent provider for discovery and authoring.",
        ),
        click.option(
            "--agent-sandbox-mode",
            type=click.Choice(["workspace-write", "read-only", "bypass"]),
            envvar="FORGE_AGENT_SANDBOX_MODE",
            default="workspace-write",
            show_default=True,
            help="Agent runtime sandbox. Use bypass only when an external boundary already enforces isolation.",
        ),
        click.option(
            "--model",
            "llm_model",
            default=None,
            help="Agent model. Explicit value wins; otherwise uses provider-specific "
            "$CODEX_MODEL/$CLAUDE_MODEL, then the registered provider default.",
        ),
        click.option("--max-turns", default=100, type=int, help="Max authoring turns."),
    ),
    click.option(
        "--max-recipes",
        default=0,
        type=int,
        help="Cap how many ranked recipes to try. 0 means uncapped: every discovered recipe is considered.",
    ),
    option_group(
        "smoke",
        SmokeOptions,
        click.option("--ab-isl", default=512, type=int, help="A/B input length."),
        click.option("--ab-osl", default=128, type=int, help="A/B output length."),
        click.option(
            "--server-extra",
            default="",
            help=(
                "Extra serving args for the smoke launch (e.g. '--kv-cache-dtype fp8'). "
                "A model whose engine refuses to start without a flag can never reach the "
                "kernel the smoke exists to exercise."
            ),
        ),
    ),
    click.option(
        "--gpu-target",
        "gpu_arch",
        default="",
        help="Canonical GPU arch the author writes for (e.g. gfx950); auto-detected via rocminfo when omitted.",
    ),
    option_group(
        "layout",
        ServingLayout,
        click.option(
            "--tp",
            default=1,
            type=int,
            help="Tensor-parallel size for the serving smoke (must match the session).",
        ),
        click.option(
            "--attn-tp",
            "attn_tp_size",
            default=1,
            type=int,
            help=(
                "Attention TP shard used to stamp n_local_heads / n_local_groups "
                "(default 1; keep 1 under DP-attention even when --tp > 1)."
            ),
        ),
        click.option(
            "--block-size",
            "block_size",
            default=0,
            type=int,
            help="vLLM KV --block-size for the serving smoke (0=omit). Required for "
            "sparse-attention models that reject the default block size.",
        ),
        click.option(
            "--max-model-len",
            "max_model_len",
            default=0,
            type=int,
            help="Serving-smoke max model / context length (0 uses the smoke default 4096).",
        ),
    ),
    click.option("--verbose", "-v", is_flag=True, help="Verbose logging."),
)
def run(request: FuseRequest) -> None:
    """Diagnose a decode trace and locate a fusion opportunity for the model."""
    noise = request.noise
    if noise.harness_noise_path:
        click.echo(
            json.dumps(
                measure_harness_noise(
                    harness=noise.harness_noise_path,
                    repeat=noise.harness_noise_repeat,
                    gpu=request.gpu,
                    env_flags=noise.harness_noise_env,
                ),
                indent=2,
            )
        )
        return

    trace_path, model_path, framework = request.trace_path, request.model_path, request.framework
    stages, agent, smoke, layout = request.stages, request.agent, request.smoke, request.layout
    _require_run_options(trace_path, model_path, framework, request.output_dir)
    fuse_kernel = request.discovery.fuse_kernel.strip()
    discover_mode = _resolve_discover_mode(fuse_kernel, request.discovery.discover_mode)

    out = Path(request.output_dir)
    _setup_logging(out, request.verbose)

    log.info("forge-fuse %s | framework=%s model=%s", __version__, framework, model_path)
    selected_agent = None
    llm_model = agent.llm_model

    def require_agent_backend():
        """Lazily create and cache the one backend shared by both Agent stages."""
        nonlocal selected_agent, llm_model
        if selected_agent is not None:
            return selected_agent
        try:
            selected_agent = _create_agent_backend(
                agent.agent_backend,
                llm_model,
                agent.agent_sandbox_mode,
            )
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(f"agent backend configuration failed: {type(exc).__name__}: {exc}") from exc
        llm_model = selected_agent.runtime.model
        log.info(
            "selected Agent backend=%s model=%s",
            selected_agent.name,
            selected_agent.runtime.model,
        )
        return selected_agent

    decode_batch = request.decode.decode_batch
    diagnosis = diagnose_trace(trace_path, decode_steps=request.decode.decode_steps, decode_batch=decode_batch)
    log.info(
        "diagnosis: candidate=%s launch_bound_share=%.3f predicted_e2e_gain=%.3f busy_of_wall=%s reason=%s",
        diagnosis.is_candidate,
        diagnosis.launch_bound_share,
        diagnosis.predicted_e2e_gain,
        diagnosis.busy_fraction_of_wall,
        diagnosis.reason,
    )

    model_type = str(load_model_config(model_path).get("model_type") or "")
    if request.repo_scope and discover_mode == "patterns":
        # ``patterns`` matches a fixed template library against the trace and never asks a model anything, so there is
        # nothing in it that could explore a repository.
        raise click.UsageError("--repo-scope requires --discover llm or --discover anchored")
    install = FrameworkInstall(framework, request.framework_root)
    inputs = _DiscoveryInputs(
        trace_path=trace_path,
        model_path=model_path,
        model_type=model_type,
        install=install,
        decode_batch=decode_batch,
        attn_tp_size=layout.attn_tp_size,
        diagnosis=diagnosis,
        repo_scope=request.repo_scope,
        out=out,
        agent_factory=require_agent_backend,
    )
    if discover_mode == "anchored":
        discovery = _discover_anchored(inputs, fuse_kernel, stages.dry_run)
    elif discover_mode == "llm":
        discovery = _discover_by_llm(inputs)
    else:
        discovery = _Discovery(
            build_recipes(diagnosis, model_path=model_path, install=install, decode_batch=decode_batch)
        )
    recipes = discovery.recipes
    anchor_report = discovery.anchor_report
    llm_error = discovery.llm_error
    top_recipe = recipes[0] if recipes else None
    _log_located_recipes(discovery, dry_run=stages.dry_run)

    def publish(outcome: RunOutcome) -> tuple[dict[str, Any], Path]:
        """Write the run's manifest, as complete as the run has so far got.

        The aggregate is the only thing that points at a keeper, so it cannot wait for every
        campaign to return: a run killed in between would report REVERT while proven,
        already-smoked patches sat in the workspace. Called from ``on_keep`` too, it is never
        missing, only partial, and the end-of-run call overwrites it with the real loop /
        compile-pass / error fields before any exit.
        """
        manifest = build_manifest(
            ManifestSubject(framework=framework, model_path=model_path, model_type=model_type, diagnosis=diagnosis),
            LocatedFusions(
                recipe=top_recipe,
                candidates=recipes,
                anchor=anchor_report.to_dict() if anchor_report is not None else None,
            ),
            outcome,
        )
        if selected_agent is not None:
            manifest["agent_backend"] = selected_agent.name
            manifest["agent_model"] = selected_agent.runtime.model
            manifest["agent_sandbox_mode"] = selected_agent.runtime.sandbox_mode
        return manifest, write_manifest(manifest, out)

    validation = None
    artifacts = None
    loop_manifest = None
    compile_pass_outcome: Optional[CompilePassOutcome] = None
    loop_result = None
    # Multi-patch nomination outputs: patches_out is the list of sibling envelopes (None on the combine / single-patch
    # escape hatch so build_manifest omits the key and the legacy shape stays byte-identical); nomination_summary is
    # the round's counts.
    patches_out: Optional[list[dict[str, Any]]] = None
    nomination_summary: Optional[dict[str, Any]] = None

    split = _split_nomination(recipes, top_recipe, stages.fuse_all_confirmed)
    claims, deferred = split.claims, split.deferred
    top_recipe = split.top_recipe
    fuse_all_confirmed = split.fuse_all_confirmed
    multi_patch = split.multi_patch

    if not stages.dry_run and top_recipe is not None:
        repo_root = _framework_repo_root(top_recipe.source_file, request.framework_root)
        # Snapshot the pristine model source BEFORE authoring so a patch can be produced even when the framework is a
        # non-git pip install (git diff would otherwise be empty -> patch=null -> integrate skips the KEPT fusion).
        pristine_dir = _snapshot_fusion_source(
            repo_root, top_recipe.source_file, out, extra_files=top_recipe.extra_files
        )
        # combine folds every recipe into one unit; multi-patch authors each non-claim recipe as its own sibling and
        # runs any claims separately.
        authored = recipes if fuse_all_confirmed else (deferred if multi_patch else [top_recipe])
        ab_hint = (
            f"forge-fuse validates at the KERNEL level (compile + SNR parity + "
            f"microbench speedup), decode batch {decode_batch} isl {smoke.ab_isl} osl {smoke.ab_osl}"
        )
        target_speedup = DEFAULT_TARGET_SPEEDUP
        run_arch = _run_target_arch(request.gpu_arch)
        fusion_run = _FusionRun(
            out=out,
            repo_root=repo_root,
            pristine_dir=pristine_dir,
            framework_root=request.framework_root,
            repo_scope=request.repo_scope,
            session=ServingSession(
                framework,
                model_path,
                request.gpu,
                smoke.ab_isl,
                smoke.ab_osl,
                server_extra=smoke.server_extra,
                tp=layout.tp,
                block_size=layout.block_size,
                max_model_len=layout.max_model_len,
            ),
            author=AuthorAgent(
                backend=require_agent_backend, gpu=request.gpu, model=llm_model, max_turns=agent.max_turns
            ),
            gpu_arch=run_arch,
            target_speedup=target_speedup,
        )

        exported_ok = False

        # The multi-patch nomination runs BOTH pipelines when a run discovered both kinds: each compile-pass claim
        # through its config A/B, every authored fusion through the kernel autoloop, and every keeper becomes an
        # independent sibling in patches[].
        if multi_patch and stages.validate:
            patches_out, compile_pass_outcome, loop_result, withheld = _run_multi_patch_nomination(
                split,
                fusion_run,
                _LoopPlan(author=stages.author, max_recipes=request.max_recipes),
                _campaign_runtime(fusion_run, agent, llm_model),
                publish,
            )
            if loop_result is not None:
                validation = loop_result.best
                loop_manifest = loop_result.to_dict()
            # Strongest sibling fills the legacy singular ``artifacts`` slot so the combine-era salvage / timeout path
            # keeps finding a patch.
            if patches_out:
                strongest = patches_out[0]
                artifacts = FusionArtifacts(
                    patch=strongest.get("patch_path"),
                    repo_root=strongest.get("kernel_repo") or repo_root,
                )
                # Mirror the strongest patch under the legacy ``fusion.patch`` name.
                _mirror_legacy_patch(strongest.get("patch_path"), out)
            exported_ok = bool(patches_out)
            nomination_summary = {
                "candidates_seen": len(recipes),
                "resolved": len(claims) + len(authored),
                "selected": len(patches_out),
                # Resolved targets the lane ceiling never funded, so a starved round is not read as "everything ran
                # and kept nothing".
                "withheld": withheld,
            }
        elif top_recipe.candidate_kind == "compile_pass":
            compile_pass_outcome, artifacts = _run_single_compile_pass_claim(
                top_recipe,
                fusion_run.claim_serving(),
                ClaimGate(validate=stages.validate, target_speedup=target_speedup),
                ClaimExport(out, repo_root, pristine_dir),
            )
            exported_ok = compile_pass_outcome.kept
        elif stages.validate:
            # Validate-driven outer loop: per recipe, author -> kernel-validate -> serving-smoke with cross-attempt
            # experience injection; early-exit on the first result that is KEPT (kernel parity + speedup AND survives
            # serving).
            loop_result = _run_fusion_autoloop(
                authored,
                fusion_run,
                _LoopPlan(author=stages.author, combine=fuse_all_confirmed),
                _campaign_runtime(fusion_run, agent, llm_model),
            )
            validation = loop_result.best
            loop_manifest = loop_result.to_dict()
            exported_ok = loop_result.kept
            log.info(
                "fusion loop finished: kept=%s speedup=%s attempts=%d termination=%s",
                loop_result.kept,
                validation.kernel_speedup if validation else None,
                len(loop_result.history),
                loop_result.termination_reason,
            )
        elif stages.author:
            rc = _run_author_only(authored, fusion_run, ab_hint)
            exported_ok = rc == 0
            log.info("author finished rc=%s (no validation requested)", rc)

        artifacts = _settle_single_patch_workspace(
            fusion_run,
            top_recipe,
            RunOutcome(artifacts=artifacts, compile_pass=compile_pass_outcome, patches=patches_out),
            exported_ok,
        )

    manifest, path = publish(
        RunOutcome(
            patches=patches_out,
            validation=validation,
            artifacts=artifacts,
            loop=loop_manifest,
            compile_pass=compile_pass_outcome,
            verdict_override=_verdict_override(llm_error, anchor_report, stages.dry_run),
            error=(llm_error.to_dict() if llm_error is not None else None),
            nomination=nomination_summary,
        )
    )
    log.info("wrote manifest: %s (verdict=%s)", path, manifest["verdict"])
    _echo_run_summary(manifest, path, recipes, compile_pass_outcome, validation)
    _exit_code_for_run(llm_error, loop_result)


def _campaign_runtime(run: _FusionRun, agent: FuseAgentOptions, llm_model: Optional[str]) -> CampaignRuntime:
    """The runtime a forge-loop campaign is handed: this run's device and arch, and the agent it resolved."""
    return CampaignRuntime(
        gpu=run.session.gpu,
        gpu_target=run.gpu_arch,
        model=llm_model or "",
        agent_backend=_campaign_agent_provider(agent.agent_backend, llm_model),
        agent_sandbox_mode=agent.agent_sandbox_mode,
    )


def _publish_partial_nomination(
    publish,
    smoked: list[RecipePatch],
    patch: RecipePatch,
    *,
    out: Path,
    repo_root: str,
) -> None:
    """Record the keepers proved so far, so a kill after this one does not lose them.

    ``fusion_manifest.json`` is the only artifact that points at a keeper, so it is written
    after each keeper rather than once every campaign has returned. A wrapper killed in
    between would otherwise report REVERT with ``patch=null`` while smoked, already-published
    patches sat in the output dir.

    Args:
        publish: The run's manifest writer, or None on the paths that have no manifest.
        smoked: Siblings already past their serving smoke; ``patch`` is appended to it.
        patch: The sibling that just passed.
        out: The fusion output directory.
        repo_root: The framework checkout the patches apply to.
    """
    smoked.append(patch)
    if publish is None:
        return
    # None speedup sorts weakest, the same rule the loop's own patches[] is built on.
    smoked.sort(key=lambda p: p.micro_speedup if p.micro_speedup is not None else -1.0, reverse=True)
    best = smoked[0]
    _mirror_legacy_patch(best.patch_path, out)
    try:
        publish(
            RunOutcome(
                patches=[_recipe_patch_envelope(p, repo_root=repo_root) for p in smoked],
                artifacts=FusionArtifacts(patch=best.patch_path, repo_root=repo_root),
                loop={
                    "kept": True,
                    "best": {"kernel_speedup": best.micro_speedup},
                    "best_env_flag": best.env_flag,
                    "termination_reason": "in_progress",
                },
            )
        )
    except OSError as exc:
        # A manifest that could not be written is not a reason to drop a proven keeper; the
        # end-of-run write gets another chance at it.
        log.warning("could not publish the partial nomination: %s", exc)


def _recipe_patch_envelope(patch: RecipePatch, *, repo_root: str) -> dict[str, Any]:
    """One entry of the manifest ``patches[]`` for an authored fusion sibling."""
    return {
        "kernel_name": patch.kernel_name,
        "patch_path": patch.patch_path,
        "target_file": patch.source_file,
        "kernel_repo": repo_root,
        "snapshot_dir": patch.snapshot_dir,
        "base_commit": patch.base_commit,
        "micro_speedup": patch.micro_speedup,
        # The env flag that activates the fused path; the consumer sets it on the re-baseline server or the patch is
        # measured un-fused and REVERTED.
        "env_flag": patch.env_flag,
        "kind": "fusion",
    }


def _compile_pass_envelope(claim, outcome, artifacts, *, repo_root: str) -> dict[str, Any]:
    """One entry of the manifest ``patches[]`` for a claimed compile pass."""
    return {
        "kernel_name": claim.pattern_id,
        "patch_path": artifacts.patch if artifacts is not None else None,
        "target_file": claim.source_file,
        "kernel_repo": repo_root,
        "snapshot_dir": "",
        "base_commit": "",
        "micro_speedup": None,
        "serving_speedup": outcome.speedup if outcome is not None else None,
        "kind": "compile_pass",
    }


def _mirror_legacy_patch(patch_path: Optional[str], out: Path) -> None:
    """Copy the strongest sibling patch to the legacy ``out/fusion.patch`` name."""
    if not patch_path:
        return
    src = Path(patch_path)
    dst = out / "fusion.patch"
    try:
        if src.resolve() == dst.resolve():
            return
        if src.is_file():
            shutil.copy2(src, dst)
    except OSError as exc:
        log.warning("could not mirror %s to legacy fusion.patch: %s", patch_path, exc)


def _run_multi_patch_nomination(
    split: _NominationSplit,
    run: _FusionRun,
    plan: _LoopPlan,
    campaign: CampaignRuntime,
    publish=None,
) -> tuple[list[dict[str, Any]], Optional[CompilePassOutcome], Optional[LoopResult], int]:
    """Run BOTH pipelines and collect every keeper as an independent sibling; ``plan.max_recipes`` spans both."""
    claims, authored, max_recipes, repo_root = split.claims, split.deferred, plan.max_recipes, run.repo_root
    patches: list[dict[str, Any]] = []

    # One ceiling across both pipelines, spent in ``rank_recipes`` order: a claim is a deterministic flip, so an
    # authoring loop must not crowd it out.
    claims_budget = _recipe_ceiling(len(claims), max_recipes)
    remaining = max_recipes - claims_budget if max_recipes > 0 else len(authored)
    authored_budget = min(len(authored), max(0, remaining))
    withheld = (len(claims) - claims_budget) + (len(authored) - authored_budget)
    if withheld:
        log.info(
            "multi-patch: lane ceiling of %d target(s) withholds %d lower-ranked target(s)",
            max_recipes,
            withheld,
        )
    claims = claims[:claims_budget]
    # Sliced here: the autoloop reads a ceiling of 0 as uncapped, not as exhausted.
    authored = authored[:authored_budget]

    # Authored recipes first: the autoloop owns the shared git workspace, smokes each keeper, and returns siblings
    # strongest-first.
    loop_result: Optional[LoopResult] = None
    if authored:
        loop_result = _run_fusion_autoloop(
            authored,
            run,
            # multi-patch: one sibling per keeper
            dataclasses.replace(plan, max_recipes=authored_budget, combine=False),
            campaign,
            publish=publish,
        )
        for patch in loop_result.patches:
            patches.append(_recipe_patch_envelope(patch, repo_root=repo_root))
        log.info(
            "multi-patch: authored loop kept=%s siblings=%d termination=%s",
            loop_result.kept,
            len(loop_result.patches),
            loop_result.termination_reason,
        )

    # Then the claims, each its own transaction.
    kept_claims: list[tuple[float, dict[str, Any]]] = []
    # Report a claim outcome on the manifest's singular ``compile_pass`` slot even when nothing kept: a REJECTED claim
    # must still be visible as kept=False, not dropped to null (which reads as "no claim ran").
    reported_outcome: Optional[CompilePassOutcome] = None
    strongest_kept_speedup = -1.0
    for claim in claims:
        outcome, arts = _run_single_compile_pass_claim(
            claim,
            run.claim_serving(),
            ClaimGate(validate=True, target_speedup=run.target_speedup),
            ClaimExport(run.out, repo_root, run.pristine_dir, f"fusion_{_safe_artifact_id(claim.pattern_id)}.patch"),
        )
        if outcome.kept and arts is not None and arts.patch:
            env = _compile_pass_envelope(claim, outcome, arts, repo_root=repo_root)
            speedup = outcome.speedup if outcome.speedup is not None else -1.0
            kept_claims.append((speedup, env))
            if reported_outcome is None or not reported_outcome.kept or speedup > strongest_kept_speedup:
                reported_outcome = outcome
                strongest_kept_speedup = speedup
        elif reported_outcome is None or not reported_outcome.kept:
            # Only a rejected claim so far; keep the latest so the manifest is not null.
            reported_outcome = outcome
    kept_claims.sort(key=lambda item: item[0], reverse=True)
    patches.extend(env for _key, env in kept_claims)

    return patches, reported_outcome, loop_result, withheld


def _recipe_ceiling(discovered: int, max_recipes: int) -> int:
    """How many ranked recipes to try out of ``discovered``."""
    return min(discovered, max_recipes) if max_recipes > 0 else discovered


def _combined_recipe(recipes: list[Recipe]) -> Recipe:
    """Fold several confirmed recipes into ONE unit for the loop."""
    base = recipes[0]
    flags = list(dict.fromkeys(f for r in recipes for f in r.env_flag.split() if f))
    return Recipe(
        pattern_id="+".join(r.pattern_id for r in recipes),
        description="; ".join(r.description for r in recipes),
        env_flag=" ".join(flags),
        source_file=base.source_file,
        # The union, minus whichever file became the combined call site: every folded
        # recipe's files must stay tracked or combining would silently narrow the
        # edit scope to the first recipe's.
        extra_files=[path for path in _recipe_files(recipes) if path != base.source_file],
        source_hints=[h for r in recipes for h in r.source_hints],
        fusion_math="\n".join(f"[{r.pattern_id}] {r.fusion_math}" for r in recipes),
        eager_reference_hint="; ".join(r.eager_reference_hint for r in recipes),
        shapes=base.shapes,
        matched_categories=sorted({c for r in recipes for c in r.matched_categories}),
        trigger_share=max(r.trigger_share for r in recipes),
        rocm_native=any(r.rocm_native for r in recipes),
    )


def _run_target_arch(gpu_arch: str) -> str:
    """One GPU arch for the whole run.

    The author tunes for it, so a mismatch would have it writing for a chip the
    run is not on.
    """
    run_arch = canon_arch(gpu_arch) or canon_arch(detect_arch())
    if run_arch:
        log.info("target GPU arch: %s", run_arch)
    else:
        log.warning("GPU arch undetectable; the author will not be told a target ISA")
    return run_arch


def _open_shadow_workspace(repo_root: str, loop_recipes: list, out: Path, campaign_files: list[str]):
    """The git workspace the campaigns keep and revert in, or None when there is none.

    Every recipe's fused module is tracked at the baseline, not just the one
    about to run: the loop keeps with ``git add -u``, which cannot commit a file
    created mid-campaign.
    """
    shadow = ensure_git_workspace(
        repo_root,
        loop_recipes[0].source_file,
        git_dir=str(out / "shadow.git"),
        extra_paths=tuple(fused_module_path(r) for r in loop_recipes),
        # A fusion whose call sites span packages is keepable only if every one of
        # those packages is in the index.
        scope_files=campaign_files,
    )
    if shadow is None:
        log.error("no git workspace over %s: the forge-loop cannot keep or revert a candidate there", repo_root)
    return shadow


@dataclass
class _NominationSplit:
    """How a run's recipes divide between the two non-interchangeable pipelines.

    A claim and an authored kernel are validated and exported by different
    machinery (config A/B vs kernel parity + microbench), so a single UNIT
    cannot do both.
    """

    claims: list
    deferred: list
    top_recipe: Any
    fuse_all_confirmed: bool
    multi_patch: bool


def _split_nomination(recipes: list, top_recipe, fuse_all_confirmed: bool) -> _NominationSplit:
    """Divide the recipes by candidate kind, narrowing a combine run to its claim."""
    claims = [r for r in recipes if r.candidate_kind == "compile_pass"]
    deferred = [r for r in recipes if r.candidate_kind != "compile_pass"]
    if not (fuse_all_confirmed and claims and deferred):
        return _NominationSplit(claims, deferred, top_recipe, fuse_all_confirmed, not fuse_all_confirmed)
    log.info(
        "combine: claiming compile pass %s first; deferring %s",
        claims[0].pattern_id,
        ", ".join(r.pattern_id for r in deferred),
    )
    # Combine narrowed to the claim; NOT the multi-patch path.
    return _NominationSplit(claims, deferred, claims[0], False, False)


def _snapshot_recipe_sources(loop_recipes: list, repo_root: str, out: Path) -> dict[str, str]:
    """A pristine snapshot per recipe, which the multi-patch export diffs each sibling against."""
    snapshots: dict[str, str] = {}
    for recipe in loop_recipes:
        snap = _snapshot_fusion_source(
            repo_root,
            recipe.source_file,
            out,
            subdir=f".pristine_{_safe_artifact_id(recipe.pattern_id)}",
            extra_files=recipe.extra_files,
        )
        if snap:
            snapshots[recipe.pattern_id] = snap
    return snapshots


def _export_recipe_sibling(recipe, *, repo_root: str, out: Path, pristine: str, repo_scope: bool):
    """This recipe's own patch, scoped to its fused module, or None when there is nothing to export.

    Exported while the recipe's edits are still live in the shared tree -- the
    loop has not reset to base yet.
    """
    if not pristine:
        log.warning("no pristine snapshot for kept recipe %s; cannot export its sibling", recipe.pattern_id)
        return None
    arts = export_artifacts(
        repo_root,
        FusionFiles(
            recipe.source_file,
            extra_files=recipe.extra_files,
            fused_module=fused_module_path(recipe),
            repo_scope=repo_scope,
        ),
        out / f"fusion_{_safe_artifact_id(recipe.pattern_id)}.patch",
        pristine_dir=pristine,
    )
    if not (arts and arts.patch):
        log.warning("kept recipe %s produced no patch on export", recipe.pattern_id)
        return None
    return arts


def _reject_fusion_sibling(recipe, vr, arts, disposition: str) -> None:
    """Withdraw a sibling the serving smoke refused, and the patch that was exported for it.

    The run rejected it, so its exported patch must not outlive the decision: a
    later reader has no way to tell it apart from one that passed.
    """
    vr.kept = False
    vr.kernel_speedup = None
    if disposition == "serving_crash":
        vr.correctness_passed = False
    with contextlib.suppress(OSError):
        Path(arts.patch).unlink()
    log.warning("dropping fusion sibling %s from nomination (%s)", recipe.pattern_id, disposition)


def _run_fusion_autoloop(recipes, run: _FusionRun, plan: _LoopPlan, campaign: CampaignRuntime, publish=None):
    """Try each ranked recipe as one forge-loop campaign."""
    out, repo_root, pristine_dir, repo_scope = run.out, run.repo_root, run.pristine_dir, run.repo_scope
    session, author, combine = run.session, plan.author, plan.combine
    originals = {r.pattern_id: r for r in recipes}
    loop_recipes = [_combined_recipe(recipes)] if (combine and len(recipes) > 1) else recipes
    # Every file any recipe edits: the shadow index has to admit all of them up front, because it is built once and
    # shared by every campaign below.
    campaign_files = list(dict.fromkeys(_recipe_files(loop_recipes)))
    tracked_roots = _tracked_roots(repo_root, campaign_files)

    # Per-recipe pristine snapshots for the multi-patch export.
    multi_patch = not combine
    recipe_pristine = _snapshot_recipe_sources(loop_recipes, repo_root, out) if multi_patch and repo_root else {}

    keep_bar = run.target_speedup

    # Only the authoring path runs campaigns.
    shadow = None
    if author and loop_recipes:
        shadow = _open_shadow_workspace(repo_root, loop_recipes, out, campaign_files)
        if shadow is None:
            return LoopResult(kept=False, best=None, best_recipe=None, termination_reason="no_git_workspace")

    campaign_experiments: dict[str, str] = {}

    def _harness_path_for(recipe) -> str:
        return str(out / f"kernel_harness_{_safe_artifact_id(recipe.pattern_id)}.py")

    def campaign_fn(recipe, experience: str):
        if not author:
            return validate_existing_source(
                recipe,
                repo_root=repo_root,
                gpu=session.gpu,
                harness_path=_harness_path_for(recipe),
                target_speedup=keep_bar,
            )
        # A fresh campaign refuses to start where the previous one left state, and the loop anchors that state to the
        # workspace rather than to ``--experiments-dir``.
        shutil.rmtree(Path(shadow.root) / LOOP_CAMPAIGN_STATE, ignore_errors=True)
        # Score every recipe against the UNFUSED framework: the previous campaign's commits are otherwise still in the
        # tree, and the two changes get reported stacked as if they were one.
        if not shadow.reset_to_base():
            return ValidationResult(
                correctness_passed=False,
                max_abs_err=None,
                rtol=None,
                kernel_speedup=None,
                eager_us=None,
                fused_us=None,
                kept=False,
                note="CAMPAIGN FAILED: could not restore the unfused baseline",
                correctness_measured=False,
            )
        # After the reset, so the loop's anchor bench measures the unfused tree.
        harness_path = _harness_path_for(recipe)
        ready, reason = _author_baseline_harness(
            recipe,
            harness_path=harness_path,
            repo_root=repo_root,
            out=out,
            agent=run.author,
        )
        if not ready:
            # Not this recipe's failure: recording it as one would file a wrong lesson that the next recipe's campaign
            # is then prompted with.
            raise FusionAbort(f"no harness for {recipe.pattern_id}: {reason}")
        outcome = run_recipe_campaign(
            recipe,
            CampaignWorkspace(root=shadow.root, output_dir=str(out), git_env=shadow.env),
            CampaignBrief(
                harness_path=harness_path,
                experience=experience,
                fused_module=fused_module_path(recipe),
                repo_scope=repo_scope,
                tracked_roots=tracked_roots,
            ),
            campaign,
            target_speedup=keep_bar,
        )
        if outcome.experiment_id:
            campaign_experiments[recipe.pattern_id] = outcome.experiment_id
        return outcome.result

    # Every sibling that has passed its serving smoke so far, so a run killed mid-campaign still
    # publishes the ones it already proved.
    smoked: list[RecipePatch] = []

    def on_keep(recipe, vr):
        """Export the just-kept recipe's OWN sibling patch before the next reset."""
        if not multi_patch or not repo_root:
            return None
        pristine = recipe_pristine.get(recipe.pattern_id, "")
        arts = _export_recipe_sibling(recipe, repo_root=repo_root, out=out, pristine=pristine, repo_scope=repo_scope)
        if arts is None:
            return None
        # Serving smoke EACH keeper, not just the strongest: a sibling that boots and crashes real decode only reveals
        # it on a full server boot, and Hyperloom's integrate lane is serial at ~25 min per patch.
        disposition, note, _termination = _run_serving_smoke(recipe, vr.note or "", session, out)
        vr.note = note
        if disposition in ("not_wired", "serving_crash"):
            _reject_fusion_sibling(recipe, vr, arts, disposition)
            return None
        patch = RecipePatch(
            kernel_name=recipe.pattern_id,
            patch_path=arts.patch,
            source_file=recipe.source_file,
            micro_speedup=vr.kernel_speedup,
            snapshot_dir=pristine,
            base_commit=shadow.base_commit if shadow is not None else "",
            # The fused path is env-gated; the flag has to reach the e2e re-baseline or integrate measures the
            # un-fused path (see RecipePatch).
            env_flag=recipe.env_flag,
        )
        _publish_partial_nomination(publish, smoked, patch, out=out, repo_root=repo_root)
        return patch

    cfg = LoopConfig(
        max_recipes=_recipe_ceiling(len(loop_recipes), plan.max_recipes),
        target_speedup=keep_bar,
        output_dir=str(out),
    )
    try:
        result = run_fusion_loop(
            loop_recipes,
            framework=session.framework,
            campaign_fn=campaign_fn,
            config=cfg,
            on_keep=on_keep if multi_patch else None,
        )
        # The combine path folds everything into ONE recipe whose edits are still live in the tree, so the strongest
        # (only) keeper is smoked here.
        if multi_patch and shadow is not None:
            shadow.reset_to_base()
        elif not multi_patch and result.kept and result.best_recipe is not None:
            apply_serving_gate(result, session, out, repo_root=repo_root, pristine_dir=pristine_dir)
    except FusionAbort as exc:
        log.error("fusion run aborted (infrastructure failure): %s", exc)
        result = LoopResult(
            kept=False,
            best=None,
            best_recipe=None,
            termination_reason="harness_author_failed",
        )
    finally:
        if shadow is not None:
            # That state is scratch, and the workspace is a framework install.
            shutil.rmtree(Path(shadow.root) / LOOP_CAMPAIGN_STATE, ignore_errors=True)
            shadow.dispose()

    # Only this scope knows which forge-loop run answered which recipe.
    for iteration in result.history:
        iteration.experiment_id = campaign_experiments.get(iteration.pattern_id, "")

    # Report against the original recipes so a combined run still names them.
    if result.best_recipe is not None:
        result.best_recipe = originals.get(result.best_recipe.pattern_id, result.best_recipe)
    return result


def validate_existing_source(
    recipe,
    *,
    repo_root: str,
    gpu: str,
    harness_path: str,
    target_speedup: float,
):
    """Score the source as it stands, for --no-author runs."""
    runner = HarnessKernelRunner(
        harness_path=harness_path,
        workdir=repo_root or ".",
        gpu=gpu,
        env_flags=dict.fromkeys(recipe.env_flag.split(), "1"),
    )
    return validate_recipe(recipe, runner, target_speedup=target_speedup)


def measure_harness_noise(
    *,
    harness: str,
    repeat: int = 20,
    gpu: str = "0",
    env_flags: tuple[str, ...] = (),
    workdir: str = ".",
) -> dict[str, object]:
    """Measure how much the same harness varies on this machine."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    flags = dict.fromkeys(env_flags, "1")
    speedups: list[float] = []
    eager: list[float] = []
    fused: list[float] = []
    failures = 0
    for i in range(max(1, repeat)):
        runner = HarnessKernelRunner(
            harness_path=harness,
            workdir=workdir,
            framework_root=workdir,
            gpu=gpu,
            env_flags=flags,
        )
        bench = runner.microbench(
            Recipe(
                pattern_id="noise-probe",
                description="",
                env_flag=" ".join(env_flags),
                source_file="",
                source_hints=[],
                fusion_math="",
                eager_reference_hint="",
                shapes={},
                matched_categories=[],
                trigger_share=0.0,
            )
        )
        if bench.skipped or not bench.eager_us or not bench.fused_us:
            failures += 1
            continue
        eager.append(float(bench.eager_us))
        fused.append(float(bench.fused_us))
        speedups.append(float(bench.eager_us) / float(bench.fused_us))
        log.info("run %d/%d: %.4fx", i + 1, repeat, speedups[-1])

    report: dict[str, object] = {"runs": repeat, "usable": len(speedups), "failed": failures}
    if len(speedups) >= 2:
        mean = statistics.fmean(speedups)
        sd = statistics.stdev(speedups)
        cv = sd / mean if mean else 0.0
        report.update(
            {
                "speedup_mean": round(mean, 4),
                "speedup_stdev": round(sd, 5),
                "speedup_cv": round(cv, 5),
                "speedup_min": round(min(speedups), 4),
                "speedup_max": round(max(speedups), 4),
                "spread_pct": round((max(speedups) - min(speedups)) / mean * 100.0, 2),
                "eager_us_mean": round(statistics.fmean(eager), 3),
                "fused_us_mean": round(statistics.fmean(fused), 3),
                # How far out the 3% improvement bar sits.
                "bar_in_sigmas": round(0.03 / (cv * (2**0.5)), 2) if cv else None,
                "verdict": (
                    "the 3% bar is within noise"
                    if cv and 0.03 / (cv * (2**0.5)) < 2.0
                    else "the 3% bar is outside noise"
                ),
            }
        )
    return report


def _package_root(source_file: str) -> str:
    """Top install dir containing ``source_file``'s package (site-packages-style root)."""
    if not source_file:
        return ""
    p = Path(source_file).resolve()
    d = p.parent
    while (d.parent != d) and (d / "__init__.py").is_file():
        d = d.parent
    return str(d)


def _framework_repo_root(source_file: str, framework_root: str) -> str:
    """Repo/install root that patch paths are relative to (for patch export)."""
    import subprocess

    start = source_file or framework_root
    if not start:
        return framework_root or ""
    start_dir = str(Path(start).parent if Path(start).suffix else start)
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        r = git(
            "-C",
            start_dir,
            "rev-parse",
            "--show-toplevel",
            check=False,
            timeout=30,
        )
        if r.returncode == 0:
            toplevel = r.stdout.strip()
            if source_file and _git_tracks(toplevel, source_file):
                return toplevel
            # Inside a git work tree but the framework file is untracked (venv in a git project): use the package
            # root, not the project root.
            return _package_root(source_file) or toplevel or framework_root or ""
    # Not a git work tree at all (plain pip install).
    return _package_root(source_file) or framework_root or ""
