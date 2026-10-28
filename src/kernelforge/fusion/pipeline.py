# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forge-fuse pipeline: diagnose a decode trace, then locate, author, validate and export a fusion."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kernelforge.tools.gpu_arch import canon_arch, detect_arch

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
from .anchor import AnchorReport
from .diagnose import diagnose_trace
from .discovery_stage import _Discovery, _DiscoveryInputs, _discover, _log_located_recipes, _resolve_discover_mode
from .errors import FuseError, FuseUsageError
from .emit import FusionFiles, export_artifacts, restore_exported_changes
from .llm_failure import LlmUnavailableError
from .locate import FrameworkInstall, _framework_repo_root
from .loop import FusionAbort, LoopConfig, LoopResult, RecipePatch, run_fusion_loop
from .models import CompilePassOutcome, FusionArtifacts, Recipe, ValidationResult
from .shadow_repo import ensure_git_workspace
from .request import FuseAgentOptions, FuseRequest
from .report import (
    ANCHOR_RESOLVED_VERDICT,
    LLM_UNAVAILABLE_VERDICT,
    LocatedFusions,
    ManifestSubject,
    RunOutcome,
    build_manifest,
    write_manifest,
)
from .shapes import load_model_config
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
# Exit code for "infrastructure failure before fusion was attempted" (no git workspace, harness not authored, etc.).
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
        raise FuseUsageError(f"Missing option(s): {', '.join(missing)}.")


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


def _run_summary(manifest: dict[str, Any], path: Path, recipes: list, compile_pass_outcome, validation) -> dict:
    """The one-line summary a caller reads instead of the manifest.

    A compile_pass run has no kernel-level ValidationResult, so ITS verdict is
    reported rather than a null that reads as "no validation ran".
    """
    return {
        "verdict": manifest["verdict"],
        "manifest": str(path),
        "patterns": [r.pattern_id for r in recipes],
        "speedup": (
            compile_pass_outcome.speedup
            if compile_pass_outcome is not None
            else (validation.kernel_speedup if validation else None)
        ),
        "kept": (
            compile_pass_outcome.kept if compile_pass_outcome is not None else (validation.kept if validation else None)
        ),
        "error": manifest["error"],
        "agent_backend": manifest.get("agent_backend"),
        "agent_model": manifest.get("agent_model"),
        "agent_sandbox_mode": manifest.get("agent_sandbox_mode"),
    }


def _exit_code_for_run(llm_error, loop: Optional[dict[str, Any]]) -> int:
    """Non-zero when the run failed for a reason an exit-code watcher must see.

    The manifest is the contract, but a run that never reached the model, or one
    whose pipeline never had a chance to fuse anything, must also be visible to
    anything that only watches exit codes.
    """
    if llm_error is not None:
        return EXIT_LLM_UNAVAILABLE
    if (
        loop is not None
        and not loop["kept"]
        and loop["termination_reason"] in ("no_git_workspace", "harness_author_failed", "serving_unconfirmed")
    ):
        return EXIT_INFRASTRUCTURE_FAILURE
    return 0


def _settle_single_patch_workspace(run: _FusionRun, top_recipe, exported_ok: bool):
    """Export what a single-patch run produced and put the framework tree back as it was.

    None of this runs for the other two paths. The multi-patch nomination already
    exported each sibling and restored the tree to base inside its own
    transaction, so exporting again would diff the wrong (reset) tree and file a
    bogus discard; a compile_pass claim likewise runs inside its own restore
    transaction and authors no modules.
    """
    artifacts = None
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
class FuseResult:
    """What one forge-fuse run reports: the one-line summary a caller reads, and the status it leaves with."""

    summary: dict[str, Any]
    exit_code: int


class _AgentBackendCache:
    """Create the one agent backend both agent stages share, on first use, and remember the model it resolved."""

    def __init__(self, agent: FuseAgentOptions) -> None:
        self._agent = agent
        self.selected: Any = None
        self.llm_model = agent.llm_model

    def __call__(self) -> Any:
        if self.selected is not None:
            return self.selected
        try:
            self.selected = _create_agent_backend(
                self._agent.agent_backend,
                self.llm_model,
                self._agent.agent_sandbox_mode,
            )
        except FuseError:
            raise
        except Exception as exc:  # isolation boundary: provider construction, reported as the run's refusal
            raise FuseError(f"agent backend configuration failed: {type(exc).__name__}: {exc}") from exc
        self.llm_model = self.selected.runtime.model
        log.info(
            "selected Agent backend=%s model=%s",
            self.selected.name,
            self.selected.runtime.model,
        )
        return self.selected


@dataclass(frozen=True)
class _ManifestPublisher:
    """Write the run's manifest, as complete as the run has so far got.

    The aggregate is the only thing that points at a keeper, so it cannot wait for every
    campaign to return: a run killed in between would report REVERT while proven,
    already-smoked patches sat in the workspace. Called from ``on_keep`` too, it is never
    missing, only partial, and the end-of-run call overwrites it with the real loop /
    compile-pass / error fields before any exit.
    """

    subject: ManifestSubject
    split: _NominationSplit
    discovery: _Discovery
    backend: _AgentBackendCache
    out: Path

    def __call__(self, outcome: RunOutcome) -> tuple[dict[str, Any], Path]:
        anchor_report = self.discovery.anchor_report
        manifest = build_manifest(
            self.subject,
            LocatedFusions(
                recipe=self.split.top_recipe,
                candidates=self.discovery.recipes,
                anchor=anchor_report.to_dict() if anchor_report is not None else None,
            ),
            outcome,
        )
        selected_agent = self.backend.selected
        if selected_agent is not None:
            manifest["agent_backend"] = selected_agent.name
            manifest["agent_model"] = selected_agent.runtime.model
            manifest["agent_sandbox_mode"] = selected_agent.runtime.sandbox_mode
        return manifest, write_manifest(manifest, self.out)


def _diagnose(request: FuseRequest) -> Any:
    """Diagnose the decode trace: how launch-bound it is, and what fusing could gain end to end."""
    diagnosis = diagnose_trace(
        request.trace_path, decode_steps=request.decode.decode_steps, decode_batch=request.decode.decode_batch
    )
    log.info(
        "diagnosis: candidate=%s launch_bound_share=%.3f predicted_e2e_gain=%.3f busy_of_wall=%s reason=%s",
        diagnosis.is_candidate,
        diagnosis.launch_bound_share,
        diagnosis.predicted_e2e_gain,
        diagnosis.busy_fraction_of_wall,
        diagnosis.reason,
    )
    return diagnosis


def fuse(request: FuseRequest) -> FuseResult:
    """Diagnose a decode trace and locate, author and validate a fusion opportunity for the model.

    Raises :class:`FuseUsageError` for an option combination the run cannot honour and :class:`FuseError` for a run
    that cannot start.
    """
    trace_path, model_path, framework = request.trace_path, request.model_path, request.framework
    stages, agent = request.stages, request.agent
    _require_run_options(trace_path, model_path, framework, request.output_dir)
    fuse_kernel = request.discovery.fuse_kernel.strip()
    discover_mode = _resolve_discover_mode(fuse_kernel, request.discovery.discover_mode)

    out = Path(request.output_dir)
    _setup_logging(out, request.verbose)

    log.info("forge-fuse %s | framework=%s model=%s", __version__, framework, model_path)
    backend = _AgentBackendCache(agent)
    diagnosis = _diagnose(request)

    model_type = str(load_model_config(model_path).get("model_type") or "")
    inputs = _DiscoveryInputs(
        trace_path=trace_path,
        model_path=model_path,
        model_type=model_type,
        install=FrameworkInstall(framework, request.framework_root),
        decode_batch=request.decode.decode_batch,
        attn_tp_size=request.layout.attn_tp_size,
        diagnosis=diagnosis,
        repo_scope=request.repo_scope,
        out=out,
        agent_factory=backend,
    )
    discovery = _discover(inputs, discover_mode, fuse_kernel, stages.dry_run)
    _log_located_recipes(discovery, dry_run=stages.dry_run)
    split = _split_nomination(discovery.recipes, stages.fuse_all_confirmed)
    publish = _ManifestPublisher(
        ManifestSubject(framework=framework, model_path=model_path, model_type=model_type, diagnosis=diagnosis),
        split,
        discovery,
        backend,
        out,
    )

    outcome = RunOutcome()
    if not stages.dry_run and split.top_recipe is not None:
        author = AuthorAgent(backend=backend, gpu=request.gpu, model=backend.llm_model, max_turns=agent.max_turns)
        outcome = _run_fusion_stages(request, discovery.recipes, split, author, publish)
    llm_error = discovery.llm_error
    manifest, path = publish(
        dataclasses.replace(
            outcome,
            verdict_override=_verdict_override(llm_error, discovery.anchor_report, stages.dry_run),
            error=(llm_error.to_dict() if llm_error is not None else None),
        )
    )
    log.info("wrote manifest: %s (verdict=%s)", path, manifest["verdict"])
    return FuseResult(
        summary=_run_summary(manifest, path, discovery.recipes, outcome.compile_pass, outcome.validation),
        exit_code=_exit_code_for_run(llm_error, outcome.loop),
    )


def _run_fusion_stages(
    request: FuseRequest, recipes: list, split: _NominationSplit, author: AuthorAgent, publish
) -> RunOutcome:
    """Author, validate and export what discovery located, as the outcome the run's manifest reports."""
    top_recipe, stages, smoke, layout = split.top_recipe, request.stages, request.smoke, request.layout
    out = Path(request.output_dir)
    repo_root = _framework_repo_root(top_recipe.source_file, request.framework_root)
    # Snapshot the pristine model source BEFORE authoring so a patch can be produced even when the framework is a
    # non-git pip install (git diff would otherwise be empty -> patch=null -> integrate skips the KEPT fusion).
    pristine_dir = _snapshot_fusion_source(repo_root, top_recipe.source_file, out, extra_files=top_recipe.extra_files)
    fusion_run = _FusionRun(
        out=out,
        repo_root=repo_root,
        pristine_dir=pristine_dir,
        framework_root=request.framework_root,
        repo_scope=request.repo_scope,
        session=ServingSession(
            request.framework,
            request.model_path,
            request.gpu,
            smoke.ab_isl,
            smoke.ab_osl,
            server_extra=smoke.server_extra,
            tp=layout.tp,
            block_size=layout.block_size,
            max_model_len=layout.max_model_len,
        ),
        author=author,
        gpu_arch=_run_target_arch(request.gpu_arch),
        target_speedup=DEFAULT_TARGET_SPEEDUP,
    )
    if not (split.multi_patch and stages.validate):
        return _run_single_patch(request, fusion_run, recipes, split)
    # The multi-patch nomination runs BOTH pipelines when a run discovered both kinds: each compile-pass claim
    # through its config A/B, every authored fusion through the kernel autoloop, and every keeper becomes an
    # independent sibling in patches[].
    patches, compile_pass, loop_result, withheld = _run_multi_patch_nomination(
        split,
        fusion_run,
        _LoopPlan(author=stages.author, max_recipes=request.max_recipes),
        _campaign_runtime(fusion_run, request.agent, author.model),
        publish,
    )
    artifacts = None
    # Strongest sibling fills the legacy singular ``artifacts`` slot so the combine-era salvage / timeout path
    # keeps finding a patch.
    if patches:
        strongest = patches[0]
        artifacts = FusionArtifacts(
            patch=strongest.get("patch_path"), repo_root=strongest.get("kernel_repo") or repo_root
        )
        # Mirror the strongest patch under the legacy ``fusion.patch`` name.
        _mirror_legacy_patch(strongest.get("patch_path"), out)
    return RunOutcome(
        validation=loop_result.best if loop_result is not None else None,
        artifacts=artifacts,
        loop=loop_result.to_dict() if loop_result is not None else None,
        compile_pass=compile_pass,
        patches=patches,
        nomination={
            "candidates_seen": len(recipes),
            "resolved": len(split.claims) + len(split.deferred),
            "selected": len(patches),
            # Resolved targets the lane ceiling never funded, so a starved round is not read as "everything ran
            # and kept nothing".
            "withheld": withheld,
        },
    )


def _run_single_patch(request: FuseRequest, run: _FusionRun, recipes: list, split: _NominationSplit) -> RunOutcome:
    """Fuse the run's one patch: claim its compile pass, or author it, validating when asked, and export it."""
    stages, top_recipe = request.stages, split.top_recipe
    if top_recipe.candidate_kind == "compile_pass":
        compile_pass, artifacts = _run_single_compile_pass_claim(
            top_recipe,
            run.claim_serving(),
            ClaimGate(validate=stages.validate, target_speedup=run.target_speedup),
            ClaimExport(run.out, run.repo_root, run.pristine_dir),
        )
        return RunOutcome(artifacts=artifacts, compile_pass=compile_pass)
    # combine folds every recipe into one unit; multi-patch authors each non-claim recipe as its own sibling and
    # runs any claims separately.
    authored = recipes if split.fuse_all_confirmed else (split.deferred if split.multi_patch else [top_recipe])
    if stages.validate:
        # Validate-driven outer loop: per recipe, author -> kernel-validate -> serving-smoke with cross-attempt
        # experience injection; early-exit on the first result that is KEPT (kernel parity + speedup AND survives
        # serving).
        loop_result = _run_fusion_autoloop(
            authored,
            run,
            _LoopPlan(author=stages.author, combine=split.fuse_all_confirmed),
            _campaign_runtime(run, request.agent, run.author.model),
        )
        log.info(
            "fusion loop finished: kept=%s speedup=%s attempts=%d termination=%s",
            loop_result.kept,
            loop_result.best.kernel_speedup if loop_result.best else None,
            len(loop_result.history),
            loop_result.termination_reason,
        )
        return RunOutcome(
            validation=loop_result.best,
            loop=loop_result.to_dict(),
            artifacts=_settle_single_patch_workspace(run, top_recipe, loop_result.kept),
        )
    exported_ok = False
    if stages.author:
        ab_hint = (
            "forge-fuse validates at the KERNEL level (compile + SNR parity + microbench speedup), "
            f"decode batch {request.decode.decode_batch} isl {request.smoke.ab_isl} osl {request.smoke.ab_osl}"
        )
        rc = _run_author_only(authored, run, ab_hint)
        exported_ok = rc == 0
        log.info("author finished rc=%s (no validation requested)", rc)
    return RunOutcome(artifacts=_settle_single_patch_workspace(run, top_recipe, exported_ok))


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
    run_arch = canon_arch(gpu_arch) or detect_arch()
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


def _split_nomination(recipes: list, fuse_all_confirmed: bool) -> _NominationSplit:
    """Divide the recipes by candidate kind, narrowing a combine run to its claim."""
    claims = [r for r in recipes if r.candidate_kind == "compile_pass"]
    deferred = [r for r in recipes if r.candidate_kind != "compile_pass"]
    if not (fuse_all_confirmed and claims and deferred):
        return _NominationSplit(
            claims, deferred, recipes[0] if recipes else None, fuse_all_confirmed, not fuse_all_confirmed
        )
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
