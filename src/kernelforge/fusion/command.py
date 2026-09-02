# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point for `kernelforge forge-fuse`.

Usage:
    kernelforge forge-fuse --trace <kineto.json[.gz]> --model-path <dir> \\
        --framework sglang --output-dir <dir> [--dry-run] [--fuse-all-confirmed]

``--dry-run`` diagnoses the trace, locates fusible patterns, and emits the JSON
manifest with the localized recipe skeleton (no authoring, no GPU). A full run
additionally drives the validate-driven autoloop (author -> kernel-level validate
with cross-attempt experience) and fills in ``validation`` / ``fusion_loop`` /
``artifacts``. Kernel-level only; e2e serving A/B is Hyperloom's job.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import click

from kernelforge.agent_backends.registry import (
    create_registered_backend,
    get_agent_provider,
    resolve_agent_runtime,
    select_default_agent_provider,
)

from . import __version__
from .author import (
    AUTHOR_RC_FAILED,
    AUTHOR_RC_SAFETY,
    build_multi_author_prompt,
    run_author,
)
from .campaign import (
    LOOP_CAMPAIGN_STATE,
    _safe_artifact_id,
    build_campaign_program_md,
    fused_module_path,
    run_recipe_campaign,
)
from .diagnose import diagnose_trace
from .discover import discover_recipes, registered_agent_llm_fn
from .emit import _git_tracks, _is_fused_module_name, export_artifacts, restore_exported_changes
from .gpu_arch import canon_arch, detect_arch
from .harness_contract import harness_contract
from .llm_failure import LlmUnavailableError
from .locate import build_recipes, resolve_framework_source_file
from .loop import FusionAbort, LoopConfig, LoopResult, run_fusion_loop
from .models import CompilePassOutcome, Recipe, ValidationResult
from .shadow_repo import ensure_git_workspace
from .report import LLM_UNAVAILABLE_VERDICT, build_manifest, write_manifest
from .shapes import load_model_config, resolve_decode_shapes
from .validate import (
    DEFAULT_TARGET_SPEEDUP,
    KERNEL_KEEP_CHECKPOINT,
    HarnessKernelRunner,
    fused_symbol_invocation_evidence,
    serving_smoke,
    serving_smoke_verdict,
    validate_recipe,
)
from .vllm_passes import (
    TargetRuntime,
    enable_pass_in_source,
    resolve_target_runtime,
    verify_pass_enabled,
)
from kernelforge.llm.git import git

log = logging.getLogger("forge_fusion")

# Exit code for "the run never reached the model". Distinct from 1 so a caller
# can tell an outage apart from a real fusion failure.
EXIT_LLM_UNAVAILABLE = 3
# Exit code for "infrastructure failure before fusion was attempted": no git
# workspace, harness could not be authored, etc. Lets callers distinguish a
# setup/environment problem from a genuine "nothing to fuse" answer.
EXIT_INFRASTRUCTURE_FAILURE = 4
_AGENT_SANDBOX_MODES = frozenset({"workspace-write", "read-only", "bypass"})


def _credential_shape() -> tuple[bool, bool]:
    """Return whether OpenAI-side and Anthropic-side credentials are configured."""
    openai = any(os.environ.get(name, "").strip() for name in ("OPENAI_API_KEY", "SAFE_API_KEY", "FORGE_API_KEY"))
    anthropic = any(os.environ.get(name, "").strip() for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")) or any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
    )
    return bool(openai), bool(anthropic)


def _resolve_agent_choice(
    agent_backend: str,
    llm_model: Optional[str],
) -> tuple[str, str]:
    """Resolve provider from credentials, then model from provider precedence."""
    requested = (agent_backend or "auto").strip().lower()
    if requested == "auto":
        has_openai, has_anthropic = _credential_shape()
        if has_openai and not has_anthropic:
            provider = "codex"
        elif has_anthropic and not has_openai:
            provider = "claude"
        elif has_openai and has_anthropic:
            # Use the project's existing first-available default without passing
            # a model, so dual-provider selection never guesses from model prefixes.
            provider = select_default_agent_provider().name
        else:
            raise click.UsageError(
                "--agent-backend auto found no OpenAI or Anthropic credentials; "
                "configure OPENAI_API_KEY for Codex, ANTHROPIC_API_KEY/"
                "ANTHROPIC_AUTH_TOKEN for Claude, or pass an explicit backend "
                "with its provider configuration"
            )
    else:
        provider = get_agent_provider(requested).name

    registration = get_agent_provider(provider)
    provider_env = "CODEX_MODEL" if provider == "codex" else "CLAUDE_MODEL"
    model = str(llm_model or "").strip() or os.environ.get(provider_env, "").strip() or registration.default_model
    return provider, model


def _resolve_agent_sandbox_mode(explicit: Optional[str]) -> str:
    """Resolve and validate the global Agent sandbox policy for this run."""
    mode = (
        str(explicit or "").strip().lower()
        or os.environ.get("FORGE_AGENT_SANDBOX_MODE", "").strip().lower()
        or "workspace-write"
    )
    if mode not in _AGENT_SANDBOX_MODES:
        choices = ", ".join(sorted(_AGENT_SANDBOX_MODES))
        raise click.UsageError(f"unsupported agent sandbox mode {mode!r}; choose one of: {choices}")
    return mode


# Per-attempt agent wall clock. Two hours is what a source-level fusion needs: it
# authors a kernel and then boots the model twice for the A/B. Overridable because
# the loop grants this budget to EVERY attempt, so an operator sizing a campaign
# against an outer timeout has to be able to bound a single one.
_AGENT_TIMEOUT_DEFAULT_SEC = 7200


def _agent_timeout_sec() -> int:
    """Resolve the per-attempt agent wall clock, defaulting to two hours."""
    raw = os.environ.get("FORGE_FUSION_AGENT_TIMEOUT_SEC", "").strip()
    if not raw:
        return _AGENT_TIMEOUT_DEFAULT_SEC
    try:
        value = int(raw)
    except ValueError as exc:
        raise click.UsageError(
            f"FORGE_FUSION_AGENT_TIMEOUT_SEC must be an integer number of seconds, got {raw!r}"
        ) from exc
    if value <= 0:
        raise click.UsageError("FORGE_FUSION_AGENT_TIMEOUT_SEC must be greater than zero")
    return value


def _create_agent_backend(
    agent_backend: str,
    llm_model: Optional[str],
    agent_sandbox_mode: Optional[str] = None,
):
    """Create one no-cross-provider-fallback backend for the complete run."""
    sandbox_mode = _resolve_agent_sandbox_mode(agent_sandbox_mode)
    provider, model = _resolve_agent_choice(agent_backend, llm_model)
    runtime = resolve_agent_runtime(
        provider,
        model=model,
        timeout_sec=_agent_timeout_sec(),
        reasoning_effort="high",
        sandbox_mode=sandbox_mode,
        fallback_provider="",
    )
    return create_registered_backend(runtime)


def _author_harness_target(repo_root: str, out: Path) -> str:
    """Return a unique in-worktree harness target for the Agent session."""
    if not repo_root:
        return ""
    digest = hashlib.sha256(str(out.resolve()).encode("utf-8")).hexdigest()[:12]
    return str(Path(repo_root).resolve() / ".forge_fusion" / f"kernel_harness_{digest}.py")


_STAGED_HARNESS_RE = re.compile(r"^kernel_harness_[0-9a-f]{12}\.py$")


def _is_staged_harness_name(name: str) -> bool:
    """Whether a staging entry is a harness some run staged, per the name above."""
    return bool(_STAGED_HARNESS_RE.match(name))


def _author_module_dirs(source_files: list[str]) -> list[str]:
    """Directories in which the author may create new fused helper modules.

    Exactly the directories the export path scans (see
    :func:`emit._fusion_scoped_paths`), so a helper the author workspace guard keeps
    is a helper the emitted patch carries. Nominating a wider scope — the harness
    directory, say — would let an authored module survive the run and never reach
    the Hyperloom handoff.
    """
    dirs: list[str] = []
    for source_file in source_files:
        if not source_file:
            continue
        parent = str(Path(source_file).parent)
        if parent not in dirs:
            dirs.append(parent)
    return dirs


def _prepare_author_harness(
    author_harness_path: str,
    harness_path: str,
    *,
    inherited: bool,
) -> tuple[bool, str, bool]:
    """Prepare the in-worktree harness target without exposing an outside path.

    Returns ``(ready, reason, deterministic)``. ``deterministic`` is what lets the
    caller refuse to spend the loop's whole attempt budget on a failure that
    cannot change: ``author_harness_path`` is a pure function of the repo root and
    the output directory, so a symlink on that path is there again next attempt,
    and an existing staging target is self-perpetuating. An ``OSError`` while
    creating or copying is weather and stays retryable.
    """
    if not author_harness_path:
        return True, "", False
    target = Path(author_harness_path)
    created_parent = not target.parent.exists()
    try:
        if target.resolve(strict=False) != target:
            return False, "author harness staging path contains a symlink", True
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            return False, "author harness staging target already exists", True
        if inherited:
            source = Path(harness_path)
            if not source.is_file():
                return True, "", False
            shutil.copy2(source, target)
    except OSError as exc:
        if created_parent:
            with contextlib.suppress(OSError):
                target.parent.rmdir()
        return False, f"could not prepare author harness target: {type(exc).__name__}", False
    return True, "", False


def _finish_author_harness(
    author_harness_path: str,
    harness_path: str,
    *,
    inherited: bool,
    author_ok: bool,
) -> tuple[bool, str]:
    """Publish a fresh harness, verify an inherited one, and remove staging."""
    if not author_harness_path:
        return True, ""
    target = Path(author_harness_path)
    final = Path(harness_path)
    ok = True
    reason = ""
    try:
        if inherited:
            if not target.is_file() or not final.is_file():
                ok = False
                reason = "inherited harness disappeared during authoring"
            elif target.read_bytes() != final.read_bytes() or stat.S_IMODE(target.stat().st_mode) != stat.S_IMODE(
                final.stat().st_mode
            ):
                ok = False
                reason = "author modified the inherited harness"
        elif author_ok and target.is_file():
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, final)
    except OSError as exc:
        ok = False
        reason = f"could not finalize author harness: {type(exc).__name__}"
    finally:
        try:
            if target.exists() or target.is_symlink():
                target.unlink()
            # Running the staged harness is what this directory is for, and the
            # interpreter writes __pycache__ beside the module it just executed.
            # That byproduct is the framework's own, not foreign content, so
            # leaving it to block the rmdir turned a successful authoring turn
            # into a failure -- and one that repeats forever, because every
            # retry runs the harness again and recreates it.
            shutil.rmtree(target.parent / "__pycache__", ignore_errors=True)
            target.parent.rmdir()
        except OSError:
            # Anything else left behind is a workspace-safety violation and must
            # be surfaced rather than deleted broadly.
            if target.exists():
                ok = False
                reason = reason or "author harness staging path could not be removed"
            elif target.parent.exists():
                try:
                    leftovers = sorted(p.name for p in target.parent.iterdir())
                except OSError:
                    leftovers = []
                # The directory is per-repo while the digest is per-output-dir, so
                # a sibling harness belongs to another run. Failing on one turned a
                # finished authoring turn into AUTHOR FAILED, and deleting it is
                # not ours to do while that run may still be executing it.
                foreign = [name for name in leftovers if not _is_staged_harness_name(name)]
                if foreign or not leftovers:
                    ok = False
                    reason = reason or (
                        "author harness staging path could not be removed"
                        + (f" (left behind: {', '.join(foreign)})" if foreign else "")
                    )
    return ok, reason


def _author_baseline_harness(
    recipe,
    *,
    harness_path: str,
    repo_root: str,
    out: Path,
    gpu: str,
    llm_model: Optional[str],
    max_turns: int,
    backend,
) -> tuple[bool, str]:
    """Write the harness ``recipe``'s campaign benchmarks, before it starts.

    The loop anchors its speedup by benching the unfused framework ahead of its
    first Implementer session, and this harness is what the driver runs to do
    it. Authored inside the campaign it arrives a step too late: without the
    anchor no candidate can be scored, so nothing can ever be kept.

    The harness encodes one chain, so it is per recipe: measuring a candidate
    against another chain's harness compares it to the wrong baseline.
    """
    Path(harness_path).unlink(missing_ok=True)
    staging = _author_harness_target(repo_root, out)
    ready, reason, _deterministic = _prepare_author_harness(staging, harness_path, inherited=False)
    if not ready:
        return False, reason
    target = staging or harness_path
    prompt = (
        build_campaign_program_md(recipe, harness_path="")
        + harness_contract(target, recipe.env_flag)
        + "\nWrite ONLY that harness. Do not edit the framework source and do "
        "not create any other file; the fused kernel is authored after this.\n"
    )
    (out / "harness_prompt.md").write_text(prompt, encoding="utf-8")
    rc = run_author(
        prompt,
        workdir=repo_root or ".",
        log_path=str(out / "harness_author.log"),
        gpu=gpu,
        model=llm_model,
        max_turns=max_turns,
        backend=backend,
        timeout_s=_agent_timeout_sec(),
        target_files=[target],
        new_module_dirs=[],
    )
    published, error = _finish_author_harness(staging, harness_path, inherited=False, author_ok=rc == 0)
    if rc != 0:
        return False, f"harness author exited {rc}"
    return published, error


def _author_rc_after_harness(rc: int, *, harness_ok: bool) -> int:
    """Fold a harness-finalization failure into the author's return code.

    Retryable on purpose: the bucket mixes an author that rewrote the inherited
    harness with a plain OSError while publishing it, and only the first is
    deterministic. It must not replace a verdict the author already reached,
    though -- a safety stop is decided identically on every attempt, so turning
    it into a retryable failure sends the loop back to re-run a recipe that is
    rejected the same way, and the budget goes to proving it again.
    """
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


@click.command("forge-fuse")
@click.version_option(version=__version__)
@click.option(
    "--trace",
    "trace_path",
    default="",
    type=click.Path(),
    help="Decode kineto trace (*.trace.json[.gz]), captured with CUDA graphs disabled.",
)
@click.option("--model-path", default="", help="Path to the model directory (must contain config.json).")
@click.option(
    "--framework",
    default=None,
    type=click.Choice(["sglang", "vllm", "vllm-aiter"]),
    help="Target inference framework.",
)
@click.option(
    "--output-dir",
    default="",
    type=click.Path(),
    help="Output directory for the manifest + logs.",
)
@click.option(
    "--harness-noise",
    "harness_noise_path",
    default="",
    hidden=True,
    help="Diagnostic: repeat one kernel-validation harness and report its variance.",
)
@click.option(
    "--harness-noise-repeat",
    default=20,
    type=int,
    hidden=True,
    help="Diagnostic: how many times to repeat the harness.",
)
@click.option(
    "--harness-noise-env",
    "harness_noise_env",
    multiple=True,
    hidden=True,
    help="Diagnostic: env flag to set for each harness run (repeatable).",
)
@click.option(
    "--framework-root",
    default="",
    help="Explicit framework source root (else auto-detect the installed package).",
)
@click.option("--decode-batch", default=16, type=int, help="Representative decode batch size (T) for shapes.")
@click.option(
    "--decode-steps",
    default=0,
    type=int,
    help="Decode steps captured in the trace (to normalize kernels/step).",
)
@click.option(
    "--discover",
    "discover_mode",
    type=click.Choice(["patterns", "llm"]),
    default="patterns",
    help="Recipe discovery: 'patterns' (template library) or 'llm' (LLM reads trace+source, autonomous).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Diagnose + locate only; emit manifest with recipe skeleton (no author/validate).",
)
@click.option("--author/--no-author", default=True, help="Author the fused kernel via the LLM (non-dry-run).")
@click.option("--validate/--no-validate", default=True, help="Run the A/B decode validation (non-dry-run).")
@click.option(
    "--fuse-all-confirmed",
    is_flag=True,
    help="Author ALL source-confirmed patterns together (not just the top), and "
    "A/B all their flags. A compile-pass candidate cannot be authored with "
    "them, so it is claimed alone and the rest wait for a later round.",
)
@click.option("--gpu", default="0", help="HIP device id for author + A/B.")
@click.option(
    "--agent-backend",
    type=click.Choice(["auto", "claude", "codex"]),
    default="auto",
    show_default=True,
    help="Registered Agent provider for discovery and authoring.",
)
@click.option(
    "--agent-sandbox-mode",
    type=click.Choice(["workspace-write", "read-only", "bypass"]),
    envvar="FORGE_AGENT_SANDBOX_MODE",
    default="workspace-write",
    show_default=True,
    help="Agent runtime sandbox. Use bypass only when an external boundary already enforces isolation.",
)
@click.option(
    "--model",
    "llm_model",
    default=None,
    help="Agent model. Explicit value wins; otherwise uses provider-specific "
    "$CODEX_MODEL/$CLAUDE_MODEL, then the registered provider default. "
    "``--model`` is accepted as an alias (Hyperloom forge-fuse spelling).",
)
@click.option("--max-turns", default=100, type=int, help="Max authoring turns.")
@click.option("--ab-isl", default=512, type=int, help="A/B input length.")
@click.option("--ab-osl", default=128, type=int, help="A/B output length.")
@click.option(
    "--bench-extra",
    default="",
    help="Extra bench_one_batch args (e.g. '--attention-backend triton').",
)
@click.option(
    "--server-extra",
    default="",
    help=(
        "Extra serving args for the smoke launch (e.g. '--kv-cache-dtype fp8'). "
        "A model whose engine refuses to start without a flag can never reach the "
        "kernel the smoke exists to exercise."
    ),
)
@click.option(
    "--gpu-target",
    "gpu_arch",
    default="",
    help="Canonical GPU arch the author writes for (e.g. gfx950); auto-detected via rocminfo when omitted.",
)
@click.option(
    "--tp",
    default=1,
    type=int,
    help="Tensor-parallel size for the serving smoke (must match the session).",
)
@click.option(
    "--block-size",
    "block_size",
    default=0,
    type=int,
    help="vLLM KV --block-size for the serving smoke (0=omit). Required for "
    "sparse-attention models that reject the default block size.",
)
@click.option(
    "--max-model-len",
    "max_model_len",
    default=0,
    type=int,
    help="Serving-smoke max model / context length (0 uses the smoke default 4096).",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
def run(
    trace_path: str,
    model_path: str,
    framework: str,
    output_dir: str,
    harness_noise_path: str,
    harness_noise_repeat: int,
    harness_noise_env: tuple[str, ...],
    framework_root: str,
    decode_batch: int,
    decode_steps: int,
    discover_mode: str,
    dry_run: bool,
    author: bool,
    validate: bool,
    fuse_all_confirmed: bool,
    gpu: str,
    agent_backend: str,
    agent_sandbox_mode: str,
    llm_model: Optional[str],
    max_turns: int,
    ab_isl: int,
    ab_osl: int,
    bench_extra: str,
    server_extra: str,
    gpu_arch: str,
    tp: int,
    block_size: int,
    max_model_len: int,
    verbose: bool,
) -> None:
    """Diagnose a decode trace and locate a fusion opportunity for the model."""
    if harness_noise_path:
        click.echo(
            json.dumps(
                measure_harness_noise(
                    harness=harness_noise_path,
                    repeat=harness_noise_repeat,
                    gpu=gpu,
                    env_flags=harness_noise_env,
                ),
                indent=2,
            )
        )
        return

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

    out = Path(output_dir)
    _setup_logging(out, verbose)

    log.info("forge-fuse %s | framework=%s model=%s", __version__, framework, model_path)
    selected_agent = None

    def require_agent_backend():
        """Lazily create and cache the one backend shared by both Agent stages."""
        nonlocal selected_agent, llm_model
        if selected_agent is not None:
            return selected_agent
        try:
            selected_agent = _create_agent_backend(
                agent_backend,
                llm_model,
                agent_sandbox_mode,
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

    diagnosis = diagnose_trace(trace_path, decode_steps=decode_steps, decode_batch=decode_batch)
    log.info(
        "diagnosis: candidate=%s launch_bound_share=%.3f predicted_e2e_gain=%.3f busy_of_wall=%s reason=%s",
        diagnosis.is_candidate,
        diagnosis.launch_bound_share,
        diagnosis.predicted_e2e_gain,
        diagnosis.busy_fraction_of_wall,
        diagnosis.reason,
    )

    model_type = str(load_model_config(model_path).get("model_type") or "")
    llm_error: LlmUnavailableError | None = None
    if discover_mode == "llm":
        # LLM-autonomous discovery: the model reads the launch-bound profile + the
        # real source and proposes fusible chains itself (not capped to templates).
        shapes = resolve_decode_shapes(model_path, decode_batch=decode_batch)
        source_file, _source_note = resolve_framework_source_file(
            model_path, framework, framework_root=framework_root, model_type=model_type
        )
        try:
            discovery_agent = require_agent_backend()
            discovery_workdir = _framework_repo_root(source_file, framework_root) or str(
                Path(source_file).parent if source_file else Path.cwd()
            )
            recipes = discover_recipes(
                diagnosis,
                model_type=model_type,
                framework=framework,
                source_file=source_file,
                shapes=shapes,
                trace_path=trace_path,
                # Forwarded for the same reason build_recipes gets it: each
                # proposal is checked against THIS install's compile-pass config,
                # and that verdict rewrites the pattern id. Left unset, the check
                # probes whichever vLLM is importable here, so the run can judge
                # the wrong install and store under a different key than a run
                # that passed the flag.
                framework_root=framework_root,
                llm_fn=registered_agent_llm_fn(
                    discovery_agent,
                    model=discovery_agent.runtime.model,
                    workdir=discovery_workdir,
                    protected_files=[source_file] if source_file else [],
                    log_path=str(out / "discovery_llm.txt"),
                ),
            )
        except LlmUnavailableError as exc:
            # The model was never reached, so this run knows nothing about the
            # kernel. Recipes stay empty, but the verdict below must not be the
            # one an empty list normally produces.
            llm_error = exc
            recipes = []
            log.error(
                "discovery could not reach the LLM (%s after %d attempt(s)): %s",
                exc.kind,
                exc.attempts,
                exc,
            )
        else:
            log.info("discovery(llm) proposed %d fusion(s)", len(recipes))
    else:
        recipes = build_recipes(
            diagnosis,
            model_path=model_path,
            framework=framework,
            framework_root=framework_root,
            decode_batch=decode_batch,
        )
    top_recipe = recipes[0] if recipes else None
    if recipes:
        log.info(
            "located %d candidate recipe(s): %s",
            len(recipes),
            ", ".join(f"{r.pattern_id}({r.trigger_share:.2f})" for r in recipes),
        )
    elif llm_error is not None:
        log.error(
            "no fusion recipe located because the LLM was unreachable "
            "(verdict: %s) — this is NOT a no_opportunity result",
            LLM_UNAVAILABLE_VERDICT,
        )
    else:
        log.info("no fusion recipe located (verdict: no_opportunity)")
    validation = None
    artifacts = None
    loop_manifest = None
    compile_pass_outcome: Optional[CompilePassOutcome] = None
    loop_result = None

    # A claim and an authored kernel are validated and exported by different,
    # non-interchangeable machinery (config A/B vs kernel parity + microbench),
    # so one run cannot do both. Clearing the flag is what narrows this run to
    # the claim alone; the authored candidates stay on the manifest for a later
    # round. Refusing would fail a run the caller cannot fix, the flag being on
    # by default.
    claims = [r for r in recipes if r.candidate_kind == "compile_pass"]
    deferred = [r for r in recipes if r.candidate_kind != "compile_pass"]
    if fuse_all_confirmed and claims and deferred:
        top_recipe = claims[0]
        fuse_all_confirmed = False
        log.info(
            "claiming compile pass %s first; deferring %s",
            top_recipe.pattern_id,
            ", ".join(r.pattern_id for r in deferred),
        )

    if not dry_run and top_recipe is not None:
        repo_root = _framework_repo_root(top_recipe.source_file, framework_root)
        # Snapshot the pristine model source BEFORE authoring so a patch can be
        # produced even when the framework is a non-git pip install (git diff would
        # otherwise be empty -> patch=null -> integrate skips the KEPT fusion).
        pristine_dir = _snapshot_fusion_source(repo_root, top_recipe.source_file, out)
        authored = recipes if fuse_all_confirmed else [top_recipe]
        ab_hint = (
            f"forge-fuse validates at the KERNEL level (compile + SNR parity + "
            f"microbench speedup), decode batch {decode_batch} isl {ab_isl} osl {ab_osl}"
        )
        target_speedup = DEFAULT_TARGET_SPEEDUP
        # One arch value for the whole run: the author tunes for it, so a
        # mismatch would have it writing for a chip the run is not on.
        run_arch = canon_arch(gpu_arch) or canon_arch(detect_arch())
        if run_arch:
            log.info("target GPU arch: %s", run_arch)
        else:
            log.warning("GPU arch undetectable; the author will not be told a target ISA")

        exported_ok = False

        if top_recipe.candidate_kind == "compile_pass":
            # The flip edits a LIVE install, so every exit path must restore it and
            # the patch must be diffed against the pre-run snapshot (not HEAD, which
            # would sweep in unrelated uncommitted edits).
            runtime = resolve_target_runtime(framework, framework_root=framework_root)
            with _live_file_restored(top_recipe.source_file):
                compile_pass_outcome = _run_compile_pass(
                    top_recipe,
                    runtime=runtime,
                    model_path=model_path,
                    gpu=gpu,
                    validate=validate,
                    out=out,
                    isl=ab_isl,
                    osl=ab_osl,
                    target_speedup=target_speedup,
                )
                log.info(
                    "compile pass %s: kept=%s speedup=%s note=%s",
                    top_recipe.compile_pass_flag,
                    compile_pass_outcome.kept,
                    compile_pass_outcome.speedup,
                    compile_pass_outcome.note,
                )
                if repo_root and compile_pass_outcome.kept:
                    artifacts = export_artifacts(
                        repo_root,
                        top_recipe.source_file,
                        out,
                        pristine_dir=pristine_dir,
                        snapshot_diff_only=True,
                    )
            compile_pass_outcome.reverted = True  # the context manager just did it
            exported_ok = compile_pass_outcome.kept
        elif validate:
            # Validate-driven outer loop: per recipe, author -> kernel-validate ->
            # serving-smoke with cross-attempt experience injection; early-exit on the
            # first result that is KEPT (kernel parity + speedup AND survives serving).
            loop_result = _run_fusion_autoloop(
                authored,
                framework=framework,
                out=out,
                repo_root=repo_root,
                author=author,
                gpu=gpu,
                llm_model=llm_model,
                target_speedup=target_speedup,
                keep_threshold=target_speedup,
                combine=fuse_all_confirmed,
                model_path=model_path,
                gpu_arch=run_arch,
                agent_backend=agent_backend,
                agent_sandbox_mode=agent_sandbox_mode,
                server_extra=server_extra,
                ab_isl=ab_isl,
                ab_osl=ab_osl,
                max_turns=max_turns,
                agent_factory=require_agent_backend,
                pristine_dir=pristine_dir,
                tp=tp,
                block_size=block_size,
                max_model_len=max_model_len,
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
        elif author:
            # Author-only (validation disabled): keep the single-pass authoring path.
            # Same prompt contract as the loop path -- the hardware it is targeting,
            # where the harness goes, and the bar to clear. One harness covers the
            # whole prompt here, which authors every recipe at once.
            harness_path = str(out / "kernel_harness.py")
            author_harness_path = _author_harness_target(repo_root, out)
            ready, harness_error, harness_fatal = _prepare_author_harness(
                author_harness_path,
                harness_path,
                inherited=False,
            )
            if not ready:
                rc = AUTHOR_RC_SAFETY if harness_fatal else AUTHOR_RC_FAILED
                _append_author_rejection(str(out / "author.log"), harness_error)
                log.error("author harness preparation failed: %s", harness_error)
            else:
                prompt_harness_path = author_harness_path or harness_path
                author_sources = [r.source_file for r in authored if r.source_file]
                prompt = build_multi_author_prompt(
                    [r.to_dict() for r in authored],
                    framework=framework,
                    ab_hint=ab_hint,
                    target_speedup=target_speedup,
                    harness_path=prompt_harness_path,
                    gpu_arch=run_arch,
                    model_path=model_path,
                )
                (out / "author_prompt.md").write_text(prompt, encoding="utf-8")
                rc = run_author(
                    prompt,
                    workdir=repo_root or ".",
                    log_path=str(out / "author.log"),
                    gpu=gpu,
                    model=llm_model,
                    max_turns=max_turns,
                    backend=require_agent_backend,
                    timeout_s=_agent_timeout_sec(),
                    target_files=[*author_sources, prompt_harness_path],
                    new_module_dirs=_author_module_dirs(author_sources),
                )
                harness_ok, harness_error = _finish_author_harness(
                    author_harness_path,
                    harness_path,
                    inherited=False,
                    author_ok=rc == 0,
                )
                if not harness_ok:
                    rc = _author_rc_after_harness(rc, harness_ok=harness_ok)
                    _append_author_rejection(
                        str(out / "author.log"),
                        harness_error,
                    )
                    log.error("author harness finalization failed: %s", harness_error)
            exported_ok = rc == 0
            log.info("author finished rc=%s (no validation requested)", rc)

        # Only export a patch when the run produced a USABLE fusion (validate path:
        # kernel parity + speedup AND serving survived). A crashing / near-miss attempt
        # must NOT leave an exported patch behind. The compile-pass branch already
        # exported and restored inside its own transaction.
        if repo_root and exported_ok and compile_pass_outcome is None:
            artifacts = export_artifacts(repo_root, top_recipe.source_file, out, pristine_dir=pristine_dir)

        # The exported patch is taken back out of the framework. Gated on the
        # patch rather than on how the run reached it, so neither branch above
        # can accidentally skip the restore.
        #
        # A compile_pass claim is excluded: it exports and restores inside its own
        # ``_live_file_restored`` transaction, so restoring again here would act on
        # a tree it already put back.
        if repo_root and artifacts and artifacts.patch and compile_pass_outcome is None:
            restore_exported_changes(repo_root, artifacts, pristine_dir=pristine_dir)
        # A compile_pass claim runs inside its own restore transaction and authors
        # no modules, so this rollback has nothing to do there and would only file
        # a bogus ".failed" attempt.
        if repo_root and pristine_dir and compile_pass_outcome is None and _needs_discard(exported_ok, artifacts):
            # Nothing usable came out, so leave the framework exactly as found
            # rather than carrying unvalidated code into whatever runs next.
            _discard_failed_attempt(repo_root, top_recipe.source_file, out, pristine_dir)

    manifest = build_manifest(
        framework=framework,
        model_path=model_path,
        model_type=model_type,
        diagnosis=diagnosis,
        recipe=top_recipe,
        candidates=recipes,
        validation=validation,
        artifacts=artifacts,
        loop=loop_manifest,
        compile_pass=compile_pass_outcome,
        verdict_override=(LLM_UNAVAILABLE_VERDICT if llm_error is not None else ""),
        error=(llm_error.to_dict() if llm_error is not None else None),
    )
    if selected_agent is not None:
        manifest["agent_backend"] = selected_agent.name
        manifest["agent_model"] = selected_agent.runtime.model
        manifest["agent_sandbox_mode"] = selected_agent.runtime.sandbox_mode
    path = write_manifest(manifest, out)
    log.info("wrote manifest: %s (verdict=%s)", path, manifest["verdict"])
    # A compile_pass run has no kernel-level ValidationResult, so report ITS verdict
    # instead of a null that reads as "no validation ran".
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
    if llm_error is not None:
        # Exit non-zero as well: the manifest is the contract, but a run that
        # never reached the model must also be visible to anything that only
        # watches exit codes.
        raise SystemExit(EXIT_LLM_UNAVAILABLE)
    if (
        loop_result is not None
        and not loop_result.kept
        and loop_result.termination_reason in ("no_git_workspace", "harness_author_failed", "serving_unconfirmed")
    ):
        # Infrastructure failure: the pipeline never had a chance to fuse anything.
        # Distinct from 0 (no_opportunity / exhausted) and EXIT_LLM_UNAVAILABLE.
        #
        # A KEPT run is NOT one of these even when the smoke went unconfirmed: it
        # produced a validated kernel and a patch, and exiting non-zero would have
        # Hyperloom read the whole run as failed and discard exactly the KEEP this
        # deferral exists to preserve.
        raise SystemExit(EXIT_INFRASTRUCTURE_FAILURE)


def _combined_recipe(recipes: list[Recipe]) -> Recipe:
    """Fold several confirmed recipes into ONE unit for the loop.

    ``--fuse-all-confirmed`` means "stack all confirmed fusions and measure the
    COMBINED gain" (matching the proven multi-fusion result), not "try them one at
    a time and stop at the first that clears the bar". So the loop treats the set
    as a single recipe: the author writes all fusions together, and validation
    toggles ALL their env flags. ``env_flag`` becomes the space-joined set.
    """
    base = recipes[0]
    flags = list(dict.fromkeys(f for r in recipes for f in r.env_flag.split() if f))
    return Recipe(
        pattern_id="+".join(r.pattern_id for r in recipes),
        description="; ".join(r.description for r in recipes),
        env_flag=" ".join(flags),
        source_file=base.source_file,
        source_hints=[h for r in recipes for h in r.source_hints],
        fusion_math="\n".join(f"[{r.pattern_id}] {r.fusion_math}" for r in recipes),
        eager_reference_hint="; ".join(r.eager_reference_hint for r in recipes),
        shapes=base.shapes,
        matched_categories=sorted({c for r in recipes for c in r.matched_categories}),
        trigger_share=max(r.trigger_share for r in recipes),
        rocm_native=any(r.rocm_native for r in recipes),
    )


def _run_fusion_autoloop(
    recipes,
    *,
    framework: str,
    out: Path,
    repo_root: str,
    author: bool,
    gpu: str,
    llm_model: Optional[str],
    target_speedup: float,
    keep_threshold: float | None = None,
    combine: bool = False,
    model_path: str = "",
    gpu_arch: str = "",
    agent_backend: str = "",
    agent_sandbox_mode: str = "",
    server_extra: str = "",
    ab_isl: int,
    ab_osl: int,
    max_turns: int,
    agent_factory,
    pristine_dir: str = "",
    tp: int = 1,
    block_size: int = 0,
    max_model_len: int = 0,
):
    """Try each ranked recipe as one forge-loop campaign.

    The loop owns authoring, validation and keep/revert; this only establishes
    what it needs -- a git workspace over the framework tree, and per recipe a
    harness and a driver -- then runs the serving smoke on whatever was kept.

    When ``combine`` is set, all confirmed recipes are folded into ONE unit so
    the campaign stacks every fusion and measures the COMBINED gain.
    """
    originals = {r.pattern_id: r for r in recipes}
    loop_recipes = [_combined_recipe(recipes)] if (combine and len(recipes) > 1) else recipes

    # The author aims at ``target_speedup`` (raised when a record was inherited);
    # the gate keeps anything above ``keep_threshold`` (absolute). Splitting them
    # is what stops an inherited record from discarding a usable patch.
    keep_bar = target_speedup if keep_threshold is None else keep_threshold

    # Only the authoring path runs campaigns. Without one there is nothing to
    # keep or revert, and the placeholders below would empty the very modules
    # ``--no-author`` exists to score.
    shadow = None
    if author and loop_recipes:
        # Every recipe's fused module is tracked at the baseline, not just the
        # one about to run: the loop keeps with ``git add -u``, which cannot
        # commit a file created mid-campaign.
        shadow = ensure_git_workspace(
            repo_root,
            loop_recipes[0].source_file,
            git_dir=str(out / "shadow.git"),
            extra_paths=tuple(fused_module_path(r) for r in loop_recipes),
        )
        if shadow is None:
            log.error(
                "no git workspace over %s: the forge-loop cannot keep or revert a candidate there",
                repo_root,
            )
            return LoopResult(
                kept=False,
                best=None,
                best_recipe=None,
                termination_reason="no_git_workspace",
            )

    campaign_experiments: dict[str, str] = {}

    def _harness_path_for(recipe) -> str:
        return str(out / f"kernel_harness_{_safe_artifact_id(recipe.pattern_id)}.py")

    def campaign_fn(recipe, experience: str):
        if not author:
            return validate_existing_source(
                recipe,
                repo_root=repo_root,
                gpu=gpu,
                harness_path=_harness_path_for(recipe),
                target_speedup=keep_bar,
            )
        # A fresh campaign refuses to start where the previous one left state,
        # and the loop anchors that state to the workspace rather than to
        # ``--experiments-dir``. Without this the SECOND recipe is rejected.
        shutil.rmtree(Path(shadow.root) / LOOP_CAMPAIGN_STATE, ignore_errors=True)
        # Score every recipe against the UNFUSED framework: the previous
        # campaign's commits are otherwise still in the tree, and the two
        # changes get reported stacked as if they were one. Cannot lose a win,
        # because run_fusion_loop returns the instant a campaign KEEPs.
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
            )
        # After the reset, so the loop's anchor bench measures the unfused tree.
        harness_path = _harness_path_for(recipe)
        ready, reason = _author_baseline_harness(
            recipe,
            harness_path=harness_path,
            repo_root=repo_root,
            out=out,
            gpu=gpu,
            llm_model=llm_model,
            max_turns=max_turns,
            backend=agent_factory,
        )
        if not ready:
            # Not this recipe's failure: recording it as one would file a wrong
            # lesson that the next recipe's campaign is then prompted with.
            raise FusionAbort(f"no harness for {recipe.pattern_id}: {reason}")
        outcome = run_recipe_campaign(
            recipe,
            workspace=shadow.root,
            harness_path=harness_path,
            output_dir=str(out),
            experience=experience,
            gpu=gpu,
            gpu_target=gpu_arch,
            target_speedup=keep_bar,
            model=llm_model or "",
            agent_backend=agent_backend,
            agent_sandbox_mode=agent_sandbox_mode,
            shadow_env=shadow.env,
            fused_module=fused_module_path(recipe),
        )
        if outcome.experiment_id:
            campaign_experiments[recipe.pattern_id] = outcome.experiment_id
        return outcome.result

    cfg = LoopConfig(
        max_recipes=len(loop_recipes),
        target_speedup=keep_bar,
        output_dir=str(out),
    )
    try:
        result = run_fusion_loop(
            loop_recipes,
            framework=framework,
            campaign_fn=campaign_fn,
            config=cfg,
        )
        if result.kept and result.best_recipe is not None:
            apply_serving_gate(
                result,
                framework=framework,
                out=out,
                gpu=gpu,
                model_path=model_path,
                isl=ab_isl,
                osl=ab_osl,
                server_extra=server_extra,
                repo_root=repo_root,
                pristine_dir=pristine_dir,
                tp=tp,
                block_size=block_size,
                max_model_len=max_model_len,
            )
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


def apply_serving_gate(
    result,
    *,
    framework: str,
    out: Path,
    gpu: str,
    model_path: str,
    isl: int,
    osl: int,
    server_extra: str = "",
    repo_root: str = "",
    pristine_dir: str = "",
    tp: int = 1,
    block_size: int = 0,
    max_model_len: int = 0,
) -> None:
    """Boot the real server once; only a fused-kernel fault demotes a KEEP.

    Parity and the microbench run on small shapes with no CUDA graph, so a
    kernel that allocates or host-syncs per call passes both and still crashes
    the captured decode loop. Booting costs tens of minutes, hence once.

    Session ``tp`` / KV ``block_size`` / ``max_model_len`` must match real serving
    (sparse vLLM rejects the default block size). A failure the smoke does not
    attribute to the kernel is not a kernel loss: keep the micro KEEP so
    Hyperloom e2e can still verify it.
    """
    if not (_serving_check_enabled() and model_path and result.best_recipe):
        return
    recipe = result.best_recipe
    flags = {f: "1" for f in recipe.env_flag.split()}
    safe_id = _safe_artifact_id(recipe.pattern_id)
    vr = result.best
    smoke_block = int(block_size) if int(block_size or 0) > 0 else None
    smoke_mml = int(max_model_len) if int(max_model_len or 0) > 0 else 4096
    # Export BEFORE the smoke, so a forge-fuse killed while serving still leaves an
    # applicable patch. ``pristine_dir`` is what makes that possible on a non-git
    # framework (a pip install has nothing for `git diff` to report), and the
    # checkpoint is written only once the patch is on disk: it is the completion
    # marker Hyperloom salvages on, so it must never point at a missing patch.
    exported = _export_salvage_patch(
        out,
        getattr(recipe, "source_file", ""),
        repo_root=repo_root,
        pristine_dir=pristine_dir,
    )
    if exported:
        _write_kernel_keep_checkpoint(out, recipe, vr, repo_root=repo_root)
    else:
        log.warning(
            "no fusion patch could be exported for %s; a killed run cannot be salvaged",
            recipe.pattern_id,
        )
    # Cheapest gate first, and the only one that catches a fusion nothing calls:
    # the smoke would boot, decode and PASS, because stock code is what ran.
    wired, wiring = fused_symbol_invocation_evidence(getattr(recipe, "source_file", ""))
    if not wired:
        result.kept = False
        vr.kept = False
        vr.kernel_speedup = None
        vr.note = (
            f"KERNEL OK but NOT WIRED IN: {wiring}. The microbench measured the fused "
            f"entry point directly, so its speedup says nothing about the served model, "
            f"whose end-to-end gain is exactly zero. | LESSON: authoring the fused module "
            f"is half the deliverable -- replace the ORIGINAL call site in the framework's "
            f"forward path with a call to the fused entry point, under the same env gate, "
            f"and leave the unfused code as the fallback branch."
        )
        result.termination_reason = "not_wired"
        _clear_kernel_keep_checkpoint(out)
        log.warning("fusion not wired into %s: %s", recipe.pattern_id, wiring)
        return
    log.info("fusion wiring confirmed for %s: %s", recipe.pattern_id, wiring)
    verdict = serving_smoke_verdict(
        model_path,
        flags,
        framework=framework,
        gpu=gpu,
        isl=isl,
        osl=osl,
        server_extra=server_extra,
        log_path=str(out / f"serving_smoke_{safe_id}.log"),
        tp=tp,
        block_size=smoke_block,
        max_model_len=smoke_mml,
    )
    reason = verdict.reason
    if verdict.ok:
        vr.note = f"{vr.note} | SERVING SMOKE OK"
        log.info("serving smoke OK for %s", recipe.pattern_id)
        return
    if verdict.blames_kernel:
        result.kept = False
        vr.kept = False
        vr.correctness_passed = False
        vr.kernel_speedup = None
        vr.note = (
            f"KERNEL OK but SERVING CRASHED (CUDA-graph-ON decode): {reason} "
            f"| LESSON: the kernel is NOT CUDA-graph-capture safe. Use a STATIC "
            f"launch grid (no data-dependent grid size), pre-allocate every "
            f"scratch/output tensor ONCE outside the fused path (no per-call "
            f"torch.empty/zeros/cat), avoid host<->device syncs, and index "
            f"strictly in bounds for every token count. Re-author CUDA-graph safe."
        )
        result.termination_reason = "serving_crash"
        _clear_kernel_keep_checkpoint(out)
        log.warning("serving smoke FAILED for %s: %s", recipe.pattern_id, reason)
        return
    vr.note = (
        f"{vr.note} | SERVING SMOKE UNCONFIRMED at stage {verdict.stage} "
        f"(defer e2e): {reason} | LESSON: the GPU did not fault, so nothing here "
        f"is evidence against the kernel. Do not re-author to fix it; Hyperloom "
        f"e2e is the KEEP/REVERT gate."
    )
    result.termination_reason = "serving_unconfirmed"
    log.warning(
        "serving smoke unconfirmed for %s at stage %s (keeping micro KEEP): %s",
        recipe.pattern_id,
        verdict.stage,
        reason,
    )


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
        env_flags={f: "1" for f in recipe.env_flag.split()},
    )
    return validate_recipe(recipe, runner, target_speedup=target_speedup)


def _serving_check_enabled() -> bool:
    """Serving smoke is ON by default; ``FORGE_FUSION_SERVING_CHECK=0`` disables it."""
    return os.environ.get("FORGE_FUSION_SERVING_CHECK", "1") != "0"


def _export_salvage_patch(
    out: Path,
    source_file: str,
    *,
    repo_root: str = "",
    pristine_dir: str = "",
) -> bool:
    """Write ``fusion.patch`` for the edits made so far; report whether one exists.

    ``pristine_dir`` is required for a non-git framework tree: ``git diff`` reports
    nothing for a pip install, so without the snapshot baseline the export is empty
    and there is nothing for Hyperloom to apply.
    """
    if not repo_root or not source_file:
        return False
    # This output directory may be reused. Invalidate the previous run's
    # completion marker and patch BEFORE asking export to produce this run's
    # artifact; otherwise an empty export can accidentally bless stale bytes.
    _clear_kernel_keep_checkpoint(out)
    try:
        artifacts = export_artifacts(
            repo_root,
            source_file,
            out,
            pristine_dir=pristine_dir or None,
        )
    except Exception as exc:  # noqa: BLE001 — export must never fail the gate.
        log.warning("fusion patch export failed: %s: %s", type(exc).__name__, exc)
        return False
    if not artifacts.patch:
        return False
    patch = Path(artifacts.patch)
    return patch.is_file() and patch.stat().st_size > 0


def _write_kernel_keep_checkpoint(out: Path, recipe, vr, *, repo_root: str = "") -> None:
    """Persist a micro KEEP so a killed forge-fuse process can still be salvaged.

    Written atomically: a reader that finds this file must find a COMPLETE record,
    since it is what Hyperloom treats as "this run produced a salvageable KEEP".
    """
    payload = {
        "kept": True,
        "kernel_speedup": getattr(vr, "kernel_speedup", None),
        "eager_us": getattr(vr, "eager_us", None),
        "fused_us": getattr(vr, "fused_us", None),
        "env_flag": getattr(recipe, "env_flag", ""),
        "pattern_id": getattr(recipe, "pattern_id", ""),
        "source_file": getattr(recipe, "source_file", ""),
        "repo_root": repo_root,
        "note": getattr(vr, "note", ""),
    }
    path = out / KERNEL_KEEP_CHECKPOINT
    tmp = path.with_suffix(".json.tmp")
    with contextlib.suppress(OSError):
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)


def _clear_kernel_keep_checkpoint(out: Path) -> None:
    """Drop salvage artifacts after a real fused-kernel serving crash."""
    for name in (KERNEL_KEEP_CHECKPOINT, "fusion.patch"):
        with contextlib.suppress(OSError):
            (out / name).unlink()


def measure_harness_noise(
    *,
    harness: str,
    repeat: int = 20,
    gpu: str = "0",
    env_flags: tuple[str, ...] = (),
    workdir: str = ".",
) -> dict[str, object]:
    """Measure how much the same harness varies on this machine.

    The KEEP bar and the plateau noise floor (2%) are assumptions about
    measurement stability that were never checked against a real GPU. If the
    run-to-run spread here is comparable to those numbers, then "beat the previous
    result by 3%" is partly deciding on noise -- which matters most for the
    inherited floor, where a 3% margin gates whether a result is recorded at all.

    Repeats one harness unchanged, so everything except measurement noise is held
    constant. Report ``speedup_cv`` (relative standard deviation) against
    ``bar_in_sigmas``: a bar worth trusting sits several sigma out.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    flags = {name: "1" for name in env_flags}
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
                # How far out the 3% improvement bar sits. Comparing two independent
                # measurements roughly doubles the variance, hence the sqrt(2).
                "bar_in_sigmas": round(0.03 / (cv * (2**0.5)), 2) if cv else None,
                "verdict": (
                    "the 3% bar is within noise"
                    if cv and 0.03 / (cv * (2**0.5)) < 2.0
                    else "the 3% bar is outside noise"
                ),
            }
        )
    return report


@contextlib.contextmanager
def _live_file_restored(path: str):
    """Guarantee byte-exact restoration of a live framework file on EVERY exit.

    The compile-pass path edits an INSTALLED framework, so a failed smoke, an empty
    patch or an exception must not leave the install silently modified. Restoring
    the pre-run bytes (not ``git checkout``, which would reset to HEAD and discard
    unrelated uncommitted edits) also keeps any pre-existing modifications intact,
    and because the exported patch is diffed against the same pre-run snapshot,
    those modifications never leak into it.
    """
    target = Path(path) if path else None
    original: Optional[bytes] = None
    mode: Optional[int] = None
    if target is not None and target.is_file():
        try:
            original = target.read_bytes()
            mode = target.stat().st_mode
        except OSError as exc:
            log.warning("cannot snapshot %s for restore: %s", path, exc)
            original = None
    try:
        yield
    finally:
        # Stay in the ``finally`` without returning: a return here would swallow
        # an exception from the body when there was nothing to restore.
        if original is not None and target is not None:
            try:
                if target.read_bytes() != original:
                    target.write_bytes(original)
                    if mode is not None:
                        os.chmod(target, mode)
                    log.info("restored %s to its pre-run contents", path)
            except OSError as exc:
                log.error("FAILED to restore %s (%s): the install may be left modified", path, exc)


def _serving_arm(
    label: str,
    *,
    framework: str,
    model_path: str,
    gpu: str,
    out: Path,
    isl: int,
    osl: int,
    server_extra: str = "",
    launcher_exe: str,
    env_flags: Optional[dict] = None,
) -> tuple[bool, str, dict]:
    """One serving arm of the compile-pass A/B; returns ``(ok, reason, metrics)``."""
    metrics: dict = {}
    ok, reason = serving_smoke(
        model_path,
        env_flags or {},
        framework=framework,
        gpu=gpu,
        isl=isl,
        osl=osl,
        server_extra=server_extra,
        launcher_exe=launcher_exe,
        metrics=metrics,
        log_path=str(out / f"compile_pass_{label}.log"),
    )
    log.info("compile pass %s arm: ok=%s tok_s=%s reason=%s", label, ok, metrics.get("tok_s"), reason)
    return ok, reason, metrics


def _run_compile_pass(
    recipe: Recipe,
    *,
    runtime: TargetRuntime,
    model_path: str,
    gpu: str,
    validate: bool,
    out: Path,
    isl: int,
    osl: int,
    target_speedup: float,
) -> CompilePassOutcome:
    """Claim a fusion the framework implements but ships switched OFF.

    The edit itself is deterministic and LLM-free, but "the server booted" proves
    nothing: flipping a class default is a no-op for any flag something else
    overrides, and an enabled pass can still fail to match the model or cost
    throughput. So the flip is confirmed against the target's RESOLVED config and
    then measured by a disabled/enabled A/B on the same runtime, model and request
    shape, with the same speedup bar the authoring path uses.

    Caller MUST revert the file unless ``kept``; this function does not restore.
    """
    flag = recipe.compile_pass_flag
    outcome = CompilePassOutcome(
        flag=flag, config_file=recipe.source_file, source="default", target_speedup=target_speedup
    )
    if runtime.error:
        outcome.note = f"target runtime not pinned: {runtime.error}"
        return outcome

    baseline: dict = {}
    if validate:
        ok, reason, baseline = _serving_arm(
            "baseline_disabled",
            framework=runtime.framework,
            model_path=model_path,
            gpu=gpu,
            out=out,
            isl=isl,
            osl=osl,
            launcher_exe=runtime.launcher_exe,
        )
        if not ok or not baseline.get("tok_s"):
            outcome.note = f"baseline (pass disabled) arm failed: {reason}"
            return outcome
        outcome.baseline_tok_s = baseline.get("tok_s")

    if not enable_pass_in_source(recipe.source_file, flag):
        outcome.note = (
            f"no disabled default to flip for {flag} in {recipe.source_file} (already enabled, or the flag is absent)"
        )
        return outcome
    log.info("flipped native compile pass %s in %s", flag, recipe.source_file)

    # Did the edit actually change what the target RESOLVES? A level or any other
    # override would silently win, making the patch behaviourally empty.
    state = verify_pass_enabled(flag, python=runtime.python, require_root=runtime.require_root)
    outcome.enabled_after_edit = state.enabled
    outcome.source = state.source or outcome.source
    if state.enabled is not True:
        outcome.note = (
            f"after the edit the target still resolves {flag}="
            f"{state.enabled} (source={state.source}, error={state.error[:120]}): "
            f"the patch would have no effect"
        )
        return outcome

    if not validate:
        outcome.kept = True
        outcome.note = "validation disabled: edit confirmed to change the resolved config, but NO serving A/B was run"
        return outcome

    ok, reason, enabled = _serving_arm(
        "enabled",
        framework=runtime.framework,
        model_path=model_path,
        gpu=gpu,
        out=out,
        isl=isl,
        osl=osl,
        launcher_exe=runtime.launcher_exe,
        # Fusion passes report what they rewrote at debug level; without this the
        # run cannot tell "fused N sites" from "matched nothing".
        env_flags={"VLLM_LOGGING_LEVEL": "DEBUG"},
    )
    outcome.enabled_tok_s = enabled.get("tok_s")
    outcome.pass_activated = enabled.get("pass_activated")
    outcome.activation_evidence = list(enabled.get("activation_evidence") or [])
    if not ok or not outcome.enabled_tok_s:
        outcome.note = f"enabled arm failed: {reason}"
        return outcome
    outcome.validated = True
    if outcome.pass_activated is False:
        outcome.note = (
            "the pass ran but matched NOTHING in this model's graph (0 sites fused): the flip buys nothing here"
        )
        return outcome
    outcome.speedup = outcome.enabled_tok_s / float(outcome.baseline_tok_s or 0.0 or 1.0)
    if outcome.speedup < target_speedup:
        outcome.note = (
            f"enabled arm is not faster: {outcome.baseline_tok_s} -> "
            f"{outcome.enabled_tok_s} tok/s (speedup {outcome.speedup:.3f} "
            f"< target {target_speedup})"
        )
        return outcome
    outcome.kept = True
    outcome.note = (
        f"A/B kept: {outcome.baseline_tok_s} -> {outcome.enabled_tok_s} tok/s "
        f"(speedup {outcome.speedup:.3f}), pass_activated="
        f"{outcome.pass_activated}"
    )
    return outcome


_FUSED_INVENTORY = ".fused_siblings"


def _read_fused_inventory(snapshot_dir: Path) -> set[str] | None:
    """Names of the fused-looking modules that existed BEFORE authoring.

    ``None`` means the inventory was never recorded, which has to be treated as
    "nothing is known to be author-created" rather than as an empty set.
    """
    listing = snapshot_dir / _FUSED_INVENTORY
    try:
        raw = listing.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Unreadable and undecodable are the same answer here: this runs while
        # cleaning up, so it must not raise, and an inventory it cannot trust
        # means it deletes nothing.
        return None
    return {line.strip() for line in raw.splitlines() if line.strip()}


def _needs_discard(exported_ok: bool, artifacts) -> bool:
    """Whether the framework is still carrying changes nobody exported.

    A run that produced a patch has already been restored by
    :func:`restore_exported_changes`. Everything else -- a rejected attempt, and
    equally an accepted one whose export came back empty -- leaves edits behind
    that no artifact records, so they have to be rolled back here. The empty-export
    case is easy to miss because the run looks successful right up to the point
    where there is nothing to show for it.
    """
    if not exported_ok:
        return True
    return not (artifacts and artifacts.patch)


def _discard_failed_attempt(repo_root: str, source_file: str, out: Path, pristine_dir: str) -> None:
    """Put the framework back as it was after a run that produced nothing usable.

    A KEPT run exports a patch and then restores; a failed run used to do neither,
    leaving the framework carrying code that never passed validation. The fused
    path is env-gated so a default import is unlikely to hit it, but a pip-installed
    package silently modified is a trap for whoever uses that machine next -- and
    inheriting a floor makes failed runs MORE common, so this got likelier.

    The attempt is preserved under ``out/.failed`` first: discarding it outright
    would throw away the only record of what the author actually wrote.

    Modules the author created are deleted, identified against the inventory the
    snapshot recorded rather than by name -- a framework file such as
    ``diffusion_gemma.py`` matches the marker and must survive. The inventory is
    used instead of the snapshot's own files because copying a sibling is allowed
    to fail without failing the run, and a failed copy would otherwise make a
    framework file look author-created. Deleting from ``site-packages`` on that
    basis is not a mistake worth risking, so a missing inventory removes nothing.
    """
    if not repo_root or not source_file or not pristine_dir:
        return
    _snapshot_fusion_source(repo_root, source_file, out, subdir=".failed")
    _reset_fusion_source(repo_root, source_file, pristine_dir=pristine_dir)

    for candidate in _author_created_modules(source_file, pristine_dir):
        with contextlib.suppress(OSError):
            candidate.unlink()
            log.info("discarded author-created module %s", candidate.name)


def _author_created_modules(source_file: str, pristine_dir: str) -> list[Path]:
    """Fused-looking modules that appeared beside the source during this run.

    Identified against the inventory the snapshot recorded rather than by name: a
    framework file such as ``fused_moe.py`` matches the marker and must survive.
    The inventory is used instead of the snapshot's own files because copying a
    sibling is allowed to fail without failing the run, and a failed copy would
    otherwise make a framework file look author-created. Deleting from
    ``site-packages`` on that basis is not a mistake worth risking, so a missing
    inventory claims nothing.

    Shared by the two paths that have to account for these modules -- discarding a
    failed attempt, and adopting a KB patch over one -- because a divergence
    between them is invisible until a framework install is already polluted.
    """
    if not source_file or not pristine_dir:
        return []
    pre_existing = _read_fused_inventory(Path(pristine_dir))
    if pre_existing is None:
        return []
    model_dir = Path(source_file).parent
    if not model_dir.is_dir():
        return []
    # The inventory lists the source's SIBLINGS, so the source is absent from it
    # by construction -- and a model file can itself be named like a fused module
    # (``fused_moe.py``). Skipping it explicitly keeps the caller from deleting the
    # file a restore just put back.
    source_resolved = Path(source_file).resolve()
    found: list[Path] = []
    for candidate in sorted(model_dir.glob("*.py")):
        if not _is_fused_module_name(candidate.name) or candidate.name in pre_existing:
            continue
        if candidate.resolve() == source_resolved:
            continue
        found.append(candidate)
    return found


def _snapshot_fusion_source(repo_root: str, source_file: str, out: Path, subdir: str = ".pristine") -> str:
    """Copy the pristine model source (pre-authoring) into ``out/<subdir>/<rel>``.

    Lets :func:`emit.export_artifacts` produce a patch by diffing snapshot-vs-live
    when the framework is a non-git pip install (git diff is empty there). Returns
    the snapshot root, or "" when nothing could be snapshotted.
    """
    if not source_file or not Path(source_file).is_file():
        return ""
    pdir = out / subdir
    root = Path(repo_root).resolve() if repo_root else None

    def _rel(p: Path) -> str:
        if root:
            with contextlib.suppress(ValueError):
                return str(p.resolve().relative_to(root))
        return p.name

    def _snap(f: Path) -> bool:
        dest = pdir / _rel(f)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            return True
        except OSError as exc:
            log.warning("could not snapshot pristine fusion source %s: %s", f, exc)
            return False

    # The MAIN source snapshot is mandatory: without it, export would diff a
    # missing snapshot ("") vs the live edited file and emit the whole file as a
    # bogus "new file". Fail closed (return "") in that case.
    src = Path(source_file)
    if not _snap(src):
        return ""

    # Also snapshot any pre-existing *_fused*/*_fusion* module beside it, so export
    # can tell an author-created NEW module from a pre-existing framework file that
    # merely matches the marker. Failure here is non-fatal.
    #
    # Their NAMES are recorded separately from the copies, because a rollback
    # deletes what is not on this list: listing a directory is reliable, copying
    # into it is not, and a file missing only because its copy failed must not
    # look author-created.
    model_dir = src.parent
    if model_dir.is_dir():
        siblings = [
            f for f in sorted(model_dir.glob("*.py")) if _is_fused_module_name(f.name) and f.resolve() != src.resolve()
        ]
        with contextlib.suppress(OSError):
            (pdir / _FUSED_INVENTORY).write_text("".join(f"{f.name}\n" for f in siblings), encoding="utf-8")
        for f in siblings:
            _snap(f)
    return str(pdir)


def _reset_fusion_source(repo_root: str, source_file: str, pristine_dir: str = "") -> None:
    """Revert the tracked model source file to its committed baseline (best-effort).

    Called before each author attempt so a failed attempt does not leave broken
    edits for the next one. Untracked files (e.g. a stale ``*_fused.py``) are left
    in place — they are not imported by the reverted eager source and deleting by
    pattern could remove unrelated modules.

    Non-git framework (pip install): git checkout cannot revert, so restore the
    source file from the pre-authoring ``pristine_dir`` snapshot when available.

    Only the MAIN source file is restored here, on both paths. Author-created
    ``*_fused*`` siblings are the caller's to clear, because identifying them needs
    the pristine inventory (:func:`_author_created_modules`) rather than a name
    pattern that would also match framework modules. They must not be left for the
    next attempt: the author guard inventories the module directory when it starts,
    so a leftover name reads as pre-existing and re-authoring the same fusion is
    rejected for touching it.
    """
    import subprocess

    if not repo_root or not source_file:
        return
    # Use the SAME rel scheme as _snapshot_fusion_source (basename fallback when the
    # source is not under repo_root) so the pristine restore below can find the snap.
    try:
        rel = str(Path(source_file).resolve().relative_to(Path(repo_root).resolve()))
        rel_is_repo_relative = True
    except ValueError:
        rel = Path(source_file).name
        rel_is_repo_relative = False
    # Decide by whether the source file is git-TRACKED, NOT merely inside a work
    # tree — a pip framework under a project-local venv/site-packages is untracked,
    # so `git checkout` is a no-op there and we must restore from the snapshot.
    # (Aligned with export_artifacts / restore_exported_changes.)
    if not _git_tracks(repo_root, source_file):
        if pristine_dir:
            snap = Path(pristine_dir) / rel
            if snap.is_file():
                with contextlib.suppress(OSError):
                    Path(source_file).write_text(snap.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        return  # untracked: nothing git can revert
    if not rel_is_repo_relative:
        return  # git checkout below needs a repo-relative path
    try:
        status = git(
            "-C",
            repo_root,
            "status",
            "--porcelain",
            "--",
            rel,
            check=False,
            timeout=30,
        )
        src = Path(repo_root) / rel
        if status.returncode == 0 and status.stdout.strip() and src.is_file():
            backup = (
                Path(os.environ.get("USER_DATA_PATH") or "/tmp")
                / "forge_fusion"
                / "source_backups"
                / str(time.time_ns())
                / rel
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup)
            log.warning("backed up dirty fusion source before reset: %s", backup)
        git("-C", repo_root, "checkout", "--", rel, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("could not reset %s: %s", rel, exc)


def _package_root(source_file: str) -> str:
    """Top install dir containing ``source_file``'s package (site-packages-style root).

    Walks up while a parent has ``__init__.py`` and returns the dir ABOVE the
    top package (e.g. ``.../qwen3.py`` -> ``.../site-packages``). Used as the patch
    repo_root for a non-git (pip-installed) framework so exported diff paths are
    package-relative and apply cleanly at that root.

    NOTE: assumes every intermediate package level ships an ``__init__.py`` (true
    for vLLM/sglang). A PEP 420 namespace package (no ``__init__.py``) would stop
    the walk early and yield a deeper-than-expected root; revisit if such a
    framework appears.
    """
    if not source_file:
        return ""
    p = Path(source_file).resolve()
    d = p.parent
    while (d.parent != d) and (d / "__init__.py").is_file():
        d = d.parent
    return str(d)


def _framework_repo_root(source_file: str, framework_root: str) -> str:
    """Repo/install root that patch paths are relative to (for patch export).

    Uses the git work-tree root ONLY when ``source_file`` is actually git-TRACKED
    there. A pip-installed framework frequently lives under a git work tree (e.g. a
    project-local ``.venv``/``site-packages``) yet is untracked; returning the
    project root then makes patch paths project-relative and non-appliable at the
    package root. In that case (and for a plain pip install) fall back to the
    package install root so exported diff paths stay package-relative.
    """
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
            # Inside a git work tree but the framework file is untracked (venv in a
            # git project): use the package root, not the project root.
            return _package_root(source_file) or toplevel or framework_root or ""
    # Not a git work tree at all (plain pip install).
    return _package_root(source_file) or framework_root or ""


# The command is registered on the kernelforge CLI as `forge-fuse`; this alias
# keeps `python -m kernelforge.fusion.command` working for direct debugging.
main = run


if __name__ == "__main__":
    main()
