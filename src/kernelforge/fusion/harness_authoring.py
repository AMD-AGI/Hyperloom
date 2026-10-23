# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author and align the baseline validation harness before a fusion campaign starts."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from .author import AuthorAgent, AuthorWorkspace, run_author
from .campaign import CampaignBrief, build_campaign_program_md
from .harness_contract import harness_contract
from .shapes import harness_group_dim_mismatch
from .validate import HarnessKernelRunner, eager_trace_alignment
from .agent_selection import _agent_timeout_sec

log = logging.getLogger("forge_fusion")


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


def _prepare_author_harness(author_harness_path: str) -> tuple[bool, str, bool]:
    """Prepare the in-worktree harness target without exposing an outside path."""
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
    except OSError as exc:
        if created_parent:
            with contextlib.suppress(OSError):
                target.parent.rmdir()
        return False, f"could not prepare author harness target: {type(exc).__name__}", False
    return True, "", False


def _finish_author_harness(author_harness_path: str, harness_path: str, *, author_ok: bool) -> tuple[bool, str]:
    """Publish the authored harness and remove staging."""
    if not author_harness_path:
        return True, ""
    target = Path(author_harness_path)
    final = Path(harness_path)
    ok = True
    reason = ""
    try:
        if author_ok and target.is_file():
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, final)
    except OSError as exc:
        ok = False
        reason = f"could not finalize author harness: {type(exc).__name__}"
    finally:
        try:
            if target.exists() or target.is_symlink():
                target.unlink()
            # Running the staged harness is what this directory is for, and the interpreter writes __pycache__ beside
            # the module it just executed.
            shutil.rmtree(target.parent / "__pycache__", ignore_errors=True)
            target.parent.rmdir()
        except OSError:
            # Anything else left behind is a workspace-safety violation and must be surfaced rather than deleted
            # broadly.
            if target.exists():
                ok = False
                reason = reason or "author harness staging path could not be removed"
            elif target.parent.exists():
                try:
                    leftovers = sorted(p.name for p in target.parent.iterdir())
                except OSError:
                    leftovers = []
                # The directory is per-repo while the digest is per-output-dir, so a sibling harness belongs to
                # another run.
                foreign = [name for name in leftovers if not _is_staged_harness_name(name)]
                if foreign or not leftovers:
                    ok = False
                    reason = reason or (
                        "author harness staging path could not be removed"
                        + (f" (left behind: {', '.join(foreign)})" if foreign else "")
                    )
    return ok, reason


def _harness_alignment_attempts() -> int:
    """How many times the harness may be re-authored onto the right code path."""
    raw = os.environ.get("FORGE_HARNESS_ALIGN_ATTEMPTS", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("bad FORGE_HARNESS_ALIGN_ATTEMPTS=%r; using 3", raw)
        return 3


def _probe_eager_alignment(
    recipe,
    *,
    harness_path: str,
    repo_root: str,
    gpu: str,
) -> tuple[Optional[bool], str, list[str]]:
    """Run the fresh harness once and check its eager arm against the trace.

    This costs one harness run before the campaign starts, which buys the only
    chance to catch a baseline measured against the wrong implementation. Once
    the loop is running, every number it produces is relative to this arm and
    nothing re-examines it.
    """
    runner = HarnessKernelRunner(
        harness_path=harness_path,
        workdir=repo_root or ".",
        framework_root=repo_root,
        gpu=gpu,
        env_flags=dict.fromkeys(recipe.env_flag.split(), "1"),
    )
    report = runner.report(recipe)
    observed = [str(k) for k in (report.get("eager_kernels") or []) if str(k).strip()]
    aligned, reason = eager_trace_alignment(observed, getattr(recipe, "trace_kernels", {}))
    if aligned is None and report.get("eager_matches_trace") is False:
        # The harness could not name the kernels but knows it did not match. Its own
        # verdict is worth more than our inability to check it.
        return False, "harness reported eager_matches_trace=false", observed
    return aligned, reason, observed


def _author_baseline_harness(
    recipe,
    *,
    harness_path: str,
    repo_root: str,
    out: Path,
    agent: AuthorAgent,
) -> tuple[bool, str]:
    """Write the harness ``recipe``'s campaign benchmarks, before it starts.

    The harness is re-authored while its eager arm demonstrably runs a different
    code path from the one the trace recorded: a wrong baseline is not a smaller
    version of a right one, and every later stage inherits it.
    """
    attempts = _harness_alignment_attempts()
    feedback = ""
    published, error = False, "harness was never authored"

    for attempt in range(1, attempts + 1):
        Path(harness_path).unlink(missing_ok=True)
        staging = _author_harness_target(repo_root, out)
        ready, reason, _deterministic = _prepare_author_harness(staging)
        if not ready:
            return False, reason
        target = staging or harness_path
        prompt = (
            build_campaign_program_md(recipe, CampaignBrief(harness_path=""))
            + harness_contract(target, recipe.env_flag)
            + "\nWrite ONLY that harness. Do not edit the framework source and do "
            "not create any other file; the fused kernel is authored after this.\n" + feedback
        )
        suffix = "" if attempt == 1 else f".retry{attempt}"
        (out / f"harness_prompt{suffix}.md").write_text(prompt, encoding="utf-8")
        rc = run_author(
            prompt,
            AuthorWorkspace(workdir=repo_root or ".", log_path=str(out / "harness_author.log"), target_files=[target]),
            dataclasses.replace(agent, timeout_s=_agent_timeout_sec()),
        )
        published, error = _finish_author_harness(staging, harness_path, author_ok=rc == 0)
        if rc != 0:
            return False, f"harness author exited {rc}"
        if not published:
            return published, error

        shape_why = harness_group_dim_mismatch(
            Path(harness_path).read_text(encoding="utf-8", errors="replace"),
            getattr(recipe, "shapes", {}) or {},
        )
        if shape_why:
            log.warning(
                "harness attempt %d/%d used the wrong group axis: %s",
                attempt,
                attempts,
                shape_why,
            )
            feedback = (
                "\n## FIX REQUIRED (group axis)\n"
                f"{shape_why}\n"
                "Rebuild the tensors from `n_local_groups` / `o_groups` in the "
                "shapes block. Do not set G from gqa_groups or num_attention_heads.\n"
            )
            continue

        aligned, why, observed = _probe_eager_alignment(
            recipe,
            harness_path=harness_path,
            repo_root=repo_root,
            gpu=agent.gpu,
        )
        if aligned is None:
            log.info("harness eager-arm alignment unchecked: %s", why)
            return published, error
        if aligned:
            log.info("harness eager arm matches the trace: %s", why)
            return published, error

        log.warning(
            "harness attempt %d/%d measured the wrong code path: %s",
            attempt,
            attempts,
            why,
        )
        feedback = _alignment_feedback(recipe, why, observed)

    if feedback.startswith("\n## FIX REQUIRED (group axis)"):
        return False, f"harness used the wrong group axis after {attempts} attempts"
    return False, f"harness eager arm never matched the traced kernels after {attempts} attempts"


def _alignment_feedback(recipe, why: str, observed: list[str]) -> str:
    """Tell the next harness attempt exactly which code path it actually hit."""
    evidence = getattr(recipe, "trace_kernels", {}) or {}
    expected = [str(evidence.get("anchor") or "")]
    for key in ("before", "after"):
        expected += [str(n) for n in (evidence.get(key) or [])]
    expected = [e for e in expected if e]
    seen = "\n".join(f"    {k}" for k in observed[:20]) or "    (the harness named none)"
    want = "\n".join(f"    {k}" for k in expected[:20])
    return f"""
## Your previous harness measured the WRONG code path — rewrite it
{why}

Kernels your eager arm actually launched:
{seen}

Kernels the trace recorded for this fusion:
{want}

The functions you called are not the ones that run in the served model. Do not
adjust the ones you picked: go back to the model's forward pass and follow the
calls through to whatever issues the kernels above — the module the layer really
instantiates, the mixin the attention backend really inherits, the branch this
configuration really takes. Symbols that merely look right are how the previous
attempt got here. Verify with the profiler before you report anything.
"""
