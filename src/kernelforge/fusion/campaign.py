# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run one forge-loop campaign per fusion recipe.

The forge-loop is designed to be shelled out as an isolated, hard-killable
subprocess, so the fusion pipeline reuses it verbatim rather than keeping a
second author-validate loop of its own. One recipe is one campaign: the loop's
own iteration control does the repeated authoring, and its scoring decides keep
or revert against the pristine (unfused) anchor.

What stays outside the loop is what the loop has no notion of -- diagnosing a
trace, choosing which chain to fuse, and booting a real server to prove the
kernel survives CUDA-graph capture.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from kernelforge.fusion.driver_shim import write_driver
from kernelforge.fusion.models import Recipe, ValidationResult
from kernelforge.fusion.shadow_repo import SHADOW_BRANCH
from kernelforge.fusion.validate import DEFAULT_TARGET_SPEEDUP
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

log = logging.getLogger("forge_fusion")

# The kernel backend that carries the decode-fusion authoring discipline.
FUSION_KERNEL_BACKEND = "fusion"

# Knowledge-base producer for records this pipeline authors. A producer owns its
# own candidate index, so fusion records never rank against a kernel campaign's.
FUSION_PRODUCER = "fusion"

# Per-recipe wall clock. Authoring a fused kernel and proving parity is a much
# shorter job than a full kernel-optimization campaign, and the outer loop still
# has other recipes to try.
DEFAULT_MAX_HOURS = 2.0

# The loop's campaign config and run state. ``CampaignConfigStore`` anchors it
# to the workspace, so ``--experiments-dir`` does not move it.
LOOP_CAMPAIGN_STATE = "forge_experiments"


def fused_module_path(recipe: Recipe) -> str:
    """Where the author must write this recipe's fused kernel.

    Derived before the campaign rather than left to the author, because a module
    created mid-campaign stays untracked: ``git add -u`` cannot commit it and
    ``git restore`` cannot revert it, so a rejected attempt's edits to it survive
    into the next one and a kept commit does not describe what was benchmarked.

    The name keeps the ``*_fused*`` marker :func:`emit._is_fused_module_name`
    recognizes, so the export and rollback paths still classify it correctly.
    """
    stem = Path(recipe.source_file).stem or "model"
    tag = re.sub(r"[^A-Za-z0-9]+", "_", recipe.pattern_id).strip("_").lower()[:48]
    return str(Path(recipe.source_file).parent / f"{stem}_fused_{tag or 'chain'}.py")


def _forge_loop_argv() -> list[str]:
    """Invoke forge-loop with the same interpreter and package as this process.

    An editable install or a multi-venv PATH could otherwise launch a different
    installed version than the code running right now.
    """
    if sys.executable:
        return [sys.executable, "-m", "kernelforge.cli"]
    exe = shutil.which("kernelforge")
    return [exe] if exe else ["kernelforge"]


@dataclass
class CampaignOutcome:
    """What one forge-loop campaign produced for a recipe."""

    result: ValidationResult
    experiment_id: str = ""


def _failed_campaign(note: str) -> CampaignOutcome:
    """A campaign that produced no verdict for the recipe to be judged on."""
    return CampaignOutcome(
        result=ValidationResult(
            correctness_passed=False,
            max_abs_err=None,
            rtol=None,
            kernel_speedup=None,
            eager_us=None,
            fused_us=None,
            kept=False,
            note=f"CAMPAIGN FAILED: {note}",
        )
    )


def _read_result_json(result_json: str) -> dict:
    """Read the campaign result the loop wrote to ``--result-json``."""
    try:
        return json.loads(Path(result_json).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        log.error("no usable forge-loop result at %s: %s", result_json, exc)
        return {}


def _read_harness_reports(report_log: str) -> list[dict]:
    """Every harness report the driver recorded during one campaign, in order."""
    reports: list[dict] = []
    try:
        text = Path(report_log).read_text(encoding="utf-8")
    except OSError:
        return reports
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            report = json.loads(line)
        except ValueError:
            continue
        if isinstance(report, dict):
            reports.append(report)
    return reports


def _best_harness_report(reports: list[dict], best_ms) -> dict:
    """The recorded report describing the candidate the loop settled on.

    The driver runs per baseline, per validation and per benchmark, so the last
    report is whatever ran last, not what was kept. The shim derives the wall
    time the loop reports as its best from ``fused_us``, so matching on it names
    the right report; without a best, the fastest fused report is that candidate.
    """
    usable = [
        r for r in reports if r.get("compiled") and isinstance(r.get("fused_us"), (int, float)) and not r.get("skipped")
    ]
    if not usable:
        return {}
    if isinstance(best_ms, (int, float)):
        return min(usable, key=lambda r: abs(float(r["fused_us"]) / 1000.0 - float(best_ms)))
    return min(usable, key=lambda r: float(r["fused_us"]))


def _worst_parity(report: dict) -> tuple[float | None, float | None]:
    """``(max_abs_err, snr_db)`` of the least accurate shape the harness compared.

    The shape the driver scores the loop on, so the manifest records the error
    that decided correctness rather than an average that hides it.
    """
    parity = report.get("parity") or []
    errs = [p.get("max_abs_err") for p in parity if isinstance(p.get("max_abs_err"), (int, float))]
    snrs = [p.get("snr_db") for p in parity if isinstance(p.get("snr_db"), (int, float))]
    return (max(errs) if errs else None, min(snrs) if snrs else None)


def _to_validation_result(payload: dict, target_speedup: float, reports: list[dict] | None = None) -> ValidationResult:
    """Translate the loop's campaign result into the fusion verdict shape.

    The loop reports its search outcome, not a per-candidate verdict. It anchors
    ``mean_case_speedup`` at 1.0 before the first iteration, so the number alone
    does not say a candidate was committed -- ``best_commit`` does. The keep
    decision is re-made here against the fusion bar, which is higher than the
    loop's own per-iteration improvement threshold.

    ``kernel_speedup`` is the loop's own number; the parity and per-arm timings
    come from the harness report behind it. See :class:`ValidationResult` for
    what that mixed provenance means for a reader.
    """
    speedup = payload.get("mean_case_speedup")
    speedup = float(speedup) if isinstance(speedup, (int, float)) else None
    committed = bool(str(payload.get("best_commit") or "").strip())
    kept = committed and speedup is not None and speedup >= target_speedup
    report = _best_harness_report(reports or [], payload.get("best_ms"))
    max_abs_err, snr_db = _worst_parity(report)
    eager_us = report.get("eager_us")
    fused_us = report.get("fused_us")
    if committed and not report:
        log.warning("no harness report recorded; manifest parity and timings stay null")
    return ValidationResult(
        correctness_passed=committed,
        max_abs_err=max_abs_err,
        # The harness reports SNR and absolute error, never a relative tolerance.
        rtol=None,
        kernel_speedup=speedup,
        eager_us=float(eager_us) if isinstance(eager_us, (int, float)) else None,
        fused_us=float(fused_us) if isinstance(fused_us, (int, float)) else None,
        kept=kept,
        note=(
            f"forge-loop best iteration {payload.get('best_iteration')}: "
            f"{payload.get('best_ms')} ms vs {payload.get('baseline_ms')} ms baseline"
            + (f", worst-shape SNR {snr_db:.2f} dB" if snr_db is not None else "")
            if committed
            else "forge-loop produced no validated candidate"
        ),
    )


def _safe_artifact_id(value: str, max_length: int = 80) -> str:
    """Return a bounded, filesystem-safe identifier for run artifacts.

    ``Recipe.pattern_id`` is not usable in a filename: LLM-proposed recipes are
    named ``llm:<pattern>`` and :func:`_combined_recipe` joins them with ``+``, so
    a combined id carries ``:`` and grows with the number of candidates. NFS
    rejects ``:`` in a path component with EINVAL, and every filesystem caps a
    component at NAME_MAX, so the raw id cannot be interpolated into a path.

    Truncation alone is unsafe because combined ids share long prefixes, so an
    over-long id keeps a readable prefix and gets a digest of the FULL original
    appended to keep distinct recipes distinct.
    """
    raw = str(value or "")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-") or "recipe"
    if len(safe) <= max_length:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{safe[:prefix_length].rstrip('._-')}_{digest}"


def build_forge_loop_command(
    recipe: Recipe,
    *,
    workspace: str,
    driver_path: str,
    experiments_dir: str,
    result_json: str,
    program_md_file: str,
    gpu_target: str = "",
    max_hours: float = DEFAULT_MAX_HOURS,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    supervisor_backend: str = "",
    model: str = "",
    agent_backend: str = "",
    agent_sandbox_mode: str = "",
    fused_module: str = "",
) -> list[str]:
    """Assemble the forge-loop invocation for one recipe.

    The fusion pipeline owns discovery, the harness and the serving gate, so the
    loop is told to skip its own task preparation.

    Experience is filed under the ``fusion`` producer and keyed on the chain,
    because several chains in one model file would otherwise share an address.
    Warm-start stays off: replaying a stored rewiring on top of a tree this
    pipeline has already prepared has to be proven before it is automatic.
    """
    source_files = [recipe.source_file] + ([fused_module] if fused_module else [])
    cmd = _forge_loop_argv() + [
        "forge-loop",
        "--workspace",
        workspace,
        "--kernel",
        recipe.source_file,
        "--driver",
        driver_path,
        "--experiments-dir",
        experiments_dir,
        "--result-json",
        result_json,
        "--program-md-file",
        program_md_file,
        "--snr-threshold",
        str(snr_threshold),
        "--max-hours",
        str(max(1.0, max_hours)),
        "--kernel-backend",
        FUSION_KERNEL_BACKEND,
        # The loop refuses a workspace on an unnamed, main or master branch.
        "--git-branch",
        SHADOW_BRANCH,
        "--task-type",
        "repository",
        "--source-files",
        ",".join(source_files),
        # Discovery already picked the chain and the harness already exists; the
        # loop's single-path preparer has a different contract and must not
        # rewrite either.
        "--no-prepare-task",
        "--experience-kb",
        "--producer",
        FUSION_PRODUCER,
        "--operator-name",
        recipe.pattern_id,
        "--no-kb-warmstart",
        # A fusion campaign is single-lane, and says so rather than inheriting
        # the loop's default. A lane is a full copy of the workspace measured on
        # its own, which is the one thing fusion cannot do: the benchmark and the
        # serving gate import the framework from its real install path, so every
        # lane would edit a copy and time the tree none of them touched. The
        # driver is outside the workspace as well -- it lives beside the run's
        # other artifacts -- and a round refuses to hand lanes a driver it cannot
        # copy. Concurrent lanes also need a provider that declares stop_hooks
        # and session_env, and that refusal lands before the first iteration, so
        # inheriting the default would fail runs on backends fusion otherwise
        # supports.
        "--lanes",
        "1",
    ]
    if gpu_target:
        cmd += ["--gpu-target", gpu_target]
    if supervisor_backend:
        cmd += ["--supervisor-backend", supervisor_backend]
    if model:
        cmd += ["--model", model]
    # The loop resolves its runtime from Config defaults, so a provider or
    # sandbox the caller chose would silently become `bypass` in the process
    # that actually edits the framework.
    if agent_backend:
        cmd += ["--agent-backend", agent_backend]
    if agent_sandbox_mode:
        cmd += ["--agent-sandbox-mode", agent_sandbox_mode]
    return cmd


def run_recipe_campaign(
    recipe: Recipe,
    *,
    workspace: str,
    harness_path: str,
    output_dir: str,
    experience: str = "",
    gpu: str = "0",
    gpu_target: str = "",
    max_hours: float = DEFAULT_MAX_HOURS,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    supervisor_backend: str = "",
    model: str = "",
    agent_backend: str = "",
    agent_sandbox_mode: str = "",
    shadow_env: dict[str, str] | None = None,
    fused_module: str = "",
) -> CampaignOutcome:
    """Author and validate one recipe by running a forge-loop campaign.

    ``shadow_env`` is empty unless the framework is a git checkout, where the
    shadow cannot be reached through a ``.git`` pointer file and the loop needs
    ``GIT_DIR`` in its environment to keep its commits out of that repository.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = _safe_artifact_id(recipe.pattern_id)

    env_flags = tuple(f for f in (recipe.env_flag or "").split() if f)
    report_log = str(out / f"harness_reports_{stem}.jsonl")
    Path(report_log).unlink(missing_ok=True)
    driver_path = write_driver(
        out / f"driver_{stem}.py",
        harness_path,
        env_flags,
        report_log=report_log,
        case_id=stem,
        fused_module=fused_module,
    )

    program_md_file = str(out / f"program_{stem}.md")
    Path(program_md_file).write_text(
        build_campaign_program_md(
            recipe,
            harness_path=harness_path,
            experience=experience,
            fused_module=fused_module,
        ),
        encoding="utf-8",
    )

    # Removed before the run, not just written after it: a campaign that dies
    # without writing one would otherwise hand the previous run's KEEP back.
    result_json = str(out / f"forge_loop_{stem}.json")
    Path(result_json).unlink(missing_ok=True)
    cmd = build_forge_loop_command(
        recipe,
        workspace=workspace,
        driver_path=driver_path,
        experiments_dir=str(out / "forge_experiments"),
        result_json=result_json,
        program_md_file=program_md_file,
        gpu_target=gpu_target,
        max_hours=max_hours,
        snr_threshold=snr_threshold,
        supervisor_backend=supervisor_backend,
        model=model,
        agent_backend=agent_backend,
        agent_sandbox_mode=agent_sandbox_mode,
        fused_module=fused_module,
    )

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = gpu
    env.update(shadow_env or {})
    log.info("forge-loop campaign for %s: %s", recipe.pattern_id, " ".join(cmd))

    collected: list[str] = []
    log_path = out / f"forge_loop_{stem}.log"
    try:
        # Same process group: an orchestrator timing this pipeline out signals
        # the group, and a detached campaign would go on holding the GPU and
        # editing the framework after the parent reported a failure.
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            collected.append(line)
            sys.stdout.write(line)
        returncode = proc.wait()
    except (OSError, subprocess.SubprocessError) as exc:
        log.error("forge-loop campaign for %s could not run: %s", recipe.pattern_id, exc)
        return _failed_campaign(f"{type(exc).__name__}: {exc}")

    log_path.write_text("".join(collected), encoding="utf-8")
    if returncode != 0:
        log.error("forge-loop campaign for %s exited %s", recipe.pattern_id, returncode)
        return _failed_campaign(f"forge-loop exited {returncode}")
    payload = _read_result_json(result_json)
    return CampaignOutcome(
        result=_to_validation_result(payload, target_speedup, _read_harness_reports(report_log)),
        experiment_id=str(payload.get("experiment_id") or ""),
    )


def build_campaign_program_md(
    recipe: Recipe, *, harness_path: str, experience: str = "", fused_module: str = ""
) -> str:
    """The task document handed to the loop's implementer for one recipe.

    Only the per-recipe facts live here. The durable authoring discipline is the
    fusion kernel backend's system prompt, so it is not repeated.

    The fused-module path is stated as a hard requirement because it is the one
    instruction the loop cannot recover from being ignored: a kernel written
    elsewhere is untracked, so the campaign scores a candidate it cannot keep.

    ``harness_path`` names an existing file to run, never one to write: the
    harness is authored before the campaign and is the measurement it is scored
    by. The authoring pass passes ``""`` and states its own contract instead.
    """
    hints = "\n".join(f"  - {h}" for h in recipe.source_hints) or "  (none recorded)"
    shapes = json.dumps(recipe.shapes or {}, indent=2, sort_keys=True)
    experience_block = f"\n## What earlier attempts established\n{experience}\n" if experience else ""
    module_block = (
        f"""
## Where the fused kernel goes (MANDATORY)
Write the fused kernel into exactly this file, which already exists and is empty:
    {fused_module}
Do NOT create any other new module. Only this file and the framework source file
above are tracked, and the loop can neither keep nor revert anything else — a
kernel written elsewhere scores as a validated candidate that then vanishes.
"""
        if fused_module
        else ""
    )
    harness_block = (
        f"""
## Kernel-validation harness (READ-ONLY)
The harness already exists at:
    {harness_path}
The driver the loop runs executes that harness and reads its JSON output. Do NOT
modify or recreate it. It matches the glob ``*harness*.py`` and is protected by
the in-session gate — any attempt to edit it will be rejected.
"""
        if harness_path
        else ""
    )
    return f"""# Fuse the {recipe.pattern_id} chain

## Target
- Framework source file to edit: {recipe.source_file}
- Env flag gating the fusion: {recipe.env_flag}
- {recipe.description}
{module_block}{harness_block}
## What to fuse
{recipe.fusion_math}

## How to localize it in the source
Grep the file for these anchors and fuse the chain they mark:
{hints}

## Representative decode shapes
{shapes}

## Correctness reference
{recipe.eager_reference_hint}
{experience_block}"""
