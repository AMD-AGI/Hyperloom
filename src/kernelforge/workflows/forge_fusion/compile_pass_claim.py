# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claim a vLLM compile pass that is implemented but off by default, by A/B serving throughput."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .emit import FusionFiles, export_artifacts
from .models import CompilePassOutcome, FusionArtifacts, Recipe
from .validate import DecodeLoad, SmokeServer, serving_smoke
from .vllm_passes import TargetRuntime, enable_pass_in_source, resolve_target_runtime, verify_pass_enabled

log = logging.getLogger("forge_fusion")


@dataclass(frozen=True)
class ClaimServing:
    """Where a claim's A/B arms serve: the framework install, the model, the device and the decode lengths."""

    framework: str
    framework_root: str
    model_path: str
    gpu: str
    isl: int
    osl: int


@dataclass(frozen=True)
class ClaimGate:
    """Whether a claim is A/B validated, and the serving speedup it must clear."""

    validate: bool
    target_speedup: float


@dataclass(frozen=True)
class ClaimExport:
    """Where a kept claim's patch goes: the output dir, the framework checkout, its pristine snapshot, the name."""

    out: Path
    repo_root: str
    pristine_dir: str
    patch_name: str = "fusion.patch"


def _run_single_compile_pass_claim(
    claim: Recipe,
    serving: ClaimServing,
    gate: ClaimGate,
    export: ClaimExport,
) -> tuple[CompilePassOutcome, Optional[FusionArtifacts]]:
    """Claim ONE compile pass: flip the default, A/B it, export its patch, restore."""
    runtime = resolve_target_runtime(serving.framework, framework_root=serving.framework_root)
    artifacts: Optional[FusionArtifacts] = None
    with _live_file_restored(claim.source_file):
        outcome = _run_compile_pass(claim, runtime, serving, gate, export.out)
        log.info(
            "compile pass %s: kept=%s speedup=%s note=%s",
            claim.compile_pass_flag,
            outcome.kept,
            outcome.speedup,
            outcome.note,
        )
        if export.repo_root and outcome.kept:
            artifacts = export_artifacts(
                export.repo_root,
                FusionFiles(claim.source_file),
                export.out / export.patch_name,
                pristine_dir=export.pristine_dir,
                snapshot_diff_only=True,
            )
    outcome.reverted = True  # the context manager just did it
    return outcome, artifacts


@contextlib.contextmanager
def _live_file_restored(path: str):
    """Guarantee byte-exact restoration of a live framework file on EVERY exit."""
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
        # Stay in the ``finally`` without returning: a return here would swallow an exception from the body when there
        # was nothing to restore.
        if original is not None and target is not None:
            try:
                if target.read_bytes() != original:
                    target.write_bytes(original)
                    if mode is not None:
                        os.chmod(target, mode)
                    log.info("restored %s to its pre-run contents", path)
            except OSError as exc:
                log.error("FAILED to restore %s (%s): the install may be left modified", path, exc)


def _serving_arm(label: str, server: SmokeServer, load: DecodeLoad, out: Path) -> tuple[bool, str, dict]:
    """One serving arm of the compile-pass A/B; returns ``(ok, reason, metrics)``."""
    metrics: dict = {}
    ok, reason = serving_smoke(server, load=load, metrics=metrics, log_path=str(out / f"compile_pass_{label}.log"))
    log.info("compile pass %s arm: ok=%s tok_s=%s reason=%s", label, ok, metrics.get("tok_s"), reason)
    return ok, reason, metrics


def _record_enabled_arm(outcome: CompilePassOutcome, ok: bool, reason: str, enabled: dict) -> CompilePassOutcome:
    """Record the enabled serving arm on ``outcome`` and decide whether the flip is kept."""
    target_speedup = outcome.target_speedup
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


def _run_compile_pass(
    recipe: Recipe,
    runtime: TargetRuntime,
    serving: ClaimServing,
    gate: ClaimGate,
    out: Path,
) -> CompilePassOutcome:
    """Claim a fusion the framework implements but ships switched OFF."""
    validate, target_speedup = gate.validate, gate.target_speedup
    load = DecodeLoad(isl=serving.isl, osl=serving.osl)

    def arm_server(env_flags: dict) -> SmokeServer:
        return SmokeServer(
            serving.model_path,
            env_flags,
            framework=runtime.framework,
            gpu=serving.gpu,
            launcher_exe=runtime.launcher_exe,
        )

    flag = recipe.compile_pass_flag
    outcome = CompilePassOutcome(
        flag=flag, config_file=recipe.source_file, source="default", target_speedup=target_speedup
    )
    if runtime.error:
        outcome.note = f"target runtime not pinned: {runtime.error}"
        return outcome

    baseline: dict = {}
    if validate:
        ok, reason, baseline = _serving_arm("baseline_disabled", arm_server({}), load, out)
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

    # Did the edit actually change what the target RESOLVES?
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

    # Fusion passes report what they rewrote at debug level; without this the run cannot tell "fused N sites" from
    # "matched nothing".
    ok, reason, enabled = _serving_arm("enabled", arm_server({"VLLM_LOGGING_LEVEL": "DEBUG"}), load, out)
    return _record_enabled_arm(outcome, ok, reason, enabled)
