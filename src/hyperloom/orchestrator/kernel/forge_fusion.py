# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run KernelForge's kernel fusion (``kernelforge.cli forge-fuse``) for a Hyperloom fusion request."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from hyperloom.common.codex_session import CodexSessionError
from hyperloom.common.env import is_truthy
from hyperloom.common.jsonio import read_json
from hyperloom.common.llm_config import AGENT_BACKEND_CHOICES
from kernelforge.agent_backends import watchdog_timeout_sec
from kernelforge.fusion.shadow_repo import reclaim_killed_shadow

from ..actions.executors._kernel_agent_tool import _run_subprocess

# The manifest verdict for "discovery never reached the model", added in manifest schema v2 alongside the ``error``
# block.
LLM_UNAVAILABLE_VERDICT = "llm_unavailable"

# ``fusion_loop.termination_reason`` values for a run that died before the loop ran a single attempt.
INFRA_ABORT_REASONS = frozenset({"harness_author_failed", "no_git_workspace"})
DEFAULT_TIMEOUT_SEC = 7200
_AGENT_BACKENDS = frozenset(AGENT_BACKEND_CHOICES)


def _validated_agent_backend(value: Any) -> str:
    """Return a canonical forge-fusion agent backend or raise."""
    backend = str(value or "").strip().lower()
    if backend not in _AGENT_BACKENDS:
        raise ValueError(f"agent_backend={value!r} is invalid; choose one of: {', '.join(sorted(_AGENT_BACKENDS))}")
    return backend


def _validated_agent_sandbox_mode(value: Any) -> str:
    """Delegate sandbox validation, including bypass opt-in, to Hyperloom."""
    stated = str(value or "").strip()
    if not stated:
        raise ValueError("agent_sandbox_mode is required")

    from hyperloom.common.codex_session import (
        resolve_codex_sandbox_mode,
    )

    return resolve_codex_sandbox_mode(sandbox_mode=stated)


def _author_env(agent_backend: str) -> dict[str, str]:
    """The environment forge-fuse runs its author under, without crossing provider shapes."""
    env = dict(os.environ)
    if _validated_agent_backend(agent_backend) in {"codex", "deepseek"}:
        return env

    from hyperloom.common import llm_config

    options = llm_config.claude_sdk_env_options(
        env=env,
        component="forge",
        operation="author_kernel",
    )
    resolved_env = options.get("env")
    if isinstance(resolved_env, dict):
        # Exactly the synthesizable subset: mirroring the subscription token into a key slot is what would disable it,
        # so the registry decides which forms may be copied here rather than a list kept in step by hand.
        for name in llm_config.ANTHROPIC_SYNTHESIZABLE_KEY_ENVS:
            value = str(resolved_env.get(name) or "").strip()
            if value:
                env.setdefault(name, value)
    # The authoring child gets this environment, not the resolved copy above, so the tag has to be merged in here or
    # the run arrives at the gateway anonymous.
    from hyperloom.common.llm_attribution import inject_env
    from hyperloom.common.llm_stability_env import apply_llm_stability_env

    inject_env(env, component="forge", operation="author_kernel", source=env)
    # claude's bypassPermissions refuses to start under root unless IS_SANDBOX=1.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.setdefault("IS_SANDBOX", "1")
    apply_llm_stability_env(env)
    return env


def _add_opt(cmd: list[str], args: dict[str, Any], key: str, flag: str, *, required: bool = False) -> None:
    val = args.get(key)
    if val in (None, ""):
        if required:
            raise ValueError(f"{key} is required")
        return
    cmd.extend([flag, str(val)])


def _build_cmd(args: dict[str, Any]) -> list[str]:
    agent_backend = _validated_agent_backend(args.get("agent_backend"))
    agent_sandbox_mode = _validated_agent_sandbox_mode(args.get("agent_sandbox_mode"))
    cmd = [sys.executable, "-m", "kernelforge.cli", "forge-fuse"]
    _add_opt(cmd, args, "trace_path", "--trace", required=True)
    _add_opt(cmd, args, "model_path", "--model-path", required=True)
    _add_opt(cmd, args, "framework", "--framework", required=True)
    _add_opt(cmd, args, "output_dir", "--output-dir", required=True)
    _add_opt(cmd, args, "discover_mode", "--discover")
    cmd.extend(["--agent-backend", agent_backend])
    _add_opt(cmd, args, "llm_model", "--model", required=True)
    cmd.extend(["--agent-sandbox-mode", agent_sandbox_mode])
    _add_opt(cmd, args, "max_turns", "--max-turns")
    # Omitted by the caller when no lane ceiling was derived; forge-fuse reads both an absent flag and an explicit 0
    # as "try every discovered recipe".
    _add_opt(cmd, args, "max_recipes", "--max-recipes")
    _add_opt(cmd, args, "gpu", "--gpu")
    _add_opt(cmd, args, "decode_batch", "--decode-batch")
    _add_opt(cmd, args, "ab_isl", "--ab-isl")
    _add_opt(cmd, args, "ab_osl", "--ab-osl")
    _add_opt(cmd, args, "framework_root", "--framework-root")
    _add_opt(cmd, args, "tp", "--tp")
    # Attention is not always sharded by the serving --tp (DP-attention splits it),
    # and forge-fuse stamps the harness group/head dims from this.
    _add_opt(cmd, args, "attn_tp", "--attn-tp")
    _add_opt(cmd, args, "block_size", "--block-size")
    _add_opt(cmd, args, "max_model_len", "--max-model-len")
    # Nominate one independent sibling patch per confirmed pattern by default (the multi-patch contract).
    if bool(args.get("fuse_all_confirmed", False)):
        cmd.append("--fuse-all-confirmed")
    if is_truthy(args.get("verbose", False)):
        cmd.append("--verbose")
    return cmd


def _timeout_sec(args: dict[str, Any]) -> int:
    """Resolve the forge-fusion subprocess wall-clock timeout."""
    raw = args.get("timeout") or args.get("timeout_sec") or os.environ.get("FORGE_FUSION_TIMEOUT")
    try:
        return max(1, int(float(raw or DEFAULT_TIMEOUT_SEC)))
    except (OverflowError, TypeError, ValueError):
        return DEFAULT_TIMEOUT_SEC


def _attempted(loop: dict[str, Any]) -> bool:
    """Whether ``fusion_loop`` reports at least one completed attempt."""
    try:
        return int(loop.get("attempts") or 0) > 0
    except (TypeError, ValueError):
        return False


def _llm_unavailable_fields(m: dict[str, Any]) -> dict[str, Any]:
    """Result fields for a run whose discovery never reached the model, with the reason the manifest gave."""
    detail = m.get("error") if isinstance(m.get("error"), dict) else {}
    kind = str(detail.get("kind") or "unknown")
    attempts = detail.get("attempts")
    tried = f" after {attempts} attempt(s)" if attempts else ""
    message = str(detail.get("message") or "no detail reported")
    return {
        "verdict": LLM_UNAVAILABLE_VERDICT,
        "error_class": LLM_UNAVAILABLE_VERDICT,
        "error": f"forge-fusion never reached the LLM ({kind}{tried}): {message}"[:1500],
    }


def _fusion_target_file(fusion: dict[str, Any], patches: Any) -> str:
    """The file the kept fusion patches: the strongest sibling's on the multi-patch path, else the recipe's."""
    src_file = str(fusion.get("source_file") or "")
    if isinstance(patches, list) and patches:
        # On the multi-patch path the producer mirrors the STRONGEST sibling into the singular ``artifacts`` slot, but
        # ``fusion.source_file`` still names the top RECIPE -- which may be a compile-pass claim whose file differs
        # from the strongest authored patch.
        src_file = str(patches[0].get("target_file") or patches[0].get("source_file") or src_file)
    return src_file


def _outcome_fields(
    m: dict[str, Any], loop: dict[str, Any], compile_pass: dict[str, Any], kept: bool, src_file: str
) -> dict[str, Any]:
    """Result fields for a run that reached a verdict: its flags, its artifacts and the KEEP or REVERT it implies."""
    artifacts = m.get("artifacts") or {}
    changed = [c.get("path") for c in (artifacts.get("changes") or []) if c.get("path")]
    if compile_pass:
        # The win is a flipped default in the framework's own source, so the patch carries it and there is no runtime
        # flag.
        speedup = compile_pass.get("speedup")
        fields: dict[str, Any] = {
            "compile_pass_flag": compile_pass.get("flag"),
            "serving_speedup": speedup,
        }
    else:
        speedup = (loop.get("best") or {}).get("kernel_speedup")
        best_flags = str(loop.get("best_env_flag") or "").split()
        fields = {
            # Fused arm = all confirmed flags ON; baseline arm = same flags OFF.
            "env_flags": dict.fromkeys(best_flags, "1"),
            "baseline_env_flags": dict.fromkeys(best_flags, "0"),
            "best_pattern": loop.get("best_pattern"),
        }

    # Integrate applies a fusion from a patch file, its root and a target, and returns without any of them, while
    # ``ok`` satisfies the KERNEL-entry idempotency gate -- so reported as a success such a run is both dropped and
    # never retried.
    patch = artifacts.get("patch")
    if kept:
        # A falsy patch fails the first entry before the patch-file entry is read.
        for name, present in (
            ("a patch", bool(patch)),
            ("the patch file it named", Path(str(patch)).is_file()),
            ("a target file", bool(src_file)),
            ("a patch root", bool(artifacts.get("repo_root"))),
        ):
            if not present:
                # The failed/REVERT defaults are the retryable shape.
                fields["error_class"] = "fusion_artifact_missing"
                fields["error"] = f"forge-fuse kept a fusion but exported no {name}"
                return fields

    fields.update(
        {
            "status": "ok" if kept else "complete",
            "micro_decision": "candidate" if kept else "no_improvement",
            "decision": "KEEP" if kept else "REVERT",
            "kept": kept,
            "kernel_speedup": speedup,
            "artifact_files": changed,
            "patch": patch,
            # For integrate's patch-apply path.
            "source_file": src_file,
            # The root the patch was exported against, which may be a site-packages dir.
            "kernel_repo": str(artifacts.get("repo_root") or ""),
            "verdict": m.get("verdict"),
            # KernelForge validated this on its own -- kernel parity plus a serving smoke for an authored fusion, a
            # serving A/B for a claimed compile pass -- but integrate is what confirms the real e2e gain here.
            "requires_e2e_validation": kept,
        }
    )

    patches = m.get("patches")
    if isinstance(patches, list):
        # Carry the sibling envelopes through untouched so the consumer can enqueue each independently.
        fields["patches"] = [dict(p) for p in patches]
        nomination = m.get("nomination")
        fields["nomination"] = dict(nomination) if isinstance(nomination, dict) else None
    return fields


def _normalize_manifest(output_dir: str, rc: int) -> dict[str, Any]:
    """Map forge-fusion's ``fusion_manifest.json`` -> Hyperloom result contract."""
    result: dict[str, Any] = {
        "status": "failed",
        "engine": "forge_fusion",
        "micro_decision": "failed",
        "decision": "REVERT",
        "kept": False,
        "kernel_speedup": None,
        "env_flags": {},
        "baseline_env_flags": {},
        "artifact_files": [],
        "patch": None,
        "requires_e2e_validation": False,
        "workspace": output_dir,
    }
    # ``Path("")`` is the current directory.
    manifest_path = Path(output_dir) / "fusion_manifest.json"
    if not manifest_path.is_file():
        result["error"] = f"no fusion_manifest.json at {manifest_path} (forge-fusion rc={rc})"
        return result
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"fusion_manifest.json parse error: {exc!r}"
        return result

    loop = m.get("fusion_loop") or {}
    compile_pass = m.get("compile_pass") or {}
    fusion = m.get("fusion") or {}
    # The nomination envelope: a list (possibly empty) on the multi-patch path, None on the combine / single-patch
    # path.
    patches = m.get("patches")
    multi_patch = isinstance(patches, list)
    # KernelForge runs the compile-pass shortcut INSTEAD of the authoring loop, so exactly one of these is populated.
    kept = bool(patches) if multi_patch else bool(loop.get("kept") or compile_pass.get("kept"))

    if str(m.get("verdict") or "").strip().lower() == LLM_UNAVAILABLE_VERDICT and not kept:
        # forge-fusion never reached the model, so this run holds no opinion about the kernel.
        result.update(_llm_unavailable_fields(m))
        return result

    aborted = str(loop.get("termination_reason") or "").strip().lower()
    if aborted in INFRA_ABORT_REASONS and not kept and not compile_pass and not _attempted(loop):
        # The loop stopped on infrastructure, not on a judgement about the kernel: the harness-authoring turn failed,
        # or there was no git workspace to author in.
        located = str(fusion.get("env_flag") or "")
        result.update(
            {
                "verdict": m.get("verdict"),
                "error_class": aborted,
                # Explicit rather than re-deriving the reason set on the consumer side: the orchestrator bounds how
                # often it re-runs an abort, and a copied constant there would drift from the one above.
                "infrastructure_abort": True,
                "error": (
                    f"forge-fusion aborted on infrastructure ({aborted}); "
                    f"the loop reported no completed attempts and no result. "
                    f"Recipe located by discovery: {located or 'unknown'}"
                )[:1500],
            }
        )
        return result

    result.update(_outcome_fields(m, loop, compile_pass, kept, _fusion_target_file(fusion, patches)))
    return result


_DIFF_TARGET_RE = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)
#: ``driver_shim`` renders this as a module-level literal, so it reads back without importing.
_DRIVER_ENV_FLAGS_RE = re.compile(r"^ENV_FLAGS\s*=\s*(.+)$", re.MULTILINE)

#: Where a campaign's best manifest sits under the shadow repo it was built in.
_SHADOW_EXPERIMENTS_MARKER = "/forge_experiments/"


def _campaign_env_flag(root: Path, stem: str) -> str:
    """Read back the env flags the campaign gated its fused path behind.

    ``run_campaign`` splits ``recipe.env_flag`` on whitespace and renders the result into
    ``driver_<stem>.py`` beside the loop result, so the driver is where a killed run still
    says which flags turn the fusion on.
    """
    driver = root / f"driver_{stem}.py"
    try:
        source = driver.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = _DRIVER_ENV_FLAGS_RE.search(source)
    if not match:
        return ""
    try:
        flags = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return ""
    if isinstance(flags, str):
        flags = (flags,)
    if not isinstance(flags, (tuple, list)):
        return ""
    return " ".join(str(flag).strip() for flag in flags if str(flag).strip())


def _patch_target_file(patch_path: Path) -> str:
    """Return the first path a unified diff touches, or empty when it names none."""
    try:
        head = patch_path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return ""
    match = _DIFF_TARGET_RE.search(head)
    return match.group(1) if match else ""


def _published_best_patch(best_manifest: Path) -> Path | None:
    """Return the patch the shadow repo published for a campaign's best iteration."""
    try:
        manifest = json.loads(best_manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    rel = str(manifest.get("patch_path") or "")
    if not rel:
        return None
    # ``patch_path`` is relative to the experiments root, which holds the ``best/`` the manifest sits in.
    patch = best_manifest.parent.parent / rel
    return patch if patch.is_file() else None


def _campaign_patches_on_disk(root: Path) -> list[dict[str, Any]]:
    """Collect the per-recipe keepers a killed run left proof of.

    ``run_campaign`` writes ``forge_loop_<pattern_id>.json`` as each campaign returns and
    publishes its winning iteration to the shadow repo's ``forge_experiments/best/``, while
    ``on_keep`` exports ``fusion_<pattern_id>.patch`` only once the loop gates the keeper and
    the aggregate ``fusion_manifest.json`` only once every campaign has returned. A run
    killed at any of those points leaves proven work with nothing the aggregate can see.

    Args:
        root: The fusion output directory.

    Returns:
        Nomination rows for every campaign whose own loop result says it won, strongest
        first.
    """
    rows: list[dict[str, Any]] = []
    for loop_file in sorted(root.glob("forge_loop_*.json")):
        stem = loop_file.name[len("forge_loop_") : -len(".json")]
        try:
            loop = json.loads(loop_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loop, dict) or not loop.get("improved"):
            continue
        best_manifest = str(loop.get("best_manifest") or "")
        patch_file = root / f"fusion_{stem}.patch"
        if not patch_file.is_file():
            published = _published_best_patch(Path(best_manifest)) if best_manifest else None
            if published is None:
                continue
            # Exported as ``on_keep`` would have: the published copy sits in the workspace's own
            # forge_experiments, which the next Forge campaign in that repository archives away before
            # the queued integrate reads it.
            shutil.copyfile(published, patch_file)
        target_file = _patch_target_file(patch_file)
        if not target_file:
            continue
        env_flag = _campaign_env_flag(root, stem)
        if not env_flag:
            # An authored fusion is env-gated, and the flag is the only way the fused path
            # reaches the re-baseline server. Salvaging the patch without it queues a win to be
            # measured on the eager path and REVERTed, which loses it one stage later instead.
            sys.stderr.write(
                f"forge_fusion: not salvaging {stem}: its driver names no env flag, and without one "
                "the patch would be re-baselined un-gated and rejected\n"
            )
            continue
        rows.append(
            {
                "kernel_name": stem,
                "patch_path": str(patch_file),
                "target_file": target_file,
                "env_flag": env_flag,
                "kernel_repo": best_manifest.split(_SHADOW_EXPERIMENTS_MARKER)[0]
                if _SHADOW_EXPERIMENTS_MARKER in best_manifest
                else "",
                "micro_speedup": float(loop.get("total_speedup") or 0.0),
                "kind": "fusion",
            }
        )
    rows.sort(key=lambda row: row["micro_speedup"], reverse=True)
    return rows


def _salvaged_speedup(
    ckpt: dict[str, Any], manifest: dict[str, Any], loop: dict[str, Any], compile_pass: dict[str, Any]
) -> Any:
    """The checkpointed kernel speedup, else the one the aggregate manifest recorded."""
    speedup = ckpt.get("kernel_speedup")
    if manifest:
        # A checkpointed ``kernel_speedup`` of 0 is a measurement, so the manifest answers
        # for it only when the manifest is there to answer.
        speedup = speedup or (loop.get("best") or {}).get("kernel_speedup") or compile_pass.get("speedup")
    return speedup


def salvage_forge_fusion_from_workspace(output_dir: str) -> dict[str, Any] | None:
    """Rebuild a KEEP result from pre-smoke checkpoint + patch after a kill."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    # A killed run leaves either file truncated or mid-write, so neither is read strictly.
    read = partial(read_json, default={}, require_dict=True, errors="replace")
    manifest = read(root / "fusion_manifest.json")
    # Only a checkpoint that kept says anything about the run.
    ckpt = read(root / "kernel_keep_checkpoint.json")
    ckpt = ckpt if ckpt.get("kept") else {}
    loop, compile_pass, artifacts, fusion = (
        manifest.get(block) or {} for block in ("fusion_loop", "compile_pass", "artifacts", "fusion")
    )
    kept = bool(ckpt or loop.get("kept") or compile_pass.get("kept"))
    speedup = _salvaged_speedup(ckpt, manifest, loop, compile_pass)
    env_flag, source_file, repo_root = (
        str(ckpt.get(key) or fallback or "")
        for key, fallback in (
            ("env_flag", loop.get("best_env_flag")),
            ("source_file", fusion.get("source_file")),
            ("repo_root", artifacts.get("repo_root")),
        )
    )
    patch = str(artifacts.get("patch") or "")
    siblings = manifest.get("patches")
    patch_path = root / "fusion.patch"
    if patch_path.is_file():
        patch = str(patch_path)
    if not kept or not Path(patch).is_file():
        campaign = _campaign_patches_on_disk(root)
        if not campaign:
            return None
        strongest = campaign[0]
        siblings = campaign
        patch = strongest["patch_path"]
        speedup = strongest["micro_speedup"]
        # There is no checkpoint and no aggregate on this path, so the top-level flags are
        # empty too; the strongest row read its own out of the campaign's driver.
        env_flag, source_file, repo_root = (
            current or strongest[key]
            for current, key in ((env_flag, "env_flag"), (source_file, "target_file"), (repo_root, "kernel_repo"))
        )
    flags = env_flag.split()
    result: dict[str, Any] = {
        "status": "ok",
        "engine": "forge_fusion",
        "micro_decision": "candidate",
        "decision": "KEEP",
        "kept": True,
        "kernel_speedup": speedup,
        "env_flags": dict.fromkeys(flags, "1"),
        "baseline_env_flags": dict.fromkeys(flags, "0"),
        "artifact_files": [],
        "patch": str(patch),
        "source_file": source_file,
        "kernel_repo": repo_root,
        "requires_e2e_validation": True,
        "salvaged": True,
        "workspace": str(root),
    }
    if isinstance(siblings, list):
        # The consumer reads the nomination contract, so every sibling the manifest recorded is carried through beside
        # the singular slots.
        result["patches"] = siblings
    return result


def _timeout_result(output_dir: str, timeout_sec: int, exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    """Shape a timed-out forge-fusion run; salvage a micro KEEP when present."""
    salvaged = salvage_forge_fusion_from_workspace(output_dir)
    cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or []))
    timeout_error = f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}"
    if salvaged:
        salvaged["error_class"] = "subprocess_timeout"
        salvaged["error"] = timeout_error
        return salvaged
    return {
        "status": "failed",
        "engine": "forge_fusion",
        "micro_decision": "failed",
        "decision": "REVERT",
        "kept": False,
        "kernel_speedup": None,
        "env_flags": {},
        "baseline_env_flags": {},
        "artifact_files": [],
        "patch": None,
        "requires_e2e_validation": False,
        "workspace": str(output_dir or ""),
        "error_class": "subprocess_timeout",
        "error": timeout_error,
    }


async def run_forge_fusion(args: dict[str, Any]) -> dict[str, Any]:
    """Run forge-fuse for ``args`` and return its result in Hyperloom's fusion contract."""
    try:
        cmd = _build_cmd(args)
    except (ValueError, CodexSessionError) as exc:
        return {
            "status": "failed",
            "engine": "forge_fusion",
            "micro_decision": "failed",
            "decision": "REVERT",
            "kept": False,
            "error_class": exc.__class__.__name__,
            "error": repr(exc),
        }
    env = _author_env(str(args.get("agent_backend") or ""))
    output_dir = str(args.get("output_dir") or "")
    # The output dir is keyed on the task, so a run that dies before writing new artifacts must not salvage the
    # previous run's KEEP as its own.
    output_root = Path(output_dir or ".")
    for stale_name in (
        "fusion_manifest.json",
        "kernel_keep_checkpoint.json",
        "fusion.patch",
    ):
        (output_root / stale_name).unlink(missing_ok=True)
    for stale_glob in ("fusion_*.patch", "forge_loop_*.json"):
        for stale in output_root.glob(stale_glob):
            stale.unlink(missing_ok=True)
    # The lane's budget is when the work must end, and every recipe's forge-loop is told so. The kill waits a watchdog
    # grace past it to backstop a run that hangs, not to race one finishing on time.
    budget_sec = _timeout_sec(args)
    cmd += ["--deadline-unix", str(time.time() + budget_sec)]
    timeout_sec = int(watchdog_timeout_sec(budget_sec))
    try:
        rc, _stdout, _stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec, env=env, label="forge_fusion")
    except subprocess.TimeoutExpired as exc:
        result = _timeout_result(output_dir, timeout_sec, exc)
    else:
        result = _normalize_manifest(output_dir, rc)
    # A run killed before its own reset left the candidate it was measuring live in the framework tree.
    await asyncio.to_thread(reclaim_killed_shadow, output_dir)
    return result
