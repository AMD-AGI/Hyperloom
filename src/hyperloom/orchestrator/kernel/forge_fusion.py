# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run KernelForge's kernel fusion (``kernelforge.cli forge-fuse``) for a Hyperloom fusion request."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.codex_session import CodexSessionError
from hyperloom.common.env import is_truthy

from ..actions.executors._kernel_agent_tool import _run_subprocess

# The manifest verdict for "discovery never reached the model", added in manifest schema v2 alongside the ``error``
# block.
LLM_UNAVAILABLE_VERDICT = "llm_unavailable"

# ``fusion_loop.termination_reason`` values for a run that died before the loop ran a single attempt.
INFRA_ABORT_REASONS = frozenset({"harness_author_failed", "no_git_workspace"})
DEFAULT_TIMEOUT_SEC = 7200
_AGENT_BACKENDS = frozenset({"claude", "codex"})


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
    if _validated_agent_backend(agent_backend) == "codex":
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


class _ManifestRead(NamedTuple):
    """A parsed ``fusion_manifest.json``, or why it could not be read."""

    manifest: dict[str, Any] | None
    error: str


class _FusionManifest(NamedTuple):
    """The blocks of a ``fusion_manifest.json`` the Hyperloom result contract is built from."""

    raw: dict[str, Any]
    loop: dict[str, Any]
    compile_pass: dict[str, Any]
    patches: Any
    multi_patch: bool
    kept: bool


def _failed_fusion_result(output_dir: str) -> dict[str, Any]:
    """The REVERT-shaped result of a forge-fusion run that produced no usable fusion."""
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
    }


def _read_fusion_manifest(output_dir: str, rc: int) -> _ManifestRead:
    """Parse the ``fusion_manifest.json`` a forge-fusion run left in ``output_dir``."""
    manifest_path = Path(output_dir or ".") / "fusion_manifest.json"
    if not manifest_path.is_file():
        return _ManifestRead(None, f"no fusion_manifest.json at {manifest_path} (forge-fusion rc={rc})")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return _ManifestRead(None, f"fusion_manifest.json parse error: {exc!r}")
    return _ManifestRead(manifest, "")


def _fusion_manifest_blocks(raw: dict[str, Any]) -> _FusionManifest:
    """Split a parsed manifest into the blocks the result contract reads."""
    loop = raw.get("fusion_loop") or {}
    compile_pass = raw.get("compile_pass") or {}
    # The nomination envelope: a list (possibly empty) on the multi-patch path, None on the combine / single-patch
    # path.
    patches = raw.get("patches")
    multi_patch = isinstance(patches, list)
    # KernelForge runs the compile-pass shortcut INSTEAD of the authoring loop, so exactly one of these is populated.
    kept = bool(patches) if multi_patch else bool(loop.get("kept") or compile_pass.get("kept"))
    return _FusionManifest(raw, loop, compile_pass, patches, multi_patch, kept)


def _llm_unavailable_error(raw: dict[str, Any]) -> str:
    """Spell out how far discovery got before it gave up on reaching the model."""
    detail = raw.get("error") if isinstance(raw.get("error"), dict) else {}
    kind = str(detail.get("kind") or "unknown")
    attempts = detail.get("attempts")
    tried = f" after {attempts} attempt(s)" if attempts else ""
    message = str(detail.get("message") or "no detail reported")
    return f"forge-fusion never reached the LLM ({kind}{tried}): {message}"[:1500]


def _infrastructure_abort_error(raw: dict[str, Any], aborted: str) -> str:
    """Spell out which recipe discovery had located when the infrastructure gave way."""
    located = str((raw.get("fusion") or {}).get("env_flag") or "")
    return (
        f"forge-fusion aborted on infrastructure ({aborted}); "
        f"the loop reported no completed attempts and no result. "
        f"Recipe located by discovery: {located or 'unknown'}"
    )[:1500]


def _aborted_on_infrastructure(manifest: _FusionManifest, aborted: str) -> bool:
    """True when the loop stopped on infrastructure rather than on a judgement about the kernel."""
    return (
        aborted in INFRA_ABORT_REASONS
        and not manifest.kept
        and not manifest.compile_pass
        and not _attempted(manifest.loop)
    )


def _abort_result_fields(manifest: _FusionManifest) -> dict[str, Any] | None:
    """Result fields for a run that ended before judging the kernel, ``None`` when it judged it."""
    raw = manifest.raw
    if str(raw.get("verdict") or "").strip().lower() == LLM_UNAVAILABLE_VERDICT and not manifest.kept:
        # forge-fusion never reached the model, so this run holds no opinion about the kernel.
        return {
            "verdict": LLM_UNAVAILABLE_VERDICT,
            "error_class": LLM_UNAVAILABLE_VERDICT,
            "error": _llm_unavailable_error(raw),
        }
    aborted = str(manifest.loop.get("termination_reason") or "").strip().lower()
    if not _aborted_on_infrastructure(manifest, aborted):
        return None
    # The harness-authoring turn failed, or there was no git workspace to author in.
    return {
        "verdict": raw.get("verdict"),
        "error_class": aborted,
        # Explicit rather than re-deriving the reason set on the consumer side: the orchestrator bounds how
        # often it re-runs an abort, and a copied constant there would drift from the one above.
        "infrastructure_abort": True,
        "error": _infrastructure_abort_error(raw, aborted),
    }


def _fusion_source_file(manifest: _FusionManifest) -> str:
    """The framework file a kept fusion changes."""
    src_file = str((manifest.raw.get("fusion") or {}).get("source_file") or "")
    if not (manifest.multi_patch and manifest.patches):
        return src_file
    # On the multi-patch path the producer mirrors the STRONGEST sibling into the singular ``artifacts`` slot, but
    # ``fusion.source_file`` still names the top RECIPE -- which may be a compile-pass claim whose file differs
    # from the strongest authored patch.
    strongest = manifest.patches[0]
    return str(strongest.get("target_file") or strongest.get("source_file") or src_file)


def _fusion_kernel_speedup(manifest: _FusionManifest) -> Any:
    """The micro speedup the run measured for the fusion."""
    if manifest.compile_pass:
        return manifest.compile_pass.get("speedup")
    return (manifest.loop.get("best") or {}).get("kernel_speedup")


def _fusion_activation_fields(manifest: _FusionManifest) -> dict[str, Any]:
    """Result fields naming how the fused path is turned on, and what the producer measured for it."""
    if manifest.compile_pass:
        # The win is a flipped default in the framework's own source, so the patch carries it and there is no runtime
        # flag.
        return {
            "compile_pass_flag": manifest.compile_pass.get("flag"),
            "serving_speedup": manifest.compile_pass.get("speedup"),
        }
    best_flags = str(manifest.loop.get("best_env_flag") or "").split()
    return {
        # Fused arm = all confirmed flags ON; baseline arm = same flags OFF.
        "env_flags": dict.fromkeys(best_flags, "1"),
        "baseline_env_flags": dict.fromkeys(best_flags, "0"),
        "best_pattern": manifest.loop.get("best_pattern"),
    }


def _missing_keep_artifact(artifacts: dict[str, Any], patch: Any, source_file: str) -> str:
    """Name the artifact a kept fusion failed to export (``""`` when it exported them all).

    Integrate applies a fusion from a patch file, its root and a target, and returns without any of them, while
    ``ok`` satisfies the KERNEL-entry idempotency gate -- so reported as a success such a run is both dropped and
    never retried.
    """
    for name, present in (
        ("a patch", bool(patch)),
        ("the patch file it named", bool(patch) and Path(str(patch)).is_file()),
        ("a target file", bool(source_file)),
        ("a patch root", bool(artifacts.get("repo_root"))),
    ):
        if not present:
            return name
    return ""


def _normalize_manifest(output_dir: str, rc: int) -> dict[str, Any]:
    """Map forge-fusion's ``fusion_manifest.json`` -> Hyperloom result contract."""
    result = _failed_fusion_result(output_dir)
    read = _read_fusion_manifest(output_dir, rc)
    if read.manifest is None:
        result["error"] = read.error
        return result

    manifest = _fusion_manifest_blocks(read.manifest)
    abort_fields = _abort_result_fields(manifest)
    if abort_fields is not None:
        result.update(abort_fields)
        return result

    artifacts = read.manifest.get("artifacts") or {}
    patch = artifacts.get("patch")
    source_file = _fusion_source_file(manifest)
    kept = manifest.kept
    result.update(_fusion_activation_fields(manifest))

    missing = _missing_keep_artifact(artifacts, patch, source_file) if kept else ""
    if missing:
        # The failed/REVERT defaults above are the retryable shape.
        result["error_class"] = "fusion_artifact_missing"
        result["error"] = f"forge-fuse kept a fusion but exported no {missing}"
        return result

    result.update(
        {
            "status": "ok" if kept else "complete",
            "micro_decision": "candidate" if kept else "no_improvement",
            "decision": "KEEP" if kept else "REVERT",
            "kept": kept,
            "kernel_speedup": _fusion_kernel_speedup(manifest),
            "artifact_files": [c.get("path") for c in (artifacts.get("changes") or []) if c.get("path")],
            "patch": patch,
            # For integrate's patch-apply path.
            "source_file": source_file,
            # The root the patch was exported against, which may be a site-packages dir.
            "kernel_repo": str(artifacts.get("repo_root") or ""),
            "verdict": read.manifest.get("verdict"),
            # KernelForge validated this on its own -- kernel parity plus a serving smoke for an authored fusion, a
            # serving A/B for a claimed compile pass -- but integrate is what confirms the real e2e gain here.
            "requires_e2e_validation": kept,
        }
    )

    if manifest.multi_patch:
        # Carry the sibling envelopes through untouched so the consumer can enqueue each independently.
        result["patches"] = [dict(p) for p in manifest.patches]
        nomination = read.manifest.get("nomination")
        result["nomination"] = dict(nomination) if isinstance(nomination, dict) else None

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
            patch_file = published
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


class _SalvagedKeep(NamedTuple):
    """What a killed forge-fusion run left on disk of a fusion it had kept."""

    kept: bool = False
    speedup: Any = None
    env_flag: str = ""
    source_file: str = ""
    repo_root: str = ""
    patch: Any = None
    siblings: Any = None

    def filled_from(self, weaker: _SalvagedKeep) -> _SalvagedKeep:
        """These values, with every blank field taken from ``weaker``."""
        return _SalvagedKeep(
            kept=self.kept or weaker.kept,
            speedup=self.speedup or weaker.speedup,
            env_flag=self.env_flag or weaker.env_flag,
            source_file=self.source_file or weaker.source_file,
            repo_root=self.repo_root or weaker.repo_root,
            patch=self.patch or weaker.patch,
            siblings=weaker.siblings if weaker.siblings is not None else self.siblings,
        )


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Parse ``path`` as a JSON object, or ``{}`` when it is absent or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _keep_from_exported_patch(patch_path: Path) -> _SalvagedKeep:
    """The patch forge-fuse exported beside the run, which outranks every recorded copy."""
    return _SalvagedKeep(patch=str(patch_path)) if patch_path.is_file() else _SalvagedKeep()


def _keep_from_checkpoint(ckpt_path: Path) -> _SalvagedKeep:
    """What the pre-smoke keep checkpoint recorded."""
    ckpt = _read_json_dict(ckpt_path)
    if not ckpt.get("kept"):
        return _SalvagedKeep()
    return _SalvagedKeep(
        kept=True,
        speedup=ckpt.get("kernel_speedup"),
        env_flag=str(ckpt.get("env_flag") or ""),
        source_file=str(ckpt.get("source_file") or ""),
        repo_root=str(ckpt.get("repo_root") or ""),
    )


def _keep_from_manifest(manifest_path: Path) -> _SalvagedKeep:
    """What the aggregate ``fusion_manifest.json`` recorded."""
    manifest = _read_json_dict(manifest_path)
    if not manifest:
        return _SalvagedKeep()
    loop = manifest.get("fusion_loop") or {}
    compile_pass = manifest.get("compile_pass") or {}
    artifacts = manifest.get("artifacts") or {}
    return _SalvagedKeep(
        kept=bool(loop.get("kept") or compile_pass.get("kept")),
        speedup=(loop.get("best") or {}).get("kernel_speedup") or compile_pass.get("speedup"),
        env_flag=str(loop.get("best_env_flag") or ""),
        source_file=str((manifest.get("fusion") or {}).get("source_file") or ""),
        repo_root=str(artifacts.get("repo_root") or ""),
        patch=artifacts.get("patch"),
        siblings=manifest.get("patches"),
    )


def _keep_from_campaign_patches(root: Path, carried: _SalvagedKeep) -> _SalvagedKeep | None:
    """``carried`` rebuilt around the strongest per-recipe keeper on disk, ``None`` when there is none."""
    campaign = _campaign_patches_on_disk(root)
    if not campaign:
        return None
    strongest = campaign[0]
    return _SalvagedKeep(
        kept=True,
        speedup=strongest["micro_speedup"],
        # There is no checkpoint and no aggregate on this path, so the top-level flags are
        # empty too; the strongest row read its own out of the campaign's driver.
        env_flag=carried.env_flag or strongest["env_flag"],
        source_file=carried.source_file or strongest["target_file"],
        repo_root=carried.repo_root or strongest["kernel_repo"],
        patch=strongest["patch_path"],
        siblings=campaign,
    )


def _salvaged_keep_result(keep: _SalvagedKeep, root: Path) -> dict[str, Any]:
    """The KEEP-shaped fusion result a salvaged run publishes."""
    flags = [f for f in keep.env_flag.split() if f]
    result: dict[str, Any] = {
        "status": "ok",
        "engine": "forge_fusion",
        "micro_decision": "candidate",
        "decision": "KEEP",
        "kept": True,
        "kernel_speedup": keep.speedup,
        "env_flags": dict.fromkeys(flags, "1"),
        "baseline_env_flags": dict.fromkeys(flags, "0"),
        "artifact_files": [],
        "patch": str(keep.patch),
        "source_file": keep.source_file,
        "kernel_repo": keep.repo_root,
        "requires_e2e_validation": True,
        "salvaged": True,
        "workspace": str(root),
    }
    if isinstance(keep.siblings, list):
        # The consumer reads the nomination contract, so every sibling the manifest recorded is carried through beside
        # the singular slots.
        result["patches"] = keep.siblings
    return result


def salvage_forge_fusion_from_workspace(output_dir: str) -> dict[str, Any] | None:
    """Rebuild a KEEP result from pre-smoke checkpoint + patch after a kill."""
    root = Path(output_dir or "")
    if not root.is_dir():
        return None
    carried = _SalvagedKeep()
    for recorded in (
        _keep_from_exported_patch(root / "fusion.patch"),
        _keep_from_checkpoint(root / "kernel_keep_checkpoint.json"),
        _keep_from_manifest(root / "fusion_manifest.json"),
    ):
        carried = carried.filled_from(recorded)
    if not carried.kept or not carried.patch or not Path(str(carried.patch)).is_file():
        from_campaign = _keep_from_campaign_patches(root, carried)
        if from_campaign is None:
            return None
        carried = from_campaign
    return _salvaged_keep_result(carried, root)


def _timeout_result(output_dir: str, timeout_sec: int, exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    """Shape a timed-out forge-fusion run; salvage a micro KEEP when present."""
    cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or []))
    result = salvage_forge_fusion_from_workspace(output_dir) or _failed_fusion_result(output_dir)
    result["error_class"] = "subprocess_timeout"
    result["error"] = f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}"
    return result


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
    timeout_sec = _timeout_sec(args)
    try:
        rc, _stdout, _stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec, env=env, label="forge_fusion")
    except subprocess.TimeoutExpired as exc:
        return _timeout_result(output_dir, timeout_sec, exc)
    return _normalize_manifest(output_dir, rc)
