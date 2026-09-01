#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run KernelForge's kernel fusion as a Hyperloom kernel-agent tool.

The orchestrator writes an input JSON with one validated agent backend, model,
and sandbox policy and calls this script; the autonomous fusion pipeline itself
lives in KernelForge and is invoked as ``kernelforge forge-fuse``.

It emits a ``fusion_manifest.json``; this wrapper normalizes that into the
Hyperloom kernel-result contract (a ``FORGE_FUSION_RESULT_BEGIN/END`` stdout
sentinel + an on-disk ``result.json``) that ``run_fusion_handler`` parses. A KEPT
result already carries validation on the KernelForge side -- kernel parity plus a
serving smoke for an authored fusion, a serving A/B for a claimed compile pass --
so ``requires_e2e_validation`` is set for the orchestrator's integrate/re-baseline
gate to confirm the end-to-end gain and apply the patch + env flags.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

# Sibling import: kernel-agent tools cannot rely on the ``hyperloom`` import root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _io_utils import truthy  # noqa: E402

sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parent / "backends"))
from _llm_stability_env import apply_llm_stability_env  # noqa: E402

sys.path.pop(0)

RESULT_BEGIN = "FORGE_FUSION_RESULT_BEGIN"
RESULT_END = "FORGE_FUSION_RESULT_END"

# The manifest verdict for "discovery never reached the model", added in manifest
# schema v2 alongside the ``error`` block. It is NOT a statement about the kernel,
# so it must not be normalized into an optimization outcome -- see
# _normalize_manifest.
LLM_UNAVAILABLE_VERDICT = "llm_unavailable"

# ``fusion_loop.termination_reason`` values for a run that died before the loop
# ran a single attempt. Like an LLM outage, this is infrastructure failing rather
# than a statement about the kernel -- see _normalize_manifest. Keyed on the
# termination reason, which is KernelForge's contract for why the loop stopped,
# so a new abort path inherits the handling instead of silently regressing into
# the no-KEEP shape.
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

    from hyperloom.common.codex_session import (  # noqa: PLC0415 - standalone import-light
        resolve_codex_sandbox_mode,
    )

    return resolve_codex_sandbox_mode(sandbox_mode=stated)


def _inject_author_gateway_env(agent_backend: str) -> None:
    """Prepare the selected author runtime without crossing provider shapes.

    Codex needs none of the Claude-only auth aliases, root sandbox escape, or
    stability variables, so its environment is left untouched. Claude keeps the
    established behavior: credential alias resolution is delegated to
    :mod:`hyperloom.common.llm_config`, then Claude-specific process defaults are
    applied. Selection is driven only by the explicit backend contract, never by
    a model-name prefix. A ``CLAUDE_CODE_OAUTH_TOKEN`` is inherited as-is and
    deliberately not mirrored into a key var, since either one would disable it.
    """
    if _validated_agent_backend(agent_backend) == "codex":
        return

    from hyperloom.common import llm_config  # noqa: PLC0415 - standalone import-light

    options = llm_config.claude_sdk_env_options(
        env=os.environ,
        component="forge",
        operation="author_kernel",
    )
    resolved_env = options.get("env")
    if isinstance(resolved_env, dict):
        # Exactly the synthesizable subset: mirroring the subscription token
        # into a key slot is what would disable it, so the registry decides
        # which forms may be copied here rather than a list kept in step by hand.
        for name in llm_config.ANTHROPIC_SYNTHESIZABLE_KEY_ENVS:
            value = str(resolved_env.get(name) or "").strip()
            if value:
                os.environ.setdefault(name, value)
    # The authoring child inherits this process's environment, not the resolved
    # copy above, so the tag has to be merged in here or the run arrives at the
    # gateway anonymous. It is merged rather than copied from ``resolved_env``
    # because that copy has been through ``_expand_env_refs``: writing it back
    # would publish the operator's gateway secret in this process's environment,
    # whereas merging preserves the ``${VAR}`` the child resolves for itself.
    from hyperloom.common.llm_attribution import inject_env  # noqa: PLC0415 - standalone import-light

    inject_env(os.environ, component="forge", operation="author_kernel")
    # claude's bypassPermissions refuses to start under root unless IS_SANDBOX=1.
    # Only set it when actually running as root so we do not defeat the guard
    # for non-root sessions that never needed the escape hatch.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    apply_llm_stability_env(os.environ)


def _load_input_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"--input-json must contain a JSON object: {path}")
    return data


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
    _add_opt(cmd, args, "gpu", "--gpu")
    _add_opt(cmd, args, "decode_batch", "--decode-batch")
    _add_opt(cmd, args, "ab_isl", "--ab-isl")
    _add_opt(cmd, args, "ab_osl", "--ab-osl")
    _add_opt(cmd, args, "framework_root", "--framework-root")
    _add_opt(cmd, args, "tp", "--tp")
    _add_opt(cmd, args, "block_size", "--block-size")
    _add_opt(cmd, args, "max_model_len", "--max-model-len")
    # Author all source-confirmed patterns together by default; set
    # fuse_all_confirmed=false to author only the top recipe.
    if bool(args.get("fuse_all_confirmed", True)):
        cmd.append("--fuse-all-confirmed")
    if truthy(args.get("verbose", False)):
        cmd.append("--verbose")
    return cmd


def _timeout_sec(args: dict[str, Any]) -> int:
    """Resolve the forge-fusion subprocess wall-clock timeout."""
    raw = args.get("timeout") or args.get("timeout_sec") or os.environ.get("FORGE_FUSION_TIMEOUT")
    try:
        return max(1, int(float(raw or DEFAULT_TIMEOUT_SEC)))
    except (OverflowError, TypeError, ValueError):
        return DEFAULT_TIMEOUT_SEC


def _new_session_kwargs() -> dict[str, bool]:
    """Popen kwargs that isolate the child into its own killable session."""
    return {"start_new_session": True} if os.name == "posix" else {}


def _terminate_process_tree(proc: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    """Best-effort teardown for the forge-fusion subprocess and descendants."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            pgid = os.getpgid(proc.pid)
            own_pgid = os.getpgid(0)
        except OSError:
            pgid = None
            own_pgid = None
        if pgid is not None and pgid != own_pgid:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=grace_seconds)
                return
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
                return
    try:
        proc.terminate()
        proc.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


def _run_with_tree_timeout(cmd: list[str], timeout_sec: int) -> subprocess.CompletedProcess:
    """Run forge-fusion in a killable process group and reap it on timeout."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **_new_session_kwargs(),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = ""
        raise subprocess.TimeoutExpired(cmd, timeout_sec, output=stdout, stderr=stderr)


def _attempted(loop: dict[str, Any]) -> bool:
    """Whether ``fusion_loop`` reports at least one completed attempt.

    Read defensively: the count crosses a repository boundary, and a non-numeric
    value must not read as "attempted" -- that would drop the run back into the
    ``complete``/``no_improvement`` shape this guards against.
    """
    try:
        return int(loop.get("attempts") or 0) > 0
    except (TypeError, ValueError):
        return False


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
        "workspace": str(output_dir or ""),
    }
    manifest_path = Path(output_dir or ".") / "fusion_manifest.json"
    if not manifest_path.is_file():
        result["error"] = f"no fusion_manifest.json at {manifest_path} (forge-fusion rc={rc})"
        return result
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"fusion_manifest.json parse error: {exc!r}"
        return result

    loop = m.get("fusion_loop") or {}
    compile_pass = m.get("compile_pass") or {}
    # KernelForge runs the compile-pass shortcut INSTEAD of the authoring loop, so
    # exactly one of these is populated. ``validation`` is not consulted: it is the
    # same object as ``fusion_loop.best``.
    kept = bool(loop.get("kept") or compile_pass.get("kept"))

    if str(m.get("verdict") or "").strip().lower() == LLM_UNAVAILABLE_VERDICT and not kept:
        # forge-fusion never reached the model, so this run holds no opinion about the
        # kernel. The default no-KEEP shape below would report it as
        # ``complete``/``no_improvement``, which is wrong twice over: it records an
        # outage as an optimization result, AND ``complete`` satisfies the KERNEL-entry
        # idempotency gate (``_fusion_required_before_kernel_opt``), so one gateway
        # blip would skip fusion for the whole remaining session and the model would
        # never be fusion-optimized at all. The timeout shape below is the established
        # way to say "infrastructure failed, this is retryable".
        #
        # Guarded on ``not kept`` so this can never discard a validated fusion. That
        # combination should be impossible -- forge-fusion only overrides the verdict
        # when discovery raised, in which case it proposed no recipes and the loop
        # never ran -- but that invariant lives in another repo and nothing here can
        # enforce it, while the cost of being wrong is throwing away a measured patch.
        #
        # ``result`` still carries its failed/REVERT/not-kept defaults from above, so
        # only the outage's identity has to be added.
        detail = m.get("error") if isinstance(m.get("error"), dict) else {}
        kind = str(detail.get("kind") or "unknown")
        attempts = detail.get("attempts")
        tried = f" after {attempts} attempt(s)" if attempts else ""
        message = str(detail.get("message") or "no detail reported")
        result.update(
            {
                "verdict": LLM_UNAVAILABLE_VERDICT,
                "error_class": LLM_UNAVAILABLE_VERDICT,
                "error": f"forge-fusion never reached the LLM ({kind}{tried}): {message}"[:1500],
            }
        )
        return result

    aborted = str(loop.get("termination_reason") or "").strip().lower()
    if aborted in INFRA_ABORT_REASONS and not kept and not compile_pass and not _attempted(loop):
        # The loop stopped on infrastructure, not on a judgement about the kernel:
        # the harness-authoring turn failed, or there was no git workspace to
        # author in. Discovery had already run, so a recipe is named in the
        # manifest; nothing ever got to measure it.
        #
        # The default no-KEEP shape below would report that as
        # ``complete``/``no_improvement`` -- an outage recorded as an optimization
        # result, AND ``complete`` satisfies the KERNEL-entry idempotency gate
        # (``_fusion_required_before_kernel_opt``), so one abort would skip fusion
        # for the whole remaining session even though every later entry could have
        # retried it. Same reasoning, and the same retryable shape, as the LLM
        # outage above.
        #
        # Guarded on ``not kept`` and ``not compile_pass`` so a measured result is
        # never discarded. The attempt count is a weaker signal than it looks:
        # KernelForge's abort handler builds a fresh ``LoopResult`` whose history
        # is empty, so ``attempts`` reads 0 even when earlier recipes ran full
        # campaigns. It is kept as a cheap filter for the ordinary case, but the
        # load-bearing guards are the two above -- they are what a real
        # measurement would set.
        #
        # ``result`` still carries its failed/REVERT/not-kept defaults from above,
        # so only the abort's identity has to be added. The located recipe rides
        # in the message rather than in ``env_flags``: nothing measured it, and
        # ``env_flags`` means "flags this run confirmed".
        located = str((m.get("fusion") or {}).get("env_flag") or "")
        result.update(
            {
                "verdict": m.get("verdict"),
                "error_class": aborted,
                # Explicit rather than re-deriving the reason set on the consumer
                # side: the orchestrator bounds how often it re-runs an abort, and
                # a copied constant there would drift from the one above.
                "infrastructure_abort": True,
                "error": (
                    f"forge-fusion aborted on infrastructure ({aborted}); "
                    f"the loop reported no completed attempts and no result. "
                    f"Recipe located by discovery: {located or 'unknown'}"
                )[:1500],
            }
        )
        return result

    artifacts = m.get("artifacts") or {}
    changed = [c.get("path") for c in (artifacts.get("changes") or []) if c.get("path")]
    src_file = str((m.get("fusion") or {}).get("source_file") or "")

    if compile_pass:
        # The win is a flipped default in the framework's own source, so the patch
        # carries it and there is no runtime flag. The speedup is a serving tok/s
        # ratio, not a microbenchmark one, hence the fields naming its origin.
        speedup = compile_pass.get("speedup")
        result.update(
            {
                "compile_pass_flag": compile_pass.get("flag"),
                "serving_speedup": speedup,
            }
        )
    else:
        speedup = (loop.get("best") or {}).get("kernel_speedup")
        best_flags = str(loop.get("best_env_flag") or "").split()
        result.update(
            {
                # Fused arm = all confirmed flags ON; baseline arm = same flags OFF.
                "env_flags": {f: "1" for f in best_flags},
                "baseline_env_flags": {f: "0" for f in best_flags},
                "best_pattern": loop.get("best_pattern"),
            }
        )

    # Integrate applies a fusion from a patch file, its root and a target, and
    # returns without any of them, while ``ok`` satisfies the KERNEL-entry
    # idempotency gate -- so reported as a success such a run is both dropped
    # and never retried. Verified rather than assumed: the invariant that the
    # producer sets them together lives in another repository.
    patch = artifacts.get("patch")
    if kept:
        for name, present in (
            ("a patch", bool(patch)),
            ("the patch file it named", bool(patch) and Path(str(patch)).is_file()),
            ("a target file", bool(src_file)),
            ("a patch root", bool(artifacts.get("repo_root"))),
        ):
            if not present:
                # The failed/REVERT defaults above are the retryable shape.
                result["error_class"] = "fusion_artifact_missing"
                result["error"] = f"forge-fuse kept a fusion but exported no {name}"
                return result

    result.update(
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
            # The root the patch was exported against, which may be a
            # site-packages dir. KernelForge sets it exactly when it sets a
            # patch, so it is present whenever integrate needs it.
            "kernel_repo": str(artifacts.get("repo_root") or ""),
            "verdict": m.get("verdict"),
            # KernelForge validated this on its own -- kernel parity plus a serving
            # smoke for an authored fusion, a serving A/B for a claimed compile pass
            # -- but integrate is what confirms the real e2e gain here.
            "requires_e2e_validation": kept,
        }
    )
    return result


def salvage_forge_fusion_from_workspace(output_dir: str) -> dict[str, Any] | None:
    """Rebuild a KEEP result from pre-smoke checkpoint + patch after a kill.

    ``forge-fuse`` is often SIGKILLed during serving smoke (default 7200s). The
    micro KEEP and ``fusion.patch`` are written before smoke so Hyperloom can
    still hand them to formal e2e integrate.
    """
    root = Path(output_dir or "")
    if not root.is_dir():
        return None
    ckpt_path = root / "kernel_keep_checkpoint.json"
    patch_path = root / "fusion.patch"
    manifest_path = root / "fusion_manifest.json"
    kept = False
    speedup = None
    env_flag = ""
    source_file = ""
    repo_root = ""
    patch = None
    if ckpt_path.is_file():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            ckpt = {}
        if isinstance(ckpt, dict) and ckpt.get("kept"):
            kept = True
            speedup = ckpt.get("kernel_speedup")
            env_flag = str(ckpt.get("env_flag") or "")
            source_file = str(ckpt.get("source_file") or "")
            repo_root = str(ckpt.get("repo_root") or "")
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if isinstance(manifest, dict):
            loop = manifest.get("fusion_loop") or {}
            compile_pass = manifest.get("compile_pass") or {}
            if loop.get("kept") or compile_pass.get("kept"):
                kept = True
            speedup = speedup or (loop.get("best") or {}).get("kernel_speedup") or compile_pass.get("speedup")
            env_flag = env_flag or str(loop.get("best_env_flag") or "")
            source_file = source_file or str((manifest.get("fusion") or {}).get("source_file") or "")
            artifacts = manifest.get("artifacts") or {}
            if artifacts.get("patch"):
                patch = artifacts.get("patch")
            repo_root = repo_root or str(artifacts.get("repo_root") or "")
    if patch_path.is_file():
        patch = str(patch_path)
    if not kept or not patch or not Path(str(patch)).is_file():
        return None
    flags = [f for f in env_flag.split() if f]
    return {
        "status": "ok",
        "engine": "forge_fusion",
        "micro_decision": "candidate",
        "decision": "KEEP",
        "kept": True,
        "kernel_speedup": speedup,
        "env_flags": {f: "1" for f in flags},
        "baseline_env_flags": {f: "0" for f in flags},
        "artifact_files": [],
        "patch": str(patch),
        "source_file": source_file,
        "kernel_repo": repo_root,
        "requires_e2e_validation": True,
        "salvaged": True,
        "workspace": str(root),
    }


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


def _as_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _relay_streams(stdout: Any, stderr: Any) -> None:
    out = _as_text(stdout)
    err = _as_text(stderr)
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)


def _emit(result: dict[str, Any], output_dir: str) -> None:
    """Write result.json (disk fallback) + print the stdout sentinel."""
    if output_dir:
        try:
            (Path(output_dir) / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except OSError:
            pass
    print(f"\n{RESULT_BEGIN}\n{json.dumps(result, sort_keys=True)}\n{RESULT_END}", flush=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperloom wrapper for forge-fusion")
    p.add_argument("--input-json", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(list(argv or sys.argv[1:]))
        payload = _load_input_json(args.input_json)
        cmd = _build_cmd(payload)
    except Exception as exc:  # noqa: BLE001 - structured wrapper failure
        print(
            json.dumps(
                {
                    "status": "failed",
                    "engine": "forge_fusion",
                    "micro_decision": "failed",
                    "decision": "REVERT",
                    "kept": False,
                    "error_class": exc.__class__.__name__,
                    "error": repr(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2

    _inject_author_gateway_env(str(payload.get("agent_backend") or ""))
    output_dir = str(payload.get("output_dir") or "")
    # The output dir is keyed on the task, so a run that dies before writing new
    # artifacts must not salvage the previous run's KEEP as its own.
    output_root = Path(output_dir or ".")
    for stale_name in (
        "fusion_manifest.json",
        "kernel_keep_checkpoint.json",
        "fusion.patch",
    ):
        (output_root / stale_name).unlink(missing_ok=True)
    timeout_sec = _timeout_sec(payload)
    try:
        proc = _run_with_tree_timeout(cmd, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _relay_streams(getattr(exc, "stdout", None), getattr(exc, "stderr", None))
        result = _timeout_result(output_dir, timeout_sec, exc)
        _emit(result, output_dir)
        return 124

    _relay_streams(proc.stdout, proc.stderr)
    result = _normalize_manifest(output_dir, proc.returncode)
    _emit(result, output_dir)
    # Mirror the subprocess exit: non-zero only when the subprocess itself failed.
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
