#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run forge-fusion as a Hyperloom kernel-agent tool.

The orchestrator writes an input JSON and calls this script; the autonomous
fusion loop itself lives in the standalone ``forge_fusion`` package.

forge-fusion emits a ``fusion_manifest.json``; this wrapper normalizes that into
the Hyperloom kernel-result contract (a ``FORGE_FUSION_RESULT_BEGIN/END`` stdout
sentinel + an on-disk ``result.json``) that ``run_fusion_handler`` parses. A KEPT
fusion already carries kernel-parity AND serving-smoke validation, so
``requires_e2e_validation`` is set for the orchestrator's integrate/re-baseline
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

# forge-fusion's manifest verdict for "discovery never reached the model", added in
# manifest schema v2 alongside the ``error`` block. It is NOT a statement about the
# kernel, so it must not be normalized into an optimization outcome -- see
# _normalize_manifest.
LLM_UNAVAILABLE_VERDICT = "llm_unavailable"
DEFAULT_TIMEOUT_SEC = 7200


def _inject_author_gateway_env() -> None:
    """Seed the ``claude`` author subprocess's gateway auth from the OpenAI-proxy env.

    forge-fusion's ``author`` stage drives the ``claude`` CLI, which authenticates
    via ``ANTHROPIC_*``. Hyperloom's session env only carries ``OPENAI_BASE_URL`` /
    ``SAFE_API_KEY`` for the OpenAI-compatible LLM proxy, so derive the
    ``ANTHROPIC_*`` equivalents here. Only fills what is absent; explicit operator
    values always win.
    """
    openai_base = str(os.environ.get("OPENAI_BASE_URL") or "").strip()
    if openai_base and not os.environ.get("ANTHROPIC_BASE_URL"):
        # Strip trailing /v1 (claude appends its own).
        os.environ["ANTHROPIC_BASE_URL"] = openai_base[:-3] if openai_base.endswith("/v1") else openai_base
    token = str(os.environ.get("SAFE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if token:
        os.environ.setdefault("ANTHROPIC_API_KEY", token)
        os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", token)
    # claude's bypassPermissions refuses to start under root unless IS_SANDBOX=1.
    # Only set it when actually running as root so we do not defeat the guard
    # for non-root sessions that never needed the escape hatch.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    apply_llm_stability_env(os.environ)


def _package_root(source_file: str) -> str:
    """Install dir above the top package containing ``source_file`` (site-packages).

    Mirrors forge-fusion's export root for a non-git (pip-installed) framework so the
    patch's package-relative paths apply here. Assumes each package level ships an
    ``__init__.py`` (true for vLLM/sglang).
    """
    if not source_file:
        return ""
    d = Path(source_file).resolve().parent
    while d.parent != d and (d / "__init__.py").is_file():
        d = d.parent
    return str(d)


def _git_toplevel(path: str) -> str:
    """Best-effort repo root the patch paths are relative to (for integrate's apply).

    Uses the git work-tree root ONLY when ``path`` is a git-TRACKED file there. A
    pip-installed framework often lives under a git project's ``.venv`` yet is
    untracked; returning the project root would make the package-relative patch fail
    to apply. In that case (and for a plain pip install) use the package root, which
    matches the root forge-fusion exported the patch against.
    """
    if not path:
        return ""
    try:
        parent = str(Path(path).parent)
        r = subprocess.run(
            ["git", "-C", parent, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            toplevel = r.stdout.strip()
            try:
                rel = str(Path(path).resolve().relative_to(Path(toplevel).resolve()))
                tracked = subprocess.run(
                    ["git", "-C", toplevel, "ls-files", "--error-unmatch", "--", rel],
                    capture_output=True, text=True, timeout=10,
                )
                if tracked.returncode == 0:
                    return toplevel
            except ValueError:
                pass
            return _package_root(path) or toplevel
    except (OSError, subprocess.SubprocessError):
        return _package_root(path)
    return _package_root(path)


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
    cmd = [sys.executable, "-m", "forge_fusion.cli", "run"]
    _add_opt(cmd, args, "trace_path", "--trace", required=True)
    _add_opt(cmd, args, "model_path", "--model-path", required=True)
    _add_opt(cmd, args, "framework", "--framework", required=True)
    _add_opt(cmd, args, "output_dir", "--output-dir", required=True)
    _add_opt(cmd, args, "discover_mode", "--discover")
    _add_opt(cmd, args, "llm_model", "--llm-model")
    _add_opt(cmd, args, "max_turns", "--max-turns")
    _add_opt(cmd, args, "gpu", "--gpu")
    _add_opt(cmd, args, "decode_batch", "--decode-batch")
    _add_opt(cmd, args, "ab_isl", "--ab-isl")
    _add_opt(cmd, args, "ab_osl", "--ab-osl")
    _add_opt(cmd, args, "framework_root", "--framework-root")
    # Author all source-confirmed patterns together by default; set
    # fuse_all_confirmed=false to author only the top recipe.
    if bool(args.get("fuse_all_confirmed", True)):
        cmd.append("--fuse-all-confirmed")
    if not bool(args.get("author", True)):
        cmd.append("--no-author")
    if not bool(args.get("validate", True)):
        cmd.append("--no-validate")
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
    val = m.get("validation") or {}
    kept = bool(loop.get("kept") or val.get("kept"))

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
        result.update({
            "verdict": LLM_UNAVAILABLE_VERDICT,
            "error_class": LLM_UNAVAILABLE_VERDICT,
            "error": f"forge-fusion never reached the LLM ({kind}{tried}): {message}"[:1500],
        })
        return result

    best = loop.get("best") or {}
    speedup = best.get("kernel_speedup") or val.get("kernel_speedup")
    best_flags = str(loop.get("best_env_flag") or "").split()
    artifacts = m.get("artifacts") or {}
    changed = [c.get("path") for c in (artifacts.get("changes") or []) if c.get("path")]
    src_file = str((m.get("fusion") or {}).get("source_file") or "")

    result.update(
        {
            "status": "ok" if kept else "complete",
            "micro_decision": "candidate" if kept else "no_improvement",
            "decision": "KEEP" if kept else "REVERT",
            "kept": kept,
            "kernel_speedup": speedup,
            # Fused arm = all confirmed flags ON; baseline arm = same flags OFF.
            "env_flags": {f: "1" for f in best_flags},
            "baseline_env_flags": {f: "0" for f in best_flags},
            "artifact_files": changed,
            "patch": artifacts.get("patch"),
            # For integrate's patch-apply path.
            "source_file": src_file,
            # Prefer the root forge-fusion exported the patch against (authoritative,
            # may be a site-packages dir); fall back to deriving it from src_file.
            "kernel_repo": str(artifacts.get("repo_root") or "")
            or (_git_toplevel(src_file) if src_file else ""),
            "best_pattern": loop.get("best_pattern"),
            "verdict": m.get("verdict"),
            # A KEPT fusion passed kernel parity + serving smoke; the orchestrator
            # still confirms the real e2e gain via integrate.
            "requires_e2e_validation": kept,
        }
    )
    return result


def _timeout_result(output_dir: str, timeout_sec: int, exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    """Shape a timed-out forge-fusion run as a normal REVERT result."""
    cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or []))
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
        "error": f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}",
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

    _inject_author_gateway_env()
    output_dir = str(payload.get("output_dir") or "")
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
