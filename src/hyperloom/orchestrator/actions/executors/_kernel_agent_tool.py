# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-agent tool invocation primitives, and kernel-patch apply/revert/finalize through them."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from hyperloom.common.git_safety import safe_directory_args
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows

log = logging.getLogger(__name__)

# Kernel-agent shell tools root; read lazily so late env injection wins.
_KERNEL_AGENT_ROOT_ENV = "HYPERLOOM_KERNEL_AGENT_ROOT"


def _kernel_agent_root_from_env() -> Path | None:
    """Read the kernel-agent install root from the environment at call time.

    Resolved lazily on every call so a late ``os.environ`` injection by the CLI
    preflight still wins.

    Returns:
        Path | None: The kernel-agent root as a :class:`~pathlib.Path`, or
            ``None`` when ``HYPERLOOM_KERNEL_AGENT_ROOT`` is unset or empty.
    """
    raw = os.environ.get(_KERNEL_AGENT_ROOT_ENV)
    if not raw:
        return None
    return Path(raw)


HandlerResult = dict[str, Any]


_APPLY_TOOL_MODULE: Any | None = None


def _kernel_agent_root_error() -> str | None:
    """Validate that the kernel-agent install root is configured and present.

    Returns:
        str | None: A human-readable error message when the root env var is
            unset or points at a missing directory, or ``None`` when the root
            exists and is usable.
    """
    root = _kernel_agent_root_from_env()
    if root is None:
        return (
            f"{_KERNEL_AGENT_ROOT_ENV} is not set; run "
            "src/hyperloom/inference_optimizer/assets/install.sh and source $KERNEL_AGENT_ENV "
            "(default: $USER_DATA_PATH/runtime/kernel-agent.env.sh)"
        )
    if not root.is_dir():
        return f"{_KERNEL_AGENT_ROOT_ENV} does not exist: {root}"
    return None


def _kernel_agent_tool_path(tool_name: str) -> Path:
    """Resolve the absolute path to a kernel-agent shell tool.

    Args:
        tool_name (str): File name of the tool under ``<root>/tools/`` (for
            example ``tracelens_analysis.py``).

    Returns:
        Path: The resolved path to the requested tool.

    Raises:
        RuntimeError: If the kernel-agent root is unset/missing, or the named
            tool does not exist under ``<root>/tools/``.
    """
    err = _kernel_agent_root_error()
    if err:
        raise RuntimeError(err)
    root = _kernel_agent_root_from_env()
    assert root is not None
    path = root / "tools" / tool_name
    if not path.is_file():
        raise RuntimeError(f"kernel-agent tool not found: {path}")
    return path


def _load_apply_tool() -> Any:
    """Lazily import and cache the kernel-agent ``apply_kernel_patch.py`` module.

    Loaded by file path via :mod:`importlib.util` and memoized in the module
    global ``_APPLY_TOOL_MODULE`` so subsequent calls reuse the same module.

    Returns:
        Any: The imported ``apply_kernel_patch`` module object.

    Raises:
        RuntimeError: If the kernel-agent root/tool path cannot be resolved.
        ImportError: If the module cannot be loaded from its resolved path.
    """
    global _APPLY_TOOL_MODULE
    if _APPLY_TOOL_MODULE is not None:
        return _APPLY_TOOL_MODULE
    path = _kernel_agent_tool_path("apply_kernel_patch.py")
    spec = importlib.util.spec_from_file_location("hyperloom_apply_kernel_patch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load apply_kernel_patch.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _APPLY_TOOL_MODULE = module
    return module


def _artifact_paths_from_payload(payload: dict) -> list[str]:
    """Normalize compiled-artifact paths from a payload into a list of strings.

    Accepts either ``artifact_paths`` or ``compiled_artifact_paths``; a single
    string is wrapped into a one-element list and falsy entries are dropped.

    Args:
        payload (dict): Request payload that may carry artifact path(s).

    Returns:
        list[str]: The collected artifact paths (possibly empty).
    """
    raw = payload.get("artifact_paths") or payload.get("compiled_artifact_paths") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _final_content_snapshot(
    *,
    patch_path: str,
    snapshot_dir: str | None,
    repo_root: str | None,
) -> str | None:
    """Return a snapshot dir holding the patch's FINAL bytes, materializing if needed.

    ``snapshot_dir`` means two different things on the two sides of the
    nomination wire. The fusion exporter records its *pre-authoring pristine*
    snapshot -- the baseline it diffed AGAINST -- on ``RecipePatch.snapshot_dir``,
    and that value rides the envelope into the pending record. ``apply_kernel_patch``
    reads the same field as the *post-patch final* contents it copies FROM. A
    pristine dir can never satisfy that: it is missing, by construction, every
    module the fusion authored, so the apply pre-flight refuses the whole patch
    with "snapshot missing content for <...>_fused_<recipe>.py" and a real KEEP
    is lost.

    Rather than trust the field, check it: a usable snapshot has the final bytes
    for every path the patch writes. When it does not, materialize one from the
    patch itself. Materialization failure returns the original value so apply
    reports the real error instead of this helper's.
    """
    if not (patch_path.endswith(".patch") and repo_root):
        return snapshot_dir
    try:
        descriptors = _load_apply_tool().parse_patch_manifest(
            Path(patch_path).read_text(encoding="utf-8", errors="replace")
        )
        writes = [str(d.get("path") or "") for d in descriptors if d.get("op") == "write"]
    except Exception:  # noqa: BLE001 — an unreadable patch is apply's error to report.
        return snapshot_dir
    if not writes:
        return snapshot_dir
    if snapshot_dir and all((Path(snapshot_dir) / rel).exists() for rel in writes):
        return snapshot_dir
    try:
        return materialize_unified_patch_snapshot(
            patch_path=patch_path,
            repo_root=repo_root,
            snapshot_dir=Path(patch_path).parent / "integrate_snapshot",
        )
    except Exception:
        log.exception("integrate: could not materialize a final-content snapshot for %s", patch_path)
        return snapshot_dir


def _maybe_apply_kernel_patch(
    payload: dict,
    *,
    session_dir: Path,
    kernel_id: str | None,
) -> HandlerResult:
    """Apply a kernel patch via the kernel-agent ``apply_kernel_patch`` tool.

    Resolves a backup root under the session's patches dir when none is given,
    then delegates to the tool with rebuild / dry-run / target options pulled
    from the payload.

    Args:
        payload (dict): Request payload carrying ``patch_path`` plus
            ``target_file`` / ``source_file`` and optional apply/rebuild flags.
        session_dir (Path): Session directory used to derive the backup root.
        kernel_id (str | None): Kernel identifier for backup namespacing;
            falls back to ``payload['kernel_id']`` or ``"anon"``.

    Returns:
        HandlerResult: A ``status="skipped"`` result when required inputs are
            missing, otherwise the tool's apply result dict.
    """
    patch_path = str(payload.get("patch_path") or "").strip()
    target_file = str(payload.get("target_file") or payload.get("source_file") or "").strip()
    if not patch_path or not target_file:
        return {
            "status": "skipped",
            "reason": "missing patch_path or target_file/source_file",
        }
    from hyperloom.inference_optimizer.session.session_paths import fs_safe_id, patches_dir

    kid = str(kernel_id or payload.get("kernel_id") or "")
    # Same fold as the integrate workspace: a fusion sibling keys this dir by its
    # ``llm:<recipe>`` operator name, which ``mkdir`` rejects on some filesystems.
    backup_root = payload.get("backup_root") or (patches_dir(session_dir, fs_safe_id(kid)) / "backup")
    tool = _load_apply_tool()
    # Snapshot mode: a snapshot dir of byte-exact final files lands atomically.
    snapshot_dir = str(payload.get("snapshot_dir") or "").strip() or None
    repo_root = str(payload.get("kernel_repo") or payload.get("repo") or "").strip() or None
    snapshot_dir = _final_content_snapshot(
        patch_path=patch_path,
        snapshot_dir=snapshot_dir,
        repo_root=repo_root,
    )
    return tool.apply_kernel_patch(
        patch_path=patch_path,
        target_file=target_file,
        backup_root=backup_root,
        kernel_id=kid,
        artifact_paths=_artifact_paths_from_payload(payload),
        rebuild_command=payload.get("rebuild_command"),
        rebuild_timeout_sec=int(payload.get("rebuild_timeout_sec", 1800)),
        skip_rebuild=bool(payload.get("skip_rebuild", False)),
        dry_run=bool(payload.get("dry_run_patch", False)),
        snapshot_dir=snapshot_dir,
        repo_root=repo_root,
        producer_manifest=(str(payload.get("producer_manifest") or "").strip() or None),
    )


def materialize_unified_patch_snapshot(
    *,
    patch_path: str | Path,
    repo_root: str | Path,
    snapshot_dir: str | Path | None = None,
) -> str:
    """Materialize final file contents for apply_kernel_patch snapshot mode.

    Applies a ``forge-fusion`` unified diff to a minimal throwaway mirror of the
    touched files and returns that mirror path (snapshot mode treats the diff as
    a manifest with final bytes under ``snapshot_dir``).
    """
    patch = Path(patch_path).resolve()
    root = Path(repo_root).resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"patch_path does not exist: {patch}")
    if not root.is_dir():
        raise FileNotFoundError(f"kernel repo does not exist: {root}")

    tool = _load_apply_tool()
    patch_text = patch.read_text(encoding="utf-8", errors="replace")
    descriptors = tool.parse_patch_manifest(patch_text)
    if not descriptors:
        raise ValueError(f"patch has no file operations: {patch}")

    # Paths the patch CREATES: these must be produced by ``git apply``, never
    # pre-seeded with a base, or apply fails "already exists". Everything else
    # is a modify whose base we must supply. ``is_new`` comes from
    # ``parse_patch_manifest`` (single source of truth for both the path
    # normalization and the create/modify disposition), which avoids a second,
    # drift-prone parse of the raw patch text.
    _new_file_paths = {
        str(desc.get("path") or "") for desc in descriptors if desc.get("op") == "write" and desc.get("is_new")
    }

    snap = Path(snapshot_dir) if snapshot_dir is not None else patch.parent / "fusion_snapshot"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)

    for desc in descriptors:
        rel = Path(str(desc.get("path") or ""))
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe patch path: {rel}")
        dst = snap / rel
        base = subprocess.run(
            ["git", *safe_directory_args(["-C", str(root), "show", f"HEAD:{rel.as_posix()}"])],
            capture_output=True,
            timeout=60,
        )
        if base.returncode == 0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(base.stdout)
        elif rel.as_posix() not in _new_file_paths:
            # ``git show HEAD:`` failed and this is a MODIFY (not a create):
            # non-git repo_root (e.g. vLLM/sglang under site-packages/
            # dist-packages) or an untracked-but-present file. Fall back to the
            # on-disk source. forge-fusion (PR #75) emits the patch for these
            # non-git frameworks; without this fallback the snapshot lacks the
            # base file and ``git apply`` fails "<path>: No such file or
            # directory". New files are intentionally left for ``git apply`` to
            # create.
            src = root / rel
            if not src.is_file():
                # Neither git HEAD nor the on-disk layout has the base. Surface
                # a precise error here instead of the opaque ``git apply`` "No
                # such file or directory" that would otherwise follow.
                raise FileNotFoundError(
                    f"patch base missing for {rel.as_posix()}: not in git HEAD and not on disk under {root}"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    # ``git apply <path>`` rejects an otherwise valid final hunk when the patch
    # artifact lacks a trailing newline (observed in legacy KB records). Feed a
    # normalized in-memory copy so materialization is tolerant without mutating
    # the content-addressed downloaded artifact.
    normalized_patch_text = patch_text if patch_text.endswith(("\n", "\r")) else f"{patch_text}\n"
    # Pin the work tree to ``snap``. Without this, ``git apply`` resolves paths
    # against whatever repository encloses ``snap`` -- and when the session dir
    # lives INSIDE a checkout (a session under the Hyperloom repo itself), every
    # hunk is reported "Skipped patch ..." while git still exits 0. The snapshot
    # then comes back empty and the failure surfaces later as the far more
    # confusing "snapshot missing final content".
    apply_env = {
        **os.environ,
        "GIT_DIR": str(snap / ".git_materialize"),
        "GIT_WORK_TREE": str(snap),
        "GIT_CEILING_DIRECTORIES": str(snap.parent),
    }
    proc = subprocess.run(
        ["git", "apply", "--unsafe-paths", "-"],
        cwd=snap,
        input=normalized_patch_text,
        capture_output=True,
        text=True,
        timeout=60,
        env=apply_env,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"could not materialize patch snapshot: {msg[:500]}")

    for desc in descriptors:
        if desc.get("op") == "write" and not (snap / str(desc["path"])).is_file():
            raise RuntimeError(f"snapshot missing final content for {desc['path']}")
    return str(snap)


def _maybe_revert_kernel_patch(apply_result: HandlerResult) -> HandlerResult:
    """Revert a kernel patch using its apply manifest.

    A manifest is enough; the apply's ``status`` is not required, so a partial
    apply reverts the files it managed to touch. Gating on ``status == "ok"``
    used to leave exactly those applied.

    Args:
        apply_result: Apply metadata carrying ``manifest_path``.

    Returns:
        The revert result, or an explicit failure result.
    """
    if not apply_result.get("manifest_path"):
        return {"status": "skipped", "reason": "no applied patch manifest"}
    try:
        return _load_apply_tool().revert_kernel_patch(apply_result["manifest_path"])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_class": "patch_revert_exception",
            "error": repr(exc),
            "manifest_path": str(apply_result["manifest_path"]),
        }


def _maybe_finalize_kernel_patch(
    apply_result: HandlerResult,
) -> HandlerResult:
    """Delete patch backups after a KEEP becomes durable."""
    if apply_result.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": "patch apply did not complete",
        }
    if not apply_result.get("manifest_path"):
        return {"status": "skipped", "reason": "no applied patch manifest"}
    try:
        return _load_apply_tool().finalize_kernel_patch(apply_result["manifest_path"])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_class": "patch_finalize_exception",
            "error": repr(exc),
            "manifest_path": str(apply_result["manifest_path"]),
        }


def _tool_label(cmd: list[str]) -> str:
    """Name the tool a command runs, for the progress note.

    Args:
        cmd (list[str]): The command and arguments.

    Returns:
        str: The first ``.py`` argument's stem, else the executable's name.
    """
    for arg in cmd:
        text = str(arg)
        if text.endswith(".py"):
            return Path(text).stem
    return Path(str(cmd[0])).name if cmd else "subprocess"


async def _run_subprocess(
    cmd: list[str],
    *,
    timeout_sec: int,
) -> tuple[int, str, str]:
    """Run a bounded subprocess without blocking the reactor.

    Args:
        cmd: The command and arguments to run.
        timeout_sec: Per-run timeout in seconds.

    Returns:
        A tuple of ``(returncode, stdout, stderr)``.
    """
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(float(timeout_sec))
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be finite and positive")

    def _run(on_output: Callable[[], None]) -> tuple[int, str, str]:
        """Run the command synchronously in a worker thread.

        Copies the environment, injects the Ray GCS address in multi-node mode,
        and prepends the venv ``bin`` to ``PATH``. Launches the child in its own
        POSIX session and, on timeout, reaps the whole process group so a hung
        grandchild dies with the wrapper. Mirrors ``subprocess.run``: captures
        stdout/stderr and re-raises ``TimeoutExpired``.

        Args:
            on_output: Liveness callback invoked per line the child emits.

        Returns:
            tuple[int, str, str]: ``(returncode, stdout, stderr)``.

        Raises:
            subprocess.TimeoutExpired: When the command exceeds ``timeout_sec``.
        """
        env = os.environ.copy()
        from ._multi_node_env import (
            is_multi_node,
            ray_gcs_address_from_state,
            infera_ssh_env_from_state,
        )
        from ._subprocess_kill import run_with_session_kill

        if is_multi_node():
            # Infera backend: route GEAK GPU work to a pod over SSH (no Ray).
            # infera_ssh_env_from_state() returns {} for RayJob/single-node, so
            # the RAY_ADDRESS path below is unchanged for those.
            ssh_env = infera_ssh_env_from_state()
            if ssh_env:
                env.update(ssh_env)
            addr = "" if ssh_env else ray_gcs_address_from_state()
            if addr:
                env.setdefault("RAY_ADDRESS", addr)
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        # The heartbeat around this call is only as honest as the child's
        # flushing: block-buffered on a pipe, it looks dead between flushes.
        # ``setdefault`` so an operator who set this deliberately still wins.
        env.setdefault("PYTHONUNBUFFERED", "1")
        # ``run_with_session_kill`` reaps the whole descendant tree on every exit path.
        cp = run_with_session_kill(
            cmd,
            env=env,
            timeout=timeout_sec,
            text=True,
            on_output=on_output,
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""

    async with heartbeat_while_output_flows(unit="kernel_tool", label=_tool_label(cmd)) as activity:
        return await asyncio.to_thread(_run, activity.note)


def _shape_tool_result(rc: int, stdout: str, stderr: str) -> HandlerResult:
    """Wrap a kernel-agent tool's exit + stdout into our schema (prefer the tool's own JSON, synthesize only on parse failure).

    Args:
        rc: The tool's process return code.
        stdout: The tool's captured standard output.
        stderr: The tool's captured standard error.

    Returns:
        The tool's own JSON result (status filled from ``rc`` if absent), or a
        synthesized failure result when stdout has no parseable JSON.
    """
    parsed = _parse_tool_stdout(stdout)
    if parsed and set(parsed) == {"raw_stdout_tail"}:
        # Unparseable output is not a result. Inferring ``ok`` from rc==0 here
        # made a tool whose output we could not read indistinguishable from one
        # that succeeded: the roofline executor read status=ok, recorded an
        # empty analysis over the real one, and the leg reported success while
        # twenty minutes of GPU evidence went in the bin.
        return {
            "status": "failed",
            "error_class": "tool_output_unparseable",
            "error": ("tool exited rc=%d but its stdout held no JSON object" % rc),
            "returncode": rc,
            "raw_stdout_tail": parsed["raw_stdout_tail"],
            "stderr_tail": stderr[-2000:] if stderr.strip() else "",
        }
    if parsed:
        # Trust the tool's own status; else infer from rc.
        if "status" not in parsed:
            parsed["status"] = "ok" if rc == 0 else "failed"
        if rc != 0:
            parsed.setdefault("returncode", rc)
            if stderr.strip():
                parsed.setdefault("stderr_tail", stderr[-2000:])
        return parsed
    return {
        "status": "failed" if rc != 0 else "ok",
        "returncode": rc,
        "error": (stderr or stdout)[-2000:],
    }


def _parse_tool_stdout(stdout: str) -> dict[str, Any]:
    """Parse a tool's stdout into a dict, surviving non-JSON noise.

    Tries the whole stdout as a JSON object first; if that fails, scans
    backwards for the last line that is a standalone JSON object. As a last
    resort returns the stdout tail under ``raw_stdout_tail``.

    Args:
        stdout (str): Captured standard output from a kernel-agent tool.

    Returns:
        dict[str, Any]: The parsed JSON object, an empty dict for empty input,
            or ``{"raw_stdout_tail": ...}`` when no JSON object is found.
    """
    text = stdout.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    # Fallback: scan for the last JSON object on its own line.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    # Last: a pretty-printed object opening at the start of a line. A tool that
    # indents its result spans many lines, so neither whole-text nor per-line
    # parsing sees it, and it is exactly the tools with a lot to say that
    # indent. tracelens_analysis returned a megabyte of hot-kernel analysis this
    # way, interleaved with progress chatter and followed by an import banner;
    # every field of it was dropped and the run still reported ``ok``.
    # ``raw_decode`` stops at the end of the object, so trailing noise is fine.
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"^\{", text, re.MULTILINE)]
    for start in reversed(starts):
        try:
            obj, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {"raw_stdout_tail": text[-2000:]}
