# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side script entrypoints: compose the shell entrypoints that ship multi_node/scripts to pods, parse their JSON."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common import aiter_jit_cache

from ._internal.server_args_safety import ServerArgsRejected, prepare_shell_safe_extra_args


_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def read_pod_script(name: str) -> str:
    """Read a pod-side script from ``multi_node/scripts/`` (embedded into the dashboard entrypoint at submit time)."""
    p = _SCRIPTS_DIR / name
    if not p.is_file():
        raise RuntimeError(f"missing pod-side script: {p}. Did you trim the multi_node/scripts/ directory?")
    return p.read_text(encoding="utf-8")


KERNEL_NODE_OPS_DEPS = (Path(aiter_jit_cache.__file__).resolve(), _SCRIPTS_DIR / "patch_path_safety.py")
SHAPE_GATE_DEPS = (_SCRIPTS_DIR / "sglang_shape_gate.py",)
LAUNCHER_DEPS = (Path(__file__).parent / "_internal" / "server_args_safety.py", *SHAPE_GATE_DEPS)


def read_bundled_pod_python_script(main: str, deps: tuple[Path, ...]) -> str:
    """Bundle a pod script with its dependencies registered under their own module names.

    Pods receive one file, so each dependency is registered in ``sys.modules``
    before the script runs and keeps importing it by name, on the driver and on
    any Ray worker that deserializes a function closing over it.
    """
    chunks = ["import sys, types\n"]
    for dep in deps:
        source = dep.read_text(encoding="utf-8")
        chunks.append(
            f"_dependency = types.ModuleType({dep.stem!r})\n"
            f"_dependency.__file__ = {dep.name!r}\n"
            f"sys.modules[{dep.stem!r}] = _dependency\n"
            f"exec(compile({source!r}, _dependency.__file__, 'exec'), _dependency.__dict__)\n"
        )
    chunks.append(f"exec(compile({read_pod_script(main)!r}, {main!r}, 'exec'), globals())\n")
    return "".join(chunks)


def _build_restart_entrypoint(
    args: argparse.Namespace,
    pid_file: str,
    log_file: str,
) -> str:
    """Compose the single-node restart entrypoint (heredoc kill_server.sh + launch_server.sh; IR-5 PID-file kill)."""
    framework = args.framework.lower()
    if framework not in ("sglang", "vllm"):
        raise RuntimeError(f"unsupported framework: {args.framework!r} (use sglang or vllm)")

    # extra_args is spliced after ``--`` into a shell entrypoint and word-split by launch_server.sh into argv.
    try:
        safe_extra_args = prepare_shell_safe_extra_args(
            args.extra_args or "",
            context="restart-server --extra-args",
        )
    except ServerArgsRejected as exc:
        raise RuntimeError(str(exc)) from exc

    kill_sh = read_pod_script("kill_server.sh")
    launch_sh = read_pod_script("launch_server.sh")

    wait_flag = "--no-wait-health" if args.no_wait_health else "--wait-health"

    return (
        "set -euo pipefail; "
        f'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
        f"cat > \"$WORK_DIR/kill_server.sh\" <<'__MN_KILL_EOF__'\n"
        f"{kill_sh}__MN_KILL_EOF__\n"
        f"cat > \"$WORK_DIR/launch_server.sh\" <<'__MN_LAUNCH_EOF__'\n"
        f"{launch_sh}__MN_LAUNCH_EOF__\n"
        'chmod +x "$WORK_DIR/kill_server.sh" "$WORK_DIR/launch_server.sh"; '
        f'"$WORK_DIR/kill_server.sh" {shlex.quote(str(pid_file))}; '
        # shlex.quote every interpolated value so a path/arg carrying shell metacharacters is a single argv token,
        # never shell control syntax.
        f'"$WORK_DIR/launch_server.sh" {shlex.quote(str(framework))} '
        f"{shlex.quote(str(args.model))} {shlex.quote(str(args.tp))} "
        f"{shlex.quote(str(pid_file))} {shlex.quote(str(log_file))} "
        f"{wait_flag} -- {safe_extra_args}"
    )


# Common entrypoint preamble: sources the bootstrap env file so PATH points at /opt/venv/bin (no-op when bootstrap was
# skipped).
_MN_ENTRYPOINT_PREAMBLE = (
    "set -euo pipefail; "
    "if [ -f /etc/profile.d/hyperloom-env.sh ]; then "
    "source /etc/profile.d/hyperloom-env.sh; "
    "fi; "
    'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
)


def _build_kill_single_entrypoint(pid_file: str) -> str:
    """Compose a head-pod entrypoint that runs only kill_server.sh."""
    kill_sh = read_pod_script("kill_server.sh")
    return (
        "set -euo pipefail; "
        f'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
        f"cat > \"$WORK_DIR/kill_server.sh\" <<'__MN_KILL_EOF__'\n"
        f"{kill_sh}__MN_KILL_EOF__\n"
        'chmod +x "$WORK_DIR/kill_server.sh"; '
        f'"$WORK_DIR/kill_server.sh" {shlex.quote(str(pid_file))}'
    )


def _build_multinode_kill_entrypoint(pid_dir: str) -> str:
    """Compose the head-pod entrypoint that kills every rank's server via heredoc-embedded kill_multinode.py (fans out via ray actors)."""
    py = read_pod_script("kill_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/kill_multinode.py\" <<'__MN_KILL_PY_EOF__'\n"
        f"{py}__MN_KILL_PY_EOF__\n"
        f'python3 "$WORK_DIR/kill_multinode.py" '
        f"--pid-dir {shlex.quote(str(pid_dir))}"
    )


def _json_object_candidates(text: str) -> Iterator[str]:
    """Yield balanced ``{...}`` blocks from the end of ``text`` backward, ignoring braces inside JSON strings."""
    depth = 0
    in_str = False
    end = -1
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if depth and ch == '"':
            j = i
            while j and text[j - 1] == "\\":
                j -= 1
            if (i - j) % 2 == 0:
                in_str = not in_str
        elif in_str:
            continue
        elif ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{" and depth:
            depth -= 1
            if depth == 0:
                yield text[i : end + 1]


def _extract_launcher_summary(launch_logs: str) -> dict:
    """Parse the JSON summary launch_multinode.py writes to stdout: the last ``{...}`` block that parses; ``{}`` on failure."""
    for candidate in _json_object_candidates(launch_logs or ""):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def _build_multinode_launch_entrypoint(
    args: argparse.Namespace,
    nnodes: int,
    pid_dir: str,
    log_dir: str,
) -> str:
    """Compose the head-pod entrypoint that spawns one rank per node via heredoc-embedded launch_multinode.py."""
    py = read_bundled_pod_python_script("launch_multinode.py", LAUNCHER_DEPS)
    wait_flag = "--no-wait-health" if args.no_wait_health else ""
    try:
        extra_args = prepare_shell_safe_extra_args(
            args.extra_args,
            context="rayjob restart-server --extra-args",
        )
    except ServerArgsRejected as exc:
        raise RuntimeError(str(exc)) from exc
    # Pin SGLANG_TORCH_PROFILER_DIR to a shared-FS path from env; empty => skip the flag.
    profiler_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    profiler_arg = f"--torch-profiler-dir {shlex.quote(str(profiler_dir))} " if profiler_dir else ""
    # Expert-parallel size; ep <= 1 emits no flag.
    ep_arg = f"--ep {args.ep} " if args.ep > 1 else ""
    # Forward PD knobs only when disaggregated (else keep the aggregated default).
    pd_args = ""
    if args.pd_mode == "disaggregated":
        pn = args.pd_prefill_nodes
        dn = args.pd_decode_nodes
        ptp = args.pd_prefill_tp
        dtp = args.pd_decode_tp
        tb = args.pd_transfer_backend.strip()
        ib = args.pd_ib_device.strip()
        bp = args.pd_bootstrap_port
        chunks = [
            "--pd-mode disaggregated",
            f"--pd-prefill-nodes {pn}",
            f"--pd-decode-nodes {dn}",
        ]
        if ptp > 0:
            chunks.append(f"--pd-prefill-tp {ptp}")
        if dtp > 0:
            chunks.append(f"--pd-decode-tp {dtp}")
        if tb:
            chunks.append(f"--pd-transfer-backend {shlex.quote(str(tb))}")
        if ib:
            chunks.append(f"--pd-ib-device {shlex.quote(str(ib))}")
        if bp > 0:
            chunks.append(f"--pd-bootstrap-port {bp}")
        pd_args = " ".join(chunks) + " "
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/launch_multinode.py\" <<'__MN_LAUNCH_PY_EOF__'\n"
        f"{py}__MN_LAUNCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/launch_multinode.py" '
        # shlex.quote framework/model so a value with shell metacharacters stays a single argv token (tp/nnodes are
        # int-coerced above).
        f"--framework {shlex.quote(str(args.framework))} "
        f"--model {shlex.quote(str(args.model))} "
        f"--tp {args.tp!s} --nnodes {nnodes!s} "
        f"--pid-dir {shlex.quote(str(pid_dir))} --log-dir {shlex.quote(str(log_dir))} "
        f"{ep_arg}{profiler_arg}{pd_args}{wait_flag} --extra-args {shlex.quote(str(extra_args))}"
    )


def _build_multinode_router_entrypoint(
    args: argparse.Namespace,
    prefill_url: str,
    decode_url: str,
    pid_dir: str,
    log_dir: str,
) -> str:
    """Compose the head-pod entrypoint that detaches the PD router (rank 0 only, binds 8888) via heredoc-embedded launch_router.py."""
    py = read_pod_script("launch_router.py")
    pid_file = f"{pid_dir.rstrip('/')}/router.pid"
    log_file = f"{log_dir.rstrip('/')}/router.log"
    vllm_router_cmd = (getattr(args, "pd_vllm_router_cmd", "") or "").strip()
    vrc_arg = f"--vllm-router-cmd {shlex.quote(str(vllm_router_cmd))} " if vllm_router_cmd else ""
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/launch_router.py\" <<'__MN_ROUTER_PY_EOF__'\n"
        f"{py}__MN_ROUTER_PY_EOF__\n"
        f'python3 "$WORK_DIR/launch_router.py" '
        f"--framework {args.framework!s} "
        f"--prefill-url {shlex.quote(str(prefill_url))} --decode-url {shlex.quote(str(decode_url))} "
        f"--pid-file {shlex.quote(str(pid_file))} --log-file {shlex.quote(str(log_file))} "
        f"{vrc_arg}"
    )


def _kernel_patch_pod_files() -> str:
    """Ship the kernel patch driver and its fixed sibling imports to Ray pods."""
    return "".join(
        f"cat > \"$WORK_DIR/{path.name}\" <<'__MN_{path.stem.upper()}_EOF__'\n"
        f"{path.read_text(encoding='utf-8')}__MN_{path.stem.upper()}_EOF__\n"
        for path in (*KERNEL_NODE_OPS_DEPS, _SCRIPTS_DIR / "kernel_patch_multinode.py")
    )


@dataclass(frozen=True)
class KernelPatchApply:
    """One kernel patch to apply on every pod: the target file, the base64 patch, and where backups and JIT builds go."""

    target_path: str
    patch_b64: str
    backup_dir: str
    kernel_id: str
    jit_build_dir: str = ""


def _build_multinode_apply_patch_entrypoint(patch: KernelPatchApply, timeout_sec: int) -> str:
    """Compose the head-pod entrypoint that fans out a kernel patch to every pod via heredoc-embedded kernel_patch_multinode.py."""
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"{_kernel_patch_pod_files()}"
        f'python3 "$WORK_DIR/kernel_patch_multinode.py" apply '
        f"--target-path {shlex.quote(str(patch.target_path))} "
        f"--patch-b64 {shlex.quote(str(patch.patch_b64))} "
        f"--backup-dir {shlex.quote(str(patch.backup_dir))} "
        f"--kernel-id {shlex.quote(str(patch.kernel_id))} "
        f"--jit-build-dir {shlex.quote(str(patch.jit_build_dir))} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_revert_patch_entrypoint(
    target_path: str,
    backup_map_json: str,
    timeout_sec: int,
    records_json: str = "",
) -> str:
    """Compose the head-pod entrypoint that fans out a revert via heredoc-embedded kernel_patch_multinode.py (``backup_map_json`` from the matching apply)."""
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"{_kernel_patch_pod_files()}"
        f'python3 "$WORK_DIR/kernel_patch_multinode.py" revert '
        f"--target-path {shlex.quote(str(target_path))} "
        f"--records-json {shlex.quote(str(records_json))} "
        f"--backup-map-json {shlex.quote(str(backup_map_json))} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_finalize_patch_entrypoint(
    records_json: str,
    timeout_sec: int,
) -> str:
    """Compose the head-pod entrypoint that finalizes accepted backups."""
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"{_kernel_patch_pod_files()}"
        f'python3 "$WORK_DIR/kernel_patch_multinode.py" finalize '
        f"--records-json {shlex.quote(str(records_json))} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_apply_tracelens_patch_entrypoint(
    tracelens_root: str,
    sglang_version_pin: str,
    patch_set: str = "",
) -> str:
    """Compose the head-pod entrypoint fanning out the TraceLens patch set via heredoc-embedded apply_tracelens_patch_multinode.py."""
    py = read_bundled_pod_python_script("apply_tracelens_patch_multinode.py", SHAPE_GATE_DEPS)
    extra_args = ""
    if sglang_version_pin:
        extra_args += f" --sglang-version-pin {shlex.quote(str(sglang_version_pin))}"
    if patch_set:
        extra_args += f" --patch-set {shlex.quote(str(patch_set))}"
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f'cat > "$WORK_DIR/apply_tracelens_patch_multinode.py" '
        f"<<'__MN_TLPATCH_PY_EOF__'\n"
        f"{py}__MN_TLPATCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/apply_tracelens_patch_multinode.py" '
        f"--tracelens-root {shlex.quote(str(tracelens_root))}"
        f"{extra_args}"
    )


def extract_pod_json(logs: str) -> dict | None:
    """Parse the last top-level JSON document from an interleaved Ray Dashboard job_logs blob (the in-pod scripts emit one)."""
    for candidate in _json_object_candidates(logs or ""):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None
