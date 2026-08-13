#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Launcher-side preflight for hyperloom.inference_optimizer.

Usage:
    python src/hyperloom/inference_optimizer/tools/preflight_optimizer.py MODEL_PATH
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


STALE_PROCESS_PATTERNS = (
    "hyperloom.inference_optimizer.cli",
    "Magpie",
    "sglang.launch_server",
    "vllm.entrypoints",
)


def _read_cmdline(pid: str) -> str:
    """Read and decode the ``/proc/<pid>/cmdline`` for the given pid.

    Args:
        pid: The process id (as a string) whose command line to read.

    Returns:
        The space-joined command line, or ``""`` when it cannot be read.
    """
    try:
        raw = pathlib.Path("/proc", pid, "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "ignore")


def _print_torch_visibility() -> bool:
    """Print torch CUDA visibility and report whether a device is usable.

    Returns:
        ``True`` when torch imports and reports at least one available CUDA
        device, ``False`` otherwise (including on import failure).
    """
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        print("torch_check_error=", type(exc).__name__, str(exc)[:300])
        return False

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    print("torch_cuda_available=", available)
    print("torch_cuda_device_count=", count)
    return available and count > 0


def _print_rocm_snapshot() -> None:
    """Print a ``rocm-smi`` memory-use snapshot (best-effort, never raises)."""
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        print("rocm_smi=missing")
        return

    try:
        result = subprocess.run(
            [rocm_smi, "--showmemuse", "--json"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception as exc:
        print("rocm_smi_error=", type(exc).__name__, str(exc)[:300])
        return

    print("rocm_smi_rc=", result.returncode)
    if result.stdout.strip():
        print(result.stdout.strip()[:2000])
    if result.stderr.strip():
        print("rocm_smi_stderr=", result.stderr.strip()[:500])


def _forge_loop_options(forge_path: str) -> set[str] | None:
    """Return the options the installed forge-loop accepts, hidden ones included.

    Args:
        forge_path: Directory holding ``kernel_agents``, as ``$FORGE_PATH`` names
            it; empty relies on an installed package.

    Returns:
        The accepted option strings, or ``None`` when KernelForge cannot be
        imported or exposes no ``forge-loop`` command.
    """
    if forge_path and forge_path not in sys.path:
        sys.path.insert(0, forge_path)
    try:
        from kernel_agents import cli  # type: ignore[import-not-found]
    except Exception as exc:
        print("forge_loop_contract=unavailable", type(exc).__name__, str(exc)[:200])
        return None

    command = getattr(cli, "main", None)
    command = getattr(command, "commands", {}).get("forge-loop") if command else None
    if command is None:
        print("forge_loop_contract=unavailable no forge-loop command")
        return None

    accepted: set[str] = set()
    for param in command.params:
        accepted.update(param.opts)
        accepted.update(param.secondary_opts)
    return accepted


def _print_forge_loop_contract() -> bool:
    """Report whether the installed forge-loop accepts every option we pass.

    click rejects an unknown option while parsing, before running anything, so a
    KernelForge that retired one of these loses every forge attempt with rc=2 --
    hours into a run, and only in the per-attempt logs. Saying so at launch costs
    a second and names the option.

    Returns:
        ``True`` when the contract holds or KernelForge is absent (a run without
        it simply has no forge backend to break), ``False`` on a real mismatch.
    """
    accepted = _forge_loop_options(os.environ.get("FORGE_PATH", "").strip())
    if accepted is None:
        return True

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    backends = repo_root / "hyperloom" / "agents" / "kernel" / "tools" / "backends"
    if str(backends) not in sys.path:
        sys.path.insert(0, str(backends))
    try:
        import forge_submit  # type: ignore[import-not-found]
    except Exception as exc:
        print("forge_loop_contract=unchecked", type(exc).__name__, str(exc)[:200])
        return True

    missing = sorted(set(forge_submit.FORGE_LOOP_OPTIONS) - accepted)
    if missing:
        print(
            "forge_loop_contract_missing=" + ",".join(missing),
            file=sys.stderr,
        )
        print(
            "the installed forge-loop rejects these, so every forge attempt "
            "would exit rc=2; install a KernelForge that has them or stop "
            "passing them",
            file=sys.stderr,
        )
        return False

    print("forge_loop_contract_ok=", len(forge_submit.FORGE_LOOP_OPTIONS))
    return True


def _find_stale_processes() -> list[tuple[str, str]]:
    """Scan ``/proc`` for running processes matching known stale patterns.

    Returns:
        A list of ``(pid, cmdline)`` tuples for processes (other than this one)
        whose command line matches one of :data:`STALE_PROCESS_PATTERNS`.
    """
    matches: list[tuple[str, str]] = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        text = _read_cmdline(pid)
        if text and any(pattern in text for pattern in STALE_PROCESS_PATTERNS):
            matches.append((pid, text[:300]))
    return matches


def main() -> int:
    """Run launcher preflight checks and return a process exit code.

    Validates the model path, prints torch/ROCm visibility, checks the forge-loop
    argv contract against the installed KernelForge, and reports any stale
    optimizer/server processes.

    Returns:
        ``0`` when every check passes, otherwise ``2``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Model directory to optimize.")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path)
    ok = True

    if not model_path.is_dir():
        print(f"model_path_missing={model_path}", file=sys.stderr)
        ok = False
    else:
        print(f"model_path_ok={model_path}")

    if not _print_torch_visibility():
        ok = False

    if not _print_forge_loop_contract():
        ok = False

    _print_rocm_snapshot()

    stale = _find_stale_processes()
    for pid, cmdline in stale:
        print(f"existing_process {pid}: {cmdline}")
    if stale:
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
