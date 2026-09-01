# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``recover`` ActionRunner — release leaked GPU VRAM.

Counterpart of the Robustness ``gpu_memory_leaked`` signal: when a crashed
server leaves the ROCm KFD tables attributing VRAM to dead PIDs, every
subsequent server start aborts on insufficient free memory. Invoked via
``delegate{action_name="recover", ...}`` (Robustness-only by PolicyGate)
for a soft cleanup: ``pgrep`` stale owners, SIGTERM, wait
``SERVER_KILL_WAIT_S``, SIGKILL survivors.

We deliberately do NOT reload amdgpu, reset the GPU, restart the pod/Ray
head, or touch persistent runtime config (would kill the optimizer, need
root, or affect other tenants). A failed recover surfaces as
``state == "needs_review"``.

Returned dict (the subset below is persisted to
``runs/recover/<task_id>/result.json``)::

    {
        "state":                  "succeeded" | "needs_review",
        "reason":                 echoed from params,
        "force_gpu_cleanup":      bool,
        "killed_pids":            [{pid, cmd, signal}, ...],  # actually killed
        "pre_free_mb_per_gpu":    [{gpu_id, free_mb}, ...],   # before cleanup
        "mid_free_mb_per_gpu":    [{gpu_id, free_mb}, ...],   # after kills
        "error_class":            str,                        # only on failure
        "cpu_only_sandbox":       True,   # multi-node short-circuit only
        "workspace":              str,    # return value only, not persisted
        "result_path":            str,    # return value only, not persisted
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ...loop.sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


# Process cmdline patterns for servers/benchmarks that can pin VRAM and must
# be killed during recovery (``benchmark_serving`` can hold KV-cache).
_OWNER_PATTERNS: tuple[str, ...] = (
    "sglang.launch_server",
    "sglang.srt",
    "vllm.entrypoints",
    "vllm serve",
    "EngineCore",
    "Magpie",
    "benchmark_serving",
)


def _is_multi_node_sandbox() -> bool:
    """True when running in multi-node mode (nodes >= 2).

    In multi-node mode (both Infera and RayJob), the optimizer sandbox does
    NOT own the inference server GPUs — those live in remote pods (Infera
    worker pods or RayJob head/worker pods). The sandbox may be scheduled on
    a GPU node and see local ``/dev/kfd`` + ``rocm-smi``, but:
      - The local GPUs run OTHER workloads (not ours).
      - ``rocm-smi --showmeminfo`` reports those workloads' VRAM usage.
      - ``_all_recovered`` sees low free_mb and returns False.
      - The orchestration LLM then proposes ``recover`` every tick forever.

    In multi-node mode the local GPU probe is skipped entirely; remote GPU
    health is handled by the Infera restart-server / kill-inference path
    (SSH to the actual pods).

    Single-node (``is_multi_node() == False``) is unaffected — the sandbox
    IS the GPU pod, so the local rocm-smi probe is meaningful.

    Returns:
        ``True`` when running in multi-node mode (nodes >= 2); ``False`` for
        single-node or when the mode cannot be determined.
    """
    try:
        from ._multi_node_env import is_multi_node

        return is_multi_node()
    except Exception:  # noqa: BLE001 — never block recovery; default to single-node
        log.warning("_is_multi_node_sandbox detection failed; assuming single-node", exc_info=True)
        return False


class RecoverExecutor:
    """Executable form of the ``recover`` action.

    Side-effect-free construction (all work in :meth:`__call__`); stateless
    besides the read-only tunables, so a single module-level instance suffices.
    """

    # Time we wait between SIGTERM and SIGKILL for a stuck owner.
    SERVER_KILL_WAIT_S: float = 5.0
    # Free MiB above which a GPU is considered healthy. Matches the
    # robustness-agent default ``GpuLeakConfig.free_mb_threshold``.
    FREE_MB_HEALTHY: float = 500.0
    # Owner patterns enforced by ``_kill_stale_owners``.
    OWNER_PATTERNS: tuple[str, ...] = _OWNER_PATTERNS

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the GPU recovery sequence and report the outcome.

        Probes GPU memory, optionally TERM/KILLs stale owner processes, and
        re-probes to decide success. Writes ``result.json`` to the task
        workspace when one is available.

        Args:
            ctx (RunnerContext): The action runner context carrying the
                task params (``reason``, ``force_gpu_cleanup``) and extras.

        Returns:
            dict[str, Any]: The recovery result, including ``state``
            (``"succeeded"`` / ``"needs_review"``), per-stage GPU memory
            probes, and killed PIDs.
        """
        params: dict[str, Any] = dict(getattr(ctx.task, "params", {}) or {})
        reason = str(params.get("reason", ""))
        force_cleanup = bool(params.get("force_gpu_cleanup", False))
        workspace = self._workspace_dir(ctx)

        log.info(
            "recover_executor: start reason=%r force=%s",
            reason,
            force_cleanup,
        )

        # Multi-node (Infera or RayJob): the serving GPUs live on remote pods.
        # Even when this sandbox lands on a GPU node, local rocm-smi reports
        # OTHER workloads' VRAM, so _all_recovered would never pass and the
        # orchestrator would propose recover forever. Short-circuit to success;
        # remote VRAM cleanup goes through the restart-server / kill-inference path.
        if _is_multi_node_sandbox():
            log.info(
                "recover_executor: infera CPU-only sandbox detected; skipping "
                "local rocm-smi probe (GPUs are on remote pods)."
            )
            result = {
                "state": "succeeded",
                "reason": reason,
                "force_gpu_cleanup": force_cleanup,
                "cpu_only_sandbox": True,
                "killed_pids": [],
                "pre_free_mb_per_gpu": [],
                "mid_free_mb_per_gpu": [],
            }
            if workspace is not None:
                await asyncio.to_thread(self._write_result_json, workspace, result)
                result["workspace"] = str(workspace)
                result["result_path"] = str(workspace / "result.json")
            log.info("recover_executor: succeeded (cpu_only_sandbox; no-op)")
            return result

        # 1) Probe pre-cleanup memory.
        pre = await asyncio.to_thread(self._probe_gpu_free_mb)

        # 2) Soft cleanup — TERM/KILL stale owners.
        killed: list[dict[str, Any]] = []
        if force_cleanup:
            killed = await asyncio.to_thread(self._kill_stale_owners)
        else:
            log.info("recover_executor: force_gpu_cleanup=false; skipping kill stage")

        # 3) Probe after kills.
        mid = await asyncio.to_thread(self._probe_gpu_free_mb)

        succeeded = self._all_recovered(mid)
        result: dict[str, Any] = {
            "state": "succeeded" if succeeded else "needs_review",
            "reason": reason,
            "force_gpu_cleanup": force_cleanup,
            "killed_pids": killed,
            "pre_free_mb_per_gpu": pre,
            "mid_free_mb_per_gpu": mid,
        }
        if not succeeded:
            result["error_class"] = "gpu_unhealthy_after_soft_cleanup"

        if workspace is not None:
            await asyncio.to_thread(self._write_result_json, workspace, result)
            result["workspace"] = str(workspace)
            result["result_path"] = str(workspace / "result.json")

        log.info(
            "recover_executor: %s killed=%d healthy_gpus=%d/%d",
            result["state"],
            len(killed),
            sum(1 for g in mid if g.get("free_mb", 0) >= self.FREE_MB_HEALTHY),
            len(mid),
        )
        return result

    # workspace
    def _workspace_dir(self, ctx: RunnerContext) -> Path | None:
        """Resolve the task workspace directory from the runner context.

        Args:
            ctx (RunnerContext): The action runner context.

        Returns:
            Path | None: The workspace path, or ``None`` when none is
            configured or the value is not path-like.
        """
        ws = (ctx.extra or {}).get("workspace")
        if not ws:
            return None
        try:
            return Path(ws)
        except (TypeError, ValueError):
            return None

    def _write_result_json(self, workspace: Path, payload: dict[str, Any]) -> None:
        """Write the recovery result payload to ``workspace/result.json``.

        Args:
            workspace (Path): Destination workspace directory.
            payload (dict[str, Any]): The recovery result to serialize.

        Returns:
            None: Errors are logged and swallowed (best-effort write).
        """
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "result.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning(
                "recover_executor: failed to write result.json to %s: %s",
                workspace,
                exc,
            )

    # GPU probe (rocm-smi --showmeminfo vram --csv)
    def _probe_gpu_free_mb(self) -> list[dict[str, Any]]:
        """Probe per-GPU free VRAM via ``rocm-smi --showmeminfo vram``.

        Returns:
            list[dict[str, Any]]: One snapshot per visible GPU, or an
            empty list when ``rocm-smi`` is unavailable or the probe
            fails.
        """
        if not shutil.which("rocm-smi"):
            return []
        try:
            proc = subprocess.run(
                [
                    "rocm-smi",
                    "--showmeminfo",
                    "vram",
                    "--csv",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.warning("recover_executor: rocm-smi probe failed: %s", exc)
            return []
        if proc.returncode != 0:
            log.warning(
                "recover_executor: rocm-smi exit=%d stderr=%s",
                proc.returncode,
                proc.stderr.strip()[:200],
            )
            return []
        return self._parse_rocm_smi_vram_csv(proc.stdout)

    @staticmethod
    def _parse_rocm_smi_vram_csv(text: str) -> list[dict[str, Any]]:
        """Parse rocm-smi `--showmeminfo vram --csv` output.

        Format (one block):

            device,VRAM Total Memory (B),VRAM Total Used Memory (B)
            card0,206158430208,205678182400
            card1,206158430208,205678182400

        Args:
            text (str): The raw ``rocm-smi --csv`` stdout to parse.

        Returns:
            list[dict[str, Any]]: ``{gpu_id, vram_total_mb, vram_used_mb,
            free_mb}`` per visible card, sorted by ``gpu_id``.
        """
        by_id: dict[int, dict[str, Any]] = {}
        header: list[str] | None = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                header = None
                continue
            cells = [c.strip() for c in line.split(",")]
            if not cells:
                continue
            if cells[0].lower() == "device":
                header = cells
                continue
            if header is None or not cells[0].lower().startswith("card"):
                continue
            try:
                gpu_id = int(cells[0][4:])
            except ValueError:
                continue
            snap = by_id.setdefault(gpu_id, {"gpu_id": gpu_id})
            for col_idx in range(1, min(len(cells), len(header))):
                h = header[col_idx]
                try:
                    val = float(cells[col_idx])
                except ValueError:
                    continue
                if h == "VRAM Total Memory (B)":
                    snap["vram_total_mb"] = val / (1024.0 * 1024.0)
                elif h == "VRAM Total Used Memory (B)":
                    snap["vram_used_mb"] = val / (1024.0 * 1024.0)
        out: list[dict[str, Any]] = []
        for k in sorted(by_id):
            snap = by_id[k]
            used = snap.get("vram_used_mb")
            total = snap.get("vram_total_mb")
            if isinstance(used, (int, float)) and isinstance(total, (int, float)):
                snap["free_mb"] = max(0.0, float(total) - float(used))
            out.append(snap)
        return out

    def _all_recovered(self, gpus: list[dict[str, Any]]) -> bool:
        """Return whether every probed GPU is above the healthy floor.

        Args:
            gpus (list[dict[str, Any]]): Per-GPU memory snapshots.

        Returns:
            bool: ``True`` iff the list is non-empty and every GPU's
            ``free_mb`` is at least :attr:`FREE_MB_HEALTHY`.
        """
        if not gpus:
            # No probe -> treat as unhealthy.
            return False
        return all(
            isinstance(snap.get("free_mb"), (int, float)) and snap["free_mb"] >= self.FREE_MB_HEALTHY for snap in gpus
        )

    # soft cleanup — pgrep + kill loop
    def _kill_stale_owners(self) -> list[dict[str, Any]]:
        """SIGTERM then SIGKILL stale owners matching :data:`OWNER_PATTERNS`.

        Returns one record per signalled PID (cmdline at discovery + final
        signal name ``"TERM"`` / ``"KILL"``).

        Returns:
            One record per signalled PID, or ``[]`` when none were stale.
        """
        candidates = self._discover_stale_pids()
        if not candidates:
            return []
        killed: list[dict[str, Any]] = []
        for entry in candidates:
            pid = entry["pid"]
            if self._send_signal(pid, signal.SIGTERM):
                entry["signal"] = "TERM"
                killed.append(entry)
        if not killed:
            return []
        # Wait then SIGKILL survivors of the TERMed set (no re-discover).
        time.sleep(self.SERVER_KILL_WAIT_S)
        for entry in killed:
            pid = entry["pid"]
            if self._pid_alive(pid) and self._send_signal(pid, signal.SIGKILL):
                entry["signal"] = "KILL"
        return killed

    def _discover_stale_pids(self) -> list[dict[str, Any]]:
        """Run ``pgrep -a -f -- <pattern>`` per owner pattern (matches the
        full cmdline) and return unique PID records, excluding our own PID.

        Returns:
            Unique PID records matching the owner patterns, or ``[]`` when
            ``pgrep`` is unavailable or nothing matched.
        """
        if not shutil.which("pgrep"):
            log.warning("recover_executor: pgrep not on PATH; skipping kill stage")
            return []
        own_pid = os.getpid()
        seen: dict[int, dict[str, Any]] = {}
        for pattern in self.OWNER_PATTERNS:
            try:
                proc = subprocess.run(
                    ["pgrep", "-a", "-f", "--", pattern],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                log.warning("recover_executor: pgrep(%r) failed: %s", pattern, exc)
                continue
            if proc.returncode not in (0, 1):  # 1 = no matches
                continue
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                cmd = parts[1]
                if pid == own_pid:
                    continue
                # Confirm the pattern via plain substring (pgrep's regex is permissive).
                if pattern not in cmd:
                    continue
                seen[pid] = {"pid": pid, "cmd": cmd, "pattern": pattern}
        return list(seen.values())

    def _send_signal(self, pid: int, sig: signal.Signals) -> bool:
        """Send a signal to a PID, tolerating dead/forbidden processes.

        Args:
            pid (int): Target process id.
            sig (signal.Signals): The signal to deliver.

        Returns:
            bool: ``True`` if the signal was delivered; ``False`` if the
            process is gone or permission was denied.
        """
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            log.warning(
                "recover_executor: cannot signal pid=%d sig=%s: %s",
                pid,
                sig.name,
                exc,
            )
            return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Check whether a process is still alive via signal 0.

        Args:
            pid (int): Target process id.

        Returns:
            bool: ``True`` if the process exists; ``False`` otherwise.
        """
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


# Module-level callable for ``register_executor("recover", recover_executor)``.
recover_executor = RecoverExecutor()


def probe_gpu_free_mb() -> list[dict[str, Any]]:
    """Per-GPU free VRAM, for callers that need the probe without the action.

    Blocking (shells out to ``rocm-smi``); call via ``asyncio.to_thread``.

    Returns:
        list[dict[str, Any]]: One entry per visible GPU, or ``[]`` when the
        probe is unavailable.
    """
    return recover_executor._probe_gpu_free_mb()


__all__ = [
    "RecoverExecutor",
    "probe_gpu_free_mb",
    "recover_executor",
]
