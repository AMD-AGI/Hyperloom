# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``recover`` ActionRunner — release leaked GPU VRAM."""

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
from ._server_lifecycle import _pid_cmdline as _shared_pid_cmdline


log = logging.getLogger(__name__)


# Process cmdline patterns for servers/benchmarks that can pin VRAM and must be killed during recovery
# (``benchmark_serving`` can hold KV-cache).
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
    """True when running in multi-node mode (nodes >= 2)."""
    try:
        from ._multi_node_env import is_multi_node

        return is_multi_node()
    except Exception:  # noqa: BLE001 — never block recovery; default to single-node
        log.warning("_is_multi_node_sandbox detection failed; assuming single-node", exc_info=True)
        return False


class RecoverExecutor:
    """Executable form of the ``recover`` action."""

    # Time we wait between SIGTERM and SIGKILL for a stuck owner.
    SERVER_KILL_WAIT_S: float = 5.0
    # Free MiB above which a GPU is considered healthy.
    FREE_MB_HEALTHY: float = 500.0
    # Owner patterns enforced by ``_kill_stale_owners``.
    OWNER_PATTERNS: tuple[str, ...] = _OWNER_PATTERNS
    _pid_cmdline = staticmethod(_shared_pid_cmdline)

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the GPU recovery sequence and report the outcome."""
        params: dict[str, Any] = dict(getattr(ctx.task, "params", {}) or {})
        reason = str(params.get("reason", ""))
        force_cleanup = bool(params.get("force_gpu_cleanup", False))
        workspace = self._workspace_dir(ctx)
        self._active_session_dir = self._session_dir(ctx)

        log.info(
            "recover_executor: start reason=%r force=%s",
            reason,
            force_cleanup,
        )

        # Multi-node (Infera or RayJob): the serving GPUs live on remote pods.
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
        from ._multi_node_env import uses_external_server

        killed: list[dict[str, Any]] = []
        if not force_cleanup:
            log.info("recover_executor: force_gpu_cleanup=false; skipping kill stage")
        elif uses_external_server():
            # OWNER_PATTERNS match by cmdline, so the engine behind an external
            # endpoint would be TERM/KILLed here even though we never launched
            # it -- restarting the server the benchmark is measuring. The GPU
            # probe above still applies: single-node, those GPUs are local.
            log.info("recover_executor: external server; skipping kill stage (engine is not ours)")
        else:
            killed = await asyncio.to_thread(self._kill_stale_owners)

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
        """Resolve the task workspace directory from the runner context."""
        ws = (ctx.extra or {}).get("workspace")
        if not ws:
            return None
        try:
            return Path(ws)
        except (TypeError, ValueError):
            return None

    def _session_dir(self, ctx: RunnerContext) -> Path | None:
        """Resolve the session directory that owns recover pidfiles."""
        sd = (ctx.extra or {}).get("session_dir")
        if not sd:
            return None
        try:
            return Path(sd)
        except (TypeError, ValueError):
            return None

    def _write_result_json(self, workspace: Path, payload: dict[str, Any]) -> None:
        """Write the recovery result payload to ``workspace/result.json``."""
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
        """Probe per-GPU free VRAM via ``rocm-smi --showmeminfo vram``."""
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
        """Parse rocm-smi `--showmeminfo vram --csv` output."""
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
        """Return whether every probed GPU is above the healthy floor."""
        if not gpus:
            # No probe -> treat as unhealthy.
            return False
        return all(
            isinstance(snap.get("free_mb"), (int, float)) and snap["free_mb"] >= self.FREE_MB_HEALTHY for snap in gpus
        )

    # soft cleanup — session pidfiles + kill loop
    def _kill_stale_owners(self) -> list[dict[str, Any]]:
        """SIGTERM then SIGKILL owners recorded in this session's pidfiles."""
        candidates = self._discover_stale_pids()
        if not candidates:
            return []
        killed: list[dict[str, Any]] = []
        for entry in candidates:
            pid = entry["pid"]
            cmd = str(entry.get("cmd", ""))
            pattern = next((marker for marker in self.OWNER_PATTERNS if marker in cmd), None)
            if pattern is None:
                log.warning(
                    "recover_executor: pid %d is not a recognized session owner; not signalling",
                    pid,
                )
                self._remove_finished_pidfile(entry)
                continue
            entry["pattern"] = pattern
            pgid = entry.get("pgid")
            sent = (
                self._send_group_signal(int(pgid), signal.SIGTERM) or self._send_signal(pid, signal.SIGTERM)
                if isinstance(pgid, int)
                else self._send_signal(pid, signal.SIGTERM)
            )
            if sent:
                entry["signal"] = "TERM"
                killed.append(entry)
        if not killed:
            return []
        # Wait then SIGKILL survivors of the TERMed set (no re-discover).
        time.sleep(self.SERVER_KILL_WAIT_S)
        for entry in killed:
            pid = entry["pid"]
            pgid = entry.get("pgid")
            if isinstance(pgid, int):
                still_owned = bool(self._process_group_owner_cmd(pgid))
                alive = self._process_group_alive(pgid)
                if alive and still_owned:
                    sent = self._send_group_signal(pgid, signal.SIGKILL)
                else:
                    current_cmd = self._pid_cmdline(pid)
                    pid_owned = any(marker in current_cmd for marker in self.OWNER_PATTERNS)
                    sent = self._pid_alive(pid) and pid_owned and self._send_signal(pid, signal.SIGKILL)
            else:
                current_cmd = self._pid_cmdline(pid)
                still_owned = any(marker in current_cmd for marker in self.OWNER_PATTERNS)
                sent = self._pid_alive(pid) and still_owned and self._send_signal(pid, signal.SIGKILL)
            if sent:
                entry["signal"] = "KILL"
            self._remove_finished_pidfile(entry, force=bool(sent))
        return killed

    def _send_group_signal(self, pgid: int, sig: signal.Signals) -> bool:
        """Signal an owned process group without touching our own group."""
        if os.name != "posix" or pgid <= 0 or pgid == os.getpgrp():
            return False
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError as exc:
            log.warning("recover_executor: cannot signal pgid=%d sig=%s: %s", pgid, sig.name, exc)
            return False

    @staticmethod
    def _process_group_alive(pgid: int) -> bool:
        """Return whether a POSIX process group still has members."""
        if os.name != "posix":
            return False
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _process_group_owner_cmd(self, pgid: int) -> str:
        """Return a recognized owner cmdline from ``pgid``, if one exists."""
        try:
            entries = list(Path("/proc").iterdir())
        except OSError:
            return ""
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                entry_pgid = int(stat.rsplit(")", 1)[1].split()[2])
            except (IndexError, OSError, ValueError):
                continue
            if entry_pgid != pgid:
                continue
            cmd = self._pid_cmdline(int(entry.name))
            if any(marker in cmd for marker in self.OWNER_PATTERNS):
                return cmd
        return ""

    def _remove_finished_pidfile(self, entry: dict[str, Any], *, force: bool = False) -> None:
        """Remove a pidfile after its recorded process group has exited."""
        pid_file = entry.get("pid_file")
        if not isinstance(pid_file, str):
            return
        pgid = entry.get("pgid")
        alive = self._pid_alive(entry["pid"])
        if isinstance(pgid, int):
            alive = alive or self._process_group_alive(pgid)
        if alive and not force:
            return
        path = Path(pid_file)
        for candidate in (path, path.with_suffix(".json")):
            try:
                candidate.unlink()
            except OSError:
                # Best-effort cleanup: an absent or locked pidfile is not an error.
                pass

    def _discover_stale_pids(self) -> list[dict[str, Any]]:
        """Return unique PIDs from this session's ``runs/**/*.pid`` files."""
        session_dir = getattr(self, "_active_session_dir", None)
        if session_dir is None:
            return []
        runs = Path(session_dir) / "runs"
        if not runs.is_dir():
            return []
        own_pid = os.getpid()
        seen: dict[int, dict[str, Any]] = {}
        for pid_file in sorted(runs.rglob("*.pid")):
            try:
                parts = pid_file.read_text(encoding="utf-8").split()
            except OSError:
                continue
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == own_pid:
                continue
            try:
                pgid = int(parts[1]) if len(parts) > 1 else pid
            except ValueError:
                pgid = pid
            cmd = self._pid_cmdline(pid)
            if not cmd and pgid != own_pid:
                cmd = self._process_group_owner_cmd(pgid)
            seen[pid] = {
                "pid": pid,
                "pgid": pgid,
                "pid_file": str(pid_file),
                "cmd": cmd,
                "pattern": "session_pidfile",
            }
        return list(seen.values())

    def _send_signal(self, pid: int, sig: signal.Signals) -> bool:
        """Send a signal to a PID, tolerating dead/forbidden processes."""
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
        """Check whether a process is still alive via signal 0."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


# Module-level callable for ``register_executor("recover", recover_executor)``.
recover_executor = RecoverExecutor()


def probe_gpu_free_mb() -> list[dict[str, Any]]:
    """Per-GPU free VRAM, for callers that need the probe without the action."""
    return recover_executor._probe_gpu_free_mb()


__all__ = [
    "RecoverExecutor",
    "probe_gpu_free_mb",
    "recover_executor",
]
