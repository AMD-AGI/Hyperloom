# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Real ``recover`` ActionRunner — release leaked GPU VRAM.

The recover action is the inference_optimizer counterpart of the
Robustness ``gpu_memory_leaked`` signal: when the resident server
process (sglang / vLLM EngineCore / Magpie wrapper) has crashed and the
ROCm KFD driver tables still attribute VRAM allocations to dead PIDs,
every subsequent server start aborts with ``Free memory on device cuda:N
(0.0/191.98 GiB) ... less than gpu-memory-utilization``. The optimizer
loops on the failing server start and burns the remaining budget.

``RecoverExecutor`` is invoked via ``delegate{action_name="recover",
params={force_gpu_cleanup: True, ...}}`` (Robustness-only by PolicyGate)
and performs a tiered, escalating cleanup:

1. **soft** — list stale owners with ``pgrep -af <pattern>``, SIGTERM
   each PID, wait ``SERVER_KILL_WAIT_S``, SIGKILL any survivors.
2. **hard** (env-gated) — when soft cleanup leaves VRAM still pinned
   AND ``HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1`` is exported, shell out to
   ``rocm-smi --gpureset --gpu=all`` with a 30s subprocess timeout and
   capture stdout / stderr / returncode for the result.json audit.

We deliberately do NOT:

* reload the ``amdgpu`` kernel module,
* restart the pod / container / Ray head,
* touch ``~/.claude/config.json`` or any other persistent runtime config.

These would either kill the running optimizer (so resume would be the
only recovery surface) or require root that the typical sandbox lacks.
A failed recover surfaces as ``state == "needs_review"`` and the
Robustness escalate_strategy_change advisory tells Orchestration to
fall back to a deterministic ``report`` proposal.

Wire-up::

    sub.register_executor("recover", recover_executor)

Returned dict (also persisted to ``runs/recover/<task_id>/result.json``):

    {
        "state":                  "succeeded" | "needs_review",
        "reason":                 echoed from params,
        "force_gpu_cleanup":      bool,
        "allow_reset_env":        bool,                       # gate state
        "killed_pids":            [{pid, cmd, signal}, ...],  # actually killed
        "pre_free_mb_per_gpu":    [{gpu_id, free_mb}, ...],   # before cleanup
        "mid_free_mb_per_gpu":    [{gpu_id, free_mb}, ...],   # after kills
        "post_free_mb_per_gpu":   [{gpu_id, free_mb}, ...],   # after gpureset
        "gpureset_attempted":     bool,
        "gpureset_result":        {returncode, stdout, stderr, error} | {},
        "error_class":            str,                        # only on failure
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

from ..sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


# Owner patterns we treat as "legitimate VRAM owners". The list mirrors
# the robustness-agent extension in 2026-05 plus a couple of extra
# patterns we want to clean up if they're stuck (``benchmark_serving``
# can hold KV-cache, ``cudagraph_capture`` workers leak after a server
# crash mid-capture).
_OWNER_PATTERNS: tuple[str, ...] = (
    "sglang.launch_server",
    "sglang.srt",
    "vllm.entrypoints",
    "vllm serve",
    "EngineCore",
    "Magpie",
    "benchmark_serving",
)


def _env_gate_allows_gpureset() -> bool:
    """``HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1`` opt-in for hard recovery."""
    return os.getenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", "").strip() == "1"


def _is_multi_node_sandbox() -> bool:
    """True when running in multi-node mode (nodes >= 2).

    In multi-node mode (both Dynamo and RayJob), the optimizer sandbox does
    NOT own the inference server GPUs — those live in remote pods (Dynamo
    worker pods or RayJob head/worker pods). The sandbox may be scheduled on
    a GPU node and see local ``/dev/kfd`` + ``rocm-smi``, but:
      - The local GPUs run OTHER workloads (not ours).
      - ``rocm-smi --showmeminfo`` reports those workloads' VRAM usage.
      - ``_all_recovered`` sees low free_mb and returns False.
      - The orchestration LLM then proposes ``recover`` every tick forever.

    The fix: skip the local GPU probe entirely in multi-node mode. Remote
    GPU health is handled by the Dynamo restart-server / kill-inference
    path (SSH to the actual pods).

    Single-node (``is_multi_node() == False``) is unaffected — the sandbox
    IS the GPU pod, so local rocm-smi / gpureset are meaningful.
    """
    try:
        from ._multi_node_env import is_multi_node
        return is_multi_node()
    except Exception:  # noqa: BLE001 - never block recovery on import error
        return False


class RecoverExecutor:
    """Executable form of the ``recover`` action.

    Construction does not touch the host; all side-effects happen inside
    :meth:`__call__`. The class is stateless besides the read-only
    tunables below so we can keep a single module-level instance bound
    in :data:`recover_executor`.
    """

    # Time we wait between SIGTERM and SIGKILL for a stuck owner.
    SERVER_KILL_WAIT_S: float = 5.0
    # Hard timeout for the ``rocm-smi --gpureset`` shell-out.
    GPURESET_TIMEOUT_S: float = 30.0
    # Free MiB above which a GPU is considered healthy. Matches the
    # robustness-agent default ``GpuLeakConfig.free_mb_threshold``.
    FREE_MB_HEALTHY: float = 500.0
    # Owner patterns enforced by ``_kill_stale_owners``.
    OWNER_PATTERNS: tuple[str, ...] = _OWNER_PATTERNS

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        params: dict[str, Any] = dict(getattr(ctx.task, "params", {}) or {})
        reason = str(params.get("reason", ""))
        force_cleanup = bool(params.get("force_gpu_cleanup", False))
        allow_reset = _env_gate_allows_gpureset()
        workspace = self._workspace_dir(ctx)

        log.info(
            "recover_executor: start reason=%r force=%s allow_reset_env=%s",
            reason, force_cleanup, allow_reset,
        )

        # Dynamo CPU-only sandbox: no local GPUs to reclaim (they live on remote
        # pods reached over SSH). Calling rocm-smi here deadlocks on the kfd
        # ioctl and makes recover loop. Short-circuit to success so the
        # orchestrator stops proposing recover; remote VRAM cleanup, when
        # needed, is handled by the dynamo restart-server / kill-inference path.
        if _is_multi_node_sandbox():
            log.info(
                "recover_executor: dynamo CPU-only sandbox detected; skipping "
                "local rocm-smi probe + gpureset (GPUs are on remote pods)."
            )
            result = {
                "state": "succeeded",
                "reason": reason,
                "force_gpu_cleanup": force_cleanup,
                "allow_reset_env": allow_reset,
                "cpu_only_sandbox": True,
                "killed_pids": [],
                "pre_free_mb_per_gpu": [],
                "mid_free_mb_per_gpu": [],
                "post_free_mb_per_gpu": [],
                "gpureset_attempted": False,
                "gpureset_result": {},
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
            log.info(
                "recover_executor: force_gpu_cleanup=false; skipping kill stage"
            )

        # 3) Probe after kills.
        mid = await asyncio.to_thread(self._probe_gpu_free_mb)

        # 4) Hard cleanup (gpureset) — gated by env + force_cleanup.
        gpureset_result: dict[str, Any] | None = None
        if (
            force_cleanup
            and allow_reset
            and not self._all_recovered(mid)
        ):
            gpureset_result = await asyncio.to_thread(
                self._try_rocm_smi_gpureset
            )

        # 5) Final probe (only matters if we attempted gpureset).
        post = (
            await asyncio.to_thread(self._probe_gpu_free_mb)
            if gpureset_result is not None
            else mid
        )

        succeeded = self._all_recovered(post)
        result: dict[str, Any] = {
            "state": "succeeded" if succeeded else "needs_review",
            "reason": reason,
            "force_gpu_cleanup": force_cleanup,
            "allow_reset_env": allow_reset,
            "killed_pids": killed,
            "pre_free_mb_per_gpu": pre,
            "mid_free_mb_per_gpu": mid,
            "post_free_mb_per_gpu": post,
            "gpureset_attempted": gpureset_result is not None,
            "gpureset_result": gpureset_result or {},
        }
        if not succeeded:
            result["error_class"] = (
                "gpu_unhealthy_after_gpureset"
                if gpureset_result is not None
                else "gpu_unhealthy_after_soft_cleanup"
            )

        if workspace is not None:
            await asyncio.to_thread(self._write_result_json, workspace, result)
            result["workspace"] = str(workspace)
            result["result_path"] = str(workspace / "result.json")

        log.info(
            "recover_executor: %s killed=%d gpureset=%s healthy_gpus=%d/%d",
            result["state"],
            len(killed),
            result["gpureset_attempted"],
            sum(1 for g in post if g.get("free_mb", 0) >= self.FREE_MB_HEALTHY),
            len(post),
        )
        return result

    # ------------------------------------------------------------------
    # workspace
    # ------------------------------------------------------------------
    def _workspace_dir(self, ctx: RunnerContext) -> Path | None:
        ws = (ctx.extra or {}).get("workspace")
        if not ws:
            return None
        try:
            return Path(ws)
        except (TypeError, ValueError):
            return None

    def _write_result_json(self, workspace: Path, payload: dict[str, Any]) -> None:
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "result.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning(
                "recover_executor: failed to write result.json to %s: %s",
                workspace, exc,
            )

    # ------------------------------------------------------------------
    # GPU probe (rocm-smi --showmeminfo vram --csv)
    # ------------------------------------------------------------------
    def _probe_gpu_free_mb(self) -> list[dict[str, Any]]:
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
                proc.returncode, proc.stderr.strip()[:200],
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

        Returns ``[{gpu_id, vram_total_mb, vram_used_mb, free_mb}, ...]``
        per visible card, sorted by gpu_id.
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
        if not gpus:
            # Without a probe we can't claim recovery; treat as unhealthy.
            return False
        return all(
            isinstance(snap.get("free_mb"), (int, float))
            and snap["free_mb"] >= self.FREE_MB_HEALTHY
            for snap in gpus
        )

    # ------------------------------------------------------------------
    # soft cleanup — pgrep + kill loop
    # ------------------------------------------------------------------
    def _kill_stale_owners(self) -> list[dict[str, Any]]:
        """SIGTERM then SIGKILL stale owners matching :data:`OWNER_PATTERNS`.

        Returns one record per PID we actually sent a signal to. Each
        record captures the cmdline at discovery time and the final
        signal name (``"TERM"`` or ``"KILL"``).
        """
        candidates = self._discover_stale_pids()
        if not candidates:
            return []
        killed: list[dict[str, Any]] = []
        # First pass: SIGTERM everyone.
        for entry in candidates:
            pid = entry["pid"]
            if self._send_signal(pid, signal.SIGTERM):
                entry["signal"] = "TERM"
                killed.append(entry)
        if not killed:
            return []
        # Wait, then SIGKILL anyone still alive. We do not re-discover
        # the candidates here — the goal is to kill exactly the set we
        # already TERMed, not to chase newly-spawned processes.
        time.sleep(self.SERVER_KILL_WAIT_S)
        for entry in killed:
            pid = entry["pid"]
            if self._pid_alive(pid) and self._send_signal(pid, signal.SIGKILL):
                entry["signal"] = "KILL"
        return killed

    def _discover_stale_pids(self) -> list[dict[str, Any]]:
        """Run pgrep for each owner pattern and return unique PID records.

        We use ``pgrep -a -f -- '<pattern>'`` so the pattern is matched
        anywhere in the full cmdline (matches Magpie wrappers whose
        argv[0] is ``python``). The ``--`` guards against patterns that
        could be mistaken for flags.

        Excludes our own PID so the recover task can't kill itself.
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
                log.warning(
                    "recover_executor: pgrep(%r) failed: %s", pattern, exc
                )
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
                # Defence-in-depth: confirm the pattern truly appears in
                # the cmdline (pgrep's regex engine is permissive on some
                # platforms). Use plain substring so callers can pass
                # literal owner strings without escaping.
                if pattern not in cmd:
                    continue
                seen[pid] = {"pid": pid, "cmd": cmd, "pattern": pattern}
        return list(seen.values())

    def _send_signal(self, pid: int, sig: signal.Signals) -> bool:
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            log.warning(
                "recover_executor: cannot signal pid=%d sig=%s: %s",
                pid, sig.name, exc,
            )
            return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    # ------------------------------------------------------------------
    # hard cleanup — rocm-smi --gpureset (env-gated)
    # ------------------------------------------------------------------
    def _try_rocm_smi_gpureset(self) -> dict[str, Any]:
        if not shutil.which("rocm-smi"):
            return {"error": "rocm-smi not on PATH"}
        log.warning(
            "recover_executor: HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1; "
            "attempting `rocm-smi --gpureset --gpu=all` (best effort, "
            "typically requires root)"
        )
        try:
            proc = subprocess.run(
                ["rocm-smi", "--gpureset", "--gpu=all"],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.GPURESET_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            log.warning(
                "recover_executor: gpureset timed out after %.0fs",
                self.GPURESET_TIMEOUT_S,
            )
            return {
                "returncode": None,
                "error": "timeout",
                "timeout_s": self.GPURESET_TIMEOUT_S,
                "stdout": (exc.stdout or "")[:400] if exc.stdout else "",
                "stderr": (exc.stderr or "")[:400] if exc.stderr else "",
            }
        except (FileNotFoundError, OSError) as exc:
            log.warning("recover_executor: gpureset failed to launch: %s", exc)
            return {"error": f"launch_failed: {exc}"}
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }


# Module-level callable so callers can do
# ``register_executor("recover", recover_executor)``.
recover_executor = RecoverExecutor()


__all__ = [
    "RecoverExecutor",
    "recover_executor",
]
