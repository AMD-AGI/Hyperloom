# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GpuBackend that answers measurements from an oracle socket (no GPU).

Protocol (one JSON line each way per variant), matching hyperloom-eval::

    -> {"name", "extra_server_args", "extra_envs", "remove_args", "unset_envs",
        "base_extra_args", "base_extra_envs"}
    <- {"outcome": str | null, "gain_pct": float | null, "accuracy_drop_pts": float | null,
        "match": str, "duration_sec": float}

Noise, failures, and wall time are owned by the oracle; this module only forwards requests and shapes
``VariantResult`` rows. Answers are kept for the Coordinator's lifetime: the warmup and decision rounds bench the
same variant and the oracle answers once.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shlex
import socket
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .protocol import NO_GPU_ERROR

_ENCODING = "utf-8"


def _no_effect_gain(key: tuple) -> float:
    """Stable gain in [-1.0, +0.5)% when the oracle has no recorded outcome, under any KEEP bar."""
    digest = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big")
    return -1.0 + 1.5 * digest / 2**32


def _launch_key(args: str, envs: dict) -> frozenset:
    """A launch as the set of its argv tokens and env pairs, harness controls aside."""
    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").split()
    return frozenset(tokens) | frozenset(f"{k}={v}" for k, v in (envs or {}).items() if k != "RUN_EVAL")


class _OracleClient:
    """One JSON line each way over a connected socket; with no socket every variant is unmeasured."""

    def __init__(self, fd: int | None, time_scale: float) -> None:
        self.time_scale = time_scale
        self._lock = asyncio.Lock()
        self._io = None
        if fd is not None:
            sock = socket.socket(fileno=fd)
            self._io = (sock.makefile("r", encoding=_ENCODING), sock.makefile("w", encoding=_ENCODING))

    def _ask(self, request: dict) -> dict:
        reader, writer = self._io
        writer.write(json.dumps(request) + "\n")
        writer.flush()
        return json.loads(reader.readline() or "{}")

    async def measure(self, request: dict) -> dict:
        if self._io is None:
            return {"outcome": None, "match": "no_oracle", "duration_sec": 0.0}
        async with self._lock:
            return await asyncio.to_thread(self._ask, request)


def _not_dumpable() -> None:
    """A process running as the same user cannot open this one's ``/proc/<pid>/fd`` or read its memory."""
    import ctypes

    ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE, 0


def claim_oracle_fd(fd: int) -> int:
    """Move the oracle socket off ``fd`` onto a private descriptor and point ``fd`` at /dev/null."""
    _not_dumpable()
    private = os.dup(fd)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, fd)
    os.close(devnull)
    return private


def connect_oracle(path: Path, token: str, wait_sec: float = 300.0) -> int:
    """Connect to a Unix-socket oracle and present the one-time token; return the connection's descriptor."""
    _not_dumpable()
    deadline = time.monotonic() + wait_sec
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(path))
            break
        except OSError:
            sock.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(1.0)
    sock.sendall((token + "\n").encode())
    if sock.makefile("r", encoding=_ENCODING).readline().strip() != "ok":
        raise RuntimeError("oracle refused the connection")
    return sock.detach()


def oracle_fd_from_env() -> int | None:
    """``HYPERLOOM_ORACLE_FD``, else ``HYPERLOOM_ORACLE_PATH`` with ``HYPERLOOM_ORACLE_TOKEN``; ``None`` for neither."""
    raw = (os.environ.get("HYPERLOOM_ORACLE_FD") or "").strip()
    if raw:
        return claim_oracle_fd(int(raw))
    path = (os.environ.get("HYPERLOOM_ORACLE_PATH") or "").strip()
    token = (os.environ.get("HYPERLOOM_ORACLE_TOKEN") or "").strip()
    if path and token:
        return connect_oracle(Path(path), token)
    return None


class ReplayGpuBackend:
    """Answer ``run_grid`` from an oracle; every other GPU operation fails as on a host without a device."""

    name = "replay"

    def __init__(self, *, oracle_fd: int | None = None, time_scale: float = 1.0) -> None:
        self._oracle = _OracleClient(oracle_fd, time_scale)
        self._coordinator: Any = None
        self._answered: dict[tuple, tuple[Any, float | None]] = {}
        self._launched: dict[frozenset, float] = {}
        self._accuracy_drops: dict[str, float] = {}

    def bind_coordinator(self, coordinator: Any) -> None:
        """Answer for ``coordinator``: anchors and accuracy read its shared state; earlier answers are dropped."""
        self._coordinator = coordinator
        self._answered.clear()
        self._launched.clear()
        self._accuracy_drops.clear()

    def launch(self, cmd: Sequence[str], **kwargs: Any) -> Any:
        raise RuntimeError(NO_GPU_ERROR)

    def _anchor(self, base_extra_args: str, base_extra_envs: dict) -> float:
        """The base launch's answered throughput when it was one of ours, else the live current best."""
        launched = self._launched.get(_launch_key(base_extra_args, base_extra_envs))
        if launched:
            return launched
        state = self._coordinator.shared_state
        return float((state.current_best or {}).get("tput") or getattr(state, "baseline_tput", 0.0) or 0.0)

    async def run_grid(self, **kwargs: Any) -> list[Any]:
        from hyperloom.orchestrator.actions.executors._grid_base import VariantResult

        base_extra_args = str(kwargs.get("base_extra_args") or "")
        base_extra_envs = dict(kwargs.get("base_extra_envs") or {})
        output_root = Path(kwargs["output_root"])
        slot = str(output_root)
        # Trees up to v1.1.4 predate fields such as ``benchmark_valid``.
        fields = set(inspect.signature(VariantResult).parameters)
        make = lambda **kw: VariantResult(**{k: v for k, v in kw.items() if k in fields})  # noqa: E731
        results: list[Any] = []
        for variant in kwargs.get("grid") or []:
            key = (variant.name, variant.extra_server_args, json.dumps(variant.extra_envs, sort_keys=True))
            if key in self._answered:
                result, drop = self._answered[key]
                if drop is not None:
                    self._accuracy_drops[slot] = drop
                results.append(result)
                continue
            answer = await self._oracle.measure(
                {
                    "name": variant.name,
                    "extra_server_args": variant.extra_server_args,
                    "extra_envs": dict(variant.extra_envs),
                    "remove_args": list(getattr(variant, "remove_args", []) or []),
                    "unset_envs": list(getattr(variant, "unset_envs", []) or []),
                    "base_extra_args": base_extra_args,
                    "base_extra_envs": dict(base_extra_envs),
                }
            )
            await asyncio.sleep(float(answer.get("duration_sec") or 0.0) * self._oracle.time_scale)
            anchor = self._anchor(base_extra_args, base_extra_envs)
            gain = answer.get("gain_pct")
            if answer.get("outcome") is None:
                gain = _no_effect_gain(key)
            if anchor > 0 and gain is not None:
                result = make(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="succeeded",
                    output_throughput=anchor * (1.0 + float(gain) / 100.0),
                    completed_requests=1,
                    benchmark_valid=True,
                    reported_success=True,
                    request_error_rate=0.0,
                    workspace=slot,
                    returncode=0,
                    note=variant.note,
                )
                own = _launch_key(
                    f"{base_extra_args} {variant.extra_server_args}",
                    {**base_extra_envs, **dict(variant.extra_envs)},
                )
                self._launched[own] = result.output_throughput
            else:
                # A recorded failure reads as a server that died before measuring.
                result = make(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    workspace=slot,
                    returncode=1,
                    error="server_lifecycle server phase exited with 1 before the benchmark completed",
                    error_class="magpie_nonzero_invalid_measurement",
                    note=variant.note,
                )
            drop = answer.get("accuracy_drop_pts")
            drop = float(drop) if isinstance(drop, (int, float)) else None
            if drop is not None:
                self._accuracy_drops[slot] = drop
            self._answered[key] = (result, drop)
            results.append(result)
        return results

    def parse_eval_results(
        self,
        workspace: Path | str,
        framework: str | None = None,
        benchmark_mode: str = "",
    ) -> Mapping[str, Any]:
        """The baseline accuracy, less the drop the oracle answered for this bench slot."""
        baseline = self._coordinator.shared_state.baseline_accuracy
        if not baseline:
            return {"accuracy": None}
        drop = self._accuracy_drops.get(str(workspace), 0.0)
        scale = 1.0 if baseline > 1.0 else 0.01
        return {"accuracy": max(0.0, baseline - max(drop, 0.0) * scale)}

    async def apply_patch(self, **kwargs: Any) -> Mapping[str, Any]:
        raise RuntimeError(NO_GPU_ERROR)

    async def kernel_job(self, **kwargs: Any) -> Mapping[str, Any]:
        raise RuntimeError(NO_GPU_ERROR)

    async def profile(self, **kwargs: Any) -> Mapping[str, Any]:
        raise RuntimeError(NO_GPU_ERROR)
