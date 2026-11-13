# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The four ranks of a seam test, each its own process (``seam_rank.py``), so a rank that a failed
wait stops is seen stopping, alone, with its exit status and the time it took."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

TP = 4
WEIGHT_SEED = 20261112
WORKER = Path(__file__).with_name("seam_rank.py")

# replay: graphs of three widths replayed interleaved, rounds of the order below
REPLAY_WIDTHS = (1, 4, 16)
REPLAY_ORDER = (1, 4, 16, 16, 4, 1, 4, 1, 16)
REPLAY_ROUNDS = 3
# launches issued back to back, three inputs in turn: a mailbox half still holds the launch two
# before, so that launch's answer must not be the current one
BACK_TO_BACK = 24
BACK_TO_BACK_WIDTH = 8
BACK_TO_BACK_SEEDS = (31, 32, 33)
# launches from the epoch below, across its wrap to 0
WRAP_FROM = 2**32 - 3
WRAP_LAUNCHES = 6

FAULT_WIDTH = 4
LAUNCHES_UNDER_FAULT = 4  # a rank stalled in launch 2 leaves the others waiting in launch 3
MARGIN_S = 30.0  # past a wait's deadline: the stop's propagation and the process teardown


@dataclass(frozen=True)
class RankRun:
    rank: int
    returncode: int
    stdout: str
    stderr: str
    exited: float  # wall clock

    def lines(self, tag: str) -> list[str]:
        return [line[len(tag) + 1 :] for line in self.stdout.splitlines() if line.startswith(f"{tag} ")]

    def verdicts(self) -> list[dict]:
        return [json.loads(line) for line in self.lines("RESULT")]

    @property
    def finished(self) -> bool:
        return "FINISHED" in self.stdout.splitlines()

    @property
    def launched(self) -> float | None:
        """When the rank issued its first launch of a fault scenario (wall clock)."""
        found = self.lines("LAUNCH")
        return float(found[0]) if found else None

    def tail(self, lines: int = 40) -> str:
        err = "\n".join(self.stderr.splitlines()[-lines:])
        return f"rank {self.rank} exited {self.returncode}; stderr tail:\n{err}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_ranks(*args: str, env: dict[str, str] | None = None, timeout: float) -> list[RankRun]:
    """Run ``seam_rank.py *args`` as ranks 0 .. TP-1 on GPUs 0 .. TP-1 and wait for all of them."""
    port = str(_free_port())
    with tempfile.TemporaryDirectory() as tmp:
        logs, procs = [], []
        for rank in range(TP):
            out, err = Path(tmp, f"{rank}.out"), Path(tmp, f"{rank}.err")
            logs.append((out, err))
            rank_env = {
                **os.environ,
                **(env or {}),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(TP),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": port,
            }
            with out.open("w") as o, err.open("w") as e:
                procs.append(subprocess.Popen([sys.executable, str(WORKER), *args], env=rank_env, stdout=o, stderr=e))
        exited: dict[int, float] = {}
        end = time.monotonic() + timeout
        while len(exited) < TP and time.monotonic() < end:
            for rank, proc in enumerate(procs):
                if rank not in exited and proc.poll() is not None:
                    exited[rank] = time.time()
            time.sleep(0.1)
        hung = [rank for rank in range(TP) if rank not in exited]
        for rank in hung:
            procs[rank].kill()
            procs[rank].wait()
            exited[rank] = time.time()
        runs = [
            RankRun(rank, proc.returncode, logs[rank][0].read_text(), logs[rank][1].read_text(), exited[rank])
            for rank, proc in enumerate(procs)
        ]
    if hung:
        raise AssertionError(
            f"rank(s) {hung} still running after {timeout} s\n" + "\n".join(runs[r].tail() for r in hung)
        )
    return runs
