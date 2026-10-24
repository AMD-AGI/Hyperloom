# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A directory tree that looks like the kernel's list of GPU processes and like ``/proc``, for the card-occupancy tests.

The layout is the one the rule pack's ``kfd_process_listing`` describes: a topology of node directories (a CPU node
has ``gpu_id`` 0), and a directory per process id with ``vram_<gpu id>`` files and a ``queues`` directory.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from hyperloom.common.outside_facts import INITIAL_PID_NAMESPACE

__all__ = ["KernelTree"]

_MOUNT_LINE = "36 35 0:3 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc {options}\n"


class KernelTree:
    """The kernel's list and ``/proc`` of a machine with ``gpus`` GPUs (after ``cpus`` CPU nodes) under ``root``."""

    def __init__(
        self,
        root: Path,
        *,
        gpus: int = 8,
        cpus: int = 2,
        pid_namespace: str = INITIAL_PID_NAMESPACE,
        proc_options: str = "rw",
    ) -> None:
        self.kfd = root / "kfd"
        self.procfs = root / "proc"
        self._gpus = gpus
        for node in range(cpus + gpus):
            directory = self.kfd / "topology" / "nodes" / str(node)
            directory.mkdir(parents=True)
            (directory / "gpu_id").write_text(f"{0 if node < cpus else self.gpu_id(node - cpus)}\n")
            minor = 0 if node < cpus else self.render_minor(node - cpus)
            (directory / "properties").write_text(f"cpu_cores_count 0\ndrm_render_minor {minor}\nlocation_id 8\n")
        (self.kfd / "proc").mkdir()
        (self.procfs / "self" / "ns").mkdir(parents=True)
        os.symlink(pid_namespace, self.procfs / "self" / "ns" / "pid")
        (self.procfs / "self" / "mountinfo").write_text(_MOUNT_LINE.format(options=proc_options))

    @staticmethod
    def render_minor(card: int) -> int:
        """The minor of the ``card``-th GPU's render node, as on a real machine (128, 136, ...)."""
        return 128 + 8 * card

    @staticmethod
    def gpu_id(card: int) -> int:
        """The kernel's id of the ``card``-th GPU: a number that is not the card's own, as on a real machine."""
        return 28851 + 17 * card

    def process(
        self,
        pid: int,
        *,
        parent: int = 1,
        command: Sequence[str] = ("python3", "work.py"),
        state: str = "S",
        environ: Sequence[str] = (),
    ) -> None:
        """A live process in ``/proc``."""
        directory = self.procfs / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(f"{pid} ({Path(command[0]).name}) {state} {parent} {pid} {pid} 0 -1\n")
        (directory / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
        (directory / "comm").write_text(f"{Path(command[0]).name}\n")
        (directory / "environ").write_bytes(b"\0".join(item.encode() for item in environ))

    def holds(self, pid: int, card: int, *, vram: int = 1 << 30, queues: int = 1) -> None:
        """The kernel's list says ``pid`` has ``vram`` bytes resident on ``card`` and ``queues`` queues on it."""
        directory = self.kfd / "proc" / str(pid)
        (directory / "queues").mkdir(parents=True, exist_ok=True)
        (directory / f"vram_{self.gpu_id(card)}").write_text(f"{vram}\n")
        existing = len(list((directory / "queues").iterdir()))
        for number in range(existing, existing + queues):
            (directory / "queues" / str(number)).mkdir()
            (directory / "queues" / str(number) / "gpuid").write_text(f"{self.gpu_id(card)}\n")

    def renders(self, pid: int, card: int, *, kib: int, client: str | None = None, descriptor: int = 3) -> None:
        """A process of this container has ``card``'s render node open and its fdinfo says ``kib`` KiB of video memory;
        ``client`` names the open DRM file (descriptors that share it say the same), by default one of its own."""
        directory = self.procfs / str(pid)
        (directory / "fd").mkdir(exist_ok=True)
        (directory / "fdinfo").mkdir(exist_ok=True)
        os.symlink(f"/dev/dri/renderD{self.render_minor(card)}", directory / "fd" / str(descriptor))
        text = f"pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client or f'{pid}-{descriptor}'}\ndrm-memory-vram:\t{kib} KiB\n"
        (directory / "fdinfo" / str(descriptor)).write_text(text)

    def opened_only(self, pid: int) -> None:
        """``pid`` opened the interface on every GPU (as a monitor does): an entry with no memory and no queue."""
        directory = self.kfd / "proc" / str(pid)
        (directory / "queues").mkdir(parents=True, exist_ok=True)
        for card in range(self._gpus):
            (directory / f"vram_{self.gpu_id(card)}").write_text("0\n")

    def exit(self, pid: int) -> None:
        """The process ends: it leaves ``/proc`` at once and the kernel's list a moment later (it is left here)."""
        for path in sorted((self.procfs / str(pid)).iterdir()):
            path.unlink()
        (self.procfs / str(pid)).rmdir()
