# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GPUs of the kernel's compute topology, numbered the way ``ROCR_VISIBLE_DEVICES`` numbers them, the PCI address
of each and the video memory in use on it.

The topology (the rule pack's ``kfd_process_listing``) has a node per device. A GPU node's ``gpu_id`` is the id the
kernel's process list names that GPU by, and the ROCm runtime numbers the GPU nodes it can read in node order: that
number is a card's here, in ``ROCR_VISIBLE_DEVICES`` and in the card pool. A container is refused (permission denied)
the ``gpu_id`` and ``properties`` of every GPU node it is not given, and the runtime leaves those nodes out when it
numbers the cards; so does this reader. A view that is refused every GPU node, or a card it cannot read, is not one
whose cards are free: its error is ``denied``.

A card's PCI address is the one its node's properties state, which is how a tool that numbers cards by PCI address
(rocm-smi, amd-smi) is matched to it. A card's memory is read from the driver's counters (the rule pack's
``amdgpu_vram_counters``) in the device directory of the render node its node of the topology names. The DRM class
directory lists every GPU of the machine, other containers' as well, and neither its order nor rocm-smi's is the
runtime's, so a card is never found by a position.

:data:`KFD_PATH` and :data:`DRM_PATH` are read when a reader is called, not when it is defined, so a test points every
reader of the machine at a tree of its own at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from hyperloom.common.outside_facts import (
    DRM_CLASS_ROOT,
    DRM_DEVICE_DIR,
    DRM_RENDER_NODE_PREFIX,
    DRM_VRAM_TOTAL_FILE,
    DRM_VRAM_USED_FILE,
    KFD_DOMAIN_KEY,
    KFD_LOCATION_ID_KEY,
    KFD_NODE_GPU_ID,
    KFD_NODE_PROPERTIES,
    KFD_NODES,
    KFD_RENDER_MINOR_KEY,
    KFD_ROOT,
)

__all__ = [
    "DRM_PATH",
    "KFD_PATH",
    "VRAM_BUSY_FRACTION",
    "GpuNode",
    "GpuVram",
    "KfdUnreadable",
    "card_pci_addresses",
    "card_vram",
    "list_names",
    "read_number",
    "render_minor",
    "select_gpus",
    "vram_busy",
]

KFD_PATH = Path(KFD_ROOT)
DRM_PATH = Path(DRM_CLASS_ROOT)
#: Share of a card's VRAM above which something is resident on it.
VRAM_BUSY_FRACTION = 0.01
_MIB = 1 << 20
#: What reading an entry that is gone raises (a process's, which went with it).
_GONE = (FileNotFoundError, ProcessLookupError, NotADirectoryError)
_PERMISSION_DENIED = "permission denied"


class KfdUnreadable(Exception):
    """The kernel's files cannot be read as the reader needs them; the message says why, and ``denied`` whether it is
    because this reader's view refuses them (permission denied) rather than because they are missing or malformed."""

    def __init__(self, message: str, *, denied: bool = False) -> None:
        super().__init__(message)
        self.denied = denied


class GpuNode(NamedTuple):
    """A GPU of the kernel's topology: its node number and the id the kernel's process list names it by."""

    node: int
    gpu_id: int


class _Topology(NamedTuple):
    """The GPUs of the topology this reader can read, in the order the runtime numbers them, and how many GPU nodes its
    view refuses."""

    gpus: list[GpuNode]
    denied: int


class GpuVram(NamedTuple):
    """Per-GPU VRAM usage in MiB."""

    used_mib: float
    total_mib: float


def vram_busy(gpu: GpuVram) -> bool:
    """Whether something holds VRAM on ``gpu``."""
    return gpu.used_mib > gpu.total_mib * VRAM_BUSY_FRACTION


def read_number(path: Path) -> int | None:
    """The number in a file of the kernel's; ``None`` when the entry went away while it was read.

    Raises:
        KfdUnreadable: When this view is refused the file (``denied``), or it does not hold a number.
    """
    try:
        return int(path.read_text(encoding="ascii").strip())
    except _GONE:
        return None
    except PermissionError as exc:
        raise KfdUnreadable(f"{path} cannot be read: {_PERMISSION_DENIED}", denied=True) from exc
    except (OSError, ValueError) as exc:
        raise KfdUnreadable(f"{path} does not hold a number ({type(exc).__name__})") from exc


def list_names(directory: Path) -> list[str]:
    """The names in a directory of the kernel's; none when it went away.

    Raises:
        KfdUnreadable: When this view is refused the directory (``denied``).
    """
    try:
        return [entry.name for entry in directory.iterdir()]
    except _GONE:
        return []
    except PermissionError as exc:
        raise KfdUnreadable(f"{directory} cannot be listed: {_PERMISSION_DENIED}", denied=True) from exc


def _gpu_nodes(kfd: Path) -> _Topology:
    """The GPUs of the topology, in the order the runtime numbers them: the nodes that are not a CPU, by node number,
    leaving out those this reader's view refuses (another container's GPUs), as the runtime does.

    Raises:
        KfdUnreadable: When the topology cannot be listed, a node goes away, or every GPU node is refused (``denied``).
    """
    try:
        nodes = sorted(int(entry.name) for entry in (kfd / KFD_NODES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise KfdUnreadable(f"the GPU topology of {kfd} cannot be listed ({type(exc).__name__})") from exc
    found = []
    denied = 0
    for node in nodes:
        try:
            gpu_id = read_number(kfd / KFD_NODES / str(node) / KFD_NODE_GPU_ID)
        except KfdUnreadable as exc:
            if not exc.denied:
                raise
            denied += 1
            continue
        if gpu_id is None:
            raise KfdUnreadable(f"node {node} of the GPU topology went away while it was read")
        if gpu_id:
            found.append(GpuNode(node, gpu_id))
    if denied and not found:
        raise KfdUnreadable(
            f"every one of the {denied} GPU nodes of {kfd} is refused to this view: {_PERMISSION_DENIED}", denied=True
        )
    return _Topology(found, denied)


def select_gpus(kfd: Path, cards: Sequence[int]) -> dict[int, GpuNode]:
    """The GPU of the topology under ``kfd`` each of ``cards`` is.

    Raises:
        KfdUnreadable: When the topology cannot be read (see :func:`_gpu_nodes`), or a card is not among the GPUs the
            kernel lists: a card that is not there is not a free one. It is ``denied`` when this view is refused GPU
            nodes, one of which may be that card.
    """
    gpus, denied = _gpu_nodes(kfd)
    missing = [card for card in cards if not 0 <= card < len(gpus)]
    if missing and denied:
        raise KfdUnreadable(
            f"card(s) {missing} are not among the {len(gpus)} GPUs this view can read ({denied} more are refused to it: "
            f"{_PERMISSION_DENIED})",
            denied=True,
        )
    if missing:
        raise KfdUnreadable(f"card(s) {missing} are not among the {len(gpus)} GPUs the kernel lists")
    return {card: gpus[card] for card in cards}


def _properties(kfd: Path, node: int) -> dict[str, str]:
    """The properties of a GPU's topology node, by key; none when they cannot be read."""
    try:
        text = (kfd / KFD_NODES / str(node) / KFD_NODE_PROPERTIES).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return {key: value.strip() for key, _, value in (line.partition(" ") for line in text.splitlines())}


def render_minor(kfd: Path, node: int) -> int | None:
    """The minor of the render node of a GPU's topology node, or ``None`` when its properties do not say."""
    value = _properties(kfd, node).get(KFD_RENDER_MINOR_KEY, "")
    return int(value) if value.isdigit() and int(value) > 0 else None


def card_pci_addresses(cards: Sequence[int]) -> dict[int, str]:
    """The PCI address of each of ``cards`` (``0000:f5:00.0``), as the domain and location_id of its node state it.

    Raises:
        KfdUnreadable: When a card is not one this view can read (see :func:`select_gpus`), or its node states no PCI
            address.
    """
    kfd = KFD_PATH
    return {card: _pci_address(kfd, gpu) for card, gpu in select_gpus(kfd, cards).items()}


def _pci_address(kfd: Path, gpu: GpuNode) -> str:
    properties = _properties(kfd, gpu.node)
    domain, location = properties.get(KFD_DOMAIN_KEY, ""), properties.get(KFD_LOCATION_ID_KEY, "")
    if not (domain.isdigit() and location.isdigit()):
        raise KfdUnreadable(f"node {gpu.node} of the GPU topology of {kfd} states no PCI address")
    bus, devfn = divmod(int(location), 256)
    return f"{int(domain):04x}:{bus:02x}:{devfn >> 3:02x}.{devfn & 7:x}"


def card_vram(cards: Sequence[int] | None = None) -> dict[int, GpuVram]:
    """The video memory of each of ``cards``, or of every GPU this view can read when ``None``, by card.

    Raises:
        KfdUnreadable: When a card is not one this view can read (see :func:`select_gpus`), its node names no render
            node, or the counters of that render node are not there or state no memory.
    """
    kfd = KFD_PATH
    gpus = dict(enumerate(_gpu_nodes(kfd).gpus)) if cards is None else select_gpus(kfd, cards)
    return {card: _vram(kfd, gpu) for card, gpu in gpus.items()}


def _vram(kfd: Path, gpu: GpuNode) -> GpuVram:
    minor = render_minor(kfd, gpu.node)
    if minor is None:
        raise KfdUnreadable(f"node {gpu.node} of the GPU topology of {kfd} names no render node")
    device = DRM_PATH / f"{DRM_RENDER_NODE_PREFIX}{minor}" / DRM_DEVICE_DIR
    used, total = read_number(device / DRM_VRAM_USED_FILE), read_number(device / DRM_VRAM_TOTAL_FILE)
    if used is None or total is None or total <= 0:
        raise KfdUnreadable(f"{device} states no video memory (used {used}, total {total})")
    return GpuVram(used / _MIB, total / _MIB)
