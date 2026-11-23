# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One GPU as its driver describes it: the record, its ``amd-smi static`` reader, and what depends on the arch alone.

A session reads its card once per process start and publishes it in :data:`SESSION_CARD_ENV`; every consumer, in
this process or one that inherits its environment, reads it back with :func:`session_card`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from hyperloom_common import amd_smi
from hyperloom_common.amd_smi import AmdSmiError

__all__ = [
    "PEAK_PRECISIONS",
    "SESSION_CARD_ENV",
    "GpuCard",
    "board_label",
    "cards_of",
    "one_kind",
    "peak_rate",
    "publish_session_card",
    "read_cards",
    "session_card",
    "session_gfx",
]

#: Holds the session's card as JSON, for this process and every child that inherits its environment.
SESSION_CARD_ENV = "HYPERLOOM_GPU_CARD"

PEAK_PRECISIONS = ("fp32", "fp16", "bf16", "fp8", "fp4")

#: FP32 runs at the same rate on every arch that publishes an FP32 peak.
_FP32_RATE = 256
#: Dense matrix FLOPs one compute unit completes per engine-clock cycle, by arch and precision. A card's vendor peak
#: is its compute units x its top engine clock x this rate; an arch or a precision without an entry has no published
#: peak to reproduce. BF16 runs at the FP16 rate on every arch, so the table holds the one figure.
_PEAK_RATES: dict[str, dict[str, int]] = {
    "gfx942": {"fp32": _FP32_RATE, "fp16": 2048, "fp8": 4096},
    "gfx950": {"fp32": _FP32_RATE, "fp16": 4096, "fp8": 8192, "fp4": 16384},
    "gfx1201": {"fp32": _FP32_RATE, "fp16": 1024, "fp8": 2048},
    "gfx1100": {"fp32": _FP32_RATE, "fp16": 512},
    "gfx1151": {"fp16": 512},
}

_CU_KEYS = ("num_compute_units", "compute_units", "num_cu", "cu_count")
#: Each unit ``vram.size`` is reported in, as the power of 1024 it is of a MiB; amd-smi's ``MB`` is binary.
_UNIT_POWER_OF_MIB = {"B": -2, "KB": -1, "KIB": -1, "MB": 0, "MIB": 0, "GB": 1, "GIB": 1, "TB": 2, "TIB": 2}
#: A name amd-smi prints when libdrm does not know the card: the device id itself.
_BARE_DEVICE_ID = re.compile(r"0x[0-9a-f]+", re.IGNORECASE)

_BRAND_WORDS = frozenset({"amd", "instinct", "radeon", "pro", "ai", "graphics"})
_VARIANT_WORDS = frozenset({"hf", "vf", "oam", "dual", "slot"})


@dataclass(frozen=True)
class GpuCard:
    """One GPU as ``amd-smi static`` reports it."""

    #: Marketing name; empty when the driver has none for the card.
    name: str
    #: PCI device id, lower-case hex without ``0x``; empty when unreported.
    device_id: str
    gfx: str
    compute_units: int
    memory_mib: int
    memory_bandwidth_gbps: float | None
    #: Top level of the system clock.
    engine_clock_mhz: int | None

    def same_kind(self, other: GpuCard) -> bool:
        """Whether ``other`` computes like this card: same arch, compute units, memory and top engine clock."""
        return (self.gfx, self.compute_units, self.memory_mib, self.engine_clock_mhz) == (
            other.gfx,
            other.compute_units,
            other.memory_mib,
            other.engine_clock_mhz,
        )

    def peak_tflops(self, precision: str) -> float | None:
        """Vendor dense peak at ``precision`` in TFLOPS; ``None`` without an engine clock or a rate for the arch."""
        rate = peak_rate(self.gfx, precision)
        if rate is None or self.engine_clock_mhz is None:
            return None
        return round(self.compute_units * self.engine_clock_mhz * rate / 1e6, 1)

    def describe(self) -> str:
        """The card's name and the facts that make its kind, for a message that tells two cards apart."""
        clock = f"{self.engine_clock_mhz} MHz" if self.engine_clock_mhz is not None else "no engine clock"
        return f"{self.name or 'unnamed card'} ({self.gfx}, {self.compute_units} CUs, {self.memory_mib} MiB, {clock})"


def peak_rate(gfx: str, precision: str) -> int | None:
    """Dense matrix FLOPs per compute unit per engine-clock cycle on ``gfx`` at ``precision``; ``None`` when unknown."""
    return _PEAK_RATES.get(gfx, {}).get("fp16" if precision == "bf16" else precision)


def board_label(name: str) -> str:
    """The label a card's marketing name is filed under: its model designation, lower case; empty when it has none."""
    words = [word for word in re.split(r"[\s_]+", name.strip().lower()) if word]
    kept = "".join(word for word in words if word not in _BRAND_WORDS | _VARIANT_WORDS)
    label = re.sub(r"[^a-z0-9]", "", kept)
    # A label names files, so one that would start with a digit ("8060s") keeps its family word.
    return f"radeon{label}" if label[:1].isdigit() and "radeon" in words else label


def cards_of(document: Any) -> dict[int, GpuCard]:
    """The cards an ``amd-smi static --json`` document describes, by amd-smi index.

    Raises:
        AmdSmiError: a card reports no arch, compute-unit count or memory size.
    """
    return {row["gpu"]: _card(row) for row in amd_smi.gpu_rows(document)}


def read_cards() -> list[GpuCard]:
    """This machine's cards in ROCm order: card ``n`` is the one a visible-device mask calls ``n``.

    Raises:
        AmdSmiError: ``amd-smi`` could not be read, reports no card, or describes one incompletely.
    """
    cards = cards_of(amd_smi.read_amd_smi_json("static"))
    if not cards:
        raise AmdSmiError("amd-smi static reports no GPU")
    try:
        return [cards[index] for index in amd_smi.gpus_in_rocm_order()]
    except KeyError as exc:
        raise AmdSmiError(f"amd-smi list names card {exc.args[0]}, which amd-smi static does not describe") from exc


def one_kind(cards: Iterable[GpuCard]) -> GpuCard:
    """The card that stands for all of ``cards``.

    Raises:
        ValueError: there is no card, or they are not all the same kind; the message lists each kind.
    """
    cards = list(cards)
    if not cards:
        raise ValueError("no GPU to describe")
    kinds: list[list[GpuCard]] = []
    for card in cards:
        same = next((kind for kind in kinds if kind[0].same_kind(card)), None)
        if same is None:
            kinds.append([card])
        else:
            same.append(card)
    if len(kinds) > 1:
        described = "; ".join(f"{len(kind)}x {kind[0].describe()}" for kind in kinds)
        raise ValueError(f"the GPUs are not one kind of card: {described}")
    return cards[0]


def publish_session_card(card: GpuCard | None) -> None:
    """Make ``card`` the one :func:`session_card` answers with from here on; ``None`` publishes that there is none."""
    if card is None:
        os.environ.pop(SESSION_CARD_ENV, None)
    else:
        os.environ[SESSION_CARD_ENV] = json.dumps(asdict(card))


def session_card(environ: Mapping[str, str] = os.environ) -> GpuCard | None:
    """The card the session published; ``None`` when it read none."""
    published = environ.get(SESSION_CARD_ENV)
    return GpuCard(**json.loads(published)) if published else None


def session_gfx(environ: Mapping[str, str] = os.environ) -> str:
    """The arch of the card the session published; empty when it read none."""
    card = session_card(environ)
    return card.gfx if card else ""


def _card(row: dict[str, Any]) -> GpuCard:
    asic, vram = _block(row, "asic"), _block(row, "vram")
    gfx = _text(asic.get("target_graphics_version")).lower()
    compute_units = next((count for key in _CU_KEYS if (count := _positive_int(asic.get(key)))), None)
    memory_mib = _memory_mib(vram.get("size"))
    if not (gfx and compute_units and memory_mib):
        required = (("arch", gfx), ("compute-unit count", compute_units), ("memory size", memory_mib))
        missing = ", no ".join(what for what, value in required if not value)
        raise AmdSmiError(f"amd-smi static: card {row['gpu']} reports no {missing}")
    name = _text(asic.get("market_name"))
    return GpuCard(
        name="" if _BARE_DEVICE_ID.fullmatch(name) else name,
        device_id=_device_id(asic.get("device_id")),
        gfx=gfx,
        compute_units=compute_units,
        memory_mib=memory_mib,
        memory_bandwidth_gbps=_number(vram.get("max_bandwidth")),
        engine_clock_mhz=_engine_clock_mhz(row.get("clock")),
    )


def _block(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    block = row.get(key)
    return block if isinstance(block, dict) else {}


def _text(value: Any) -> str:
    """A string amd-smi reported, stripped; empty for its ``N/A`` and for anything that is not a string."""
    text = value.strip() if isinstance(value, str) else ""
    return "" if text.upper() == "N/A" else text


def _number(block: Any) -> float | None:
    """The number in a reading: bare, as ``{"value": n, "unit": ...}``, or either with the number in a string."""
    value = block.get("value") if isinstance(block, dict) else block
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return amd_smi.metric_value(value)


def _positive_int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 1 else None


def _device_id(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return f"{value:04x}"
    text = _text(value).lower().removeprefix("0x")
    return text if text and all(char in "0123456789abcdef" for char in text) else ""


def _memory_mib(size: Any) -> int | None:
    """``vram.size`` in MiB; a bare number is in amd-smi's ``MB``."""
    unit = _text(size.get("unit")) if isinstance(size, dict) else ""
    value, power = _number(size), _UNIT_POWER_OF_MIB.get(unit.upper() or "MB")
    return _positive_int(value * 1024.0**power) if value is not None and power is not None else None


def _engine_clock_mhz(clock: Any) -> int | None:
    levels = _block(_block(clock if isinstance(clock, dict) else {}, "sys"), "frequency_levels")
    return max((mhz for level in levels.values() if (mhz := _positive_int(level))), default=None)
