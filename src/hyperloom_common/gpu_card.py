# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One GPU as ``amd-smi static`` describes it, the card a session publishes for its consumers, and per-arch peak rates."""

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

#: The session's card as JSON; children of the session inherit it.
SESSION_CARD_ENV = "HYPERLOOM_GPU_CARD"

PEAK_PRECISIONS = ("fp32", "fp16", "bf16", "fp8", "fp4")

_FP32_RATE = 256
#: Dense matrix FLOPs per compute unit per engine-clock cycle. BF16 runs at the FP16 rate.
_PEAK_RATES: dict[str, dict[str, int]] = {
    "gfx942": {"fp32": _FP32_RATE, "fp16": 2048, "fp8": 4096},
    "gfx950": {"fp32": _FP32_RATE, "fp16": 4096, "fp8": 8192, "fp4": 16384},
    "gfx1201": {"fp32": _FP32_RATE, "fp16": 1024, "fp8": 2048},
    "gfx1100": {"fp32": _FP32_RATE, "fp16": 512},
    "gfx1151": {"fp16": 512},
}

#: amd-smi prints the device id as the name of a card libdrm does not know.
_BARE_DEVICE_ID = re.compile(r"0x[0-9a-f]+", re.IGNORECASE)
_BRAND_WORDS = frozenset({"amd", "instinct", "radeon", "pro", "ai", "graphics"})
_VARIANT_WORDS = frozenset({"hf", "vf", "oam", "dual", "slot"})


@dataclass(frozen=True)
class GpuCard:
    """One GPU as ``amd-smi static`` reports it."""

    #: Empty when the driver has no marketing name for the card.
    name: str
    #: Lower-case hex without ``0x``.
    device_id: str
    gfx: str
    compute_units: int
    memory_mib: int
    memory_bandwidth_gbps: float | None
    #: Top level of the system clock.
    engine_clock_mhz: int | None

    def same_kind(self, other: GpuCard) -> bool:
        """Whether ``other`` has this card's arch, compute units, memory and top engine clock."""
        return (self.gfx, self.compute_units, self.memory_mib, self.engine_clock_mhz) == (
            other.gfx,
            other.compute_units,
            other.memory_mib,
            other.engine_clock_mhz,
        )

    def peak_tflops(self, precision: str) -> float | None:
        """Vendor dense peak in TFLOPS; ``None`` without an engine clock or a rate for the arch."""
        rate = peak_rate(self.gfx, precision)
        if rate is None or self.engine_clock_mhz is None:
            return None
        return round(self.compute_units * self.engine_clock_mhz * rate / 1e6, 1)

    def describe(self) -> str:
        """The card's name and kind, for a message."""
        clock = f"{self.engine_clock_mhz} MHz" if self.engine_clock_mhz is not None else "no engine clock"
        return f"{self.name or 'unnamed card'} ({self.gfx}, {self.compute_units} CUs, {self.memory_mib} MiB, {clock})"


def peak_rate(gfx: str, precision: str) -> int | None:
    """Dense matrix FLOPs per compute unit per engine-clock cycle; ``None`` when unknown."""
    return _PEAK_RATES.get(gfx, {}).get("fp16" if precision == "bf16" else precision)


def board_label(name: str) -> str:
    """The model designation in a marketing name, lower case: ``AMD Instinct MI355X`` is ``mi355x``."""
    words = [word for word in re.split(r"[\s_]+", name.strip().lower()) if word]
    kept = "".join(word for word in words if word not in _BRAND_WORDS | _VARIANT_WORDS)
    label = re.sub(r"[^a-z0-9]", "", kept)
    # A label names files, so one starting with a digit ("8060s") keeps its family word.
    return f"radeon{label}" if label[:1].isdigit() and "radeon" in words else label


def cards_of(document: Any) -> dict[int, GpuCard]:
    """The cards of an ``amd-smi static --json`` document, by amd-smi index.

    Raises:
        AmdSmiError: a card reports no arch, compute-unit count or memory size.
    """
    return {row["gpu"]: _card(row) for row in amd_smi.gpu_rows(document)}


def read_cards() -> list[GpuCard]:
    """This machine's cards, indexed by ROCm device number; raises :class:`AmdSmiError` when there is none to read."""
    cards = cards_of(amd_smi.read_amd_smi_json("static"))
    if not cards:
        raise AmdSmiError("amd-smi static reports no GPU")
    return [cards[index] for index in amd_smi.gpus_in_rocm_order()]


def one_kind(cards: Iterable[GpuCard]) -> GpuCard:
    """The card that stands for all of ``cards``; raises ``ValueError`` listing the kinds unless there is exactly one."""
    cards = list(cards)
    kinds = [card for index, card in enumerate(cards) if not any(card.same_kind(seen) for seen in cards[:index])]
    if len(kinds) != 1:
        counted = "; ".join(f"{sum(map(kind.same_kind, cards))}x {kind.describe()}" for kind in kinds)
        raise ValueError(f"the GPUs are not one kind of card: {counted or 'there is none'}")
    return kinds[0]


def publish_session_card(card: GpuCard | None) -> None:
    """Make ``card``, or no card, what :func:`session_card` answers with."""
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
    compute_units = _positive_int(asic.get("num_compute_units"))
    memory_mib = _positive_int(vram.get("size"))  # amd-smi's "MB" is binary
    if not (gfx and compute_units and memory_mib):
        required = (("arch", gfx), ("compute-unit count", compute_units), ("memory size", memory_mib))
        missing = ", no ".join(what for what, value in required if not value)
        raise AmdSmiError(f"amd-smi static: card {row['gpu']} reports no {missing}")
    name = _text(asic.get("market_name"))
    return GpuCard(
        name="" if _BARE_DEVICE_ID.fullmatch(name) else name,
        device_id=_text(asic.get("device_id")).lower().removeprefix("0x"),
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
    """A string amd-smi reported; empty for ``N/A`` and for anything else."""
    text = value.strip() if isinstance(value, str) else ""
    return "" if text.upper() == "N/A" else text


def _number(block: Any) -> float | None:
    """A reading's number: bare or ``{"value": n, "unit": ...}``, as a number or a numeric string."""
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


def _engine_clock_mhz(clock: Any) -> int | None:
    levels = _block(_block(clock if isinstance(clock, dict) else {}, "sys"), "frequency_levels")
    return max((mhz for level in levels.values() if (mhz := _positive_int(level))), default=None)
