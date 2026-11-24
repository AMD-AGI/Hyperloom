# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One GPU as its machine describes it, the card a session publishes for its consumers, and per-arch peak rates."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess  # nosec B404 - fixed probe commands.
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from hyperloom_common import amd_smi
from hyperloom_common.amd_smi import AmdSmiError

__all__ = [
    "PEAK_PRECISIONS",
    "PROBES",
    "SESSION_CARD_ENV",
    "GpuCard",
    "board_label",
    "cards_of",
    "one_kind",
    "peak_rate",
    "probed_cards",
    "publish_session_card",
    "read_cards",
    "rocr_uuids",
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

#: Unmasked, so torch numbers the cards as ROCm does.
_TORCH_PROBE = (
    "import json, os; "
    "[os.environ.pop(v, None) for v in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')]; "
    "import torch; "
    "facts = ('name', 'gcnArchName', 'total_memory'); "
    "cards = map(torch.cuda.get_device_properties, range(torch.cuda.device_count())); "
    "print(json.dumps({'torch': [{f: getattr(c, f, None) for f in facts} for c in cards]}))"
)
#: The commands a machine is asked for its cards with. Only amd-smi reports compute units, bandwidth and clock: torch
#: counts an RDNA card's compute units by half.
PROBES = (
    "amd-smi static --json",
    "rocm-smi --showproductname --showmeminfo vram --json",
    f"python3 -c {shlex.quote(_TORCH_PROBE)}",
)
_PROBE_TIMEOUT_SEC = 120.0

#: amd-smi prints the device id as the name of a card libdrm does not know.
_BARE_DEVICE_ID = re.compile(r"0x[0-9a-f]+", re.IGNORECASE)
_BRAND_WORDS = frozenset({"amd", "instinct", "radeon", "pro", "ai", "graphics"})
_VARIANT_WORDS = frozenset({"hf", "vf", "oam", "dual", "slot"})


@dataclass(frozen=True)
class GpuCard:
    """One GPU as its machine reports it; a fact is ``None`` when the probe that read it does not report it."""

    #: Empty when the driver has no marketing name for the card.
    name: str
    #: Lower-case hex without ``0x``.
    device_id: str
    gfx: str
    compute_units: int | None
    memory_mib: int
    memory_bandwidth_gbps: float | None
    #: Top level of the system clock.
    engine_clock_mhz: int | None

    def same_kind(self, other: GpuCard) -> bool:
        """Whether ``other`` has this card's arch, compute units and memory."""
        return (self.gfx, self.compute_units, self.memory_mib) == (other.gfx, other.compute_units, other.memory_mib)

    def peak_tflops(self, precision: str) -> float | None:
        """Vendor dense peak in TFLOPS; ``None`` without compute units, an engine clock or a rate for the arch."""
        rate = peak_rate(self.gfx, precision)
        if rate is None or self.compute_units is None or self.engine_clock_mhz is None:
            return None
        return round(self.compute_units * self.engine_clock_mhz * rate / 1e6, 1)

    def describe(self) -> str:
        """The card's name and kind, for a message."""
        facts = ((self.compute_units, "CUs"), (self.memory_mib, "MiB"), (self.engine_clock_mhz, "MHz"))
        known = ", ".join(f"{value} {unit}" for value, unit in facts if value is not None)
        return f"{self.name or 'unnamed card'} ({self.gfx}, {known})"


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


def probed_cards(output: str) -> list[GpuCard]:
    """The cards in the first JSON document of ``output`` that describes any: what one of :data:`PROBES` printed,
    possibly among other lines. rocm-smi's and torch's are in ROCm order, amd-smi's in its own.

    Raises:
        AmdSmiError: no document describes a card, or one describes a card without its arch or memory size.
    """
    decoder = json.JSONDecoder()
    for start in re.finditer(r"[\[{]", output):
        try:
            document, _ = decoder.raw_decode(output, start.start())
        except ValueError:
            continue
        cards = _document_cards(document)
        if cards:
            return cards
    raise AmdSmiError(f"no GPU is described in {output.strip()[-300:]!r}")


def read_cards() -> list[GpuCard]:
    """This machine's cards, indexed by ROCm device number, from the first of :data:`PROBES` that describes any.

    Raises:
        AmdSmiError: no probe describes a card; the message gives each one's reason.
    """
    reasons = []
    try:
        cards = cards_of(amd_smi.read_amd_smi_json("static"))
        if cards:
            return [cards[index] for index in amd_smi.gpus_in_rocm_order()]
        reasons.append("amd-smi static reports no GPU")
    except AmdSmiError as exc:
        reasons.append(str(exc))
    for probe in PROBES[1:]:
        try:
            return probed_cards(_probe_output(probe))
        except AmdSmiError as exc:
            reasons.append(str(exc))
    raise AmdSmiError("; ".join(reasons))


def rocr_uuids() -> list[str]:
    """What a UUID device mask calls each card (``gpu-<serial>``, lower case), by ROCm device number.

    Raises:
        AmdSmiError: ``amd-smi`` cannot be read.
    """
    rows = {row["gpu"]: _block(row, "asic") for row in amd_smi.gpu_rows(amd_smi.read_amd_smi_json("static"))}
    serials = (_text(rows[index].get("asic_serial")) for index in amd_smi.gpus_in_rocm_order())
    return ["gpu-" + serial.lower().removeprefix("0x") for serial in serials]


def one_kind(cards: Iterable[GpuCard]) -> GpuCard:
    """The card that stands for all of ``cards``, a named one when any is; raises ``ValueError`` listing the kinds
    unless there is exactly one."""
    cards = list(cards)
    kinds = [card for index, card in enumerate(cards) if not any(card.same_kind(seen) for seen in cards[:index])]
    if len(kinds) != 1:
        counted = "; ".join(f"{sum(map(kind.same_kind, cards))}x {kind.describe()}" for kind in kinds)
        raise ValueError(f"the GPUs are not one kind of card: {counted or 'there is none'}")
    return next((card for card in cards if board_label(card.name)), cards[0])


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


def _probe_output(probe: str) -> str:
    tool = probe.split()[0]
    try:
        done = subprocess.run(  # nosec B603 - one of PROBES.
            shlex.split(probe), capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SEC, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AmdSmiError(f"{tool} could not run: {exc}") from exc
    if done.returncode != 0:
        said = done.stderr.strip().splitlines()[-1:]
        raise AmdSmiError(f"{tool} exited {done.returncode}" + (f": {said[0][:200]}" if said else ""))
    return done.stdout


def _document_cards(document: Any) -> list[GpuCard]:
    if isinstance(document, dict) and "torch" in document:
        return [_torch_card(facts) for facts in document["torch"]]
    if isinstance(document, dict) and "gpu_data" not in document:
        rows = [row for key, row in document.items() if re.fullmatch(r"card\d+", key) and isinstance(row, dict)]
        return [_rocm_smi_card(row) for row in sorted(rows, key=lambda row: int(row.get("Node ID") or 0))]
    return list(cards_of(document).values())


def _rocm_smi_card(row: dict[str, Any]) -> GpuCard:
    gfx, memory_bytes = _text(row.get("GFX Version")).lower(), _positive_int(row.get("VRAM Total Memory (B)"))
    if not (gfx and memory_bytes):
        raise AmdSmiError("rocm-smi reports a card without its arch or memory size")
    return GpuCard(
        name=_text(row.get("Card Series")),
        device_id=_text(row.get("Card Model")).lower().removeprefix("0x"),
        gfx=gfx,
        compute_units=None,
        memory_mib=memory_bytes >> 20,
        memory_bandwidth_gbps=None,
        engine_clock_mhz=None,
    )


def _torch_card(facts: dict[str, Any]) -> GpuCard:
    gfx, memory_bytes = _text(facts.get("gcnArchName")).lower().split(":")[0], _positive_int(facts.get("total_memory"))
    if not (gfx and memory_bytes):
        raise AmdSmiError("torch reports a card without its arch or memory size")
    return GpuCard(
        name=_text(facts.get("name")),
        device_id="",
        gfx=gfx,
        compute_units=None,
        memory_mib=memory_bytes >> 20,
        memory_bandwidth_gbps=None,
        engine_clock_mhz=None,
    )


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
    """The top level of the system clock. A lower level is the live reading, which can exceed it under load."""
    levels = _block(_block(clock if isinstance(clock, dict) else {}, "sys"), "frequency_levels")
    return _positive_int(list(levels.values())[-1]) if levels else None
