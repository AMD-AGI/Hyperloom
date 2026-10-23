# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which card a campaign works on: pinned, kept while free, moved off when taken, waited for when all are taken."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from kernelforge.cli.options import DeviceOptions
from kernelforge.evaluation.gpu_choice import DEVICE_ENV, GpuChoice, free_card, parse_cards
from kernelforge.loop.runner import IterationLoop


def _probe(*answers):
    """rocm-smi's answers, one per look; the last repeats."""
    answers = list(answers)
    return lambda _threshold: answers.pop(0) if len(answers) > 1 else answers[0]


async def test_the_current_card_is_kept_while_free_and_left_when_taken() -> None:
    choice = GpuChoice(gpu="3", pool=("3", "5"), poll_sec=1)
    assert await free_card(choice, "3", on_wait=print, probe=_probe({"3": True, "5": True})) == "3"
    assert await free_card(choice, "3", on_wait=print, probe=_probe({"3": False, "5": True})) == "5"


async def test_a_round_waits_while_every_card_of_the_pool_is_taken(monkeypatch) -> None:
    slept, waited = [], []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("kernelforge.evaluation.gpu_choice.asyncio.sleep", sleep)
    choice = GpuChoice(gpu="0", pool=("0", "1"), poll_sec=7)
    probe = _probe({"0": False, "1": False}, {"0": False, "1": True})
    assert await free_card(choice, "0", on_wait=waited.append, probe=probe) == "1"
    assert waited == [["0", "1"]] and slept == [7]


async def test_without_a_pool_or_an_answer_the_card_stays() -> None:
    assert await free_card(GpuChoice(gpu="2"), "2", on_wait=print, probe=_probe({"2": False})) == "2"
    assert await free_card(GpuChoice(gpu="2", pool=("2", "4")), "2", on_wait=print, probe=_probe(None)) == "2"


def test_the_options_pin_the_card_every_child_sees(monkeypatch) -> None:
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    choice = DeviceOptions(gpu_card=" 6 ", gpu_pool="6, 7,", gpu_free_vram_mb=512, gpu_poll_sec=30).choice()
    assert choice == GpuChoice(gpu="6", pool=("6", "7"), free_vram_mb=512, poll_sec=30)
    choice.pin()
    assert os.environ[DEVICE_ENV] == "6"
    assert parse_cards("") == [] and parse_cards("0,1") == ["0", "1"]


async def test_the_loop_moves_off_a_taken_card_before_the_round_and_says_so(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(DEVICE_ENV, "0")
    events = []
    loop = SimpleNamespace(
        ic=SimpleNamespace(gpu=GpuChoice(gpu="0", pool=("0", "1"))),
        state_store=SimpleNamespace(append_event=events.append),
    )

    async def taken(choice, current, *, on_wait):
        return "1"

    monkeypatch.setattr("kernelforge.loop.runner.free_card", taken)
    await IterationLoop._ensure_free_gpu(loop, 4)
    assert os.environ[DEVICE_ENV] == "1"
    assert events[-1]["type"] == "gpu_switch" and events[-1]["card"] == "1" and events[-1]["previous"] == "0"
