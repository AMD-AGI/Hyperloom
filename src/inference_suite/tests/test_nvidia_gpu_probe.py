# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the pre-boot GPU check on a host with no ``amd-smi``.

The check reads whichever SMI reports the host's cards: ``amd-smi`` where ROCm is installed, and ``nvidia-smi`` on a
host without it. On that second shape the cards are ``nvidia-smi``'s indices, and a card is held only when the
compute processes on it account for more than the same 1 GiB — a desktop compositor draws on the card without
holding a context a server would collide with, so it must not read as a holder. The figure ``nvidia-smi`` prints
per process is in MiB; a driver that has none -- WDDM, and the unified-memory parts -- answers ``[N/A]``, which reads
as nothing held.
"""

from __future__ import annotations

import subprocess

import pytest

from inference_suite.runtime import gpus

_GIB_MIB = (1 << 30) >> 20
#: The real tool choice, before the module fixture pins it to the NVIDIA path.
_SMI_TOOL = gpus._smi_tool


@pytest.fixture(autouse=True)
def nvidia_host(monkeypatch) -> None:
    """Describe a host whose cards are read through ``nvidia-smi``."""
    monkeypatch.setattr(gpus, "_smi_tool", lambda: "nvidia-smi")


def _serve(monkeypatch, *, cards, compute):
    """Answer the two queries the check makes from the given rows, and record which columns were asked for."""
    asked: list[str] = []

    def query(columns: str):
        asked.append(columns)
        return {"gpu=index,uuid": cards, "compute-apps=gpu_uuid,used_memory": compute}[columns]

    monkeypatch.setattr(gpus, "_nvidia_query", query)
    return asked


def test_idle_cards_pass(monkeypatch) -> None:
    asked = _serve(monkeypatch, cards=[["0", "GPU-aaa"], ["1", "GPU-bbb"]], compute=[])
    assert gpus._refusals([0, 1], set()) == []
    assert sorted(asked) == ["compute-apps=gpu_uuid,used_memory", "gpu=index,uuid"]


def test_a_card_held_by_a_compute_process_is_refused_with_its_gpu(monkeypatch) -> None:
    _serve(monkeypatch, cards=[["0", "GPU-aaa"]], compute=[["GPU-aaa", str(_GIB_MIB + 1)]])
    (refusal,) = gpus._refusals([0], set())
    assert refusal.held is True
    assert "GPU 0" in refusal.reason
    assert "nvidia-smi card 0" in refusal.reason


def test_a_card_whose_contexts_fit_under_one_gib_is_not_held(monkeypatch) -> None:
    """The same threshold the AMD path uses: a card that only had a context opened on it is free."""
    _serve(monkeypatch, cards=[["0", "GPU-aaa"]], compute=[["GPU-aaa", str(_GIB_MIB)]])
    assert gpus._refusals([0], set()) == []


def test_a_process_whose_memory_the_driver_will_not_report_does_not_hold_the_card(monkeypatch) -> None:
    """A WDDM or unified-memory host lists every graphics client and no figure for any of them."""
    _serve(monkeypatch, cards=[["0", "GPU-aaa"]], compute=[["GPU-aaa", "[N/A]"], ["GPU-aaa", "[N/A]"]])
    assert gpus._refusals([0], set()) == []


def test_the_tool_is_the_one_on_the_host_amd_smi_first(monkeypatch) -> None:
    monkeypatch.setattr(gpus, "_smi_tool", _SMI_TOOL)
    monkeypatch.setattr(gpus.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool == "amd-smi" else None)
    assert gpus._smi_tool() == "amd-smi"

    monkeypatch.setattr(gpus.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool == "nvidia-smi" else None)
    assert gpus._smi_tool() == "nvidia-smi"

    monkeypatch.setattr(gpus.shutil, "which", lambda _tool: None)
    assert gpus._smi_tool() == "amd-smi"


def test_a_process_listed_on_every_card_holds_only_the_one_it_has_memory_on(monkeypatch) -> None:
    _serve(
        monkeypatch,
        cards=[["0", "GPU-aaa"], ["1", "GPU-bbb"]],
        compute=[["GPU-bbb", str(2 * _GIB_MIB)]],
    )
    (refusal,) = gpus._refusals([0, 1], set())
    assert "GPU 1" in refusal.reason


def test_a_gpu_the_host_does_not_have_is_not_visible(monkeypatch) -> None:
    _serve(monkeypatch, cards=[["0", "GPU-aaa"]], compute=[])
    (refusal,) = gpus._refusals([1], set())
    assert refusal.held is False
    assert "not visible" in refusal.reason


def test_a_query_that_cannot_run_refuses_rather_than_assuming_free(monkeypatch) -> None:
    def unavailable(columns: str):
        raise OSError("nvidia-smi exited 9")

    monkeypatch.setattr(gpus, "_nvidia_query", unavailable)
    (refusal,) = gpus._refusals([0], set())
    assert refusal.held is False
    assert "cannot be checked" in refusal.reason


def test_a_reading_reports_each_row_as_trimmed_fields(monkeypatch) -> None:
    def run(argv, **_kwargs):
        assert argv[:2] == ["nvidia-smi", "--query-gpu=index,uuid"]
        return subprocess.CompletedProcess(argv, 0, stdout=" 0 , GPU-aaa \n\n1,GPU-bbb\n", stderr="")

    monkeypatch.setattr(gpus.subprocess, "run", run)
    assert gpus._nvidia_query("gpu=index,uuid") == [["0", "GPU-aaa"], ["1", "GPU-bbb"]]


def test_a_reading_that_fails_says_what_the_tool_said(monkeypatch) -> None:
    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 9, stdout="", stderr="No devices were found")

    monkeypatch.setattr(gpus.subprocess, "run", run)
    with pytest.raises(OSError, match="No devices were found"):
        gpus._nvidia_query("gpu=index,uuid")


def test_a_tool_that_is_not_there_says_so(monkeypatch) -> None:
    def run(_argv, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpus.subprocess, "run", run)
    with pytest.raises(OSError):
        gpus._nvidia_query("gpu=index,uuid")
