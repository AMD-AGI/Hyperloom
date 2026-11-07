# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the pre-boot GPU check: a round's GPUs must be visible to ``amd-smi`` and hold no process, and a refusal
names each GPU and why."""

from __future__ import annotations

import pytest

from hyperloom_common.amd_smi import AmdSmiError
from inference_suite.runtime import gpus

_IDLE = [{"process_info": "No running processes detected"}]


def _host(busy: dict[int, list[dict]], count: int = 8) -> list[dict]:
    return [{"gpu": gpu, "process_list": busy.get(gpu, _IDLE)} for gpu in range(count)]


def _answer(monkeypatch: pytest.MonkeyPatch, payload: object) -> list[tuple[str, ...]]:
    asked: list[tuple[str, ...]] = []

    def read(*args: str) -> object:
        asked.append(args)
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(gpus, "read_amd_smi_json", read)
    return asked


def test_visible_idle_gpus_pass(monkeypatch):
    asked = _answer(monkeypatch, _host({5: [{"process_info": {"name": "python3", "pid": 77}}]}))
    assert gpus.unavailable_gpus((0, 1, 2, 3)) is None
    assert asked == [("process",)]


@pytest.mark.parametrize(
    "entry",
    [{"process_info": {"name": "python3", "pid": 4242}}, {"name": "python3", "pid": 4242}],
    ids=["nested", "flat"],
)
def test_a_gpu_another_process_holds_is_named_with_its_holder(monkeypatch, entry):
    _answer(monkeypatch, {"gpu_data": _host({1: [entry]})})
    assert gpus.unavailable_gpus((0, 1)) == "GPU 1: held by pid 4242 (python3)"


def test_a_gpu_the_host_does_not_list_is_not_visible(monkeypatch):
    _answer(monkeypatch, _host({2: [{"process_info": {"name": "vllm", "pid": 9}}]}, count=4))
    assert gpus.unavailable_gpus((2, 6)) == (
        "GPU 2: held by pid 9 (vllm); GPU 6: not visible (the host lists GPUs 0, 1, 2, 3)"
    )


def test_a_host_amd_smi_cannot_read_refuses_rather_than_assuming_free(monkeypatch):
    _answer(monkeypatch, AmdSmiError("amd-smi process could not run: No such file"))
    assert gpus.unavailable_gpus((0,)) == "GPUs 0 cannot be checked: amd-smi process could not run: No such file"
