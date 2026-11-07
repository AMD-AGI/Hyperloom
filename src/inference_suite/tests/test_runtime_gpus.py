# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the pre-boot GPU check: a round's GPUs must be visible to ``amd-smi`` and hold no process, and a refusal
names each GPU and why.

A round's GPUs are ``HIP_VISIBLE_DEVICES`` numbers, which ROCm assigns in KFD node order while ``amd-smi`` indexes cards
in PCI order, so the check must translate one into the other.
"""

from __future__ import annotations

import pytest

from hyperloom_common.amd_smi import AmdSmiError
from inference_suite.runtime import gpus

_IDLE = [{"process_info": "No running processes detected"}]
#: An MI355X board's amd-smi indices in ROCm order: HIP GPU 0 is amd-smi card 3, HIP GPU 1 is amd-smi card 0, ...
_MI355X_ROCM_ORDER = [3, 0, 2, 1, 7, 4, 6, 5]


def _host(busy: dict[int, list[dict]], count: int = 8) -> list[dict]:
    """``amd-smi process`` rows, keyed by amd-smi index."""
    return [{"gpu": gpu, "process_list": busy.get(gpu, _IDLE)} for gpu in range(count)]


def _answer(monkeypatch: pytest.MonkeyPatch, payload: object, order: list[int] | None = None) -> list[tuple[str, ...]]:
    asked: list[tuple[str, ...]] = []

    def read(*args: str) -> object:
        asked.append(args)
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(gpus, "read_amd_smi_json", read)
    rocm_order = list(range(8)) if order is None else order
    monkeypatch.setattr(gpus, "gpus_in_rocm_order", lambda: list(rocm_order))
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
    assert gpus.unavailable_gpus((0, 1)) == "GPU 1 (amd-smi card 1): held by pid 4242 (python3)"


def test_a_gpu_the_host_does_not_have_is_not_visible(monkeypatch):
    _answer(monkeypatch, _host({2: [{"process_info": {"name": "vllm", "pid": 9}}]}, count=4), order=[0, 1, 2, 3])
    assert gpus.unavailable_gpus((2, 6)) == (
        "GPU 2 (amd-smi card 2): held by pid 9 (vllm); GPU 6: not visible (the host has GPUs 0-3)"
    )


def test_a_held_card_is_found_by_its_rocm_number_not_its_amd_smi_index(monkeypatch):
    # amd-smi card 0 is HIP GPU 1; amd-smi card 6 is HIP GPU 6.
    _answer(monkeypatch, _host({0: [{"name": "other", "pid": 5}], 6: [{"name": "vllm", "pid": 6}]}), _MI355X_ROCM_ORDER)
    assert gpus.unavailable_gpus((0, 2, 3)) is None
    assert gpus.unavailable_gpus((0, 1)) == "GPU 1 (amd-smi card 0): held by pid 5 (other)"
    assert gpus.unavailable_gpus((4, 5, 6, 7)) == "GPU 6 (amd-smi card 6): held by pid 6 (vllm)"


def test_a_card_amd_smi_process_omits_is_not_visible(monkeypatch):
    _answer(monkeypatch, _host({}, count=7), _MI355X_ROCM_ORDER)
    assert gpus.unavailable_gpus((4,)) == "GPU 4 (amd-smi card 7): not listed by amd-smi process"


def test_a_host_amd_smi_cannot_read_refuses_rather_than_assuming_free(monkeypatch):
    _answer(monkeypatch, AmdSmiError("amd-smi process could not run: No such file"))
    assert gpus.unavailable_gpus((0,)) == "GPUs 0 cannot be checked: amd-smi process could not run: No such file"


def test_a_host_whose_rocm_order_is_unknown_refuses(monkeypatch):
    _answer(monkeypatch, _host({}))

    def unknown() -> list[int]:
        raise AmdSmiError("amd-smi list reports a card without its KFD node, so its ROCm number is unknown")

    monkeypatch.setattr(gpus, "gpus_in_rocm_order", unknown)
    assert gpus.unavailable_gpus((0, 1)) == (
        "GPUs 0, 1 cannot be checked: amd-smi list reports a card without its KFD node, so its ROCm number is unknown"
    )
