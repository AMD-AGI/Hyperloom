# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the pre-boot GPU check: a round's GPUs must be visible to ``amd-smi`` and no process may have memory
resident on them beyond the 1 GiB a process that merely opened the card may keep there, and a refusal names each GPU
and why.

A round's GPUs are ``ROCR_VISIBLE_DEVICES`` numbers, which ROCm assigns in KFD node order while ``amd-smi`` indexes
cards in PCI order, so the check must translate one into the other. A holder is a process with more than 1 GiB in the
kernel's per-process, per-GPU figure (``/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>``) for that GPU; a container's
refused GPU nodes are left out and its GPUs numbered among the readable ones. amd-smi's figure, with its rule that an
entry reporting no memory is a holder, decides only when that list cannot be read, and the reason is logged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hyperloom_common.amd_smi import AmdSmiError
from inference_suite.contracts.config import Deadlines
from inference_suite.contracts.report import FailedStep
from inference_suite.runtime import gpus
from inference_suite.runtime.watch import RoundGate, Stop

_IDLE = [{"process_info": "No running processes detected"}]
#: An MI355X board's amd-smi indices in ROCm order: HIP GPU 0 is amd-smi card 3, HIP GPU 1 is amd-smi card 0, ...
_MI355X_ROCM_ORDER = [3, 0, 2, 1, 7, 4, 6, 5]


class _Clock:
    """Stands in for ``time`` in :mod:`inference_suite.runtime.gpus`: sleeping advances it at once."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(gpus, "time", fake)
    return fake


@pytest.fixture(autouse=True)
def no_kfd(monkeypatch, tmp_path) -> Path:
    """A host whose kernel list is not there, so amd-smi's figure decides unless a test builds one."""
    monkeypatch.setattr(gpus, "_KFD_ROOT", tmp_path / "no-kfd")
    return tmp_path / "no-kfd"


def _gate(cancel=lambda: "") -> RoundGate:
    return RoundGate(Deadlines(boot_sec=600, round_sec=3600, eval_sec=600, silence_sec=600), cancel, 0.0)


def _refused(round_gpus: tuple[int, ...]) -> str | None:
    """The detail of the GPU refusal :func:`gpus.await_free_gpus` ends with, or ``None`` once the GPUs are free."""
    stop = gpus.await_free_gpus(round_gpus, _gate())
    if stop is None:
        return None
    assert stop.step is FailedStep.GPUS_UNAVAILABLE
    return stop.detail


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
    assert _refused((0, 1, 2, 3)) is None
    assert asked == [("process",)]


@pytest.mark.parametrize(
    "entry",
    [{"process_info": {"name": "python3", "pid": 4242}}, {"name": "python3", "pid": 4242}],
    ids=["nested", "flat"],
)
def test_a_gpu_another_process_holds_is_named_with_its_holder(monkeypatch, entry):
    _answer(monkeypatch, {"gpu_data": _host({1: [entry]})})
    assert _refused((0, 1)) == "GPU 1 (amd-smi card 1): held by pid 4242 (python3)"


def _opened(pid: int, vram: int) -> dict:
    """An ``amd-smi process --json`` entry of ``pid`` with ``vram`` bytes resident on the card."""
    memory = {key: {"value": 0, "unit": "B"} for key in ("gtt_mem", "cpu_mem")}
    memory["vram_mem"] = {"value": vram, "unit": "B"}
    return {"process_info": {"name": "python3", "pid": pid, "memory_usage": memory}}


def test_a_process_listed_on_every_card_holds_only_the_one_it_has_memory_on(monkeypatch):
    # A HIP process with no device mask opens every card: amd-smi lists it on all of them, with memory on one.
    held = 7  # amd-smi card 7 is HIP GPU 4 on an MI355X board
    _answer(
        monkeypatch,
        _host({card: [_opened(4242, 4 << 30 if card == held else 0)] for card in range(8)}),
        _MI355X_ROCM_ORDER,
    )

    assert _refused((3, 5, 6, 7)) is None
    assert _refused((4,)) == "GPU 4 (amd-smi card 7): held by pid 4242 (python3)"


def test_a_process_that_only_opened_a_card_does_not_hold_it(monkeypatch):
    # Recorded per-card drm-memory-vram: an orchestrator masked to card 3 and a scheduler masked to cards 4-7 keep
    # 36-52 KiB on every card, and a server masked to card 5 keeps 1.7 MiB on the cards it does not use.
    kib = 1 << 10
    _answer(
        monkeypatch,
        _host(
            {
                card: [
                    _opened(1822180, 52 * kib if card >= 4 else 48 * kib),
                    _opened(2608305, 52 * kib if card == 3 else 36 * kib),
                    _opened(2768962, 247_247_288 * kib if card == 5 else 1724 * kib),
                ]
                for card in range(8)
            }
        ),
    )

    assert _refused((4, 6, 7)) is None
    assert _refused((5,)) == "GPU 5 (amd-smi card 5): held by pid 2768962 (python3)"


@pytest.mark.parametrize(("vram", "held"), [(1 << 30, False), ((1 << 30) + 1, True)])
def test_a_card_is_held_only_above_one_gib_resident(monkeypatch, vram, held):
    # Up to 1 GiB leaves room for a runtime that maps larger staging buffers on a card it does not serve on.
    _answer(monkeypatch, _host({1: [_opened(4242, vram)]}))
    expected = "GPU 1 (amd-smi card 1): held by pid 4242 (python3)" if held else None
    assert _refused((1,)) == expected


def test_a_gpu_the_host_does_not_have_is_not_visible(monkeypatch):
    _answer(monkeypatch, _host({2: [{"process_info": {"name": "vllm", "pid": 9}}]}, count=4), order=[0, 1, 2, 3])
    assert _refused((2, 6)) == (
        "GPU 2 (amd-smi card 2): held by pid 9 (vllm); GPU 6: not visible (the host has GPUs 0-3)"
    )


def test_a_held_card_is_found_by_its_rocm_number_not_its_amd_smi_index(monkeypatch):
    # amd-smi card 0 is HIP GPU 1; amd-smi card 6 is HIP GPU 6.
    _answer(monkeypatch, _host({0: [{"name": "other", "pid": 5}], 6: [{"name": "vllm", "pid": 6}]}), _MI355X_ROCM_ORDER)
    assert _refused((0, 2, 3)) is None
    assert _refused((0, 1)) == "GPU 1 (amd-smi card 0): held by pid 5 (other)"
    assert _refused((4, 5, 6, 7)) == "GPU 6 (amd-smi card 6): held by pid 6 (vllm)"


def test_a_card_amd_smi_process_omits_is_not_visible(monkeypatch):
    _answer(monkeypatch, _host({}, count=7), _MI355X_ROCM_ORDER)
    assert _refused((4,)) == "GPU 4 (amd-smi card 7): not listed by amd-smi process"


def test_a_host_amd_smi_cannot_read_refuses_rather_than_assuming_free(monkeypatch):
    _answer(monkeypatch, AmdSmiError("amd-smi process could not run: No such file"))
    assert _refused((0,)) == "GPUs 0 cannot be checked: amd-smi process could not run: No such file"


def test_a_host_whose_rocm_order_is_unknown_refuses(monkeypatch):
    _answer(monkeypatch, _host({}))

    def unknown() -> list[int]:
        raise AmdSmiError("amd-smi list reports a card without its KFD node, so its ROCm number is unknown")

    monkeypatch.setattr(gpus, "gpus_in_rocm_order", unknown)
    assert _refused((0, 1)) == (
        "GPUs 0, 1 cannot be checked: amd-smi list reports a card without its KFD node, so its ROCm number is unknown"
    )


_HELD = {1: [_opened(4242, 8 << 30)]}
_HOLDER = "GPU 1 (amd-smi card 1): held by pid 4242 (python3)"


def _readings(monkeypatch: pytest.MonkeyPatch, payloads: list[object]) -> list[float]:
    """Answer ``amd-smi process`` with ``payloads`` in turn, the last one for good; return when each was read."""
    read_at: list[float] = []
    monkeypatch.setattr(gpus, "gpus_in_rocm_order", lambda: list(range(8)))

    def read(*_args: str) -> object:
        read_at.append(gpus.time.monotonic())
        return payloads[min(len(read_at), len(payloads)) - 1]

    monkeypatch.setattr(gpus, "read_amd_smi_json", read)
    return read_at


def test_a_holder_that_exits_within_the_release_window_is_waited_out(monkeypatch, clock):
    # A torn-down server's ranks stay <defunct> with memory listed for ~25 s after the round that ran them ended.
    read_at = _readings(monkeypatch, [_host(_HELD), _host(_HELD), _host({})])

    assert gpus.await_free_gpus((0, 1), _gate()) is None
    assert read_at == [0.0, gpus.RELEASE_STEP_SEC, 2 * gpus.RELEASE_STEP_SEC]


def test_a_holder_that_outlasts_the_release_window_is_refused_with_its_name(monkeypatch, clock):
    read_at = _readings(monkeypatch, [_host(_HELD)])

    stop = gpus.await_free_gpus((0, 1), _gate())

    assert stop == Stop(FailedStep.GPUS_UNAVAILABLE, _HOLDER)
    assert read_at[-1] == pytest.approx(gpus.RELEASE_WAIT_SEC)
    assert clock.now == pytest.approx(gpus.RELEASE_WAIT_SEC)


@pytest.mark.parametrize(
    ("payload", "cards", "detail"),
    [
        (_host(_HELD, count=4), 4, f"{_HOLDER}; GPU 6: not visible (the host has GPUs 0-3)"),
        (_host(_HELD, count=6), 8, f"{_HOLDER}; GPU 6 (amd-smi card 6): not listed by amd-smi process"),
        (AmdSmiError("amd-smi process timed out"), 8, "GPUs 1, 6 cannot be checked: amd-smi process timed out"),
    ],
    ids=["not-visible", "not-listed", "amd-smi-error"],
)
def test_a_gpu_no_wait_can_free_is_refused_at_once(monkeypatch, clock, payload, cards, detail):
    _answer(monkeypatch, payload, list(range(cards)))

    assert gpus.await_free_gpus((1, 6), _gate()) == Stop(FailedStep.GPUS_UNAVAILABLE, detail)
    assert clock.slept == 0.0


def test_a_cancel_during_the_wait_stops_it_at_the_next_poll(monkeypatch, clock):
    _readings(monkeypatch, [_host(_HELD)])
    cancel_at = gpus.RELEASE_STEP_SEC + 1.0

    stop = gpus.await_free_gpus((1,), _gate(lambda: "received SIGTERM" if clock.now >= cancel_at else ""))

    assert stop == Stop(FailedStep.CANCELLED, "received SIGTERM")
    assert clock.now - cancel_at <= gpus.POLL_SEC


_GIB = 1 << 30
#: The kernel's id of each GPU, in ROCm order (the GPU topology nodes 2-9 after the CPU nodes 0 and 1).
_GPU_IDS = [41_000 + gpu for gpu in range(8)]
#: A DeepSeek V4.1 vLLM server at TP4 on ROCm GPUs 0-3 (recorded on MI355X, 2026-11-12): each worker has 264 GiB
#: resident on its own GPU and none on GPUs 4-7, while amd-smi process lists every worker on all eight cards with the
#: same figure on each: TP0's 283,944,206,336 B, and 2 MiB for TP1-3.
_TP_WORKERS = {283_952 + rank: rank for rank in range(4)}
_TP0_VRAM_BYTES = 283_944_206_336


def _kfd(root: Path, resident: dict[int, dict[int, int]], *, gpu_ids: list[int] = _GPU_IDS) -> Path:
    """A fake ``/sys/class/kfd/kfd``: two CPU nodes then a node per GPU, and ``resident[pid][rocm gpu]`` bytes."""
    for node, gpu_id in enumerate([0, 0, *gpu_ids]):
        (root / "topology" / "nodes" / str(node)).mkdir(parents=True)
        (root / "topology" / "nodes" / str(node) / "gpu_id").write_text(f"{gpu_id}\n", encoding="ascii")
    (root / "proc").mkdir(parents=True)
    for pid, per_gpu in resident.items():
        (root / "proc" / str(pid)).mkdir()
        for gpu, vram in per_gpu.items():
            if gpu >= len(gpu_ids):  # a GPU this topology leaves out has no file
                continue
            (root / "proc" / str(pid) / f"vram_{gpu_ids[gpu]}").write_text(f"{vram}\n", encoding="ascii")
    return root


def _vllm_tp4_on_gpus_0_to_3(monkeypatch) -> None:
    """amd-smi's view of :data:`_TP_WORKERS`: every worker on every card, with TP0's figure on all of them."""
    rows = {
        card: [_opened(pid, _TP0_VRAM_BYTES if rank == 0 else 2 << 20) for pid, rank in _TP_WORKERS.items()]
        for card in range(8)
    }
    _answer(monkeypatch, _host(rows), _MI355X_ROCM_ORDER)


def _kernel_view_of_tp4() -> dict[int, dict[int, int]]:
    return {pid: {gpu: 264 * _GIB if gpu == rank else 0 for gpu in range(8)} for pid, rank in _TP_WORKERS.items()}


def test_the_kernels_per_gpu_figure_frees_the_gpus_a_multi_card_server_only_opened(monkeypatch, no_kfd):
    _vllm_tp4_on_gpus_0_to_3(monkeypatch)
    _kfd(no_kfd, _kernel_view_of_tp4())

    assert _refused((4, 5, 6, 7)) is None
    assert _refused((0, 1)) == (
        "GPU 0 (amd-smi card 3): held by pid 283952 (python3); GPU 1 (amd-smi card 0): held by pid 283953 (python3)"
    )


def test_a_process_the_kernel_lists_but_amd_smi_does_not_is_still_a_holder(monkeypatch, no_kfd):
    _answer(monkeypatch, _host({}), _MI355X_ROCM_ORDER)
    _kfd(no_kfd, {5150: {6: 8 * _GIB}, 5151: {6: _GIB}})

    assert _refused((6,)) == "GPU 6 (amd-smi card 6): held by pid 5150 (unnamed)"


def test_an_amd_smi_entry_without_a_memory_figure_holds_nothing_while_the_kernels_list_is_read(monkeypatch, no_kfd):
    # amd-smi's no-figure rule is part of its fallback only: the kernel lists nothing resident on GPU 1.
    _answer(monkeypatch, _host({1: [{"process_info": {"name": "python3", "pid": 4242}}]}))
    _kfd(no_kfd, {})

    assert _refused((1,)) is None


def test_without_the_kernels_list_amd_smis_figure_decides(monkeypatch):
    _vllm_tp4_on_gpus_0_to_3(monkeypatch)

    # TP0's figure on every card refuses GPUs the server never used: the reason the kernel's figure comes first.
    assert _refused((4,)) == "GPU 4 (amd-smi card 7): held by pid 283952 (python3)"


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("no-interface", "no AMD GPU driver interface in this view"),
        ("refused-topology", "the topology"),
        ("refused-processes", "the process list"),
        ("every-node-refused", "every one of the 8 GPU nodes"),
        ("refused-vram", f"vram_{_GPU_IDS[5]} is refused to this view"),
        ("refused-node", "this view reads 7 GPUs in the kernel's topology and amd-smi lists 8"),
        ("fewer-gpus", "this view reads 7 GPUs in the kernel's topology and amd-smi lists 8"),
    ],
)
def test_a_kernel_list_this_view_cannot_fully_read_falls_back_to_amd_smi_and_says_why(
    monkeypatch, no_kfd, caplog, damage, reason
):
    _vllm_tp4_on_gpus_0_to_3(monkeypatch)
    if damage == "fewer-gpus":
        _kfd(no_kfd, _kernel_view_of_tp4(), gpu_ids=_GPU_IDS[:7])
    elif damage != "no-interface":
        _kfd(no_kfd, _kernel_view_of_tp4())
    if damage == "refused-topology":
        _refuse_listing(monkeypatch, no_kfd / "topology" / "nodes")
    if damage == "refused-processes":
        _refuse_listing(monkeypatch, no_kfd / "proc")
    if damage == "every-node-refused":
        _refuse(monkeypatch, *(no_kfd / "topology" / "nodes" / str(node) / "gpu_id" for node in range(2, 10)))
    if damage == "refused-vram":
        _refuse(monkeypatch, no_kfd / "proc" / "283953" / f"vram_{_GPU_IDS[5]}")
    if damage == "refused-node":
        _refuse(monkeypatch, no_kfd / "topology" / "nodes" / "9" / "gpu_id")

    with caplog.at_level(logging.WARNING, logger=gpus.__name__):
        assert _refused((5,)) == "GPU 5 (amd-smi card 4): held by pid 283952 (python3)"

    # One line per check, however many times the held GPU is read again within the release window.
    [line] = [record.getMessage() for record in caplog.records if record.name == gpus.__name__]
    assert line.startswith("GPUs 5: the kernel's per-GPU memory list cannot be read (")
    assert reason in line
    assert line.endswith("amd-smi's figure decides who holds them")


def _refuse(monkeypatch, *refused: Path) -> None:
    """Make reading ``refused`` fail as the kernel fails it for a file this view's device cgroup does not grant."""
    read_text = Path.read_text

    def read(path: Path, *args, **kwargs) -> str:
        if path in refused:
            raise PermissionError(1, "Operation not permitted", str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)


def _refuse_listing(monkeypatch, refused: Path) -> None:
    """Make listing the directory ``refused`` fail with permission denied."""
    iterdir = Path.iterdir

    def listing(path: Path):
        if path == refused:
            raise PermissionError(13, "Permission denied", str(path))
        return iterdir(path)

    monkeypatch.setattr(Path, "iterdir", listing)


def test_a_container_numbers_its_gpus_among_the_nodes_it_is_not_refused(monkeypatch, no_kfd, caplog):
    # A container shown the host's GPUs 4-7 is refused the gpu_id of the four other GPU nodes; ROCm numbers its GPUs
    # 0-3, and amd-smi lists four cards. Its own TP2 server on its GPUs 0 and 1 shows on every card with rank 0's
    # figure, so amd-smi alone would refuse its free GPUs 2 and 3.
    server = {7001: 0, 7002: 1}
    _answer(
        monkeypatch,
        _host({card: [_opened(pid, _TP0_VRAM_BYTES) for pid in server] for card in range(4)}, count=4),
        [0, 1, 2, 3],
    )
    _kfd(no_kfd, {pid: {4 + gpu: 264 * _GIB} for pid, gpu in server.items()})
    _refuse(monkeypatch, *(no_kfd / "topology" / "nodes" / str(node) / "gpu_id" for node in range(2, 6)))

    with caplog.at_level(logging.WARNING, logger=gpus.__name__):
        assert _refused((2, 3)) is None
        assert _refused((0, 1)) == (
            "GPU 0 (amd-smi card 0): held by pid 7001 (python3); GPU 1 (amd-smi card 1): held by pid 7002 (python3)"
        )
    assert not caplog.records


def test_topology_nodes_are_taken_in_numeric_order(monkeypatch, no_kfd):
    # Ten GPUs on nodes 2-11: GPU 8 is node 10, which a lexical sort would put before node 2.
    gpu_ids = [52_000 + gpu for gpu in range(10)]
    _answer(monkeypatch, _host({}, count=10), list(range(10)))
    _kfd(no_kfd, {5150: {8: 8 * _GIB}}, gpu_ids=gpu_ids)

    assert _refused((0, 1, 7, 9)) is None
    assert _refused((8,)) == "GPU 8 (amd-smi card 8): held by pid 5150 (unnamed)"


def test_a_process_with_no_vram_file_for_a_gpu_holds_nothing_there(monkeypatch, no_kfd):
    # pid 6002 has a file for GPU 3 only: on GPU 2 only pid 6001 is a holder, and nothing raises for the missing file.
    _answer(monkeypatch, _host({}))
    _kfd(no_kfd, {6001: {2: 8 * _GIB}, 6002: {3: 8 * _GIB}})

    assert _refused((2,)) == "GPU 2 (amd-smi card 2): held by pid 6001 (unnamed)"


def test_a_process_that_ended_before_its_files_were_read_holds_nothing(monkeypatch, no_kfd):
    _answer(monkeypatch, _host({}))
    _kfd(no_kfd, {})
    (no_kfd / "proc" / "777").mkdir()  # listed, but its vram files were gone by the time they were read

    assert _refused((0, 1, 2)) is None
