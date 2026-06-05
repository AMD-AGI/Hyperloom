"""Unit tests for TraceLens arch JSON pre-report benchmarking."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tracelens_arch_benchmark as tab  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MI355X", "MI355X"),
        ("mi355x", "MI355X"),
        ("  mi300x  ", "MI300X"),
        ("", ""),
    ],
)
def test_normalize_platform(raw: str, expected: str) -> None:
    assert tab.normalize_platform(raw) == expected


@pytest.mark.parametrize(
    ("env", "device_count", "expected"),
    [
        ({"HIP_VISIBLE_DEVICES": "3"}, None, 1),
        ({"CUDA_VISIBLE_DEVICES": "0,1"}, None, 2),
        ({}, 4, 4),
        ({}, 0, 0),
    ],
)
def test_count_visible_gpus(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    device_count: int | None,
    expected: int,
) -> None:
    for var in tab.VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with patch.object(tab, "list_candidate_physical_gpus") as mock_list:
        if env.get("HIP_VISIBLE_DEVICES") == "3":
            mock_list.return_value = [3]
        elif env.get("CUDA_VISIBLE_DEVICES") == "0,1":
            mock_list.return_value = [0, 1]
        elif device_count == 4:
            mock_list.return_value = [0, 1, 2, 3]
        else:
            mock_list.return_value = []
        assert tab.count_visible_gpus() == expected


def test_list_candidate_physical_gpus_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in tab.VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,5")
    assert tab.list_candidate_physical_gpus() == [2, 5]


def test_pin_single_physical_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in tab.VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    tab.pin_single_physical_gpu(4)
    assert os.environ["HIP_VISIBLE_DEVICES"] == "4"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
    assert os.environ["ROCR_VISIBLE_DEVICES"] == "4"


def test_select_idle_gpu_picks_first_free_from_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in tab.VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    def _idle(logical_idx: int, *, util_threshold: int = 5):
        return logical_idx == 1, f"logical={logical_idx}"

    fake_mod = type(
        "microbench_utils",
        (),
        {"check_gpu_idle": staticmethod(_idle)},
    )
    monkeypatch.setitem(
        sys.modules,
        "TraceLens.PerfModel.benchmarking.microbench_utils",
        fake_mod,
    )

    selected = tab.select_idle_gpu()
    assert selected == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_select_idle_gpu_reuses_single_idle_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in tab.VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    fake_mod = type(
        "microbench_utils",
        (),
        {"check_gpu_idle": staticmethod(lambda *_a, **_k: (True, "idle"))},
    )
    monkeypatch.setitem(
        sys.modules,
        "TraceLens.PerfModel.benchmarking.microbench_utils",
        fake_mod,
    )

    selected = tab.select_idle_gpu()
    assert selected == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_select_idle_gpu_raises_when_all_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in tab.VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    fake_mod = type(
        "microbench_utils",
        (),
        {"check_gpu_idle": staticmethod(lambda *_a, **_k: (False, "busy"))},
    )
    monkeypatch.setitem(
        sys.modules,
        "TraceLens.PerfModel.benchmarking.microbench_utils",
        fake_mod,
    )

    with pytest.raises(RuntimeError, match="no unoccupied GPU"):
        tab.select_idle_gpu()


def test_ensure_gpu_arch_json_uses_bundled_spec(tmp_path: Path) -> None:
    bundled = tmp_path / "MI300X.json"
    bundled.write_text("{}", encoding="utf-8")
    with patch.object(tab, "resolve_arch_json_path", return_value=bundled):
        path = tab.ensure_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI300X",
            log=lambda _msg: None,
            run_command=lambda *_a, **_k: 0,
        )
    assert path == bundled


def test_ensure_gpu_arch_json_runs_microbench_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _run_command(cmd, *, cwd, timeout_s):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"name": "MI300X", "mem_bw_gbps": 5000, "memory_gb": 192}),
            encoding="utf-8",
        )
        return 0

    with patch.object(tab, "resolve_arch_json_path", return_value=None), patch.object(
        tab,
        "select_idle_gpu",
        return_value=2,
    ):
        path = tab.ensure_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            log=lambda _msg: None,
            run_command=_run_command,
            timeout_s=60,
        )

    assert path == tmp_path / "TraceLens/Agent/Analysis/utils/arch/MI355X.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "MI355X"
    assert "-m" in calls[0]
    assert "TraceLens.PerfModel.benchmarking.microbench" in calls[0]
    assert "--warmup" in calls[0] and "20" in calls[0]
    assert "--rep" in calls[0] and "50" in calls[0]
    assert "--allow-busy" in calls[0]
    assert calls[0][calls[0].index("--output") + 1].endswith(
        "TraceLens/Agent/Analysis/utils/arch/MI355X.json"
    )
