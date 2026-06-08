"""Unit tests for TraceLens arch JSON pre-report benchmarking."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "tracelens_arch_benchmark",
    _TOOLS_DIR / "tracelens_arch_benchmark.py",
)
tab = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tab)

_VISIBLE_DEVICE_VARS = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


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


def test_list_candidate_physical_gpus_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,5")
    assert tab.list_candidate_physical_gpus() == [2, 5]


def test_single_physical_gpu_env_pins_subprocess_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    env = tab.single_physical_gpu_env(4)

    assert env["HIP_VISIBLE_DEVICES"] == "4"
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["ROCR_VISIBLE_DEVICES"] == "4"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2"


def test_select_idle_gpu_picks_first_free_from_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    def _idle(logical_idx: int, *, util_threshold: int = 5):
        return logical_idx == 1, f"logical={logical_idx}"

    monkeypatch.setattr(tab, "check_gpu_idle", _idle)

    selected = tab.select_idle_gpu()
    assert selected == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2"


def test_select_idle_gpu_reuses_single_idle_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    monkeypatch.setattr(
        tab, "check_gpu_idle", lambda *_a, **_k: (True, "idle")
    )

    selected = tab.select_idle_gpu()
    assert selected == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_select_idle_gpu_raises_when_all_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    monkeypatch.setattr(
        tab, "check_gpu_idle", lambda *_a, **_k: (False, "busy")
    )

    with pytest.raises(RuntimeError, match="no unoccupied GPU"):
        tab.select_idle_gpu()


def test_populate_gpu_arch_json_uses_bundled_spec(tmp_path: Path) -> None:
    bundled = tmp_path / "MI300X.json"
    bundled.write_text("{}", encoding="utf-8")
    with patch.object(tab, "resolve_arch_json_path", return_value=bundled):
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI300X",
            log=lambda _msg: None,
            run_command=lambda *_a, **_k: 0,
        )
    assert path == bundled


def test_populate_gpu_arch_json_runs_microbench_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    captured_env: dict[str, str] | None = None

    def _run_command(cmd, *, cwd, timeout_s, env=None):
        nonlocal captured_env
        captured_env = env
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
        path = tab.populate_gpu_arch_json(
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
    assert captured_env is not None
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "2"
