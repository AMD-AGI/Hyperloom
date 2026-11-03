"""Unit tests for TraceLens arch JSON pre-report benchmarking."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS as _VISIBLE_DEVICE_VARS
from hyperloom.orchestrator.trace_analysis import tracelens_arch_benchmark as tab


@pytest.fixture(scope="module")
def tla():
    """Import the TraceLens-checkout module without executing ``main()``."""
    from hyperloom.orchestrator.trace_analysis import _tracelens_checkout

    return _tracelens_checkout


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


def test_list_candidate_physical_gpus_reads_legacy_rocr_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HSA_VISIBLE_DEVICES", "4,5")
    assert tab.list_candidate_physical_gpus() == [4, 5]


def test_list_candidate_physical_gpus_applies_hip_mask_within_rocr(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,0")
    assert tab.list_candidate_physical_gpus() == [6, 4]


def test_list_candidate_physical_gpus_uuid_mask_is_logical(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-a1b2,GPU-c3d4")
    assert tab.list_candidate_physical_gpus() == [0, 1]


def test_single_physical_gpu_env_pins_subprocess_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("HSA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "0,1")

    env = tab.single_physical_gpu_env(5)

    assert env["HIP_VISIBLE_DEVICES"] == "5"
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    for var in ("ROCR_VISIBLE_DEVICES", "HSA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        assert var not in env
    assert os.environ.get("ROCR_VISIBLE_DEVICES") == "4,5"


def test_single_physical_gpu_env_matches_tracelens_idle_gate() -> None:
    """The microbench child's own idle gate must probe the GPU the child runs on."""
    from TraceLens.PerfModel.benchmarking.microbench_utils import resolve_physical_device

    env = tab.single_physical_gpu_env(6, base_env={"ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "0,1,2,3"})
    with patch.dict(os.environ, env, clear=True):
        assert resolve_physical_device(0)[0] == 6


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
            json.dumps(
                {
                    "name": "MI300X",
                    "mem_bw_gbps": 5000,
                    "memory_gb": 192,
                    "max_achievable_tflops": {
                        "matrix_bf16": 1000,
                        "matrix_fp8": 2000,
                        "matrix_fp4": 0,  # unmeasured -> dropped by sanitizer
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[2]),
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
    # 0-valued MAF key is dropped so roofline never divides by zero.
    assert payload["max_achievable_tflops"] == {"matrix_bf16": 1000, "matrix_fp8": 2000}
    assert "-m" in calls[0]
    assert "TraceLens.PerfModel.benchmarking.microbench" in calls[0]
    assert "--warmup" in calls[0] and "20" in calls[0]
    assert "--rep" in calls[0] and "50" in calls[0]
    assert calls[0][calls[0].index("--output") + 1].endswith("TraceLens/Agent/Analysis/utils/arch/MI355X.json")
    assert captured_env is not None
    assert captured_env["HIP_VISIBLE_DEVICES"] == "2"
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "2"


def test_populate_gpu_arch_json_skips_busy_gpu(tmp_path: Path) -> None:
    """A GPU the microbench's idle gate rejects (exit 2) is skipped for the next candidate."""
    tried: list[str] = []

    def _run_command(cmd, *, cwd, timeout_s, env=None):
        tried.append(env["HIP_VISIBLE_DEVICES"])
        if env["HIP_VISIBLE_DEVICES"] == "4":
            return 2
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"name": "MI355X", "mem_bw_gbps": 8000, "max_achievable_tflops": {"matrix_bf16": 1000}}),
            encoding="utf-8",
        )
        return 0

    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[4, 5]),
    ):
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path, platform="MI355X", log=lambda _msg: None, run_command=_run_command
        )

    assert tried == ["4", "5"]
    assert path == tmp_path / "TraceLens/Agent/Analysis/utils/arch/MI355X.json"


def test_populate_skips_microbench_when_internal_extension_enabled(
    tmp_path: Path,
) -> None:
    """Internal extension backfills MAF, so no microbenchmark runs even when the bundled arch spec is missing."""
    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus") as candidates,
    ):
        path = tab.internal_extension_arch_json("MI355X", log=lambda _msg: None)

    assert path is None
    candidates.assert_not_called()


def test_populate_internal_extension_returns_bundled_spec_without_benchmark(
    tmp_path: Path,
) -> None:
    """When the internal extension is enabled and a bundled spec exists, it is returned as an artifact without running the microbenchmark."""
    bundled = tmp_path / "MI300X.json"
    bundled.write_text("{}", encoding="utf-8")

    with (
        patch.object(tab, "resolve_arch_json_path", return_value=bundled),
        patch.object(tab, "list_candidate_physical_gpus") as candidates,
    ):
        path = tab.internal_extension_arch_json("MI300X", log=lambda _msg: None)

    assert path == bundled
    candidates.assert_not_called()


def test_populate_external_raises_on_microbench_failure(
    tmp_path: Path,
) -> None:
    """Open-source path surfaces a non-zero microbenchmark exit code when the hyperloom fallback is also unavailable."""
    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "write_hyperloom_arch_spec", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[0]),
    ):
        with pytest.raises(RuntimeError, match="exit code 7"):
            tab.populate_gpu_arch_json(
                tracelens_root=tmp_path,
                platform="MI355X",
                log=lambda _msg: None,
                run_command=lambda *_a, **_k: 7,
            )


def test_populate_falls_back_to_hyperloom_when_microbench_unavailable(
    tmp_path: Path,
) -> None:
    """Every GPU busy -> the hyperloom achievable spec is used instead of raising."""
    fallback = tmp_path / "MI355X.json"

    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[0, 1]),
        patch.object(tab, "write_hyperloom_arch_spec", return_value=fallback),
    ):
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            log=lambda _msg: None,
            run_command=lambda *_a, **_k: 2,
        )

    assert path == fallback


def test_populate_falls_back_to_hyperloom_on_zeroed_measured_spec(
    tmp_path: Path,
) -> None:
    """A measured spec with all-zero MAF falls back to the hyperloom spec."""
    fallback = tmp_path / "MI355X.json"

    def _run_command(cmd, *, cwd, timeout_s, env=None):
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "name": "MI355X",
                    "mem_bw_gbps": 8000,
                    "max_achievable_tflops": {"matrix_bf16": 0, "matrix_fp8": 0},
                }
            ),
            encoding="utf-8",
        )
        return 0

    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[0]),
        patch.object(tab, "write_hyperloom_arch_spec", return_value=fallback),
    ):
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            log=lambda _msg: None,
            run_command=_run_command,
        )

    assert path == fallback


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 600),  # unset -> floor
        ("", 600),  # empty -> floor
        ("   ", 600),  # whitespace -> floor
        ("abc", 600),  # non-numeric -> floor (no ValueError)
        ("12.5", 600),  # float string -> floor (int() would raise)
        ("0", 600),  # below floor -> clamped up
        ("-5", 600),  # negative -> clamped up to floor
        ("300", 600),  # under floor -> floor
        ("600", 600),  # exactly floor
        ("1800", 1800),  # valid override
        ("  900  ", 900),  # surrounding whitespace tolerated
    ],
)
def test_resolve_arch_benchmark_timeout_s(tla, monkeypatch: pytest.MonkeyPatch, raw, expected) -> None:
    """Malformed TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC falls back to the 600s floor instead of raising ValueError before the microbenchmark."""
    if raw is None:
        monkeypatch.delenv(tla.ARCH_BENCHMARK_TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(tla.ARCH_BENCHMARK_TIMEOUT_ENV, raw)
    assert tla._resolve_arch_benchmark_timeout_s() == expected


def test_sanitize_drops_zero_maf_keys(tmp_path: Path) -> None:
    """0-valued MAF entries are dropped so roofline skips (not divides by) them."""
    payload = {
        "name": "MI355X",
        "mem_bw_gbps": 5000,
        "max_achievable_tflops": {"matrix_bf16": 900, "matrix_fp8": 0, "matrix_int8": 0.0},
    }
    changed = tab._sanitize_measured_arch_spec(
        payload, platform="MI355X", out_path=tmp_path / "MI355X.json", log=lambda _m: None
    )
    assert changed is True
    assert payload["max_achievable_tflops"] == {"matrix_bf16": 900}


def test_sanitize_keeps_spec_with_all_positive(tmp_path: Path) -> None:
    payload = {
        "name": "MI355X",
        "mem_bw_gbps": 5000,
        "max_achievable_tflops": {"matrix_bf16": 900, "matrix_fp8": 1800},
    }
    changed = tab._sanitize_measured_arch_spec(
        payload, platform="MI355X", out_path=tmp_path / "MI355X.json", log=lambda _m: None
    )
    assert changed is False
    assert payload["max_achievable_tflops"] == {"matrix_bf16": 900, "matrix_fp8": 1800}


def test_sanitize_raises_when_all_maf_zero(tmp_path: Path) -> None:
    payload = {
        "name": "MI355X",
        "mem_bw_gbps": 5000,
        "max_achievable_tflops": {"matrix_bf16": 0, "matrix_fp8": 0},
    }
    with pytest.raises(RuntimeError, match="no positive max_achievable_tflops"):
        tab._sanitize_measured_arch_spec(
            payload,
            platform="MI355X",
            out_path=tmp_path / "MI355X.json",
            log=lambda _m: None,
        )


def test_sanitize_raises_when_maf_missing(tmp_path: Path) -> None:
    payload = {"name": "MI355X", "mem_bw_gbps": 5000}
    with pytest.raises(RuntimeError, match="no max_achievable_tflops"):
        tab._sanitize_measured_arch_spec(
            payload,
            platform="MI355X",
            out_path=tmp_path / "MI355X.json",
            log=lambda _m: None,
        )


@pytest.mark.parametrize("mem_bw", [0, 0.0, -1, None, "abc"])
def test_sanitize_raises_on_non_positive_bandwidth(tmp_path: Path, mem_bw) -> None:
    payload = {
        "name": "MI355X",
        "mem_bw_gbps": mem_bw,
        "max_achievable_tflops": {"matrix_bf16": 900},
    }
    with pytest.raises(RuntimeError, match="non-positive mem_bw_gbps"):
        tab._sanitize_measured_arch_spec(
            payload,
            platform="MI355X",
            out_path=tmp_path / "MI355X.json",
            log=lambda _m: None,
        )
