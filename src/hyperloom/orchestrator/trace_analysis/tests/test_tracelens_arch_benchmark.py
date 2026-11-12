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
        spec = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI300X",
            fallback_dir=tmp_path / "session",
            log=lambda _msg: None,
            run_command=lambda *_a, **_k: 0,
        )
    assert spec == tab.ArchSpec(bundled, tab.ArchSpecSource.BUNDLED)
    assert spec.health_warning("MI300X") is None


def test_populate_gpu_arch_json_runs_microbench_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    captured_env: dict[str, str] | None = None

    def _run_command(cmd, *, cwd, env=None):
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
        spec = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            fallback_dir=tmp_path / "session",
            log=lambda _msg: None,
            run_command=_run_command,
        )

    path = tmp_path / "TraceLens/Agent/Analysis/utils/arch/MI355X.json"
    assert spec == tab.ArchSpec(path, tab.ArchSpecSource.MEASURED)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "MI355X"
    # 0-valued MAF key is dropped so roofline never divides by zero.
    assert payload["max_achievable_tflops"] == {"matrix_bf16": 1000, "matrix_fp8": 2000}
    assert "-m" in calls[0]
    assert "TraceLens.PerfModel.benchmarking.microbench" in calls[0]
    assert "--warmup" in calls[0] and "20" in calls[0]
    assert "--rep" in calls[0] and "50" in calls[0]
    assert captured_env is not None
    assert captured_env["HIP_VISIBLE_DEVICES"] == "2"
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "2"


def test_populate_gpu_arch_json_skips_busy_gpu(tmp_path: Path) -> None:
    """A GPU the microbench's idle gate rejects (exit 2) is skipped for the next candidate."""
    tried: list[str] = []

    def _run_command(cmd, *, cwd, env=None):
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
        spec = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            fallback_dir=tmp_path / "session",
            log=lambda _msg: None,
            run_command=_run_command,
        )

    assert tried == ["4", "5"]
    assert spec.path == tmp_path / "TraceLens/Agent/Analysis/utils/arch/MI355X.json"


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
        spec = tab.internal_extension_arch_json("MI300X", log=lambda _msg: None)

    assert spec == tab.ArchSpec(bundled, tab.ArchSpecSource.BUNDLED)
    candidates.assert_not_called()


def test_populate_external_raises_on_microbench_failure(
    tmp_path: Path,
) -> None:
    """Open-source path surfaces a non-zero microbenchmark exit code when the vendor-peak fallback is also unavailable."""
    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "write_vendor_peak_arch_spec", return_value=False),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[0]),
    ):
        with pytest.raises(RuntimeError, match="exit code 7"):
            tab.populate_gpu_arch_json(
                tracelens_root=tmp_path,
                platform="MI355X",
                fallback_dir=tmp_path / "session",
                log=lambda _msg: None,
                run_command=lambda *_a, **_k: 7,
            )


def test_populate_falls_back_to_vendor_peaks_in_the_session_not_the_checkout(tmp_path: Path) -> None:
    """Every GPU busy: the vendor-peak spec lands in the session's fallback dir, never in the TraceLens checkout, and
    the session's next run reuses it rather than wait on the microbenchmark again."""
    checkout, session = tmp_path / "TraceLens", tmp_path / "session"
    runs: list[str] = []

    def _busy(cmd, *, cwd, env=None):
        runs.append(env["HIP_VISIBLE_DEVICES"])
        return 2

    def _populate() -> tab.ArchSpec:
        return tab.populate_gpu_arch_json(
            tracelens_root=checkout, platform="MI355X", fallback_dir=session, log=lambda _msg: None, run_command=_busy
        )

    with (
        patch.object(tab, "resolve_arch_json_path", return_value=None),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[0, 1]),
    ):
        first = _populate()
        second = _populate()

    assert first == second == tab.ArchSpec(session / "MI355X.json", tab.ArchSpecSource.VENDOR_PEAK)
    assert json.loads(first.path.read_text(encoding="utf-8")) == tab.build_hyperloom_arch_spec("MI355X")
    assert runs == ["0", "1"]
    assert not checkout.exists()
    warning = first.health_warning("mi355x")
    assert warning is not None and warning["code"] == "gpu_arch_json_vendor_peak" and "MI355X" in warning["message"]


def test_a_rejected_measurement_never_reaches_the_checkout(tmp_path: Path) -> None:
    """A measured spec with all-zero MAF is not published where later sessions would read it as bundled."""
    checkout = tmp_path / "TraceLens"

    def _run_command(cmd, *, cwd, env=None):
        out = Path(cmd[cmd.index("--output") + 1])
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
    ):
        spec = tab.populate_gpu_arch_json(
            tracelens_root=checkout,
            platform="MI355X",
            fallback_dir=tmp_path / "session",
            log=lambda _msg: None,
            run_command=_run_command,
        )

    assert spec.source is tab.ArchSpecSource.VENDOR_PEAK
    assert not checkout.exists()


def test_a_vendor_peak_spec_left_in_the_checkout_is_measured_over(tmp_path: Path) -> None:
    """The vendor-peak spec older versions wrote into the arch dir is not bundled: the board is measured, and the
    measurement replaces it."""
    checkout = tmp_path / "TraceLens"
    planted = tab.default_arch_output_path(checkout, "MI355X")
    planted.parent.mkdir(parents=True)
    planted.write_text(json.dumps(tab.build_hyperloom_arch_spec("MI355X"), indent=4) + "\n", encoding="utf-8")
    measured = {"name": "MI355X", "mem_bw_gbps": 6000.0, "max_achievable_tflops": {"matrix_bf16": 1500.0}}

    def _run_command(cmd, *, cwd, env=None):
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(measured), encoding="utf-8")
        return 0

    with (
        patch.object(tab, "_collect_arch_jsons", return_value={"MI355X": str(planted)}),
        patch.object(tab, "list_candidate_physical_gpus", return_value=[0]),
    ):
        assert tab.resolve_arch_json_path("MI355X") is None
        spec = tab.populate_gpu_arch_json(
            tracelens_root=checkout,
            platform="MI355X",
            fallback_dir=tmp_path / "session",
            log=lambda _msg: None,
            run_command=_run_command,
        )
        assert tab.resolve_arch_json_path("MI355X") == planted

    assert spec == tab.ArchSpec(planted, tab.ArchSpecSource.MEASURED)
    assert json.loads(planted.read_text(encoding="utf-8")) == measured


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
    tab._sanitize_measured_arch_spec(payload, platform="MI355X", out_path=tmp_path / "MI355X.json", log=lambda _m: None)
    assert payload["max_achievable_tflops"] == {"matrix_bf16": 900}


def test_sanitize_keeps_spec_with_all_positive(tmp_path: Path) -> None:
    payload = {
        "name": "MI355X",
        "mem_bw_gbps": 5000,
        "max_achievable_tflops": {"matrix_bf16": 900, "matrix_fp8": 1800},
    }
    tab._sanitize_measured_arch_spec(payload, platform="MI355X", out_path=tmp_path / "MI355X.json", log=lambda _m: None)
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
