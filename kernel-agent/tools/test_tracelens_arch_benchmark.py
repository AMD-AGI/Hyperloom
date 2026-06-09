"""Unit tests for TraceLens arch JSON pre-report benchmarking.

These cover both the TraceLens-internal (extension-enabled) and external
(open-source-only) paths so the MAF backfill gate is validated for both
deployments (#364), plus the microbenchmark mechanics themselves (#390).
"""

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


@pytest.fixture(scope="module")
def tla():
    """Import tracelens_analysis.py as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "tracelens_analysis_under_test", _TOOLS_DIR / "tracelens_analysis.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

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
            internal_extension_enabled=False,
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

    with patch.object(tab, "resolve_arch_json_path", return_value=None), patch.object(
        tab,
        "select_idle_gpu",
        return_value=2,
    ):
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            internal_extension_enabled=False,
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
    assert calls[0][calls[0].index("--output") + 1].endswith(
        "TraceLens/Agent/Analysis/utils/arch/MI355X.json"
    )
    assert captured_env is not None
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "2"


def test_populate_skips_microbench_when_internal_extension_enabled(
    tmp_path: Path,
) -> None:
    """Internal extension backfills MAF, so no microbenchmark runs even when
    the bundled arch spec is missing (#390 gate)."""
    calls: list[list[str]] = []

    def _run_command(cmd, *, cwd, timeout_s, env=None):
        calls.append(cmd)
        return 0

    with patch.object(tab, "resolve_arch_json_path", return_value=None), patch.object(
        tab, "select_idle_gpu"
    ) as select_idle:
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI355X",
            internal_extension_enabled=True,
            log=lambda _msg: None,
            run_command=_run_command,
        )

    assert path is None
    assert calls == []
    select_idle.assert_not_called()


def test_populate_internal_extension_returns_bundled_spec_without_benchmark(
    tmp_path: Path,
) -> None:
    """When the internal extension is enabled and a bundled spec exists, it is
    returned as an artifact without running the microbenchmark."""
    bundled = tmp_path / "MI300X.json"
    bundled.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def _run_command(cmd, *, cwd, timeout_s, env=None):
        calls.append(cmd)
        return 0

    with patch.object(tab, "resolve_arch_json_path", return_value=bundled):
        path = tab.populate_gpu_arch_json(
            tracelens_root=tmp_path,
            platform="MI300X",
            internal_extension_enabled=True,
            log=lambda _msg: None,
            run_command=_run_command,
        )

    assert path == bundled
    assert calls == []


def test_populate_external_raises_on_microbench_failure(
    tmp_path: Path,
) -> None:
    """Open-source path surfaces a non-zero microbenchmark exit code."""
    with patch.object(tab, "resolve_arch_json_path", return_value=None), patch.object(
        tab, "select_idle_gpu", return_value=0
    ):
        with pytest.raises(RuntimeError, match="exit code 7"):
            tab.populate_gpu_arch_json(
                tracelens_root=tmp_path,
                platform="MI355X",
                internal_extension_enabled=False,
                log=lambda _msg: None,
                run_command=lambda *_a, **_k: 7,
            )


def _install_fake_tracelens(
    monkeypatch: pytest.MonkeyPatch, leaf: str, attr: str, value
) -> None:
    """Inject a fake ``TraceLens.…`` package tree so a lazy ``from … import``
    succeeds without the real TraceLens being present."""
    import sys
    import types

    parts = leaf.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    setattr(sys.modules[leaf], attr, value)


def test_get_check_gpu_idle_resolves_after_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A None cache (TraceLens not importable at process start) is re-resolved
    once TraceLens is installed by tracelens_analysis.main() (#390)."""
    monkeypatch.setattr(tab, "check_gpu_idle", None)
    sentinel = lambda *_a, **_k: (True, "idle")  # noqa: E731
    _install_fake_tracelens(
        monkeypatch,
        "TraceLens.PerfModel.benchmarking.microbench_utils",
        "check_gpu_idle",
        sentinel,
    )
    assert tab._get_check_gpu_idle() is sentinel
    assert tab.check_gpu_idle is sentinel  # cached on first success


def test_get_collect_arch_jsons_resolves_after_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same lazy re-resolution for the arch-JSON collector (#390)."""
    monkeypatch.setattr(tab, "_collect_arch_jsons", None)
    sentinel = lambda: {"MI355X": "/tmp/MI355X.json"}  # noqa: E731
    _install_fake_tracelens(
        monkeypatch,
        "TraceLens.Agent.Analysis.utils.arch_utils",
        "_collect_arch_jsons",
        sentinel,
    )
    assert tab._get_collect_arch_jsons() is sentinel
    assert tab._collect_arch_jsons is sentinel


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 600),       # unset -> floor
        ("", 600),         # empty -> floor
        ("   ", 600),      # whitespace -> floor
        ("abc", 600),      # non-numeric -> floor (no ValueError)
        ("12.5", 600),     # float string -> floor (int() would raise)
        ("0", 600),        # below floor -> clamped up
        ("-5", 600),       # negative -> clamped up to floor
        ("300", 600),      # under floor -> floor
        ("600", 600),      # exactly floor
        ("1800", 1800),    # valid override
        ("  900  ", 900),  # surrounding whitespace tolerated
    ],
)
def test_resolve_arch_benchmark_timeout_s(
    tla, monkeypatch: pytest.MonkeyPatch, raw, expected
) -> None:
    """Malformed TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC falls back to the 600s
    floor instead of raising ValueError before the microbenchmark (#390)."""
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
            payload, platform="MI355X", out_path=tmp_path / "MI355X.json",
            log=lambda _m: None,
        )


def test_sanitize_raises_when_maf_missing(tmp_path: Path) -> None:
    payload = {"name": "MI355X", "mem_bw_gbps": 5000}
    with pytest.raises(RuntimeError, match="no max_achievable_tflops"):
        tab._sanitize_measured_arch_spec(
            payload, platform="MI355X", out_path=tmp_path / "MI355X.json",
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
            payload, platform="MI355X", out_path=tmp_path / "MI355X.json",
            log=lambda _m: None,
        )
