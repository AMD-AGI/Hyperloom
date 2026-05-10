"""P2-3 tests: backends / params / sweep executors + grid runner shared logic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors import (
    DEFAULT_BACKENDS_GRID,
    DEFAULT_NCCL_GRID,
    DEFAULT_PARAMS_GRID,
    DEFAULT_CONC_VALUES,
    DEFAULT_ISL_OSL,
    BackendsExecutor,
    ParamsExecutor,
    SweepExecutor,
    discover_backend_flags,
)
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
    _build_variant_yaml,
    pick_winners,
    run_grid,
)
from inference_optimizer.orchestrator.action_executors.baseline import (
    _materialize_config_with_envs,
)
from inference_optimizer.orchestrator.action_executors.params import (
    DEFAULT_VLLM_PARAMS_GRID,
)
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import SubAgentRunner
from inference_optimizer.storage import SqliteConnection


# ===========================================================================
# Shared fixtures
# ===========================================================================
def _write_baseline_yaml(path: Path) -> None:
    """Minimal YAML in our repo's expected shape."""
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "sglang_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _write_vllm_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "vllm_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float = 800.0,
                     ttft_ms: float = 140.0, e2el_ms: float = 2500.0,
                     success: bool = True) -> Path:
    """Create a fake Magpie workspace inside `slot` and return the path."""
    workspace = slot / "benchmark_sglang_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": success,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": tput / 256,
            "output_throughput": tput,
            "total_token_throughput": tput * 2,
            "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {
            "ttft": {"mean_ms": ttft_ms, "p99_ms": ttft_ms + 20},
            "e2el": {"mean_ms": e2el_ms, "p99_ms": e2el_ms + 60},
        },
    }))
    return workspace


# ===========================================================================
# discover_backend_flags
# ===========================================================================
def test_discover_backend_flags_returns_empty_when_missing(tmp_path):
    flags = discover_backend_flags(server_args_path=tmp_path / "nope.py")
    assert flags == []


def test_discover_backend_flags_parses_attributes(tmp_path):
    """Self-contained parse against a tiny fake server_args.py."""
    fake = tmp_path / "server_args.py"
    fake.write_text(
        "class ServerArgs:\n"
        "    def __init__(self):\n"
        "        self.attention_backend = 'flashinfer'\n"
        "        self.enable_overlap_schedule = False\n"
        "        self.disable_overlap_schedule = False\n"
        "        self.enable_fused_moe = False\n"
        "        self.unrelated_field = 0\n"
    )
    flags = discover_backend_flags(server_args_path=fake)
    assert "--attention-backend" in flags
    assert "--enable-overlap-schedule" in flags
    assert "--disable-overlap-schedule" in flags
    assert "--enable-fused-moe" in flags
    assert "--unrelated-field" not in flags


# ===========================================================================
# pick_winners
# ===========================================================================
def test_pick_winners_above_threshold():
    results = [
        VariantResult(name="a", extra_sglang_args="", extra_envs={},
                       status="succeeded", output_throughput=850.0),  # +6.25%
        VariantResult(name="b", extra_sglang_args="", extra_envs={},
                       status="succeeded", output_throughput=900.0),  # +12.5%
        VariantResult(name="c", extra_sglang_args="", extra_envs={},
                       status="succeeded", output_throughput=805.0),  # +0.625% — below 1%
        VariantResult(name="d", extra_sglang_args="", extra_envs={},
                       status="failed"),
    ]
    winners = pick_winners(results, baseline_tput=800.0, keep_threshold_pct=1.0)
    # Only a + b clear the +1% threshold; c is +0.625% (not a winner),
    # d is failed (not eligible).
    assert {w.name for w in winners} == {"a", "b"}


def test_pick_winners_default_threshold():
    """Default threshold is 1% — items at exactly +1% are NOT winners."""
    results = [
        VariantResult(name="exact_1pct",
                       extra_sglang_args="", extra_envs={},
                       status="succeeded", output_throughput=808.0),
        VariantResult(name="just_above",
                       extra_sglang_args="", extra_envs={},
                       status="succeeded", output_throughput=810.0),
    ]
    winners = pick_winners(results, baseline_tput=800.0)
    assert [w.name for w in winners] == ["just_above"]


# ===========================================================================
# run_grid — exercised with a stubbed subprocess.run
# ===========================================================================
@pytest.mark.asyncio
async def test_run_grid_writes_per_variant_yaml_and_parses_report(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    # Stub subprocess.run to fabricate workspaces with throughput
    # proportional to the variant index.
    def _fake_run(cmd, *args, **kwargs):
        # cmd[5] is --benchmark-config <path>; --output-dir is at cmd[7]
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # Tput = 800 + 50*i where i is variant index encoded in name.
        i = int(slot.name.split("_")[1])
        _fake_workspace(slot, tput=800.0 + 50.0 * i)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [
        GridVariant("a", "--attention-backend aiter"),
        GridVariant("b", "--attention-backend triton"),
        GridVariant("c", "--enable-fused-moe"),
    ]
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="--mem-fraction-static 0.85",
            grid=grid, output_root=output_root, variant_timeout_sec=10,
        )
    assert len(results) == 3
    # Each got its own YAML with EXTRA_SGLANG_ARGS containing both base + variant args.
    for i, r in enumerate(results):
        assert r.status == "succeeded"
        assert r.output_throughput == 800.0 + 50.0 * i
        slot = output_root / f"variant_{i:02d}_{grid[i].name}"
        cfg = yaml.safe_load((slot / "config.yaml").read_text())
        envs = cfg["benchmark"]["envs"]
        assert "--mem-fraction-static 0.85" in envs["EXTRA_SGLANG_ARGS"]
        assert grid[i].extra_sglang_args in envs["EXTRA_SGLANG_ARGS"]


@pytest.mark.asyncio
async def test_run_grid_keeps_valid_measurement_with_report_failure_and_nonzero_rc(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0, success=False)
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="cleanup failed",
        )

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
        side_effect=_fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("valid_warning")],
            output_root=output_root,
            variant_timeout_sec=10,
        )

    assert len(results) == 1
    assert results[0].status == "succeeded"
    assert results[0].reported_success is False
    assert results[0].returncode == 1
    assert results[0].output_throughput == 900.0
    assert results[0].completed_requests == 80
    assert "benchmark_report_success_false" in results[0].nonfatal_warnings
    assert "magpie_nonzero_after_valid_measurement" in results[0].nonfatal_warnings
    winners = pick_winners(results, baseline_tput=800.0)
    assert [w.name for w in winners] == ["valid_warning"]


@pytest.mark.asyncio
async def test_run_grid_keeps_going_on_subprocess_failure(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    def _fake_run(cmd, *args, **kwargs):
        # First variant fails (rc=1), second succeeds.
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        i = int(slot.name.split("_")[1])
        if i == 0:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="boom",
            )
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [GridVariant("a"), GridVariant("b")]
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=10,
        )
    assert len(results) == 2
    assert results[0].status == "failed"
    assert "boom" in results[0].error
    assert results[1].status == "succeeded"
    assert results[1].output_throughput == 900.0


@pytest.mark.asyncio
async def test_run_grid_writes_variant_extra_envs(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [
        GridVariant(
            "sglang_multi_stream_overlap",
            extra_envs={"SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1"},
        ),
    ]
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=10,
        )

    assert results[0].status == "succeeded"
    cfg = yaml.safe_load(
        (output_root / "variant_00_sglang_multi_stream_overlap" / "config.yaml")
        .read_text()
    )
    assert cfg["benchmark"]["envs"]["SGLANG_OPT_USE_MULTI_STREAM_OVERLAP"] == "1"


@pytest.mark.asyncio
async def test_run_grid_writes_vllm_extra_args_for_vllm_configs(tmp_path):
    base = tmp_path / "vllm.yaml"
    _write_vllm_yaml(base)
    output_root = tmp_path / "out"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [GridVariant("vllm_block_size_256", "--block-size 256")]
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="--kv-cache-dtype fp8",
            grid=grid, output_root=output_root, variant_timeout_sec=10,
        )

    assert results[0].status == "succeeded"
    cfg = yaml.safe_load(
        (output_root / "variant_00_vllm_block_size_256" / "config.yaml")
        .read_text()
    )
    envs = cfg["benchmark"]["envs"]
    assert envs["EXTRA_VLLM_ARGS"] == "--kv-cache-dtype fp8 --block-size 256"
    assert "EXTRA_SGLANG_ARGS" not in envs


def test_baseline_materialize_uses_vllm_extra_args_env(tmp_path):
    base = tmp_path / "vllm.yaml"
    _write_vllm_yaml(base)
    out = tmp_path / "out"
    out.mkdir()

    materialized = _materialize_config_with_envs(
        base,
        out,
        extra_server_args="--block-size 256",
    )
    cfg = yaml.safe_load(materialized.read_text())
    envs = cfg["benchmark"]["envs"]
    assert envs["EXTRA_VLLM_ARGS"] == "--block-size 256"
    assert "EXTRA_SGLANG_ARGS" not in envs


def test_grid_variant_yaml_preserves_base_extra_args_and_env_overrides(
    tmp_path, monkeypatch,
):
    """Backends/params grid materialization must match baseline env handling.

    Regression: grid runner used to ignore TP/ISL/OSL env overrides and
    overwrite base EXTRA_VLLM_ARGS, so DSR1 ran backend variants with TP=1 and
    dropped the model-required `--block-size 1`.
    """
    base = tmp_path / "vllm.yaml"
    _write_vllm_yaml(base)

    # Simulate a model-specific asset yaml with a required global vLLM arg.
    cfg = yaml.safe_load(base.read_text())
    envs = cfg["benchmark"]["envs"]
    envs["EXTRA_VLLM_ARGS"] = "--block-size 1"
    base.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("ISL", "1024")
    monkeypatch.setenv("OSL", "1024")
    monkeypatch.setenv("MAX_MODEL_LEN", "6144")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    out = _build_variant_yaml(
        base,
        base_extra_args="--kv-cache-dtype fp8",
        variant=GridVariant("vllm_max_seqs", "--max-num-seqs 512"),
        output_subdir=tmp_path / "variant",
        model_path="/wekafs/models/DeepSeek-R1-0528",
        gpu_type="mi355x",
    )
    rendered = yaml.safe_load(out.read_text())
    bench = rendered["benchmark"]
    envs = bench["envs"]

    assert bench["model"] == "/wekafs/models/DeepSeek-R1-0528"
    assert bench["runner_type"] == "mi355x"
    assert "benchmark_script" not in bench
    assert envs["TP"] == 8
    assert envs["CONC"] == 64
    assert envs["ISL"] == 1024
    assert envs["OSL"] == 1024
    assert envs["MAX_MODEL_LEN"] == 6144
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert envs["EXTRA_VLLM_ARGS"] == (
        "--block-size 1 --kv-cache-dtype fp8 --max-num-seqs 512"
    )


# ===========================================================================
# BackendsExecutor / ParamsExecutor / SweepExecutor — end-to-end via SubAgentRunner
# ===========================================================================
@pytest.fixture
def sub_agent_runner(tmp_path):
    db = SqliteConnection(tmp_path / "db.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    runner = SubAgentRunner(locks, tr)
    yield runner, tr, tmp_path
    db.close()


@pytest.mark.asyncio
async def test_backends_executor_picks_best_and_winners(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "backends-out"

    # Stub: variant index → tput
    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        i = int(slot.name.split("_")[1])
        _fake_workspace(slot, tput=800.0 + 30.0 * i)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [
        {"name": f"v{i}", "extra_sglang_args": f"--flag-{i}"}
        for i in range(3)
    ]
    task = await tr.create(
        kind="backends",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            "variant_timeout_sec": 10,
        },
        idempotency_key="be-1",
    )
    sub.register_executor("backends", BackendsExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    out = res.result
    assert out["grid_size"] == 3
    # base_tput=800 → +1% threshold = 808 → variants i=1 (830) & i=2 (860)
    # are winners; i=0 (800) is not.
    winner_names = {w["name"] for w in out["winners"]}
    assert winner_names == {"v1", "v2"}
    assert out["best_variant"]["name"] == "v2"
    assert out["best_gain_pct"] == pytest.approx((860 - 800) / 800 * 100)


@pytest.mark.asyncio
async def test_params_executor_with_default_grid(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "params-out"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    task = await tr.create(
        kind="params",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "variant_timeout_sec": 10,
        },
        idempotency_key="pa-1",
    )
    sub.register_executor("params", ParamsExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result["grid_size"] == len(DEFAULT_PARAMS_GRID)
    # Every variant returned 900 vs base 800 → all winners (>1%)
    assert len(res.result["winners"]) == len(DEFAULT_PARAMS_GRID)


def test_default_grids_are_non_empty():
    assert len(DEFAULT_BACKENDS_GRID) >= 5
    assert len(DEFAULT_PARAMS_GRID)   >= 12
    assert len(DEFAULT_NCCL_GRID)     >= 1
    assert all(isinstance(v, GridVariant) for v in DEFAULT_BACKENDS_GRID)
    names = {v.name for v in DEFAULT_PARAMS_GRID}
    assert {"cuda_graph_max_bs_64", "chunked_prefill_128k",
            "max_prefill_tokens_64k"} <= names


def test_default_params_grid_includes_inferencex_sglang_candidates():
    """Keep the SGLang default search aligned with high-signal InferenceX knobs."""
    by_name = {v.name: v for v in DEFAULT_PARAMS_GRID}

    assert by_name["disable_radix_cache"].extra_sglang_args == "--disable-radix-cache"
    assert by_name["tokenizer_workers_16"].extra_sglang_args == "--tokenizer-worker-num 16"
    assert by_name["stream_interval_50"].extra_sglang_args == "--stream-interval 50"
    assert by_name["max_running_requests_256"].extra_sglang_args == "--max-running-requests 256"

    assert by_name["sglang_multi_stream_overlap"].extra_envs == {
        "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1",
    }
    assert by_name["sglang_flashmla_tilelang"].extra_envs == {
        "SGLANG_HACK_FLASHMLA_BACKEND": "tilelang",
    }
    assert by_name["sglang_tilelang_indexer"].extra_envs == {
        "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
    }


def test_default_vllm_params_grid_includes_inferencex_candidates():
    by_name = {v.name: v for v in DEFAULT_VLLM_PARAMS_GRID}

    assert by_name["vllm_kv_cache_fp8"].extra_sglang_args == "--kv-cache-dtype fp8"
    assert by_name["vllm_block_size_256"].extra_sglang_args == "--block-size 256"
    assert by_name["vllm_no_prefix_cache"].extra_sglang_args == (
        "--no-enable-prefix-caching"
    )
    assert by_name["vllm_fp4_indexer_cache"].extra_sglang_args == (
        "--attention_config.use_fp4_indexer_cache=True"
    )
    assert (
        "FULL_AND_PIECEWISE"
        in by_name["vllm_full_piecewise_compile"].extra_sglang_args
    )


@pytest.mark.asyncio
async def test_params_executor_uses_vllm_grid_for_vllm_config(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "vllm.yaml"
    _write_vllm_yaml(base)
    output_dir = tmp_path / "params-vllm-out"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    task = await tr.create(
        kind="params",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "variant_timeout_sec": 10,
        },
        idempotency_key="pa-vllm-1",
    )
    sub.register_executor("params", ParamsExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result["grid_size"] == len(DEFAULT_VLLM_PARAMS_GRID)
    cfg = yaml.safe_load(
        (output_dir / "variant_00_vllm_kv_cache_fp8" / "config.yaml").read_text()
    )
    assert cfg["benchmark"]["envs"]["EXTRA_VLLM_ARGS"] == "--kv-cache-dtype fp8"


@pytest.mark.asyncio
async def test_sweep_executor_returns_pareto_front(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "sweep-out"

    # Make tput vary with conc and isl: higher conc × lower isl → higher tput
    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        cfg = yaml.safe_load((slot / "config.yaml").read_text())
        envs = cfg["benchmark"]["envs"]
        conc = int(envs.get("CONC", 8))
        isl = int(envs.get("ISL", 1024))
        osl = int(envs.get("OSL", 1024))
        tput = 50.0 * conc + 100000.0 / (isl + 1) + 5000.0 / (osl + 1)
        e2el = 50.0 + isl / 10.0 + osl / 5.0
        _fake_workspace(slot, tput=tput, e2el_ms=e2el)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    task = await tr.create(
        kind="sweep",
        params={
            "config_path":     str(base),
            "output_dir":      str(output_dir),
            "conc_values":     [4, 16],
            "isl_osl_configs": ["1024:1024", "2048:1024"],
            "variant_timeout_sec": 10,
        },
        idempotency_key="sw-1",
    )
    sub.register_executor("sweep", SweepExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    out = res.result
    assert out["grid_size"] == 4  # 2 conc × 2 isl/osl
    assert {(e["conc"], e["isl"], e["osl"]) for e in out["sweep_grid"]} == {
        (4, 1024, 1024), (4, 2048, 1024), (16, 1024, 1024), (16, 2048, 1024),
    }
    # Pareto must include CONC=16 ISL=1024 (best tput) and may include
    # smaller-conc/lower-latency ones; at minimum: non-empty + each entry
    # is non-dominated within the set.
    assert out["pareto_front"], "pareto front shouldn't be empty"
    for f in out["pareto_front"]:
        for o in out["sweep_grid"]:
            if o is f or o["status"] != "succeeded":
                continue
            dominates = (
                o["output_throughput"] >= f["output_throughput"]
                and o["e2el_mean_ms"] <= f["e2el_mean_ms"]
                and (o["output_throughput"] > f["output_throughput"]
                     or o["e2el_mean_ms"] < f["e2el_mean_ms"])
            )
            assert not dominates, (
                f"pareto entry {f['name']} dominated by {o['name']}"
            )


def test_sweep_default_grid_size():
    expected = len(DEFAULT_CONC_VALUES) * len(DEFAULT_ISL_OSL)
    assert expected == 9  # 3×3


# ===========================================================================
# Workload-contract regression (baseline 4367 tok/s vs variants ~360 tok/s bug)
# ===========================================================================
# Before the `_workload_envs.materialize_config_with_envs` call was added to
# params/backends/sweep, every variant rendered from the shipped YAML's
# smoke defaults (CONC=8 / ISL=256 / OSL=256 / TP=1) regardless of what the
# operator exported, so on the user's vLLM 32B/8-GPU run the baseline ran
# at CONC=64/ISL=1024/OSL=1024 (4367 tok/s) while every variant ran at
# CONC=8/ISL=256/OSL=256 (~360 tok/s). These tests pin the fix.
@pytest.mark.asyncio
async def test_params_variants_inherit_process_env_workload(
    sub_agent_runner, tmp_path, monkeypatch,
):
    """`os.environ["CONC"]=64` must reach the variant YAML even though the
    base YAML's hardcoded default is CONC=8 and the variant defines no CONC
    override. Same for ISL/OSL/TP — pinning the workload-contract reuse fix.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)  # YAML defaults: TP=1 CONC=8 ISL=256 OSL=256
    output_dir = tmp_path / "params-out"

    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("ISL", "1024")
    monkeypatch.setenv("OSL", "1024")
    monkeypatch.setenv("TP", "8")

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [{"name": "no_workload_override", "extra_sglang_args": "--flag-x"}]
    task = await tr.create(
        kind="params",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "variant_timeout_sec": 10,
        },
        idempotency_key="pa-workload-1",
    )
    sub.register_executor("params", ParamsExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    cfg = yaml.safe_load(
        (output_dir / "variant_00_no_workload_override" / "config.yaml")
        .read_text()
    )
    envs = cfg["benchmark"]["envs"]
    assert envs["CONC"] == 64, f"variant lost process-env CONC; got {envs.get('CONC')}"
    assert envs["ISL"] == 1024
    assert envs["OSL"] == 1024
    assert envs["TP"] == 8


@pytest.mark.asyncio
async def test_backends_variants_inherit_process_env_workload(
    sub_agent_runner, tmp_path, monkeypatch,
):
    """Same workload-contract pin for BackendsExecutor."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "backends-out"

    monkeypatch.setenv("CONC", "32")
    monkeypatch.setenv("TP", "4")

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    grid = [{"name": "no_workload_override", "extra_sglang_args": "--flag-y"}]
    task = await tr.create(
        kind="backends",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "variant_timeout_sec": 10,
        },
        idempotency_key="be-workload-1",
    )
    sub.register_executor("backends", BackendsExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    cfg = yaml.safe_load(
        (output_dir / "variant_00_no_workload_override" / "config.yaml")
        .read_text()
    )
    envs = cfg["benchmark"]["envs"]
    assert envs["CONC"] == 32
    assert envs["TP"] == 4


@pytest.mark.asyncio
async def test_sweep_per_variant_envs_still_win_over_baseline_workload(
    sub_agent_runner, tmp_path, monkeypatch,
):
    """Sweep deliberately overrides CONC/ISL/OSL per variant — those must
    still beat the baseline-materialized contract (last-wins ordering in
    `_grid_runner._build_variant_yaml`)."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "sweep-out"

    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("TP", "8")  # baseline contract: TP=8 must propagate

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    task = await tr.create(
        kind="sweep",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "conc_values": [4],  # sweep overrides CONC=4
            "isl_osl_configs": ["1024:1024"],
            "variant_timeout_sec": 10,
        },
        idempotency_key="sw-workload-1",
    )
    sub.register_executor("sweep", SweepExecutor())
    with patch("inference_optimizer.orchestrator.action_executors._grid_runner.subprocess.run",
                side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"

    cfg = yaml.safe_load(
        (output_dir / "variant_00_conc4_isl1024_osl1024" / "config.yaml")
        .read_text()
    )
    envs = cfg["benchmark"]["envs"]
    assert envs["CONC"] == "4", "sweep variant CONC override must win over baseline contract"
    assert envs["ISL"] == "1024"
    assert envs["OSL"] == "1024"
    assert envs["TP"] == 8, "TP must still inherit from process env (baseline contract)"


def test_baseline_executor_surfaces_materialized_config_path(tmp_path):
    """BaselineExecutor result must include `materialized_config` so the
    Coordinator can store it in SharedState.baseline_config_path and plumb
    it forward to params/backends/sweep tasks."""
    from inference_optimizer.orchestrator.action_executors.baseline import (
        BaselineExecutor,
    )
    from inference_optimizer.orchestrator.task_registry import TaskRegistry
    from inference_optimizer.orchestrator.resource_lock import (
        ResourceLockManager, SqliteLeaseBackend,
    )
    from inference_optimizer.orchestrator.sub_agent_runner import SubAgentRunner
    from inference_optimizer.storage import SqliteConnection
    import asyncio

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "baseline-out"

    async def _go():
        db = SqliteConnection(tmp_path / "db.db")
        try:
            locks = ResourceLockManager(SqliteLeaseBackend(db))
            tr = TaskRegistry(db)
            sub = SubAgentRunner(locks, tr)
            task = await tr.create(
                kind="baseline",
                params={
                    "config_path": str(base),
                    "output_dir": str(output_dir),
                    "timeout_sec": 30,
                },
                idempotency_key="bl-workload-surface-1",
            )

            def _fake_run(cmd, *args, **kwargs):
                out_idx = cmd.index("--output-dir")
                slot = Path(cmd[out_idx + 1])
                _fake_workspace(slot, tput=900.0)
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="ok", stderr="",
                )

            sub.register_executor("baseline", BaselineExecutor(cwd=tmp_path))
            with patch(
                "inference_optimizer.orchestrator.action_executors.baseline.subprocess.run",
                side_effect=_fake_run,
            ):
                return await sub.run_task(task)
        finally:
            db.close()

    res = asyncio.run(_go())
    assert res.state == "succeeded"
    materialized = res.result.get("materialized_config")
    assert materialized, "baseline result missing `materialized_config`"
    assert Path(materialized).exists()
    assert Path(materialized).name == "baseline_config.with_envs.yaml"
